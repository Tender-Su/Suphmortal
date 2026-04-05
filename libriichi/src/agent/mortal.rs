use super::{BatchAgent, InvisibleState};
use crate::array::Simple2DArray;
use crate::consts::ACTION_SPACE;
use crate::consts::obs_shape;
use crate::consts::oracle_obs_shape;
use crate::mjai::{Event, EventExt, Metadata};
use crate::state::PlayerState;
use crate::{must_tile, tu8};
use std::cell::UnsafeCell;
use std::mem;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use crossbeam::sync::WaitGroup;
use ndarray::prelude::*;
use numpy::{PyArray2, PyArray3};
use parking_lot::Mutex;
use pyo3::intern;
use pyo3::prelude::*;

pub struct MortalBatchAgent {
    engine: PyObject,
    is_oracle: bool,
    version: u32,
    enable_quick_eval: bool,
    enable_rule_based_agari_guard: bool,
    enable_metadata: bool,
    use_obs_encode_into: bool,
    profile_enabled: bool,
    name: String,
    player_ids: Vec<u8>,

    actions: Vec<usize>,
    alt_actions: Vec<usize>,
    q_values: Vec<[f32; ACTION_SPACE]>,
    masks_recv: Vec<[bool; ACTION_SPACE]>,
    is_greedy: Vec<bool>,
    last_eval_elapsed: Duration,
    last_batch_size: usize,

    evaluated: bool,
    quick_eval_reactions: Vec<Option<Event>>,
    perf: AgentPerfStats,
    state_batch: Arc<DirectStateBatch>,
    mask_batch: Arc<DirectMaskBatch>,
    batch_input_owner: Py<BatchInputOwner>,

    wg: WaitGroup,
    sync_fields: Arc<Mutex<SyncFields>>,
}

#[pyclass]
struct BatchInputOwner {
    _state_batch_owner: Arc<DirectStateBatch>,
    _mask_batch_owner: Arc<DirectMaskBatch>,
    invisible_state_batch: Option<Array3<f32>>,
}

struct DirectStateBatch {
    rows: usize,
    capacity: usize,
    data: UnsafeCell<Box<[f32]>>,
}

struct DirectMaskBatch {
    capacity: usize,
    data: UnsafeCell<Box<[[bool; ACTION_SPACE]]>>,
}

// SAFETY: The backing storage is fixed-size and never reallocated after
// construction. Callers only mutate disjoint slot ranges reserved under
// `SyncFields`, and `evaluate()` waits for all encoding workers before reading.
unsafe impl Send for DirectStateBatch {}
// SAFETY: See the `Send` justification above.
unsafe impl Sync for DirectStateBatch {}
// SAFETY: The backing storage is fixed-size and workers only write to disjoint
// slot indices reserved under `SyncFields`.
unsafe impl Send for DirectMaskBatch {}
// SAFETY: See the `Send` justification above.
unsafe impl Sync for DirectMaskBatch {}

struct SyncFields {
    invisible_states: Vec<Option<Array2<f32>>>,
    action_idxs: Vec<usize>,
    kan_action_idxs: Vec<Option<usize>>,
    next_slot: usize,
}

#[derive(Default)]
struct AgentPerfStats {
    evaluate_calls: usize,
    total_batch_items: usize,
    max_batch_size: usize,
    wait_encode_elapsed: Duration,
    batch_pack_elapsed: Duration,
    py_call_elapsed: Duration,
    py_extract_elapsed: Duration,
    post_eval_elapsed: Duration,
    reaction_decode_elapsed: Duration,
}

fn fill_invisible_state_batch(
    owner: &mut BatchInputOwner,
    states: &[Option<Array2<f32>>],
    field_name: &str,
) -> Result<()> {
    ensure!(!states.is_empty(), "empty {field_name} batch");
    let first_dim = states[0]
        .as_ref()
        .context("missing invisible state for batch slot 0")?
        .raw_dim();
    let batch = owner
        .invisible_state_batch
        .as_mut()
        .context("missing invisible_state_batch allocation")?;
    ensure!(
        batch.dim().0 >= states.len()
            && batch.dim().1 == first_dim[0]
            && batch.dim().2 == first_dim[1],
        "insufficient preallocated invisible_state_batch capacity",
    );
    for (index, state) in states.iter().enumerate() {
        let state = state
            .as_ref()
            .with_context(|| format!("missing {field_name} at batch index {index}"))?;
        ensure!(
            state.raw_dim() == first_dim,
            "mismatched {field_name} shape at batch index {index}",
        );
        let mut dst = batch.slice_mut(s![index, .., ..]);
        dst.as_slice_mut()
            .context("fill_invisible_state_batch expected contiguous output slice")?
            .copy_from_slice(
                state
                    .as_slice()
                    .context("fill_invisible_state_batch expected contiguous input slice")?,
            );
    }
    Ok(())
}

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        },
        Err(_) => default,
    }
}

fn agent_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("MORTAL_AGENT_PROFILE", false))
}

fn encode_obs_legacy(
    state: &PlayerState,
    version: u32,
    at_kan_select: bool,
) -> (Simple2DArray<34, f32>, [bool; ACTION_SPACE]) {
    let (feature, mask) = state.encode_obs(version, at_kan_select);
    let mut feature_simple = Simple2DArray::new(feature.nrows());
    feature_simple.as_mut_slice().copy_from_slice(
        feature
            .as_slice()
            .expect("encode_obs produced a non-contiguous array"),
    );
    let mask = mask
        .to_vec()
        .try_into()
        .expect("encode_obs produced an unexpected mask length");
    (feature_simple, mask)
}

impl DirectStateBatch {
    fn new(capacity: usize, rows: usize) -> Self {
        Self {
            rows,
            capacity,
            data: UnsafeCell::new(vec![0.0; capacity * rows * 34].into_boxed_slice()),
        }
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    fn slot_len(&self) -> usize {
        self.rows * 34
    }

    fn write_slot(
        &self,
        slot: usize,
        state: &PlayerState,
        version: u32,
        at_kan_select: bool,
        use_obs_encode_into: bool,
    ) -> [bool; ACTION_SPACE] {
        let slot_slice = self.slot_slice_mut(slot);
        slot_slice.fill(0.0);
        let mut mask = [false; ACTION_SPACE];
        if use_obs_encode_into {
            state.encode_obs_row_major_into(version, at_kan_select, slot_slice, &mut mask);
            mask
        } else {
            let (feature, mask) = encode_obs_legacy(state, version, at_kan_select);
            slot_slice.copy_from_slice(feature.as_slice());
            mask
        }
    }

    fn view(&self, batch_size: usize) -> Result<ArrayView3<'_, f32>> {
        ensure!(
            batch_size <= self.capacity,
            "state batch view exceeds capacity"
        );
        let len = batch_size * self.slot_len();
        let data = self.data_slice(len);
        ArrayView3::from_shape((batch_size, self.rows, 34), data)
            .context("failed to build contiguous state batch view")
    }

    fn slot_slice_mut(&self, slot: usize) -> &mut [f32] {
        assert!(slot < self.capacity, "state batch slot out of range");
        let start = slot * self.slot_len();
        let end = start + self.slot_len();
        // SAFETY: The boxed slice is allocated once in `new()` and never
        // reallocated. Each slot is reserved exactly once per batch under
        // `SyncFields`, so concurrent workers only write to disjoint ranges.
        unsafe { &mut (&mut *self.data.get())[start..end] }
    }

    fn data_slice(&self, len: usize) -> &[f32] {
        // SAFETY: `evaluate()` calls this only after waiting for all encoding
        // workers to finish, so there are no concurrent writers. The backing
        // storage is fixed-size and valid for the lifetime of `self`.
        unsafe { &(&*self.data.get())[..len] }
    }
}

impl DirectMaskBatch {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            data: UnsafeCell::new(vec![[false; ACTION_SPACE]; capacity].into_boxed_slice()),
        }
    }

    fn write_slot(&self, slot: usize, mask: [bool; ACTION_SPACE]) {
        assert!(slot < self.capacity, "mask batch slot out of range");
        // SAFETY: The boxed slice is fixed-size and each slot is reserved once
        // per batch under `SyncFields`, so concurrent workers only write to
        // disjoint indices.
        unsafe {
            (&mut *self.data.get())[slot] = mask;
        }
    }

    fn view(&self, batch_size: usize) -> Result<ArrayView2<'_, bool>> {
        ensure!(
            batch_size <= self.capacity,
            "mask batch view exceeds capacity"
        );
        // SAFETY: `evaluate()` calls this only after waiting for all encoding
        // workers to finish, so there are no concurrent writers.
        let rows = unsafe { &(&*self.data.get())[..batch_size] };
        ArrayView2::from_shape((batch_size, ACTION_SPACE), rows.as_flattened())
            .context("failed to build contiguous mask batch view")
    }
}

impl MortalBatchAgent {
    pub fn new(engine: PyObject, player_ids: &[u8]) -> Result<Self> {
        ensure!(player_ids.iter().all(|&id| matches!(id, 0..=3)));

        let (
            name,
            is_oracle,
            version,
            enable_quick_eval,
            enable_rule_based_agari_guard,
            enable_metadata,
        ) = Python::with_gil(|py| {
            let obj = engine.bind_borrowed(py);
            ensure!(
                obj.getattr("react_batch")?.is_callable(),
                "missing method react_batch",
            );

            let name = obj.getattr("name")?.extract()?;
            let is_oracle: bool = obj.getattr("is_oracle")?.extract()?;
            let version = obj.getattr("version")?.extract()?;
            let enable_quick_eval = obj.getattr("enable_quick_eval")?.extract()?;
            let enable_rule_based_agari_guard =
                obj.getattr("enable_rule_based_agari_guard")?.extract()?;
            let enable_metadata = obj.getattr("enable_metadata")?.extract()?;
            Ok((
                name,
                is_oracle,
                version,
                enable_quick_eval,
                enable_rule_based_agari_guard,
                enable_metadata,
            ))
        })?;

        let size = player_ids.len();
        let state_rows = obs_shape(version).0;
        let invisible_state_rows = is_oracle.then(|| oracle_obs_shape(version).0);
        let max_pending_slots = size * 2;
        let quick_eval_reactions = if enable_quick_eval {
            vec![None; size]
        } else {
            vec![]
        };
        let state_batch = Arc::new(DirectStateBatch::new(max_pending_slots, state_rows));
        let mask_batch = Arc::new(DirectMaskBatch::new(max_pending_slots));
        let batch_input_owner = Python::with_gil(|py| {
            Py::new(
                py,
                BatchInputOwner {
                    _state_batch_owner: Arc::clone(&state_batch),
                    _mask_batch_owner: Arc::clone(&mask_batch),
                    invisible_state_batch: invisible_state_rows
                        .map(|rows| Array3::<f32>::zeros((max_pending_slots, rows, 34))),
                },
            )
        })?;
        let sync_fields = Arc::new(Mutex::new(SyncFields {
            invisible_states: Vec::with_capacity(max_pending_slots),
            action_idxs: vec![0; size],
            kan_action_idxs: vec![None; size],
            next_slot: 0,
        }));

        Ok(Self {
            engine,
            is_oracle,
            version,
            enable_quick_eval,
            enable_rule_based_agari_guard,
            enable_metadata,
            use_obs_encode_into: env_flag("MORTAL_1V3_ENABLE_ENCODE_INTO", true),
            profile_enabled: agent_profile_enabled(),
            name,
            player_ids: player_ids.to_vec(),

            actions: vec![],
            alt_actions: vec![],
            q_values: vec![],
            masks_recv: vec![],
            is_greedy: vec![],
            last_eval_elapsed: Duration::ZERO,
            last_batch_size: 0,

            evaluated: false,
            quick_eval_reactions,
            perf: AgentPerfStats::default(),
            state_batch,
            mask_batch,
            batch_input_owner,

            wg: WaitGroup::new(),
            sync_fields,
        })
    }

    fn evaluate(&mut self) -> Result<()> {
        let wait_started = self.profile_enabled.then(Instant::now);
        mem::take(&mut self.wg).wait();
        if let Some(started) = wait_started {
            self.perf.wait_encode_elapsed += started.elapsed();
        }
        let mut sync_fields = self.sync_fields.lock();

        if sync_fields.next_slot == 0 {
            return Ok(());
        }

        let start = Instant::now();
        self.last_batch_size = sync_fields.next_slot;
        self.perf.evaluate_calls += 1;
        self.perf.total_batch_items += self.last_batch_size;
        self.perf.max_batch_size = self.perf.max_batch_size.max(self.last_batch_size);
        let invisible_capacity = sync_fields.invisible_states.capacity();
        let invisible_states = self
            .is_oracle
            .then(|| mem::take(&mut sync_fields.invisible_states));
        if self.is_oracle {
            sync_fields.invisible_states = Vec::with_capacity(invisible_capacity);
        }
        sync_fields.next_slot = 0;
        drop(sync_fields);

        let state_view = self.state_batch.view(self.last_batch_size)?;
        let mask_view = self.mask_batch.view(self.last_batch_size)?;
        let states = Python::with_gil(|py| -> Result<_> {
            let pack_started = self.profile_enabled.then(Instant::now);
            let owner_bound = self.batch_input_owner.clone_ref(py).into_bound(py);
            {
                let mut owner = owner_bound.borrow_mut();
                if let Some(ref batch) = invisible_states {
                    fill_invisible_state_batch(&mut owner, batch, "invisible_state")?;
                }
            }
            if let Some(started) = pack_started {
                self.perf.batch_pack_elapsed += started.elapsed();
            }
            let owner = owner_bound.borrow();
            let states =
                unsafe { PyArray3::borrow_from_array(&state_view, owner_bound.clone().into_any()) };
            let masks =
                unsafe { PyArray2::borrow_from_array(&mask_view, owner_bound.clone().into_any()) };
            let invisible_states = owner
                .invisible_state_batch
                .as_ref()
                .filter(|_| invisible_states.is_some())
                .map(|batch| unsafe {
                    PyArray3::borrow_from_array(
                        &batch.slice(s![..self.last_batch_size, .., ..]),
                        owner_bound.clone().into_any(),
                    )
                });
            let args = (states, masks, invisible_states);
            if self.enable_metadata {
                let py_call_started = self.profile_enabled.then(Instant::now);
                let response = self
                    .engine
                    .bind_borrowed(py)
                    .call_method1(intern!(py, "react_batch"), args)
                    .context("failed to execute `react_batch` on Python engine")?;
                if let Some(started) = py_call_started {
                    self.perf.py_call_elapsed += started.elapsed();
                }
                let py_extract_started = self.profile_enabled.then(Instant::now);
                let (actions, q_values, masks_recv, is_greedy) = response
                    .extract()
                    .context("failed to extract to Rust type")?;
                if let Some(started) = py_extract_started {
                    self.perf.py_extract_elapsed += started.elapsed();
                }
                Ok((
                    actions,
                    Some(q_values),
                    Some(masks_recv),
                    Some(is_greedy),
                    None,
                ))
            } else {
                let py_call_started = self.profile_enabled.then(Instant::now);
                let response = self
                    .engine
                    .bind_borrowed(py)
                    .call_method1(intern!(py, "react_batch_action_only"), args)
                    .context("failed to execute `react_batch_action_only` on Python engine")?;
                if let Some(started) = py_call_started {
                    self.perf.py_call_elapsed += started.elapsed();
                }
                let py_extract_started = self.profile_enabled.then(Instant::now);
                let (actions, alt_actions) = response
                    .extract()
                    .context("failed to extract action-only response to Rust type")?;
                if let Some(started) = py_extract_started {
                    self.perf.py_extract_elapsed += started.elapsed();
                }
                Ok((actions, None, None, None, Some(alt_actions)))
            }
        })?;
        let post_eval_started = self.profile_enabled.then(Instant::now);
        self.actions = states.0;
        self.q_values = states.1.unwrap_or_default();
        self.masks_recv = states.2.unwrap_or_default();
        self.is_greedy = states.3.unwrap_or_default();
        self.alt_actions = states.4.unwrap_or_default();
        if let Some(started) = post_eval_started {
            self.perf.post_eval_elapsed += started.elapsed();
        }

        self.last_eval_elapsed = Instant::now()
            .checked_duration_since(start)
            .unwrap_or(Duration::ZERO);

        Ok(())
    }

    fn gen_meta(&self, state: &PlayerState, action_idx: usize) -> Metadata {
        let q_values = self.q_values[action_idx];
        let masks = self.masks_recv[action_idx];
        let is_greedy = self.is_greedy[action_idx];

        let mut mask_bits = 0;
        let q_values_compact = q_values
            .into_iter()
            .zip(masks)
            .enumerate()
            .filter(|&(_, (_, m))| m)
            .map(|(i, (q, _))| {
                mask_bits |= 0b1 << i;
                q
            })
            .collect();

        Metadata {
            q_values: Some(q_values_compact),
            mask_bits: Some(mask_bits),
            is_greedy: Some(is_greedy),
            shanten: Some(state.shanten()),
            at_furiten: Some(state.at_furiten()),
            ..Default::default()
        }
    }
}

impl BatchAgent for MortalBatchAgent {
    #[inline]
    fn name(&self) -> String {
        self.name.clone()
    }

    #[inline]
    fn oracle_obs_version(&self) -> Option<u32> {
        self.is_oracle.then_some(self.version)
    }

    #[inline]
    fn uses_event_log(&self) -> bool {
        false
    }

    fn set_scene(
        &mut self,
        index: usize,
        _: &[EventExt],
        state: &PlayerState,
        invisible_state: Option<InvisibleState>,
    ) -> Result<()> {
        self.evaluated = false;
        let cans = state.last_cans();

        if self.enable_quick_eval
            && cans.can_discard
            && !cans.can_riichi
            && !cans.can_tsumo_agari
            && !cans.can_ankan
            && !cans.can_kakan
            && !cans.can_ryukyoku
        {
            let candidates = state.discard_candidates_aka();
            let mut only_candidate = None;
            for (tile, &flag) in candidates.iter().enumerate() {
                if !flag {
                    continue;
                }
                match only_candidate.take() {
                    None => only_candidate = Some(tile),
                    Some(_) => break,
                }
            }

            if let Some(tile_id) = only_candidate {
                let actor = self.player_ids[index];
                let pai = must_tile!(tile_id);
                let tsumogiri = state.last_self_tsumo().is_some_and(|t| t == pai);
                let ev = Event::Dahai {
                    actor,
                    pai,
                    tsumogiri,
                };
                self.quick_eval_reactions[index] = Some(ev);
                return Ok(());
            }
        }

        let need_kan_select = if !cans.can_ankan && !cans.can_kakan {
            false
        } else if !self.enable_quick_eval {
            true
        } else {
            state.ankan_candidates().len() + state.kakan_candidates().len() > 1
        };

        let version = self.version;
        let use_obs_encode_into = self.use_obs_encode_into;
        let is_oracle = self.is_oracle;
        let state_batch = Arc::clone(&self.state_batch);
        let mask_batch = Arc::clone(&self.mask_batch);
        let (action_slot, kan_slot) = {
            let mut sync_fields = self.sync_fields.lock();
            let reserve_slot = |sync_fields: &mut SyncFields| -> Result<usize> {
                ensure!(
                    sync_fields.next_slot < state_batch.capacity(),
                    "exceeded preallocated state batch slots",
                );
                let slot = sync_fields.next_slot;
                sync_fields.next_slot += 1;
                if is_oracle {
                    sync_fields.invisible_states.push(None);
                }
                Ok(slot)
            };

            let kan_slot = need_kan_select
                .then(|| reserve_slot(&mut sync_fields))
                .transpose()?;
            let action_slot = reserve_slot(&mut sync_fields)?;
            sync_fields.action_idxs[index] = action_slot;
            sync_fields.kan_action_idxs[index] = kan_slot;
            (action_slot, kan_slot)
        };
        let state = state.clone();
        let sync_fields = Arc::clone(&self.sync_fields);
        let wg = self.wg.clone();
        rayon::spawn(move || {
            let _wg = wg;

            // Encode features in parallel within the game batch to utilize
            // multiple cores, as this can be very CPU-intensive, especially for
            // the sp feature (since v4).
            if let Some(kan_slot) = kan_slot {
                let kan_mask =
                    state_batch.write_slot(kan_slot, &state, version, true, use_obs_encode_into);
                mask_batch.write_slot(kan_slot, kan_mask);
                if is_oracle {
                    let mut sync_fields = sync_fields.lock();
                    sync_fields.invisible_states[kan_slot] = invisible_state.clone();
                }
            }

            let mask =
                state_batch.write_slot(action_slot, &state, version, false, use_obs_encode_into);
            mask_batch.write_slot(action_slot, mask);
            if is_oracle {
                let mut sync_fields = sync_fields.lock();
                sync_fields.invisible_states[action_slot] = invisible_state;
            }
        });

        Ok(())
    }

    fn get_reaction(
        &mut self,
        index: usize,
        _: &[EventExt],
        state: &PlayerState,
        _: Option<InvisibleState>,
    ) -> Result<EventExt> {
        if self.enable_quick_eval {
            if let Some(ev) = self.quick_eval_reactions[index].take() {
                return Ok(EventExt::no_meta(ev));
            }
        }

        if !self.evaluated {
            self.evaluate()?;
            self.evaluated = true;
        }
        let start = Instant::now();
        let decode_started = self.profile_enabled.then_some(start);

        let mut sync_fields = self.sync_fields.lock();
        let action_idx = sync_fields.action_idxs[index];
        let kan_select_idx = sync_fields.kan_action_idxs[index].take();

        let actor = self.player_ids[index];
        let akas_in_hand = state.akas_in_hand();
        let cans = state.last_cans();

        let orig_action = self.actions[action_idx];
        let action =
            if self.enable_rule_based_agari_guard && orig_action == 43 && !state.rule_based_agari()
            {
                // The engine wants agari, but the rule-based engine is against
                // it. In rule-based agari guard mode, it will force to execute
                // the best alternative option other than agari.
                if self.enable_metadata {
                    let mut q_values = self.q_values[action_idx];
                    q_values[43] = f32::MIN;
                    q_values
                        .iter()
                        .enumerate()
                        .max_by(|(_, l), (_, r)| l.total_cmp(r))
                        .unwrap()
                        .0
                } else {
                    self.alt_actions[action_idx]
                }
            } else {
                orig_action
            };

        let event = match action {
            0..=36 => {
                ensure!(
                    cans.can_discard,
                    "failed discard check: {}",
                    state.brief_info()
                );

                let pai = must_tile!(action);
                let tsumogiri = state.last_self_tsumo().is_some_and(|t| t == pai);
                Event::Dahai {
                    actor,
                    pai,
                    tsumogiri,
                }
            }

            37 => {
                ensure!(
                    cans.can_riichi,
                    "failed riichi check: {}",
                    state.brief_info()
                );

                Event::Reach { actor }
            }

            38 => {
                ensure!(
                    cans.can_chi_low,
                    "failed chi low check: {}",
                    state.brief_info()
                );

                let pai = state
                    .last_kawa_tile()
                    .context("invalid state: no last kawa tile")?;
                let first = pai.next();

                let can_akaize_consumed = match pai.as_u8() {
                    tu8!(3m) | tu8!(4m) => akas_in_hand[0],
                    tu8!(3p) | tu8!(4p) => akas_in_hand[1],
                    tu8!(3s) | tu8!(4s) => akas_in_hand[2],
                    _ => false,
                };
                let consumed = if can_akaize_consumed {
                    [first.akaize(), first.next().akaize()]
                } else {
                    [first, first.next()]
                };
                Event::Chi {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed,
                }
            }
            39 => {
                ensure!(
                    cans.can_chi_mid,
                    "failed chi mid check: {}",
                    state.brief_info()
                );

                let pai = state
                    .last_kawa_tile()
                    .context("invalid state: no last kawa tile")?;

                let can_akaize_consumed = match pai.as_u8() {
                    tu8!(4m) | tu8!(6m) => akas_in_hand[0],
                    tu8!(4p) | tu8!(6p) => akas_in_hand[1],
                    tu8!(4s) | tu8!(6s) => akas_in_hand[2],
                    _ => false,
                };
                let consumed = if can_akaize_consumed {
                    [pai.prev().akaize(), pai.next().akaize()]
                } else {
                    [pai.prev(), pai.next()]
                };
                Event::Chi {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed,
                }
            }
            40 => {
                ensure!(
                    cans.can_chi_high,
                    "failed chi high check: {}",
                    state.brief_info()
                );

                let pai = state
                    .last_kawa_tile()
                    .context("invalid state: no last kawa tile")?;
                let last = pai.prev();

                let can_akaize_consumed = match pai.as_u8() {
                    tu8!(6m) | tu8!(7m) => akas_in_hand[0],
                    tu8!(6p) | tu8!(7p) => akas_in_hand[1],
                    tu8!(6s) | tu8!(7s) => akas_in_hand[2],
                    _ => false,
                };
                let consumed = if can_akaize_consumed {
                    [last.prev().akaize(), last.akaize()]
                } else {
                    [last.prev(), last]
                };
                Event::Chi {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed,
                }
            }

            41 => {
                ensure!(cans.can_pon, "failed pon check: {}", state.brief_info());

                let pai = state
                    .last_kawa_tile()
                    .context("invalid state: no last kawa tile")?;

                let can_akaize_consumed = match pai.as_u8() {
                    tu8!(5m) => akas_in_hand[0],
                    tu8!(5p) => akas_in_hand[1],
                    tu8!(5s) => akas_in_hand[2],
                    _ => false,
                };
                let consumed = if can_akaize_consumed {
                    [pai.akaize(), pai.deaka()]
                } else {
                    [pai.deaka(); 2]
                };
                Event::Pon {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed,
                }
            }

            42 => {
                ensure!(
                    cans.can_daiminkan || cans.can_ankan || cans.can_kakan,
                    "failed kan check: {}",
                    state.brief_info()
                );

                let ankan_candidates = state.ankan_candidates();
                let kakan_candidates = state.kakan_candidates();

                let tile = if let Some(kan_idx) = kan_select_idx {
                    let tile = must_tile!(self.actions[kan_idx]);
                    ensure!(
                        ankan_candidates.contains(&tile) || kakan_candidates.contains(&tile),
                        "kan choice not in kan candidates: {}",
                        state.brief_info()
                    );
                    tile
                } else if cans.can_daiminkan {
                    state
                        .last_kawa_tile()
                        .context("invalid state: no last kawa tile")?
                } else if cans.can_ankan {
                    ankan_candidates[0]
                } else {
                    kakan_candidates[0]
                };

                if cans.can_daiminkan {
                    let consumed = if tile.is_aka() {
                        [tile.deaka(); 3]
                    } else {
                        [tile.akaize(), tile, tile]
                    };
                    Event::Daiminkan {
                        actor,
                        target: cans.target_actor,
                        pai: tile,
                        consumed,
                    }
                } else if cans.can_ankan && ankan_candidates.contains(&tile.deaka()) {
                    Event::Ankan {
                        actor,
                        consumed: [tile.akaize(), tile, tile, tile],
                    }
                } else {
                    let can_akaize_target = match tile.as_u8() {
                        tu8!(5m) => akas_in_hand[0],
                        tu8!(5p) => akas_in_hand[1],
                        tu8!(5s) => akas_in_hand[2],
                        _ => false,
                    };
                    let (pai, consumed) = if can_akaize_target {
                        (tile.akaize(), [tile.deaka(); 3])
                    } else {
                        (tile.deaka(), [tile.akaize(), tile.deaka(), tile.deaka()])
                    };
                    Event::Kakan {
                        actor,
                        pai,
                        consumed,
                    }
                }
            }

            43 => {
                ensure!(
                    cans.can_agari(),
                    "failed hora check: {}",
                    state.brief_info(),
                );

                Event::Hora {
                    actor,
                    target: cans.target_actor,
                    deltas: None,
                    ura_markers: None,
                }
            }

            44 => {
                ensure!(
                    cans.can_ryukyoku,
                    "failed ryukyoku check: {}",
                    state.brief_info()
                );

                Event::Ryukyoku { deltas: None }
            }

            // 45
            _ => Event::None,
        };

        if !self.enable_metadata {
            if let Some(started) = decode_started {
                self.perf.reaction_decode_elapsed += started.elapsed();
            }
            return Ok(EventExt::no_meta(event));
        }

        let mut meta = self.gen_meta(state, action_idx);
        let eval_time_ns = Instant::now()
            .checked_duration_since(start)
            .unwrap_or(Duration::ZERO)
            .saturating_add(self.last_eval_elapsed)
            .as_nanos()
            .try_into()
            .unwrap_or(u64::MAX);

        meta.eval_time_ns = Some(eval_time_ns);
        meta.batch_size = Some(self.last_batch_size);
        meta.kan_select = kan_select_idx.map(|kan_idx| Box::new(self.gen_meta(state, kan_idx)));

        if let Some(started) = decode_started {
            self.perf.reaction_decode_elapsed += started.elapsed();
        }

        Ok(EventExt {
            event,
            meta: Some(meta),
        })
    }
}

impl Drop for MortalBatchAgent {
    fn drop(&mut self) {
        if !self.profile_enabled || self.perf.evaluate_calls == 0 {
            return;
        }

        let avg_batch_size = self.perf.total_batch_items as f64 / self.perf.evaluate_calls as f64;
        log::info!(
            "mortal agent profile: name={} players={} eval_calls={} avg_batch_size={:.1} max_batch_size={} wait_encode={:.3}s batch_pack={:.3}s py_call={:.3}s py_extract={:.3}s post_eval={:.3}s reaction_decode={:.3}s",
            self.name,
            self.player_ids.len(),
            self.perf.evaluate_calls,
            avg_batch_size,
            self.perf.max_batch_size,
            self.perf.wait_encode_elapsed.as_secs_f64(),
            self.perf.batch_pack_elapsed.as_secs_f64(),
            self.perf.py_call_elapsed.as_secs_f64(),
            self.perf.py_extract_elapsed.as_secs_f64(),
            self.perf.post_eval_elapsed.as_secs_f64(),
            self.perf.reaction_decode_elapsed.as_secs_f64(),
        );
    }
}

#[cfg(test)]
mod test {
    use super::{DirectMaskBatch, DirectStateBatch};
    use crate::consts::ACTION_SPACE;

    #[test]
    fn direct_mask_batch_view_preserves_order() {
        let mut first = [false; ACTION_SPACE];
        first[0] = true;
        first[2] = true;
        let mut second = [false; ACTION_SPACE];
        second[1] = true;
        let batch = DirectMaskBatch::new(2);
        batch.write_slot(0, first);
        batch.write_slot(1, second);

        let view = batch.view(2).unwrap();
        assert!(view[[0, 2]]);
        assert!(view[[1, 1]]);
        assert!(!view[[1, 2]]);
    }

    #[test]
    fn direct_state_batch_view_preserves_order() {
        let batch = DirectStateBatch::new(2, 2);
        batch.slot_slice_mut(0)[0] = 1.0;
        batch.slot_slice_mut(0)[34] = 2.0;
        batch.slot_slice_mut(1)[1] = 3.0;
        batch.slot_slice_mut(1)[35] = 4.0;

        let view = batch.view(2).unwrap();
        assert_eq!(view[[0, 0, 0]], 1.0);
        assert_eq!(view[[0, 1, 0]], 2.0);
        assert_eq!(view[[1, 0, 1]], 3.0);
        assert_eq!(view[[1, 1, 1]], 4.0);
    }
}
