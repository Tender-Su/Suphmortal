use super::{BatchAgent, InvisibleState};
use crate::array::Simple2DArray;
use crate::consts::ACTION_SPACE;
use crate::consts::obs_shape;
use crate::mjai::{Event, EventExt, Metadata};
use crate::state::PlayerState;
use crate::{must_tile, tu8};
use std::mem;
use std::sync::Arc;
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

    wg: WaitGroup,
    sync_fields: Arc<Mutex<SyncFields>>,
}

struct SyncFields {
    states: Vec<Simple2DArray<34, f32>>,
    invisible_states: Vec<Array2<f32>>,
    masks: Vec<[bool; ACTION_SPACE]>,
    action_idxs: Vec<usize>,
    kan_action_idxs: Vec<Option<usize>>,
}

fn stack_state_batch(states: Vec<Simple2DArray<34, f32>>, field_name: &str) -> Result<Array3<f32>> {
    ensure!(!states.is_empty(), "empty {field_name} batch");
    let batch_size = states.len();
    let first_rows = states[0].rows();
    let mut stacked = Array3::<f32>::zeros((batch_size, first_rows, 34));
    for (index, state) in states.into_iter().enumerate() {
        ensure!(
            state.rows() == first_rows,
            "mismatched {field_name} shape at batch index {index}",
        );
        let mut view = stacked.slice_mut(s![index, .., ..]);
        let buf = view
            .as_slice_mut()
            .context("stack_state_batch expected contiguous output slice")?;
        buf.copy_from_slice(state.as_slice());
    }
    Ok(stacked)
}

fn stack_array2_state_batch(states: Vec<Array2<f32>>, field_name: &str) -> Result<Array3<f32>> {
    ensure!(!states.is_empty(), "empty {field_name} batch");
    let batch_size = states.len();
    let first_dim = states[0].raw_dim();
    let mut stacked = Array3::<f32>::zeros((batch_size, first_dim[0], first_dim[1]));
    for (index, state) in states.into_iter().enumerate() {
        ensure!(
            state.raw_dim() == first_dim,
            "mismatched {field_name} shape at batch index {index}",
        );
        stacked.slice_mut(s![index, .., ..]).assign(&state);
    }
    Ok(stacked)
}

fn stack_mask_batch(masks: Vec<[bool; ACTION_SPACE]>) -> Result<Array2<bool>> {
    ensure!(!masks.is_empty(), "empty mask batch");
    let batch_size = masks.len();
    let mut stacked = Array2::<bool>::default((batch_size, ACTION_SPACE));
    for (index, mask) in masks.into_iter().enumerate() {
        let mut view = stacked.slice_mut(s![index, ..]);
        let buf = view
            .as_slice_mut()
            .context("stack_mask_batch expected contiguous output slice")?;
        buf.copy_from_slice(&mask);
    }
    Ok(stacked)
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
            let is_oracle = obj.getattr("is_oracle")?.extract()?;
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
        let quick_eval_reactions = if enable_quick_eval {
            vec![None; size]
        } else {
            vec![]
        };
        let sync_fields = Arc::new(Mutex::new(SyncFields {
            states: vec![],
            invisible_states: vec![],
            masks: vec![],
            action_idxs: vec![0; size],
            kan_action_idxs: vec![None; size],
        }));

        Ok(Self {
            engine,
            is_oracle,
            version,
            enable_quick_eval,
            enable_rule_based_agari_guard,
            enable_metadata,
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

            wg: WaitGroup::new(),
            sync_fields,
        })
    }

    fn evaluate(&mut self) -> Result<()> {
        // Wait for all feature encodings to complete.
        mem::take(&mut self.wg).wait();
        let mut sync_fields = self.sync_fields.lock();

        if sync_fields.states.is_empty() {
            return Ok(());
        }

        let start = Instant::now();
        self.last_batch_size = sync_fields.states.len();
        let states = mem::take(&mut sync_fields.states);
        let masks = mem::take(&mut sync_fields.masks);
        let invisible_states = self
            .is_oracle
            .then(|| mem::take(&mut sync_fields.invisible_states));
        drop(sync_fields);

        let states = Python::with_gil(|py| -> Result<_> {
            let states = PyArray3::from_owned_array(py, stack_state_batch(states, "state")?);
            let masks = PyArray2::from_owned_array(py, stack_mask_batch(masks)?);
            let invisible_states = invisible_states
                .map(|batch| stack_array2_state_batch(batch, "invisible_state"))
                .transpose()?
                .map(|batch| PyArray3::from_owned_array(py, batch));

            let args = (states, masks, invisible_states);
            if self.enable_metadata {
                let (actions, q_values, masks_recv, is_greedy) = self
                    .engine
                    .bind_borrowed(py)
                    .call_method1(intern!(py, "react_batch"), args)
                    .context("failed to execute `react_batch` on Python engine")?
                    .extract()
                    .context("failed to extract to Rust type")?;
                Ok((
                    actions,
                    Some(q_values),
                    Some(masks_recv),
                    Some(is_greedy),
                    None,
                ))
            } else {
                let (actions, alt_actions) = self
                    .engine
                    .bind_borrowed(py)
                    .call_method1(intern!(py, "react_batch_action_only"), args)
                    .context("failed to execute `react_batch_action_only` on Python engine")?
                    .extract()
                    .context("failed to extract action-only response to Rust type")?;
                Ok((actions, None, None, None, Some(alt_actions)))
            }
        })?;
        self.actions = states.0;
        self.q_values = states.1.unwrap_or_default();
        self.masks_recv = states.2.unwrap_or_default();
        self.is_greedy = states.3.unwrap_or_default();
        self.alt_actions = states.4.unwrap_or_default();

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
        let state = state.clone();
        let sync_fields = Arc::clone(&self.sync_fields);
        let wg = self.wg.clone();
        rayon::spawn(move || {
            let _wg = wg;

            // Encode features in parallel within the game batch to utilize
            // multiple cores, as this can be very CPU-intensive, especially for
            // the sp feature (since v4).
            let state_rows = obs_shape(version).0;
            let kan = need_kan_select.then(|| {
                let mut kan_feature = Simple2DArray::new(state_rows);
                let mut kan_mask = [false; ACTION_SPACE];
                state.encode_obs_into(version, true, &mut kan_feature, &mut kan_mask);
                (kan_feature, kan_mask)
            });
            let mut feature = Simple2DArray::new(state_rows);
            let mut mask = [false; ACTION_SPACE];
            state.encode_obs_into(version, false, &mut feature, &mut mask);

            let SyncFields {
                states,
                invisible_states,
                masks,
                action_idxs,
                kan_action_idxs,
            } = &mut *sync_fields.lock();
            if let Some((kan_feature, kan_mask)) = kan {
                kan_action_idxs[index] = Some(states.len());
                states.push(kan_feature);
                masks.push(kan_mask);
                if let Some(invisible_state) = invisible_state.clone() {
                    invisible_states.push(invisible_state);
                }
            }

            action_idxs[index] = states.len();
            states.push(feature);
            masks.push(mask);
            if let Some(invisible_state) = invisible_state {
                invisible_states.push(invisible_state);
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

        Ok(EventExt {
            event,
            meta: Some(meta),
        })
    }
}

#[cfg(test)]
mod test {
    use super::{stack_mask_batch, stack_state_batch};
    use crate::array::Simple2DArray;
    use crate::consts::ACTION_SPACE;

    #[test]
    fn stack_state_batch_preserves_order() {
        let mut first = Simple2DArray::new(2);
        first.assign(0, 0, 1.0);
        first.assign(0, 1, 2.0);
        first.assign(1, 0, 3.0);
        first.assign(1, 1, 4.0);
        let mut second = Simple2DArray::new(2);
        second.assign(0, 0, 5.0);
        second.assign(0, 1, 6.0);
        second.assign(1, 0, 7.0);
        second.assign(1, 1, 8.0);
        let stacked = stack_state_batch(vec![first, second], "state").unwrap();

        assert_eq!(stacked.shape(), &[2, 2, 34]);
        assert_eq!(stacked[[0, 1, 0]], 3.0);
        assert_eq!(stacked[[1, 0, 1]], 6.0);
    }

    #[test]
    fn stack_mask_batch_preserves_order() {
        let mut first = [false; ACTION_SPACE];
        first[0] = true;
        first[2] = true;
        let mut second = [false; ACTION_SPACE];
        second[1] = true;
        let stacked = stack_mask_batch(vec![first, second]).unwrap();

        assert_eq!(stacked.shape(), &[2, ACTION_SPACE]);
        assert!(stacked[[0, 2]]);
        assert!(stacked[[1, 1]]);
        assert!(!stacked[[1, 2]]);
    }
}
