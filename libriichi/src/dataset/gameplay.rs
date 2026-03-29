use super::{Grp, Invisible};
use crate::array::Simple2DArray;
use crate::chi_type::ChiType;
use crate::consts::{ACTION_SPACE, obs_shape, oracle_obs_shape};
use crate::mjai::Event;
use crate::state::PlayerState;
use crate::tile::Tile;
use std::array;
use std::fs::File;
use std::io;
use std::mem;
use std::path::Path;

use ahash::AHashSet;
use anyhow::{Context, Result, bail};
use derivative::Derivative;
use flate2::read::GzDecoder;
use ndarray::prelude::*;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json as json;
use tinyvec::ArrayVec;

const CONTEXT_META_DIM: usize = 8;
const DANGER_DISCARD_DIM: usize = 37;
const DANGER_PLAYER_DIM: usize = 3;

#[inline]
fn danger_ron_loss(ron: i32, honba: u8) -> u32 {
    ron.max(0) as u32 + u32::from(honba) * 300
}

#[inline]
fn estimated_player_sample_capacity(event_count: usize) -> usize {
    event_count.div_ceil(4).saturating_add(8).max(16)
}

#[pyclass]
#[derive(Derivative)]
#[derivative(Debug)]
pub struct GameplayLoader {
    #[pyo3(get)]
    version: u32,
    #[pyo3(get)]
    oracle: bool,
    #[pyo3(get)]
    player_names: Vec<String>,
    #[pyo3(get)]
    excludes: Vec<String>,
    #[pyo3(get)]
    trust_seed: bool,
    #[pyo3(get)]
    always_include_kan_select: bool,
    #[pyo3(get)]
    augmented: bool,
    #[pyo3(get)]
    track_opponent_states: bool,
    #[pyo3(get)]
    track_danger_labels: bool,

    #[derivative(Debug = "ignore")]
    player_names_set: AHashSet<String>,
    #[derivative(Debug = "ignore")]
    excludes_set: AHashSet<String>,
}

#[pyclass]
#[derive(Clone, Default)]
pub struct Gameplay {
    // per move
    pub obs: Vec<f32>,
    pub invisible_obs: Vec<Array2<f32>>,
    pub actions: Vec<i64>,
    pub masks: Vec<bool>,
    pub at_kyoku: Vec<u8>,
    pub dones: Vec<bool>,
    pub apply_gamma: Vec<bool>,
    pub at_turns: Vec<u8>,
    pub shantens: Vec<i8>,
    pub context_meta: Vec<u16>,
    pub opponent_shanten: Vec<u8>,
    pub opponent_tenpai: Vec<u8>,
    pub danger_valid: Vec<bool>,
    pub danger_any: Vec<bool>,
    pub danger_value: Vec<f32>,
    pub danger_player_mask: Vec<bool>,

    // per game
    pub grp: Grp, // actually per kyoku though
    pub player_id: u8,
    pub player_name: String,
    pub sample_count: usize,
    pub version: u32,
}

struct LoaderContext<'a> {
    config: &'a GameplayLoader,
    invisibles: Option<&'a [Invisible]>,

    state: PlayerState,
    kyoku_idx: usize,

    // fields below are only used for oracle
    opponent_states: [PlayerState; 3],
    from_rinshan: bool,
    yama_idx: usize,
    rinshan_idx: usize,
    obs_scratch: Simple2DArray<34, f32>,
    mask_scratch: [bool; ACTION_SPACE],
}

#[derive(Debug, Serialize, Deserialize)]
struct EventCacheEntry {
    path: String,
    events: Vec<Event>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EventCacheChunk {
    format: String,
    entries: Vec<EventCacheEntry>,
}

#[pymethods]
impl GameplayLoader {
    #[new]
    #[pyo3(signature = (
        version,
        *,
        oracle = true,
        player_names = None,
        excludes = None,
        trust_seed = false,
        always_include_kan_select = true,
        augmented = false,
        track_opponent_states = false,
        track_danger_labels = false,
    ))]
    fn new(
        version: u32,
        oracle: bool,
        player_names: Option<Vec<String>>,
        excludes: Option<Vec<String>>,
        trust_seed: bool,
        always_include_kan_select: bool,
        augmented: bool,
        track_opponent_states: bool,
        track_danger_labels: bool,
    ) -> Self {
        let player_names = player_names.unwrap_or_default();
        let player_names_set = player_names.iter().cloned().collect();
        let excludes = excludes.unwrap_or_default();
        let excludes_set = excludes.iter().cloned().collect();
        Self {
            version,
            oracle,
            player_names,
            excludes,
            trust_seed,
            always_include_kan_select,
            augmented,
            track_opponent_states,
            track_danger_labels,
            player_names_set,
            excludes_set,
        }
    }

    // Nested result is too hard to handle...
    fn load_log(&self, raw_log: &str) -> Result<Vec<Gameplay>> {
        let events = self.parse_events(raw_log, true)?;
        self.load_events(&events)
    }

    #[pyo3(name = "load_gz_log_files")]
    fn load_gz_log_files_py(&self, gzip_filenames: Vec<String>) -> Result<Vec<Vec<Gameplay>>> {
        self.load_log_files(gzip_filenames)
    }

    #[pyo3(name = "load_log_files")]
    fn load_log_files_py(&self, filenames: Vec<String>) -> Result<Vec<Vec<Gameplay>>> {
        self.load_log_files(filenames)
    }

    #[pyo3(name = "load_logs")]
    fn load_logs_py(&self, raw_logs: Vec<String>) -> Result<Vec<Vec<Gameplay>>> {
        self.load_logs(raw_logs)
    }

    #[pyo3(name = "build_event_cache_file")]
    fn build_event_cache_file_py(
        &self,
        filenames: Vec<String>,
        output_filename: String,
    ) -> Result<usize> {
        self.build_event_cache_file(filenames, output_filename)
    }

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

impl GameplayLoader {
    fn parse_events(&self, raw_log: &str, apply_augmentation: bool) -> Result<Vec<Event>> {
        let mut events = raw_log
            .lines()
            .map(json::from_str)
            .collect::<Result<Vec<Event>, _>>()
            .context("failed to parse log")?;
        if apply_augmentation && self.augmented {
            events.iter_mut().for_each(Event::augment);
        }
        Ok(events)
    }

    fn read_raw_log_from_file(&self, filename: &str) -> Result<String> {
        let file = File::open(filename)?;
        let raw = if Path::new(filename)
            .extension()
            .is_some_and(|s| s.eq_ignore_ascii_case("gz"))
        {
            io::read_to_string(GzDecoder::new(file))?
        } else {
            io::read_to_string(file)?
        };
        Ok(raw)
    }

    fn load_event_cache_file(&self, filename: &str) -> Result<Vec<Vec<Gameplay>>> {
        let file = File::open(filename)?;
        let reader = io::BufReader::new(file);
        let mut decoder = zstd::stream::read::Decoder::new(reader)?;
        let chunk: EventCacheChunk = rmp_serde::from_read(&mut decoder)
            .with_context(|| format!("failed to deserialize event cache {filename}"))?;

        chunk
            .entries
            .into_par_iter()
            .map(|mut entry| {
                if self.augmented {
                    entry.events.iter_mut().for_each(Event::augment);
                }
                self.load_events(&entry.events)
            })
            .collect()
    }

    pub fn load_log_files<V, S>(&self, filenames: V) -> Result<Vec<Vec<Gameplay>>>
    where
        V: IntoParallelIterator<Item = S>,
        S: AsRef<str>,
    {
        let nested = filenames
            .into_par_iter()
            .map(|f| {
                let filename = f.as_ref();
                let inner = || {
                    if filename.ends_with(".events.zst") {
                        self.load_event_cache_file(filename)
                    } else {
                        let raw = self.read_raw_log_from_file(filename)?;
                        self.load_log(&raw).map(|v| vec![v])
                    }
                };
                inner().with_context(|| format!("error when reading {filename}"))
            })
            .collect::<Result<Vec<Vec<Vec<Gameplay>>>>>()?;

        Ok(nested.into_iter().flatten().collect())
    }

    pub fn load_logs<V, S>(&self, raw_logs: V) -> Result<Vec<Vec<Gameplay>>>
    where
        V: IntoParallelIterator<Item = S>,
        S: AsRef<str>,
    {
        raw_logs
            .into_par_iter()
            .map(|raw| self.load_log(raw.as_ref()))
            .collect()
    }

    pub fn load_gz_log_files<V, S>(&self, gzip_filenames: V) -> Result<Vec<Vec<Gameplay>>>
    where
        V: IntoParallelIterator<Item = S>,
        S: AsRef<str>,
    {
        self.load_log_files(gzip_filenames)
    }

    pub fn build_event_cache_file<V, S>(
        &self,
        filenames: V,
        output_filename: String,
    ) -> Result<usize>
    where
        V: IntoParallelIterator<Item = S>,
        S: AsRef<str>,
    {
        let entries = filenames
            .into_par_iter()
            .map(|f| {
                let filename = f.as_ref();
                let inner = || -> Result<EventCacheEntry> {
                    let raw = self.read_raw_log_from_file(filename)?;
                    let events = self.parse_events(&raw, false)?;
                    Ok(EventCacheEntry {
                        path: filename.to_owned(),
                        events,
                    })
                };
                inner().with_context(|| format!("error when caching {filename}"))
            })
            .collect::<Result<Vec<_>>>()?;

        let count = entries.len();
        let chunk = EventCacheChunk {
            format: "gameplay_event_cache_v1".to_owned(),
            entries,
        };

        let file = File::create(&output_filename)?;
        let writer = io::BufWriter::new(file);
        let mut encoder = zstd::stream::write::Encoder::new(writer, 3)?;
        let mut serializer = rmp_serde::Serializer::new(Vec::new()).with_struct_map();
        chunk
            .serialize(&mut serializer)
            .with_context(|| format!("failed to serialize event cache to {output_filename}"))?;
        let bytes = serializer.into_inner();
        use std::io::Write;
        encoder.write_all(&bytes)?;
        encoder.finish()?;
        Ok(count)
    }

    pub fn load_events(&self, events: &[Event]) -> Result<Vec<Gameplay>> {
        let invisibles = self.oracle.then(|| Invisible::new(events, self.trust_seed));

        let [Event::StartGame { names, .. }, ..] = events else {
            bail!("empty or invalid game log");
        };
        let player_ids = names
            .iter()
            .enumerate()
            .filter(|&(_, name)| {
                if !self.player_names_set.is_empty() {
                    return self.player_names_set.contains(name);
                }
                if !self.excludes_set.is_empty() {
                    return !self.excludes_set.contains(name);
                }
                true
            })
            .map(|(i, _)| i as u8)
            .collect::<ArrayVec<[_; 4]>>();
        let grp = Grp::load_events(events)?;
        // `events.len()` is the whole-log raw event count, while each Gameplay only
        // stores one player's supervised decision samples. Reserving from the raw
        // event count over-allocates the heavy obs/mask buffers by several times.
        let sample_capacity = estimated_player_sample_capacity(events.len());

        let mut gameplays = player_ids
            .into_iter()
            .map(|player_id| {
                (
                    Gameplay::new_with_capacity(
                        player_id,
                        grp.clone(),
                        self.version,
                        sample_capacity,
                    ),
                    LoaderContext {
                        config: self,
                        invisibles: invisibles.as_deref(),
                        state: PlayerState::new(player_id),
                        kyoku_idx: 0,
                        opponent_states: array::from_fn(|i| {
                            PlayerState::new((player_id + i as u8 + 1) % 4)
                        }),
                        from_rinshan: false,
                        yama_idx: 0,
                        rinshan_idx: 0,
                        obs_scratch: Simple2DArray::new(obs_shape(self.version).0),
                        mask_scratch: [false; ACTION_SPACE],
                    },
                )
            })
            .collect::<Vec<_>>();

        for wnd in events.windows(4) {
            let wnd: &[Event; 4] = wnd.try_into().unwrap();
            for (gameplay, ctx) in &mut gameplays {
                gameplay.extend_from_event_window(ctx, wnd)?;
            }
        }

        Ok(gameplays
            .into_iter()
            .map(|(mut gameplay, _)| {
                gameplay.finalize_round_markers();
                gameplay
            })
            .collect())
    }
}

#[pymethods]
impl Gameplay {
    fn take_obs<'py>(&mut self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f32>>> {
        let batch = self.take_obs_batch_array();
        batch
            .outer_iter()
            .map(|v| PyArray2::from_owned_array(py, v.to_owned()))
            .collect()
    }
    fn take_obs_batch<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        PyArray3::from_owned_array(py, self.take_obs_batch_array())
    }
    fn take_invisible_obs<'py>(&mut self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f32>>> {
        mem::take(&mut self.invisible_obs)
            .into_iter()
            .map(|v| PyArray2::from_owned_array(py, v))
            .collect()
    }
    fn take_invisible_obs_batch<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        PyArray3::from_owned_array(py, self.take_invisible_obs_batch_array())
    }
    fn take_actions(&mut self) -> Vec<i64> {
        mem::take(&mut self.actions)
    }
    fn take_actions_batch<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(py, mem::take(&mut self.actions))
    }
    fn take_masks<'py>(&mut self, py: Python<'py>) -> Vec<Bound<'py, PyArray1<bool>>> {
        let batch = self.take_masks_batch_array();
        batch
            .outer_iter()
            .map(|v| PyArray1::from_owned_array(py, v.to_owned()))
            .collect()
    }
    fn take_masks_batch<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<bool>> {
        PyArray2::from_owned_array(py, self.take_masks_batch_array())
    }
    fn take_at_kyoku(&mut self) -> Vec<u8> {
        mem::take(&mut self.at_kyoku)
    }
    fn take_at_kyoku_batch<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        PyArray1::from_vec(py, mem::take(&mut self.at_kyoku))
    }
    fn take_dones(&mut self) -> Vec<bool> {
        mem::take(&mut self.dones)
    }
    fn take_apply_gamma(&mut self) -> Vec<bool> {
        mem::take(&mut self.apply_gamma)
    }
    fn take_at_turns(&mut self) -> Vec<u8> {
        mem::take(&mut self.at_turns)
    }
    fn take_shantens(&mut self) -> Vec<i8> {
        mem::take(&mut self.shantens)
    }
    fn take_context_meta_batch<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<u16>> {
        PyArray2::from_owned_array(py, self.take_context_meta_batch_array())
    }
    fn take_opponent_shanten_batch<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        PyArray2::from_owned_array(py, self.take_opponent_shanten_batch_array())
    }
    fn take_opponent_tenpai_batch<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        PyArray2::from_owned_array(py, self.take_opponent_tenpai_batch_array())
    }
    fn take_danger_valid_batch<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_owned_array(py, self.take_danger_valid_batch_array())
    }
    fn take_danger_any_batch<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<bool>> {
        PyArray2::from_owned_array(py, self.take_danger_any_batch_array())
    }
    fn take_danger_value_batch<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        PyArray2::from_owned_array(py, self.take_danger_value_batch_array())
    }
    fn take_danger_player_mask_batch<'py>(
        &mut self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray3<bool>> {
        PyArray3::from_owned_array(py, self.take_danger_player_mask_batch_array())
    }

    fn take_grp(&mut self) -> Grp {
        mem::take(&mut self.grp)
    }

    const fn take_player_id(&self) -> u8 {
        self.player_id
    }
}

impl Gameplay {
    fn take_obs_batch_array(&mut self) -> Array3<f32> {
        let obs_channels = obs_shape(self.version).0;
        let values = mem::take(&mut self.obs);
        let sample_count = values.len() / (obs_channels * 34);
        Array3::from_shape_vec((sample_count, obs_channels, 34), values).unwrap()
    }

    fn take_invisible_obs_batch_array(&mut self) -> Array3<f32> {
        let channels = oracle_obs_shape(self.version).0;
        let values = mem::take(&mut self.invisible_obs);
        let sample_count = values.len();
        let mut arr = Array3::zeros((sample_count, channels, 34));
        for (idx, obs) in values.into_iter().enumerate() {
            arr.index_axis_mut(Axis(0), idx).assign(&obs);
        }
        arr
    }

    fn take_masks_batch_array(&mut self) -> Array2<bool> {
        let values = mem::take(&mut self.masks);
        let sample_count = values.len() / ACTION_SPACE;
        Array2::from_shape_vec((sample_count, ACTION_SPACE), values).unwrap()
    }

    fn take_opponent_shanten_batch_array(&mut self) -> Array2<u8> {
        let values = mem::take(&mut self.opponent_shanten);
        let sample_count = values.len() / 3;
        Array2::from_shape_vec((sample_count, 3), values).unwrap()
    }

    fn take_context_meta_batch_array(&mut self) -> Array2<u16> {
        let values = mem::take(&mut self.context_meta);
        let sample_count = values.len() / CONTEXT_META_DIM;
        Array2::from_shape_vec((sample_count, CONTEXT_META_DIM), values).unwrap()
    }

    fn take_opponent_tenpai_batch_array(&mut self) -> Array2<u8> {
        let values = mem::take(&mut self.opponent_tenpai);
        let sample_count = values.len() / 3;
        Array2::from_shape_vec((sample_count, 3), values).unwrap()
    }

    fn take_danger_any_batch_array(&mut self) -> Array2<bool> {
        let values = mem::take(&mut self.danger_any);
        let sample_count = values.len() / DANGER_DISCARD_DIM;
        Array2::from_shape_vec((sample_count, DANGER_DISCARD_DIM), values).unwrap()
    }

    fn take_danger_valid_batch_array(&mut self) -> Array1<bool> {
        let values = mem::take(&mut self.danger_valid);
        Array1::from_shape_vec(values.len(), values).unwrap()
    }

    fn take_danger_value_batch_array(&mut self) -> Array2<f32> {
        let values = mem::take(&mut self.danger_value);
        let sample_count = values.len() / DANGER_DISCARD_DIM;
        Array2::from_shape_vec((sample_count, DANGER_DISCARD_DIM), values).unwrap()
    }

    fn take_danger_player_mask_batch_array(&mut self) -> Array3<bool> {
        let values = mem::take(&mut self.danger_player_mask);
        let sample_count = values.len() / (DANGER_DISCARD_DIM * DANGER_PLAYER_DIM);
        Array3::from_shape_vec(
            (sample_count, DANGER_DISCARD_DIM, DANGER_PLAYER_DIM),
            values,
        )
        .unwrap()
    }

    fn new_with_capacity(player_id: u8, grp: Grp, version: u32, sample_capacity: usize) -> Self {
        let obs_capacity = sample_capacity * obs_shape(version).0 * 34;
        let mask_capacity = sample_capacity * ACTION_SPACE;
        Self {
            obs: Vec::with_capacity(obs_capacity),
            invisible_obs: Vec::with_capacity(sample_capacity),
            actions: Vec::with_capacity(sample_capacity),
            masks: Vec::with_capacity(mask_capacity),
            at_kyoku: Vec::with_capacity(sample_capacity),
            dones: Vec::with_capacity(sample_capacity),
            apply_gamma: Vec::with_capacity(sample_capacity),
            at_turns: Vec::with_capacity(sample_capacity),
            shantens: Vec::with_capacity(sample_capacity),
            context_meta: Vec::with_capacity(sample_capacity * CONTEXT_META_DIM),
            opponent_shanten: Vec::with_capacity(sample_capacity * 3),
            opponent_tenpai: Vec::with_capacity(sample_capacity * 3),
            danger_valid: Vec::with_capacity(sample_capacity),
            danger_any: Vec::with_capacity(sample_capacity * DANGER_DISCARD_DIM),
            danger_value: Vec::with_capacity(sample_capacity * DANGER_DISCARD_DIM),
            danger_player_mask: Vec::with_capacity(
                sample_capacity * DANGER_DISCARD_DIM * DANGER_PLAYER_DIM,
            ),
            grp,
            player_id,
            player_name: String::new(),
            sample_count: 0,
            version: version as u32,
        }
    }

    fn finalize_round_markers(&mut self) {
        self.dones = self.at_kyoku.windows(2).map(|w| w[1] > w[0]).collect();
        self.dones.push(true);
    }

    fn push_danger_labels(
        &mut self,
        state: &PlayerState,
        opponent_states: &[PlayerState; DANGER_PLAYER_DIM],
        at_kan_select: bool,
    ) {
        let danger_valid = !at_kan_select && state.last_cans().can_discard;
        self.danger_valid.push(danger_valid);

        let mut any = [false; DANGER_DISCARD_DIM];
        let mut value = [0_f32; DANGER_DISCARD_DIM];
        let mut player_mask = [false; DANGER_DISCARD_DIM * DANGER_PLAYER_DIM];

        if danger_valid {
            let honba = state.honba();
            for (discard_idx, eligible) in state.discard_candidates_aka().into_iter().enumerate() {
                if !eligible {
                    continue;
                }

                let tile = Tile::new_unchecked(discard_idx as u8);
                let player_mask_row =
                    &mut player_mask[discard_idx * DANGER_PLAYER_DIM..][..DANGER_PLAYER_DIM];

                for (opponent_idx, opponent_state) in opponent_states.iter().enumerate() {
                    if let Some(point) = opponent_state.ron_point_on_tile(tile) {
                        player_mask_row[opponent_idx] = true;
                        any[discard_idx] = true;
                        value[discard_idx] =
                            value[discard_idx].max(danger_ron_loss(point.ron, honba) as f32);
                    }
                }
            }
        }

        self.danger_any.extend_from_slice(&any);
        self.danger_value.extend_from_slice(&value);
        self.danger_player_mask.extend_from_slice(&player_mask);
    }

    #[cfg(test)]
    fn load_events_by_player(
        config: &GameplayLoader,
        events: &[Event],
        player_id: u8,
        invisibles: Option<&[Invisible]>,
    ) -> Result<Self> {
        let grp = Grp::load_events(events)?;
        let mut data = Self::new_with_capacity(
            player_id,
            grp,
            config.version,
            estimated_player_sample_capacity(events.len()),
        );

        let mut ctx = LoaderContext {
            config,
            invisibles,
            state: PlayerState::new(player_id),
            kyoku_idx: 0,
            // end_state: EndState::Passive,
            opponent_states: array::from_fn(|i| PlayerState::new((player_id + i as u8 + 1) % 4)),
            from_rinshan: false,
            yama_idx: 0,
            rinshan_idx: 0,
            obs_scratch: Simple2DArray::new(obs_shape(config.version).0),
            mask_scratch: [false; ACTION_SPACE],
        };

        // It is guaranteed that there are at least 4 events.
        // tsumo/dahai -> ryukyoku/hora -> end kyoku -> end game
        for wnd in events.windows(4) {
            data.extend_from_event_window(&mut ctx, wnd.try_into().unwrap())?;
        }

        data.finalize_round_markers();

        Ok(data)
    }

    fn extend_from_event_window(
        &mut self,
        ctx: &mut LoaderContext<'_>,
        wnd: &[Event; 4],
    ) -> Result<()> {
        let LoaderContext {
            config,
            invisibles,
            state,
            kyoku_idx,
            opponent_states,
            from_rinshan,
            yama_idx,
            rinshan_idx,
            ..
        } = ctx;

        let cur = &wnd[0];
        let next = if matches!(wnd[1], Event::ReachAccepted { .. } | Event::Dora { .. }) {
            &wnd[2]
        } else {
            &wnd[1]
        };

        match cur {
            Event::StartGame { names, .. } => {
                self.player_name.clone_from(&names[self.player_id as usize]);
            }
            Event::EndKyoku => *kyoku_idx += 1,
            _ => (),
        }

        if invisibles.is_some() {
            match cur {
                Event::EndKyoku => {
                    *from_rinshan = false;
                    *yama_idx = 0;
                    *rinshan_idx = 0;
                }
                Event::Tsumo { .. } => {
                    if *from_rinshan {
                        *rinshan_idx += 1;
                        *from_rinshan = false;
                    } else {
                        *yama_idx += 1;
                    }
                }
                Event::Ankan { .. } | Event::Kakan { .. } | Event::Daiminkan { .. } => {
                    *from_rinshan = true;
                }
                _ => (),
            };
        }

        if invisibles.is_some() || config.track_opponent_states || config.track_danger_labels {
            for s in opponent_states {
                s.update(cur)?;
            }
        }

        let cans = state.update(cur)?;
        if !cans.can_act() {
            return Ok(());
        }

        let mut kan_select = None;
        let label_opt = match *next {
            Event::Dahai { pai, .. } => Some(pai.as_usize()),
            Event::Reach { .. } => Some(37),
            Event::Chi {
                actor,
                pai,
                consumed,
                ..
            } if actor == self.player_id => match ChiType::new(consumed, pai) {
                ChiType::Low => Some(38),
                ChiType::Mid => Some(39),
                ChiType::High => Some(40),
            },
            Event::Pon { actor, .. } if actor == self.player_id => Some(41),
            Event::Daiminkan { actor, pai, .. } if actor == self.player_id => {
                if config.always_include_kan_select {
                    kan_select = Some(pai.deaka().as_usize());
                }
                Some(42)
            }
            Event::Kakan { pai, .. } => {
                if config.always_include_kan_select || state.kakan_candidates().len() > 1 {
                    kan_select = Some(pai.deaka().as_usize());
                }
                Some(42)
            }
            Event::Ankan { consumed, .. } => {
                if config.always_include_kan_select || state.ankan_candidates().len() > 1 {
                    kan_select = Some(consumed[0].deaka().as_usize());
                }
                Some(42)
            }
            Event::Ryukyoku { .. } if cans.can_ryukyoku => Some(44),
            _ => {
                let mut ret = None;

                let has_any_ron = matches!(wnd[1], Event::Hora { .. });
                if has_any_ron {
                    // Check if the POV is one of those who made Hora.
                    for ev in &wnd[1..] {
                        match *ev {
                            Event::EndKyoku { .. } => break,
                            Event::Hora { actor, .. } if actor == self.player_id => {
                                ret = Some(43);
                                break;
                            }
                            _ => (),
                        };
                    }
                }

                if ret.is_none() {
                    // It is now proven there is no ron from the POV.
                    if cans.can_chi() && matches!(next, Event::Tsumo { .. })
                        || (cans.can_pon || cans.can_daiminkan || cans.can_ron_agari)
                            && !has_any_ron
                    {
                        // Can chi, but actively denied instead of being
                        // interrupted by other's pon/daiminkan/ron.
                        //
                        // or
                        //
                        // Can pon/daiminkan/ron, but actively denied
                        // instead of being interrupted by other's ron.
                        ret = Some(45);
                    }
                }

                ret
            }
        };

        if let Some(label) = label_opt {
            self.add_entry(ctx, false, label);
            if let Some(kan) = kan_select {
                self.add_entry(ctx, true, kan);
            }
        }
        Ok(())
    }

    fn add_entry(&mut self, ctx: &mut LoaderContext<'_>, at_kan_select: bool, label: usize) {
        ctx.obs_scratch.reset();
        ctx.mask_scratch.fill(false);
        ctx.state.encode_obs_into(
            ctx.config.version,
            at_kan_select,
            &mut ctx.obs_scratch,
            &mut ctx.mask_scratch,
        );
        self.obs.extend_from_slice(ctx.obs_scratch.as_slice());
        self.actions.push(label as i64);
        self.masks.extend_from_slice(&ctx.mask_scratch);
        self.at_kyoku.push(ctx.kyoku_idx as u8);
        // only discard and kan will discount
        self.apply_gamma.push(label <= 37);
        self.at_turns.push(ctx.state.at_turn());
        self.shantens.push(ctx.state.shanten());
        let scores = ctx.state.scores();
        let self_score = scores[0];
        let rank = ctx.state.rank() as usize;
        let mut sorted_scores = scores;
        sorted_scores.sort_by(|a, b| b.cmp(a));
        let up_gap = if rank > 0 {
            (sorted_scores[rank - 1] - self_score).max(0)
        } else {
            i32::MAX
        };
        let down_gap = if rank < 3 {
            (self_score - sorted_scores[rank + 1]).max(0)
        } else {
            i32::MAX
        };
        let encode_gap = |gap: i32| -> u16 {
            if gap == i32::MAX {
                u16::MAX
            } else {
                (gap.clamp(0, i32::from(u16::MAX) * 100) / 100) as u16
            }
        };

        self.context_meta.push(u16::from(ctx.state.at_turn()));
        self.context_meta
            .push(u16::from(ctx.state.is_south_or_later()));
        self.context_meta.push(u16::from(ctx.state.is_oya()));
        self.context_meta.push(u16::from(ctx.state.is_all_last()));
        self.context_meta.push(u16::from(ctx.state.rank()));
        self.context_meta
            .push(u16::from(ctx.state.num_opp_riichi_accepted()));
        self.context_meta.push(encode_gap(up_gap));
        self.context_meta.push(encode_gap(down_gap));
        if ctx.invisibles.is_some() || ctx.config.track_opponent_states {
            for opponent_state in &ctx.opponent_states {
                let shanten = opponent_state.shanten();
                let shanten_bucket = if shanten <= 0 {
                    0
                } else if shanten == 1 {
                    1
                } else if shanten == 2 {
                    2
                } else {
                    3
                };
                self.opponent_shanten.push(shanten_bucket);
                self.opponent_tenpai.push(u8::from(shanten <= 0));
            }
        }
        if ctx.config.track_danger_labels {
            self.push_danger_labels(&ctx.state, &ctx.opponent_states, at_kan_select);
        }
        self.sample_count += 1;

        if let Some(invisibles) = ctx.invisibles {
            let invisible_obs = invisibles[ctx.kyoku_idx].encode(
                &ctx.opponent_states,
                ctx.yama_idx,
                ctx.rinshan_idx,
                ctx.config.version,
            );
            self.invisible_obs.push(invisible_obs);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn danger_ron_loss_adds_honba() {
        assert_eq!(danger_ron_loss(5800, 0), 5800);
        assert_eq!(danger_ron_loss(5800, 1), 6100);
        assert_eq!(danger_ron_loss(7700, 3), 8600);
    }

    fn assert_gameplay_eq(actual: &Gameplay, expected: &Gameplay) {
        assert_eq!(actual.player_id, expected.player_id);
        assert_eq!(actual.player_name, expected.player_name);
        assert_eq!(actual.actions, expected.actions);
        assert_eq!(actual.at_kyoku, expected.at_kyoku);
        assert_eq!(actual.dones, expected.dones);
        assert_eq!(actual.apply_gamma, expected.apply_gamma);
        assert_eq!(actual.at_turns, expected.at_turns);
        assert_eq!(actual.shantens, expected.shantens);
        assert_eq!(actual.context_meta, expected.context_meta);
        assert_eq!(actual.opponent_shanten, expected.opponent_shanten);
        assert_eq!(actual.opponent_tenpai, expected.opponent_tenpai);
        assert_eq!(actual.danger_valid, expected.danger_valid);
        assert_eq!(actual.danger_any, expected.danger_any);
        assert_eq!(actual.danger_value, expected.danger_value);
        assert_eq!(actual.danger_player_mask, expected.danger_player_mask);
        assert_eq!(actual.grp.rank_by_player, expected.grp.rank_by_player);
        assert_eq!(actual.grp.final_scores, expected.grp.final_scores);
        assert_eq!(actual.grp.feature, expected.grp.feature);
        assert_eq!(actual.obs, expected.obs);
        assert_eq!(actual.invisible_obs, expected.invisible_obs);
        assert_eq!(actual.masks, expected.masks);
    }

    #[test]
    fn load_events_matches_per_player_replay() {
        let raw_log = r#"
{"type":"start_game","names":["A","B","C","D"]}
{"type":"start_kyoku","bakaze":"S","dora_marker":"5m","kyoku":4,"honba":0,"kyotaku":0,"oya":3,"scores":[35300,3000,38400,23300],"tehais":[["4m","5mr","8m","1p","3p","3p","5p","2s","5sr","9s","W","P","P"],["2m","3m","5m","7m","7p","9p","4s","5s","5s","6s","7s","7s","E"],["3m","5m","6m","2p","6p","9p","1s","5s","8s","9s","S","S","C"],["1m","4m","3p","4p","5pr","7p","1s","2s","7s","8s","W","N","P"]]}
{"type":"tsumo","actor":3,"pai":"F"}
{"type":"dahai","actor":3,"pai":"1m","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"5p"}
{"type":"dahai","actor":0,"pai":"W","tsumogiri":false}
{"type":"tsumo","actor":1,"pai":"9m"}
{"type":"dahai","actor":1,"pai":"E","tsumogiri":false}
{"type":"tsumo","actor":2,"pai":"N"}
{"type":"dahai","actor":2,"pai":"9p","tsumogiri":false}
{"type":"tsumo","actor":3,"pai":"2p"}
{"type":"dahai","actor":3,"pai":"N","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"6m"}
{"type":"dahai","actor":0,"pai":"9s","tsumogiri":false}
{"type":"tsumo","actor":1,"pai":"7m"}
{"type":"dahai","actor":1,"pai":"9m","tsumogiri":false}
{"type":"tsumo","actor":2,"pai":"3s"}
{"type":"dahai","actor":2,"pai":"2p","tsumogiri":false}
{"type":"tsumo","actor":3,"pai":"4s"}
{"type":"dahai","actor":3,"pai":"W","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"1m"}
{"type":"dahai","actor":0,"pai":"1m","tsumogiri":true}
{"type":"tsumo","actor":1,"pai":"9m"}
{"type":"dahai","actor":1,"pai":"9m","tsumogiri":true}
{"type":"tsumo","actor":2,"pai":"3m"}
{"type":"dahai","actor":2,"pai":"N","tsumogiri":false}
{"type":"tsumo","actor":3,"pai":"2s"}
{"type":"dahai","actor":3,"pai":"F","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"2m"}
{"type":"dahai","actor":0,"pai":"2s","tsumogiri":false}
{"type":"tsumo","actor":1,"pai":"1m"}
{"type":"dahai","actor":1,"pai":"5m","tsumogiri":false}
{"type":"tsumo","actor":2,"pai":"3p"}
{"type":"dahai","actor":2,"pai":"3p","tsumogiri":true}
{"type":"pon","actor":0,"target":2,"pai":"3p","consumed":["3p","3p"]}
{"type":"dahai","actor":0,"pai":"2m","tsumogiri":false}
{"type":"tsumo","actor":1,"pai":"6p"}
{"type":"dahai","actor":1,"pai":"9p","tsumogiri":false}
{"type":"tsumo","actor":2,"pai":"6s"}
{"type":"dahai","actor":2,"pai":"C","tsumogiri":false}
{"type":"tsumo","actor":3,"pai":"7p"}
{"type":"dahai","actor":3,"pai":"P","tsumogiri":false}
{"type":"pon","actor":0,"target":3,"pai":"P","consumed":["P","P"]}
{"type":"dahai","actor":0,"pai":"1p","tsumogiri":false}
{"type":"tsumo","actor":1,"pai":"7s"}
{"type":"dahai","actor":1,"pai":"5s","tsumogiri":false}
{"type":"tsumo","actor":2,"pai":"3s"}
{"type":"dahai","actor":2,"pai":"9s","tsumogiri":false}
{"type":"tsumo","actor":3,"pai":"2m"}
{"type":"dahai","actor":3,"pai":"1s","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"1p"}
{"type":"dahai","actor":0,"pai":"1p","tsumogiri":true}
{"type":"tsumo","actor":1,"pai":"7m"}
{"type":"dahai","actor":1,"pai":"4s","tsumogiri":false}
{"type":"chi","actor":2,"target":1,"pai":"4s","consumed":["5s","6s"]}
{"type":"dahai","actor":2,"pai":"6p","tsumogiri":false}
{"type":"chi","actor":3,"target":2,"pai":"6p","consumed":["5pr","7p"]}
{"type":"dahai","actor":3,"pai":"7p","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"1s"}
{"type":"dahai","actor":0,"pai":"1s","tsumogiri":true}
{"type":"tsumo","actor":1,"pai":"1s"}
{"type":"reach","actor":1}
{"type":"dahai","actor":1,"pai":"1s","tsumogiri":true}
{"type":"reach_accepted","actor":1}
{"type":"tsumo","actor":2,"pai":"9s"}
{"type":"dahai","actor":2,"pai":"8s","tsumogiri":false}
{"type":"tsumo","actor":3,"pai":"4p"}
{"type":"dahai","actor":3,"pai":"4p","tsumogiri":true}
{"type":"tsumo","actor":0,"pai":"4m"}
{"type":"dahai","actor":0,"pai":"4m","tsumogiri":true}
{"type":"tsumo","actor":1,"pai":"1p"}
{"type":"dahai","actor":1,"pai":"1p","tsumogiri":true}
{"type":"tsumo","actor":2,"pai":"8m"}
{"type":"dahai","actor":2,"pai":"8m","tsumogiri":true}
{"type":"tsumo","actor":3,"pai":"C"}
{"type":"dahai","actor":3,"pai":"C","tsumogiri":true}
{"type":"tsumo","actor":0,"pai":"2s"}
{"type":"dahai","actor":0,"pai":"2s","tsumogiri":true}
{"type":"tsumo","actor":1,"pai":"8p"}
{"type":"hora","actor":1,"target":1,"deltas":[-1000,4000,-1000,-2000]}
{"type":"end_kyoku"}
{"type":"end_game"}
"#;

        let loader = GameplayLoader::new(4, false, None, None, false, true, false, false, false);
        let events = loader.parse_events(raw_log.trim(), false).unwrap();
        let actual = loader.load_events(&events).unwrap();
        let expected = [0_u8, 1, 2, 3]
            .into_iter()
            .map(|player_id| Gameplay::load_events_by_player(&loader, &events, player_id, None))
            .collect::<Result<Vec<_>>>()
            .unwrap();

        assert_eq!(actual.len(), expected.len());
        for (actual_gameplay, expected_gameplay) in actual.iter().zip(expected.iter()) {
            assert_gameplay_eq(actual_gameplay, expected_gameplay);
        }
    }

    #[test]
    fn estimated_player_sample_capacity_stays_below_raw_event_count_and_covers_actual_samples() {
        let raw_log = r#"
{"type":"start_game","names":["A","B","C","D"]}
{"type":"start_kyoku","bakaze":"S","dora_marker":"5m","kyoku":4,"honba":0,"kyotaku":0,"oya":3,"scores":[35300,3000,38400,23300],"tehais":[["4m","5mr","8m","1p","3p","3p","5p","2s","5sr","9s","W","P","P"],["2m","3m","5m","7m","7p","9p","4s","5s","5s","6s","7s","7s","E"],["3m","5m","6m","2p","6p","9p","1s","5s","8s","9s","S","S","C"],["1m","4m","3p","4p","5pr","7p","1s","2s","7s","8s","W","N","P"]]}
{"type":"tsumo","actor":3,"pai":"F"}
{"type":"dahai","actor":3,"pai":"1m","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"5p"}
{"type":"dahai","actor":0,"pai":"W","tsumogiri":false}
{"type":"tsumo","actor":1,"pai":"9m"}
{"type":"dahai","actor":1,"pai":"E","tsumogiri":false}
{"type":"tsumo","actor":2,"pai":"N"}
{"type":"dahai","actor":2,"pai":"9p","tsumogiri":false}
{"type":"tsumo","actor":3,"pai":"2p"}
{"type":"dahai","actor":3,"pai":"N","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"6m"}
{"type":"dahai","actor":0,"pai":"9s","tsumogiri":false}
{"type":"tsumo","actor":1,"pai":"7m"}
{"type":"dahai","actor":1,"pai":"9m","tsumogiri":false}
{"type":"tsumo","actor":2,"pai":"3s"}
{"type":"dahai","actor":2,"pai":"2p","tsumogiri":false}
{"type":"tsumo","actor":3,"pai":"4s"}
{"type":"dahai","actor":3,"pai":"W","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"1m"}
{"type":"dahai","actor":0,"pai":"1m","tsumogiri":true}
{"type":"tsumo","actor":1,"pai":"9m"}
{"type":"dahai","actor":1,"pai":"9m","tsumogiri":true}
{"type":"tsumo","actor":2,"pai":"3m"}
{"type":"dahai","actor":2,"pai":"N","tsumogiri":false}
{"type":"tsumo","actor":3,"pai":"2s"}
{"type":"dahai","actor":3,"pai":"F","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"2m"}
{"type":"dahai","actor":0,"pai":"2s","tsumogiri":false}
{"type":"tsumo","actor":1,"pai":"1m"}
{"type":"dahai","actor":1,"pai":"5m","tsumogiri":false}
{"type":"tsumo","actor":2,"pai":"3p"}
{"type":"dahai","actor":2,"pai":"3p","tsumogiri":true}
{"type":"pon","actor":0,"target":2,"pai":"3p","consumed":["3p","3p"]}
{"type":"dahai","actor":0,"pai":"2m","tsumogiri":false}
{"type":"tsumo","actor":1,"pai":"6p"}
{"type":"dahai","actor":1,"pai":"9p","tsumogiri":false}
{"type":"tsumo","actor":2,"pai":"6s"}
{"type":"dahai","actor":2,"pai":"C","tsumogiri":false}
{"type":"tsumo","actor":3,"pai":"7p"}
{"type":"dahai","actor":3,"pai":"P","tsumogiri":false}
{"type":"pon","actor":0,"target":3,"pai":"P","consumed":["P","P"]}
{"type":"dahai","actor":0,"pai":"1p","tsumogiri":false}
{"type":"tsumo","actor":1,"pai":"7s"}
{"type":"dahai","actor":1,"pai":"5s","tsumogiri":false}
{"type":"tsumo","actor":2,"pai":"3s"}
{"type":"dahai","actor":2,"pai":"9s","tsumogiri":false}
{"type":"tsumo","actor":3,"pai":"2m"}
{"type":"dahai","actor":3,"pai":"1s","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"1p"}
{"type":"dahai","actor":0,"pai":"1p","tsumogiri":true}
{"type":"tsumo","actor":1,"pai":"7m"}
{"type":"dahai","actor":1,"pai":"4s","tsumogiri":false}
{"type":"chi","actor":2,"target":1,"pai":"4s","consumed":["5s","6s"]}
{"type":"dahai","actor":2,"pai":"6p","tsumogiri":false}
{"type":"chi","actor":3,"target":2,"pai":"6p","consumed":["5pr","7p"]}
{"type":"dahai","actor":3,"pai":"7p","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"1s"}
{"type":"dahai","actor":0,"pai":"1s","tsumogiri":true}
{"type":"tsumo","actor":1,"pai":"1s"}
{"type":"reach","actor":1}
{"type":"dahai","actor":1,"pai":"1s","tsumogiri":true}
{"type":"reach_accepted","actor":1}
{"type":"tsumo","actor":2,"pai":"9s"}
{"type":"dahai","actor":2,"pai":"8s","tsumogiri":false}
{"type":"tsumo","actor":3,"pai":"4p"}
{"type":"dahai","actor":3,"pai":"4p","tsumogiri":true}
{"type":"tsumo","actor":0,"pai":"4m"}
{"type":"dahai","actor":0,"pai":"4m","tsumogiri":true}
{"type":"tsumo","actor":1,"pai":"1p"}
{"type":"dahai","actor":1,"pai":"1p","tsumogiri":true}
{"type":"tsumo","actor":2,"pai":"8m"}
{"type":"dahai","actor":2,"pai":"8m","tsumogiri":true}
{"type":"tsumo","actor":3,"pai":"C"}
{"type":"dahai","actor":3,"pai":"C","tsumogiri":true}
{"type":"tsumo","actor":0,"pai":"2s"}
{"type":"dahai","actor":0,"pai":"2s","tsumogiri":true}
{"type":"tsumo","actor":1,"pai":"8p"}
{"type":"hora","actor":1,"target":1,"deltas":[-1000,4000,-1000,-2000]}
{"type":"end_kyoku"}
{"type":"end_game"}
"#;

        let loader = GameplayLoader::new(4, false, None, None, false, true, false, false, false);
        let events = loader.parse_events(raw_log.trim(), false).unwrap();
        let estimated = estimated_player_sample_capacity(events.len());
        let actual = loader.load_events(&events).unwrap();

        assert!(estimated < events.len());
        for gameplay in actual {
            assert!(gameplay.sample_count <= estimated);
        }
    }

    #[test]
    fn repeated_take_batch_reads_return_empty_arrays() {
        let version = 4;
        let obs_channels = obs_shape(version).0;
        let mut gameplay = Gameplay::new_with_capacity(0, Grp::default(), version, 2);
        gameplay.sample_count = 2;
        gameplay.obs = vec![0.0; 2 * obs_channels * 34];
        gameplay.masks = vec![true; 2 * ACTION_SPACE];
        gameplay.context_meta = vec![0; 2 * CONTEXT_META_DIM];

        let first_obs = gameplay.take_obs_batch_array();
        assert_eq!(first_obs.shape(), &[2, obs_channels, 34]);
        let second_obs = gameplay.take_obs_batch_array();
        assert_eq!(second_obs.shape(), &[0, obs_channels, 34]);

        let first_masks = gameplay.take_masks_batch_array();
        assert_eq!(first_masks.shape(), &[2, ACTION_SPACE]);
        let second_masks = gameplay.take_masks_batch_array();
        assert_eq!(second_masks.shape(), &[0, ACTION_SPACE]);

        let first_context = gameplay.take_context_meta_batch_array();
        assert_eq!(first_context.shape(), &[2, CONTEXT_META_DIM]);
        let second_context = gameplay.take_context_meta_batch_array();
        assert_eq!(second_context.shape(), &[0, CONTEXT_META_DIM]);
    }
}
