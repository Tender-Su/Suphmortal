use super::board::{Board, BoardState, Poll};
use super::result::GameResult;
use crate::agent::BatchAgent;
use crate::mjai::EventExt;
use std::sync::OnceLock;
use std::time::Duration;
use std::time::Instant;
use std::{array, mem};

use anyhow::{Result, ensure};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::prelude::*;

pub struct BatchGame {
    /// 8 for hanchan and 4 for tonpuu
    pub length: u8,
    pub init_scores: [i32; 4],
    pub disable_progress_bar: bool,
}

#[derive(Default)]
struct ArenaPerfStats {
    run_elapsed: Duration,
    poll_elapsed: Duration,
    board_poll_elapsed: Duration,
    set_scene_elapsed: Duration,
    commit_elapsed: Duration,
    get_reaction_elapsed: Duration,
    start_game_elapsed: Duration,
    end_kyoku_elapsed: Duration,
    end_game_elapsed: Duration,
}

fn arena_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MORTAL_ARENA_PROFILE")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

impl ArenaPerfStats {
    fn log_summary(
        &self,
        cycles: usize,
        actions: usize,
        games: usize,
        record_kyoku_log: bool,
        store_game_logs: bool,
    ) {
        let run_s = self.run_elapsed.as_secs_f64();
        let pct = |part: Duration| {
            if run_s <= f64::EPSILON {
                0.0
            } else {
                part.as_secs_f64() * 100.0 / run_s
            }
        };
        log::info!(
            "arena profile: games={} cycles={} actions={} record_kyoku_log={} store_game_logs={} run={:.3}s poll={:.3}s ({:.1}%) board_poll={:.3}s ({:.1}%) set_scene={:.3}s ({:.1}%) commit={:.3}s ({:.1}%) get_reaction={:.3}s ({:.1}%) start_game={:.3}s ({:.1}%) end_kyoku={:.3}s ({:.1}%) end_game={:.3}s ({:.1}%)",
            games,
            cycles,
            actions,
            record_kyoku_log,
            store_game_logs,
            run_s,
            self.poll_elapsed.as_secs_f64(),
            pct(self.poll_elapsed),
            self.board_poll_elapsed.as_secs_f64(),
            pct(self.board_poll_elapsed),
            self.set_scene_elapsed.as_secs_f64(),
            pct(self.set_scene_elapsed),
            self.commit_elapsed.as_secs_f64(),
            pct(self.commit_elapsed),
            self.get_reaction_elapsed.as_secs_f64(),
            pct(self.get_reaction_elapsed),
            self.start_game_elapsed.as_secs_f64(),
            pct(self.start_game_elapsed),
            self.end_kyoku_elapsed.as_secs_f64(),
            pct(self.end_kyoku_elapsed),
            self.end_game_elapsed.as_secs_f64(),
            pct(self.end_game_elapsed),
        );
    }
}

#[derive(Clone, Copy, Default)]
pub struct Index {
    /// For `Game` to find a specific `Agent` (game -> agent).
    pub agent_idx: usize,
    /// For `Agent` to find a specific player ID (agent -> game).
    pub player_id_idx: usize,
}

#[derive(Default)]
struct Game {
    length: u8,
    seed: (u64, u64),
    indexes: [Index; 4],
    record_kyoku_log: bool,
    store_game_log: bool,
    profile_enabled: bool,

    oracle_obs_versions: [Option<u32>; 4],
    invisible_state_cache: [Option<Array2<f32>>; 4],

    last_reactions: [EventExt; 4], // cached for poll phase

    board: BoardState,
    kyoku: u8,
    honba: u8,
    kyotaku: u8,
    scores: [i32; 4],
    game_log: Vec<Vec<EventExt>>,

    kyoku_started: bool,
    ended: bool,
    /// Used in 西入 where the oya and another player get to 30000 at the same
    /// time, but the game continues because oya is not the top.
    ///
    /// As per [Tenhou's rule](https://tenhou.net/man/#RULE):
    ///
    /// > サドンデスルールは、30000点(供託未収)以上になった時点で終了、ただし親の
    /// > 連荘がある場合は連荘を優先する
    in_renchan: bool,
}

impl Game {
    /// Returns iff any player in the game can act or the game has ended.
    fn poll(
        &mut self,
        agents: &mut [Box<dyn BatchAgent>],
        perf: &mut Option<ArenaPerfStats>,
    ) -> Result<()> {
        let poll_started = self.profile_enabled.then(Instant::now);
        if self.ended {
            if let (Some(started), Some(perf)) = (poll_started, perf.as_mut()) {
                perf.poll_elapsed += started.elapsed();
            }
            return Ok(());
        }

        if !self.kyoku_started {
            // after W4
            // or, after all-last
            //   and, oya is not in renchan (if oya is in renchan, it would already have been ended in the renchan owari check)
            //   and, anyone has more than 30k
            if self.kyoku >= self.length + 4
                || self.kyoku >= self.length
                    && !self.in_renchan
                    && self.scores.iter().any(|&s| s >= 30000)
            {
                self.ended = true;
                return Ok(());
            }

            let mut next_board = Board {
                kyoku: self.kyoku,
                honba: self.honba,
                kyotaku: self.kyotaku,
                scores: self.scores,
                ..Default::default()
            };
            next_board.init_from_seed(self.seed);
            self.board = next_board.into_state(self.record_kyoku_log);
            self.kyoku_started = true;
        }

        let reactions = mem::take(&mut self.last_reactions);
        let board_poll_started = self.profile_enabled.then(Instant::now);
        let poll = self.board.poll(reactions)?;
        if let (Some(started), Some(perf)) = (board_poll_started, perf.as_mut()) {
            perf.board_poll_elapsed += started.elapsed();
        }
        match poll {
            Poll::InGame => {
                let ctx = self.board.agent_context();
                for (player_id, state) in ctx.player_states.iter().enumerate() {
                    if !state.last_cans().can_act() {
                        continue;
                    }

                    let invisible_state = self.oracle_obs_versions[player_id]
                        .map(|ver| self.board.encode_oracle_obs(player_id as u8, ver));
                    self.invisible_state_cache[player_id].clone_from(&invisible_state);

                    let idx = self.indexes[player_id];
                    let set_scene_started = self.profile_enabled.then(Instant::now);
                    agents[idx.agent_idx].set_scene(
                        idx.player_id_idx,
                        ctx.log,
                        state,
                        invisible_state,
                    )?;
                    if let (Some(started), Some(perf)) = (set_scene_started, perf.as_mut()) {
                        perf.set_scene_elapsed += started.elapsed();
                    }
                }
            }

            Poll::End => {
                self.kyoku_started = false;
                self.in_renchan = false;

                for idx in &self.indexes {
                    let end_kyoku_started = self.profile_enabled.then(Instant::now);
                    agents[idx.agent_idx].end_kyoku(idx.player_id_idx)?;
                    if let (Some(started), Some(perf)) = (end_kyoku_started, perf.as_mut()) {
                        perf.end_kyoku_elapsed += started.elapsed();
                    }
                }

                let kyoku_result = self.board.end();
                self.kyotaku = kyoku_result.kyotaku_left;
                self.scores = kyoku_result.scores;

                let logs = self.board.take_log();
                if self.store_game_log {
                    self.game_log.push(logs);
                }

                let has_tobi = self.scores.iter().any(|&s| s < 0);
                if has_tobi {
                    self.ended = true;
                    if let (Some(started), Some(perf)) = (poll_started, perf.as_mut()) {
                        perf.poll_elapsed += started.elapsed();
                    }
                    return Ok(());
                }

                if kyoku_result.has_abortive_ryukyoku {
                    self.honba += 1;
                    if let (Some(started), Some(perf)) = (poll_started, perf.as_mut()) {
                        perf.poll_elapsed += started.elapsed();
                    }
                    return self.poll(agents, perf);
                }

                if !kyoku_result.can_renchan {
                    self.kyoku += 1;
                    if kyoku_result.has_hora {
                        self.honba = 0;
                    } else {
                        self.honba += 1;
                    }
                    if let (Some(started), Some(perf)) = (poll_started, perf.as_mut()) {
                        perf.poll_elapsed += started.elapsed();
                    }
                    return self.poll(agents, perf);
                }

                // renchan owari conditions:
                // 1. can renchan
                // 2. is at all-last
                // 3. oya has at least 30000
                // 4. oya is the top
                let oya = kyoku_result.kyoku as usize % 4;
                if kyoku_result.kyoku >= self.length - 1 && self.scores[oya] >= 30000 {
                    let top = kyoku_result
                        .scores
                        .iter()
                        .enumerate()
                        .min_by_key(|&(_, &s)| -s)
                        .map(|(i, _)| i)
                        .unwrap();
                    if top == oya {
                        self.ended = true;
                        if let (Some(started), Some(perf)) = (poll_started, perf.as_mut()) {
                            perf.poll_elapsed += started.elapsed();
                        }
                        return Ok(());
                    }
                }

                // renchan
                self.in_renchan = true;
                self.honba += 1;
                if let (Some(started), Some(perf)) = (poll_started, perf.as_mut()) {
                    perf.poll_elapsed += started.elapsed();
                }
                return self.poll(agents, perf);
            }
        };

        if let (Some(started), Some(perf)) = (poll_started, perf.as_mut()) {
            perf.poll_elapsed += started.elapsed();
        }
        Ok(())
    }

    fn commit(
        &mut self,
        agents: &mut [Box<dyn BatchAgent>],
        perf: &mut Option<ArenaPerfStats>,
    ) -> Result<Option<GameResult>> {
        let commit_started = self.profile_enabled.then(Instant::now);
        if self.ended {
            if self.kyotaku > 0 {
                *self.scores.iter_mut().min_by_key(|s| -**s).unwrap() += self.kyotaku as i32 * 1000;
            }

            let names = if self.store_game_log {
                array::from_fn(|i| agents[self.indexes[i].agent_idx].name())
            } else {
                Default::default()
            };
            let game_result = GameResult {
                names,
                scores: self.scores,
                seed: self.seed,
                game_log: if self.store_game_log {
                    mem::take(&mut self.game_log)
                } else {
                    vec![]
                },
            };

            for idx in &self.indexes {
                let end_game_started = self.profile_enabled.then(Instant::now);
                agents[idx.agent_idx].end_game(idx.player_id_idx, &game_result)?;
                if let (Some(started), Some(perf)) = (end_game_started, perf.as_mut()) {
                    perf.end_game_elapsed += started.elapsed();
                }
            }
            if let (Some(started), Some(perf)) = (commit_started, perf.as_mut()) {
                perf.commit_elapsed += started.elapsed();
            }
            return Ok(Some(game_result));
        }

        let ctx = self.board.agent_context();
        for (player_id, state) in ctx.player_states.iter().enumerate() {
            if !state.last_cans().can_act() {
                continue;
            }

            let invisible_state = self.invisible_state_cache[player_id].take();

            let idx = self.indexes[player_id];
            let get_reaction_started = self.profile_enabled.then(Instant::now);
            self.last_reactions[player_id] = agents[idx.agent_idx].get_reaction(
                idx.player_id_idx,
                ctx.log,
                state,
                invisible_state,
            )?;
            if let (Some(started), Some(perf)) = (get_reaction_started, perf.as_mut()) {
                perf.get_reaction_elapsed += started.elapsed();
            }
        }

        if let (Some(started), Some(perf)) = (commit_started, perf.as_mut()) {
            perf.commit_elapsed += started.elapsed();
        }
        Ok(None)
    }
}

impl BatchGame {
    pub const fn tenhou_hanchan(disable_progress_bar: bool) -> Self {
        Self {
            length: 8,
            init_scores: [25000; 4],
            disable_progress_bar,
        }
    }

    pub fn run(
        &self,
        agents: &mut [Box<dyn BatchAgent>],
        indexes: &[[Index; 4]],
        seeds: &[(u64, u64)],
        store_game_logs: bool,
    ) -> Result<Vec<GameResult>> {
        ensure!(!agents.is_empty());
        ensure!(!indexes.is_empty());
        ensure!(
            indexes.len() == seeds.len(),
            "expected `indexes.len() == seeds.len()`, got {} and {}",
            indexes.len(),
            seeds.len(),
        );

        let record_kyoku_log = store_game_logs || agents.iter().any(|agent| agent.uses_event_log());
        let profile_enabled = arena_profile_enabled();
        let mut perf = profile_enabled.then_some(ArenaPerfStats::default());
        let run_started = profile_enabled.then(Instant::now);

        let mut games = indexes
            .iter()
            .zip(seeds)
            .enumerate()
            .map(|(game_idx, (idxs, &seed))| {
                let mut oracle_obs_versions = [None; 4];
                for (i, idx) in idxs.iter().enumerate() {
                    let start_game_started = profile_enabled.then(Instant::now);
                    agents[idx.agent_idx].start_game(idx.player_id_idx)?;
                    if let (Some(started), Some(perf)) = (start_game_started, perf.as_mut()) {
                        perf.start_game_elapsed += started.elapsed();
                    }
                    oracle_obs_versions[i] = agents[idx.agent_idx].oracle_obs_version();
                }

                let game = Box::new(Game {
                    length: self.length,
                    seed,
                    indexes: *idxs,
                    record_kyoku_log,
                    store_game_log: store_game_logs,
                    profile_enabled,
                    scores: self.init_scores,
                    oracle_obs_versions,
                    ..Default::default()
                });
                Ok((game_idx, game))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut game_results = vec![GameResult::default(); games.len()];
        let mut to_remove = vec![];
        let mut cycles = 0;
        let mut actions = 0;

        let bar = if self.disable_progress_bar {
            None
        } else {
            let bar = ProgressBar::new(games.len() as u64);
            const TEMPLATE: &str =
                "{spinner:.cyan} {msg}\n[{elapsed_precise}] [{wide_bar}] {pos}/{len} {percent:>3}%";
            let style = ProgressStyle::with_template(TEMPLATE)?
                .tick_chars(".oO°Oo*")
                .progress_chars("#-");
            bar.set_style(style);
            bar.enable_steady_tick(Duration::from_millis(150));
            Some(bar)
        };

        while !games.is_empty() {
            for (_, game) in &mut games {
                game.poll(agents, &mut perf)?;
            }

            for (idx_for_rm, (game_idx, game)) in games.iter_mut().enumerate() {
                if let Some(game_result) = game.commit(agents, &mut perf)? {
                    game_results[*game_idx] = game_result;
                    to_remove.push(idx_for_rm);
                }
            }

            for idx_for_rm in to_remove.drain(..).rev() {
                games.swap_remove(idx_for_rm);
                if let Some(bar) = &bar {
                    bar.inc(1);
                }
            }

            cycles += 1;
            actions += games.len();

            if let Some(bar) = &bar {
                let secs = bar.elapsed().as_secs_f64();
                bar.set_message(format!(
                    "cycles: {cycles} ({:.3} cycle/s), actions: {actions} ({:.3} action/s)",
                    cycles as f64 / secs,
                    actions as f64 / secs,
                ));
            }
        }
        if let Some(bar) = bar {
            bar.abandon();
        }
        if let (Some(started), Some(perf)) = (run_started, perf.as_mut()) {
            perf.run_elapsed += started.elapsed();
            perf.log_summary(
                cycles,
                actions,
                game_results.len(),
                record_kyoku_log,
                store_game_logs,
            );
        }

        Ok(game_results)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::agent::Tsumogiri;

    #[test]
    fn tsumogiri() {
        let g = BatchGame::tenhou_hanchan(true);
        let mut agents = [
            Box::new(Tsumogiri::new_batched(&[0, 1, 2, 3]).unwrap()) as _,
            Box::new(Tsumogiri::new_batched(&[3, 2, 1, 0]).unwrap()) as _,
        ];
        let indexes = &[
            [
                Index {
                    agent_idx: 0,
                    player_id_idx: 0,
                },
                Index {
                    agent_idx: 0,
                    player_id_idx: 1,
                },
                Index {
                    agent_idx: 1,
                    player_id_idx: 1,
                },
                Index {
                    agent_idx: 1,
                    player_id_idx: 0,
                },
            ],
            [
                Index {
                    agent_idx: 1,
                    player_id_idx: 3,
                },
                Index {
                    agent_idx: 1,
                    player_id_idx: 2,
                },
                Index {
                    agent_idx: 0,
                    player_id_idx: 2,
                },
                Index {
                    agent_idx: 0,
                    player_id_idx: 3,
                },
            ],
        ];

        g.run(&mut agents, indexes, &[(1009, 0), (1021, 0)], false)
            .unwrap();
    }
}
