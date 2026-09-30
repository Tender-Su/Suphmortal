from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor

from mortal.config import config as global_config
from mortal.core.checkpoint_utils import load_brain_state_with_input_bridge
from mortal.core.config_utils import (
    coerce_bool as _as_bool,
    get_dict_section as _cfg_section,
)
from mortal.core.model import (
    Brain,
    DangerAuxNet,
    ExpectedRewardNet,
    FuroRegretHead,
    HandValueRegretHead,
    OpponentStateAuxNet,
    TileEfficiencyRegretHead,
    ValueHead,
)

_EPS = 1e-8
_DISCARD_DIM = 37
_CHI_ACTIONS = (38, 39, 40)
_CALL_ACTIONS = (38, 39, 40, 41, 42)
_PASS_ACTION = 45
_RIICHI_ACTION = 37
_AGARI_ACTION = 43
_RYUKYOKU_ACTION = 44


@dataclass(frozen=True)
class SearchConfig:
    enabled: bool = False
    top_k: int = 5
    belief_samples: int = 8
    planner_blend: float = 0.65
    score_temperature: float = 0.85
    min_policy_entropy: float = 1.15
    min_margin: float = 0.08
    min_opp_tenpai_prob: float = 0.30
    min_danger_prob: float = 0.22
    risk_weight: float = 1.30
    variance_weight: float = 0.35
    tile_eff_weight: float = 0.20
    hand_value_weight: float = 0.18
    call_regret_weight: float = 0.22
    riichi_bonus: float = 0.06
    agari_bonus: float = 0.60
    pass_bonus: float = 0.05
    safe_discard_bias: float = 0.08
    hard_entropy_min: float = 1.05
    hard_margin_max: float = 0.06
    hard_danger_min: float = 0.28

    @classmethod
    def from_config(cls, config_dict: Any) -> SearchConfig:
        cfg = _cfg_section(config_dict, "search")
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            top_k=max(int(cfg.get("top_k", 5) or 5), 1),
            belief_samples=max(int(cfg.get("belief_samples", 8) or 8), 1),
            planner_blend=float(cfg.get("planner_blend", 0.65) or 0.65),
            score_temperature=max(float(cfg.get("score_temperature", 0.85) or 0.85), 0.05),
            min_policy_entropy=float(cfg.get("min_policy_entropy", 1.15) or 1.15),
            min_margin=float(cfg.get("min_margin", 0.08) or 0.08),
            min_opp_tenpai_prob=float(cfg.get("min_opp_tenpai_prob", 0.30) or 0.30),
            min_danger_prob=float(cfg.get("min_danger_prob", 0.22) or 0.22),
            risk_weight=float(cfg.get("risk_weight", 1.30) or 1.30),
            variance_weight=float(cfg.get("variance_weight", 0.35) or 0.35),
            tile_eff_weight=float(cfg.get("tile_eff_weight", 0.20) or 0.20),
            hand_value_weight=float(cfg.get("hand_value_weight", 0.18) or 0.18),
            call_regret_weight=float(cfg.get("call_regret_weight", 0.22) or 0.22),
            riichi_bonus=float(cfg.get("riichi_bonus", 0.06) or 0.06),
            agari_bonus=float(cfg.get("agari_bonus", 0.60) or 0.60),
            pass_bonus=float(cfg.get("pass_bonus", 0.05) or 0.05),
            safe_discard_bias=float(cfg.get("safe_discard_bias", 0.08) or 0.08),
            hard_entropy_min=float(cfg.get("hard_entropy_min", 1.05) or 1.05),
            hard_margin_max=float(cfg.get("hard_margin_max", 0.06) or 0.06),
            hard_danger_min=float(cfg.get("hard_danger_min", 0.28) or 0.28),
        )


@dataclass(frozen=True)
class SearchDistillConfig:
    enabled: bool = False
    weight: float = 0.05
    hard_only: bool = True
    min_teacher_gap: float = 0.03

    @classmethod
    def from_config(cls, config_dict: Any) -> SearchDistillConfig:
        cfg = _cfg_section(config_dict, "search_distill")
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            weight=float(cfg.get("weight", 0.05) or 0.05),
            hard_only=_as_bool(cfg.get("hard_only", True), default=True),
            min_teacher_gap=float(cfg.get("min_teacher_gap", 0.03) or 0.03),
        )


@dataclass
class BeliefSamples:
    discard_risk_samples: Tensor
    discard_risk_mean: Tensor
    discard_risk_std: Tensor
    global_pressure: Tensor
    opponent_tenpai_prob: Tensor
    danger_prob: Tensor


@dataclass
class SearchPlan:
    final_probs: Tensor
    planner_probs: Tensor
    planner_scores: Tensor
    active_mask: Tensor
    hard_mask: Tensor
    entropy: Tensor
    margin: Tensor
    global_pressure: Tensor
    teacher_gap: Tensor
    belief: BeliefSamples


class BeliefSampler:
    def __init__(self, cfg: SearchConfig):
        self.cfg = cfg

    def sample(
        self,
        *,
        masks: Tensor,
        opponent_outputs: Optional[tuple[tuple[Tensor, ...], tuple[Tensor, ...]]] = None,
        danger_outputs: Optional[tuple[Tensor, Tensor, Tensor]] = None,
    ) -> BeliefSamples:
        batch_size = masks.shape[0]
        device = masks.device
        dtype = torch.float32

        if opponent_outputs is not None:
            shanten_logits, tenpai_logits = opponent_outputs
            opp_tenpai_prob = torch.stack(
                [logits.softmax(-1)[..., 1] for logits in tenpai_logits],
                dim=-1,
            ).to(dtype=dtype)
            shanten_support = torch.arange(
                shanten_logits[0].shape[-1],
                dtype=dtype,
                device=device,
            )
            opp_expected_shanten = torch.stack(
                [(logits.softmax(-1) * shanten_support).sum(-1) for logits in shanten_logits],
                dim=-1,
            ).to(dtype=dtype)
        else:
            opp_tenpai_prob = torch.zeros((batch_size, 3), dtype=dtype, device=device)
            opp_expected_shanten = torch.full((batch_size, 3), 2.5, dtype=dtype, device=device)

        if danger_outputs is not None:
            any_logits, value_pred, player_logits = danger_outputs
            danger_prob = any_logits.sigmoid().to(dtype=dtype)
            danger_value = value_pred.sigmoid().to(dtype=dtype)
            player_prob = player_logits.sigmoid().to(dtype=dtype)
        else:
            danger_prob = torch.zeros((batch_size, _DISCARD_DIM), dtype=dtype, device=device)
            danger_value = torch.zeros((batch_size, _DISCARD_DIM), dtype=dtype, device=device)
            player_prob = torch.zeros((batch_size, _DISCARD_DIM, 3), dtype=dtype, device=device)

        pressure_prob = (
            opp_tenpai_prob
            * (1.20 - 0.15 * opp_expected_shanten.clamp(min=0.0, max=3.0))
        ).clamp(0.0, 1.0)
        sample_count = self.cfg.belief_samples
        sampled_pressure = torch.bernoulli(
            pressure_prob.unsqueeze(0).expand(sample_count, -1, -1)
        )
        sampled_risk = danger_prob.unsqueeze(0).expand(sample_count, -1, -1).clone()
        player_weight = player_prob.unsqueeze(0) * sampled_pressure.unsqueeze(-2)
        sampled_risk = sampled_risk + 0.35 * player_weight.sum(-1)
        sampled_risk = sampled_risk + 0.40 * danger_value.unsqueeze(0)
        sampled_risk = sampled_risk.clamp(0.0, 2.5)
        legal_discards = masks[:, :_DISCARD_DIM].to(dtype=dtype)
        sampled_risk = sampled_risk * legal_discards.unsqueeze(0)
        discard_risk_mean = sampled_risk.mean(0)
        discard_risk_std = sampled_risk.std(0, unbiased=False)
        global_pressure = (
            0.55 * pressure_prob.max(-1).values
            + 0.45 * discard_risk_mean.max(-1).values
        ).clamp(0.0, 2.5)
        return BeliefSamples(
            discard_risk_samples=sampled_risk,
            discard_risk_mean=discard_risk_mean,
            discard_risk_std=discard_risk_std,
            global_pressure=global_pressure,
            opponent_tenpai_prob=opp_tenpai_prob,
            danger_prob=danger_prob,
        )


class LocalSearchPlanner:
    def __init__(self, cfg: SearchConfig):
        self.cfg = cfg
        self.belief_sampler = BeliefSampler(cfg)

    def _candidate_mask(self, policy_probs: Tensor, masks: Tensor, belief: BeliefSamples) -> Tensor:
        candidate_mask = torch.zeros_like(masks)
        legal_count = masks.sum(-1)
        top_k = min(self.cfg.top_k, masks.shape[-1])
        legal_probs = policy_probs.masked_fill(~masks, 0.0)
        topk_indices = legal_probs.topk(k=top_k, dim=-1).indices
        candidate_mask.scatter_(1, topk_indices, True)

        legal_discards = masks[:, :_DISCARD_DIM]
        if bool(legal_discards.any()):
            safe_discard = belief.discard_risk_mean.masked_fill(~legal_discards, float("inf")).argmin(-1, keepdim=True)
            candidate_mask.scatter_(1, safe_discard, True)
            risky_discard = belief.discard_risk_mean.masked_fill(~legal_discards, float("-inf")).argmax(-1, keepdim=True)
            candidate_mask.scatter_(1, risky_discard, True)

        candidate_mask |= (legal_count <= top_k).unsqueeze(-1) & masks
        candidate_mask &= masks
        return candidate_mask

    def plan(
        self,
        *,
        policy_logits: Tensor,
        policy_probs: Tensor,
        masks: Tensor,
        opponent_outputs: Optional[tuple[tuple[Tensor, ...], tuple[Tensor, ...]]] = None,
        danger_outputs: Optional[tuple[Tensor, Tensor, Tensor]] = None,
        tile_eff_pred: Optional[Tensor] = None,
        hand_value_pred: Optional[Tensor] = None,
        furo_regret_pred: Optional[Tensor] = None,
    ) -> SearchPlan:
        if not self.cfg.enabled:
            entropy = -(policy_probs * policy_probs.clamp_min(_EPS).log()).sum(-1)
            margin = policy_probs.topk(k=min(2, policy_probs.shape[-1]), dim=-1).values
            second = margin[:, 1] if margin.shape[-1] > 1 else torch.zeros_like(margin[:, 0])
            empty_belief = self.belief_sampler.sample(
                masks=masks,
                opponent_outputs=opponent_outputs,
                danger_outputs=danger_outputs,
            )
            return SearchPlan(
                final_probs=policy_probs,
                planner_probs=policy_probs,
                planner_scores=policy_logits,
                active_mask=torch.zeros_like(entropy, dtype=torch.bool),
                hard_mask=torch.zeros_like(entropy, dtype=torch.bool),
                entropy=entropy,
                margin=margin[:, 0] - second,
                global_pressure=empty_belief.global_pressure,
                teacher_gap=torch.zeros_like(entropy),
                belief=empty_belief,
            )

        belief = self.belief_sampler.sample(
            masks=masks,
            opponent_outputs=opponent_outputs,
            danger_outputs=danger_outputs,
        )
        entropy = -(policy_probs * policy_probs.clamp_min(_EPS).log()).sum(-1)
        top2 = policy_probs.topk(k=min(2, policy_probs.shape[-1]), dim=-1).values
        second = top2[:, 1] if top2.shape[-1] > 1 else torch.zeros_like(top2[:, 0])
        margin = top2[:, 0] - second
        legal_count = masks.sum(-1)
        has_call = masks[:, _CHI_ACTIONS[0]:_PASS_ACTION + 1].any(-1)
        opp_signal = belief.opponent_tenpai_prob.max(-1).values
        danger_signal = belief.discard_risk_mean.max(-1).values
        active_mask = (
            (legal_count > 1)
            & (
                (entropy >= self.cfg.min_policy_entropy)
                | (margin <= self.cfg.min_margin)
                | (opp_signal >= self.cfg.min_opp_tenpai_prob)
                | (danger_signal >= self.cfg.min_danger_prob)
                | has_call
            )
        )

        planner_scores = policy_logits.clone()
        candidate_mask = self._candidate_mask(policy_probs, masks, belief)
        discard_idx = torch.arange(_DISCARD_DIM, device=policy_logits.device)
        if _DISCARD_DIM > 0:
            discard_penalty = (
                self.cfg.risk_weight * belief.discard_risk_mean
                + self.cfg.variance_weight * belief.discard_risk_std
            )
            discard_bonus = torch.zeros_like(discard_penalty)
            if tile_eff_pred is not None:
                discard_bonus = discard_bonus + self.cfg.tile_eff_weight * tile_eff_pred[:, :_DISCARD_DIM]
            if hand_value_pred is not None:
                discard_bonus = discard_bonus + self.cfg.hand_value_weight * hand_value_pred[:, :_DISCARD_DIM].sigmoid()
            safest = belief.discard_risk_mean.argmin(-1, keepdim=True)
            safe_bonus = torch.zeros_like(discard_bonus)
            safe_bonus.scatter_(1, safest, self.cfg.safe_discard_bias)
            planner_scores[:, :_DISCARD_DIM] = (
                planner_scores[:, :_DISCARD_DIM] - discard_penalty + discard_bonus + safe_bonus
            )

        global_pressure = belief.global_pressure
        if furo_regret_pred is not None:
            call_bonus = self.cfg.call_regret_weight * furo_regret_pred[:, 0:1]
            pass_bonus = self.cfg.call_regret_weight * furo_regret_pred[:, 1:2]
            for action_id in _CALL_ACTIONS:
                planner_scores[:, action_id:action_id + 1] = (
                    planner_scores[:, action_id:action_id + 1]
                    + call_bonus
                    - 0.45 * global_pressure.unsqueeze(-1)
                )
            planner_scores[:, _PASS_ACTION:_PASS_ACTION + 1] = (
                planner_scores[:, _PASS_ACTION:_PASS_ACTION + 1]
                + pass_bonus
                + self.cfg.pass_bonus * global_pressure.unsqueeze(-1)
            )

        planner_scores[:, _RIICHI_ACTION:_RIICHI_ACTION + 1] = (
            planner_scores[:, _RIICHI_ACTION:_RIICHI_ACTION + 1]
            + self.cfg.riichi_bonus * (1.0 - global_pressure.clamp(max=1.0)).unsqueeze(-1)
        )
        planner_scores[:, _AGARI_ACTION:_AGARI_ACTION + 1] = (
            planner_scores[:, _AGARI_ACTION:_AGARI_ACTION + 1] + self.cfg.agari_bonus
        )
        planner_scores[:, _RYUKYOKU_ACTION:_RYUKYOKU_ACTION + 1] = (
            planner_scores[:, _RYUKYOKU_ACTION:_RYUKYOKU_ACTION + 1]
            - 0.10 * (1.0 - global_pressure.clamp(max=1.0)).unsqueeze(-1)
        )

        inactive_fill = policy_logits.masked_fill(~masks, float("-inf"))
        planner_scores = torch.where(candidate_mask, planner_scores, inactive_fill)
        planner_scores = planner_scores.masked_fill(~masks, float("-inf"))
        planner_probs = torch.softmax(planner_scores / self.cfg.score_temperature, dim=-1)
        blended = torch.lerp(policy_probs, planner_probs, self.cfg.planner_blend)
        final_probs = blended.masked_fill(~masks, 0.0)
        final_probs = final_probs / final_probs.sum(-1, keepdim=True).clamp_min(_EPS)
        final_probs = torch.where(active_mask.unsqueeze(-1), final_probs, policy_probs)
        teacher_gap = (planner_probs - policy_probs).abs().sum(-1)
        hard_mask = active_mask & (
            (entropy >= self.cfg.hard_entropy_min)
            | (margin <= self.cfg.hard_margin_max)
            | (global_pressure >= self.cfg.hard_danger_min)
        )

        return SearchPlan(
            final_probs=final_probs,
            planner_probs=planner_probs,
            planner_scores=planner_scores,
            active_mask=active_mask,
            hard_mask=hard_mask,
            entropy=entropy,
            margin=margin,
            global_pressure=global_pressure,
            teacher_gap=teacher_gap,
            belief=belief,
        )


@dataclass
class SearchRuntimeBundle:
    cfg: SearchConfig
    planner: LocalSearchPlanner
    opponent_aux_net: Optional[OpponentStateAuxNet] = None
    danger_aux_net: Optional[DangerAuxNet] = None
    tile_eff_net: Optional[TileEfficiencyRegretHead] = None
    furo_regret_net: Optional[FuroRegretHead] = None
    hand_value_regret_net: Optional[HandValueRegretHead] = None
    oracle_brain: Optional[Brain] = None
    value_net: Optional[ValueHead] = None
    exp_reward_net: Optional[ExpectedRewardNet] = None

    def enabled(self) -> bool:
        return self.cfg.enabled

    def plan(
        self,
        *,
        phi: Tensor,
        policy_logits: Tensor,
        policy_probs: Tensor,
        masks: Tensor,
    ) -> SearchPlan:
        with torch.inference_mode():
            opponent_outputs = self.opponent_aux_net(phi.detach()) if self.opponent_aux_net is not None else None
            danger_outputs = self.danger_aux_net(phi.detach()) if self.danger_aux_net is not None else None
            tile_eff_pred = self.tile_eff_net(phi.detach()) if self.tile_eff_net is not None else None
            furo_pred = self.furo_regret_net(phi.detach()) if self.furo_regret_net is not None else None
            hand_value_pred = self.hand_value_regret_net(phi.detach()) if self.hand_value_regret_net is not None else None
            return self.planner.plan(
                policy_logits=policy_logits,
                policy_probs=policy_probs,
                masks=masks,
                opponent_outputs=opponent_outputs,
                danger_outputs=danger_outputs,
                tile_eff_pred=tile_eff_pred,
                hand_value_pred=hand_value_pred,
                furo_regret_pred=furo_pred,
            )

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in (
            "opponent_aux_net",
            "danger_aux_net",
            "tile_eff_net",
            "furo_regret_net",
            "hand_value_regret_net",
            "oracle_brain",
            "value_net",
            "exp_reward_net",
        ):
            module = getattr(self, key)
            if module is not None:
                payload[key] = module.state_dict()
        if payload:
            payload["search_cfg"] = self.cfg.__dict__.copy()
        return payload

    def load_payload(self, payload: Optional[dict[str, Any]]) -> None:
        if not isinstance(payload, dict):
            return
        for key in (
            "opponent_aux_net",
            "danger_aux_net",
            "tile_eff_net",
            "furo_regret_net",
            "hand_value_regret_net",
            "value_net",
            "exp_reward_net",
        ):
            module = getattr(self, key)
            state = payload.get(key)
            if module is not None and isinstance(state, dict):
                module.load_state_dict(state)
        oracle_state = payload.get("oracle_brain")
        if self.oracle_brain is not None and isinstance(oracle_state, dict):
            load_brain_state_with_input_bridge(self.oracle_brain, oracle_state)


def _maybe_compile(module: Optional[torch.nn.Module], enabled: bool) -> Optional[torch.nn.Module]:
    if module is None or not enabled:
        return module
    module.compile()
    return module


def _state_has_module(state: dict[str, Any], key: str) -> bool:
    return isinstance(state.get(key), dict)


def build_search_runtime_bundle_from_state(
    state: dict[str, Any],
    *,
    device: torch.device,
    enable_compile: bool = False,
    search_cfg: Optional[SearchConfig] = None,
) -> Optional[SearchRuntimeBundle]:
    if not isinstance(state, dict):
        return None
    state_cfg = state.get("config", {})
    cfg_source = (
        state_cfg
        if isinstance(state_cfg, dict) and "search" in state_cfg
        else global_config
    )
    cfg = search_cfg or SearchConfig.from_config(cfg_source)
    if not cfg.enabled:
        return None

    bundle = SearchRuntimeBundle(
        cfg=cfg,
        planner=LocalSearchPlanner(cfg),
    )
    if _state_has_module(state, "opponent_aux_net"):
        bundle.opponent_aux_net = _maybe_compile(OpponentStateAuxNet().to(device).eval(), enable_compile)
        bundle.opponent_aux_net.load_state_dict(state["opponent_aux_net"])
    if _state_has_module(state, "danger_aux_net"):
        bundle.danger_aux_net = _maybe_compile(DangerAuxNet().to(device).eval(), enable_compile)
        bundle.danger_aux_net.load_state_dict(state["danger_aux_net"])
    if _state_has_module(state, "tile_eff_net"):
        bundle.tile_eff_net = _maybe_compile(TileEfficiencyRegretHead().to(device).eval(), enable_compile)
        bundle.tile_eff_net.load_state_dict(state["tile_eff_net"])
    if _state_has_module(state, "furo_regret_net"):
        bundle.furo_regret_net = _maybe_compile(FuroRegretHead().to(device).eval(), enable_compile)
        bundle.furo_regret_net.load_state_dict(state["furo_regret_net"])
    if _state_has_module(state, "hand_value_regret_net"):
        bundle.hand_value_regret_net = _maybe_compile(HandValueRegretHead().to(device).eval(), enable_compile)
        bundle.hand_value_regret_net.load_state_dict(state["hand_value_regret_net"])
    if _state_has_module(state, "value_net"):
        saved_cfg = state.get("config", {})
        value_cfg = _cfg_section(saved_cfg, "value")
        num_players = int(value_cfg.get("num_players", 4) or 4)
        bundle.value_net = _maybe_compile(
            ValueHead(
                num_players=num_players,
                zero_sum=bool(value_cfg.get("exact_zero_sum", False)),
            ).to(device).eval(),
            enable_compile,
        )
        bundle.value_net.load_state_dict(state["value_net"])
    if _state_has_module(state, "exp_reward_net"):
        saved_cfg = state.get("config", {})
        value_cfg = _cfg_section(saved_cfg, "value")
        num_players = int(value_cfg.get("num_players", 4) or 4)
        bundle.exp_reward_net = _maybe_compile(ExpectedRewardNet(num_players=num_players).to(device).eval(), enable_compile)
        bundle.exp_reward_net.load_state_dict(state["exp_reward_net"])
    if _state_has_module(state, "oracle_brain"):
        saved_cfg = state.get("config", {})
        resnet_cfg = _cfg_section(saved_cfg, "resnet")
        control_cfg = _cfg_section(saved_cfg, "control")
        version = int(control_cfg.get("version", 4) or 4)
        brain = Brain(
            version=version,
            num_blocks=int(resnet_cfg.get("num_blocks", 40) or 40),
            conv_channels=int(resnet_cfg.get("conv_channels", 192) or 192),
            is_oracle=True,
            Norm="GN",
        ).to(device).eval()
        load_brain_state_with_input_bridge(brain, state["oracle_brain"])
        bundle.oracle_brain = _maybe_compile(brain, enable_compile)
    return bundle


def build_search_runtime_bundle_from_state_file(
    state_file: str,
    *,
    device: torch.device,
    enable_compile: bool = False,
    search_cfg: Optional[SearchConfig] = None,
) -> Optional[SearchRuntimeBundle]:
    if not state_file:
        return None
    checkpoint_path = Path(state_file)
    if not checkpoint_path.exists():
        return None
    state = torch.load(checkpoint_path, weights_only=False, map_location=torch.device("cpu"))
    return build_search_runtime_bundle_from_state(
        state,
        device=device,
        enable_compile=enable_compile,
        search_cfg=search_cfg,
    )
