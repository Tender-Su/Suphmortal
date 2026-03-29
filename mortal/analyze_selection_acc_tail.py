from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIDELITY_ROOT = REPO_ROOT / "logs" / "stage05_fidelity"
DEFAULT_REPORT_JSON = REPO_ROOT / "logs" / "selection_acc_tail_audit.json"
DEFAULT_REPORT_MD = REPO_ROOT / "logs" / "selection_acc_tail_audit.md"
NEAR_TIE_THRESHOLDS = (1e-4, 2e-4, 5e-4, 1e-3)


def safe_float(value: object, default: float = float("-inf")) -> float:
    try:
        if value is None:
            return default
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def is_selector_round(payload: dict[str, Any]) -> bool:
    ranking = payload.get("ranking")
    if not ranking or not isinstance(ranking, list):
        return False
    first = ranking[0]
    return isinstance(first, dict) and "selection_quality_score" in first


def collect_rounds(fidelity_root: Path) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    for path in sorted(fidelity_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not is_selector_round(payload):
            continue
        rounds.append({"path": path, "payload": payload})
    return rounds


def group_key_for_entry(entry: dict[str, Any]) -> str:
    group = entry.get("eligibility_group")
    if group:
        return str(group)
    meta = entry.get("candidate_meta")
    if isinstance(meta, dict) and meta.get("protocol_arm"):
        return str(meta["protocol_arm"])
    return "__all__"


def current_sort_key(entry: dict[str, Any]) -> tuple[float, ...]:
    sort_key = entry.get("sort_key")
    if isinstance(sort_key, list):
        return tuple(safe_float(value) for value in sort_key)
    return simplified_sort_key(entry)


def simplified_sort_key(entry: dict[str, Any]) -> tuple[float, ...]:
    recent = safe_float(
        entry.get("comparison_recent_loss", entry.get("recent_policy_loss")),
        float("inf"),
    )
    old = safe_float(
        entry.get("comparison_old_regression_loss", entry.get("old_regression_policy_loss")),
        float("inf"),
    )
    return (
        1 if entry.get("valid") else 0,
        1 if entry.get("eligible") else 0,
        safe_float(entry.get("selection_quality_score")),
        -recent if math.isfinite(recent) else float("-inf"),
        -old if math.isfinite(old) else float("-inf"),
    )


def make_empty_summary() -> dict[str, Any]:
    return {
        "files": 0,
        "group_decisions": 0,
        "eligible_groups": 0,
        "eligible_entries": 0,
        "groups_with_2plus_eligible": 0,
        "top1_selection_ties": 0,
        "groups_with_any_selection_equal_pairs": 0,
        "selection_equal_pair_count": 0,
        "winner_changed_without_acc_tail": 0,
        "order_changed_without_acc_tail": 0,
        "near_tie_top12_counts": {str(threshold): 0 for threshold in NEAR_TIE_THRESHOLDS},
        "min_top12_gap": None,
        "max_top12_gap": None,
        "by_round": {},
        "winner_change_examples": [],
        "selection_tie_examples": [],
        "near_tie_examples": {str(threshold): [] for threshold in NEAR_TIE_THRESHOLDS},
    }


def summarize_rounds(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    summary = make_empty_summary()
    summary["files"] = len(rounds)
    by_round = Counter()

    for item in rounds:
        path: Path = item["path"]
        payload = item["payload"]
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in payload["ranking"]:
            by_group[group_key_for_entry(entry)].append(entry)

        for group, entries in by_group.items():
            summary["group_decisions"] += 1
            eligible = [entry for entry in entries if entry.get("valid") and entry.get("eligible")]
            if not eligible:
                continue
            summary["eligible_groups"] += 1
            summary["eligible_entries"] += len(eligible)
            by_round[payload.get("round_name", path.name)] += 1

            current_sorted = sorted(eligible, key=current_sort_key, reverse=True)
            simplified_sorted = sorted(eligible, key=simplified_sort_key, reverse=True)

            scores = [safe_float(entry.get("selection_quality_score")) for entry in eligible]
            top_score = max(scores)
            top_ties = [entry for entry in eligible if safe_float(entry.get("selection_quality_score")) == top_score]
            if len(top_ties) > 1:
                summary["top1_selection_ties"] += 1
                if len(summary["selection_tie_examples"]) < 8:
                    summary["selection_tie_examples"].append({
                        "file": str(path),
                        "round": payload.get("round_name", path.name),
                        "group": group,
                        "score": top_score,
                        "arms": [entry.get("arm_name") for entry in top_ties],
                    })

            equal_pairs = 0
            for index, left_score in enumerate(scores):
                for right_score in scores[index + 1:]:
                    if left_score == right_score:
                        equal_pairs += 1
            if equal_pairs > 0:
                summary["groups_with_any_selection_equal_pairs"] += 1
                summary["selection_equal_pair_count"] += equal_pairs

            if len(current_sorted) >= 2:
                summary["groups_with_2plus_eligible"] += 1
                top12_gap = (
                    safe_float(current_sorted[0].get("selection_quality_score"))
                    - safe_float(current_sorted[1].get("selection_quality_score"))
                )
                if summary["min_top12_gap"] is None or top12_gap < summary["min_top12_gap"]:
                    summary["min_top12_gap"] = top12_gap
                if summary["max_top12_gap"] is None or top12_gap > summary["max_top12_gap"]:
                    summary["max_top12_gap"] = top12_gap
                for threshold in NEAR_TIE_THRESHOLDS:
                    if top12_gap <= threshold:
                        label = str(threshold)
                        summary["near_tie_top12_counts"][label] += 1
                        if len(summary["near_tie_examples"][label]) < 8:
                            summary["near_tie_examples"][label].append({
                                "file": str(path),
                                "round": payload.get("round_name", path.name),
                                "group": group,
                                "gap": top12_gap,
                                "top1": current_sorted[0].get("arm_name"),
                                "top2": current_sorted[1].get("arm_name"),
                            })

            if current_sorted[0].get("arm_name") != simplified_sorted[0].get("arm_name"):
                summary["winner_changed_without_acc_tail"] += 1
                if len(summary["winner_change_examples"]) < 8:
                    summary["winner_change_examples"].append({
                        "file": str(path),
                        "round": payload.get("round_name", path.name),
                        "group": group,
                        "current_winner": current_sorted[0].get("arm_name"),
                        "simplified_winner": simplified_sorted[0].get("arm_name"),
                        "current_selection_score": current_sorted[0].get("selection_quality_score"),
                        "simplified_selection_score": simplified_sorted[0].get("selection_quality_score"),
                    })

            current_order = [entry.get("arm_name") for entry in current_sorted]
            simplified_order = [entry.get("arm_name") for entry in simplified_sorted]
            if current_order != simplified_order:
                summary["order_changed_without_acc_tail"] += 1

    summary["by_round"] = dict(by_round)
    return summary


def build_report(fidelity_root: Path) -> dict[str, Any]:
    rounds = collect_rounds(fidelity_root)
    p1_mainline_rounds = [
        item for item in rounds
        if any(
            marker in str(item["payload"].get("round_name", item["path"].name))
            for marker in ("p1_protocol_decide_round", "p1_winner_refine_round", "p1_ablation_round")
        )
    ]
    p1_historical_solo_rounds = [
        item for item in rounds
        if "p1_solo_round" in str(item["payload"].get("round_name", item["path"].name))
    ]
    return {
        "fidelity_root": str(fidelity_root),
        "selector_round_files": [str(item["path"]) for item in rounds],
        "all_selector_rounds": summarize_rounds(rounds),
        "p1_mainline_rounds": summarize_rounds(p1_mainline_rounds),
        "p1_historical_solo_rounds": summarize_rounds(p1_historical_solo_rounds),
        "note": (
            "winner_changed_without_acc_tail/order_changed_without_acc_tail compare the current "
            "stored full sort_key against a simplified key that keeps only "
            "valid -> eligible -> selection_quality_score -> -recent_policy_loss -> "
            "-old_regression_policy_loss."
        ),
    }


def render_scope_markdown(title: str, summary: dict[str, Any]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        f"- files: `{summary['files']}`",
        f"- group decisions: `{summary['group_decisions']}`",
        f"- eligible groups: `{summary['eligible_groups']}`",
        f"- eligible entries: `{summary['eligible_entries']}`",
        f"- groups with >=2 eligible: `{summary['groups_with_2plus_eligible']}`",
        f"- top1 exact selection-score ties: `{summary['top1_selection_ties']}`",
        f"- groups with any exact selection-score equal pair: `{summary['groups_with_any_selection_equal_pairs']}`",
        f"- exact equal selection-score pairs: `{summary['selection_equal_pair_count']}`",
        f"- winner changes after removing acc tail: `{summary['winner_changed_without_acc_tail']}`",
        f"- eligible-order changes after removing acc tail: `{summary['order_changed_without_acc_tail']}`",
        f"- smallest top1-top2 selection gap: `{summary['min_top12_gap']}`",
        f"- largest top1-top2 selection gap: `{summary['max_top12_gap']}`",
        "",
        "### Near Ties",
        "",
    ]
    for threshold, count in summary["near_tie_top12_counts"].items():
        lines.append(f"- top1-top2 gap <= `{threshold}`: `{count}`")

    lines.extend([
        "",
        "### Example Near Ties",
        "",
    ])
    for threshold, examples in summary["near_tie_examples"].items():
        if not examples:
            continue
        lines.append(f"- threshold `{threshold}`:")
        for example in examples:
            lines.append(
                "  - "
                f"`{example['round']}` / `{example['group']}` / gap=`{example['gap']}` / "
                f"`{example['top1']}` vs `{example['top2']}`"
            )

    lines.extend([
        "",
        "### Winner-Change Examples",
        "",
    ])
    if not summary["winner_change_examples"]:
        lines.append("- none")
    else:
        for example in summary["winner_change_examples"]:
            lines.append(
                "- "
                f"`{example['round']}` / `{example['group']}` / current=`{example['current_winner']}` / "
                f"simplified=`{example['simplified_winner']}`"
            )

    lines.append("")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Selection Acc Tail Audit",
        "",
        f"- fidelity_root: `{report['fidelity_root']}`",
        f"- selector round files: `{len(report['selector_round_files'])}`",
        f"- note: {report['note']}",
        "",
    ]
    lines.extend(render_scope_markdown("All Selector Rounds", report["all_selector_rounds"]))
    lines.extend(render_scope_markdown("P1 Mainline Rounds", report["p1_mainline_rounds"]))
    lines.extend(render_scope_markdown("P1 Historical Solo Rounds", report["p1_historical_solo_rounds"]))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fidelity-root", type=Path, default=DEFAULT_FIDELITY_ROOT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()

    report = build_report(args.fidelity_root)
    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(render_markdown(report), encoding="utf-8")

    print(args.report_json)
    print(args.report_md)


if __name__ == "__main__":
    main()
