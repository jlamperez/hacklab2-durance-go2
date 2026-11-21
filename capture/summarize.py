"""Summarize the real event-camera recordings as Markdown.

    python -m capture.summarize                    # data/river_samples
    python -m capture.summarize --plan-session sat_follow2

Reads every session folder's takes.csv (guided capture) and lists folders of unlabelled
recordings. With --plan-session, also lists the takes of the plan that this session has not
recorded yet. Needs h5py only for unlabelled .h5 files.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_rows(folder: Path) -> list[dict]:
    with (folder / "takes.csv").open(newline="", encoding="utf-8") as f:
        return sorted(csv.DictReader(f), key=lambda r: int(r["take"]))


def action_s(r: dict) -> float:
    return (int(r["t_end_us"]) - int(r["t_start_us"])) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO / "data" / "river_samples"))
    ap.add_argument("--plan-session", default="", help="list the plan's takes this session still lacks")
    args = ap.parse_args()
    root = Path(args.root)
    out: list[str] = []

    sessions = sorted(p for p in root.iterdir() if (p / "takes.csv").exists())
    loose = sorted(p for p in root.iterdir() if p.is_dir() and not (p / "takes.csv").exists()
                   and (any(p.glob("*.h5")) or any(p.glob("*.bin"))))

    out.append("| Session | Takes | Action time | Events | Size on disk |\n|---|---|---|---|---|")
    for s in sessions:
        rows = load_rows(s)
        size = sum(f.stat().st_size for f in s.glob("*.h5")) / 2**20
        out.append(f"| {s.name} | {len(rows)} | {sum(map(action_s, rows)):.0f} s | "
                   f"{sum(int(r['n_events']) for r in rows):,} | {size:.1f} MB (.h5) |")
    for s in loose:
        n = len(list(s.glob("*.h5"))) or len(list(s.glob("*.bin")))
        size = sum(f.stat().st_size for f in s.glob("*.h5")) / 2**20
        out.append(f"| {s.name} (no labels) | {n} | | | {size:.1f} MB (.h5) |")

    out.append("\n### By label (all sessions)\n")
    out.append("| Label | Takes | Action time | Median events/s | Sessions |\n|---|---|---|---|---|")
    by_label: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for s in sessions:
        for r in load_rows(s):
            by_label[r["label"]].append((s.name, r))
    for label in sorted(by_label, key=lambda k: (-len(by_label[k]), k)):
        items = by_label[label]
        rates = sorted(int(r["n_events"]) / max(action_s(r) + 1.5, 0.1) for _, r in items)
        out.append(f"| {label} | {len(items)} | {sum(action_s(r) for _, r in items):.0f} s | "
                   f"{rates[len(rates) // 2]:,.0f} | {', '.join(sorted({s for s, _ in items}))} |")

    for s in sessions:
        out.append(f"\n### {s.name}\n")
        out.append("| Take | Label | Phase | Distance | Camera | Ball | Events | Action | Marks |\n"
                   "|---|---|---|---|---|---|---|---|---|")
        for r in load_rows(s):
            marks = len([m for m in r["marks_us"].split() if m])
            out.append(f"| {r['take']} | {r['label']} | {r['phase']} | {r['distance_m']} | {r['camera']} | "
                       f"{r['ball']} | {int(r['n_events']):,} | {action_s(r):.1f} s | {marks or ''} |")

    if args.plan_session:
        from capture.record import PLANS, session_plan
        plan = PLANS[session_plan(root / args.plan_session)]()
        done = {int(r["take"]) for r in load_rows(root / args.plan_session)}
        missing = [i for i in range(len(plan)) if i not in done]
        out.append(f"\n### Not recorded yet in {args.plan_session} ({len(missing)} of {len(plan)} plan takes)\n")
        out.append("| Take | Phase | Label | Instruction |\n|---|---|---|---|")
        for i in missing:
            t = plan[i]
            out.append(f"| {i} | {t.phase} | {t.label} | {t.instruction.split('. ')[0]} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
