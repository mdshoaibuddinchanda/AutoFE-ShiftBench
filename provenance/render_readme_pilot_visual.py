"""Render the README's bounded-pilot execution chart from the audited report.

This chart shows task completion, same-cell runtime, and the last recorded
launch-gate state. It does not plot corrected benchmark performance or infer
results for the 25-dataset campaign.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "provenance" / "four_dataset_pilot_diagnosis.md"
DEFAULT_OUTPUT = ROOT / "provenance" / "figures" / "four_dataset_pilot_status.svg"
READINESS_SOURCE = ROOT / "provenance" / "reviewer1_launch_readiness_v5.json"


def _cells(report: str, label: str) -> list[str]:
    prefix = f"| {label} |"
    matches = [line for line in report.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one report row for {label!r}; found {len(matches)}")
    return [cell.strip() for cell in matches[0].strip("|").split("|")]


def _integer(value: str) -> int:
    return int(value.replace(",", ""))


def _section(report: str, heading: str) -> str:
    marker = f"## {heading}\n"
    if report.count(marker) != 1:
        raise ValueError(f"Expected exactly one section named {heading!r}")
    return report.split(marker, 1)[1].split("\n## ", 1)[0]


def render(source: Path = SOURCE, output: Path = DEFAULT_OUTPUT) -> Path:
    report = source.read_text(encoding="utf-8")
    completion_section = _section(report, "Final accounting and verdict")
    profile_section = _section(report, "Paired performance verification")
    policies = {}
    for name in ("row_level", "group_aware"):
        cells = _cells(completion_section, f"`{name}`")
        intended, success = _integer(cells[1]), _integer(cells[2])
        terminal_failures, remaining = _integer(cells[6]), _integer(cells[7])
        if intended != success or terminal_failures or remaining:
            raise ValueError(f"Pilot completion changed for {name}; review chart wording")
        policies[name] = (intended, success)

    baseline = _cells(profile_section, "Post-retry baseline")
    optimized = _cells(profile_section, "Both changes, committed main branch")
    if baseline[1] != "560 / 560" or optimized[1] != "560 / 560":
        raise ValueError("Paired profile coverage changed; review chart wording")
    baseline_seconds = float(baseline[2])
    optimized_seconds = float(optimized[2])
    if baseline_seconds <= 0 or optimized_seconds <= 0:
        raise ValueError("Paired profile timings must be positive")
    scale = 380 / max(baseline_seconds, optimized_seconds)
    baseline_width = baseline_seconds * scale
    optimized_width = optimized_seconds * scale
    speedup = baseline_seconds / optimized_seconds
    readiness = json.loads(READINESS_SOURCE.read_text(encoding="utf-8")) if READINESS_SOURCE.exists() else {}
    recorded_ready = (
        readiness.get("all_seven_ready") is True
        and len(readiness.get("runs", [])) == 7
        and all(item.get("ready_for_frozen_command") is True for item in readiness["runs"])
    )
    gate_label = (
        "Last recorded gate: READY; full run not started" if recorded_ready
        else "25-dataset campaign: launch gate pending"
    )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="420" viewBox="0 0 1100 420" role="img" aria-labelledby="title desc">
  <title id="title">Bounded four-dataset pilot execution status</title>
  <desc id="desc">Row-level and group-aware policies each completed {policies['row_level'][1]:,} of {policies['row_level'][0]:,} tasks with zero terminal failures. On a separate matched 560-task profile, baseline runtime was {baseline_seconds:.1f} seconds and the committed implementation took {optimized_seconds:.1f} seconds. {gate_label}. These are code-verification measurements, not corrected scientific results.</desc>
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: #152438; }}
    .title {{ font-size: 25px; font-weight: 700; }}
    .heading {{ font-size: 18px; font-weight: 700; }}
    .label {{ font-size: 14px; font-weight: 600; }}
    .value {{ font-size: 14px; font-weight: 700; }}
    .note {{ font-size: 12px; fill: #526278; }}
    .badge {{ font-size: 12px; font-weight: 700; fill: #14558a; }}
    .warning {{ fill: #77520a; }}
  </style>
  <rect width="1100" height="420" fill="#f6f8fb"/>
  <text class="title" x="40" y="44">Four-dataset execution pilot</text>
  <rect x="827" y="18" width="233" height="32" rx="16" fill="#e4f1fb"/>
  <text class="badge" x="944" y="39" text-anchor="middle">CODE VERIFICATION</text>

  <rect x="40" y="72" width="490" height="262" rx="14" fill="#fff" stroke="#dce3ec"/>
  <text class="heading" x="64" y="108">Authoritative task completion</text>
  <text class="note" x="64" y="129">Four datasets · 10 conditions · 14 pipelines · 10 classifiers</text>
  <text class="label" x="64" y="165">Row-level</text>
  <text class="value" x="505" y="165" text-anchor="end">{policies['row_level'][1]:,} / {policies['row_level'][0]:,}</text>
  <rect x="64" y="177" width="441" height="19" rx="9" fill="#e8edf3"/>
  <rect x="64" y="177" width="441" height="19" rx="9" fill="#178a65"/>
  <text class="label" x="64" y="232">Group-aware</text>
  <text class="value" x="505" y="232" text-anchor="end">{policies['group_aware'][1]:,} / {policies['group_aware'][0]:,}</text>
  <rect x="64" y="244" width="441" height="19" rx="9" fill="#e8edf3"/>
  <rect x="64" y="244" width="441" height="19" rx="9" fill="#178a65"/>
  <text class="note" x="64" y="304">Zero terminal failures in both policies</text>

  <rect x="550" y="72" width="510" height="262" rx="14" fill="#fff" stroke="#dce3ec"/>
  <text class="heading" x="574" y="108">Historical 560-task runtime</text>
  <text class="note" x="574" y="129">Same four datasets, cells, CPU workers, and outputs</text>
  <text class="label" x="574" y="165">Repaired baseline</text>
  <text class="value" x="1036" y="165" text-anchor="end">{baseline_seconds:.1f} s</text>
  <rect x="574" y="177" width="{baseline_width:.1f}" height="21" rx="10" fill="#778ba6"/>
  <text class="label" x="574" y="234">Historical implementation</text>
  <text class="value" x="1036" y="234" text-anchor="end">{optimized_seconds:.1f} s</text>
  <rect x="574" y="246" width="{optimized_width:.1f}" height="21" rx="10" fill="#286fb0"/>
  <text class="note" x="574" y="304">{speedup:.2f}× faster in this bounded profile; exact output parity</text>

  <rect x="40" y="352" width="1020" height="43" rx="10" fill="#fff2d6" stroke="#e9c98a"/>
  <text class="warning" x="58" y="379" font-size="14" font-weight="700">{gate_label}</text>
  <text class="warning" x="520" y="379" font-size="13">Scientific effects remain PENDING CORRECTED RUN.</text>
</svg>
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(render(args.source, args.output))
