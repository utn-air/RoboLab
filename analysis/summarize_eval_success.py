# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Print compact per-task and overall success rates for eval results."""

import argparse
import json
import os
from collections import OrderedDict


def resolve_results_dir(path_or_name: str) -> str:
    if os.path.isdir(path_or_name):
        return path_or_name

    candidate = os.path.join("robolab", "output", path_or_name)
    if os.path.isdir(candidate):
        return candidate

    return path_or_name


def load_results(results_dir: str) -> list[dict]:
    jsonl_path = os.path.join(results_dir, "episode_results.jsonl")
    json_path = os.path.join(results_dir, "episode_results.json")

    if os.path.exists(jsonl_path):
        episodes = []
        with open(jsonl_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON on {jsonl_path}:{line_num}: {exc}") from exc
        return episodes

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {json_path}")
        return data

    raise FileNotFoundError(
        f"Could not find episode_results.jsonl or episode_results.json in {results_dir}"
    )


def dedupe_episodes(episodes: list[dict]) -> list[dict]:
    """Keep the last result for each task/episode/env tuple."""
    deduped = OrderedDict()
    for episode in episodes:
        key = (
            episode.get("env_name"),
            episode.get("episode"),
            episode.get("env_id", 0),
        )
        deduped[key] = episode
    return list(deduped.values())


def format_rate(successes: int, total: int) -> str:
    rate = (successes / total * 100.0) if total else 0.0
    return f"{successes}/{total} ({rate:.2f}%)"


def summarize(
    episodes: list[dict],
    task_order: list[str] | None,
    expected_runs: int | None,
) -> tuple[list[tuple[str, int, int, float]], tuple[int, int, float]]:
    grouped: OrderedDict[str, list[dict]] = OrderedDict()

    if task_order:
        for task in task_order:
            grouped[task] = []

    for episode in episodes:
        task = episode.get("env_name")
        if not task:
            continue
        if task_order and task not in grouped:
            continue
        grouped.setdefault(task, []).append(episode)

    rows = []
    total_successes = 0
    total_runs = 0

    for task, task_episodes in grouped.items():
        task_episodes = sorted(
            task_episodes,
            key=lambda ep: (ep.get("episode", -1), ep.get("env_id", -1)),
        )
        successes = sum(1 for ep in task_episodes if ep.get("success"))
        runs = len(task_episodes)
        denominator = expected_runs if expected_runs is not None else runs
        rate = successes / denominator if denominator else 0.0
        rows.append((task, successes, denominator, rate))
        total_successes += successes
        total_runs += denominator

    total_rate = total_successes / total_runs if total_runs else 0.0
    return rows, (total_successes, total_runs, total_rate)


def print_table(rows: list[tuple[str, int, int, float]], total: tuple[int, int, float]) -> None:
    task_width = max([len("Task"), *(len(row[0]) for row in rows), len("TOTAL")])
    result_width = max(len("Success"), len("000/000 (100.00%)"))

    print(f"{'Task':<{task_width}}  {'Success':<{result_width}}")
    print(f"{'-' * task_width}  {'-' * result_width}")

    for task, successes, runs, rate in rows:
        result = f"{successes}/{runs} ({rate * 100.0:.2f}%)"
        print(f"{task:<{task_width}}  {result:<{result_width}}")

    total_successes, total_runs, total_rate = total
    print(f"{'-' * task_width}  {'-' * result_width}")
    print(f"{'TOTAL':<{task_width}}  {total_successes}/{total_runs} ({total_rate * 100.0:.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize eval success as per-task x/N and total x/N."
    )
    parser.add_argument(
        "results",
        help="Output folder path, or folder name under robolab/output.",
    )
    parser.add_argument(
        "--task",
        nargs="+",
        default=None,
        help="Optional task order/filter.",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=None,
        help="Expected denominator per task, e.g. 3 for three runs per task.",
    )
    args = parser.parse_args()

    results_dir = resolve_results_dir(args.results)
    episodes = dedupe_episodes(load_results(results_dir))
    rows, total = summarize(
        episodes=episodes,
        task_order=args.task,
        expected_runs=args.expected_runs,
    )

    if not rows:
        raise SystemExit(f"No matching episodes found in {results_dir}")

    print_table(rows, total)


if __name__ == "__main__":
    main()
