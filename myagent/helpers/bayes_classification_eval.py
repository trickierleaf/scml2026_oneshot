from __future__ import annotations

import argparse
import inspect
import os
import random
import subprocess
import sys
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp_tournament_test/.matplotlib")))

from scml.oneshot.agents import (
    EqualDistOneShotAgent,
    GreedyOneShotAgent,
    RandomOneShotAgent,
    SyncRandomOneShotAgent,
)
from scml.oneshot.agents.rand import RandDistOneShotAgent
from scml.utils import anac2024_oneshot
import negmas.tournaments.tournaments as negmas_tournaments

from myagent.bayesian_agent import BayesianAgent

negmas_tournaments.MAX_TASKS_PER_CHILD = None


TARGET_TYPES = {
    "GreedyOneShotAgent": "GreedyOneShotAgent",
    "RandomOneShotAgent": "NonGreedy",
    "EqualDistOneShotAgent": "NonGreedy",
    "SyncRandomOneShotAgent": "NonGreedy",
    "RandDistOneShotAgent": "NonGreedy",
}


def _agent_object(adapter):
    return getattr(adapter, "adapted_object", adapter)


def _seed_everything(seed: int):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


def _ensure_hash_seed(seed: int):
    if os.environ.get("PYTHONHASHSEED") == str(seed):
        return

    module = "myagent.helpers.bayes_classification_eval"
    if Path(sys.argv[0]).name == "bayes_clasifaication_eval.py":
        module = "myagent.helpers.bayes_clasifaication_eval"

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    completed = subprocess.run(
        [sys.executable, "-m", module, *sys.argv[1:]],
        cwd=PROJECT_ROOT,
        env=env,
    )
    raise SystemExit(completed.returncode)


def _run_seeded_tournament(args, world_done):
    opponent_pool = [
        RandomOneShotAgent,
        EqualDistOneShotAgent,
        GreedyOneShotAgent,
        SyncRandomOneShotAgent,
        RandDistOneShotAgent,
    ]
    kwargs = {
        "competitors": [BayesianAgent, GreedyOneShotAgent],
        "non_competitors": opponent_pool,
        "n_steps": args.steps,
        "n_configs": args.configs,
        "max_worlds_per_config": args.worlds,
        "n_competitors_per_world": 2,
        "tournament_path": Path(args.path),
        "parallelism": "serial",
        "compact": True,
        "forced_logs_fraction": 0.0,
        "verbose": False,
        "world_progress_callback": world_done,
    }

    parameters = inspect.signature(anac2024_oneshot).parameters
    for seed_arg in ("random_seed", "seed", "base_seed"):
        if seed_arg in parameters:
            kwargs[seed_arg] = args.seed
            break

    return anac2024_oneshot(**kwargs)


def _collect_world_predictions(world, samples):
    current_step = int(getattr(world, "current_step", 0))
    n_steps = int(getattr(world, "n_steps", 1))
    if current_step < n_steps - 1:
        return

    ids = list(world.non_system_agent_ids)
    objects = {
        agent_id: _agent_object(agent)
        for agent_id, agent in zip(ids, world.non_system_agents, strict=False)
    }

    bayes_agents = [
        (agent_id, agent)
        for agent_id, agent in objects.items()
        if isinstance(agent, BayesianAgent)
    ]

    for bayes_id, bayes in bayes_agents:
        partners = [
            partner
            for partner in bayes._all_partners()
            if partner in objects and partner != bayes_id
        ]

        for partner in partners:
            true_type = TARGET_TYPES.get(type(objects[partner]).__name__, "Other")
            if true_type == "Other":
                continue

            posteriors = bayes.opponent_posteriors(partner)
            top = max(posteriors, key=posteriors.get)

            samples.append(
                {
                    "truth": true_type,
                    "pred": bayes.opponent_type(partner),
                    "observations": int(bayes._opponent_observations[partner]),
                    "top": top,
                    "top_p": posteriors[top],
                    "partner_class": type(objects[partner]).__name__,

                    # 診断情報
                    "veto_reason": getattr(
                        bayes,
                        "_non_greedy_veto",
                        {},
                    ).get(partner),
                    "evidence_counts": dict(
                        getattr(
                            bayes,
                            "_evidence_counts",
                            {},
                        ).get(partner, {})
                    ),
                    "logits": dict(
                        getattr(
                            bayes,
                            "_opponent_logits",
                            {},
                        ).get(partner, {})
                    ),
                    "recent_logit_history": list(
                        getattr(
                            bayes,
                            "_logit_history",
                            {},
                        ).get(partner, [])
                    )[-5:],
                }
            )


def _print_report(samples):
    labels = [
        "GreedyOneShotAgent",
        "NonGreedy",
        "Unknown",
        "Other",
    ]

    correct = sum(sample["pred"] == sample["truth"] for sample in samples)
    total = len(samples)
    print(f"accuracy={correct}/{total}={correct / total if total else 0:.3f}")

    matrix = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        matrix[sample["truth"]][sample["pred"]] += 1

    print("truth,pred,count")
    for truth in labels[:2]:
        for pred in labels:
            count = matrix[truth].get(pred, 0)
            if count:
                print(f"{truth},{pred},{count}")

    greedy_to_nongreedy_reasons = Counter()
    greedy_to_unknown_reasons = Counter()
    nongreedy_to_greedy_reasons = Counter()

    greedy_to_nongreedy_evidence = Counter()
    nongreedy_to_greedy_evidence = Counter()

    greedy_to_nongreedy_observation_counts = Counter()
    nongreedy_to_greedy_observation_counts = Counter()

    greedy_to_nongreedy_top = Counter()
    nongreedy_to_greedy_top = Counter()

    greedy_to_unknown_top = Counter()
    nongreedy_to_unknown_top = Counter()

    greedy_to_unknown_top_p_bins = Counter()
    nongreedy_to_unknown_top_p_bins = Counter()

    for sample in samples:
        reason = sample.get("veto_reason") or "NO_VETO"

        if sample["truth"] == "GreedyOneShotAgent" and sample["pred"] == "NonGreedy":
            greedy_to_nongreedy_reasons[reason] += 1
            greedy_to_nongreedy_observation_counts[sample["observations"]] += 1
            greedy_to_nongreedy_top[sample["top"]] += 1

            for key, value in sample.get("evidence_counts", {}).items():
                greedy_to_nongreedy_evidence[key] += int(value)

        if sample["truth"] == "GreedyOneShotAgent" and sample["pred"] == "Unknown":
            greedy_to_unknown_reasons[reason] += 1

        if sample["truth"] == "NonGreedy" and sample["pred"] == "GreedyOneShotAgent":
            nongreedy_to_greedy_reasons[reason] += 1
            nongreedy_to_greedy_observation_counts[sample["observations"]] += 1
            nongreedy_to_greedy_top[sample["top"]] += 1

            for key, value in sample.get("evidence_counts", {}).items():
                nongreedy_to_greedy_evidence[key] += int(value)
        
        if sample["truth"] == "GreedyOneShotAgent" and sample["pred"] == "Unknown":
            greedy_to_unknown_top[sample["top"]] += 1
            greedy_to_unknown_top_p_bins[int(sample["top_p"] * 10) / 10] += 1

        if sample["truth"] == "NonGreedy" and sample["pred"] == "Unknown":
            nongreedy_to_unknown_top[sample["top"]] += 1
            nongreedy_to_unknown_top_p_bins[int(sample["top_p"] * 10) / 10] += 1

    print()
    print("diagnostics")

    print("Greedy -> NonGreedy veto reasons")
    print(greedy_to_nongreedy_reasons)

    print("Greedy -> Unknown veto reasons")
    print(greedy_to_unknown_reasons)

    print("NonGreedy -> Greedy veto reasons")
    print(nongreedy_to_greedy_reasons)

    print("Greedy -> NonGreedy evidence counts")
    print(greedy_to_nongreedy_evidence.most_common(20))

    print("NonGreedy -> Greedy evidence counts")
    print(nongreedy_to_greedy_evidence.most_common(20))

    print("Greedy -> NonGreedy observation counts")
    print(greedy_to_nongreedy_observation_counts)

    print("NonGreedy -> Greedy observation counts")
    print(nongreedy_to_greedy_observation_counts)

    print("Greedy -> NonGreedy top posterior class")
    print(greedy_to_nongreedy_top)

    print("NonGreedy -> Greedy top posterior class")
    print(nongreedy_to_greedy_top)

    print("Greedy -> Unknown top posterior class")
    print(greedy_to_unknown_top)

    print("Greedy -> Unknown top_p bins")
    print(greedy_to_unknown_top_p_bins)

    print("NonGreedy -> Unknown top posterior class")
    print(nongreedy_to_unknown_top)

    print("NonGreedy -> Unknown top_p bins")
    print(nongreedy_to_unknown_top_p_bins)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--configs", type=int, default=2)
    parser.add_argument("--worlds", type=int, default=12)
    parser.add_argument("--path", default="tmp_tournament_test/bayes_prod_eval")
    parser.add_argument("--samples", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for Python, NumPy, and SCML tournament generation if supported.",
    )
    args = parser.parse_args()
    _ensure_hash_seed(args.seed)
    _seed_everything(args.seed)

    samples = []

    def world_done(world):
        if world is not None:
            _collect_world_predictions(world, samples)

    _run_seeded_tournament(args, world_done)

    _print_report(samples)

    if args.samples:
        print("mistake_samples")
        for sample in samples:
            if sample["truth"] == sample["pred"]:
                continue

            print("=" * 80)
            print(
                "truth=", sample["truth"],
                "pred=", sample["pred"],
                "partner_class=", sample["partner_class"],
                "observations=", sample["observations"],
                "top=", sample["top"],
                "top_p=", round(sample["top_p"], 3),
            )
            print("logits=", sample.get("logits"))
            print("veto_reason=", sample.get("veto_reason"))
            print("evidence_counts=", sample.get("evidence_counts"))
            print("recent_logit_history=")
            for item in sample.get("recent_logit_history", []):
                print(item)


if __name__ == "__main__":
    main()
