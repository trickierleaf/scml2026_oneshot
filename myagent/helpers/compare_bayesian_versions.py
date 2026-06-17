from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp_tournament_test/.matplotlib")))

from scml_agents.scml2023.oneshot.team_poli_usp.quantity_oriented_agent import (
    QuantityOrientedAgent,
)
from scml_agents.scml2024.oneshot.team_miyajima_oneshot.cautious import (
    CautiousOneShotAgent,
)
from scml_agents.scml2025.oneshot.teamyuzuru.agent import CostAverseAgent

from myagent.bayesian_agent import BayesianAgent
from myagent.bayesian_agent_022 import BayesianAgent022
from myagent.helpers.runner import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a tournament with current BayesianAgent and BayesianAgent022 together."
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--configs", type=int, default=10)
    parser.add_argument("--path", type=Path, default=Path("tmp_tournament_test/bayes_ab"))
    args = parser.parse_args()

    run(
        competitors=(
            BayesianAgent,
            BayesianAgent022,
            CautiousOneShotAgent,
            CostAverseAgent,
            QuantityOrientedAgent,
        ),
        competition="oneshot",
        n_steps=args.steps,
        n_configs=args.configs,
        tournament_path=args.path,
    )


if __name__ == "__main__":
    main()
