import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp_tournament_test/.matplotlib")))

from negmas.helpers import humanize_time
from rich import print
from scml.oneshot.agents import RandomOneShotAgent, SyncRandomOneShotAgent
from scml.std.agents import SyncRandomStdAgent
from scml.utils import anac2024_oneshot, anac2024_std
from tabulate import tabulate
# Monkey patch for local simulation:
# Disable NegMAS worker recycling to avoid hangs in parallel tournament runs.
import negmas.tournaments.tournaments as negmas_tournaments
negmas_tournaments.MAX_TASKS_PER_CHILD = None

def run(
    competitors=tuple(),
    competition="oneshot",
    reveal_names=True,
    n_steps=50,
    n_configs=20,
    tournament_path=Path("tmp_tournament_test"),
    display=True,
):
    """
    **Not needed for submission.** You can use this function to test your agent.

    Args:
        competitors: A list of competitor classes
        competition: The competition type to run (possibilities are oneshot, std).
        n_steps:     The number of simulation steps.
        n_configs:   Number of different world configurations to try.
                     Different world configurations will correspond to
                     different number of factories, profiles
                     , production graphs etc

    Returns:
        None

    Remarks:

        - This function will take several minutes to run.
        - To speed it up, use a smaller `n_step` value

    """

    if competition == "oneshot":
        competitors = list(competitors) + [RandomOneShotAgent, SyncRandomOneShotAgent]
    else:
        competitors = list(competitors) + [SyncRandomStdAgent, RandomOneShotAgent]

    start = time.perf_counter()
    if competition == "std":
        runner = anac2024_std
    else:
        runner = anac2024_oneshot
    known_stages = set(tournament_path.glob("*-stage-*"))
    results = runner(
        competitors=competitors,
        verbose=True,
        n_steps=n_steps,
        n_configs=n_configs,
        tournament_path=tournament_path,
        parallelism="serial",
    )
    new_stages = sorted(
        set(tournament_path.glob("*-stage-*")) - known_stages,
        key=lambda path: path.stat().st_mtime,
    )
    if new_stages:
        results.tournament_stage_path = new_stages[-1]  # type: ignore[attr-defined]
    # just make names shorter
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(  # type: ignore
        "."
    ).str[
        -1
    ]
    # display results
    if display:
        print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))  # type: ignore
        if new_stages:
            print(f"Saved tournament data to {new_stages[-1]}")
        print(f"Finished in {humanize_time(time.perf_counter() - start)}")
    return results


def simulate(
    competitors=tuple(),
    competition="oneshot",
    n_steps=50,
    n_configs=5,
    tournament_path=Path("tmp_tournament_test"),
):
    """Run a tournament and return the folder that can be analyzed later."""
    results = run(
        competitors=competitors,
        competition=competition,
        n_steps=n_steps,
        n_configs=n_configs,
        tournament_path=tournament_path,
        display=False,
    )
    return results.tournament_stage_path  # type: ignore[attr-defined]


if __name__ == "__main__":
    import sys

    run(competition=sys.argv[1] if len(sys.argv) > 1 else "oneshot")
