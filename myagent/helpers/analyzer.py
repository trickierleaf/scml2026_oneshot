from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp_tournament_test/.matplotlib")))

import pandas as pd

AGENT_METRICS = (
    "score",
    "balance",
    "productivity",
    "shortfall_quantity",
    "shortfall_penalty",
    "storage_cost",
    "disposal_cost",
    "inventory_input",
    "inventory_output",
    "inventory_penalized",
    "bankrupt",
)

DEFAULT_MAX_EVENTS_PER_STEP = 60

OLD_OUTPUTS = (
    "agent_daily_metrics.csv",
    "agent_improvement_metrics.png",
    "contracts.csv",
    "contracts_by_day.csv",
    "contracts_by_partner.csv",
    "product_daily_metrics.csv",
    "report.md",
    "score_productivity.png",
    "score_productivity_by_step.csv",
    "score_productivity_history.csv",
    "world_daily_metrics.csv",
)

def latest_stage(root: Path = Path("tmp_tournament_test")) -> Path:
    stages = sorted(root.glob("*-stage-*"), key=lambda path: path.stat().st_mtime)
    if not stages:
        raise FileNotFoundError(f"No tournament stages found under {root}")
    return stages[-1]


def world_dirs(stage_path: Path) -> list[Path]:
    return sorted(
        path
        for path in stage_path.glob("*/*")
        if path.is_dir() and (path / "stats.parquet").exists()
    )


def clean_old_outputs(output_dir: Path) -> None:
    for name in OLD_OUTPUTS:
        path = output_dir / name
        if path.exists():
            path.unlink()


def short_type(agent_type: str) -> str:
    return str(agent_type).split(":")[-1].split(".")[-1]


def process_of(agent_name: str) -> str:
    if "@" not in agent_name:
        return ""
    return agent_name.rsplit("@", 1)[-1]


def safe_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [safe_value(item) for item in value]
    if isinstance(value, list):
        return [safe_value(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if math.isfinite(value):
            return round(value, 4)
        return None
    return value


def price_range_from_issues(issues: Any) -> str:
    text = " ".join(str(item) for item in issues) if isinstance(issues, list) else str(issues)
    match = re.search(r"unit_price:\s*\(([^)]+)\)", text)
    if not match:
        match = re.search(r"ContiguousIssue\(\(([^)]+)\),\s*unit_price\)", text)
    if not match:
        return ""
    return match.group(1).replace(" ", "")


def parsed(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        try:
            return eval(value, {"__builtins__": {}, "set": set})  # noqa: S307
        except Exception:
            return None


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def offer_parts(offer: Any) -> tuple[Any, Any, Any]:
    offer = parsed(offer)
    if isinstance(offer, (list, tuple)) and len(offer) >= 3:
        return offer[0], offer[1], offer[2]
    return None, None, None


def same_offer(left: Any, right: Any) -> bool:
    left = parsed(left)
    right = parsed(right)
    if is_missing(left) or is_missing(right):
        return False
    try:
        return tuple(left) == tuple(right)
    except TypeError:
        return False


def other_side(agent: Any, buyer: Any, seller: Any, partners: Any) -> Any:
    if agent == buyer:
        return seller
    if agent == seller:
        return buyer
    partners = parsed(partners)
    if isinstance(partners, (list, tuple)):
        for partner in partners:
            if partner != agent:
                return partner
    return None


def agent_for_process(buyer: Any, seller: Any, process: Any) -> Any:
    process = str(process)
    if process_of(str(buyer)) == process:
        return buyer
    if process_of(str(seller)) == process:
        return seller
    return None


def current_offer_sender(event: dict[str, Any]) -> Any:
    current_proposer = event.get("current_proposer")
    return event.get("current_proposer_agent") or current_proposer


def offer_sender(event: dict[str, Any], offer_index: int, offerer: Any, outcome: Any) -> Any:
    offerer_agents = parsed(event.get("new_offerer_agents")) or []
    if isinstance(offerer_agents, (list, tuple)) and offer_index < len(offerer_agents):
        return offerer_agents[offer_index]
    if offerer == event.get("current_proposer") and same_offer(outcome, event.get("current_offer")):
        return current_offer_sender(event)
    return offerer


def normalized_new_offer(entry: Any) -> tuple[Any, Any]:
    """Return (receiver, offer) from a NegMAS history new_offers entry."""
    entry = parsed(entry)
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        receiver, outcome = entry[0], entry[1]
        if isinstance(outcome, (list, tuple)) and len(outcome) >= 3:
            return receiver, outcome
    if isinstance(entry, (list, tuple)) and len(entry) >= 3:
        return None, entry
    return None, None


def infer_first_process_by_day(negotiations: pd.DataFrame) -> dict[Any, str]:
    """Infer which process (@0/@1) makes the first offer on each simulation day."""
    first_by_day: dict[Any, tuple[float, int, str]] = {}
    for _, negotiation in negotiations.iterrows():
        sim_step = negotiation.get("sim_step")
        history = parsed(negotiation.get("history"))
        if not isinstance(history, list):
            continue
        for index, event in enumerate(history):
            if not isinstance(event, dict):
                continue
            new_offers = parsed(event.get("new_offers")) or []
            if not isinstance(new_offers, (list, tuple)):
                new_offers = []
            sender = None
            if new_offers:
                receiver, outcome = normalized_new_offer(new_offers[0])
                if not is_missing(outcome):
                    sender = offer_sender(event, 0, receiver, outcome)
            if is_missing(sender):
                sender = current_offer_sender(event)
            process = process_of(str(sender)) if not is_missing(sender) else ""
            if process not in {"0", "1"}:
                continue
            relative_time = event.get("relative_time")
            try:
                relative_time = float(relative_time)
            except Exception:
                relative_time = float("inf")
            current = first_by_day.get(sim_step)
            if current is None or (relative_time, index) < (current[0], current[1]):
                first_by_day[sim_step] = (relative_time, index, process)
            break
    return {day: value[2] for day, value in first_by_day.items()}


def records(data: pd.DataFrame) -> list[dict[str, Any]]:
    if data.empty:
        return []
    return [
        {column: safe_value(value) for column, value in row.items()}
        for row in data.to_dict(orient="records")
    ]


def limited_records(data: pd.DataFrame, max_rows: int | None) -> list[dict[str, Any]]:
    """Convert a dataframe to compact records, limiting bulky per-step tables."""
    if max_rows is not None and max_rows >= 0 and len(data) > max_rows:
        data = data.head(max_rows).copy()
    return records(data)


def negotiation_flow_rows(
    world_name: str, negotiations: pd.DataFrame, negs: pd.DataFrame
) -> pd.DataFrame:
    if negotiations.empty or "history" not in negotiations.columns:
        return pd.DataFrame()

    neg_id_by_uuid = (
        dict(zip(negs["name"], negs["id"], strict=False))
        if not negs.empty and {"name", "id"}.issubset(negs.columns)
        else {}
    )
    first_process_by_day = infer_first_process_by_day(negotiations)
    rows: list[dict[str, Any]] = []
    for _, negotiation in negotiations.iterrows():
        uuid = negotiation.get("id")
        neg_id = neg_id_by_uuid.get(uuid, uuid)
        buyer = negotiation.get("buyer")
        seller = negotiation.get("seller")
        first_process = first_process_by_day.get(negotiation.get("sim_step"))
        agreement = parsed(negotiation.get("agreement"))
        final_status = negotiation.get("final_status")
        history = parsed(negotiation.get("history"))
        if not isinstance(history, list):
            history = []

        last_round = None
        last_relative_time = None
        last_time = None
        turn_number = 0
        for index, event in enumerate(history):
            if not isinstance(event, dict):
                continue
            new_offers = parsed(event.get("new_offers")) or []
            if not isinstance(new_offers, (list, tuple)):
                new_offers = []
            event_step = event.get("step", index)
            last_relative_time = event.get("relative_time")
            last_time = event.get("time")
            offer_entries = []
            for offer_index, new_offer in enumerate(new_offers):
                receiver, outcome = normalized_new_offer(new_offer)
                if is_missing(outcome):
                    continue
                sender = offer_sender(event, offer_index, receiver, outcome)
                if sender == receiver:
                    sender = other_side(receiver, buyer, seller, negotiation.get("partners"))
                if is_missing(sender) and turn_number == 0 and first_process is not None:
                    sender = agent_for_process(buyer, seller, first_process)
                if is_missing(receiver):
                    receiver = other_side(sender, buyer, seller, negotiation.get("partners"))
                offer_entries.append((offer_index, sender, receiver, outcome))

            if not offer_entries:
                current_offer = parsed(event.get("current_offer"))
                if not is_missing(current_offer):
                    sender = current_offer_sender(event)
                    if is_missing(sender) and turn_number == 0 and first_process is not None:
                        sender = agent_for_process(buyer, seller, first_process)
                    receiver = other_side(sender, buyer, seller, negotiation.get("partners"))
                    offer_entries.append((0, sender, receiver, current_offer))

            for offer_index, sender, receiver, outcome in offer_entries:
                turn_number += 1
                last_round = turn_number
                quantity, delivery_step, unit_price = offer_parts(outcome)
                rows.append(
                    {
                        "world": world_name,
                        "id": f"{uuid}:{event_step}:offer:{offer_index}",
                        "neg_id": neg_id,
                        "neg_uuid": uuid,
                        "round": turn_number,
                        "mechanism_step": event_step,
                        "relative_time": event.get("relative_time"),
                        "time": event.get("time"),
                        "event": "offer",
                        "sender": sender,
                        "receiver": receiver,
                        "quantity": quantity,
                        "delivery_step": delivery_step,
                        "unit_price": unit_price,
                        "current_proposer": event.get("current_proposer"),
                        "n_acceptances": event.get("n_acceptances"),
                        "first_process": first_process,
                        "matches_agreement": same_offer(outcome, agreement),
                        "final_status": final_status,
                        "sim_step": negotiation.get("sim_step"),
                        "buyer": buyer,
                        "seller": seller,
                        "issues": negotiation.get("issues"),
                        "price_range": price_range_from_issues(negotiation.get("issues")),
                    }
                )

        result_round = last_round
        if isinstance(result_round, int):
            result_round += 1
        quantity, delivery_step, unit_price = offer_parts(agreement)
        rows.append(
            {
                "world": world_name,
                "id": f"{uuid}:result",
                "neg_id": neg_id,
                "neg_uuid": uuid,
                "round": result_round,
                "relative_time": last_relative_time,
                "time": last_time,
                "event": "agreement" if final_status == "succeeded" else final_status,
                "sender": None,
                "receiver": None,
                "quantity": quantity,
                "delivery_step": delivery_step,
                "unit_price": unit_price,
                "current_proposer": None,
                "n_acceptances": None,
                "first_process": first_process,
                "matches_agreement": bool(final_status == "succeeded"),
                "final_status": final_status,
                "sim_step": negotiation.get("sim_step"),
                "buyer": buyer,
                "seller": seller,
                "issues": negotiation.get("issues"),
                "price_range": price_range_from_issues(negotiation.get("issues")),
            }
        )

    return pd.DataFrame(rows)


def read_parquet_columns(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        return pd.read_parquet(path)
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        data = pd.read_parquet(path)
        return data.loc[:, [column for column in columns if column in data.columns]]


def read_stage(
    stage_path: Path,
    world_names: set[str] | None = None,
    include_contracts: bool = True,
    include_actions: bool = True,
    include_flows: bool = True,
) -> dict[str, Any]:
    worlds = []
    for world_dir in world_dirs(stage_path):
        if world_names is not None and world_dir.name not in world_names:
            continue
        agents = read_parquet_columns(world_dir / "agents.parquet", ["id", "name", "type"])
        stats = pd.read_parquet(world_dir / "stats.parquet")
        contracts = (
            read_parquet_columns(
                world_dir / "contracts.parquet",
                [
                    "id",
                    "seller_name",
                    "buyer_name",
                    "delivery_time",
                    "quantity",
                    "unit_price",
                    "signed_at",
                    "concluded_at",
                    "issues",
                    "product_name",
                ],
            )
            if include_contracts
            else pd.DataFrame()
        )
        actions = (
            read_parquet_columns(
                world_dir / "actions.parquet",
                [
                    "id",
                    "neg_id",
                    "step",
                    "relative_time",
                    "time",
                    "sender",
                    "receiver",
                    "sender_agent_id",
                    "receiver_agent_id",
                    "state",
                    "quantity",
                    "delivery_step",
                    "unit_price",
                ],
            )
            if include_actions
            else pd.DataFrame()
        )
        if "step" in actions.columns:
            actions["round"] = (
                pd.to_numeric(actions["step"], errors="coerce").fillna(-1).astype(int)
                + 1
            )
        negs_path = world_dir / "negs.parquet"
        negotiations_path = world_dir / "negotiations.parquet"
        read_negotiation_tables = include_actions or include_flows
        negs = (
            read_parquet_columns(
                negs_path,
                [
                    "id",
                    "name",
                    "sim_step",
                    "has_agreement",
                    "agent0_id",
                    "agent1_id",
                    "quantity",
                    "delivery_step",
                    "unit_price",
                    "product",
                    "is_buy",
                    "buyer",
                    "seller",
                    "needed_sales0",
                    "needed_sales1",
                    "needed_supplies0",
                    "needed_supplies1",
                    "trading_price",
                ],
            )
            if read_negotiation_tables and negs_path.exists()
            else pd.DataFrame()
        )
        negotiations = (
            read_parquet_columns(
                negotiations_path,
                [
                    "id",
                    "partners",
                    "issues",
                    "final_status",
                    "agreement",
                    "history",
                    "buyer",
                    "seller",
                    "sim_step",
                ],
            )
            if read_negotiation_tables and negotiations_path.exists()
            else pd.DataFrame()
        )
        flows = (
            negotiation_flow_rows(world_dir.name, negotiations, negs)
            if include_flows
            else pd.DataFrame()
        )

        agent_names_by_id = dict(zip(agents["id"], agents["name"], strict=False))
        if include_actions and {"sender_agent_id", "receiver_agent_id"}.issubset(actions.columns):
            actions["sender_negotiator"] = actions["sender"]
            actions["receiver_negotiator"] = actions["receiver"]
            actions["sender"] = actions["sender_agent_id"].map(agent_names_by_id).fillna(
                actions["sender"]
            )
            actions["receiver"] = actions["receiver_agent_id"].map(agent_names_by_id).fillna(
                actions["receiver"]
            )

        agents = agents.loc[~agents["name"].isin(["NoAgent", "SELLER", "BUYER"])].copy()
        agents["type_short"] = agents["type"].map(short_type)
        agents["process"] = agents["name"].map(process_of)

        if include_actions and not negs.empty:
            if not negotiations.empty and {"id", "issues"}.issubset(negotiations.columns):
                negs = negs.merge(
                    negotiations.loc[:, ["id", "issues"]],
                    left_on="name",
                    right_on="id",
                    how="left",
                    suffixes=("", "_uuid"),
                )
            neg_columns = [
                column
                for column in (
                    "id",
                    "sim_step",
                    "has_agreement",
                    "agent0_id",
                    "agent1_id",
                    "quantity",
                    "delivery_step",
                    "unit_price",
                    "product",
                    "is_buy",
                    "buyer",
                    "seller",
                    "issues",
                    "needed_sales0",
                    "needed_sales1",
                    "needed_supplies0",
                    "needed_supplies1",
                    "trading_price",
                )
                if column in negs.columns
            ]
            actions = actions.merge(
                negs.loc[:, neg_columns],
                left_on="neg_id",
                right_on="id",
                how="left",
                suffixes=("", "_neg"),
            )
            if "issues" in actions.columns:
                actions["price_range"] = actions["issues"].map(price_range_from_issues)
        elif include_actions:
            actions["sim_step"] = actions.get("delivery_step", actions.get("step", 0))
            actions["has_agreement"] = actions["state"] == "agreement"
            actions["price_range"] = ""

        worlds.append(
            {
                "name": world_dir.name,
                "agents": agents,
                "stats": stats,
                "contracts": contracts,
                "actions": actions,
                "flows": flows,
            }
        )
    return {"stage": stage_path, "worlds": worlds}


def agent_metric_rows(stage_data: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for world in stage_data["worlds"]:
        stats = world["stats"]
        for _, agent in world["agents"].iterrows():
            agent_name = agent["name"]
            for step, values in stats.iterrows():
                row = {
                    "key": f"{world['name']}::{agent_name}",
                    "world": world["name"],
                    "step": int(step),
                    "agent": agent_name,
                    "type": agent["type_short"],
                    "process": agent["process"],
                }
                for metric in AGENT_METRICS:
                    column = f"{metric}_{agent_name}"
                    if column in stats:
                        row[metric] = values[column]
                if "score" in row:
                    rows.append(row)
    return pd.DataFrame(rows)


def contract_rows(stage_data: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for world in stage_data["worlds"]:
        data = world["contracts"].copy()
        if data.empty:
            continue
        data.insert(0, "world", world["name"])
        data["is_exogenous"] = (data["seller_name"] == "SELLER") | (
            data["buyer_name"] == "BUYER"
        )
        frames.append(data)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def action_rows(stage_data: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for world in stage_data["worlds"]:
        data = world["actions"].copy()
        if data.empty:
            continue
        data.insert(0, "world", world["name"])
        frames.append(data)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def flow_rows(stage_data: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for world in stage_data["worlds"]:
        data = world["flows"].copy()
        if data.empty:
            continue
        frames.append(data)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def final_ranking(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    latest = (
        history.sort_values(["key", "step"])
        .groupby("key", as_index=False)
        .tail(1)
        .copy()
    )
    penalty_columns = [
        column
        for column in ("shortfall_penalty", "disposal_cost", "shortfall_quantity")
        if column in history.columns
    ]
    totals = (
        history.groupby("key", as_index=False)[penalty_columns].sum(numeric_only=True)
        if penalty_columns
        else pd.DataFrame({"key": latest["key"]})
    )
    totals = totals.rename(
        columns={
            "shortfall_penalty": "total_shortfall_penalty",
            "disposal_cost": "total_disposal_cost",
            "shortfall_quantity": "total_shortfall_quantity",
        }
    )
    if "productivity" in history.columns:
        productivity = (
            history.groupby("key", as_index=False)["productivity"]
            .mean(numeric_only=True)
            .rename(columns={"productivity": "avg_productivity"})
        )
    else:
        productivity = pd.DataFrame({"key": latest["key"]})
    columns = [
        column
        for column in (
            "key",
            "world",
            "agent",
            "type",
            "process",
            "score",
            "balance",
        )
        if column in latest.columns
    ]
    ranking = (
        latest.loc[:, columns]
        .merge(productivity, on="key", how="left")
        .merge(totals, on="key", how="left")
    )
    return ranking.sort_values("score", ascending=False)


def agent_matches_filter(agent: dict[str, Any], agent_filter: str) -> bool:
    """Case-insensitive match against displayed agent id and short type."""
    if not agent_filter:
        return True
    needle = agent_filter.lower()
    agent_name = str(agent.get("agent", "")).lower()
    agent_type = str(agent.get("type", "")).lower()
    return needle in agent_name or needle == agent_type


def filter_agents(agents: list[dict[str, Any]], agent_filter: str) -> list[dict[str, Any]]:
    return [agent for agent in agents if agent_matches_filter(agent, agent_filter)]


def loss_amount(agent: dict[str, Any]) -> float:
    """Total world-level loss used to keep heavy detail views compact."""
    return float(agent.get("total_shortfall_penalty") or 0) + float(
        agent.get("total_disposal_cost") or 0
    )


def high_loss_half_agents(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(agents) <= 1:
        return agents
    n_keep = math.ceil(len(agents) / 2)
    return sorted(
        agents,
        key=lambda agent: (
            loss_amount(agent),
            float(agent.get("total_shortfall_quantity") or 0),
            -float(agent.get("score") or 0),
        ),
        reverse=True,
    )[:n_keep]


def select_representative_agents(
    ranked_agents: list[dict[str, Any]],
    agent_contains: str,
    per_bucket: int,
) -> list[dict[str, Any]]:
    """Pick representative examples for the target agent for each process."""
    if not ranked_agents:
        return []

    target = agent_contains or "PenaltyAwareDelayAgent"
    matched = filter_agents(ranked_agents, target)
    if not matched:
        matched = ranked_agents

    per_bucket = max(1, per_bucket)
    selected: dict[str, dict[str, Any]] = {}

    processes = sorted(
        {str(agent.get("process", "")) for agent in matched},
        key=lambda value: (value == "", value),
    )
    for process in processes:
        process_agents = [
            agent for agent in matched if str(agent.get("process", "")) == process
        ]
        process_agents = sorted(
            process_agents,
            key=lambda agent: float(agent.get("score") or -math.inf),
            reverse=True,
        )
        if not process_agents:
            continue

        def by_distance(index: int) -> list[dict[str, Any]]:
            indexed = list(enumerate(process_agents))
            return [
                agent
                for _, agent in sorted(
                    indexed,
                    key=lambda item: (abs(item[0] - index), item[0]),
                )
            ]

        middle = len(process_agents) // 2
        q3 = round((len(process_agents) - 1) * 0.25)
        q1 = round((len(process_agents) - 1) * 0.75)
        buckets: list[tuple[str, list[dict[str, Any]]]] = [
            ("good", process_agents),
            ("q3", by_distance(q3)),
            ("middle", by_distance(middle)),
            ("q1", by_distance(q1)),
            ("bad", list(reversed(process_agents))),
        ]
        label_prefix = f"@{process}" if process else "@?"
        for bucket, agents in buckets:
            n_added = 0
            for agent in agents:
                if str(agent["key"]) in selected:
                    continue
                copied = dict(agent)
                copied["bucket"] = f"{label_prefix} {bucket}"
                selected[str(agent["key"])] = copied
                n_added += 1
                if n_added >= per_bucket:
                    break
    return list(selected.values())


def summarize_contracts(data: pd.DataFrame) -> dict[str, Any]:
    if data.empty:
        return {"contracts": 0, "quantity": 0, "mean_unit_price": None, "rows": []}
    return {
        "contracts": int(len(data)),
        "quantity": safe_value(data["quantity"].sum()),
        "mean_unit_price": safe_value(data["unit_price"].mean()),
        "min_unit_price": safe_value(data["unit_price"].min()),
        "max_unit_price": safe_value(data["unit_price"].max()),
        "rows": records(
            data.loc[
                :,
                [
                    column
                    for column in (
                        "seller_name",
                        "buyer_name",
                        "partner",
                        "quantity",
                        "unit_price",
                        "price_range",
                        "product_name",
                        "signed_at",
                        "concluded_at",
                        "is_exogenous",
                    )
                    if column in data.columns
                ],
            ].sort_values(["is_exogenous", "partner", "unit_price"], ascending=[False, True, True])
        ),
    }


def world_market_summary(contracts: pd.DataFrame, world_name: str) -> pd.DataFrame:
    if contracts.empty:
        return pd.DataFrame()

    data = contracts.loc[contracts["world"] == world_name].copy()
    if data.empty or "delivery_time" not in data.columns:
        return pd.DataFrame()

    data["delivery_time"] = pd.to_numeric(data["delivery_time"], errors="coerce")
    data = data.dropna(subset=["delivery_time"]).copy()
    data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce").fillna(0)
    seller_supply = data.loc[
        (data["seller_name"] == "SELLER")
        & data["buyer_name"].map(lambda name: process_of(name) == "0")
    ]
    buyer_demand = data.loc[
        (data["buyer_name"] == "BUYER")
        & data["seller_name"].map(lambda name: process_of(name) == "1")
    ]
    agreed = data.loc[
        data["seller_name"].map(lambda name: process_of(name) == "0")
        & data["buyer_name"].map(lambda name: process_of(name) == "1")
    ]

    steps = sorted(
        set(seller_supply["delivery_time"].dropna().astype(int))
        | set(buyer_demand["delivery_time"].dropna().astype(int))
        | set(agreed["delivery_time"].dropna().astype(int))
    )
    rows = []
    for step in steps:
        seller_quantity = seller_supply.loc[
            seller_supply["delivery_time"].astype(int) == step, "quantity"
        ].sum()
        buyer_quantity = buyer_demand.loc[
            buyer_demand["delivery_time"].astype(int) == step, "quantity"
        ].sum()
        agreed_quantity = agreed.loc[
            agreed["delivery_time"].astype(int) == step, "quantity"
        ].sum()
        rows.append(
            {
                "step": step,
                "seller_target_quantity": safe_value(seller_quantity),
                "buyer_target_quantity": safe_value(buyer_quantity),
                "agreed_quantity": safe_value(agreed_quantity),
                "seller_minus_buyer": safe_value(seller_quantity - buyer_quantity),
                "unmet_buyer_quantity": safe_value(max(buyer_quantity - agreed_quantity, 0)),
                "unsold_seller_quantity": safe_value(max(seller_quantity - agreed_quantity, 0)),
            }
        )
    return pd.DataFrame(rows)


def agent_summary(
    key: str,
    history: pd.DataFrame,
    contracts: pd.DataFrame,
    actions: pd.DataFrame,
    flows: pd.DataFrame,
    max_events_per_step: int | None = DEFAULT_MAX_EVENTS_PER_STEP,
) -> dict[str, Any]:
    selected_history = history.loc[history["key"] == key].sort_values("step")
    if selected_history.empty:
        return {}
    world_name = selected_history["world"].iloc[0]
    agent_name = selected_history["agent"].iloc[0]
    market_summary = world_market_summary(contracts, world_name)
    selected_contracts = contracts.loc[
        (contracts["world"] == world_name)
        & (
            (contracts["seller_name"] == agent_name)
            | (contracts["buyer_name"] == agent_name)
        )
    ].copy()
    if "issues" in selected_contracts.columns:
        selected_contracts["price_range"] = selected_contracts["issues"].map(
            price_range_from_issues
        )
    else:
        selected_contracts["price_range"] = ""
    selected_contracts["role"] = selected_contracts.apply(
        lambda row: "seller" if row["seller_name"] == agent_name else "buyer",
        axis=1,
    )
    selected_contracts["partner"] = selected_contracts.apply(
        lambda row: row["buyer_name"] if row["seller_name"] == agent_name else row["seller_name"],
        axis=1,
    )
    selected_actions = actions.loc[
        (actions["world"] == world_name)
        & ((actions["sender"] == agent_name) | (actions["receiver"] == agent_name))
    ].copy()
    if "sim_step" not in selected_actions:
        selected_actions["sim_step"] = selected_actions.get("delivery_step", 0)
    selected_actions["partner"] = selected_actions.apply(
        lambda row: row["receiver"] if row["sender"] == agent_name else row["sender"],
        axis=1,
    )
    selected_actions["direction"] = selected_actions.apply(
        lambda row: "sent" if row["sender"] == agent_name else "received",
        axis=1,
    )
    fallback_role = "seller" if process_of(agent_name) == "0" else "buyer"
    selected_actions["role"] = selected_actions.apply(
        lambda row: "seller"
        if row.get("seller") == agent_name
        else "buyer"
        if row.get("buyer") == agent_name
        else fallback_role,
        axis=1,
    )
    selected_actions["accepted"] = selected_actions["state"].eq("agreement")
    selected_actions["failed_or_ended"] = selected_actions["state"].eq("ended")
    selected_flows = (
        flows.loc[
            (flows["world"] == world_name)
            & ((flows["buyer"] == agent_name) | (flows["seller"] == agent_name))
        ].copy()
        if not flows.empty
        else pd.DataFrame()
    )
    if not selected_flows.empty:
        selected_flows["partner"] = selected_flows.apply(
            lambda row: row["seller"] if row["buyer"] == agent_name else row["buyer"],
            axis=1,
        )
        selected_flows["direction"] = selected_flows.apply(
            lambda row: "sent"
            if row["sender"] == agent_name
            else "received"
            if row["receiver"] == agent_name
            else "",
            axis=1,
        )
        selected_flows["role"] = selected_flows.apply(
            lambda row: "seller" if row["seller"] == agent_name else "buyer",
            axis=1,
        )

    steps: dict[str, Any] = {}
    for step in sorted(selected_history["step"].dropna().astype(int).unique()):
        metrics = selected_history.loc[selected_history["step"] == step].iloc[-1].to_dict()
        market_rows = market_summary.loc[market_summary["step"] == step]
        market = market_rows.iloc[0].to_dict() if not market_rows.empty else {}
        day_contracts = selected_contracts.loc[selected_contracts["delivery_time"] == step].copy()
        buys = day_contracts.loc[day_contracts["buyer_name"] == agent_name].copy()
        sells = day_contracts.loc[day_contracts["seller_name"] == agent_name].copy()
        exo_buys = buys.loc[buys["seller_name"] == "SELLER"]
        exo_sells = sells.loc[sells["buyer_name"] == "BUYER"]
        negotiated_buys = buys.loc[buys["seller_name"] != "SELLER"]
        negotiated_sells = sells.loc[sells["buyer_name"] != "BUYER"]
        day_actions = selected_actions.loc[
            pd.to_numeric(selected_actions["sim_step"], errors="coerce").fillna(-1).astype(int)
            == step
        ].copy()
        day_flows = (
            selected_flows.loc[
                pd.to_numeric(selected_flows["sim_step"], errors="coerce")
                .fillna(-1)
                .astype(int)
                == step
            ].copy()
            if not selected_flows.empty
            else pd.DataFrame()
        )
        raw_action_count = len(day_actions)
        raw_flow_count = len(day_flows)
        if max_events_per_step is not None and max_events_per_step >= 0:
            if raw_action_count > max_events_per_step:
                day_actions = day_actions.head(max_events_per_step).copy()
            if raw_flow_count > max_events_per_step:
                day_flows = day_flows.head(max_events_per_step).copy()
        action_columns = [
            column
            for column in (
                "id",
                "partner",
                "role",
                "round",
                "mechanism_step",
                "relative_time",
                "time",
                "sender",
                "receiver",
                "direction",
                "state",
                "quantity",
                "delivery_step",
                "unit_price",
                "price_range",
                "neg_id",
                "has_agreement",
                "needed_sales0",
                "needed_sales1",
                "needed_supplies0",
                "needed_supplies1",
                "trading_price",
            )
            if column in day_actions.columns
        ]
        timeline_sort_columns = [
            column
            for column in ("round", "relative_time", "time", "id")
            if column in action_columns
        ]
        role_sort_columns = [
            column
            for column in ("partner", "neg_id", "round", "direction", "id")
            if column in action_columns
        ]
        sorted_day_actions = day_actions.loc[:, action_columns]
        flow_columns = [
            column
            for column in (
                "id",
                "partner",
                "role",
                "round",
                "mechanism_step",
                "relative_time",
                "time",
                "event",
                "direction",
                "sender",
                "receiver",
                "quantity",
                "delivery_step",
                "unit_price",
                "price_range",
                "n_acceptances",
                "first_process",
                "matches_agreement",
                "final_status",
                "neg_id",
                "neg_uuid",
            )
            if column in day_flows.columns
        ]
        flow_role_sort_columns = [
            column
            for column in ("partner", "neg_id", "round", "event", "id")
            if column in flow_columns
        ]
        flow_timeline_sort_columns = [
            column
            for column in ("round", "mechanism_step", "relative_time", "time", "neg_id", "partner", "id")
            if column in flow_columns
        ]
        sorted_day_flows = day_flows.loc[:, flow_columns] if flow_columns else pd.DataFrame()
        sorted_role_actions = sorted_day_actions.sort_values(
            role_sort_columns or action_columns[:1]
        )
        sorted_timeline_actions = sorted_day_actions.sort_values(
            timeline_sort_columns or action_columns[:1]
        )
        sorted_role_flows = sorted_day_flows.sort_values(
            flow_role_sort_columns or flow_columns[:1]
        )
        sorted_timeline_flows = sorted_day_flows.sort_values(
            flow_timeline_sort_columns or flow_columns[:1]
        )
        steps[str(step)] = {
            "metrics": {k: safe_value(v) for k, v in metrics.items()},
            "market": {k: safe_value(v) for k, v in market.items()},
            "seller": {
                "procurement": summarize_contracts(buys),
                "negotiated_sales": summarize_contracts(negotiated_sells),
                "sales_obligation": summarize_contracts(exo_sells),
            },
            "buyer": {
                "sales_obligation": summarize_contracts(exo_sells),
                "negotiated_buys": summarize_contracts(negotiated_buys),
                "supply_obligation": summarize_contracts(exo_buys),
            },
            "actions": limited_records(sorted_role_actions, max_events_per_step),
            "timeline": limited_records(sorted_timeline_actions, max_events_per_step),
            "flows": limited_records(sorted_role_flows, max_events_per_step),
            "flow_timeline": limited_records(sorted_timeline_flows, max_events_per_step),
            "truncated": {
                "actions": max(0, raw_action_count - max_events_per_step)
                if max_events_per_step is not None
                else 0,
                "flows": max(0, raw_flow_count - max_events_per_step)
                if max_events_per_step is not None
                else 0,
            },
        }

    partner_summary = (
        selected_contracts.groupby(["role", "partner", "is_exogenous"], as_index=False)
        .agg(
            contracts=("id", "count"),
            quantity=("quantity", "sum"),
            mean_unit_price=("unit_price", "mean"),
        )
        .sort_values(["role", "quantity"], ascending=[True, False])
        if not selected_contracts.empty
        else pd.DataFrame()
    )
    daily_summary = (
        selected_history.loc[
            :,
            [
                column
                for column in (
                    "step",
                    "score",
                    "balance",
                    "productivity",
                    "shortfall_quantity",
                    "shortfall_penalty",
                    "disposal_cost",
                    "inventory_penalized",
                )
                if column in selected_history.columns
            ],
        ]
        .sort_values("step")
    )
    return {
        "agent": agent_name,
        "world": world_name,
        "type": selected_history["type"].iloc[0],
        "process": selected_history["process"].iloc[0],
        "summary": {
            "daily_metrics": records(daily_summary),
            "market_supply_demand": records(market_summary),
            "partner_contracts": records(partner_summary),
        },
        "steps": steps,
    }


def build_payload(
    stage_path: Path,
    agent_contains: str = "",
    max_details: int | None = 20,
    light_agent_filter: bool = False,
    max_events_per_step: int | None = DEFAULT_MAX_EVENTS_PER_STEP,
) -> dict[str, Any]:
    ranking_stage_data = read_stage(
        stage_path,
        include_contracts=False,
        include_actions=False,
        include_flows=False,
    )
    history = agent_metric_rows(ranking_stage_data)
    ranking = final_ranking(history)
    ranked_agents = records(
        ranking.loc[
            :,
            [
                column
                for column in (
                    "key",
                    "world",
                    "agent",
                    "type",
                    "process",
                    "score",
                    "balance",
                    "avg_productivity",
                    "total_shortfall_penalty",
                    "total_disposal_cost",
                    "total_shortfall_quantity",
                )
                if column in ranking.columns
            ],
        ]
    )
    def top_bottom_agents(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if max_details is None:
            return agents
        n_top = math.ceil(max_details / 2)
        n_bottom = max_details // 2
        bottom_agents = agents[-n_bottom:] if n_bottom else []
        return list(
            {
                agent["key"]: agent
                for agent in agents[:n_top] + bottom_agents
            }.values()
        )

    if agent_contains:
        matched_agents = filter_agents(ranked_agents, agent_contains)
        detail_agents = (
            top_bottom_agents(matched_agents)
            if light_agent_filter
            else high_loss_half_agents(matched_agents)
        )
    elif max_details is None:
        detail_agents = ranked_agents
    else:
        detail_agents = top_bottom_agents(ranked_agents)

    detail_worlds = {str(agent["world"]) for agent in detail_agents}
    detail_stage_data = (
        read_stage(stage_path, world_names=detail_worlds)
        if detail_worlds
        else {"stage": stage_path, "worlds": []}
    )
    detail_history = agent_metric_rows(detail_stage_data)
    contracts = contract_rows(detail_stage_data)
    actions = action_rows(detail_stage_data)
    flows = flow_rows(detail_stage_data)
    details = {
        agent["key"]: agent_summary(
            agent["key"],
            detail_history,
            contracts,
            actions,
            flows,
            max_events_per_step,
        )
        for agent in detail_agents
    }
    payload = {
        "stage": str(stage_path),
        "ranking": ranked_agents,
        "agents": detail_agents,
        "details": details,
    }
    if detail_agents:
        payload["initial_agent_key"] = detail_agents[0]["key"]
    return payload


def build_representative_payload(
    stage_path: Path,
    agent_contains: str = "PenaltyAwareDelayAgent",
    per_bucket: int = 2,
    max_events_per_step: int | None = DEFAULT_MAX_EVENTS_PER_STEP,
) -> dict[str, Any]:
    """Build a compact report by reading details only for representative worlds."""
    ranking_stage_data = read_stage(
        stage_path,
        include_contracts=False,
        include_actions=False,
        include_flows=False,
    )
    history = agent_metric_rows(ranking_stage_data)
    ranking = final_ranking(history)
    ranked_agents = records(
        ranking.loc[
            :,
            [
                column
                for column in (
                    "key",
                    "world",
                    "agent",
                    "type",
                    "process",
                    "score",
                    "balance",
                    "avg_productivity",
                    "total_shortfall_penalty",
                    "total_disposal_cost",
                    "total_shortfall_quantity",
                )
                if column in ranking.columns
            ],
        ]
    )
    detail_agents = select_representative_agents(
        ranked_agents,
        agent_contains,
        per_bucket,
    )
    detail_worlds = {str(agent["world"]) for agent in detail_agents}
    detail_stage_data = read_stage(stage_path, world_names=detail_worlds)
    detail_history = agent_metric_rows(detail_stage_data)
    contracts = contract_rows(detail_stage_data)
    actions = action_rows(detail_stage_data)
    flows = flow_rows(detail_stage_data)
    details = {
        agent["key"]: agent_summary(
            agent["key"],
            detail_history,
            contracts,
            actions,
            flows,
            max_events_per_step,
        )
        for agent in detail_agents
    }
    payload = {
        "stage": str(stage_path),
        "ranking": records(ranking),
        "agents": detail_agents,
        "details": details,
        "representative": {
            "agent_filter": agent_contains,
            "per_bucket": per_bucket,
            "worlds": sorted(detail_worlds),
        },
    }
    if detail_agents:
        payload["initial_agent_key"] = detail_agents[0]["key"]
    return payload


def write_report(
    stage_path: Path | None = None,
    agent_contains: str = "",
    output_dir: Path | None = None,
    max_details: int | None = 20,
    light_agent_filter: bool = False,
    max_events_per_step: int | None = DEFAULT_MAX_EVENTS_PER_STEP,
) -> Path:
    stage_path = stage_path or latest_stage()
    output_dir = output_dir or stage_path / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_old_outputs(output_dir)

    payload = build_payload(
        stage_path,
        agent_contains,
        max_details,
        light_agent_filter,
        max_events_per_step,
    )
    (output_dir / "report.html").write_text(render_html(payload), encoding="utf-8")
    return output_dir


def write_representative_report(
    stage_path: Path | None = None,
    agent_contains: str = "PenaltyAwareDelayAgent",
    output_dir: Path | None = None,
    per_bucket: int = 2,
    max_events_per_step: int | None = DEFAULT_MAX_EVENTS_PER_STEP,
) -> Path:
    stage_path = stage_path or latest_stage()
    output_dir = output_dir or stage_path / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_old_outputs(output_dir)

    payload = build_representative_payload(
        stage_path,
        agent_contains,
        per_bucket,
        max_events_per_step,
    )
    (output_dir / "report.html").write_text(render_html(payload), encoding="utf-8")
    return output_dir


def render_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SCML Analysis Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --line: #d8e0ea;
      --text: #1f2933;
      --muted: #64748b;
      --accent: #0f766e;
      --soft: #e7f5f3;
      --warn: #b45309;
      --bad: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
      margin: 0;
    }}
    header {{
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 20px 28px;
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    h1 {{ font-size: 22px; margin: 0 0 6px; }}
    h2 {{ font-size: 18px; margin: 0 0 12px; }}
    h3 {{ font-size: 15px; margin: 18px 0 8px; }}
    main {{ padding: 22px 28px 40px; }}
    .meta {{ color: var(--muted); font-size: 13px; margin: 0; }}
    .grid {{
      display: grid;
      gap: 18px;
      grid-template-columns: minmax(320px, 0.95fr) minmax(420px, 1.4fr);
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      overflow: auto;
    }}
    .controls {{
      align-items: end;
      display: grid;
      gap: 12px;
      grid-template-columns: minmax(240px, 1fr) minmax(180px, 0.5fr);
      margin-bottom: 14px;
    }}
    label {{ color: var(--muted); display: grid; font-size: 12px; gap: 5px; }}
    select {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--text);
      font-size: 14px;
      padding: 8px 10px;
    }}
    table {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
    th {{
      background: #eef2f7;
      color: #334155;
      position: sticky;
      top: 0;
      text-align: left;
    }}
    th, td {{
      border-bottom: 1px solid #e5eaf0;
      padding: 7px 9px;
      white-space: nowrap;
    }}
    tr:nth-child(even) td {{ background: #fbfcfd; }}
    .rank-table tbody tr {{ cursor: pointer; }}
    .rank-table tbody tr:hover td {{ background: var(--soft); }}
    .agent-link {{
      background: transparent;
      border: 0;
      color: var(--accent);
      cursor: pointer;
      font: inherit;
      font-weight: 650;
      padding: 0;
      text-decoration: underline;
      text-underline-offset: 2px;
    }}
    .cards {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      margin-bottom: 12px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }}
    .card .name {{ color: var(--muted); font-size: 12px; }}
    .card .value {{ font-size: 20px; font-weight: 650; margin-top: 2px; }}
    .subtle {{ color: var(--muted); }}
    .recommendation {{
      background: #fff7ed;
      border: 1px solid #fed7aa;
      border-radius: 8px;
      color: #7c2d12;
      padding: 10px 12px;
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      margin: 14px 0 10px;
    }}
    .tab {{
      background: #eef2f7;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 11px;
    }}
    .tab.active {{
      background: var(--soft);
      border-color: #99d5ce;
      color: #075e58;
      font-weight: 650;
    }}
    .empty {{ color: var(--muted); padding: 8px 0; }}
    @media (max-width: 980px) {{
      .grid, .controls {{ grid-template-columns: 1fr; }}
      header {{ position: static; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>SCML シミュレーション分析</h1>
    <p class="meta" id="stage"></p>
  </header>
  <main>
    <div class="grid">
      <section class="panel">
        <h2>最終スコアランキング</h2>
        <div class="controls">
          <label>表示する world<select id="worldSelect"></select></label>
          <label>表示する type<select id="typeSelect"></select></label>
        </div>
        <div id="ranking"></div>
      </section>
      <section class="panel">
        <h2>エージェント追跡</h2>
        <div class="controls">
          <label>追跡するエージェント<select id="agentSelect"></select></label>
          <label>表示<select id="viewSelect"></select></label>
        </div>
        <div id="detail"></div>
      </section>
    </div>
  </main>
  <script id="payload" type="application/json">{data_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    const fmt = (value) => {{
      if (value === null || value === undefined || Number.isNaN(value)) return '-';
      if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(3);
      if (typeof value === 'boolean') return value ? 'true' : 'false';
      return String(value);
    }};
    const label = (agent) => `${{agent.agent}} | ${{agent.type}} | ${{agent.world}}`;
    const table = (rows, columns) => {{
      if (!rows || rows.length === 0) return '<p class="empty">データがありません。</p>';
      const head = '<tr>' + columns.map(c => `<th>${{c.label}}</th>`).join('') + '</tr>';
      const body = rows.map(row => '<tr>' + columns.map(c => `<td>${{fmt(row[c.key])}}</td>`).join('') + '</tr>').join('');
      return `<table><thead>${{head}}</thead><tbody>${{body}}</tbody></table>`;
    }};
    const metricCards = (metrics) => {{
      const keys = [
        ['score', 'score'],
        ['balance', 'balance'],
        ['productivity', 'productivity'],
        ['shortfall_penalty', 'shortfall penalty'],
        ['disposal_cost', 'disposal cost'],
        ['shortfall_quantity', 'shortfall qty'],
        ['inventory_penalized', 'inventory penalized'],
      ];
      return '<div class="cards">' + keys.filter(([key]) => key in metrics).map(([key, name]) =>
        `<div class="card"><div class="name">${{name}}</div><div class="value">${{fmt(metrics[key])}}</div></div>`
      ).join('') + '</div>';
    }};
    const recommendation = (metrics, roleData, actions) => {{
      const notes = [];
      if ((metrics.shortfall_penalty || 0) > 0) notes.push('不足ペナルティが出ています。売る約束に対して、仕入れまたは生産が足りていません。');
      if ((metrics.disposal_cost || 0) > 0) notes.push('廃棄コストが出ています。買いすぎ、または売り先不足を疑うとよさそうです。');
      if ((metrics.productivity || 0) < 0.7) notes.push('productivity が低めです。必要数量に合わせた契約成立率を上げる余地があります。');
      const ended = (actions || []).filter(a => a.state === 'ended' || a.final_status === 'failed').length;
      const agreed = (actions || []).filter(a => a.state === 'agreement' || a.event === 'agreement').length;
      if (ended > agreed) notes.push('終了した交渉が成立交渉より多めです。相手別に単価・数量の出し方を見直す価値があります。');
      if ((roleData?.buyer?.negotiated_buys?.quantity || 0) === 0 && (roleData?.seller?.negotiated_sales?.quantity || 0) === 0) notes.push('交渉による契約が少ない日です。提案数量や価格が厳しすぎる可能性があります。');
      if (notes.length === 0) notes.push('この step は大きな損失要因が見えにくいです。相手別の成立率と単価差を比較すると次の改善点が見つかりそうです。');
      return `<div class="recommendation"><strong>Codexおすすめ指標:</strong><br>${{notes.map(n => `・${{n}}`).join('<br>')}}</div>`;
    }};
    const renderRanking = () => {{
      const selectedWorld = document.getElementById('worldSelect')?.value || 'all';
      const selectedType = document.getElementById('typeSelect')?.value || 'all';
      const rows = (payload.ranking || []).filter(row =>
        (selectedWorld === 'all' || row.world === selectedWorld) &&
        (selectedType === 'all' || row.type === selectedType)
      );
      if (rows.length === 0) {{
        document.getElementById('ranking').innerHTML = '<p class="empty">データがありません。</p>';
        return;
      }}
      const headers = ['agent', 'type', '@', 'final score', 'balance', 'avg productivity', 'shortfall penalty total', 'disposal total'];
      const body = rows.map((row, index) => `
        <tr>
          <td>${{payload.details?.[row.key] ? `<button class="agent-link" data-index="${{index}}" type="button">${{fmt(row.agent)}}</button>` : fmt(row.agent)}}</td>
          <td>${{fmt(row.type)}}</td>
          <td>${{fmt(row.process)}}</td>
          <td>${{fmt(row.score)}}</td>
          <td>${{fmt(row.balance)}}</td>
          <td>${{fmt(row.avg_productivity)}}</td>
          <td>${{fmt(row.total_shortfall_penalty)}}</td>
          <td>${{fmt(row.total_disposal_cost)}}</td>
        </tr>
      `).join('');
      document.getElementById('ranking').innerHTML = `
        <table class="rank-table">
          <thead><tr>${{headers.map(header => `<th>${{header}}</th>`).join('')}}</tr></thead>
          <tbody>${{body}}</tbody>
        </table>
      `;
      document.querySelectorAll('.agent-link').forEach((button) => {{
        button.addEventListener('click', (event) => {{
          event.stopPropagation();
          document.getElementById('agentSelect').value = rows[Number(button.dataset.index)].key;
          refreshViews();
          renderDetail();
        }});
      }});
    }};
    const refreshViews = () => {{
      const key = document.getElementById('agentSelect').value;
      const detail = payload.details[key];
      const select = document.getElementById('viewSelect');
      const current = select.value;
      const steps = Object.keys(detail?.steps || {{}}).sort((a, b) => Number(a) - Number(b));
      select.innerHTML = '<option value="summary">summary</option>' + steps.map(step => `<option value="${{step}}">step ${{step}}</option>`).join('');
      select.value = [...steps, 'summary'].includes(current) ? current : 'summary';
    }};
    const summaryView = (detail) => {{
      const dailyCols = [
        {{key: 'step', label: 'step'}},
        {{key: 'score', label: 'score'}},
        {{key: 'balance', label: 'balance'}},
        {{key: 'productivity', label: 'productivity'}},
        {{key: 'shortfall_quantity', label: 'shortfall qty'}},
        {{key: 'shortfall_penalty', label: 'shortfall penalty'}},
        {{key: 'disposal_cost', label: 'disposal cost'}},
      ];
      const partnerCols = [
        {{key: 'role', label: 'role'}},
        {{key: 'partner', label: 'partner'}},
        {{key: 'is_exogenous', label: 'exogenous'}},
        {{key: 'contracts', label: 'contracts'}},
        {{key: 'quantity', label: 'quantity'}},
        {{key: 'mean_unit_price', label: 'mean unit price'}},
      ];
      const marketCols = [
        {{key: 'step', label: 'step'}},
        {{key: 'seller_target_quantity', label: '@0 sell target'}},
        {{key: 'buyer_target_quantity', label: '@1 buy target'}},
        {{key: 'agreed_quantity', label: '@0 -> @1 agreed'}},
        {{key: 'seller_minus_buyer', label: 'seller - buyer'}},
        {{key: 'unmet_buyer_quantity', label: 'buyer unmet'}},
        {{key: 'unsold_seller_quantity', label: 'seller unsold'}},
      ];
      return `
        <p class="subtle">${{detail.agent}} / ${{detail.type}} / process @${{detail.process}} / ${{detail.world}}</p>
        <h3>日ごとの主要指標</h3>${{table(detail.summary.daily_metrics, dailyCols)}}
        <h3>ワールド全体の需給関係</h3>${{table(detail.summary.market_supply_demand, marketCols)}}
        <h3>相手ごとの契約集計</h3>${{table(detail.summary.partner_contracts, partnerCols)}}
      `;
    }};
    const contractSummary = (title, summary) => `
      <div class="card">
        <div class="name">${{title}}</div>
        <div class="value">${{fmt(summary.quantity)}} 個</div>
        <div class="subtle">契約 ${{fmt(summary.contracts)}} / 平均単価 ${{fmt(summary.mean_unit_price)}}</div>
      </div>
    `;
    const activeRole = (detail, data) => {{
      const buyQuantity = data?.buyer?.negotiated_buys?.quantity || 0;
      const sellQuantity = data?.seller?.negotiated_sales?.quantity || 0;
      if (buyQuantity > 0 && sellQuantity === 0) return 'buyer';
      if (sellQuantity > 0 && buyQuantity === 0) return 'seller';
      if (buyQuantity > sellQuantity) return 'buyer';
      if (sellQuantity > buyQuantity) return 'seller';
      const roleRows = data.flows || data.actions || [];
      const buyerActions = roleRows.filter(row => row.role === 'buyer').length;
      const sellerActions = roleRows.filter(row => row.role === 'seller').length;
      if (buyerActions > sellerActions) return 'buyer';
      if (sellerActions > buyerActions) return 'seller';
      return String(detail.process) === '0' ? 'seller' : 'buyer';
    }};
    const roleActions = (actions, role) => {{
      return (actions || []).filter(row => !row.role || row.role === role);
    }};
    const marketCards = (market) => {{
      if (!market || Object.keys(market).length === 0) return '';
      return `
        <h3>ワールド全体の需給関係</h3>
        <div class="cards">
          <div class="card"><div class="name">@0 が @1 に売りたい個数</div><div class="value">${{fmt(market.seller_target_quantity)}}</div></div>
          <div class="card"><div class="name">@1 が @0 から買いたい個数</div><div class="value">${{fmt(market.buyer_target_quantity)}}</div></div>
          <div class="card"><div class="name">@0 -> @1 成立個数</div><div class="value">${{fmt(market.agreed_quantity)}}</div></div>
          <div class="card"><div class="name">売り手側 - 買い手側</div><div class="value">${{fmt(market.seller_minus_buyer)}}</div></div>
        </div>
      `;
    }};
    const stepView = (detail, step) => {{
      const data = detail.steps[step];
      const metrics = data.metrics || {{}};
      const role = activeRole(detail, data);
      const contractCols = [
        {{key: 'partner', label: 'partner'}},
        {{key: 'seller_name', label: 'seller'}},
        {{key: 'buyer_name', label: 'buyer'}},
        {{key: 'quantity', label: 'quantity'}},
        {{key: 'unit_price', label: 'unit price'}},
        {{key: 'price_range', label: 'price range'}},
        {{key: 'product_name', label: 'product'}},
        {{key: 'is_exogenous', label: 'exogenous'}},
      ];
      const flowCols = [
        {{key: 'partner', label: 'partner'}},
        {{key: 'round', label: 'round'}},
        {{key: 'mechanism_step', label: 'mechanism step'}},
        {{key: 'event', label: 'event'}},
        {{key: 'direction', label: 'sent/received'}},
        {{key: 'sender', label: 'sender'}},
        {{key: 'receiver', label: 'receiver'}},
        {{key: 'quantity', label: 'quantity'}},
        {{key: 'unit_price', label: 'unit price'}},
        {{key: 'price_range', label: 'price range'}},
        {{key: 'n_acceptances', label: 'acceptances'}},
        {{key: 'first_process', label: 'first mover @'}},
        {{key: 'matches_agreement', label: 'matches agreement'}},
        {{key: 'final_status', label: 'final status'}},
        {{key: 'neg_id', label: 'neg id'}},
      ];
      const flowTimelineCols = [
        {{key: 'round', label: 'round'}},
        {{key: 'mechanism_step', label: 'mechanism step'}},
        {{key: 'relative_time', label: 'relative time'}},
        {{key: 'partner', label: 'partner'}},
        {{key: 'event', label: 'event'}},
        {{key: 'direction', label: 'sent/received'}},
        {{key: 'sender', label: 'sender'}},
        {{key: 'receiver', label: 'receiver'}},
        {{key: 'quantity', label: 'quantity'}},
        {{key: 'unit_price', label: 'unit price'}},
        {{key: 'n_acceptances', label: 'acceptances'}},
        {{key: 'first_process', label: 'first mover @'}},
        {{key: 'matches_agreement', label: 'matches agreement'}},
        {{key: 'final_status', label: 'final status'}},
        {{key: 'neg_id', label: 'neg id'}},
      ];
      const roleLabel = role === 'seller' ? '売り手' : '買い手';
      const roleBlock = role === 'seller' ? `
        <div class="tabs"><span class="tab active">売り手として見る</span></div>
        <div class="cards">
          ${{contractSummary('仕入れ値と個数', data.seller.procurement)}}
          ${{contractSummary('成立した売値と個数', data.seller.negotiated_sales)}}
          ${{contractSummary('売るべき個数と値段', data.seller.sales_obligation)}}
        </div>
        <h3>売り手側の成立契約</h3>
        ${{table([...(data.seller.procurement.rows || []), ...(data.seller.negotiated_sales.rows || []), ...(data.seller.sales_obligation.rows || [])], contractCols)}}
      ` : `
        <div class="tabs"><span class="tab active">買い手として見る</span></div>
        <div class="cards">
          ${{contractSummary('売るべき個数と値段', data.buyer.sales_obligation)}}
          ${{contractSummary('成立した買値と個数', data.buyer.negotiated_buys)}}
          ${{contractSummary('外生仕入れ', data.buyer.supply_obligation)}}
        </div>
        <h3>買い手側の成立契約</h3>
        ${{table([...(data.buyer.sales_obligation.rows || []), ...(data.buyer.negotiated_buys.rows || []), ...(data.buyer.supply_obligation.rows || [])], contractCols)}}
      `;
      return `
        <p class="subtle">${{detail.agent}} / step ${{step}} / ${{detail.world}} / 表示ロール: ${{roleLabel}}</p>
        ${{metricCards(metrics)}}
        ${{marketCards(data.market)}}
        ${{recommendation(metrics, data, data.flows)}}
        ${{roleBlock}}
        <h3>${{roleLabel}}としての取引履歴（未成立・終了含む）</h3>
        ${{table(roleActions(data.flows, role), flowCols)}}
        <h3>1日の取引時系列</h3>
        ${{table(data.flow_timeline, flowTimelineCols)}}
      `;
    }};
    const renderDetail = () => {{
      const key = document.getElementById('agentSelect').value;
      const view = document.getElementById('viewSelect').value;
      const detail = payload.details[key];
      if (!detail) {{
        document.getElementById('detail').innerHTML = '<p class="empty">データがありません。</p>';
        return;
      }}
      document.getElementById('detail').innerHTML = view === 'summary' ? summaryView(detail) : stepView(detail, view);
    }};
    const init = () => {{
      document.getElementById('stage').textContent = `結果フォルダ: ${{payload.stage}}`;
      const worlds = [...new Set((payload.ranking || []).map(row => row.world))].sort();
      const types = [...new Set((payload.ranking || []).map(row => row.type))].sort();
      const worldSelect = document.getElementById('worldSelect');
      worldSelect.innerHTML = '<option value="all">all worlds</option>' + worlds.map(world => `<option value="${{world}}">${{world}}</option>`).join('');
      worldSelect.addEventListener('change', renderRanking);
      const typeSelect = document.getElementById('typeSelect');
      typeSelect.innerHTML = '<option value="all">all types</option>' + types.map(type => `<option value="${{type}}">${{type}}</option>`).join('');
      typeSelect.addEventListener('change', renderRanking);
      const agentSelect = document.getElementById('agentSelect');
      agentSelect.innerHTML = (payload.agents || []).map(agent => `<option value="${{agent.key}}">${{label(agent)}}</option>`).join('');
      if (payload.initial_agent_key) agentSelect.value = payload.initial_agent_key;
      agentSelect.addEventListener('change', () => {{ refreshViews(); renderDetail(); }});
      document.getElementById('viewSelect').addEventListener('change', renderDetail);
      renderRanking();
      refreshViews();
      renderDetail();
    }};
    init();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write a single interactive report.html for a SCML tournament stage. "
            "When stage_path is omitted, the newest tmp_tournament_test/*-stage-* "
            "folder is used."
        )
    )
    parser.add_argument("stage_path", nargs="?", type=Path)
    parser.add_argument(
        "--agent",
        nargs="*",
        default=[],
        help=(
            "Agent name/type filter. Matching type names are exact; matching "
            "agent ids are partial. Details include the high-loss half by default. "
            "Use '--agent light NAME' to include only the top/bottom matching agents."
        ),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--max-details",
        type=int,
        default=20,
        help="Number of top/bottom ranked agents to include in the detail pane. Ignored when --agent is used.",
    )
    parser.add_argument(
        "--all-details",
        action="store_true",
        help="Include details for every ranked agent. This can be slow for large stages.",
    )
    parser.add_argument(
        "--representative",
        action="store_true",
        help=(
            "Write a compact report for representative score-bucket worlds "
            "of the selected agent instead of reading every detailed history."
        ),
    )
    parser.add_argument(
        "--representative-per-bucket",
        type=int,
        default=2,
        help="Number of examples to include for each representative score bucket.",
    )
    parser.add_argument(
        "--max-events-per-step",
        type=int,
        default=DEFAULT_MAX_EVENTS_PER_STEP,
        help=(
            "Maximum negotiation/action rows embedded per agent per day in detail "
            "tables. Use -1 to embed all rows."
        ),
    )
    args = parser.parse_args()

    stage_path = args.stage_path or latest_stage()
    print(f"Analyzing {stage_path}")
    max_details = None if args.all_details else args.max_details
    agent_args = args.agent or []
    light_agent_filter = bool(agent_args and agent_args[0] == "light")
    agent_contains = " ".join(agent_args[1:] if light_agent_filter else agent_args)
    max_events_per_step = (
        None if args.max_events_per_step is not None and args.max_events_per_step < 0
        else args.max_events_per_step
    )
    if args.representative:
        output_dir = write_representative_report(
            stage_path,
            agent_contains or "PenaltyAwareDelayAgent",
            args.out,
            args.representative_per_bucket,
            max_events_per_step,
        )
    else:
        output_dir = write_report(
            stage_path,
            agent_contains,
            args.out,
            max_details,
            light_agent_filter,
            max_events_per_step,
        )
    print(f"Wrote {output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
