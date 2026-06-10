#!/usr/bin/env python
"""
Penalty-aware OneShot negotiation agent.

The agent evaluates offers by their effect on the estimated daily profit,
including disposal and shortfall penalties.  It delays concession when there
are few partners, but becomes much more willing to close good deals near the
end of a negotiation.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any

from negmas import Contract, Outcome, SAOResponse, SAOState, ResponseType
from scml.common import distribute
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from scml.oneshot.agents.rand import SyncRandomOneShotAgent


class MyHalfOfferHistoryAgent(SyncRandomOneShotAgent):
    """
    初手で必要量の半分を出し、その成功履歴で次回以降の送信人数を調整する。
    """

    def init(self):
        super().init()
        partners = list(
            dict.fromkeys(
                list(self.awi.my_suppliers) + list(self.awi.my_consumers)
            )
        )
        self._half_offer_history = {partner: [] for partner in partners}
        self._first_offers = {}
        self._half_logic_used_partners = set()
        self._received_offer_counts = {partner: 0 for partner in partners}
        self._shortfall_history = []
        self._day_input_contracts = 0
        self._day_output_contracts = 0
        self._day_lines = self._max_lines()
        self._day_tracking_started = False

    def _ensure_partner(self, partner):
        if partner is None:
            return
        self._half_offer_history.setdefault(partner, [])

    def _partner_from_contract(self, contract):
        try:
            for partner in contract.partners:
                if partner != self.id:
                    return partner
        except Exception:
            pass
        return None

    def _contract_offer(self, contract):
        try:
            agreement = contract.agreement
            if isinstance(agreement, dict):
                quantity = agreement.get(QUANTITY, agreement.get("quantity"))
                delivery_step = agreement.get(TIME, agreement.get("time"))
                unit_price = agreement.get(UNIT_PRICE, agreement.get("unit_price"))
            else:
                quantity = agreement[QUANTITY]
                delivery_step = agreement[TIME]
                unit_price = agreement[UNIT_PRICE]
            return (quantity, delivery_step, unit_price)
        except Exception:
            return None

    def _side_issues(self, all_partners):
        if all_partners == self.awi.my_consumers:
            return self.awi.current_output_issues
        return self.awi.current_input_issues

    def _best_price(self, all_partners):
        issues = self._side_issues(all_partners)
        if all_partners == self.awi.my_consumers:
            return issues[UNIT_PRICE].max_value
        return issues[UNIT_PRICE].min_value

    def _is_seller_to(self, partner):
        return partner in self.awi.my_consumers

    def _trading_price(self, output: bool) -> float:
        try:
            product = self.awi.my_output_product if output else self.awi.my_input_product
            prices = self.awi.trading_prices
            if isinstance(prices, dict):
                return float(prices.get(product, prices.get(str(product), 1.0)))
            return float(prices[product])
        except Exception:
            try:
                issues = self.awi.current_output_issues if output else self.awi.current_input_issues
                return float((issues[UNIT_PRICE].min_value + issues[UNIT_PRICE].max_value) / 2)
            except Exception:
                return 1.0

    def _production_cost(self) -> float:
        try:
            return float(self.awi.profile.cost)
        except Exception:
            return 0.0

    def _max_lines(self) -> int:
        return max(0, int(getattr(self.awi, "n_lines", getattr(self.awi, "max_n_lines", 0)) or 0))

    def _round(self, states) -> int:
        values = []
        for state in states.values():
            for name in ("step", "current_offer_index"):
                value = getattr(state, name, None)
                if isinstance(value, int):
                    values.append(value + 1)
                    break
        return max(values, default=0)

    def _emergency_round_reached(self, states) -> bool:
        if self._round(states) >= 19:
            return True

        for partner, state in states.items():
            for name in ("time", "t"):
                value = getattr(state, name, None)
                if value is None:
                    continue
                try:
                    if float(value) > 0.9:
                        return True
                except Exception:
                    pass

            relative_time = getattr(state, "relative_time", None)
            if relative_time is None:
                continue
            try:
                relative_time = float(relative_time)
            except Exception:
                continue

            n_steps = getattr(state, "n_steps", None)
            if n_steps is None:
                try:
                    nmi = self.get_nmi(partner)
                    n_steps = getattr(nmi, "n_steps", None)
                except Exception:
                    n_steps = None
            try:
                if n_steps and relative_time * float(n_steps) >= 19:
                    return True
            except Exception:
                pass

        return False

    def _estimate_offer_set_profit(self, offers):
        total_input = total_output = 0
        buy_cost = sell_revenue = 0.0
        for partner, offer in offers.items():
            quantity = max(0, int(offer[QUANTITY]))
            unit_price = float(offer[UNIT_PRICE])
            if self._is_seller_to(partner):
                total_output += quantity
                sell_revenue += quantity * unit_price
            else:
                total_input += quantity
                buy_cost += quantity * unit_price

        produced = min(total_input, total_output, self._max_lines())
        excess_quantity = max(0, total_input - produced)
        shortfall_quantity = max(0, total_output - produced)
        disposal = float(getattr(self.awi, "current_disposal_cost", 0.0) or 0.0)
        shortfall = float(getattr(self.awi, "current_shortfall_penalty", 0.0) or 0.0)

        return (
            sell_revenue
            - buy_cost
            - self._production_cost() * produced
            - disposal * self._trading_price(output=False) * excess_quantity
            - shortfall * self._trading_price(output=True) * shortfall_quantity
        )

    def _best_profit_offer_set(self, current_offers):
        partners = list(current_offers)
        if not partners:
            return set()

        best_partners = set()
        best_profit = -math.inf
        if len(partners) <= 12:
            subsets = itertools.chain.from_iterable(
                itertools.combinations(partners, size)
                for size in range(1, len(partners) + 1)
            )
        else:
            ranked = sorted(
                partners,
                key=lambda partner: self._estimate_offer_set_profit(
                    {partner: current_offers[partner]}
                ),
                reverse=True,
            )
            subsets = (tuple(ranked[:size]) for size in range(1, min(len(ranked), 12) + 1))

        for subset in subsets:
            subset_offers = {partner: current_offers[partner] for partner in subset}
            profit = self._estimate_offer_set_profit(subset_offers)
            quantity = sum(int(current_offers[partner][QUANTITY]) for partner in subset)
            if profit > best_profit or (
                math.isclose(profit, best_profit) and quantity > 0 and not best_partners
            ):
                best_profit = profit
                best_partners = set(subset)
        return best_partners

    def _emergency_accept_responses(self, offers, states):
        current_offers = {
            partner: offer
            for partner, offer in offers.items()
            if offer is not None and offer[TIME] == self.awi.current_step
        }
        best_partners = self._best_profit_offer_set(current_offers)
        unneeded_response = (
            SAOResponse(ResponseType.END_NEGOTIATION, None)
            if not self.awi.allow_zero_quantity
            else SAOResponse(ResponseType.REJECT_OFFER, (0, self.awi.current_step, 0))
        )
        return {
            partner: SAOResponse(ResponseType.ACCEPT_OFFER, offers[partner])
            if partner in best_partners
            else unneeded_response
            for partner in offers
        }

    def _half_quantity(self, needs, all_partners):
        issues = self._side_issues(all_partners)
        quantity = math.ceil(needs / 2)
        return max(
            issues[QUANTITY].min_value,
            min(issues[QUANTITY].max_value, quantity),
        )

    def _half_quantities(self, needs, all_partners, n_offers):
        issues = self._side_issues(all_partners)
        qmin = issues[QUANTITY].min_value
        qmax = issues[QUANTITY].max_value
        high = math.ceil(needs / 2)
        low = math.floor(needs / 2)
        return [
            max(qmin, min(qmax, high if index % 2 == 0 else low))
            for index in range(n_offers)
        ]

    def _equal_dist_quantities(self, needs, all_partners, partners):
        if not partners:
            return {}

        issues = self._side_issues(all_partners)
        qmin = int(issues[QUANTITY].min_value)
        qmax = int(issues[QUANTITY].max_value)
        target = max(0, min(int(needs), qmax * len(partners)))
        if target <= 0:
            return {partner: 0 for partner in partners}

        quantities = distribute(
            target,
            len(partners),
            mx=qmax,
            equal=True,
            allow_zero=self.awi.allow_zero_quantity,
        )
        return {
            partner: max(qmin, min(qmax, int(quantity))) if quantity > 0 else 0
            for partner, quantity in zip(partners, quantities, strict=False)
        }

    def _offer_based_quantities(self, needs, all_partners, partners, current_offers):
        """
        Re-distribute needs over already-tested partners using their last offers.

        A partner that just offered a larger quantity receives a larger share,
        but all quantities stay within the negotiation issue bounds.
        """
        if not partners:
            return {}

        issues = self._side_issues(all_partners)
        qmax = int(issues[QUANTITY].max_value)
        target = max(0, min(int(needs), qmax * len(partners)))
        if target <= 0:
            return {partner: 0 for partner in partners}

        offered = {
            partner: max(0, int(current_offers[partner][QUANTITY]))
            for partner in partners
        }
        total_offered = sum(offered.values())
        if total_offered <= 0:
            quantities = distribute(
                target,
                len(partners),
                mx=qmax,
                equal=True,
                allow_zero=self.awi.allow_zero_quantity,
            )
            return dict(zip(partners, quantities, strict=False))

        ideals = {
            partner: target * offered[partner] / total_offered
            for partner in partners
        }
        quantities = {
            partner: min(qmax, int(math.floor(ideal)))
            for partner, ideal in ideals.items()
        }
        remaining = target - sum(quantities.values())
        order = sorted(
            partners,
            key=lambda partner: (ideals[partner] - math.floor(ideals[partner]), offered[partner]),
            reverse=True,
        )

        while remaining > 0:
            changed = False
            for partner in order:
                if remaining <= 0:
                    break
                if quantities[partner] >= qmax:
                    continue
                quantities[partner] += 1
                remaining -= 1
                changed = True
            if not changed:
                break

        return quantities

    def _partner_success_rate(self, partner):
        history = self._half_offer_history.get(partner, [])
        if not history:
            return 0.5
        return sum(history) / len(history)

    def _overall_success_rate(self):
        records = [
            result
            for partner in self._half_offer_history
            for result in self._half_offer_history.get(partner, [])
        ]
        if not records:
            return 0.5
        return sum(records) / len(records)

    def _select_first_offer_partners(self, partners):
        if not partners:
            return []

        overall_success_rate = self._overall_success_rate()
        if overall_success_rate <= 0:
            n_selected = len(partners)
        else:
            n_selected = math.ceil(2 / overall_success_rate)
        n_selected = max(1, min(len(partners), n_selected))

        return sorted(
            partners,
            key=lambda partner: (
                self._partner_success_rate(partner),
                len(self._half_offer_history.get(partner, [])),
            ),
            reverse=True,
        )[:n_selected]

    def _record_previous_day_shortfall(self):
        if not self._day_tracking_started:
            return

        produced = min(
            max(0, int(self._day_input_contracts)),
            max(0, int(self._day_output_contracts)),
            max(0, int(self._day_lines)),
        )
        shortfall = max(0, int(self._day_output_contracts) - produced)
        self._shortfall_history.append(shortfall)

    def _average_shortfall(self):
        if not self._shortfall_history:
            return 0.0
        return sum(self._shortfall_history) / len(self._shortfall_history)

    def _use_equal_dist_first_offer(self):
        return len(self._shortfall_history) >= 5 and self._average_shortfall() >= 2.5

    def _active_partners(self, all_partners):
        return [partner for partner in all_partners if partner in self.negotiators]

    def _insurance_accept_partners(self, partners, current_offers, remaining_needs):
        if remaining_needs <= 0:
            return []

        lower = 0.7 * remaining_needs
        upper = 1.1 * remaining_needs
        best_partners = []
        best_diff = float("inf")
        for size in range(1, len(partners) + 1):
            for subset in itertools.combinations(partners, size):
                quantity = sum(current_offers[partner][QUANTITY] for partner in subset)
                if quantity < lower or quantity > upper:
                    continue
                diff = abs(quantity - remaining_needs)
                if diff < best_diff:
                    best_diff = diff
                    best_partners = list(subset)
        return best_partners

    def before_step(self):
        self._record_previous_day_shortfall()
        for partner in self._first_offers:
            self._ensure_partner(partner)
            self._half_offer_history[partner].append(False)
        super().before_step()
        self._first_offers = {}
        self._half_logic_used_partners = set()
        self._day_input_contracts = 0
        self._day_output_contracts = 0
        self._day_lines = self._max_lines()
        self._day_tracking_started = True
        self._received_offer_counts = {
            partner: 0
            for partner in dict.fromkeys(
                list(self.awi.my_suppliers) + list(self.awi.my_consumers)
            )
        }

    def first_proposals(self):
        step = self.awi.current_step
        proposals = {}

        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            partners = [partner for partner in all_partners if partner in self.negotiators]
            if not partners:
                continue

            if needs <= 0:
                proposals.update({partner: None for partner in partners})
                continue

            if self._use_equal_dist_first_offer():
                # 平均 shortfall が大きい間は、初手から相手全体に薄く広く配る。
                selected_quantities = self._equal_dist_quantities(
                    needs,
                    all_partners,
                    partners,
                )
            else:
                selected = set(self._select_first_offer_partners(partners))
                selected_in_offer_order = [
                    partner for partner in partners if partner in selected
                ]
                selected_quantities = dict(
                    zip(
                        selected_in_offer_order,
                        self._half_quantities(
                            needs,
                            all_partners,
                            len(selected_in_offer_order),
                        ),
                        strict=False,
                    )
                )
            price = self._best_price(all_partners)

            for partner in partners:
                if partner in selected_quantities and selected_quantities[partner] > 0:
                    offer = (selected_quantities[partner], step, price)
                    proposals[partner] = offer
                    self._first_offers[partner] = offer
                    self._half_logic_used_partners.add(partner)
                else:
                    proposals[partner] = None

        return proposals

    def counter_all(self, offers, states):
        current_offers = {
            partner: offer
            for partner, offer in offers.items()
            if offer is not None and offer[TIME] == self.awi.current_step
        }
        first_received_partners = {
            partner
            for partner in current_offers
            if self._received_offer_counts.get(partner, 0) == 0
        }
        for partner in current_offers:
            self._received_offer_counts[partner] = (
                self._received_offer_counts.get(partner, 0) + 1
            )
        if self._emergency_round_reached(states):
            return self._emergency_accept_responses(offers, states)

        responses = super().counter_all(offers, states)

        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            partners = [
                partner
                for partner in all_partners
                if partner in current_offers
            ]
            if not partners or needs <= 0:
                continue

            price = self._best_price(all_partners)
            accepted_partners = [
                partner
                for partner in partners
                if responses.get(partner) is not None
                and responses[partner].response == ResponseType.ACCEPT_OFFER
            ]
            accepted_quantity = sum(current_offers[partner][QUANTITY] for partner in accepted_partners)
            remaining_needs = max(0, needs - accepted_quantity)
            if remaining_needs <= 0:
                # The superclass already selected enough offers for this side.
                # Keep its ACCEPT/END decisions and do not add extra counters.
                continue
            active_partners = self._active_partners(all_partners)
            if len(active_partners) <= 2:
                # With few live negotiations, accept a near-enough offer as insurance.
                insurance_partners = self._insurance_accept_partners(
                    [
                        partner
                        for partner in partners
                        if partner not in accepted_partners
                    ],
                    current_offers,
                    remaining_needs,
                )
                if insurance_partners:
                    for partner in insurance_partners:
                        responses[partner] = SAOResponse(
                            ResponseType.ACCEPT_OFFER,
                            current_offers[partner],
                        )
                    accepted_partners.extend(insurance_partners)
                    accepted_quantity += sum(
                        current_offers[partner][QUANTITY]
                        for partner in insurance_partners
                    )
                    remaining_needs = max(0, needs - accepted_quantity)
                    if remaining_needs <= 0:
                        continue
            half_partners = [
                partner
                for partner in partners
                if (
                    partner not in self._half_logic_used_partners
                    or partner in first_received_partners
                )
                and partner not in accepted_partners
            ]
            half_quantities = dict(
                zip(
                    half_partners,
                    self._half_quantities(remaining_needs, all_partners, len(half_partners)),
                    strict=False,
                )
            )
            for partner, quantity in half_quantities.items():
                offer = (quantity, self.awi.current_step, price)
                responses[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)
                self._first_offers[partner] = offer
                self._half_logic_used_partners.add(partner)

            offer_based_partners = [
                partner
                for partner in partners
                if partner not in half_quantities
                and partner in self._half_logic_used_partners
                and partner not in accepted_partners
            ]
            target_needs = max(0, remaining_needs - sum(half_quantities.values()))
            if not half_quantities and not accepted_partners:
                target_needs = needs
            offer_based_quantities = self._offer_based_quantities(
                target_needs,
                all_partners,
                offer_based_partners,
                current_offers,
            )
            for partner, quantity in offer_based_quantities.items():
                if quantity <= 0 and not self.awi.allow_zero_quantity:
                    responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                    continue
                offer = (quantity, self.awi.current_step, price)
                responses[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)
                self._first_offers[partner] = offer

        return responses

    def on_negotiation_success(self, contract, mechanism):
        super().on_negotiation_success(contract, mechanism)
        partner = self._partner_from_contract(contract)
        if partner is None:
            return

        agreement = self._contract_offer(contract)
        self._ensure_partner(partner)
        if agreement is not None:
            try:
                quantity = max(0, int(agreement[QUANTITY]))
                delivery_step = agreement[TIME]
            except Exception:
                quantity = 0
                delivery_step = None
            if quantity > 0 and delivery_step == self.awi.current_step:
                if self._is_seller_to(partner):
                    self._day_output_contracts += quantity
                else:
                    self._day_input_contracts += quantity
        if partner in self._first_offers:
            self._half_offer_history[partner].append(
                agreement == self._first_offers[partner]
            )
            self._first_offers.pop(partner, None)

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        super().on_negotiation_failure(partners, annotation, mechanism, state)
        try:
            partner = next(partner for partner in partners if partner != self.id)
        except Exception:
            return

        self._ensure_partner(partner)
        if partner in self._first_offers:
            self._half_offer_history[partner].append(False)
            self._first_offers.pop(partner, None)

from .bayesian_agent import BayesianAgent
from .bayesian_agent2 import BayesianAgent2
from .rdvo_agent import RDVOOneShotAgent, EINearestNeedOneShotAgent



MyAgent = BayesianAgent

if __name__ == "__main__":
    import sys

    from scml_agents import get_agents
    from .helpers.runner import run

    winners = [
        get_agents(y, track="oneshot", winners_only=True, as_class=True)[0]
        for y in (2025, 2024, 2023)
    ]

    run(
        [MyAgent, *winners],
        sys.argv[1] if len(sys.argv) > 1 else "oneshot",
    )
