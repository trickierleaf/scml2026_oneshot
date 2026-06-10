from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any

from negmas import Contract, Outcome, ResponseType, SAOResponse
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from scml.oneshot.agents.rand import SyncRandomOneShotAgent


class RDVOOneShotAgent(SyncRandomOneShotAgent):
    """
    Residual Demand Value Optimization for SCML OneShot.

    The agent chooses offer quantities by maximizing a one-step value estimate:

        E_t = s * min(R_t, M_t) - c * M_t

    plus a small continuation value for the remaining negotiation rounds.  For
    each partner i, expected quantity is estimated as:

        e_i = p_i q_i + (1 - p_i) r_i mu_i

    where p_i is our offer acceptance rate, r_i is the counter-offer rate after
    our offer fails, and mu_i is the average quantity in those counter-offers.
    """

    def __init__(
        self,
        *args,
        max_round: int = 20,
        candidate_count: int = 6,
        future_weight: float = 0.2,
        price_concession: float = 0.20,
        max_total_offer_multiplier: float = 0.0,
        overoffer_risk_penalty: float = 0.15,
        extreme_overoffer_multiplier: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_round = max_round
        self.candidate_count = max(3, int(candidate_count))
        self.future_weight = max(0.0, float(future_weight))
        self.price_concession = max(0.0, min(1.0, float(price_concession)))
        self.max_total_offer_multiplier = max(0.0, float(max_total_offer_multiplier))
        self.overoffer_risk_penalty = max(0.0, float(overoffer_risk_penalty))
        self.extreme_overoffer_multiplier = max(1.0, float(extreme_overoffer_multiplier))

    def init(self):
        super().init()
        self._offer_trials = defaultdict(int)
        self._offer_accepts = defaultdict(int)
        self._reoffer_events = defaultdict(int)
        self._counter_offer_quantities = defaultdict(list)
        self._sent_offer_history = defaultdict(list)
        self.accepted_contracts: list[tuple[str, Outcome]] = []
        for partner in self._all_partners():
            self._ensure_partner(partner)

    def before_step(self):
        super().before_step()
        self.accepted_contracts = []
        for partner in self._all_partners():
            self._ensure_partner(partner)

    # ------------------------------------------------------------------
    # Public strategy hooks
    # ------------------------------------------------------------------

    def first_proposals(self):
        proposals = {}
        for needs, all_partners in (
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ):
            partners = [partner for partner in all_partners if partner in self.negotiators]
            if not partners:
                continue
            proposals.update(self._rdvo_side_proposals(int(needs), partners, t=0.0))
        self._record_sent_offers(proposals, states=None)
        return proposals

    def counter_all(self, offers, states):
        current_offers = {
            partner: offer
            for partner, offer in offers.items()
            if offer is not None and len(offer) > UNIT_PRICE and offer[TIME] == self.awi.current_step
        }
        self._observe_incoming_offers(current_offers, states)

        responses = {}
        t = self._relative_time(states)
        for needs, all_partners in (
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ):
            partners = [partner for partner in all_partners if partner in current_offers]
            if not partners:
                continue
            if needs <= 0:
                for partner in partners:
                    responses[partner] = self._unneeded_response()
                continue

            accepted_partners = self._rdvo_accept_subset(
                int(needs),
                partners,
                current_offers,
                t,
            )
            accepted_quantity = sum(
                int(current_offers[partner][QUANTITY])
                for partner in accepted_partners
            )
            for partner in accepted_partners:
                responses[partner] = SAOResponse(
                    ResponseType.ACCEPT_OFFER,
                    current_offers[partner],
                )

            remaining_partners = [
                partner for partner in partners if partner not in accepted_partners
            ]
            remaining_needs = max(0, int(needs) - accepted_quantity)
            if remaining_needs <= 0:
                for partner in remaining_partners:
                    responses[partner] = self._unneeded_response()
                continue

            counter_proposals = self._rdvo_side_proposals(
                remaining_needs,
                remaining_partners,
                t=t,
            )
            for partner in remaining_partners:
                offer = counter_proposals.get(partner)
                if offer is None:
                    responses[partner] = self._unneeded_response()
                else:
                    responses[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)

        self._record_sent_offers(
            {
                partner: response.outcome
                for partner, response in responses.items()
                if response is not None
                and response.response == ResponseType.REJECT_OFFER
                and response.outcome is not None
            },
            states=states,
        )
        return responses

    # ------------------------------------------------------------------
    # RDVO optimization
    # ------------------------------------------------------------------

    def _rdvo_side_proposals(self, needs: int, partners: list[str], t: float):
        if not partners:
            return {}
        if needs <= 0:
            return {partner: None for partner in partners}

        candidate_lists = [
            self._quantity_candidates(needs, partner)
            for partner in partners
        ]
        best_value = -float("inf")
        best_quantities = [0] * len(partners)
        total_offer_cap = None
        if self.max_total_offer_multiplier > 0:
            total_offer_cap = math.ceil(
                max(0, int(needs)) * self.max_total_offer_multiplier
            )

        for quantities in itertools.product(*candidate_lists):
            offered_total = sum(max(0, int(quantity)) for quantity in quantities)
            if total_offer_cap is not None and offered_total > total_offer_cap:
                continue
            expected_units = []
            for partner, quantity in zip(partners, quantities, strict=False):
                price = self._offer_price(partner, t)
                expected_units.append(
                    self._expected_units(partner, int(quantity), price)
                )
            value = self._rdvo_value(
                needs,
                partners,
                quantities,
                expected_units,
                t,
                offered_total=offered_total,
            )
            if value > best_value:
                best_value = value
                best_quantities = list(quantities)

        proposals = {}
        for partner, quantity in zip(partners, best_quantities, strict=False):
            quantity = int(quantity)
            if quantity <= 0 and not self.awi.allow_zero_quantity:
                proposals[partner] = None
                continue
            quantity = self._clamp_quantity(partner, quantity)
            proposals[partner] = (
                quantity,
                self.awi.current_step,
                self._offer_price(partner, t),
            )
        return proposals

    def _rdvo_accept_subset(self, needs: int, partners: list[str], offers, t: float):
        baseline = self._future_value(needs, partners, t)
        best_value = baseline
        best_partners = set()

        for subset in self._powerset(partners):
            if not subset:
                continue
            quantities = [int(offers[partner][QUANTITY]) for partner in subset]
            expected_units = [float(quantity) for quantity in quantities]
            value = self._rdvo_value(
                needs,
                list(subset),
                quantities,
                expected_units,
                t,
                prices={partner: float(offers[partner][UNIT_PRICE]) for partner in subset},
            )
            if value > best_value:
                best_value = value
                best_partners = set(subset)
        return best_partners

    def _rdvo_value(
        self,
        needs: int,
        partners: list[str],
        quantities,
        expected_units,
        t: float,
        prices: dict[str, float] | None = None,
        offered_total: float | None = None,
    ) -> float:
        expected_total = max(0.0, sum(expected_units))
        if expected_total <= 0:
            return self._future_value(needs, partners, t)

        unit_cost = self._side_unit_cost(partners, prices)
        immediate = self._profit_delta_for_side(partners, expected_total, prices)
        immediate -= self._overoffer_risk_cost(
            needs,
            offered_total,
            expected_total,
            unit_cost,
        )
        remaining = max(0.0, float(needs) - expected_total)
        return immediate + self._future_value(remaining, partners, t)

    def _overoffer_risk_cost(
        self,
        needs: int,
        offered_total: float | None,
        expected_total: float,
        unit_cost: float,
    ) -> float:
        if (
            offered_total is None
            or offered_total <= needs
            or self.overoffer_risk_penalty <= 0
        ):
            return 0.0

        offered_total = max(0.0, float(offered_total))
        average_success = max(0.0, min(1.0, expected_total / max(1.0, offered_total)))
        soft_over = max(0.0, offered_total - float(needs))
        extreme_start = float(needs) * self.extreme_overoffer_multiplier
        extreme_over = max(0.0, offered_total - extreme_start)
        risk_units = soft_over * average_success + extreme_over
        return self.overoffer_risk_penalty * max(0.0, unit_cost) * risk_units

    def _future_value(self, remaining_needs: float, partners: list[str], t: float) -> float:
        return 0.0

    # ------------------------------------------------------------------
    # Probability and quantity estimates
    # ------------------------------------------------------------------

    def _expected_units(self, partner, quantity: int, price: float) -> float:
        if quantity <= 0:
            return 0.0
        p = self._accept_probability(partner, quantity, price)
        r = self._reoffer_probability(partner)
        mu = self._mean_reoffer_quantity(partner)
        return p * quantity + (1.0 - p) * r * mu

    def _accept_probability(self, partner, quantity: int, price: float) -> float:
        trials = self._offer_trials[partner]
        accepts = self._offer_accepts[partner]
        base = (accepts + 1.0) / (trials + 2.0)
        return max(0.02, min(0.98, base))

    def _reoffer_probability(self, partner) -> float:
        rejected = max(0, self._offer_trials[partner] - self._offer_accepts[partner])
        return (self._reoffer_events[partner] + 1.0) / (rejected + 2.0)

    def _mean_reoffer_quantity(self, partner) -> float:
        quantities = self._counter_offer_quantities.get(partner, [])
        if quantities:
            return sum(quantities[-20:]) / len(quantities[-20:])
        return 2.0

    def _quantity_candidates(self, needs: int, partner) -> list[int]:
        qmax = max(1, int(self._issues_for(partner)[QUANTITY].max_value))
        upper = max(0, min(qmax, max(needs, math.ceil(needs * 1.2))))
        raw = {
            1,
            min(upper, math.ceil(needs * 0.25)),
            min(upper, math.ceil(needs * 0.50)),
            min(upper, math.ceil(needs * 0.80)),
            min(upper, needs),
        }
        if self.awi.allow_zero_quantity:
            raw.add(0)
        values = sorted(value for value in raw if 0 <= value <= qmax)
        if len(values) > self.candidate_count:
            keep = {values[0], values[-1], min(values, key=lambda v: abs(v - needs))}
            for value in values:
                keep.add(value)
                if len(keep) >= self.candidate_count:
                    break
            values = sorted(keep)
        return values or ([0] if self.awi.allow_zero_quantity else [1])

    # ------------------------------------------------------------------
    # Price and profit estimates
    # ------------------------------------------------------------------

    def _offer_price(self, partner, t: float) -> int:
        best = self._best_price_for_me(partner)
        worst = self._worst_price_for_me(partner)
        concession = self.price_concession * max(0.0, min(1.0, t))
        if self._is_seller_to(partner):
            return int(round(best - (best - worst) * concession))
        return int(round(best + (worst - best) * concession))

    def _side_unit_revenue(self, partners: list[str], prices: dict[str, float] | None = None) -> float:
        if not partners:
            return 0.0
        if self._is_seller_to(partners[0]):
            if prices:
                return sum(float(prices[p]) for p in partners) / len(partners)
            return float(self._offer_price(partners[0], 0.0))
        return max(0.0, self._trading_price(output=True) - self._production_cost())

    def _side_unit_cost(self, partners: list[str], prices: dict[str, float] | None = None) -> float:
        if not partners:
            return 0.0
        if self._is_seller_to(partners[0]):
            return max(0.0, self._trading_price(output=False) + self._production_cost())
        if prices:
            return sum(float(prices[p]) for p in partners) / len(partners)
        return float(self._offer_price(partners[0], 0.0))

    def _profit_delta_for_side(
        self,
        partners: list[str],
        expected_total: float,
        prices: dict[str, float] | None = None,
    ) -> float:
        input_quantity, output_quantity, buy_cost, sell_revenue = self._current_contract_totals()
        before = self._profit_from_totals(
            input_quantity,
            output_quantity,
            buy_cost,
            sell_revenue,
        )

        unit_price = self._side_unit_revenue(partners, prices)
        if not self._is_seller_to(partners[0]):
            unit_price = self._side_unit_cost(partners, prices)

        if self._is_seller_to(partners[0]):
            output_quantity += expected_total
            sell_revenue += expected_total * unit_price
        else:
            input_quantity += expected_total
            buy_cost += expected_total * unit_price

        after = self._profit_from_totals(
            input_quantity,
            output_quantity,
            buy_cost,
            sell_revenue,
        )
        return after - before

    def _profit_from_totals(
        self,
        input_quantity: float,
        output_quantity: float,
        buy_cost: float,
        sell_revenue: float,
    ) -> float:
        produced = min(float(input_quantity), float(output_quantity), float(self._max_lines()))
        excess_quantity = max(0.0, float(input_quantity) - produced)
        shortfall_quantity = max(0.0, float(output_quantity) - produced)
        disposal = float(getattr(self.awi, "current_disposal_cost", 0.0) or 0.0)
        shortfall = float(getattr(self.awi, "current_shortfall_penalty", 0.0) or 0.0)
        return (
            float(sell_revenue)
            - float(buy_cost)
            - self._production_cost() * produced
            - disposal * self._trading_price(output=False) * excess_quantity
            - shortfall * self._trading_price(output=True) * shortfall_quantity
        )

    def _current_contract_totals(self) -> tuple[float, float, float, float]:
        input_quantity = output_quantity = 0.0
        buy_cost = sell_revenue = 0.0

        for partner, offer in self.accepted_contracts:
            if offer is None or len(offer) <= UNIT_PRICE:
                continue
            quantity = max(0.0, float(offer[QUANTITY]))
            price = float(offer[UNIT_PRICE])
            if self._is_seller_to(partner):
                output_quantity += quantity
                sell_revenue += quantity * price
            else:
                input_quantity += quantity
                buy_cost += quantity * price

        exogenous_input = self._current_exogenous_quantity(output=False)
        exogenous_output = self._current_exogenous_quantity(output=True)
        input_quantity += exogenous_input
        output_quantity += exogenous_output
        buy_cost += exogenous_input * self._current_exogenous_price(output=False)
        sell_revenue += exogenous_output * self._current_exogenous_price(output=True)
        return input_quantity, output_quantity, buy_cost, sell_revenue

    def _current_exogenous_quantity(self, output: bool) -> float:
        names = (
            (
                "current_exogenous_output_quantity",
                "current_exogenous_output",
                "current_exogenous_sales_quantity",
                "current_exogenous_sales",
            )
            if output
            else (
                "current_exogenous_input_quantity",
                "current_exogenous_input",
                "current_exogenous_supplies_quantity",
                "current_exogenous_supplies",
            )
        )
        for name in names:
            value = getattr(self.awi, name, None)
            try:
                return max(0.0, float(value))
            except Exception:
                pass
        return 0.0

    def _current_exogenous_price(self, output: bool) -> float:
        names = (
            (
                "current_exogenous_output_price",
                "current_exogenous_sales_price",
            )
            if output
            else (
                "current_exogenous_input_price",
                "current_exogenous_supplies_price",
            )
        )
        for name in names:
            value = getattr(self.awi, name, None)
            try:
                return max(0.0, float(value))
            except Exception:
                pass
        return self._trading_price(output=output)

    def _trading_price(self, output: bool) -> float:
        try:
            product = self.awi.my_output_product if output else self.awi.my_input_product
            prices = self.awi.trading_prices
            if isinstance(prices, dict):
                return float(prices.get(product, prices.get(str(product), 1.0)))
            return float(prices[product])
        except Exception:
            issues = self.awi.current_output_issues if output else self.awi.current_input_issues
            return float((issues[UNIT_PRICE].min_value + issues[UNIT_PRICE].max_value) / 2)

    def _production_cost(self) -> float:
        try:
            return float(self.awi.profile.cost)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # History tracking
    # ------------------------------------------------------------------

    def _record_sent_offers(self, proposals, states=None):
        for partner, offer in proposals.items():
            if partner is None or offer is None or len(offer) <= UNIT_PRICE:
                continue
            self._ensure_partner(partner)
            round_step = self._state_step(partner, states) if states is not None else 0
            counted = round_step <= 3
            if counted:
                self._offer_trials[partner] += 1
            self._sent_offer_history[partner].append(
                {
                    "step": int(self.awi.current_step),
                    "round_step": int(round_step),
                    "counted": bool(counted),
                    "offer": tuple(offer),
                    "observed": False,
                }
            )
            if len(self._sent_offer_history[partner]) > 30:
                del self._sent_offer_history[partner][:-30]

    def _observe_incoming_offers(self, offers, states):
        for partner, offer in offers.items():
            self._ensure_partner(partner)
            sent_offer = self._latest_unobserved_sent_offer(partner)
            if sent_offer is not None and not self._same_offer(sent_offer["offer"], offer):
                sent_offer["observed"] = True
                if sent_offer.get("counted", False):
                    self._reoffer_events[partner] += 1
                    self._counter_offer_quantities[partner].append(
                        max(0, int(offer[QUANTITY]))
                    )
                    if len(self._counter_offer_quantities[partner]) > 50:
                        del self._counter_offer_quantities[partner][:-50]

    def on_negotiation_success(self, contract: Contract, mechanism: Any):
        super().on_negotiation_success(contract, mechanism)
        partner = self._contract_partner(contract)
        outcome = self._contract_outcome(contract)
        if partner is None or outcome is None:
            return
        try:
            if int(outcome[TIME]) == int(self.awi.current_step):
                self.accepted_contracts.append((partner, outcome))
        except Exception:
            self.accepted_contracts.append((partner, outcome))
        sent_offer = self._matching_sent_offer(partner, outcome)
        if sent_offer is not None:
            sent_offer["observed"] = True
            if sent_offer.get("counted", False):
                self._offer_accepts[partner] += 1

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        super().on_negotiation_failure(partners, annotation, mechanism, state)
        try:
            partner = next(partner for partner in partners if partner != self.id)
        except StopIteration:
            return
        sent_offer = self._latest_unobserved_sent_offer(partner)
        if sent_offer is not None:
            sent_offer["observed"] = True

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------

    def _all_partners(self) -> list[str]:
        return list(dict.fromkeys(list(self.awi.my_suppliers) + list(self.awi.my_consumers)))

    def _ensure_partner(self, partner):
        if partner is None:
            return
        self._offer_trials.setdefault(partner, 0)
        self._offer_accepts.setdefault(partner, 0)
        self._reoffer_events.setdefault(partner, 0)
        self._counter_offer_quantities.setdefault(partner, [])
        self._sent_offer_history.setdefault(partner, [])

    def _is_seller_to(self, partner) -> bool:
        return partner in self.awi.my_consumers

    def _issues_for(self, partner):
        return self.awi.current_output_issues if self._is_seller_to(partner) else self.awi.current_input_issues

    def _best_price_for_me(self, partner) -> int:
        issues = self._issues_for(partner)
        return int(issues[UNIT_PRICE].max_value if self._is_seller_to(partner) else issues[UNIT_PRICE].min_value)

    def _worst_price_for_me(self, partner) -> int:
        issues = self._issues_for(partner)
        return int(issues[UNIT_PRICE].min_value if self._is_seller_to(partner) else issues[UNIT_PRICE].max_value)

    def _needed_for(self, partner) -> int:
        return int(self.awi.needed_sales if self._is_seller_to(partner) else self.awi.needed_supplies)

    def _max_lines(self) -> int:
        return max(1, int(getattr(self.awi, "n_lines", getattr(self.awi, "max_n_lines", 1)) or 1))

    def _clamp_quantity(self, partner, quantity: int) -> int:
        issues = self._issues_for(partner)
        qmin = int(issues[QUANTITY].min_value)
        qmax = int(issues[QUANTITY].max_value)
        if quantity <= 0 and self.awi.allow_zero_quantity:
            return 0
        return max(qmin, min(qmax, int(quantity)))

    def _price_for_me_score(self, partner, price: float) -> float:
        issues = self._issues_for(partner)
        pmin = float(issues[UNIT_PRICE].min_value)
        pmax = float(issues[UNIT_PRICE].max_value)
        span = max(1.0, pmax - pmin)
        if self._is_seller_to(partner):
            return max(0.0, min(1.0, (float(price) - pmin) / span))
        return max(0.0, min(1.0, (pmax - float(price)) / span))

    def _price_for_opponent_score(self, partner, price: float) -> float:
        return 1.0 - self._price_for_me_score(partner, price)

    def _relative_time(self, states) -> float:
        values = [
            float(getattr(state, "relative_time", 0.0))
            for state in states.values()
            if getattr(state, "relative_time", None) is not None
        ]
        return max(0.0, min(1.0, min(values, default=0.0)))

    def _state_step(self, partner, states) -> int:
        state = states.get(partner)
        if state is None:
            return 0
        value = getattr(state, "step", None)
        return int(value) if isinstance(value, int) else 0

    def _powerset(self, iterable):
        items = list(iterable)
        return itertools.chain.from_iterable(
            itertools.combinations(items, size) for size in range(len(items) + 1)
        )

    def _unneeded_response(self):
        if self.awi.allow_zero_quantity:
            return SAOResponse(
                ResponseType.REJECT_OFFER,
                (0, self.awi.current_step, 0),
            )
        return SAOResponse(ResponseType.END_NEGOTIATION, None)

    def _contract_partner(self, contract: Contract):
        for partner in getattr(contract, "partners", ()):
            if partner != self.id:
                return partner
        return None

    def _contract_outcome(self, contract: Contract) -> Outcome | None:
        agreement = getattr(contract, "agreement", None)
        if agreement is None:
            return None
        if isinstance(agreement, dict):
            quantity = agreement.get(QUANTITY, agreement.get("quantity"))
            delivery_step = agreement.get(TIME, agreement.get("time"))
            unit_price = agreement.get(UNIT_PRICE, agreement.get("unit_price"))
            if quantity is None or delivery_step is None or unit_price is None:
                return None
            return (quantity, delivery_step, unit_price)
        if len(agreement) <= UNIT_PRICE:
            return None
        return tuple(agreement)

    def _same_offer(self, left, right) -> bool:
        if left is None or right is None:
            return False
        return (
            int(left[QUANTITY]) == int(right[QUANTITY])
            and int(left[TIME]) == int(right[TIME])
            and int(left[UNIT_PRICE]) == int(right[UNIT_PRICE])
        )

    def _latest_unobserved_sent_offer(self, partner):
        for item in reversed(self._sent_offer_history.get(partner, [])):
            if item["step"] == self.awi.current_step and not item.get("observed", False):
                return item
        return None

    def _matching_sent_offer(self, partner, outcome):
        for item in reversed(self._sent_offer_history.get(partner, [])):
            if item.get("observed", False):
                continue
            if self._same_offer(item["offer"], outcome):
                return item
        return None


class EINearestNeedOneShotAgent(RDVOOneShotAgent):
    """
    RDVO's expectation model with a simpler quantity target.

    For every candidate distribution, this agent computes each partner's
    expected fulfilled quantity e_i and chooses the offer distribution whose
    total expected quantity is closest to the current required amount.

    In this version, p_i, r_i, and mu_i are learned only from our first proposals:

        p_i  = (accepted first offers + 1) / (sent first offers + 2)
        r_i  = (counter offers to our first offers + 1)
               / (estimated rejected first offers + 2)
        mu_i = average quantity in counter offers to our first offers
               over the latest 20 observations, defaulting to 2.0
    """

    # ------------------------------------------------------------
    # First proposals
    # ------------------------------------------------------------

    def first_proposals(self):
        proposals = {}

        for needs, all_partners in (
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ):
            partners = [
                partner
                for partner in all_partners
                if partner in self.negotiators
            ]

            if not partners:
                continue

            side_proposals = self._rdvo_side_proposals(
                int(needs),
                partners,
                t=0.0,
            )

            # None は送らない
            proposals.update(
                {
                    partner: offer
                    for partner, offer in side_proposals.items()
                    if offer is not None
                }
            )

        # 初手だけを p_i, r_i, mu_i の学習対象として記録する
        self._record_sent_offers(
            proposals,
            states=None,
            is_first=True,
        )

        return proposals

    # ------------------------------------------------------------
    # Counter all
    # ------------------------------------------------------------

    def counter_all(self, offers, states):
        current_offers = {
            partner: offer
            for partner, offer in offers.items()
            if offer is not None
            and len(offer) > UNIT_PRICE
            and offer[TIME] == self.awi.current_step
        }

        self._observe_incoming_offers(current_offers, states)

        responses = {}
        t = self._relative_time(states)

        for needs, all_partners in (
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ):
            partners = [
                partner
                for partner in all_partners
                if partner in current_offers
            ]

            if not partners:
                continue

            if needs <= 0:
                for partner in partners:
                    responses[partner] = self._unneeded_response()
                continue

            accepted_partners = self._rdvo_accept_subset(
                int(needs),
                partners,
                current_offers,
                t,
            )

            accepted_quantity = sum(
                int(current_offers[partner][QUANTITY])
                for partner in accepted_partners
            )

            for partner in accepted_partners:
                responses[partner] = SAOResponse(
                    ResponseType.ACCEPT_OFFER,
                    current_offers[partner],
                )

            remaining_partners = [
                partner
                for partner in partners
                if partner not in accepted_partners
            ]

            remaining_needs = max(0, int(needs) - accepted_quantity)

            if remaining_needs <= 0:
                for partner in remaining_partners:
                    responses[partner] = self._unneeded_response()
                continue

            counter_proposals = self._rdvo_side_proposals(
                remaining_needs,
                remaining_partners,
                t=t,
            )

            for partner in remaining_partners:
                offer = counter_proposals.get(partner)

                if offer is None:
                    responses[partner] = self._unneeded_response()
                else:
                    responses[partner] = SAOResponse(
                        ResponseType.REJECT_OFFER,
                        offer,
                    )

        # カウンターオファーは学習対象にしない
        self._record_sent_offers(
            {
                partner: response.outcome
                for partner, response in responses.items()
                if response is not None
                and response.response == ResponseType.REJECT_OFFER
                and response.outcome is not None
            },
            states=states,
            is_first=False,
        )

        return responses

    # ------------------------------------------------------------
    # Quantity proposal selection
    # ------------------------------------------------------------

    def _rdvo_side_proposals(self, needs: int, partners: list[str], t: float):
        if not partners:
            return {}

        if needs <= 0:
            return {partner: None for partner in partners}

        candidate_lists = [
            self._quantity_candidates(needs, partner)
            for partner in partners
        ]

        best_score = (
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
        )
        best_quantities = [0] * len(partners)

        total_offer_cap = None
        if self.max_total_offer_multiplier > 0:
            total_offer_cap = math.ceil(
                max(0, int(needs)) * self.max_total_offer_multiplier
            )

        for quantities in itertools.product(*candidate_lists):
            offered_total = sum(
                max(0, int(quantity))
                for quantity in quantities
            )

            if total_offer_cap is not None and offered_total > total_offer_cap:
                continue

            expected_units = []

            for partner, quantity in zip(partners, quantities, strict=False):
                price = self._offer_price(partner, t)

                expected_units.append(
                    self._expected_units(
                        partner,
                        int(quantity),
                        price,
                    )
                )

            expected_total = max(0.0, sum(expected_units))
            expected_gap = abs(float(needs) - expected_total)
            expected_over = max(0.0, expected_total - float(needs))

            value = self._rdvo_value(
                needs,
                partners,
                quantities,
                expected_units,
                t,
                offered_total=offered_total,
            )

            offered_gap = abs(float(needs) - float(offered_total))

            # 小さいほど良い
            # 1. 期待成立量が必要量に近い
            # 2. 期待過剰が少ない
            # 3. RDVO価値が高い
            # 4. 実オファー合計が必要量に近い
            score = (
                expected_gap,
                expected_over,
                -value,
                offered_gap,
            )

            if score < best_score:
                best_score = score
                best_quantities = list(quantities)

        proposals = {}

        for partner, quantity in zip(partners, best_quantities, strict=False):
            quantity = int(quantity)

            # 0 は「この相手には出さない」という内部表現
            if quantity <= 0:
                proposals[partner] = None
                continue

            quantity = self._clamp_quantity(partner, quantity)

            proposals[partner] = (
                quantity,
                self.awi.current_step,
                self._offer_price(partner, t),
            )

        return proposals

    # ------------------------------------------------------------
    # Quantity candidates
    # ------------------------------------------------------------

    def _quantity_candidates(self, needs: int, partner) -> list[int]:
        qmax = max(
            1,
            int(self._issues_for(partner)[QUANTITY].max_value),
        )

        upper = max(
            0,
            min(
                qmax,
                max(needs, math.ceil(needs * 1.2)),
            ),
        )

        # 0 は実オファーではなく「この相手には出さない」という内部候補
        raw = {
            0,
            1,
            min(upper, math.ceil(needs * 0.25)),
            min(upper, math.ceil(needs * 0.50)),
            min(upper, math.ceil(needs * 0.80)),
            min(upper, needs),
        }

        values = sorted(
            value
            for value in raw
            if 0 <= value <= qmax
        )

        if len(values) > self.candidate_count:
            keep = {
                0,
                values[-1],
                min(values, key=lambda value: abs(value - needs)),
            }

            for value in values:
                keep.add(value)
                if len(keep) >= self.candidate_count:
                    break

            values = sorted(keep)

        return values or [0]

    # ------------------------------------------------------------
    # First-offer-only history tracking
    # ------------------------------------------------------------

    def _record_sent_offers(self, proposals, states=None, is_first: bool = False):
        for partner, offer in proposals.items():
            if partner is None or offer is None or len(offer) <= UNIT_PRICE:
                continue

            self._ensure_partner(partner)

            round_step = (
                self._state_step(partner, states)
                if states is not None
                else 0
            )

            # 初手オファーだけを p_i, r_i, mu_i の学習対象にする
            counted = bool(is_first)

            if counted:
                self._offer_trials[partner] += 1

            self._sent_offer_history[partner].append(
                {
                    "step": int(self.awi.current_step),
                    "round_step": int(round_step),
                    "counted": bool(counted),
                    "offer": tuple(offer),
                    "observed": False,
                }
            )

            if len(self._sent_offer_history[partner]) > 30:
                del self._sent_offer_history[partner][:-30]

    def _observe_incoming_offers(self, offers, states):
        for partner, offer in offers.items():
            self._ensure_partner(partner)

            sent_offer = self._latest_unobserved_sent_offer(partner)

            if sent_offer is None:
                continue

            if self._same_offer(sent_offer["offer"], offer):
                continue

            sent_offer["observed"] = True

            # counted=True、つまり自分の初手オファーへの counter offer だけを記録
            if sent_offer.get("counted", False):
                self._reoffer_events[partner] += 1

                self._counter_offer_quantities[partner].append(
                    max(0, int(offer[QUANTITY]))
                )

                if len(self._counter_offer_quantities[partner]) > 50:
                    del self._counter_offer_quantities[partner][:-50]

    # ------------------------------------------------------------
    # Expected quantity model
    # ------------------------------------------------------------

    def _expected_units(self, partner, quantity: int, price: float) -> float:
        if quantity <= 0:
            return 0.0

        p = self._accept_probability(partner, quantity, price)
        r = self._reoffer_probability(partner)
        mu = self._mean_reoffer_quantity(partner)

        return p * quantity + (1.0 - p) * r * mu

    def _accept_probability(self, partner, quantity: int, price: float) -> float:
        del quantity, price

        trials = self._offer_trials[partner]
        accepts = self._offer_accepts[partner]

        base = (accepts + 1.0) / (trials + 2.0)

        return max(0.02, min(0.98, base))

    def _reoffer_probability(self, partner) -> float:
        rejected = max(
            0,
            self._offer_trials[partner] - self._offer_accepts[partner],
        )

        return (self._reoffer_events[partner] + 1.0) / (rejected + 2.0)

    def _mean_reoffer_quantity(self, partner) -> float:
        quantities = self._counter_offer_quantities.get(partner, [])

        if quantities:
            recent = quantities[-20:]
            return sum(recent) / len(recent)

        return 2.0

    # ------------------------------------------------------------
    # Negotiation callbacks
    # ------------------------------------------------------------

    def on_negotiation_success(self, contract: Contract, mechanism: Any):
        super().on_negotiation_success(contract, mechanism)

        partner = self._contract_partner(contract)
        outcome = self._contract_outcome(contract)

        if partner is None or outcome is None:
            return

        sent_offer = self._matching_sent_offer(partner, outcome)

        if sent_offer is not None:
            sent_offer["observed"] = True

            # counted=True、つまり自分の初手オファーが成立した場合だけ accept と数える
            if sent_offer.get("counted", False):
                self._offer_accepts[partner] += 1

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        super().on_negotiation_failure(
            partners,
            annotation,
            mechanism,
            state,
        )

        try:
            partner = next(
                partner
                for partner in partners
                if partner != self.id
            )
        except StopIteration:
            return

        sent_offer = self._latest_unobserved_sent_offer(partner)

        if sent_offer is not None:
            sent_offer["observed"] = True


RDVOAgent = RDVOOneShotAgent
EINearestNeedAgent = EINearestNeedOneShotAgent
MyAgent = RDVOOneShotAgent
