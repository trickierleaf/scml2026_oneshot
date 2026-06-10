from __future__ import annotations

import math
import random
from collections import defaultdict
from itertools import chain, combinations
from typing import Any

from negmas import Contract, Outcome, ResponseType, SAOResponse
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE

from .rdvo_agent import RDVOOneShotAgent


class BayesianAgent2(RDVOOneShotAgent):
    """
    Greedy / non-greedy classifier using hard vetoes and softmax logits.

    OneShot-oriented version:
    - During probe days, it tests opponent behavior using the two possible price extremes.
    - After probe days:
        * If Greedy opponents exist, it targets them with opponent-favorable prices.
        * If no Greedy opponents exist, first proposals fall back to RDVO/EINearestNeed style.
    - Counter strategy:
        * Use OneShot-oriented probabilistic accept + counter optimization
          regardless of whether Greedy opponents are currently classified.
    """

    OPPONENT_TYPES = ("GreedyOneShotAgent", "NonGreedy")

    def __init__(
        self,
        *args,
        probe_days: int = 6,
        classification_threshold: float = 0.75,
        softmax_temperature: float = 1.0,
        min_observations: int = 1,
        equal: bool = False,
        overordering_max_selling: float = 0.0,
        overordering_max_buying: float = 0.2,
        overordering_min: float = 0.0,
        overordering_exp: float = 0.4,
        mismatch_exp: float = 4.0,
        overmismatch_max_selling: float = 0.0,
        overmismatch_max_buying: float = 0.3,
        undermismatch_min_selling: float = -0.4,
        undermismatch_min_buying: float = -0.2,
        counter_good_price_time: float = 0.65,
        counter_good_price_shortage_ratio: float = 0.40,
        counter_bad_price_penalty: float = 0.30,
        max_counter_partners: int = 4,
        counter_beam_width: int = 24,
        max_accept_subsets: int = 16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.probe_days = probe_days
        self.classification_threshold = classification_threshold
        self.softmax_temperature = max(0.01, float(softmax_temperature))
        self.min_observations = min_observations
        self.non_greedy_success_default = 0.5
        self.equal_distribution = equal

        self.overordering_max_selling = overordering_max_selling
        self.overordering_max_buying = overordering_max_buying
        self.overordering_min = overordering_min
        self.overordering_exp = overordering_exp
        self.mismatch_exp = mismatch_exp
        self.overmismatch_max_selling = overmismatch_max_selling
        self.overmismatch_max_buying = overmismatch_max_buying
        self.undermismatch_min_selling = undermismatch_min_selling
        self.undermismatch_min_buying = undermismatch_min_buying

        # OneShot counter parameters.
        # Since prices are basically two-valued, worst price for me is used
        # only late or under high shortage pressure.
        self.counter_good_price_time = float(counter_good_price_time)
        self.counter_good_price_shortage_ratio = float(counter_good_price_shortage_ratio)
        self.counter_bad_price_penalty = float(counter_bad_price_penalty)
        self.max_counter_partners = int(max_counter_partners)
        self.counter_beam_width = int(counter_beam_width)
        self.max_accept_subsets = int(max_accept_subsets)

    def init(self):
        self._opponent_logits = {}
        self._opponent_observations = defaultdict(int)
        self._non_greedy_veto = {}

        self._sent_first_offers = defaultdict(list)
        self._received_offer_counts = defaultdict(int)
        self._received_first_offer_history = defaultdict(list)
        self._own_first_offer_result_history = defaultdict(list)
        self._logit_history = defaultdict(list)
        self._evidence_counts = defaultdict(lambda: defaultdict(int))

        # Price-conditioned first-offer statistics.
        # label is opponent-side price label: good / bad / neutral.
        self._first_offer_trials_by_price = defaultdict(lambda: defaultdict(int))
        self._first_offer_accepts_by_price = defaultdict(lambda: defaultdict(int))
        self._first_offer_reoffers_by_price = defaultdict(lambda: defaultdict(int))
        self._first_offer_counter_quantities_by_price = defaultdict(
            lambda: defaultdict(list)
        )

        super().init()

        lines = self._max_lines()
        self.overordering_max = (
            self.overordering_max_selling
            if self.awi.my_suppliers == ["SELLER"]
            else self.overordering_max_buying
        )
        self._overmismatch_max_selling_abs = self.overmismatch_max_selling * lines
        self._overmismatch_max_buying_abs = self.overmismatch_max_buying * lines
        self._undermismatch_min_selling_abs = self.undermismatch_min_selling * lines
        self._undermismatch_min_buying_abs = self.undermismatch_min_buying * lines

        self.total_agreed_quantity = {partner: 0 for partner in self._all_partners()}

        for partner in self._all_partners():
            self._ensure_partner(partner)

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------

    def _all_partners(self) -> list[str]:
        return list(
            dict.fromkeys(list(self.awi.my_suppliers) + list(self.awi.my_consumers))
        )

    def _max_lines(self) -> int:
        return max(
            1,
            int(getattr(self.awi, "n_lines", getattr(self.awi, "max_n_lines", 1)) or 1),
        )

    def _ensure_partner(self, partner):
        if partner is None:
            return

        super()._ensure_partner(partner)

        if partner not in self._opponent_logits:
            self._opponent_logits[partner] = {
                "GreedyOneShotAgent": 0.0,
                "NonGreedy": 0.0,
            }

        self._opponent_observations.setdefault(partner, 0)
        self._sent_first_offers.setdefault(partner, [])
        self._received_first_offer_history.setdefault(partner, [])
        self._own_first_offer_result_history.setdefault(partner, [])
        self._logit_history.setdefault(partner, [])
        self._evidence_counts.setdefault(partner, defaultdict(int))

        self._first_offer_trials_by_price.setdefault(partner, defaultdict(int))
        self._first_offer_accepts_by_price.setdefault(partner, defaultdict(int))
        self._first_offer_reoffers_by_price.setdefault(partner, defaultdict(int))
        self._first_offer_counter_quantities_by_price.setdefault(
            partner,
            defaultdict(list),
        )

    def _is_seller_to(self, partner) -> bool:
        return partner in self.awi.my_consumers

    def _issues_for(self, partner):
        return (
            self.awi.current_output_issues
            if self._is_seller_to(partner)
            else self.awi.current_input_issues
        )

    def _best_price_for_me(self, partner) -> int:
        issues = self._issues_for(partner)
        return int(
            issues[UNIT_PRICE].max_value
            if self._is_seller_to(partner)
            else issues[UNIT_PRICE].min_value
        )

    def _worst_price_for_me(self, partner) -> int:
        issues = self._issues_for(partner)
        return int(
            issues[UNIT_PRICE].min_value
            if self._is_seller_to(partner)
            else issues[UNIT_PRICE].max_value
        )

    def _needed_for(self, partner) -> int:
        return int(
            self.awi.needed_sales
            if self._is_seller_to(partner)
            else self.awi.needed_supplies
        )

    def _clamp_quantity(self, partner, quantity: int) -> int:
        issues = self._issues_for(partner)
        qmin = int(issues[QUANTITY].min_value)
        qmax = int(issues[QUANTITY].max_value)
        if quantity <= 0 and self.awi.allow_zero_quantity:
            return 0
        return max(qmin, min(qmax, int(quantity)))

    def _offer(self, partner, quantity: int, price: int) -> Outcome | None:
        quantity = self._clamp_quantity(partner, quantity)
        if quantity <= 0 and not self.awi.allow_zero_quantity:
            return None
        return (quantity, self.awi.current_step, int(price))

    def _counter_or_accept_response(self, partner, current_offer, counter_offer):
        if counter_offer is None:
            return self._unneeded_response()
        if self._same_offer(counter_offer, current_offer) or self._same_quantity_offer(
            counter_offer,
            current_offer,
        ):
            return SAOResponse(
                ResponseType.ACCEPT_OFFER,
                current_offer,
            )
        return SAOResponse(
            ResponseType.REJECT_OFFER,
            counter_offer,
        )

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

    def _price_good_for_opponent(self, partner, price: float) -> bool:
        return self._price_for_opponent_score(partner, price) >= 0.5

    def _opponent_price_label(self, partner, price: float) -> str:
        score = self._price_for_opponent_score(partner, price)
        if score >= 0.80:
            return "good"
        if score <= 0.20:
            return "bad"
        return "neutral"

    def _unneeded_response(self):
        if self.awi.allow_zero_quantity:
            return SAOResponse(
                ResponseType.REJECT_OFFER,
                (0, self.awi.current_step, 0),
            )
        return SAOResponse(ResponseType.END_NEGOTIATION, None)

    # ------------------------------------------------------------------
    # Rule/logit classifier
    # ------------------------------------------------------------------

    def opponent_posteriors(self, partner) -> dict[str, float]:
        self._ensure_partner(partner)

        if partner in self._non_greedy_veto:
            return {"GreedyOneShotAgent": 0.0, "NonGreedy": 1.0}

        logits = self._opponent_logits[partner]
        scaled = {
            name: value / self.softmax_temperature
            for name, value in logits.items()
        }
        center = max(scaled.values())
        weights = {
            name: math.exp(value - center)
            for name, value in scaled.items()
        }
        total = sum(weights.values())

        if total <= 0:
            return {
                name: 1.0 / len(self.OPPONENT_TYPES)
                for name in self.OPPONENT_TYPES
            }

        return {
            name: weights[name] / total
            for name in self.OPPONENT_TYPES
        }

    def opponent_type(self, partner) -> str:
        self._ensure_partner(partner)

        if partner in self._non_greedy_veto:
            return "NonGreedy"

        if self._opponent_observations[partner] < self.min_observations:
            return "Unknown"

        posteriors = self.opponent_posteriors(partner)
        best = max(posteriors, key=posteriors.get)

        if posteriors[best] >= self.classification_threshold:
            return best

        return "Unknown"

    def _add_logit_evidence(
        self,
        partner,
        *,
        greedy: float = 0.0,
        non_greedy: float = 0.0,
        reason: str,
    ):
        self._ensure_partner(partner)

        self._opponent_logits[partner]["GreedyOneShotAgent"] += float(greedy)
        self._opponent_logits[partner]["NonGreedy"] += float(non_greedy)
        self._opponent_observations[partner] += 1

        self._logit_history[partner].append(
            {
                "step": int(self.awi.current_step),
                "greedy_delta": float(greedy),
                "non_greedy_delta": float(non_greedy),
                "logits": dict(self._opponent_logits[partner]),
                "reason": reason,
            }
        )

        if len(self._logit_history[partner]) > 50:
            del self._logit_history[partner][:-50]

    def _add_evidence_count(self, partner, name: str):
        self._ensure_partner(partner)
        self._evidence_counts[partner][name] += 1

    def _veto_non_greedy(self, partner, reason: str):
        self._ensure_partner(partner)

        if partner not in self._non_greedy_veto:
            self._non_greedy_veto[partner] = reason

        self._add_logit_evidence(
            partner,
            non_greedy=8.0,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # First-offer tracking and classification evidence
    # ------------------------------------------------------------------

    def _record_first_offer(self, partner, offer):
        if partner is None or offer is None or len(offer) <= UNIT_PRICE:
            return

        self._ensure_partner(partner)

        price_label = self._opponent_price_label(partner, offer[UNIT_PRICE])

        self._sent_first_offers[partner].append(
            {
                "step": int(self.awi.current_step),
                "offer": tuple(offer),
                "price_label": price_label,
                "observed": False,
            }
        )

        self._first_offer_trials_by_price[partner][price_label] += 1

        if len(self._sent_first_offers[partner]) > 20:
            del self._sent_first_offers[partner][:-20]

    def _matching_first_offer(self, partner, outcome):
        for item in reversed(self._sent_first_offers.get(partner, [])):
            if item["step"] != self.awi.current_step:
                continue
            if item.get("observed", False):
                continue
            if self._same_offer(item["offer"], outcome):
                return item
        return None

    def _latest_unobserved_first_offer(self, partner):
        for item in reversed(self._sent_first_offers.get(partner, [])):
            if (
                item["step"] == self.awi.current_step
                and not item.get("observed", False)
            ):
                return item
        return None

    def _observe_own_first_offer_result(self, partner, sent_offer, accepted: bool):
        if sent_offer is None or sent_offer.get("observed", False):
            return

        sent_offer["observed"] = True

        price_label = sent_offer.get("price_label", "neutral")
        accepted = bool(accepted)

        if accepted:
            self._first_offer_accepts_by_price[partner][price_label] += 1

        if price_label == "bad" and accepted:
            self._add_evidence_count(partner, "bad_first_offer_accepted")
            self._veto_non_greedy(partner, "bad_first_offer_accepted")

        elif price_label == "bad" and not accepted:
            self._add_evidence_count(partner, "bad_first_offer_rejected")
            self._add_logit_evidence(
                partner,
                greedy=0.08,
                reason="bad_first_offer_rejected",
            )

        elif price_label == "good" and accepted:
            self._add_evidence_count(partner, "good_first_offer_accepted")
            self._add_logit_evidence(
                partner,
                greedy=1.80,
                reason="good_first_offer_accepted",
            )

        elif price_label == "good":
            self._add_evidence_count(partner, "good_first_offer_rejected")
            self._add_logit_evidence(
                partner,
                non_greedy=0.70,
                reason="good_first_offer_rejected",
            )

        elif accepted:
            self._add_logit_evidence(
                partner,
                greedy=0.05,
                reason="neutral_first_offer_accepted",
            )

        else:
            self._add_logit_evidence(
                partner,
                non_greedy=0.05,
                reason="neutral_first_offer_rejected",
            )

        self._own_first_offer_result_history[partner].append(
            {
                "step": int(self.awi.current_step),
                "price_label": price_label,
                "accepted": accepted,
            }
        )

        if len(self._own_first_offer_result_history[partner]) > 30:
            del self._own_first_offer_result_history[partner][:-30]

    def _observe_own_first_offer_counter(self, partner, sent_offer, counter_offer):
        if sent_offer is None or sent_offer.get("observed", False):
            return

        price_label = sent_offer.get("price_label", "neutral")
        self._first_offer_reoffers_by_price[partner][price_label] += 1

        self._first_offer_counter_quantities_by_price[partner][price_label].append(
            max(0, int(counter_offer[QUANTITY]))
        )

        quantities = self._first_offer_counter_quantities_by_price[partner][price_label]
        if len(quantities) > 50:
            del quantities[:-50]

        self._observe_own_first_offer_result(
            partner,
            sent_offer,
            accepted=False,
        )

    def _observe_received_first_offer(self, partner, offer):
        if offer is None or len(offer) <= UNIT_PRICE:
            return

        self._ensure_partner(partner)

        if self._received_offer_counts[partner] > 0:
            self._received_offer_counts[partner] += 1
            return

        self._received_offer_counts[partner] += 1

        price_label = self._opponent_price_label(partner, offer[UNIT_PRICE])

        if price_label == "good":
            self._add_evidence_count(partner, "opponent_first_offer_good_price")
            self._add_logit_evidence(
                partner,
                greedy=0.03,
                reason="opponent_first_offer_good_price",
            )

        elif price_label == "bad":
            self._add_evidence_count(partner, "opponent_first_offer_bad_price")
            self._veto_non_greedy(partner, "opponent_first_offer_bad_price")

        self._received_first_offer_history[partner].append(
            {
                "step": int(self.awi.current_step),
                "quantity": int(offer[QUANTITY]),
                "price_label": price_label,
            }
        )

        if len(self._received_first_offer_history[partner]) > 30:
            del self._received_first_offer_history[partner][:-30]

    # ------------------------------------------------------------------
    # Strategy hooks
    # ------------------------------------------------------------------

    def before_step(self):
        super().before_step()
        self._received_offer_counts.clear()

        for partner in self._all_partners():
            self._ensure_partner(partner)

    def first_proposals(self):
        if self.awi.current_step >= self.probe_days:
            proposals = self._classified_first_proposals()
            self._record_first_offers(proposals)
            self._record_sent_offers(proposals, states=None)
            return proposals

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

            if needs <= 0:
                proposals.update({partner: None for partner in partners})
                continue

            quantities = self._distribute_quantity(
                int(needs),
                len(partners),
                mx=self._max_lines(),
                equal=True,
                allow_zero=self.awi.allow_zero_quantity,
            )

            for index, (partner, quantity) in enumerate(
                zip(partners, quantities, strict=False)
            ):
                opponent_good_probe = (self.awi.current_step + index) % 2 == 0

                price = (
                    self._worst_price_for_me(partner)
                    if opponent_good_probe
                    else self._best_price_for_me(partner)
                )

                if quantity <= 0 and not self.awi.allow_zero_quantity:
                    proposals[partner] = None
                else:
                    proposals[partner] = self._offer(partner, quantity, price)

        self._record_first_offers(proposals)
        return proposals

    def _classified_first_proposals(self):
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

            if needs <= 0:
                proposals.update({partner: None for partner in partners})
                continue

            greedy_partners = self._ranked_greedy_partners(partners)

            if not greedy_partners:
                proposals.update(
                    self._rdvo_side_proposals(
                        int(needs),
                        partners,
                        t=0.0,
                    )
                )
                continue

            proposals.update(
                self._bayesian_agent_first_side_proposals(
                    int(needs),
                    partners,
                )
            )

        return proposals

    def _record_first_offers(self, proposals):
        for partner, offer in proposals.items():
            self._record_first_offer(partner, offer)

    def counter_all(self, offers, states):
        # Observation phase.
        for partner, offer in offers.items():
            if offer is None or len(offer) <= UNIT_PRICE:
                continue
            if offer[TIME] != self.awi.current_step:
                continue

            sent_offer = self._latest_unobserved_first_offer(partner)

            if sent_offer is not None:
                if not self._same_offer(sent_offer["offer"], offer):
                    self._observe_own_first_offer_counter(
                        partner,
                        sent_offer,
                        offer,
                    )

            self._observe_received_first_offer(partner, offer)

        return self._oneshot_counter_all(offers, states)

    # ------------------------------------------------------------------
    # Classified first proposal strategy
    # ------------------------------------------------------------------

    def _ranked_greedy_partners(self, partners):
        greedy_partners = [
            partner
            for partner in partners
            if self.opponent_type(partner) == "GreedyOneShotAgent"
        ]

        return sorted(
            greedy_partners,
            key=lambda partner: self.opponent_posteriors(partner).get(
                "GreedyOneShotAgent",
                0.0,
            ),
            reverse=True,
        )

    def _bayesian_success_scaled_partners(self, partners, opponent_types):
        candidates = [
            partner
            for partner in partners
            if opponent_types.get(partner) != "GreedyOneShotAgent"
        ]

        return sorted(
            candidates,
            key=self._partner_non_greedy_initial_offer_success_rate,
            reverse=True,
        )

    def _bayesian_agent_first_side_proposals(
        self,
        needs: int,
        partners: list[str],
    ):
        proposals = {partner: None for partner in partners}

        if needs <= 0:
            return proposals

        opponent_types = {
            partner: self.opponent_type(partner)
            for partner in partners
        }

        greedy_partners = self._ranked_greedy_partners(partners)

        success_scaled_partners = self._bayesian_success_scaled_partners(
            partners,
            opponent_types,
        )

        success_rate = self._non_greedy_initial_offer_success_rate()

        selected_greedy_partners = []

        strong_greedy_partners = [
            partner
            for partner in greedy_partners
            if self._strong_initial_greedy_partner(partner)
        ]

        if len(greedy_partners) >= 2 and len(strong_greedy_partners) >= 2:
            selected_greedy_partners = strong_greedy_partners[:2]
            greedy_quantities = [
                math.ceil(needs / 2),
                math.floor(needs / 2),
            ]

            for partner, quantity in zip(
                selected_greedy_partners,
                greedy_quantities,
                strict=False,
            ):
                proposals[partner] = self._offer(
                    partner,
                    quantity,
                    self._worst_price_for_me(partner),
                )

            return proposals

        if greedy_partners:
            if len(greedy_partners) >= 2:
                selected_greedy_partners = greedy_partners[:2]
                greedy_target = math.floor(needs * 0.4)
                scaled_target = self._success_adjusted_quantity(
                    needs * 0.2,
                    success_rate,
                )
            else:
                selected_greedy_partners = greedy_partners[:1]
                greedy_target = math.floor(needs * 0.7)
                scaled_target = self._success_adjusted_quantity(
                    needs * 0.3,
                    success_rate,
                )

            for partner in selected_greedy_partners:
                proposals[partner] = self._offer(
                    partner,
                    greedy_target,
                    self._worst_price_for_me(partner),
                )

        else:
            scaled_target = self._success_adjusted_quantity(needs, success_rate)

        if len(greedy_partners) >= 3:
            for partner in greedy_partners[2:]:
                if partner not in success_scaled_partners:
                    success_scaled_partners.append(partner)

        if success_scaled_partners:
            scaled_target = max(
                int(scaled_target),
                min(2, len(success_scaled_partners)),
            )

            if self._is_process_one_agent():
                self._assign_success_weighted_quantities(
                    proposals,
                    success_scaled_partners,
                    scaled_target,
                    price_getter=self._best_price_for_me,
                    quantity_caps=self._half_quantity_caps(
                        int(needs),
                        len(success_scaled_partners),
                    ),
                )
            else:
                self._assign_equal_quantities(
                    proposals,
                    success_scaled_partners,
                    scaled_target,
                    price_getter=self._best_price_for_me,
                )

        elif greedy_partners and scaled_target > 0:
            scaled_target = max(
                int(scaled_target),
                min(2, len(selected_greedy_partners)),
            )

            self._add_equal_quantities(
                proposals,
                selected_greedy_partners,
                scaled_target,
                price_getter=self._worst_price_for_me,
            )

        return proposals

    # ------------------------------------------------------------------
    # OneShot probabilistic counter strategy
    # ------------------------------------------------------------------

    def _oneshot_counter_all(self, offers, states):
        responses = {}

        current_offers = {
            partner: offer
            for partner, offer in offers.items()
            if offer is not None
            and len(offer) > UNIT_PRICE
            and offer[TIME] == self.awi.current_step
        }

        t = self._relative_time(states)

        for needs, all_partners in (
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ):
            side_partners = [
                partner
                for partner in all_partners
                if partner in current_offers
            ]

            if not side_partners:
                continue

            if needs <= 0:
                for partner in side_partners:
                    responses[partner] = self._unneeded_response()
                continue

            side_responses = self._probabilistic_counter_side(
                int(needs),
                side_partners,
                current_offers,
                t,
            )

            responses.update(side_responses)

        # Record our counter offers as normal sent offers.
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

    def _probabilistic_counter_side(
        self,
        needs: int,
        partners: list[str],
        current_offers,
        t: float,
    ):
        best_score = None
        best_accept_set = set()
        best_counters = {}

        partners = list(partners)
        accept_subsets = self._candidate_accept_subsets(
            partners,
            current_offers,
            needs,
        )

        for accept_subset in accept_subsets:
            accept_subset = set(accept_subset)

            accepted_quantity = sum(
                int(current_offers[partner][QUANTITY])
                for partner in accept_subset
            )

            remaining_needs = max(0, int(needs) - accepted_quantity)

            remaining_partners = [
                partner
                for partner in partners
                if partner not in accept_subset
            ]

            accept_value = 0.0
            accept_price_bad_count = 0

            if accept_subset:
                accept_quantities = [
                    int(current_offers[partner][QUANTITY])
                    for partner in accept_subset
                ]
                accept_expected_units = [float(q) for q in accept_quantities]
                accept_prices = {
                    partner: float(current_offers[partner][UNIT_PRICE])
                    for partner in accept_subset
                }

                accept_value = self._rdvo_value(
                    needs,
                    list(accept_subset),
                    accept_quantities,
                    accept_expected_units,
                    t,
                    prices=accept_prices,
                    offered_total=accepted_quantity,
                )

                accept_price_bad_count = sum(
                    1
                    for partner in accept_subset
                    if self._price_for_me_score(
                        partner,
                        current_offers[partner][UNIT_PRICE],
                    )
                    <= 0.20
                )

            if remaining_needs > 0 and remaining_partners:
                (
                    counters,
                    counter_expected_total,
                    counter_value,
                    counter_bad_price_count,
                    counter_offered_total,
                ) = self._best_oneshot_counter_proposals(
                    remaining_needs,
                    remaining_partners,
                    t,
                )
            else:
                counters = {}
                counter_expected_total = 0.0
                counter_value = 0.0
                counter_bad_price_count = 0
                counter_offered_total = 0

            total_expected = float(accepted_quantity) + float(counter_expected_total)
            expected_gap = abs(float(needs) - total_expected)
            expected_over = max(0.0, total_expected - float(needs))
            total_value = accept_value + counter_value

            bad_price_count = accept_price_bad_count + counter_bad_price_count
            offered_gap = abs(float(needs) - float(accepted_quantity + counter_offered_total))

            # OneShot price is basically two-valued:
            # priority is quantity matching, then avoiding expected over,
            # then avoiding my-worst price, then value.
            score = (
                expected_gap,
                expected_over,
                bad_price_count * self.counter_bad_price_penalty,
                -total_value,
                offered_gap,
            )

            if best_score is None or score < best_score:
                best_score = score
                best_accept_set = accept_subset
                best_counters = counters

        responses = {}

        for partner in partners:
            if partner in best_accept_set:
                responses[partner] = SAOResponse(
                    ResponseType.ACCEPT_OFFER,
                    current_offers[partner],
                )
            else:
                offer = best_counters.get(partner)
                if offer is None:
                    responses[partner] = self._unneeded_response()
                else:
                    responses[partner] = self._counter_or_accept_response(
                        partner,
                        current_offers[partner],
                        offer,
                    )

        return responses

    def _candidate_accept_subsets(self, partners, current_offers, needs: int):
        partners = list(partners)
        needs = max(0, int(needs))

        candidates = [set()]

        sorted_single = sorted(
            partners,
            key=lambda partner: (
                abs(int(current_offers[partner][QUANTITY]) - needs),
                -self._price_for_me_score(
                    partner,
                    current_offers[partner][UNIT_PRICE],
                ),
            ),
        )

        for partner in sorted_single[: min(6, len(sorted_single))]:
            candidates.append({partner})

        by_good_price = sorted(
            partners,
            key=lambda partner: (
                -self._price_for_me_score(
                    partner,
                    current_offers[partner][UNIT_PRICE],
                ),
                abs(int(current_offers[partner][QUANTITY]) - needs),
            ),
        )

        total = 0
        subset = set()
        for partner in by_good_price:
            quantity = int(current_offers[partner][QUANTITY])
            if total + quantity <= needs:
                subset.add(partner)
                total += quantity

        if subset:
            candidates.append(subset)

        by_quantity = sorted(
            partners,
            key=lambda partner: int(current_offers[partner][QUANTITY]),
            reverse=True,
        )

        total = 0
        subset = set()
        for partner in by_quantity:
            quantity = int(current_offers[partner][QUANTITY])
            if abs(needs - (total + quantity)) <= abs(needs - total):
                subset.add(partner)
                total += quantity

        if subset:
            candidates.append(subset)

        good_price_partners = [
            partner
            for partner in partners
            if self._price_for_me_score(
                partner,
                current_offers[partner][UNIT_PRICE],
            )
            >= 0.80
        ]

        total = 0
        subset = set()
        for partner in sorted(
            good_price_partners,
            key=lambda partner: int(current_offers[partner][QUANTITY]),
        ):
            quantity = int(current_offers[partner][QUANTITY])
            if total + quantity <= needs:
                subset.add(partner)
                total += quantity

        if subset:
            candidates.append(subset)

        unique = []
        seen = set()
        for subset in candidates:
            key = tuple(sorted(subset))
            if key in seen:
                continue
            seen.add(key)
            unique.append(set(subset))

        return unique[: self.max_accept_subsets]

    def _rank_counter_partners(self, partners):
        return sorted(
            list(partners),
            key=lambda partner: (
                self._price_conditioned_accept_probability(
                    partner,
                    1,
                    self._best_price_for_me(partner),
                ),
                self._partner_non_greedy_initial_offer_success_rate(partner),
                self._mean_reoffer_quantity(partner),
            ),
            reverse=True,
        )

    def _best_oneshot_counter_proposals(
        self,
        needs: int,
        partners: list[str],
        t: float,
    ):
        if not partners or needs <= 0:
            return {}, 0.0, 0.0, 0, 0

        ranked_partners = self._rank_counter_partners(partners)
        active_partners = ranked_partners[: self.max_counter_partners]
        inactive_partners = ranked_partners[self.max_counter_partners :]

        quantity_allocations = self._beam_quantity_allocations_for_counter(
            needs,
            active_partners,
            t,
            beam_width=self.counter_beam_width,
        )

        best_score = None
        best_proposals = {}
        best_expected_total = 0.0
        best_value = 0.0
        best_bad_price_count = 0
        best_offered_total = 0

        for quantities, _, _, offered_total in quantity_allocations:
            price_patterns = self._counter_price_patterns(
                active_partners,
                quantities,
                t,
                needs,
            )

            for prices in price_patterns:
                expected_units = []
                price_dict = {}
                bad_price_count = 0

                for partner, quantity, price in zip(
                    active_partners,
                    quantities,
                    prices,
                    strict=False,
                ):
                    quantity = int(quantity)
                    price = int(price)

                    price_dict[partner] = float(price)

                    if quantity <= 0:
                        expected_units.append(0.0)
                        continue

                    if price == self._worst_price_for_me(partner):
                        bad_price_count += 1

                    expected_units.append(
                        self._oneshot_counter_expected_units(
                            partner,
                            quantity,
                            price,
                        )
                    )

                expected_total = max(0.0, sum(expected_units))
                expected_gap = abs(float(needs) - expected_total)
                expected_over = max(0.0, expected_total - float(needs))

                value = self._rdvo_value(
                    needs,
                    active_partners,
                    quantities,
                    expected_units,
                    t,
                    prices=price_dict,
                    offered_total=offered_total,
                )

                offered_gap = abs(float(needs) - float(offered_total))

                score = (
                    expected_gap,
                    expected_over,
                    bad_price_count * self.counter_bad_price_penalty,
                    -value,
                    offered_gap,
                )

                if best_score is None or score < best_score:
                    best_score = score
                    best_expected_total = expected_total
                    best_value = value
                    best_bad_price_count = bad_price_count
                    best_offered_total = offered_total

                    proposals = {}

                    for partner, quantity, price in zip(
                        active_partners,
                        quantities,
                        prices,
                        strict=False,
                    ):
                        quantity = int(quantity)

                        if quantity <= 0:
                            proposals[partner] = None
                        else:
                            proposals[partner] = self._offer(
                                partner,
                                quantity,
                                int(price),
                            )

                    for partner in inactive_partners:
                        proposals[partner] = None

                    best_proposals = proposals

        return (
            best_proposals,
            best_expected_total,
            best_value,
            best_bad_price_count,
            best_offered_total,
        )

    def _beam_quantity_allocations_for_counter(
        self,
        needs: int,
        partners: list[str],
        t: float,
        *,
        beam_width: int | None = None,
    ):
        needs = max(0, int(needs))
        partners = list(partners)

        if beam_width is None:
            beam_width = self.counter_beam_width

        beam = [([], [], 0.0, 0)]

        for partner in partners:
            candidates = self._quantity_candidates_for_oneshot_counter(
                needs,
                partner,
            )

            next_beam = []

            for (
                quantities_so_far,
                expected_units_so_far,
                expected_total,
                offered_total,
            ) in beam:
                for quantity in candidates:
                    quantity = int(quantity)
                    price = self._best_price_for_me(partner)

                    expected = self._oneshot_counter_expected_units(
                        partner,
                        quantity,
                        price,
                    )

                    new_quantities = quantities_so_far + [quantity]
                    new_expected_units = expected_units_so_far + [expected]
                    new_expected_total = expected_total + expected
                    new_offered_total = offered_total + max(0, quantity)

                    if self.max_total_offer_multiplier > 0:
                        cap = math.ceil(needs * self.max_total_offer_multiplier)
                        if new_offered_total > cap:
                            continue

                    if new_expected_total > needs * 1.5 + 2:
                        continue

                    next_beam.append(
                        (
                            new_quantities,
                            new_expected_units,
                            new_expected_total,
                            new_offered_total,
                        )
                    )

            def partial_key(item):
                _, _, expected_total, offered_total = item
                expected_over = max(0.0, expected_total - float(needs))
                offered_over = max(0, offered_total - needs)
                return (
                    expected_over,
                    offered_over,
                    abs(float(needs) - expected_total) * 0.2,
                    offered_total,
                )

            next_beam.sort(key=partial_key)
            beam = next_beam[:beam_width]

            if not beam:
                break

        def final_key(item):
            _, _, expected_total, offered_total = item
            expected_gap = abs(float(needs) - expected_total)
            expected_over = max(0.0, expected_total - float(needs))
            offered_gap = abs(float(needs) - float(offered_total))
            return (
                expected_gap,
                expected_over,
                offered_gap,
            )

        beam.sort(key=final_key)
        return beam

    def _counter_price_patterns(
        self,
        partners: list[str],
        quantities: list[int],
        t: float,
        remaining_needs: int,
    ):
        best_prices = [
            self._best_price_for_me(partner)
            for partner in partners
        ]

        patterns = [best_prices]

        t = max(0.0, min(1.0, float(t)))
        shortage_ratio = float(remaining_needs) / max(1.0, float(self._max_lines()))

        allow_worst = (
            t >= self.counter_good_price_time
            or shortage_ratio >= self.counter_good_price_shortage_ratio
        )

        if not allow_worst:
            return patterns

        all_worst = [
            self._worst_price_for_me(partner)
            if int(quantity) > 0
            else self._best_price_for_me(partner)
            for partner, quantity in zip(partners, quantities, strict=False)
        ]
        patterns.append(all_worst)

        active_indices = [
            index
            for index, quantity in enumerate(quantities)
            if int(quantity) > 0
        ]

        if active_indices:
            weakest = min(
                active_indices,
                key=lambda index: self._price_conditioned_accept_probability(
                    partners[index],
                    quantities[index],
                    self._best_price_for_me(partners[index]),
                ),
            )

            one_worst = list(best_prices)
            one_worst[weakest] = self._worst_price_for_me(partners[weakest])
            patterns.append(one_worst)

        greedy_indices = [
            index
            for index, partner in enumerate(partners)
            if int(quantities[index]) > 0
            and self.opponent_type(partner) == "GreedyOneShotAgent"
        ]

        if greedy_indices:
            greedy_worst = list(best_prices)
            for index in greedy_indices:
                greedy_worst[index] = self._worst_price_for_me(partners[index])
            patterns.append(greedy_worst)

        unique = []
        seen = set()
        for pattern in patterns:
            key = tuple(pattern)
            if key in seen:
                continue
            seen.add(key)
            unique.append(pattern)

        return unique

    def _quantity_candidates_for_oneshot_counter(
        self,
        needs: int,
        partner,
    ) -> list[int]:
        qmax = max(
            1,
            int(self._issues_for(partner)[QUANTITY].max_value),
        )

        upper = min(max(0, int(needs)), qmax)
        return list(range(0, upper + 1))

    def _price_conditioned_accept_probability(
        self,
        partner,
        quantity: int,
        price: float,
    ) -> float:
        label = self._opponent_price_label(partner, price)

        trials = self._first_offer_trials_by_price[partner][label]
        accepts = self._first_offer_accepts_by_price[partner][label]

        if trials > 0:
            base = (accepts + 1.0) / (trials + 2.0)
        else:
            base = self._accept_probability(partner, quantity, price)

        return max(0.02, min(0.98, base))

    def _price_conditioned_reoffer_probability(
        self,
        partner,
        price: float,
    ) -> float:
        label = self._opponent_price_label(partner, price)

        trials = self._first_offer_trials_by_price[partner][label]
        accepts = self._first_offer_accepts_by_price[partner][label]
        rejected = max(0, trials - accepts)
        reoffers = self._first_offer_reoffers_by_price[partner][label]

        if trials > 0:
            return (reoffers + 1.0) / (rejected + 2.0)

        return self._reoffer_probability(partner)

    def _price_conditioned_mean_reoffer_quantity(
        self,
        partner,
        price: float,
    ) -> float:
        label = self._opponent_price_label(partner, price)
        quantities = self._first_offer_counter_quantities_by_price[partner][label]

        if quantities:
            recent = quantities[-20:]
            return sum(recent) / len(recent)

        return self._mean_reoffer_quantity(partner)

    def _oneshot_counter_expected_units(
        self,
        partner,
        quantity: int,
        price: float,
    ) -> float:
        if quantity <= 0:
            return 0.0

        p = self._price_conditioned_accept_probability(
            partner,
            quantity,
            price,
        )
        r = self._price_conditioned_reoffer_probability(
            partner,
            price,
        )
        mu = self._price_conditioned_mean_reoffer_quantity(
            partner,
            price,
        )

        return p * quantity + (1.0 - p) * r * mu

    # ------------------------------------------------------------------
    # Success-rate utilities
    # ------------------------------------------------------------------

    def _success_adjusted_quantity(
        self,
        target_quantity: float,
        success_rate: float,
    ) -> int:
        success_rate = max(0.05, min(1.0, float(success_rate)))
        return math.ceil(float(target_quantity) / success_rate)

    def _non_greedy_initial_offer_success_rate(self) -> float:
        results = []

        for partner, history in self._own_first_offer_result_history.items():
            if self.opponent_type(partner) == "GreedyOneShotAgent":
                continue

            results.extend(
                bool(item.get("accepted", False))
                for item in history
            )

        if not results:
            return self.non_greedy_success_default

        return sum(results) / len(results)

    def _partner_non_greedy_initial_offer_success_rate(self, partner) -> float:
        history = self._own_first_offer_result_history.get(partner, [])

        if not history:
            return self._non_greedy_initial_offer_success_rate()

        return sum(
            bool(item.get("accepted", False))
            for item in history
        ) / len(history)

    def _strong_initial_greedy_partner(self, partner) -> bool:
        greedy_probability = self.opponent_posteriors(partner).get(
            "GreedyOneShotAgent",
            0.0,
        )

        if greedy_probability < 0.80:
            return False

        streak = 0

        for item in reversed(self._own_first_offer_result_history.get(partner, [])):
            if not item.get("accepted", False):
                break

            streak += 1

            if streak >= 3:
                return True

        return False

    # ------------------------------------------------------------------
    # Quantity assignment helpers
    # ------------------------------------------------------------------

    def _half_quantity_caps(self, needs: int, count: int):
        if needs <= 0 or count <= 0:
            return []

        low = int(needs) // 2
        high = int(needs) - low

        if count == 1:
            return [high]

        return [
            low if index % 2 == 0 else high
            for index in range(count)
        ]

    def _assign_equal_quantities(
        self,
        proposals,
        partners,
        target_quantity,
        price_getter,
    ):
        partners = list(partners)

        if not partners or target_quantity <= 0:
            return

        quantities = self._distribute_quantity(
            int(target_quantity),
            len(partners),
            mx=self._max_lines(),
            equal=True,
            allow_zero=self.awi.allow_zero_quantity,
        )

        for partner, quantity in zip(partners, quantities, strict=False):
            if quantity <= 0 and not self.awi.allow_zero_quantity:
                continue

            proposals[partner] = self._offer(
                partner,
                quantity,
                price_getter(partner),
            )

    def _assign_success_weighted_quantities(
        self,
        proposals,
        partners,
        target_quantity,
        price_getter,
        quantity_caps=None,
    ):
        partners = list(partners)

        if not partners or target_quantity <= 0:
            return

        target_quantity = int(target_quantity)

        if quantity_caps is not None:
            quantity_caps = [
                max(0, int(cap))
                for cap in list(quantity_caps)[: len(partners)]
            ]

            if len(quantity_caps) < len(partners):
                quantity_caps.extend([0] * (len(partners) - len(quantity_caps)))

        weights = [
            0.75 + 0.5 * self._partner_non_greedy_initial_offer_success_rate(partner)
            for partner in partners
        ]

        total_weight = sum(weights)

        if total_weight <= 0:
            self._assign_equal_quantities(
                proposals,
                partners,
                target_quantity,
                price_getter,
            )
            return

        raw_quantities = [
            target_quantity * weight / total_weight
            for weight in weights
        ]

        quantities = []

        for index, quantity in enumerate(raw_quantities):
            assigned = math.floor(quantity)

            if quantity_caps is not None:
                assigned = min(assigned, quantity_caps[index])

            quantities.append(assigned)

        remainder = target_quantity - sum(quantities)

        order = sorted(
            range(len(partners)),
            key=lambda index: (
                raw_quantities[index] - quantities[index],
                weights[index],
            ),
            reverse=True,
        )

        while remainder > 0:
            changed = False

            for index in order:
                if quantity_caps is not None and quantities[index] >= quantity_caps[index]:
                    continue

                quantities[index] += 1
                remainder -= 1
                changed = True

                if remainder <= 0:
                    break

            if not changed:
                break

        for partner, quantity in zip(partners, quantities, strict=False):
            if quantity <= 0 and not self.awi.allow_zero_quantity:
                continue

            proposals[partner] = self._offer(
                partner,
                quantity,
                price_getter(partner),
            )

    def _add_equal_quantities(
        self,
        proposals,
        partners,
        target_quantity,
        price_getter,
    ):
        partners = list(partners)

        if not partners or target_quantity <= 0:
            return

        quantities = self._distribute_quantity(
            int(target_quantity),
            len(partners),
            mx=self._max_lines(),
            equal=True,
            allow_zero=self.awi.allow_zero_quantity,
        )

        for partner, addition in zip(partners, quantities, strict=False):
            if addition <= 0 and not self.awi.allow_zero_quantity:
                continue

            current = proposals.get(partner)
            current_quantity = int(current[QUANTITY]) if current is not None else 0

            proposals[partner] = self._offer(
                partner,
                current_quantity + int(addition),
                price_getter(partner),
            )

    def _is_process_one_agent(self) -> bool:
        return str(getattr(self, "id", "")).endswith("@1")

    def _distribute_quantity(
        self,
        quantity: int,
        count: int,
        *,
        mx: int | None = None,
        equal: bool = False,
        concentrated: bool = False,
        allow_zero: bool = False,
        concentrated_idx: list[int] | None = None,
    ) -> list[int]:
        quantity = max(0, int(quantity))
        count = max(0, int(count))

        if count <= 0:
            return []

        if mx is not None:
            mx = max(0, int(mx))
            quantity = min(quantity, mx * count)

        if quantity <= 0:
            return [0] * count

        if concentrated:
            indices = list(concentrated_idx or [])
            indices += [
                index
                for index in range(count)
                if index not in indices
            ]

            values = [0] * count
            remaining = quantity

            if not allow_zero:
                for index in indices[: min(count, remaining)]:
                    values[index] = 1
                    remaining -= 1

            for index in indices:
                if remaining <= 0:
                    break

                room = remaining if mx is None else max(0, mx - values[index])
                add = min(room, remaining)
                values[index] += add
                remaining -= add

            return values

        if quantity < count and not allow_zero:
            values = [1] * quantity + [0] * (count - quantity)
            random.shuffle(values)
            return values

        if equal:
            base = quantity // count
        else:
            base = 0 if allow_zero else 1

        values = [base] * count
        remaining = quantity - base * count
        indices = list(range(count))
        random.shuffle(indices)

        while remaining > 0:
            progressed = False

            for index in indices:
                if remaining <= 0:
                    break

                if mx is not None and values[index] >= mx:
                    continue

                values[index] += 1
                remaining -= 1
                progressed = True

            if not progressed:
                break

        return values

    # ------------------------------------------------------------------
    # Optional cautious helpers kept for compatibility/debugging
    # ------------------------------------------------------------------

    def _cautious_active_partners(self, all_partners):
        active, inactive = [], []
        cutoff_step = min(self.awi.n_steps * 0.5, 50)

        for partner in all_partners:
            if partner not in self.negotiators:
                continue

            if self.awi.is_bankrupt(partner) or (
                self.awi.current_step > cutoff_step
                and self.total_agreed_quantity.get(partner, 0) == 0
            ):
                inactive.append(partner)
            else:
                active.append(partner)

        return active, inactive

    def _cautious_first_side_proposals(
        self,
        needs: int,
        partners: list[str],
    ):
        if not partners:
            return {}

        if needs <= 0:
            return {partner: None for partner in partners}

        distribution = self._cautious_distribution(
            int(needs),
            partners,
            t=0.0,
        )

        return {
            partner: self._offer(
                partner,
                quantity,
                self._best_price_for_me(partner),
            )
            if quantity > 0 or self.awi.allow_zero_quantity
            else None
            for partner, quantity in distribution.items()
        }

    def _cautious_distribution(
        self,
        needs: int,
        partners: list[str],
        t: float,
    ):
        if not partners:
            return {}

        if needs <= 0:
            return {partner: 0 for partner in partners}

        quantity = (
            int(needs * (1 + self._overordering_fraction(t)))
            if len(partners) > 1
            else int(needs)
        )

        if self.awi.current_step > self.awi.n_steps * 0.5:
            concentrated_ids = sorted(
                partners,
                key=lambda partner: self.total_agreed_quantity.get(partner, 0),
                reverse=True,
            )[:1]

            concentrated_idx = [
                index
                for index, partner in enumerate(partners)
                if partner in concentrated_ids
            ]

            quantities = self._distribute_quantity(
                quantity,
                len(partners),
                mx=self._max_lines(),
                concentrated=True,
                concentrated_idx=concentrated_idx,
                allow_zero=self.awi.allow_zero_quantity,
            )

        else:
            quantities = self._distribute_quantity(
                quantity,
                len(partners),
                mx=self._max_lines(),
                equal=self.equal_distribution,
                allow_zero=self.awi.allow_zero_quantity,
            )

        return dict(zip(partners, quantities, strict=False))

    def _best_offer_subset(
        self,
        partners,
        offers,
        needs: int,
        is_selling: bool,
        relative_time: float,
    ):
        if not partners:
            return None

        subsets = list(self._powerset(partners))[::-1]
        plus_best = None
        minus_best = None

        for partner_ids in subsets:
            offered = sum(
                int(offers[partner][QUANTITY])
                for partner in partner_ids
            )
            diff = offered - needs
            price_sum = sum(
                float(offers[partner][UNIT_PRICE])
                for partner in partner_ids
            )
            size = len(partner_ids)
            partner_ids = tuple(partner_ids)

            if diff >= 0:
                candidate = (
                    diff,
                    self._price_tiebreaker(price_sum, is_selling),
                    partner_ids,
                )

                if plus_best is None or candidate < plus_best:
                    plus_best = candidate

            if diff <= 0:
                candidate = (
                    -diff,
                    size if diff < 0 else 0,
                    self._price_tiebreaker(price_sum, is_selling),
                    diff,
                    partner_ids,
                )

                if minus_best is None or candidate < minus_best:
                    minus_best = candidate

        th_min_minus, th_max_minus = self._allowed_mismatch(
            relative_time,
            0,
            is_selling,
        )

        minus_allowed = False
        plus_allowed = False
        minus_diff = 0
        minus_partners = ()
        plus_diff = 0
        plus_partners = ()

        if minus_best is not None:
            minus_diff = int(minus_best[3])
            minus_partners = minus_best[4]
            th_min_minus, _ = self._allowed_mismatch(
                relative_time,
                len(partners.difference(minus_partners)),
                is_selling,
            )
            minus_allowed = th_min_minus <= minus_diff

        if plus_best is not None:
            plus_diff = int(plus_best[0])
            plus_partners = plus_best[2]
            _, th_max_plus = self._allowed_mismatch(
                relative_time,
                len(partners.difference(plus_partners)),
                is_selling,
            )
            plus_allowed = plus_diff <= th_max_plus

        if not minus_allowed and not plus_allowed:
            return None

        if minus_allowed and plus_allowed:
            if -minus_diff == plus_diff:
                return (
                    (minus_diff, set(minus_partners))
                    if is_selling
                    else (plus_diff, set(plus_partners))
                )

            if -minus_diff < plus_diff:
                return minus_diff, set(minus_partners)

            return plus_diff, set(plus_partners)

        if plus_allowed:
            return plus_diff, set(plus_partners)

        return minus_diff, set(minus_partners)

    def _price_tiebreaker(self, price_sum: float, is_selling: bool) -> float:
        return -price_sum if is_selling else price_sum

    def _powerset(self, iterable):
        items = list(iterable)
        return chain.from_iterable(
            combinations(items, size)
            for size in range(len(items) + 1)
        )

    def _allowed_mismatch(
        self,
        relative_time: float,
        n_others: int,
        is_selling: bool,
    ):
        del n_others

        relative_time = max(0.0, min(1.0, float(relative_time)))

        undermismatch_min = (
            self._undermismatch_min_selling_abs
            if is_selling
            else self._undermismatch_min_buying_abs
        )

        overmismatch_max = (
            self._overmismatch_max_selling_abs
            if is_selling
            else self._overmismatch_max_buying_abs
        )

        return (
            undermismatch_min * ((1.0 - relative_time) ** self.mismatch_exp),
            overmismatch_max * (relative_time ** (1.0 / self.mismatch_exp)),
        )

    def _overordering_fraction(self, t: float):
        t = max(0.0, min(1.0, float(t)))
        return self.overordering_max - (
            self.overordering_max - self.overordering_min
        ) * (t**self.overordering_exp)

    def _relative_time(self, states) -> float:
        values = [
            float(getattr(state, "relative_time", 0.0))
            for state in states.values()
            if getattr(state, "relative_time", None) is not None
        ]

        return min(values, default=0.0)

    # ------------------------------------------------------------------
    # Negotiation callbacks
    # ------------------------------------------------------------------

    def on_negotiation_success(self, contract: Contract, mechanism: Any):
        super().on_negotiation_success(contract, mechanism)

        partner = self._contract_partner(contract)
        outcome = self._contract_outcome(contract)

        if partner is None or outcome is None:
            return

        self.total_agreed_quantity[partner] = (
            self.total_agreed_quantity.get(partner, 0)
            + int(outcome[QUANTITY])
        )

        sent_offer = self._matching_first_offer(partner, outcome)

        if sent_offer is not None:
            self._observe_own_first_offer_result(
                partner,
                sent_offer,
                accepted=True,
            )

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

        sent_offer = self._latest_unobserved_first_offer(partner)

        if sent_offer is not None:
            self._observe_own_first_offer_result(
                partner,
                sent_offer,
                accepted=False,
            )

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

    def _same_quantity_offer(self, left, right) -> bool:
        if left is None or right is None:
            return False

        return int(left[QUANTITY]) == int(right[QUANTITY])


BayesianSyncRandomAgent2 = BayesianAgent2
