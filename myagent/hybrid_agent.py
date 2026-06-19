from __future__ import annotations

"""Hybrid agent that switches negotiation behaviour by environment.

- In a **Greedy** environment (most classified opponents are greedy) it behaves
  like :class:`BayesianAgent022` (its Bayesian classifier + greedy-aware
  offer/counter logic).
- In a **NonGreedy** environment it delegates offers/counters to
  :class:`CautiousOneShotAgent`.

The Bayesian classifier of :class:`BayesianAgent022` is always kept running
(every ``counter_all`` observes the incoming offers) so the environment can be
detected regardless of which "brain" currently produces the responses.

It also overrides the three subset-search helpers of :class:`BayesianAgent022`
(``_has_exact_offer_subset`` / ``_eighty_percent_acceptance_subset`` /
``_max_under_needs_subset_for_seller``).  The originals enumerate
``itertools.combinations`` over all partners, which is O(2^n).  Here they are
replaced by an equivalent subset-sum DP keyed by the offered quantity, which is
polynomial (O(n * needs)) because the needed quantity is bounded by the
production capacity.
"""

import math

from negmas import Contract, SAOResponse, ResponseType
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE

try:  # package import (``import myagent.hybrid_agent``)
    from .bayesian_agent_022 import BayesianAgent022
    from .cautious import CautiousOneShotAgent
except ImportError:  # flat import (``sys.path`` includes ``myagent``)
    from bayesian_agent_022 import BayesianAgent022
    from cautious import CautiousOneShotAgent


class HybridBayesianCautiousAgent(BayesianAgent022):
    """Greedy env -> BayesianAgent022, NonGreedy env -> CautiousOneShotAgent."""

    def __init__(
        self,
        *args,
        greedy_env_threshold: float = 0.5,
        min_env_classified: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # share of classified active partners that must be greedy for the world
        # to count as a "greedy environment".
        self.greedy_env_threshold = float(greedy_env_threshold)
        # require at least this many *classified* (non-Unknown) partners before
        # trusting an environment decision; until then keep the last decision.
        self.min_env_classified = int(min_env_classified)

    # ------------------------------------------------------------------
    # Lifecycle: keep an embedded Cautious agent sharing our AWI/negotiators
    # ------------------------------------------------------------------

    def init(self):
        super().init()
        cautious = CautiousOneShotAgent()
        # Share the simulation context so the embedded agent reads the same
        # world state and the same live set of negotiators as we do.
        cautious._awi = self._awi
        cautious._negotiators = self._negotiators
        cautious.init()
        self._cautious = cautious
        self._env_greedy_last = False

    def on_negotiation_success(self, contract: Contract, mechanism):
        super().on_negotiation_success(contract, mechanism)
        # Port CautiousOneShotAgent's per-partner volume tracking using *our*
        # id (the embedded instance has a different id, so we cannot forward the
        # callback directly without it mis-identifying the partner).
        try:
            partner_id = next(p for p in contract.partners if p != self.id)
        except StopIteration:
            return
        tracker = getattr(self._cautious, "total_agreed_quantity", None)
        if tracker is not None and partner_id in tracker:
            tracker[partner_id] += int(contract.agreement["quantity"])

    # ------------------------------------------------------------------
    # Environment detection and brain selection
    # ------------------------------------------------------------------

    def _environment_is_greedy(self) -> bool:
        partners = self._active_partners()
        greedy = 0
        classified = 0
        for partner in partners:
            opponent_type = self.opponent_type(partner)
            if opponent_type == "GreedyOneShotAgent":
                greedy += 1
                classified += 1
            elif opponent_type == "NonGreedy":
                classified += 1
        if classified < max(1, self.min_env_classified):
            return self._env_greedy_last
        self._env_greedy_last = (greedy / classified) >= self.greedy_env_threshold
        return self._env_greedy_last

    def _use_bayesian_brain(self) -> bool:
        # During the exploration window always use the Bayesian brain: its
        # alternating-price probes are what make the classification (and hence
        # the environment detection) work in the first place.
        if self._exploration_enabled():
            return True
        return self._environment_is_greedy()

    # ------------------------------------------------------------------
    # Offer / counter routing
    # ------------------------------------------------------------------

    def first_proposals(self):
        if self._use_bayesian_brain():
            return super().first_proposals()
        proposals = self._cautious.first_proposals()
        # Record what we sent so the classifier can still match accept/counter
        # results against our offers on the next round.
        self._record_sent_offers(proposals)
        return proposals

    def counter_all(self, offers, states):
        # Always feed the classifier, regardless of which brain answers.
        self._observe_incoming_offers(offers, states)

        if self._use_bayesian_brain():
            return self._current_offer_responses(offers, states)

        responses = self._cautious.counter_all(offers, states)
        self._record_response_offers(
            responses,
            relative_time=self._relative_time(states),
        )
        return responses

    def _observe_incoming_offers(self, offers, states):
        """The observation pass from ``BayesianAgent022.counter_all`` extracted so
        it runs even when the Cautious brain produces the responses."""
        current_offers = {
            partner: offer
            for partner, offer in offers.items()
            if offer is not None
        }
        for partner, offer in current_offers.items():
            if len(offer) <= UNIT_PRICE or offer[TIME] != self.awi.current_step:
                continue
            sent_offer = self._latest_unobserved_first_offer(partner)
            if sent_offer is not None and not self._same_offer(sent_offer["offer"], offer):
                self._observe_own_first_offer_counter(partner, sent_offer, offer)
            self._observe_partner_end_response(
                partner,
                self._latest_sent_offer(partner),
                ended=False,
            )
            self._observe_received_first_offer(partner, offer)
            self._observe_offer(partner, offer, states)

    # ------------------------------------------------------------------
    # Polynomial replacements for the exponential subset searches
    # ------------------------------------------------------------------

    def _reachable_quantity_subsets(self, partners, offers, cap: int, better=None):
        """Subset-sum reachability with one witness subset per reachable total.

        Returns ``{total: tuple(partner_ids)}`` for every total in ``0..cap`` that
        some subset of ``partners`` can sum to (0/1 knapsack, each partner used at
        most once).  ``better(candidate, current)`` -> ``True`` replaces the stored
        witness for a tie on ``total`` (used for price / size tie-breaking).

        Runs in O(len(partners) * cap), replacing the O(2^n) ``combinations``
        enumeration of the original helpers.
        """
        cap = int(cap)
        reachable = {0: ()}
        if cap < 0:
            return reachable
        for partner in partners:
            quantity = max(0, int(offers[partner][QUANTITY]))
            nxt = dict(reachable)
            for total, subset in reachable.items():
                new_total = total + quantity
                if new_total > cap:
                    continue
                candidate = subset + (partner,)
                current = nxt.get(new_total)
                if current is None or (better is not None and better(candidate, current)):
                    nxt[new_total] = candidate
            reachable = nxt
        return reachable

    def _has_exact_offer_subset(self, partners, offers, needs: int) -> bool:
        needs = int(needs)
        if needs <= 0:
            return False
        reachable = self._reachable_quantity_subsets(partners, offers, needs)
        subset = reachable.get(needs)
        return subset is not None and len(subset) > 0

    def _eighty_percent_acceptance_subset(self, partners, offers, needs: int):
        partners = list(partners)
        n_partners = len(partners)
        max_accept_partners = n_partners - 1
        needs = int(needs)
        if needs <= 0 or max_accept_partners <= 0:
            return None

        target = min(needs, max(1, math.ceil(needs * 0.8)))

        def better(candidate, current):
            # smallest subset first, then lexicographically smallest ids
            return (len(candidate), candidate) < (len(current), current)

        # cap at needs-1 so totals that reach exactly ``needs`` are excluded (the
        # exact-match case is handled separately by the caller).
        reachable = self._reachable_quantity_subsets(
            partners, offers, needs - 1, better
        )
        # highest reachable total => smallest gap (needs - offered)
        for total in range(needs - 1, target - 1, -1):
            subset = reachable.get(total)
            if subset and 1 <= len(subset) <= max_accept_partners:
                return subset
        return None

    def _max_under_needs_subset_for_seller(self, partners, offers, needs: int):
        partners = list(partners)
        needs = int(needs)
        if needs <= 0:
            return tuple()

        def price_value(subset):
            return sum(
                int(offers[p][QUANTITY]) * float(offers[p][UNIT_PRICE])
                for p in subset
            )

        def better(candidate, current):
            # higher revenue, then fewer partners, then larger id tuple
            key_candidate = (price_value(candidate), -len(candidate), candidate)
            key_current = (price_value(current), -len(current), current)
            return key_candidate > key_current

        reachable = self._reachable_quantity_subsets(partners, offers, needs, better)
        # maximise the total accepted quantity (<= needs)
        best_total = max(reachable)
        return reachable[best_total]
