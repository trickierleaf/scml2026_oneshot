from __future__ import annotations

"""Self-contained hybrid agent.

Routing: delegates offers/counters to the bundled ``CautiousOneShotAgent`` only
on the **buyer side of a NonGreedy environment** (empirically the only segment
where Cautious beats BA022); everywhere else (seller side, or any greedy
environment) it uses the bundled ``BayesianAgent022`` brain.  The Bayesian
classifier always runs so the environment can be detected regardless of which
brain answers.

``BayesianAgent022`` and ``CautiousOneShotAgent`` (and the ``distribute`` /
``powerset`` helpers Cautious needs) are bundled below so this module has no
dependency on the sibling agent files.  ``BayesianAgent022`` already uses a
polynomial (O(n*needs)) subset-sum search instead of the original exponential
``itertools.combinations`` enumeration.
"""

import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from negmas import Contract, Outcome, SAOResponse, ResponseType
from scml.oneshot import OneShotSyncAgent, QUANTITY, TIME, UNIT_PRICE
from scml.oneshot.agents.rand import SyncRandomOneShotAgent


def distribute(
    q: int,
    n: int,
    *,
    mx: int | None = None,
    equal=False,
    concentrated=False,
    allow_zero=False,
    concentrated_idx: list[int] = [],
) -> list[int]:
    """Distributes q values over n bins.

    Args:
        q: Quantity to distribute
        n: number of bins to distribute q over
        mx: Maximum allowed per bin. `None` for no limit
        equal: Try to make the values in each bins as equal as possible
        concentrated: If true, will try to concentrate offers in few bins. `mx` must be passed in this case
        allow_zero: Allow some bins to be zero even if that is not necessary
    """
    from collections import Counter

    from numpy.random import choice

    q, n = int(q), int(n)

    if mx is not None and q > mx * n:
        q = mx * n

    if concentrated:
        assert mx is not None
        lst = [0] * n
        if not allow_zero:
            for i in range(min(q, n)):
                lst[i] = 1
        q -= sum(lst)
        if q == 0:
            random.shuffle(lst)
            return lst
        for i in range(n):
            q += lst[i]
            lst[i] = min(mx, q)
            q -= lst[i]
        concentrated_lst = sorted(lst, reverse=True)[: len(concentrated_idx)]
        for x in concentrated_lst:
            lst.remove(x)
        random.shuffle(lst)
        for i, x in zip(concentrated_idx, concentrated_lst):
            lst.insert(i, x)
        # print(lst,concentrated_idx)
        return lst

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n
    if allow_zero:
        per = 0
    else:
        per = (q // n) if equal else 1
    q -= per * n
    r = Counter(choice(n, q))
    return [r.get(_, 0) + per for _ in range(n)]


def powerset(iterable):
    """冪集合"""
    from itertools import chain, combinations

    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))




class CautiousOneShotAgent(OneShotSyncAgent):
    """
    An agent that distributes its needs over its partners randomly.

    Args:
        equal: If given, it tries to equally distribute its needs over as many of its
               suppliers/consumers as possible
        overordering_max: Maximum fraction of needs to over-order. For example, if the
                          agent needs 5 items and this is 0.2, it will order 6 in the first
                          negotiation step.
        overordering_min: Minimum fraction of needs to over-order. Used in the last negotiation step.
        overordering_exp: Controls how fast does the over-ordering quantity go from max to min.
        concession_exp: Controls how fast does the agent concedes on matching its needs exactly.
        mismatch_max: Maximum mismtach in quantity allowed between needs and accepted offers. If
                      a fraction, it is will be this fraction of the production capacity (n_lines).
        overmismatch_max: 相手から受けた提案の総量が自身の必要取引量を超える場合に許容する過剰量(n_linesに対する割合で付与)。
        undermismatching_min_selling: 自身が売り手のときに、相手から受けた提案の総量が自身の必要取引量に満たない場合に許容する不足量(n_linesに対する割合で付与)。
        undermismatching_min_buying: 自身が買い手のときに、相手から受けた提案の総量が自身の必要取引量に満たない場合に許容する不足量(n_linesに対する割合で付与)。
    """

    def __init__(
        self,
        *args,
        equal: bool = False,
        overordering_max_selling: float = 0.0,
        overordering_max_buying: float = 0.2,
        overordering_min: float = 0.0,
        overordering_exp: float = 0.4,
        mismatch_exp: float = 4.0,
        overmismatch_max_selling: float = 0,
        overmismatch_max_buying: float = 0.3,
        undermismatch_min_selling: float = -0.4,
        undermismatch_min_buying: float = -0.2,
        **kwargs,
    ):
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
        super().__init__(*args, **kwargs)

    def init(self):
        self.overordering_max = (
            self.overordering_max_selling
            if self.awi.my_suppliers == ["SELLER"]
            else self.overordering_max_buying
        )
        self.overmismatch_max_selling *= self.awi.n_lines
        self.overmismatch_max_buying *= self.awi.n_lines
        self.undermismatch_min_selling *= self.awi.n_lines
        self.undermismatch_min_buying *= self.awi.n_lines

        # 各ラウンドでの相手一人あたりの(擬似的な)提案個数
        self.rounds_ave_offered = (
            [self.awi.n_lines / len(self.awi.my_consumers)]
            + [self.awi.n_lines / 2 / len(self.awi.my_consumers)] * 9
            + [1] * 10
            if self.awi.my_suppliers == ["SELLER"]
            else [self.awi.n_lines / len(self.awi.my_suppliers)]
            + [self.awi.n_lines / 2 / len(self.awi.my_suppliers)] * 9
            + [1] * 10
        )

        self.total_agreed_quantity = {
            k: 0
            for k in (
                self.awi.my_consumers
                if self.awi.my_suppliers == ["SELLER"]
                else self.awi.my_suppliers
            )
        }
        return super().init()

    def distribute_needs(
        self,
        t: float,
        mx: int | None = None,
        equal: bool | None = None,
        allow_zero: bool | None = None,
        concentrated: bool = False,
        concentrated_ids: list[str] = [],
    ) -> dict[str, int]:
        """Distributes my needs randomly over all my partners"""

        if equal is None:
            equal = self.equal_distribution
        if allow_zero is None:
            allow_zero = self.awi.allow_zero_quantity

        dist = dict()
        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            # find suppliers and consumers still negotiating with me
            # partners = [_ for _ in all_partners if _ in self.negotiators.keys()]
            # n_partners = len(partners)
            partners, n_partners = [], 0
            concentrated_idx = []
            for p in all_partners:
                if p not in self.negotiators.keys():
                    continue
                partners.append(p)
                if p in concentrated_ids:
                    concentrated_idx.append(n_partners)
                n_partners += 1

            # if I need nothing, end all negotiations
            if needs <= 0:
                dist.update(dict(zip(partners, [0] * n_partners)))
                continue

            # distribute my needs over my (remaining) partners.
            offering_quanitity = (
                int(needs * (1 + self._overordering_fraction(t)))
                if len(partners) > 1
                else needs
            )
            dist.update(
                dict(
                    zip(
                        partners,
                        distribute(
                            offering_quanitity,
                            n_partners,
                            mx=mx,
                            equal=equal,
                            concentrated=concentrated,
                            allow_zero=allow_zero,
                            concentrated_idx=concentrated_idx,
                        ),
                    )
                )
            )
        return dist

    def on_negotiation_success(self, contract: Contract, mechanism) -> None:
        super().on_negotiation_success(contract, mechanism)
        partner_id = [p for p in contract.partners if p != self.id][0]
        self.total_agreed_quantity[partner_id] += contract.agreement["quantity"]

    def first_proposals(self):
        # just randomly distribute my needs over my partners (with best price for me).
        s, price = self._step_and_price(best_price=True)
        # my_negotiators = [p for p in (self.awi.my_consumers if self.awi.my_suppliers==["SELLER"] else self.awi.my_suppliers) if p in self.negotiators.keys()]
        my_negotiators, not_negotiators = [], []
        if self.awi.my_suppliers == ["SELLER"]:
            for k in self.awi.my_consumers:
                if self.awi.is_bankrupt(k) or (
                    self.awi.current_step > min(self.awi.n_steps * 0.5, 50)
                    and self.total_agreed_quantity[k] == 0
                ):
                    not_negotiators.append(k)
                else:
                    my_negotiators.append(k)
            offering_quantity = (
                int(self.awi.needed_sales * (1 + self._overordering_fraction(0)))
                if len(my_negotiators) > 1
                else self.awi.needed_sales
            )
        else:
            for k in self.awi.my_suppliers:
                if self.awi.is_bankrupt(k) or (
                    self.awi.current_step > min(self.awi.n_steps * 0.5, 50)
                    and self.total_agreed_quantity[k] == 0
                ):
                    not_negotiators.append(k)
                else:
                    my_negotiators.append(k)
            offering_quantity = (
                int(self.awi.needed_supplies * (1 + self._overordering_fraction(0)))
                if len(my_negotiators) > 1
                else self.awi.needed_supplies
            )

        d = {}
        if len(my_negotiators) > 0:
            if (
                self.awi.current_step > self.awi.n_steps * 0.5
                and len(my_negotiators) > 0
            ):
                concentrated_ids = sorted(
                    my_negotiators,
                    key=lambda x: self.total_agreed_quantity[x],
                    reverse=True,
                )[:1]
                # distribution = self.distribute_needs(t=0,mx=self.awi.n_lines,allow_zero=False,concentrated=True,concentrated_ids=concentrated_ids)
                concentrated_idx = [
                    i for i, k in enumerate(my_negotiators) if k in concentrated_ids
                ]
                distribution = dict(
                    zip(
                        my_negotiators,
                        distribute(
                            offering_quantity,
                            len(my_negotiators),
                            mx=self.awi.n_lines,
                            concentrated=True,
                            concentrated_idx=concentrated_idx,
                        ),
                    )
                )
            else:
                # distribution = self.distribute_needs(t=0)
                distribution = dict(
                    zip(
                        my_negotiators,
                        distribute(offering_quantity, len(my_negotiators)),
                    )
                )

            d |= {
                k: (q, s, price) if q > 0 or self.awi.allow_zero_quantity else None
                for k, q in distribution.items()
            }
        d |= {k: None for k in not_negotiators}
        return d

    def counter_all(self, offers, states):
        response = dict()
        future_partners = {
            k for k, v in offers.items() if v[TIME] != self.awi.current_step
        }
        offers = {k: v for k, v in offers.items() if v[TIME] == self.awi.current_step}
        # process for sales and supplies independently
        for needs, all_partners, issues in [
            (
                self.awi.needed_supplies,
                self.awi.my_suppliers,
                self.awi.current_input_issues,
            ),
            (
                self.awi.needed_sales,
                self.awi.my_consumers,
                self.awi.current_output_issues,
            ),
        ]:
            # get a random price
            price = issues[UNIT_PRICE].rand()
            # find active partners in some random order
            partners = [_ for _ in all_partners if _ in offers.keys()]
            random.shuffle(partners)
            partners = set(partners)
            is_selling = all_partners == self.awi.my_consumers

            # ラウンドごとの相手一人あたりの平均提案個数を擬似的に算出
            if len(partners) > 0:
                neg_step = min(state.step for state in states.values())
                self.rounds_ave_offered[neg_step] = 0.7 * self.rounds_ave_offered[
                    neg_step
                ] + 0.3 * sum([offers[p][QUANTITY] for p in partners]) / len(partners)

                # print("round",neg_step,self.rounds_ave_offered,sum([offers[p][QUANTITY] for p in partners]))

            unneeded_response = (
                SAOResponse(ResponseType.END_NEGOTIATION, None)
                if not self.awi.allow_zero_quantity
                else SAOResponse(
                    ResponseType.REJECT_OFFER, (0, self.awi.current_step, 0)
                )
            )

            # find the set of partners that gave me the best offer set
            # (i.e. total quantity nearest to my needs)
            plist = list(powerset(partners))[::-1]
            plus_best_diff, _plus_best_expected_diff, plus_best_indx = (
                float("inf"),
                float("inf"),
                -1,
            )
            minus_best_diff, _minus_best_expected_diff, minus_best_indx = (
                -float("inf"),
                -float("inf"),
                -1,
            )
            best_diff, best_indx = float("inf"), -1

            for i, partner_ids in enumerate(plist):
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = offered - needs
                if diff >= 0:  # 必要以上の量のとき
                    if diff < plus_best_diff:
                        plus_best_diff, plus_best_indx = diff, i
                    elif diff == plus_best_diff:
                        if is_selling:  # 売り手の場合は高かったら更新
                            if sum(offers[p][UNIT_PRICE] for p in partner_ids) > sum(
                                offers[p][UNIT_PRICE] for p in plist[plus_best_indx]
                            ):
                                plus_best_diff, plus_best_indx = diff, i
                        else:  # 買い手の場合は安かったら更新
                            if sum(offers[p][UNIT_PRICE] for p in partner_ids) < sum(
                                offers[p][UNIT_PRICE] for p in plist[plus_best_indx]
                            ):
                                plus_best_diff, plus_best_indx = diff, i
                if diff <= 0:  # 必要量に満たないとき
                    if diff > minus_best_diff:
                        minus_best_diff, minus_best_indx = diff, i
                    elif diff == minus_best_diff:
                        if (
                            diff < 0 and len(partner_ids) < len(plist[minus_best_indx])
                        ):  # アクセプトする不足分をCounterOfferできる相手の数が多かったら更新
                            minus_best_diff, minus_best_indx = diff, i
                        elif diff == 0 or len(partner_ids) == len(
                            plist[minus_best_indx]
                        ):
                            if is_selling:  # 売り手の場合は高かったら更新
                                if sum(
                                    offers[p][UNIT_PRICE] for p in partner_ids
                                ) > sum(
                                    offers[p][UNIT_PRICE]
                                    for p in plist[minus_best_indx]
                                ):
                                    minus_best_diff, minus_best_indx = diff, i
                            else:  # 買い手の場合は安かったら更新
                                if sum(
                                    offers[p][UNIT_PRICE] for p in partner_ids
                                ) < sum(
                                    offers[p][UNIT_PRICE]
                                    for p in plist[minus_best_indx]
                                ):
                                    minus_best_diff, minus_best_indx = diff, i

            th_min_plus, th_max_plus = self._allowed_mismatch(
                min(state.relative_time for state in states.values()),
                len(partners.difference(plist[plus_best_indx]).union(future_partners)),
                is_selling,
            )
            th_min_minus, th_max_minus = self._allowed_mismatch(
                min(state.relative_time for state in states.values()),
                len(partners.difference(plist[minus_best_indx]).union(future_partners)),
                is_selling,
            )
            if th_min_minus <= minus_best_diff or plus_best_diff <= th_max_plus:
                if th_min_minus <= minus_best_diff and plus_best_diff <= th_max_plus:
                    if -minus_best_diff == plus_best_diff:
                        if is_selling:  # 売り手のときは、best_diff>0だとshortfall penaltyが発生するのでminus優先
                            best_diff, best_indx = minus_best_diff, minus_best_indx
                        else:  # 買い手のときは、best_diff<0だとshortfall penaltyが発生するのでplus優先
                            best_diff, best_indx = plus_best_diff, plus_best_indx
                    elif -minus_best_diff < plus_best_diff:
                        # 自身が買い手で、かつ不足分を残りの相手へのCounterOfferで補えないときは、shortfall penaltyを防ぐためplus優先
                        if (
                            not is_selling
                            and len(
                                partners.difference(plist[minus_best_indx]).union(
                                    future_partners
                                )
                            )
                            == 0
                        ):
                            best_diff, best_indx = plus_best_diff, plus_best_indx
                        else:
                            best_diff, best_indx = minus_best_diff, minus_best_indx
                    else:
                        best_diff, best_indx = plus_best_diff, plus_best_indx
                elif minus_best_diff < th_min_minus and plus_best_diff <= th_max_plus:
                    best_diff, best_indx = plus_best_diff, plus_best_indx
                else:
                    best_diff, best_indx = minus_best_diff, minus_best_indx

                partner_ids = plist[best_indx]
                others = list(partners.difference(partner_ids).union(future_partners))
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {k: unneeded_response for k in others}

                if (
                    best_diff < 0 and len(others) > 0
                ):  # 必要量に足りないとき、CounterOfferで補う
                    s, p = self._step_and_price(best_price=True)
                    t = min(states[p].relative_time for p in others)
                    offering_quanitity = (
                        int(-best_diff * (1 + self._overordering_fraction(t)))
                        if len(others) > 1
                        else -best_diff
                    )
                    if self.awi.current_step > self.awi.n_steps * 0.5:
                        concentrated_ids = sorted(
                            others,
                            key=lambda x: self.total_agreed_quantity[x],
                            reverse=True,
                        )[:1]
                        concentrated_idx = [
                            i for i, p in enumerate(others) if p in concentrated_ids
                        ]
                        distribution = dict(
                            zip(
                                others,
                                distribute(
                                    offering_quanitity,
                                    len(others),
                                    mx=self.awi.n_lines,
                                    concentrated=True,
                                    concentrated_idx=concentrated_idx,
                                ),
                            )
                        )
                    else:
                        distribution = dict(
                            zip(
                                others,
                                distribute(
                                    offering_quanitity, len(others), mx=self.awi.n_lines
                                ),
                            )
                        )
                    response.update(
                        {
                            k: (
                                unneeded_response
                                if q == 0
                                else SAOResponse(ResponseType.REJECT_OFFER, (q, s, p))
                            )
                            for k, q in distribution.items()
                        }
                    )

                continue

            # If I still do not have a good enough offer, distribute my current needs
            # randomly over my partners.
            t = min(_.relative_time for _ in states.values())
            # distribution = self.distribute_needs(t)

            partners = partners.union(future_partners)
            partners = list(partners)
            offering_quanitity = (
                int(needs * (1 + self._overordering_fraction(t)))
                if len(partners) > 1
                else needs
            )
            if self.awi.current_step > self.awi.n_steps * 0.5 and len(partners) > 0:
                concentrated_ids = sorted(
                    partners, key=lambda x: self.total_agreed_quantity[x], reverse=True
                )[:1]
                concentrated_idx = [
                    i for i, p in enumerate(partners) if p in concentrated_ids
                ]
                distribution = dict(
                    zip(
                        partners,
                        distribute(
                            offering_quanitity,
                            len(partners),
                            mx=self.awi.n_lines,
                            concentrated=True,
                            concentrated_idx=concentrated_idx,
                        ),
                    )
                )
            else:
                distribution = dict(
                    zip(
                        partners,
                        distribute(
                            offering_quanitity, len(partners), mx=self.awi.n_lines
                        ),
                    )
                )

            response.update(
                {
                    k: (
                        unneeded_response
                        if q == 0
                        else SAOResponse(
                            ResponseType.REJECT_OFFER, (q, self.awi.current_step, price)
                        )
                    )
                    for k, q in distribution.items()
                }
            )
        return response

    # def _allowed_mismatch(self, r:float, n_others:int, is_selling:bool):
    #     return self.undermismatch_min * ((1-r)**self.mismatch_exp), self.overmismatch_max * (r**self.mismatch_exp)

    def _allowed_mismatch(self, r: float, n_others: int, is_selling: bool):
        #     if is_selling:
        #         # 入荷量が一定、過剰な販売契約が良くない
        #         # 一人平均3個くらいの算段でOK?
        #         th_min = - 3 * n_others
        #         th_max = self.overmismatch_max_selling * (r**self.mismatch_exp)
        #     else:
        #         # 不足が良くない、n_othersが多いほどth_minが小さくても(不足が多くても)良い
        #         # 一人平均1.5個くらいの算段でOK?
        #         th_min = - 1.5 * n_others
        #         th_max = self.overmismatch_max_buying * (r**self.mismatch_exp)
        #     return th_min,th_max
        undermismatch_min = (
            self.undermismatch_min_selling
            if is_selling
            else self.undermismatch_min_buying
        )
        overmismatch_max = (
            self.overmismatch_max_selling
            if is_selling
            else self.overmismatch_max_buying
        )
        return undermismatch_min * ((1 - r) ** self.mismatch_exp), overmismatch_max * (
            r ** (1 / self.mismatch_exp)
        )

    def _overordering_fraction(self, t: float):
        mn, mx = self.overordering_min, self.overordering_max
        return mx - (mx - mn) * (t**self.overordering_exp)

    def _step_and_price(self, best_price=False):
        """Returns current step and a random (or max) price"""
        s = self.awi.current_step
        seller = self.awi.is_first_level
        issues = (
            self.awi.current_output_issues if seller else self.awi.current_input_issues
        )
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        if best_price:
            return s, pmax if seller else pmin
        return s, random.randint(pmin, pmax)




class BayesianAgent022(SyncRandomOneShotAgent):
    """
    SyncRandomOneShotAgent with BayesianAgent2-style Greedy/NonGreedy logits.

    The public classifier uses the same two-class softmax posterior and
    first-offer evidence as BayesianAgent2. Older multi-class evidence hooks
    are ignored for classification.
    """

    OPPONENT_TYPES = ("GreedyOneShotAgent", "NonGreedy")

    def __init__(
        self,
        *args,
        classification_threshold: float = 0.60,
        min_observations: int = 1,
        greedy_time_concession: float = 0.15,
        softmax_temperature: float = 1.0,
        counter_good_price_time: float = 0.65,
        counter_good_price_shortage_ratio: float = 0.40,
        counter_bad_price_penalty: float = 0.30,
        max_counter_partners: int = 4,
        counter_beam_width: int = 24,
        max_accept_subsets: int = 16,
        seller_quantity_bias: float = 1.0,
        buyer_quantity_bias: float = 1.0,
        seller_greedy_fill_threshold: float = 0.51,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.classification_threshold = classification_threshold
        self.min_observations = min_observations
        self.greedy_time_concession = greedy_time_concession
        self.softmax_temperature = max(0.01, float(softmax_temperature))
        self.exploration_days = 6
        self.max_exploration_days = 10
        self.unknown_exploration_ratio = 0.5
        self.min_strategy_classifications = 3
        self.sync_equal_classification_threshold = 0.50
        self.strategy_random_threshold = 0.55
        self.exploration_quantity_multiplier = 1.2
        self.small_dist_early_quantity_multiplier = 1.3
        self.small_dist_midpoint = 0.5
        self.non_greedy_success_default = 0.5
        self.classification_log_path = os.environ.get("BAYES_CLASSIFICATION_LOG")
        self.counter_good_price_time = float(counter_good_price_time)
        self.counter_good_price_shortage_ratio = float(counter_good_price_shortage_ratio)
        self.counter_bad_price_penalty = float(counter_bad_price_penalty)
        self.max_counter_partners = int(max_counter_partners)
        self.counter_beam_width = int(counter_beam_width)
        self.max_accept_subsets = int(max_accept_subsets)
        self.seller_quantity_bias = float(seller_quantity_bias)
        self.buyer_quantity_bias = float(buyer_quantity_bias)
        self.seller_greedy_fill_threshold = float(seller_greedy_fill_threshold)

    # ---------------------------------------------------------------------
    # Initialization and small utilities
    # ---------------------------------------------------------------------

    def init(self):
        super().init()
        self._opponent_logp: dict[str, dict[str, float]] = {}
        self._opponent_logits = {}
        self._opponent_observations = defaultdict(int)
        self._non_greedy_veto = {}
        self._opponent_offer_history = defaultdict(list)
        self._sent_offer_history = defaultdict(list)
        self._own_offer_result_history = defaultdict(list)
        self._own_offer_end_history = defaultdict(list)
        self._non_greedy_initial_offer_results = []
        self._partner_non_greedy_initial_offer_results = defaultdict(list)
        self._initial_good_rejection_streak_observed = defaultdict(int)
        self._received_offer_counts = defaultdict(int)
        self._received_first_offer_history = defaultdict(list)
        self._logit_history = defaultdict(list)
        self._evidence_counts = defaultdict(lambda: defaultdict(int))
        self._history_pattern_observed = defaultdict(set)
        self._first_offer_trials_by_price = defaultdict(lambda: defaultdict(int))
        self._first_offer_accepts_by_price = defaultdict(lambda: defaultdict(int))
        self._first_offer_reoffers_by_price = defaultdict(lambda: defaultdict(int))
        self._first_offer_counter_quantities_by_price = defaultdict(
            lambda: defaultdict(list)
        )
        for partner in self._all_partners():
            self._ensure_partner(partner)

    def before_step(self):
        super().before_step()
        self._received_offer_counts.clear()
        for partner in self._all_partners():
            self._ensure_partner(partner)
        self._dump_classification_log("before_step")

    def _all_partners(self):
        return list(dict.fromkeys(list(self.awi.my_suppliers) + list(self.awi.my_consumers)))

    def _ensure_partner(self, partner):
        if partner is None:
            return
        if partner not in self._opponent_logp:
            prior = -math.log(len(self.OPPONENT_TYPES))
            self._opponent_logp[partner] = {name: prior for name in self.OPPONENT_TYPES}
        if partner not in self._opponent_logits:
            self._opponent_logits[partner] = {
                "GreedyOneShotAgent": 0.0,
                "NonGreedy": 0.0,
            }
        self._sent_offer_history.setdefault(partner, [])
        self._own_offer_result_history.setdefault(partner, [])
        self._own_offer_end_history.setdefault(partner, [])
        self._partner_non_greedy_initial_offer_results.setdefault(partner, [])
        self._received_first_offer_history.setdefault(partner, [])
        self._logit_history.setdefault(partner, [])
        self._evidence_counts.setdefault(partner, defaultdict(int))
        self._history_pattern_observed.setdefault(partner, set())
        self._first_offer_trials_by_price.setdefault(partner, defaultdict(int))
        self._first_offer_accepts_by_price.setdefault(partner, defaultdict(int))
        self._first_offer_reoffers_by_price.setdefault(partner, defaultdict(int))
        self._first_offer_counter_quantities_by_price.setdefault(
            partner,
            defaultdict(list),
        )

    def _is_process_one_agent(self) -> bool:
        return str(getattr(self, "id", "")).endswith("@1")

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

    def _clamp_quantity(self, partner, quantity: int) -> int:
        issues = self._issues_for(partner)
        qmin = int(issues[QUANTITY].min_value)
        qmax = int(issues[QUANTITY].max_value)
        if quantity <= 0 and self.awi.allow_zero_quantity:
            return 0
        return max(qmin, min(qmax, int(quantity)))

    def _exploration_enabled(self) -> bool:
        if self.awi.current_step < self.exploration_days:
            return True
        if self.awi.current_step >= self.max_exploration_days:
            return False
        return self._too_many_unknown_opponents()

    def _active_partners(self):
        return [
            partner
            for partner in self._all_partners()
            if partner in getattr(self, "negotiators", {})
        ]

    def _too_many_unknown_opponents(self) -> bool:
        partners = self._active_partners()
        if not partners:
            return False
        types = [self.opponent_type(partner) for partner in partners]
        unknown_count = sum(opponent_type == "Unknown" for opponent_type in types)
        classified_count = sum(
            opponent_type not in {"Unknown", "Other"}
            for opponent_type in types
        )
        unknown_ratio = unknown_count / len(partners)
        return (
            unknown_ratio >= self.unknown_exploration_ratio
            or classified_count < min(self.min_strategy_classifications, len(partners))
        )

    def _day_progress(self) -> float:
        total_steps = getattr(self.awi, "n_steps", None)
        if total_steps is None:
            total_steps = getattr(self.awi, "n_days", None)
        if not total_steps:
            return 0.0
        return max(0.0, min(1.0, self.awi.current_step / max(1, int(total_steps) - 1)))

    def _small_dist_quantity_multiplier(self) -> float:
        progress = self._day_progress()
        if progress >= self.small_dist_midpoint:
            return 1.0
        extra = self.small_dist_early_quantity_multiplier - 1.0
        remaining = 1.0 - progress / max(0.01, self.small_dist_midpoint)
        return 1.0 + extra * remaining

    def _world_name(self) -> str:
        world = getattr(self.awi, "_world", None)
        if world is None:
            world = getattr(self.awi, "world", None)
        for name in ("name", "id"):
            value = getattr(world, name, None)
            if value is not None:
                return str(value)
        return ""

    def _dump_classification_log(self, event: str):
        if not self.classification_log_path:
            return
        partners = [
            partner
            for partner in self._all_partners()
            if partner in getattr(self, "negotiators", {})
        ]
        if not partners:
            return
        strategy_types = self._strategy_opponent_types(partners)
        row = {
            "event": event,
            "world": self._world_name(),
            "step": int(self.awi.current_step),
            "agent": self.id,
            "predictions": {
                partner: {
                    "strict": self.opponent_type(partner),
                    "strategy": strategy_types[partner],
                    "observations": int(self._opponent_observations[partner]),
                    "posteriors": self.opponent_posteriors(partner),
                }
                for partner in partners
            },
        }
        try:
            path = Path(self.classification_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception:
            pass

    def _price_for_me_score(self, partner, price: float) -> float:
        """0.0 is worst for me, 1.0 is best for me."""
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

    def _max_lines(self) -> int:
        return max(
            1,
            int(getattr(self.awi, "n_lines", getattr(self.awi, "max_n_lines", 1)) or 1),
        )

    def _relative_time(self, states) -> float:
        values = [getattr(state, "relative_time", None) for state in states.values()]
        values = [float(value) for value in values if value is not None]
        if values:
            return max(0.0, min(1.0, min(values)))
        return 0.0

    def _round_index(self, states) -> int:
        values = []
        for state in states.values():
            for name in ("step", "current_offer_index"):
                value = getattr(state, name, None)
                if isinstance(value, int):
                    values.append(value)
                    break
        return min(values, default=0)

    # ---------------------------------------------------------------------
    # Bayesian classifier
    # ---------------------------------------------------------------------

    def opponent_posteriors(self, partner) -> dict[str, float]:
        self._ensure_partner(partner)

        if partner in self._non_greedy_veto:
            return {"GreedyOneShotAgent": 0.0, "NonGreedy": 1.0}

        logits = self._opponent_logits[partner]
        scaled = {
            name: value / self.softmax_temperature
            for name, value in logits.items()
        }
        base = max(scaled.values())
        weights = {name: math.exp(value - base) for name, value in scaled.items()}
        total = sum(weights.values())
        if total <= 0:
            return {"GreedyOneShotAgent": 0.5, "NonGreedy": 0.5}
        return {name: value / total for name, value in weights.items()}

    def opponent_type(self, partner) -> str:
        self._ensure_partner(partner)

        if partner in self._non_greedy_veto:
            return "NonGreedy"

        if self._opponent_observations[partner] < self.min_observations:
            return "Unknown"

        posteriors = self.opponent_posteriors(partner)
        best = max(posteriors, key=posteriors.get)
        threshold = 0.51 if best == "GreedyOneShotAgent" else self.classification_threshold

        if posteriors[best] >= threshold:
            return best

        return "Unknown"

    def _best_non_greedy_type(self, partner, posteriors) -> str:
        if self._opponent_observations[partner] < self.min_observations:
            return "Unknown"
        candidates = {
            name: value
            for name, value in posteriors.items()
            if name != "GreedyOneShotAgent"
        }
        best = max(candidates, key=candidates.get)
        if best in {"SyncRandomDistOneShotAgent", "EqualDistOneShotAgent"}:
            if candidates[best] >= 0.30:
                return best
        elif candidates[best] >= 0.40:
            return best
        return "Unknown"

    def _greedy_binary_decision(self, partner, posteriors) -> bool | None:
        if self._opponent_observations[partner] < self.min_observations:
            return None

        greedy_p = posteriors.get("GreedyOneShotAgent", 0.0)
        non_greedy_best = max(
            posteriors.get("RandomOneShotAgent", 0.0),
            posteriors.get("SyncRandomDistOneShotAgent", 0.0),
            posteriors.get("EqualDistOneShotAgent", 0.0),
            posteriors.get("Other", 0.0),
        )

        own_history = self._own_offer_result_history.get(partner, [])[-12:]
        batch_decision = self._greedy_probe_batch_decision(partner)
        if batch_decision is not None:
            return batch_decision

        good_results = [
            item for item in own_history if bool(item.get("price_good"))
        ]
        bad_results = [
            item for item in own_history if not bool(item.get("price_good"))
        ]
        good_accepts = sum(item["accepted"] for item in good_results)
        bad_accepts = sum(item["accepted"] for item in bad_results)
        good_trials = len(good_results)
        bad_trials = len(bad_results)
        early_good_results = [
            item for item in good_results if float(item.get("relative_time", 1.0)) < 0.5
        ]
        early_bad_results = [
            item for item in bad_results if float(item.get("relative_time", 1.0)) < 0.5
        ]
        early_bad_rejections = sum(not item["accepted"] for item in early_bad_results)
        early_bad_accepts = sum(item["accepted"] for item in early_bad_results)
        equal_like_history = self._equal_like_offer_history(partner)
        if (
            early_bad_accepts >= 2
            and good_accepts == 0
            and non_greedy_best >= greedy_p * 1.20
        ):
            return False
        if (
            bad_accepts >= 2
            and good_accepts == 0
            and non_greedy_best >= greedy_p * 1.20
        ):
            return False
        if len(good_results) >= 1 and len(bad_results) >= 1:
            good_accept_rate = sum(item["accepted"] for item in good_results) / len(good_results)
            bad_accept_rate = sum(item["accepted"] for item in bad_results) / len(bad_results)
            if (
                bad_accept_rate >= 0.50
                and good_accept_rate < 0.50
                and non_greedy_best >= greedy_p * 1.20
            ):
                return False
        offer_history = self._opponent_offer_history.get(partner, [])[-8:]
        early_offers = [
            item
            for item in offer_history
            if float(item.get("time", 1.0)) < 0.5
        ]
        if early_offers:
            early_good_ratio = sum(bool(item["price_good"]) for item in early_offers) / len(early_offers)
            early_quantities = [max(0, int(item["quantity"])) for item in early_offers]
            early_small_ratio = sum(quantity <= 3 for quantity in early_quantities) / len(early_quantities)
            early_equal_like = (
                early_small_ratio >= 0.70
                and sum(early_quantities) / len(early_quantities) <= 3.5
            )
            if early_good_ratio <= 0.50 and non_greedy_best >= greedy_p * 0.90:
                return False
            if len(early_offers) >= 2 and not early_equal_like:
                if (
                    early_bad_rejections >= 1
                    and early_bad_accepts == 0
                    and greedy_p >= 0.18
                    and greedy_p >= non_greedy_best * 0.50
                ):
                    return True
                if (
                    len(early_offers) >= 3
                    and greedy_p >= 0.25
                    and greedy_p >= non_greedy_best * 0.85
                ):
                    return True
        if len(offer_history) >= 4:
            price_goods = [bool(item["price_good"]) for item in offer_history]
            quantities = [max(0, int(item["quantity"])) for item in offer_history]
            good_ratio = sum(price_goods) / len(price_goods)
            price_flip_ratio = sum(
                previous != current
                for previous, current in zip(price_goods, price_goods[1:], strict=False)
            ) / max(1, len(price_goods) - 1)
            mean_quantity = sum(quantities) / len(quantities)
            quantity_range = max(quantities) - min(quantities)
            small_ratio = sum(quantity <= 3 for quantity in quantities) / len(quantities)
            equal_like_quantity = (
                small_ratio >= 0.70
                and mean_quantity <= 3.5
            )
            early = price_goods[: max(1, len(price_goods) // 2)]
            late = price_goods[max(1, len(price_goods) // 2) :]
            early_good_ratio = sum(early) / len(early)
            late_good_ratio = sum(late) / len(late)

            if (
                good_ratio >= 0.75
                and price_flip_ratio <= 0.25
                and not equal_like_quantity
                and greedy_p >= 0.35
                and greedy_p >= non_greedy_best * 0.85
            ):
                return True
            if (
                early_good_ratio >= 0.75
                and late_good_ratio <= 0.50
                and not equal_like_quantity
                and greedy_p >= 0.30
                and greedy_p >= non_greedy_best * 0.80
            ):
                return True

            if (
                price_flip_ratio >= 0.45
                and 0.25 <= good_ratio <= 0.75
                and non_greedy_best >= greedy_p * 0.90
            ):
                return False
            if good_ratio <= 0.45 and non_greedy_best >= greedy_p * 0.80:
                return False

        if greedy_p >= 0.82 and greedy_p >= non_greedy_best * 1.20:
            return True
        if non_greedy_best >= 0.60 and greedy_p <= non_greedy_best * 0.45:
            return False
        return None

    def _greedy_probe_batch_decision(self, partner) -> bool | None:
        probes = [
            item
            for item in self._own_offer_result_history.get(partner, [])
            if float(item.get("relative_time", 1.0)) < 0.5
            and bool(item.get("initial", False))
        ][-6:]
        if len(probes) < 6:
            return None

        good = [item for item in probes if bool(item.get("price_good"))]
        bad = [item for item in probes if not bool(item.get("price_good"))]
        if len(good) < 2 or len(bad) < 2:
            return None

        good_accept_rate = sum(item["accepted"] for item in good) / len(good)
        bad_accept_rate = sum(item["accepted"] for item in bad) / len(bad)

        if bad_accept_rate == 0.0 and good_accept_rate >= 0.67:
            return True
        if bad_accept_rate >= 0.34:
            return False
        if 0.25 < good_accept_rate < 0.75 and 0.0 < bad_accept_rate < 0.75:
            return False
        return None

    def _equal_like_offer_history(self, partner) -> bool:
        history = self._opponent_offer_history.get(partner, [])[-8:]
        if len(history) < 4:
            return False
        quantities = [max(0, int(item["quantity"])) for item in history]
        mean_quantity = sum(quantities) / len(quantities)
        quantity_range = max(quantities) - min(quantities)
        small_ratio = sum(quantity <= 3 for quantity in quantities) / len(quantities)
        return small_ratio >= 0.70 and mean_quantity <= 3.5

    def _greedy_override_type(self, partner, posteriors) -> str | None:
        if self._opponent_observations[partner] < self.min_observations:
            return None
        greedy_p = posteriors.get("GreedyOneShotAgent", 0.0)
        sync_p = posteriors.get("SyncRandomDistOneShotAgent", 0.0)
        equal_p = posteriors.get("EqualDistOneShotAgent", 0.0)

        offer_history = self._opponent_offer_history.get(partner, [])[-8:]
        own_history = self._own_offer_result_history.get(partner, [])[-12:]
        offer_score = 0

        if offer_history:
            price_goods = [bool(item["price_good"]) for item in offer_history]
            good_ratio = sum(price_goods) / len(price_goods)
            price_flip_ratio = sum(
                previous != current
                for previous, current in zip(price_goods, price_goods[1:], strict=False)
            ) / max(1, len(price_goods) - 1)
            if good_ratio >= 0.75 and price_flip_ratio <= 0.25:
                offer_score += 1
            if (
                len(price_goods) >= 4
                and sum(price_goods[: len(price_goods) // 2]) / max(1, len(price_goods) // 2)
                >= 0.75
                and sum(price_goods[len(price_goods) // 2 :])
                / max(1, len(price_goods) - len(price_goods) // 2)
                <= 0.50
            ):
                offer_score += 1

        good_results = [
            item for item in own_history if bool(item.get("price_good"))
        ]
        bad_results = [
            item for item in own_history if not bool(item.get("price_good"))
        ]
        accept_score = 0
        if len(good_results) >= 2 and len(bad_results) >= 1:
            good_accept_rate = sum(item["accepted"] for item in good_results) / len(good_results)
            bad_accept_rate = sum(item["accepted"] for item in bad_results) / len(bad_results)
            if good_accept_rate >= 0.65 and bad_accept_rate <= 0.35:
                accept_score += 2

        if (
            accept_score >= 2
            and greedy_p >= 0.30
            and greedy_p >= sync_p * 0.75
            and greedy_p >= equal_p * 0.80
        ):
            return "GreedyOneShotAgent"
        if (
            accept_score >= 1
            and greedy_p >= 0.60
            and greedy_p >= sync_p * 0.90
            and greedy_p >= equal_p * 0.90
            and offer_score >= 1
        ):
            return "GreedyOneShotAgent"
        return None

    def _best_behavior_type(self, partner) -> tuple[str, float]:
        posteriors = self.opponent_posteriors(partner)
        candidates = dict(posteriors)
        best = max(candidates, key=candidates.get)
        return best, candidates[best]

    def _strategy_opponent_types(self, partners: list[str]) -> dict[str, str]:
        return {partner: self.opponent_type(partner) for partner in partners}

    def _ensure_min_dist_strategy_types(self, partners, types):
        return dict(types)

    def _add_evidence(self, partner, weights: dict[str, float], strength: float = 1.0):
        del weights, strength
        self._ensure_partner(partner)

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

    def _record_sent_offers(
        self,
        proposals,
        initial: bool = True,
        relative_time: float = 0.0,
    ):
        for partner, offer in proposals.items():
            self._record_sent_offer(
                partner,
                offer,
                initial=initial,
                relative_time=relative_time,
            )

    def _record_response_offers(self, responses, relative_time: float = 0.0):
        for partner, response in responses.items():
            if response is None or response.response != ResponseType.REJECT_OFFER:
                continue
            self._record_sent_offer(
                partner,
                response.outcome,
                initial=False,
                relative_time=relative_time,
            )

    def _record_sent_offer(
        self,
        partner,
        offer,
        initial: bool = True,
        relative_time: float = 0.0,
    ):
        if partner is None or offer is None or len(offer) <= UNIT_PRICE:
            return
        self._ensure_partner(partner)
        opponent_type = self.opponent_type(partner)
        price_label = self._opponent_price_label(partner, offer[UNIT_PRICE])
        self._sent_offer_history[partner].append(
            {
                "step": self.awi.current_step,
                "relative_time": max(0.0, min(1.0, float(relative_time))),
                "initial": bool(initial),
                "partner": partner,
                "end_response_observed": False,
                "first_result_observed": False,
                "offer": tuple(offer),
                "price_label": price_label,
                "price_good": self._price_good_for_opponent(partner, offer[UNIT_PRICE]),
                "non_greedy_initial_probe": initial
                and opponent_type
                in {
                    "RandomOneShotAgent",
                    "SyncRandomDistOneShotAgent",
                    "EqualDistOneShotAgent",
                    "Other",
                    "Unknown",
                    "NonGreedy",
                },
            }
        )
        if initial:
            self._first_offer_trials_by_price[partner][price_label] += 1
        if len(self._sent_offer_history[partner]) > 20:
            del self._sent_offer_history[partner][:-20]

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

    def _latest_sent_offer(self, partner):
        history = self._sent_offer_history.get(partner, [])
        current = [
            item
            for item in history
            if item["step"] == self.awi.current_step
        ]
        return current[-1] if current else None

    def _matching_sent_offer(self, partner, outcome):
        history = self._sent_offer_history.get(partner, [])
        for item in reversed(history):
            if item["step"] != self.awi.current_step:
                continue
            if self._same_offer(item["offer"], outcome):
                return item
        return None

    def _latest_unobserved_first_offer(self, partner):
        for item in reversed(self._sent_offer_history.get(partner, [])):
            if item["step"] != self.awi.current_step:
                continue
            if not item.get("initial", False):
                continue
            if item.get("first_result_observed", False):
                continue
            return item
        return None

    def _observe_first_offer_classification_result(
        self,
        partner,
        sent_offer,
        accepted: bool,
    ):
        if sent_offer is None or sent_offer.get("first_result_observed", False):
            return

        sent_offer["first_result_observed"] = True
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
                greedy=1.20,
                reason="good_first_offer_accepted",
            )
        elif price_label == "good":
            self._add_evidence_count(partner, "good_first_offer_rejected")
            self._add_logit_evidence(
                partner,
                non_greedy=0.15,
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

    def _observe_own_first_offer_counter(self, partner, sent_offer, counter_offer):
        if sent_offer is None or sent_offer.get("first_result_observed", False):
            return

        price_label = sent_offer.get("price_label", "neutral")
        self._first_offer_reoffers_by_price[partner][price_label] += 1
        self._first_offer_counter_quantities_by_price[partner][price_label].append(
            max(0, int(counter_offer[QUANTITY]))
        )
        quantities = self._first_offer_counter_quantities_by_price[partner][price_label]
        if len(quantities) > 50:
            del quantities[:-50]

        self._observe_first_offer_classification_result(
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

    def _partner_ended_after_sent_offer(self, sent_offer, state) -> bool:
        if sent_offer is None or state is None:
            return False
        if not bool(getattr(state, "broken", False)):
            return False
        if bool(getattr(state, "timedout", False)) or bool(getattr(state, "has_error", False)):
            return False
        if getattr(state, "agreement", None) is not None:
            return False
        return self._same_offer(getattr(state, "current_offer", None), sent_offer["offer"])

    def _observe_partner_end_response(self, partner, sent_offer, ended: bool):
        if sent_offer is None or sent_offer.get("end_response_observed", False):
            return
        sent_offer["end_response_observed"] = True

        history = self._own_offer_end_history[partner]
        history.append(
            {
                "step": self.awi.current_step,
                "relative_time": float(sent_offer.get("relative_time", 1.0)),
                "ended": bool(ended),
            }
        )
        if len(history) > 20:
            del history[:-20]
        self._observe_partner_end_pattern(partner)

    def _observe_partner_end_pattern(self, partner):
        return
        history = self._own_offer_end_history[partner][-10:]
        if len(history) < 4:
            return
        ended_count = sum(item["ended"] for item in history)
        end_rate = sum(item["ended"] for item in history) / len(history)
        evidence = {name: 0.0 for name in self.OPPONENT_TYPES}
        if ended_count >= 3 and end_rate >= 0.30:
            evidence["RandomOneShotAgent"] -= 0.90
            evidence["SyncRandomDistOneShotAgent"] += 0.08
            evidence["EqualDistOneShotAgent"] += 0.05
        elif ended_count >= 2 and end_rate >= 0.20:
            evidence["RandomOneShotAgent"] -= 0.45
        if any(abs(value) > 0 for value in evidence.values()):
            self._add_evidence(partner, evidence, strength=0.4)

    def _observe_own_offer_result(self, partner, sent_offer, accepted: bool):
        if sent_offer is None:
            return
        if sent_offer.get("initial", False):
            self._observe_first_offer_classification_result(
                partner,
                sent_offer,
                accepted=accepted,
            )
        price_good = bool(sent_offer["price_good"])
        self._own_offer_result_history[partner].append(
            {
                "step": self.awi.current_step,
                "relative_time": float(sent_offer.get("relative_time", 1.0)),
                "initial": bool(sent_offer.get("initial", False)),
                "price_good": price_good,
                "accepted": bool(accepted),
            }
        )
        if len(self._own_offer_result_history[partner]) > 30:
            del self._own_offer_result_history[partner][:-30]
        return

        early = float(sent_offer.get("relative_time", 1.0)) < 0.5
        evidence = {name: 0.0 for name in self.OPPONENT_TYPES}
        if accepted and price_good:
            evidence["GreedyOneShotAgent"] += 0.45 if early else 1.00
            evidence["RandomOneShotAgent"] -= 0.15
            evidence["SyncRandomDistOneShotAgent"] -= 0.10
            evidence["EqualDistOneShotAgent"] -= 0.05
        elif not accepted and not price_good:
            pass
        elif accepted and not price_good:
            evidence["RandomOneShotAgent"] += 0.35 if early else 0.25
            evidence["SyncRandomDistOneShotAgent"] += 0.20 if early else 0.12
            evidence["EqualDistOneShotAgent"] += 0.15 if early else 0.10
        elif not accepted and price_good:
            # Greedy can reject good-price offers when the quantity is unsuitable.
            evidence["GreedyOneShotAgent"] -= 0.10 if early else 0.45
            evidence["Other"] += 0.05
        if any(abs(value) > 0 for value in evidence.values()):
            self._add_evidence(partner, evidence, strength=0.6)
        self._own_offer_result_history[partner].append(
            {
                "step": self.awi.current_step,
                "relative_time": float(sent_offer.get("relative_time", 1.0)),
                "initial": bool(sent_offer.get("initial", False)),
                "price_good": price_good,
                "accepted": bool(accepted),
            }
        )
        if len(self._own_offer_result_history[partner]) > 30:
            del self._own_offer_result_history[partner][:-30]
        self._observe_own_offer_result_pattern(partner)
        self._observe_initial_good_price_rejection_streak(partner)

    def _observe_own_offer_result_pattern(self, partner):
        return
        history = self._own_offer_result_history[partner][-12:]
        good = [item for item in history if item["price_good"]]
        if len(good) < 3:
            return
        good_accept_rate = sum(item["accepted"] for item in good) / len(good)
        evidence = {name: 0.0 for name in self.OPPONENT_TYPES}
        if good_accept_rate >= 0.67:
            evidence["GreedyOneShotAgent"] += 1.40
            evidence["RandomOneShotAgent"] -= 0.15
            evidence["SyncRandomDistOneShotAgent"] -= 0.25
            evidence["EqualDistOneShotAgent"] -= 0.10
        elif good_accept_rate <= 0.25:
            evidence["GreedyOneShotAgent"] -= 0.90
            evidence["RandomOneShotAgent"] += 0.20
            evidence["SyncRandomDistOneShotAgent"] += 0.25
            evidence["EqualDistOneShotAgent"] += 0.15
        if any(abs(value) > 0 for value in evidence.values()):
            self._add_evidence(partner, evidence, strength=0.8)

    def _observe_initial_good_price_rejection_streak(self, partner):
        return
        history = self._own_offer_result_history[partner]
        streak = 0
        for item in reversed(history):
            if not item.get("initial", False):
                break
            if not item.get("price_good", False) or item.get("accepted", False):
                break
            streak += 1

        if streak < 2:
            return
        if streak <= self._initial_good_rejection_streak_observed[partner]:
            return
        self._initial_good_rejection_streak_observed[partner] = streak

        evidence = {name: 0.0 for name in self.OPPONENT_TYPES}
        evidence["GreedyOneShotAgent"] -= 0.65 if streak == 2 else 1.00
        evidence["RandomOneShotAgent"] += 0.12
        evidence["SyncRandomDistOneShotAgent"] += 0.15
        evidence["EqualDistOneShotAgent"] += 0.10
        self._add_evidence(partner, evidence, strength=0.7)

    def _renormalize_logp(self, partner):
        logp = self._opponent_logp[partner]
        center = max(logp.values())
        for name in logp:
            logp[name] -= center

    def _observe_offer(self, partner, offer: Outcome, states):
        if offer is None:
            return
        self._ensure_partner(partner)
        if len(offer) <= UNIT_PRICE:
            return

        quantity = int(offer[QUANTITY])
        delivery_step = offer[TIME]
        price = float(offer[UNIT_PRICE])
        t = self._relative_time(states)
        round_index = self._round_index(states)
        price_good = self._price_good_for_opponent(partner, price)
        history = self._opponent_offer_history[partner]
        history.append(
            {
                "step": self.awi.current_step,
                "round": round_index,
                "time": t,
                "quantity": quantity,
                "price": price,
                "price_good": price_good,
                "price_label": "good" if price_good else "bad",
            }
        )
        if len(history) > 30:
            del history[:-30]
        self._observe_binary_offer_history(partner)
        return

    def _add_history_pattern_logit(
        self,
        partner,
        *,
        greedy: float = 0.0,
        non_greedy: float = 0.0,
        reason: str,
    ):
        key = (int(self.awi.current_step), reason)
        if key in self._history_pattern_observed[partner]:
            return
        self._history_pattern_observed[partner].add(key)
        self._add_evidence_count(partner, reason)
        self._add_logit_evidence(
            partner,
            greedy=greedy,
            non_greedy=non_greedy,
            reason=reason,
        )

    def _observe_binary_offer_history(self, partner):
        history = self._opponent_offer_history[partner][-8:]
        if len(history) < 4:
            return

        quantities = [max(0, int(item["quantity"])) for item in history]
        price_goods = [bool(item["price_good"]) for item in history]
        if not quantities:
            return

        issues = self._issues_for(partner)
        qmax = max(1, int(issues[QUANTITY].max_value))
        mean = sum(quantities) / len(quantities)
        variance = sum((quantity - mean) ** 2 for quantity in quantities) / len(quantities)
        coefficient = math.sqrt(variance) / max(1.0, mean)
        mean_ratio = mean / qmax
        quantity_range = max(quantities) - min(quantities)
        small_ratio = sum(quantity <= max(2, math.ceil(0.35 * qmax)) for quantity in quantities) / len(quantities)
        good_ratio = sum(price_goods) / len(price_goods)
        price_flip_ratio = sum(
            previous != current
            for previous, current in zip(price_goods, price_goods[1:], strict=False)
        ) / max(1, len(price_goods) - 1)

        early = history[: max(1, len(history) // 2)]
        late = history[max(1, len(history) // 2) :]
        early_good_ratio = sum(bool(item["price_good"]) for item in early) / len(early)
        late_good_ratio = sum(bool(item["price_good"]) for item in late) / len(late)

        stable_quantity = coefficient <= 0.30 or quantity_range <= 1
        if small_ratio >= 0.70 and stable_quantity:
            self._add_history_pattern_logit(
                partner,
                non_greedy=0.70,
                reason="history_small_stable_quantity",
            )
        elif (
            small_ratio >= 0.60
            and coefficient >= 0.45
            and (good_ratio <= 0.65 or price_flip_ratio >= 0.30)
        ):
            self._add_history_pattern_logit(
                partner,
                non_greedy=0.40,
                reason="history_small_variable_quantity",
            )

        if price_flip_ratio >= 0.40 and coefficient >= 0.35 and good_ratio < 0.90:
            self._add_history_pattern_logit(
                partner,
                non_greedy=0.35,
                reason="history_random_like_flips",
            )

        if good_ratio >= 0.75 and mean_ratio >= 0.55 and stable_quantity:
            self._add_history_pattern_logit(
                partner,
                greedy=0.35,
                reason="history_large_selfish_stable",
            )

        if early_good_ratio >= 0.75 and late_good_ratio <= early_good_ratio - 0.35:
            self._add_history_pattern_logit(
                partner,
                greedy=0.85,
                reason="history_selfish_then_concedes",
            )

    def _observe_history_pattern(self, partner):
        return
        history = self._opponent_offer_history[partner]
        if len(history) < 4:
            return

        recent = history[-8:]
        quantities = [max(0, int(item["quantity"])) for item in recent]
        price_goods = [bool(item["price_good"]) for item in recent]
        if not quantities or sum(quantities) <= 0:
            return

        issues = self._issues_for(partner)
        qmax = max(1, int(issues[QUANTITY].max_value))
        mean = sum(quantities) / len(quantities)
        variance = sum((quantity - mean) ** 2 for quantity in quantities) / len(quantities)
        coefficient = math.sqrt(variance) / max(1.0, mean)
        mean_ratio = mean / qmax
        quantity_range = max(quantities) - min(quantities)
        small_ratio = sum(quantity <= 3 for quantity in quantities) / len(quantities)
        extreme_ratio = sum(quantity >= 0.8 * qmax for quantity in quantities) / len(quantities)
        good_ratio = sum(price_goods) / len(price_goods)
        good_to_bad = sum(
            previous and not current
            for previous, current in zip(price_goods, price_goods[1:], strict=False)
        )
        bad_to_good = sum(
            (not previous) and current
            for previous, current in zip(price_goods, price_goods[1:], strict=False)
        )
        price_flip_ratio = sum(
            previous != current
            for previous, current in zip(price_goods, price_goods[1:], strict=False)
        ) / max(1, len(price_goods) - 1)
        early = recent[: max(1, len(recent) // 2)]
        late = recent[max(1, len(recent) // 2) :]
        early_good_ratio = sum(bool(item["price_good"]) for item in early) / len(early)
        late_good_ratio = sum(bool(item["price_good"]) for item in late) / len(late)
        last_two_bad = len(price_goods) >= 2 and not price_goods[-1] and not price_goods[-2]
        previous_good_streak = len(price_goods) >= 4 and all(price_goods[-4:-2])

        evidence = {name: 0.0 for name in self.OPPONENT_TYPES}
        stable_quantity = coefficient <= 0.25 or quantity_range <= 1
        if small_ratio >= 0.70 and mean <= 3.5 and stable_quantity:
            if good_ratio >= 0.75 and price_flip_ratio <= 0.20:
                evidence["EqualDistOneShotAgent"] += 0.75
                evidence["RandomOneShotAgent"] -= 0.20
                evidence["GreedyOneShotAgent"] -= 0.65
            else:
                evidence["EqualDistOneShotAgent"] += 1.25
                evidence["SyncRandomDistOneShotAgent"] -= 0.20
                evidence["GreedyOneShotAgent"] -= 0.10
                evidence["RandomOneShotAgent"] -= 0.20
        elif small_ratio >= 0.70 and (coefficient >= 0.40 or quantity_range >= 3):
            evidence["SyncRandomDistOneShotAgent"] += 0.45 if good_ratio >= 0.65 else 1.00
            evidence["EqualDistOneShotAgent"] -= 0.45
        elif extreme_ratio >= 0.25:
            if good_ratio >= 0.65 and price_flip_ratio <= 0.35:
                evidence["GreedyOneShotAgent"] += 0.65
                evidence["RandomOneShotAgent"] -= 0.10
            else:
                evidence["RandomOneShotAgent"] += 0.75
                evidence["GreedyOneShotAgent"] += 0.05
            evidence["EqualDistOneShotAgent"] -= 0.50
            evidence["SyncRandomDistOneShotAgent"] -= 0.55
        elif stable_quantity and mean_ratio <= 0.65:
            evidence["EqualDistOneShotAgent"] += 0.80
            evidence["SyncRandomDistOneShotAgent"] -= 0.05
            evidence["GreedyOneShotAgent"] -= 0.15
            evidence["RandomOneShotAgent"] -= 0.10
        elif mean_ratio <= 0.65:
            evidence["SyncRandomDistOneShotAgent"] += 0.25

        if coefficient >= 0.55:
            evidence["SyncRandomDistOneShotAgent"] += 0.30 if good_ratio < 0.75 else -0.10
            evidence["GreedyOneShotAgent"] += 0.10
            evidence["EqualDistOneShotAgent"] -= 0.40
            if good_ratio >= 0.75 and price_flip_ratio <= 0.25:
                evidence["GreedyOneShotAgent"] += 0.45
                evidence["SyncRandomDistOneShotAgent"] -= 0.10
            if extreme_ratio >= 0.20 and 0.20 <= good_ratio <= 0.80:
                evidence["RandomOneShotAgent"] += 0.20 if good_ratio >= 0.65 else 0.75
                evidence["SyncRandomDistOneShotAgent"] -= 0.45

        quantity_unstable = coefficient >= 0.45 or quantity_range >= 3
        if (
            good_to_bad > 0
            and bad_to_good > 0
            and 0.20 <= good_ratio <= 0.80
            and quantity_unstable
        ):
            if extreme_ratio >= 0.20:
                evidence["RandomOneShotAgent"] += 0.65
                evidence["SyncRandomDistOneShotAgent"] -= 0.20
            else:
                evidence["RandomOneShotAgent"] += 0.25
                evidence["SyncRandomDistOneShotAgent"] += 0.65
            evidence["EqualDistOneShotAgent"] += 0.10
            evidence["GreedyOneShotAgent"] -= 0.45
        elif good_to_bad > 0 and bad_to_good > 0 and 0.20 <= good_ratio <= 0.80:
            evidence["RandomOneShotAgent"] += 0.10
            evidence["SyncRandomDistOneShotAgent"] += 0.40
            evidence["EqualDistOneShotAgent"] += 0.10
        elif price_flip_ratio >= 0.35 and good_ratio < 0.90:
            evidence["SyncRandomDistOneShotAgent"] += 0.25 if good_ratio < 0.75 else -0.05
            evidence["EqualDistOneShotAgent"] += 0.25
            evidence["RandomOneShotAgent"] += 0.08
            evidence["GreedyOneShotAgent"] -= 0.20
        if good_ratio >= 0.75 and price_flip_ratio <= 0.20:
            evidence["GreedyOneShotAgent"] += 0.85
            evidence["EqualDistOneShotAgent"] -= 0.35
            evidence["RandomOneShotAgent"] -= 0.20
            evidence["SyncRandomDistOneShotAgent"] -= 0.20
        if early_good_ratio >= 0.75 and late_good_ratio <= early_good_ratio - 0.35:
            evidence["GreedyOneShotAgent"] += 1.15
            evidence["SyncRandomDistOneShotAgent"] -= 0.15
            evidence["EqualDistOneShotAgent"] -= 0.15
            evidence["RandomOneShotAgent"] -= 0.25
        if previous_good_streak and last_two_bad and bad_to_good == 0:
            evidence["GreedyOneShotAgent"] += 0.75
            evidence["SyncRandomDistOneShotAgent"] -= 0.15
            evidence["RandomOneShotAgent"] -= 0.20

        if any(abs(value) > 0 for value in evidence.values()):
            self._add_evidence(partner, evidence, strength=0.6)

    # ---------------------------------------------------------------------
    # Strategy hooks
    # ---------------------------------------------------------------------

    def first_proposals(self):
        if self._exploration_enabled():
            base_proposals = super().first_proposals()
            proposals = self._exploration_first_proposals(base_proposals)
            self._record_sent_offers(proposals)
            return proposals

        proposals = {}
        for needs, all_partners in (
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ):
            partners = [partner for partner in all_partners if partner in self.negotiators]
            if not partners:
                continue
            proposals.update(self._greedy_only_first_proposals(int(needs), partners))

        self._record_sent_offers(proposals)
        return proposals

    def _exploration_first_proposals(self, base_proposals):
        proposals = {}
        for needs, all_partners in (
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ):
            partners = [partner for partner in all_partners if partner in self.negotiators]
            if not partners:
                continue
            if needs <= 0:
                proposals.update({partner: None for partner in partners})
                continue
            proposals.update(self._equal_dist_exploration_proposals(int(needs), partners))

        if proposals:
            return proposals
        return base_proposals

    def _equal_dist_exploration_proposals(self, needs: int, partners: list[str]):
        proposals = {partner: None for partner in partners}
        n = len(partners)
        target_quantity = max(0, math.ceil(int(needs) * self.exploration_quantity_multiplier))
        if target_quantity <= 0 or n <= 0:
            return proposals

        if target_quantity < n:
            for index, partner in enumerate(partners[:target_quantity]):
                proposals[partner] = self._offer(
                    partner,
                    1,
                    self._exploration_probe_price(partner, index),
                )
            return proposals

        base = max(1, target_quantity // n)
        if target_quantity >= 2 * n:
            base = max(base, 2)
        base = min(3, base)
        remainder = max(0, target_quantity - base * n)

        for index, partner in enumerate(partners):
            quantity = base
            if remainder > 0:
                extra = min(remainder, max(0, 3 - quantity))
                quantity += extra
                remainder -= extra
            proposals[partner] = self._offer(
                partner,
                self._clamp_quantity(partner, quantity),
                self._exploration_probe_price(partner, index),
            )
        return proposals

    def _exploration_probe_price(self, partner, index: int) -> int:
        # Alternate between prices that are good and bad for the opponent.  The
        # accept/reject pattern is especially informative for Greedy agents.
        opponent_good = (self.awi.current_step + index) % 2 == 0
        return self._worst_price_for_me(partner) if opponent_good else self._best_price_for_me(partner)

    def counter_all(self, offers, states):
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
                self._observe_own_first_offer_counter(
                    partner,
                    sent_offer,
                    offer,
                )
            self._observe_partner_end_response(
                partner,
                self._latest_sent_offer(partner),
                ended=False,
            )
            self._observe_received_first_offer(partner, offer)
            self._observe_offer(partner, offer, states)

        return self._current_offer_responses(offers, states)

    def _greedy_only_first_proposals(self, needs: int, partners: list[str]):
        proposals = {partner: None for partner in partners}
        if needs <= 0:
            return proposals

        opponent_types = self._ensure_min_dist_strategy_types(
            partners,
            {partner: self.opponent_type(partner) for partner in partners},
        )
        greedy_partners = [
            partner
            for partner in partners
            if opponent_types[partner] == "GreedyOneShotAgent"
        ]
        greedy_partners.sort(
            key=lambda partner: self.opponent_posteriors(partner).get(
                "GreedyOneShotAgent",
                0.0,
            ),
            reverse=True,
        )
        success_scaled_partners = self._success_scaled_partners(
            partners,
            opponent_types,
        )

        success_rate = self._non_greedy_initial_offer_success_rate()
        selected_greedy_partners = []
        strong_greedy_partners = [
            partner for partner in greedy_partners if self._strong_initial_greedy_partner(partner)
        ]
        if len(greedy_partners) >= 2 and len(strong_greedy_partners) >= 2:
            selected_greedy_partners = strong_greedy_partners[:2]
            greedy_quantities = [
                math.ceil(needs / 2),
                math.floor(needs / 2),
            ]
            for greedy_partner, greedy_quantity in zip(
                selected_greedy_partners,
                greedy_quantities,
                strict=False,
            ):
                proposals[greedy_partner] = self._offer(
                    greedy_partner,
                    greedy_quantity,
                    self._worst_price_for_me(greedy_partner),
                )
            return proposals

        if greedy_partners:
            if len(greedy_partners) >= 2:
                selected_greedy_partners = greedy_partners[:2]
                greedy_quantities = self._split_greedy_eighty_quantities(needs)
                scaled_target = self._success_adjusted_quantity(
                    needs * 0.2,
                    success_rate,
                )
            else:
                selected_greedy_partners = greedy_partners[:1]
                greedy_quantity = min(7, int(needs))
                greedy_quantities = [greedy_quantity]
                scaled_target = self._success_adjusted_quantity(
                    max(0, int(needs) - greedy_quantity),
                    success_rate,
                )

            for greedy_partner, greedy_quantity in zip(
                selected_greedy_partners,
                greedy_quantities,
                strict=False,
            ):
                proposals[greedy_partner] = self._offer(
                    greedy_partner,
                    greedy_quantity,
                    self._worst_price_for_me(greedy_partner),
                )
        else:
            scaled_target = self._success_adjusted_quantity(needs, success_rate)

        if len(greedy_partners) >= 3:
            for greedy_partner in greedy_partners[2:]:
                if greedy_partner not in success_scaled_partners:
                    success_scaled_partners.append(greedy_partner)

        if success_scaled_partners:
            scaled_target = self._minimum_two_dist_quantity(
                scaled_target,
                success_scaled_partners,
                opponent_types,
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

    def _success_scaled_partners(self, partners, opponent_types):
        sync_equal_types = {
            "SyncRandomDistOneShotAgent",
            "EqualDistOneShotAgent",
        }
        fallback_types = {"RandomOneShotAgent", "Other", "Unknown", "NonGreedy"}

        sync_equal_partners = [
            partner
            for partner in partners
            if opponent_types[partner] in sync_equal_types
        ]
        sync_equal_partners.sort(
            key=lambda partner: max(
                self.opponent_posteriors(partner).get(
                    "SyncRandomDistOneShotAgent",
                    0.0,
                ),
                self.opponent_posteriors(partner).get(
                    "EqualDistOneShotAgent",
                    0.0,
                ),
            ),
            reverse=True,
        )

        fallback_partners = [
            partner
            for partner in partners
            if opponent_types[partner] in fallback_types
        ]
        return sync_equal_partners + fallback_partners

    def _minimum_two_dist_quantity(self, target_quantity, partners, opponent_types):
        if not partners:
            return 0
        if int(target_quantity) <= 0:
            return 0
        return max(int(target_quantity), min(2, len(partners)))

    def _success_adjusted_quantity(self, target_quantity: float, success_rate: float) -> int:
        success_rate = max(0.05, min(1.0, float(success_rate)))
        return math.ceil(target_quantity / success_rate)

    def _split_greedy_eighty_quantities(self, needs: int) -> list[int]:
        total = max(0, math.ceil(int(needs) * 0.8))
        if total <= 0:
            return [0, 0]

        difference = 2 if total % 2 == 0 else 3
        if total < difference:
            return [total, 0]

        high = (total + difference) // 2
        low = total - high
        return [high, low]

    def _half_quantity_caps(self, needs: int, count: int):
        if needs <= 0 or count <= 0:
            return []

        low = needs // 2
        high = needs - low
        if count == 1:
            return [high]
        return [low if index % 2 == 0 else high for index in range(count)]

    def _assign_equal_quantities(self, proposals, partners, target_quantity, price_getter):
        partners = list(partners)
        if not partners or target_quantity <= 0:
            return

        base = target_quantity // len(partners)
        remainder = target_quantity - base * len(partners)
        for index, partner in enumerate(partners):
            quantity = base + (1 if index < remainder else 0)
            if quantity <= 0:
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
            self._assign_equal_quantities(proposals, partners, target_quantity, price_getter)
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
                if (
                    quantity_caps is not None
                    and quantities[index] >= quantity_caps[index]
                ):
                    continue
                quantities[index] += 1
                remainder -= 1
                changed = True
                if remainder <= 0:
                    break
            if not changed:
                break

        for partner, quantity in zip(partners, quantities, strict=False):
            if quantity <= 0:
                continue
            proposals[partner] = self._offer(
                partner,
                quantity,
                price_getter(partner),
            )

    def _add_equal_quantities(self, proposals, partners, target_quantity, price_getter):
        partners = list(partners)
        if not partners or target_quantity <= 0:
            return

        additions = {partner: 0 for partner in partners}
        base = target_quantity // len(partners)
        remainder = target_quantity - base * len(partners)
        for index, partner in enumerate(partners):
            additions[partner] = base + (1 if index < remainder else 0)

        for partner, addition in additions.items():
            if addition <= 0:
                continue
            current = proposals.get(partner)
            current_quantity = int(current[QUANTITY]) if current is not None else 0
            proposals[partner] = self._offer(
                partner,
                current_quantity + addition,
                price_getter(partner),
            )

    def _non_greedy_initial_offer_success_rate(self) -> float:
        if not self._non_greedy_initial_offer_results:
            return self.non_greedy_success_default
        return (
            sum(self._non_greedy_initial_offer_results)
            / len(self._non_greedy_initial_offer_results)
        )

    def _partner_non_greedy_initial_offer_success_rate(self, partner) -> float:
        results = self._partner_non_greedy_initial_offer_results.get(partner, [])
        if not results:
            return self._non_greedy_initial_offer_success_rate()
        return sum(results) / len(results)

    def _strong_initial_greedy_partner(self, partner) -> bool:
        greedy_probability = self.opponent_posteriors(partner).get(
            "GreedyOneShotAgent",
            0.0,
        )
        if greedy_probability < 0.80:
            return False

        streak = 0
        for item in reversed(self._own_offer_result_history.get(partner, [])):
            if not item.get("initial", False):
                continue
            if not item.get("accepted", False):
                break
            streak += 1
            if streak >= 3:
                return True
        return False

    def _record_non_greedy_initial_offer_result(self, sent_offer, accepted: bool):
        if not sent_offer or not sent_offer.get("non_greedy_initial_probe"):
            return
        partner = sent_offer.get("partner")
        self._non_greedy_initial_offer_results.append(bool(accepted))
        if len(self._non_greedy_initial_offer_results) > 100:
            del self._non_greedy_initial_offer_results[:-100]
        if partner is None:
            return
        partner_results = self._partner_non_greedy_initial_offer_results[partner]
        partner_results.append(bool(accepted))
        if len(partner_results) > 20:
            del partner_results[:-20]

    def _current_offer_responses(self, offers, states):
        base_responses = super().counter_all(offers, states)
        responses = dict(base_responses)
        t = self._relative_time(states)
        current_offers = {
            partner: offer
            for partner, offer in offers.items()
            if offer is not None and offer[TIME] == self.awi.current_step
        }

        for needs, all_partners, apply_eighty_percent_rule in (
            (self.awi.needed_supplies, self.awi.my_suppliers, True),
            (self.awi.needed_sales, self.awi.my_consumers, True),
        ):
            side_partners = [
                partner
                for partner in all_partners
                if partner in current_offers
            ]
            if not side_partners:
                continue

            seller_greedy_fill = None
            if all_partners == self.awi.my_consumers and not self._has_exact_offer_subset(
                side_partners,
                current_offers,
                int(needs),
            ):
                seller_greedy_fill = self._seller_greedy_fill_plan(
                    side_partners,
                    current_offers,
                    int(needs),
                )
            if seller_greedy_fill is not None:
                accepted_partners, greedy_partner, remaining_needs = seller_greedy_fill
                for partner in side_partners:
                    if partner in accepted_partners:
                        responses[partner] = SAOResponse(
                            ResponseType.ACCEPT_OFFER,
                            current_offers[partner],
                        )
                    elif partner == greedy_partner and remaining_needs > 0:
                        # Counter the greedy buyer to fill the remainder, but do
                        # not concede past the no-loss floor: clamp the (worst)
                        # price up to the break-even point if needed.
                        worst = self._worst_price_for_me(partner)
                        floor = self._no_loss_price_for_me(
                            partner,
                            remaining_needs,
                            {ap: current_offers[ap] for ap in accepted_partners},
                        )
                        fill_price = worst if floor is None else max(worst, floor)
                        counter_offer = self._raw_offer(
                            partner,
                            remaining_needs,
                            fill_price,
                        )
                        responses[partner] = (
                            self._unneeded_response()
                            if counter_offer is None
                            else SAOResponse(
                                ResponseType.REJECT_OFFER,
                                counter_offer,
                            )
                        )
                    else:
                        responses[partner] = self._unneeded_response()
                continue

            special_acceptance = None
            if apply_eighty_percent_rule and not self._has_exact_offer_subset(
                side_partners,
                current_offers,
                int(needs),
            ):
                special_acceptance = self._eighty_percent_acceptance_subset(
                    side_partners,
                    current_offers,
                    int(needs),
                )
            if special_acceptance is not None:
                accepted_partners = list(special_acceptance)
                accepted_quantity = sum(
                    int(current_offers[partner][QUANTITY])
                    for partner in accepted_partners
                )
                remaining_needs = max(0, int(needs) - accepted_quantity)
                remaining_partners = [
                    partner
                    for partner in side_partners
                    if partner not in accepted_partners
                ]
                opponent_types = self._strategy_opponent_types(side_partners)
                counter_partners = self._counter_remainder_partners(
                    remaining_partners,
                    opponent_types,
                    count=1,
                )

                for partner in accepted_partners:
                    responses[partner] = SAOResponse(
                        ResponseType.ACCEPT_OFFER,
                        current_offers[partner],
                    )

                counter_quantities = self._equal_counter_quantities(
                    remaining_needs,
                    counter_partners,
                )
                if len(counter_partners) == 1:
                    partner = counter_partners[0]
                    if t >= 0.95:
                        responses[partner] = self._final_single_partner_response(
                            partner,
                            current_offers[partner],
                            {
                                accepted_partner: current_offers[accepted_partner]
                                for accepted_partner in accepted_partners
                            },
                        )
                        for other in remaining_partners:
                            if other != partner:
                                responses[other] = self._unneeded_response()
                        continue
                    counter_quantities[partner] = self._conceded_counter_quantity(
                        partner,
                        counter_quantities.get(partner, 0),
                        current_offers[partner],
                        t,
                    )
                for partner in remaining_partners:
                    if partner not in counter_partners:
                        responses[partner] = self._unneeded_response()
                        continue
                    quantity = counter_quantities.get(partner, 0)
                    if quantity <= 0:
                        responses[partner] = self._unneeded_response()
                        continue
                    if len(counter_partners) == 1:
                        price = self._conceded_price_capped(
                            partner,
                            quantity,
                            t,
                            {ap: current_offers[ap] for ap in accepted_partners},
                        )
                    else:
                        price = self._best_price_for_me(partner)
                    counter_offer = self._offer(partner, quantity, price)
                    responses[partner] = self._counter_or_accept_response(
                        partner,
                        current_offers[partner],
                        counter_offer,
                    )
                continue

            accepted_partners = [
                partner
                for partner in side_partners
                if responses.get(partner) is not None
                and responses[partner].response == ResponseType.ACCEPT_OFFER
            ]
            accepted_quantity = sum(
                int(current_offers[partner][QUANTITY])
                for partner in accepted_partners
            )
            remaining_needs = max(0, int(needs) - accepted_quantity)
            counter_partners = [
                partner
                for partner in side_partners
                if partner not in accepted_partners
            ]

            if remaining_needs <= 0:
                for partner in counter_partners:
                    responses[partner] = self._unneeded_response()
                continue

            counter_quantities = self._equal_counter_quantities(
                remaining_needs,
                counter_partners,
            )
            if len(counter_partners) == 1:
                partner = counter_partners[0]
                if t >= 0.95:
                    responses[partner] = self._final_single_partner_response(
                        partner,
                        current_offers[partner],
                        {
                            accepted_partner: current_offers[accepted_partner]
                            for accepted_partner in accepted_partners
                        },
                    )
                    continue
                counter_quantities[partner] = self._conceded_counter_quantity(
                    partner,
                    counter_quantities.get(partner, 0),
                    current_offers[partner],
                    t,
                )
            for partner in counter_partners:
                quantity = counter_quantities.get(partner, 0)
                if quantity <= 0:
                    responses[partner] = self._unneeded_response()
                    continue
                if len(counter_partners) == 1:
                    price = self._conceded_price_capped(
                        partner,
                        quantity,
                        t,
                        {ap: current_offers[ap] for ap in accepted_partners},
                    )
                else:
                    price = self._best_price_for_me(partner)
                counter_offer = self._offer(partner, quantity, price)
                responses[partner] = self._counter_or_accept_response(
                    partner,
                    current_offers[partner],
                    counter_offer,
                )

        self._record_response_offers(
            responses,
            relative_time=self._relative_time(states),
        )
        return responses

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

            responses.update(
                self._probabilistic_counter_side(
                    int(needs),
                    side_partners,
                    current_offers,
                    t,
                )
            )

        self._record_response_offers(
            responses,
            relative_time=t,
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
            accept_bad_price_count = 0
            if accept_subset:
                accept_quantities = [
                    int(current_offers[partner][QUANTITY])
                    for partner in accept_subset
                ]
                accept_expected_units = [float(quantity) for quantity in accept_quantities]
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
                accept_bad_price_count = sum(
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
            bad_price_count = accept_bad_price_count + counter_bad_price_count
            offered_gap = abs(
                float(needs) - float(accepted_quantity + counter_offered_total)
            )
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
        best_prices = [self._best_price_for_me(partner) for partner in partners]
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

    def _accept_probability(self, partner, quantity: int, price: float) -> float:
        del quantity, price
        history = self._own_offer_result_history.get(partner, [])[-20:]
        if history:
            accepts = sum(bool(item.get("accepted", False)) for item in history)
            return (accepts + 1.0) / (len(history) + 2.0)
        return 0.5

    def _reoffer_probability(self, partner) -> float:
        total_reoffers = sum(
            len(quantities)
            for quantities in self._first_offer_counter_quantities_by_price[partner].values()
        )
        trials = sum(self._first_offer_trials_by_price[partner].values())
        accepts = sum(self._first_offer_accepts_by_price[partner].values())
        rejected = max(0, trials - accepts)
        if trials > 0:
            return (total_reoffers + 1.0) / (rejected + 2.0)
        return 0.5

    def _mean_reoffer_quantity(self, partner) -> float:
        quantities = []
        for values in self._first_offer_counter_quantities_by_price[partner].values():
            quantities.extend(values[-20:])
        if quantities:
            recent = quantities[-20:]
            return sum(recent) / len(recent)
        return 2.0

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
        del quantities, t
        if not partners:
            return 0.0
        value = 0.0
        for partner, expected in zip(partners, expected_units, strict=False):
            if expected <= 0:
                continue
            price = (
                float(prices[partner])
                if prices and partner in prices
                else float(self._best_price_for_me(partner))
            )
            value += float(expected) * self._price_for_me_score(partner, price)
        expected_total = max(0.0, sum(float(unit) for unit in expected_units))
        expected_gap = abs(float(needs) - expected_total)
        offered_gap = (
            abs(float(needs) - float(offered_total))
            if offered_total is not None
            else 0.0
        )
        return value - 0.05 * expected_gap - 0.02 * offered_gap

    def _conceded_counter_quantity(self, partner, desired_quantity: int, offer, t: float) -> int:
        desired_quantity = int(desired_quantity)
        if offer is None or t <= 0.5:
            return desired_quantity
        opponent_quantity = int(offer[QUANTITY])
        concession = max(0.0, min(1.0, (float(t) - 0.5) / 0.45))
        quantity = round(
            desired_quantity
            + (opponent_quantity - desired_quantity) * concession
        )
        return self._clamp_quantity(partner, quantity)

    def _final_single_partner_response(self, partner, offer, accepted_offers):
        accept_offers = dict(accepted_offers)
        accept_offers[partner] = offer
        reject_offers = dict(accepted_offers)
        try:
            accept_utility = self.ufun.from_offers(accept_offers)
            reject_utility = self.ufun.from_offers(reject_offers)
        except Exception:
            accept_utility = self._single_offer_profit_heuristic(partner, offer)
            reject_utility = 0

        if accept_utility >= reject_utility:
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
        return self._unneeded_response()

    def _single_offer_profit_heuristic(self, partner, offer) -> float:
        if offer is None:
            return 0.0
        quantity = int(offer[QUANTITY])
        price = float(offer[UNIT_PRICE])
        if self._is_seller_to(partner):
            return quantity * price
        return -quantity * price

    def _reachable_quantity_subsets(self, partners, offers, cap: int, better=None):
        """Subset-sum reachability with one witness subset per reachable total.

        Returns ``{total: tuple(partner_ids)}`` for every total in ``0..cap`` that
        some subset of ``partners`` can sum to (0/1 knapsack, each partner used at
        most once).  ``better(candidate, current)`` -> ``True`` replaces the stored
        witness for a tie on ``total`` (used for price / size tie-breaking).

        Runs in O(len(partners) * cap), replacing the O(2^n) ``combinations``
        enumeration used previously so the subset search is no longer exponential.
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

    def _seller_greedy_fill_plan(self, partners, offers, needs: int):
        if needs <= 0:
            return None

        greedy_partners = [
            partner
            for partner in partners
            if self._greedy_fill_probability(partner)
            >= self.seller_greedy_fill_threshold
        ]
        if not greedy_partners:
            return None

        greedy_partners.sort(
            key=self._greedy_fill_probability,
            reverse=True,
        )
        greedy_partner = greedy_partners[0]
        non_greedy_partners = [
            partner
            for partner in partners
            if partner != greedy_partner
        ]

        accepted_partners = self._max_under_needs_subset_for_seller(
            non_greedy_partners,
            offers,
            needs,
        )
        accepted_quantity = sum(
            int(offers[partner][QUANTITY])
            for partner in accepted_partners
        )
        remaining_needs = max(0, int(needs) - accepted_quantity)
        if remaining_needs <= 0:
            return None
        return set(accepted_partners), greedy_partner, remaining_needs

    def _greedy_fill_probability(self, partner) -> float:
        if partner in self._non_greedy_veto:
            return 0.0
        return self.opponent_posteriors(partner).get("GreedyOneShotAgent", 0.0)

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

    def _counter_remainder_partners(self, partners, opponent_types, count: int = 2):
        if not partners or count <= 0:
            return []
        preferred = self._success_scaled_partners(partners, opponent_types)
        selected = []
        for partner in preferred + list(partners):
            if partner in selected:
                continue
            selected.append(partner)
            if len(selected) >= count:
                break
        return selected

    def _equal_counter_quantities(self, needs: int, partners: list[str]) -> dict[str, int]:
        if not partners or needs <= 0:
            return {}
        quantities = {}
        remaining = int(needs)
        for index, partner in enumerate(partners):
            slots_left = len(partners) - index
            quantity = math.ceil(remaining / slots_left)
            quantity = self._clamp_quantity(partner, quantity)
            quantities[partner] = quantity
            remaining = max(0, remaining - quantity)
        return quantities

    def _unneeded_response(self):
        if self.awi.allow_zero_quantity:
            return SAOResponse(
                ResponseType.REJECT_OFFER,
                (0, self.awi.current_step, 0),
            )
        return SAOResponse(ResponseType.END_NEGOTIATION, None)

    def _type_based_first_proposals(self, needs: int, partners: list[str], base_proposals):
        opponent_types = self._strategy_opponent_types(partners)
        greedy_partners = self._ranked_partners_of_type(partners, "GreedyOneShotAgent", opponent_types)
        random_partners = self._ranked_partners_of_type(partners, "RandomOneShotAgent", opponent_types)
        small_dist_partners = [
            partner
            for partner in partners
            if opponent_types[partner] in {"SyncRandomDistOneShotAgent", "EqualDistOneShotAgent"}
        ]
        strategic_partners = set(greedy_partners[:2]) | set(random_partners) | set(small_dist_partners)
        if strategic_partners:
            proposals = {partner: None for partner in partners}
        else:
            proposals = {partner: base_proposals.get(partner) for partner in partners}
        remaining = max(0, int(needs))

        if greedy_partners:
            selected_greedy = greedy_partners[:2]
            if len(selected_greedy) == 1:
                quantities = [min(7, int(needs))]
            else:
                quantities = self._split_greedy_eighty_quantities(needs)
            for partner, quantity in zip(selected_greedy, quantities, strict=False):
                quantity = self._clamp_quantity(partner, quantity)
                proposals[partner] = self._offer(partner, quantity, self._worst_price_for_me(partner))
            remaining = max(0, remaining - sum(int(proposals[p][QUANTITY]) for p in selected_greedy if proposals[p] is not None))

        main_partners = [partner for partner in small_dist_partners if partner not in greedy_partners[:2]]
        if not greedy_partners and main_partners:
            self._assign_small_equal_quantities(
                proposals,
                main_partners,
                self._scaled_small_dist_quantity(remaining),
            )
            remaining = 0
        elif remaining > 0 and main_partners:
            self._assign_small_equal_quantities(
                proposals,
                main_partners,
                self._scaled_small_dist_quantity(remaining),
            )
            remaining = 0

        for partner in random_partners:
            if partner in greedy_partners[:2] or partner in main_partners:
                continue
            proposals[partner] = self._offer(partner, self._clamp_quantity(partner, 1), self._best_price_for_me(partner))

        for partner in partners:
            if proposals.get(partner) is None:
                continue
            quantity = int(proposals[partner][QUANTITY])
            if quantity <= 0 and not self.awi.allow_zero_quantity:
                proposals[partner] = None

        return proposals

    def _ranked_partners_of_type(self, partners, type_name: str, opponent_types=None) -> list[str]:
        if opponent_types is None:
            opponent_types = {partner: self.opponent_type(partner) for partner in partners}
        return sorted(
            [partner for partner in partners if opponent_types[partner] == type_name],
            key=lambda partner: self.opponent_posteriors(partner).get(type_name, 0.0),
            reverse=True,
        )

    def _assign_small_equal_quantities(self, proposals, partners: list[str], target_quantity: int):
        if not partners:
            return
        n = len(partners)
        if target_quantity <= 0:
            for partner in partners:
                proposals[partner] = self._offer(partner, 0, self._best_price_for_me(partner))
            return
        if target_quantity <= 3 * n:
            base = min(3, target_quantity // n)
            remainder = max(0, target_quantity - base * n)
        else:
            base = target_quantity // n
            remainder = target_quantity - base * n
        for index, partner in enumerate(partners):
            quantity = base + (1 if index < remainder else 0)
            proposals[partner] = self._offer(partner, self._clamp_quantity(partner, quantity), self._best_price_for_me(partner))

    def _scaled_small_dist_quantity(self, quantity: int) -> int:
        if quantity <= 0:
            return 0
        return max(1, math.ceil(quantity * self._small_dist_quantity_multiplier()))

    def _offer(self, partner, quantity: int, price: int) -> Outcome | None:
        quantity = self._role_biased_quantity(partner, quantity)
        quantity = self._clamp_quantity(partner, quantity)
        if quantity <= 0 and not self.awi.allow_zero_quantity:
            return None
        return (quantity, self.awi.current_step, int(price))

    def _raw_offer(self, partner, quantity: int, price: int) -> Outcome | None:
        quantity = self._clamp_quantity(partner, quantity)
        if quantity <= 0 and not self.awi.allow_zero_quantity:
            return None
        return (quantity, self.awi.current_step, int(price))

    def _role_biased_quantity(self, partner, quantity: int) -> int:
        # Sellers are penalized for over-commitment (shortfall) so bias
        # quantities down; buyers should avoid under-procurement so bias up.
        # Greedy partners keep their exact quantities; exploration probes are
        # left untouched to avoid distorting classification evidence.
        quantity = int(quantity)
        if quantity <= 0:
            return quantity
        if self._exploration_enabled():
            return quantity
        if self.opponent_type(partner) == "GreedyOneShotAgent":
            return quantity
        if self._is_seller_to(partner):
            return max(1, math.floor(quantity * self.seller_quantity_bias))
        return int(quantity * self.buyer_quantity_bias)

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

    def _adapt_proposals(self, proposals: dict[str, Outcome | None], t: float):
        adapted = dict(proposals)
        for partner, offer in proposals.items():
            if offer is None:
                continue
            opponent_type = self.opponent_type(partner)
            quantity = int(offer[QUANTITY])
            price = int(offer[UNIT_PRICE])

            if opponent_type == "RandomOneShotAgent":
                # Random agents sometimes accept extreme prices, so stay firm.
                price = self._best_price_for_me(partner)
            elif opponent_type == "GreedyOneShotAgent":
                price = self._worst_price_for_me(partner)
            elif opponent_type in {"SyncRandomDistOneShotAgent", "EqualDistOneShotAgent"}:
                price = self._best_price_for_me(partner)

            adapted[partner] = (quantity, self.awi.current_step, price)
        return adapted

    def _adapt_responses(self, responses, offers, states, t: float):
        adapted = dict(responses)
        for partner, response in list(adapted.items()):
            if response is None:
                continue
            offer = offers.get(partner)
            opponent_type = self.opponent_type(partner)

            if response.response == ResponseType.REJECT_OFFER and response.outcome is not None:
                quantity = int(response.outcome[QUANTITY])
                price = int(response.outcome[UNIT_PRICE])
                if opponent_type == "RandomOneShotAgent":
                    price = self._best_price_for_me(partner)
                elif opponent_type == "GreedyOneShotAgent":
                    price = self._worst_price_for_me(partner)
                adapted[partner] = SAOResponse(ResponseType.REJECT_OFFER, (quantity, self.awi.current_step, price))
                continue

        return adapted

    def _conceded_price_for_me(self, partner, t: float) -> int:
        best = self._best_price_for_me(partner)
        worst = self._worst_price_for_me(partner)
        concession = max(0.0, min(1.0, t + self.greedy_time_concession))
        if self._is_seller_to(partner):
            return int(round(best - (best - worst) * concession * 0.35))
        return int(round(best + (worst - best) * concession * 0.35))

    def _no_loss_price_for_me(self, partner, quantity: int, accepted_offers) -> int | None:
        """Worst (most-conceded) unit price at which proposing ``quantity`` to
        ``partner`` -- on top of ``accepted_offers`` -- is *not worse* than not
        making that deal, per the real ufun (disposal / shortfall / production
        costs included).  Returns ``None`` if the ufun is unavailable."""
        ufun = getattr(self, "ufun", None)
        if ufun is None or int(quantity) <= 0:
            return None
        issues = self._issues_for(partner)
        pmin = int(issues[UNIT_PRICE].min_value)
        pmax = int(issues[UNIT_PRICE].max_value)
        if pmax < pmin:
            return None
        step = self.awi.current_step
        accepted = dict(accepted_offers)
        try:
            base_value = ufun.from_offers(accepted)
        except Exception:
            return None
        seller = self._is_seller_to(partner)
        # iterate prices from worst-for-me to best-for-me; the first one whose
        # marginal utility is non-negative is exactly the concession floor.
        price_order = range(pmin, pmax + 1) if seller else range(pmax, pmin - 1, -1)
        for price in price_order:
            trial = dict(accepted)
            trial[partner] = (int(quantity), step, int(price))
            try:
                value = ufun.from_offers(trial)
            except Exception:
                return None
            if value >= base_value:
                return int(price)
        return None

    def _conceded_price_capped(self, partner, quantity: int, t: float, accepted_offers) -> int:
        """Time-based conceded price, but never conceded past the no-loss floor."""
        base = self._conceded_price_for_me(partner, t)
        floor = self._no_loss_price_for_me(partner, quantity, accepted_offers)
        if floor is None:
            return int(base)
        # seller: higher price is better, do not go below the floor.
        # buyer: lower price is better, do not go above the floor.
        if self._is_seller_to(partner):
            return max(int(base), floor)
        return min(int(base), floor)

    # ---------------------------------------------------------------------
    # Negotiation callbacks used as extra evidence
    # ---------------------------------------------------------------------

    def on_negotiation_success(self, contract: Contract, mechanism: Any):
        super().on_negotiation_success(contract, mechanism)
        partner = self._contract_partner(contract)
        outcome = self._contract_outcome(contract)
        if partner is None or outcome is None:
            return
        sent_offer = self._matching_sent_offer(partner, outcome)
        if sent_offer is not None:
            self._observe_partner_end_response(partner, sent_offer, ended=False)
            self._record_non_greedy_initial_offer_result(sent_offer, accepted=True)
            self._observe_own_offer_result(partner, sent_offer, accepted=True)

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        super().on_negotiation_failure(partners, annotation, mechanism, state)
        try:
            partner = next(partner for partner in partners if partner != self.id)
        except StopIteration:
            return
        sent_offer = self._latest_sent_offer(partner)
        if sent_offer is not None:
            if self._partner_ended_after_sent_offer(sent_offer, state):
                self._observe_partner_end_response(partner, sent_offer, ended=True)
            self._record_non_greedy_initial_offer_result(sent_offer, accepted=False)
            self._observe_own_offer_result(partner, sent_offer, accepted=False)

class HybridBayesianCautiousAgent(BayesianAgent022):
    """Greedy env / seller -> BayesianAgent022; NonGreedy buyer -> Cautious."""

    def __init__(
        self,
        *args,
        greedy_env_threshold: float = 0.5,
        min_env_classified: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.greedy_env_threshold = float(greedy_env_threshold)
        self.min_env_classified = int(min_env_classified)

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

    def on_negotiation_success(self, contract, mechanism):
        super().on_negotiation_success(contract, mechanism)
        # Port Cautious' per-partner volume tracking using *our* id.
        try:
            partner_id = next(p for p in contract.partners if p != self.id)
        except StopIteration:
            return
        tracker = getattr(self._cautious, "total_agreed_quantity", None)
        if tracker is not None and partner_id in tracker:
            tracker[partner_id] += int(contract.agreement["quantity"])

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
        # During exploration always use the Bayesian brain: its alternating-price
        # probes are what make the classification (and env detection) work.
        if self._exploration_enabled():
            return True
        environment_is_greedy = self._environment_is_greedy()
        is_buyer = not bool(getattr(self.awi, "is_first_level", False))
        if is_buyer and not environment_is_greedy:
            return False
        return True

    def first_proposals(self):
        if self._use_bayesian_brain():
            return super().first_proposals()
        proposals = self._cautious.first_proposals()
        self._record_sent_offers(proposals)
        return proposals

    def counter_all(self, offers, states):
        # Always feed the classifier, regardless of which brain answers.
        self._observe_incoming_offers(offers, states)
        if self._use_bayesian_brain():
            return self._current_offer_responses(offers, states)
        responses = self._cautious.counter_all(offers, states)
        self._record_response_offers(
            responses, relative_time=self._relative_time(states)
        )
        return responses

    def _observe_incoming_offers(self, offers, states):
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
                partner, self._latest_sent_offer(partner), ended=False
            )
            self._observe_received_first_offer(partner, offer)
            self._observe_offer(partner, offer, states)
