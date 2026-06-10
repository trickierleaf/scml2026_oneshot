from __future__ import annotations

import random

from typing import Any

from negmas import Contract, Outcome, SAOResponse, SAOState, ResponseType
from scml.common import distribute
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from scml.oneshot.agents.rand import SyncRandomOneShotAgent, powerset

class MyRLBestPriceAgent(SyncRandomOneShotAgent):
    """
    強化学習風 bandit 戦略エージェント

    方針:
    - equal=True
    - 相手ごとに bestprice 成立率を学習する
    - 探索:
        bestprice で数量1のオファーをばら撒く
    - 報酬:
        bestprice で取引成立したら successes +1
    - 有用度:
        q_value = successes / trials
    - 初手:
        needed * scatter_factor 人に bestprice で1個ずつオファー
    - counter_all:
        良い組み合わせがあれば受諾
        なければ、前半は bestprice、後半は random price で反対提案
    """

    def __init__(
        self,
        *args,
        scatter_factor: float = 1.3,
        epsilon: float = 0.2,
        explore_ratio: float = 0.25,
        early_time: float = 0.5,
        mismatch_exp: float = 5.0,
        mismatch_max: float = 0.3,
        **kwargs,
    ):
        super().__init__(*args, equal=True, mismatch_exp=mismatch_exp, mismatch_max=mismatch_max, **kwargs)

        # 初手で「必要量 × scatter_factor」人に1個ずつ投げる
        self.scatter_factor = scatter_factor

        # epsilon-greedy の探索率
        self.epsilon = epsilon

        # 初手で選ぶ人数のうち 25%程度を探索枠
        self.explore_ratio = explore_ratio

        # 交渉時間 t が early_time 未満なら bestprice、それ以降は random
        self.early_time = early_time

    def init(self):
        super().init()

        partners = list(
            dict.fromkeys(
                list(self.awi.my_suppliers) + list(self.awi.my_consumers)
            )
        )

        # trials: bestprice で数量1の探索オファーを送った回数
        # successes: bestprice で成立した回数
        # q_values: bestprice 成立率
        self._trials = {p: 0 for p in partners}
        self._successes = {p: 0 for p in partners}
        self._q_values = {p: 0.0 for p in partners}

    # ============================================================
    # price utilities
    # ============================================================

    def _is_seller_to(self, partner):
        """
        partner が consumer なら、自分は seller。
        partner が supplier なら、自分は buyer。
        """
        return partner in self.awi.my_consumers

    def _issues_for(self, partner):
        if self._is_seller_to(partner):
            return self.awi.current_output_issues
        return self.awi.current_input_issues

    def _best_price(self, partner):
        """
        自分にとって最良価格。
        売り手なら最高値、買い手なら最安値。
        """
        issues = self._issues_for(partner)
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        
        if self._is_seller_to(partner):
            return pmax
        return pmin

    def _worst_price(self, partner):
        """
        自分にとって最悪価格。
        売り手なら最安値、買い手なら最高値。
        """
        issues = self._issues_for(partner)
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value

        if self._is_seller_to(partner):
            return pmin
        return pmax

    def _random_price(self, partner):
        issues = self._issues_for(partner)
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        return random.randint(pmin, pmax)

    def _price_for_time(self, partner, t: float):
        """
        交渉前半は bestprice。
        後半は random price。
        """
        if t < self.early_time:
            return self._best_price(partner)
        if random.random() < 0.66:
            return self._best_price(partner)

        return self._worst_price(partner)

    # ============================================================
    # RL / bandit utilities
    # ============================================================

    def _ensure_partner(self, partner):
        if partner is None:
            return

        self._trials.setdefault(partner, 0)
        self._successes.setdefault(partner, 0)
        self._q_values.setdefault(partner, 0.0)

    def _update_q_value(self, partner):
        self._ensure_partner(partner)

        trials = self._trials.get(partner, 0)
        successes = self._successes.get(partner, 0)

        if trials <= 0:
            self._q_values[partner] = 0.0
        else:
            self._q_values[partner] = successes / trials

    def _select_partners_by_rl(self, partners, n_select):
        """
        探索ルール:
        1. trials == 0 の相手がいる場合は、まず未探索相手を優先して選ぶ
        2. すでに探索済みの相手だけになったら、
        q_value が高い有用な相手を主に選びつつ、
        一部はランダム探索する
        """
        if not partners or n_select <= 0:
            return []

        n_select = min(n_select, len(partners))

        untried = [p for p in partners if self._trials.get(p, 0) == 0]
        tried = [p for p in partners if self._trials.get(p, 0) > 0]

        selected = []

        # まだ一度も探索していない相手がいるなら、まずそこから試す
        if untried:
            random.shuffle(untried)
            selected.extend(untried[:n_select])

            if len(selected) >= n_select:
                return selected

        remaining_slots = n_select - len(selected)

        if remaining_slots <= 0:
            return selected

        # ここからは「有用な相手と取引しながら、少し他も探索」
        tried = [p for p in tried if p not in selected]

        if not tried:
            return selected

        n_explore = int(remaining_slots * self.explore_ratio)

        # epsilon によって探索量を少し増やす
        if random.random() < self.epsilon:
            n_explore = max(1, n_explore)

        n_explore = min(n_explore, len(tried), remaining_slots)
        n_exploit = remaining_slots - n_explore

        # 活用枠: q_value が高い相手
        exploit_partners = sorted(
            tried,
            key=lambda p: self._q_values.get(p, 0.0),
            reverse=True,
        )

        exploit_selected = exploit_partners[:n_exploit]
        selected.extend(exploit_selected)

        # 探索枠: まだ選ばれていない相手からランダム
        candidates = [p for p in tried if p not in selected]

        if n_explore > 0 and candidates:
            selected.extend(random.sample(candidates, min(n_explore, len(candidates))))

        return selected

    # ============================================================
    # contract / partner extraction
    # ============================================================

    def _partner_from_partners(self, partners):
        try:
            for p in partners:
                if p != self.id:
                    return p
        except Exception:
            pass
        return None

    def _partner_from_contract(self, contract):
        try:
            for p in contract.partners:
                if p != self.id:
                    return p
        except Exception:
            pass
        return None

    def _price_from_contract(self, contract):
        try:
            agreement = contract.agreement

            if isinstance(agreement, dict):
                if UNIT_PRICE in agreement:
                    return agreement[UNIT_PRICE]
                return agreement.get("unit_price", None)

            return agreement[UNIT_PRICE]
        except Exception:
            return None

    # ============================================================
    # learning callbacks
    # ============================================================

    def on_negotiation_success(self, contract, mechanism):
        super().on_negotiation_success(contract, mechanism)

        partner = self._partner_from_contract(contract)
        if partner is None:
            return

        self._ensure_partner(partner)

        price = self._price_from_contract(contract)
        if price is None:
            return

        # 報酬:
        # bestprice で成立した回数を成功として数える
        if price == self._best_price(partner):
            self._successes[partner] += 1

        self._update_q_value(partner)

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        super().on_negotiation_failure(partners, annotation, mechanism, state)

        partner = self._partner_from_partners(partners)
        if partner is None:
            return

        self._ensure_partner(partner)

        # trials は first_proposals で増やしている。
        # 失敗時は successes を増やさないので、成立率が下がる。
        self._update_q_value(partner)

    # ============================================================
    # distribute needs
    # ============================================================

    def distribute_needs(self, t: float) -> dict[str, int]:
        """
        必要量を、q_value が高い相手から優先して分配する。
        """
        dist = dict()

        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            partners = [_ for _ in all_partners if _ in self.negotiators.keys()]
            n_partners = len(partners)

            if n_partners == 0:
                continue

            if needs <= 0:
                dist.update({p: 0 for p in partners})
                continue

            target = int(needs * (1 + self._overordering_fraction(t)))
            target = max(1, target)

            # q_value が高い相手を優先
            partners = sorted(
                partners,
                key=lambda p: self._q_values.get(p, 0.0),
                reverse=True,
            )

            # allow_zero=False の distribute で破綻しないように、
            # 選ぶ人数を target 以下にする
            n_selected = min(len(partners), target)
            selected = partners[:n_selected]
            unselected = partners[n_selected:]

            quantities = distribute(
                target,
                len(selected),
                equal=True,
                allow_zero=self.awi.allow_zero_quantity,
            )

            dist.update({p: q for p, q in zip(selected, quantities)})

            for p in unselected:
                dist[p] = 0

        return dist

    # ============================================================
    # first proposals
    # ============================================================

    def first_proposals(self):
        """
        初手:
        必要量 × scatter_factor 人に、
        bestprice で数量1のオファーをばら撒く。

        例:
            needs = 5
            scatter_factor = 1.3

            int(5 * 1.3) = int(6.5) = 6

            つまり6人に bestprice で1個ずつ送る。
        """
        s = self.awi.current_step
        d = {}

        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            partners = [_ for _ in all_partners if _ in self.negotiators.keys()]

            if not partners:
                continue

            if needs <= 0:
                for p in partners:
                    d[p] = None
                continue

            n_scatter = int(needs * self.scatter_factor)
            n_scatter = max(1, min(n_scatter, len(partners)))

            selected = set(self._select_partners_by_rl(partners, n_scatter))

            for p in partners:
                if p in selected:
                    d[p] = (1, s, self._best_price(p))

                    # 探索オファーを出したので trials +1
                    self._ensure_partner(p)
                    self._trials[p] += 1
                    self._update_q_value(p)
                else:
                    d[p] = None

        return d

    # ============================================================
    # counter_all
    # ============================================================

    def counter_all(self, offers, states):
        response = dict()

        future_partners = {
            k for k, v in offers.items()
            if v[TIME] != self.awi.current_step
        }

        offers = {
            k: v for k, v in offers.items()
            if v[TIME] == self.awi.current_step
        }

        if len(states) == 0:
            t = 0.0
        else:
            t = min(_.relative_time for _ in states.values())

        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            partners = [_ for _ in all_partners if _ in offers.keys()]
            random.shuffle(partners)
            partners = set(partners)

            unneeded_response = (
                SAOResponse(ResponseType.END_NEGOTIATION, None)
                if not self.awi.allow_zero_quantity
                else SAOResponse(
                    ResponseType.REJECT_OFFER,
                    (0, self.awi.current_step, 0),
                )
            )

            # 全組み合わせを調べる
            plist = list(powerset(partners))[::-1]

            best_diff = float("inf")
            best_indx = -1
            best_q_value_sum = float("-inf")
            best_value_for_me = float("-inf")

            for i, partner_ids in enumerate(plist):
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = abs(offered - needs)

                q_value_sum = sum(
                    self._q_values.get(p, 0.0)
                    for p in partner_ids
                )

                value_for_me = 0
                for p in partner_ids:
                    q = offers[p][QUANTITY]
                    price = offers[p][UNIT_PRICE]

                    if self._is_seller_to(p):
                        value_for_me += q * price
                    else:
                        value_for_me -= q * price

                if (
                    diff < best_diff
                    or (
                        diff == best_diff
                        and q_value_sum > best_q_value_sum
                    )
                    or (
                        diff == best_diff
                        and q_value_sum == best_q_value_sum
                        and value_for_me > best_value_for_me
                    )
                ):
                    best_diff = diff
                    best_indx = i
                    best_q_value_sum = q_value_sum
                    best_value_for_me = value_for_me

            th = self._allowed_mismatch(t)

            # 良い組み合わせがあれば受諾
            if best_indx >= 0 and best_diff <= th:
                partner_ids = set(plist[best_indx])
                others = list(partners.difference(partner_ids).union(future_partners))

                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {
                    k: unneeded_response
                    for k in others
                }

                continue

            # 良い組み合わせがなければ、q_value が高い相手から分配して反対提案
            distribution = self.distribute_needs(t)

            for k, q in distribution.items():
                if k not in all_partners:
                    continue

                if q == 0:
                    response[k] = unneeded_response
                    continue

                price = self._price_for_time(k, t)

                response[k] = SAOResponse(
                    ResponseType.REJECT_OFFER,
                    (q, self.awi.current_step, price),
                )

        return response
    
class MyRLBestPriceAgent(SyncRandomOneShotAgent):
    """
    強化学習風 bandit 戦略エージェント

    方針:
    - equal=True
    - 相手ごとに bestprice 成立率を学習する
    - 探索:
        bestprice で数量1のオファーをばら撒く
    - 報酬:
        bestprice で取引成立したら successes +1
    - 有用度:
        q_value = successes / trials
    - 初手:
        needed * scatter_factor 人に bestprice で1個ずつオファー
    - counter_all:
        良い組み合わせがあれば受諾
        なければ、前半は bestprice、後半は random price で反対提案
    """

    def __init__(
        self,
        *args,
        scatter_factor: float = 1.3,
        epsilon: float = 0.2,
        explore_ratio: float = 0.25,
        early_time: float = 0.5,
        mismatch_exp: float = 5.0,
        mismatch_max: float = 0.3,
        **kwargs,
    ):
        super().__init__(*args, equal=True, mismatch_exp=mismatch_exp, mismatch_max=mismatch_max, **kwargs)

        # 初手で「必要量 × scatter_factor」人に1個ずつ投げる
        self.scatter_factor = scatter_factor

        # epsilon-greedy の探索率
        self.epsilon = epsilon

        # 初手で選ぶ人数のうち 25%程度を探索枠
        self.explore_ratio = explore_ratio

        # 交渉時間 t が early_time 未満なら bestprice、それ以降は random
        self.early_time = early_time

    def init(self):
        super().init()

        partners = list(
            dict.fromkeys(
                list(self.awi.my_suppliers) + list(self.awi.my_consumers)
            )
        )

        # trials: bestprice で数量1の探索オファーを送った回数
        # successes: bestprice で成立した回数
        # q_values: bestprice 成立率
        self._trials = {p: 0 for p in partners}
        self._successes = {p: 0 for p in partners}
        self._q_values = {p: 0.0 for p in partners}

    # ============================================================
    # price utilities
    # ============================================================

    def _is_seller_to(self, partner):
        """
        partner が consumer なら、自分は seller。
        partner が supplier なら、自分は buyer。
        """
        return partner in self.awi.my_consumers

    def _issues_for(self, partner):
        if self._is_seller_to(partner):
            return self.awi.current_output_issues
        return self.awi.current_input_issues

    def _best_price(self, partner):
        """
        自分にとって最良価格。
        売り手なら最高値、買い手なら最安値。
        """
        issues = self._issues_for(partner)
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        
        if self._is_seller_to(partner):
            return pmax
        return pmin

    def _worst_price(self, partner):
        """
        自分にとって最悪価格。
        売り手なら最安値、買い手なら最高値。
        """
        issues = self._issues_for(partner)
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value

        if self._is_seller_to(partner):
            return pmin
        return pmax

    def _random_price(self, partner):
        issues = self._issues_for(partner)
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        return random.randint(pmin, pmax)

    def _price_for_time(self, partner, t: float):
        """
        交渉前半は bestprice。
        後半も基本は bestprice だが、ときどき worst price に譲歩する。
        """
        if t < self.early_time:
            return self._best_price(partner)
        if random.random() < 0.66:
            return self._best_price(partner)

        return self._worst_price(partner)

    def _quantity_margin(self, partner, t: float):
        issues = self._issues_for(partner)
        qmin = issues[QUANTITY].min_value
        qmax = issues[QUANTITY].max_value
        span = max(0, qmax - qmin)
        remaining_time = max(0.0, min(1.0, 1.0 - t))
        return round(span * remaining_time)

    def _quantity_for_time(
        self,
        partner,
        target_quantity: int,
        t: float,
        opponent_offer: Outcome | None = None,
    ):
        if opponent_offer is None:
            quantity = target_quantity
        else:
            opponent_quantity = opponent_offer[QUANTITY]
            diff = target_quantity - opponent_quantity
            x = min(abs(diff), self._quantity_margin(partner, t))
            if diff > 0:
                quantity = opponent_quantity + x
            elif diff < 0:
                quantity = opponent_quantity - x
            else:
                quantity = opponent_quantity

        issues = self._issues_for(partner)
        qmin = issues[QUANTITY].min_value
        qmax = issues[QUANTITY].max_value
        return max(qmin, min(qmax, int(quantity)))

    # ============================================================
    # RL / bandit utilities
    # ============================================================

    def _ensure_partner(self, partner):
        if partner is None:
            return

        self._trials.setdefault(partner, 0)
        self._successes.setdefault(partner, 0)
        self._q_values.setdefault(partner, 0.0)

    def _update_q_value(self, partner):
        self._ensure_partner(partner)

        trials = self._trials.get(partner, 0)
        successes = self._successes.get(partner, 0)

        if trials <= 0:
            self._q_values[partner] = 0.0
        else:
            self._q_values[partner] = successes / trials

    def _select_partners_by_rl(self, partners, n_select):
        """
        探索ルール:
        1. trials == 0 の相手がいる場合は、まず未探索相手を優先して選ぶ
        2. すでに探索済みの相手だけになったら、
        q_value が高い有用な相手を主に選びつつ、
        一部はランダム探索する
        """
        if not partners or n_select <= 0:
            return []

        n_select = min(n_select, len(partners))

        untried = [p for p in partners if self._trials.get(p, 0) == 0]
        tried = [p for p in partners if self._trials.get(p, 0) > 0]

        selected = []

        # まだ一度も探索していない相手がいるなら、まずそこから試す
        if untried:
            random.shuffle(untried)
            selected.extend(untried[:n_select])

            if len(selected) >= n_select:
                return selected

        remaining_slots = n_select - len(selected)

        if remaining_slots <= 0:
            return selected

        # ここからは「有用な相手と取引しながら、少し他も探索」
        tried = [p for p in tried if p not in selected]

        if not tried:
            return selected

        n_explore = int(remaining_slots * self.explore_ratio)

        # epsilon によって探索量を少し増やす
        if random.random() < self.epsilon:
            n_explore = max(1, n_explore)

        n_explore = min(n_explore, len(tried), remaining_slots)
        n_exploit = remaining_slots - n_explore

        # 活用枠: q_value が高い相手
        exploit_partners = sorted(
            tried,
            key=lambda p: self._q_values.get(p, 0.0),
            reverse=True,
        )

        exploit_selected = exploit_partners[:n_exploit]
        selected.extend(exploit_selected)

        # 探索枠: まだ選ばれていない相手からランダム
        candidates = [p for p in tried if p not in selected]

        if n_explore > 0 and candidates:
            selected.extend(random.sample(candidates, min(n_explore, len(candidates))))

        return selected

    # ============================================================
    # contract / partner extraction
    # ============================================================

    def _partner_from_partners(self, partners):
        try:
            for p in partners:
                if p != self.id:
                    return p
        except Exception:
            pass
        return None

    def _partner_from_contract(self, contract):
        try:
            for p in contract.partners:
                if p != self.id:
                    return p
        except Exception:
            pass
        return None

    def _price_from_contract(self, contract):
        try:
            agreement = contract.agreement

            if isinstance(agreement, dict):
                if UNIT_PRICE in agreement:
                    return agreement[UNIT_PRICE]
                return agreement.get("unit_price", None)

            return agreement[UNIT_PRICE]
        except Exception:
            return None

    # ============================================================
    # learning callbacks
    # ============================================================

    def on_negotiation_success(self, contract, mechanism):
        super().on_negotiation_success(contract, mechanism)

        partner = self._partner_from_contract(contract)
        if partner is None:
            return

        self._ensure_partner(partner)

        price = self._price_from_contract(contract)
        if price is None:
            return

        # 報酬:
        # bestprice で成立した回数を成功として数える
        if price == self._best_price(partner):
            self._successes[partner] += 1

        self._update_q_value(partner)

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        super().on_negotiation_failure(partners, annotation, mechanism, state)

        partner = self._partner_from_partners(partners)
        if partner is None:
            return

        self._ensure_partner(partner)

        # trials は first_proposals で増やしている。
        # 失敗時は successes を増やさないので、成立率が下がる。
        self._update_q_value(partner)

    # ============================================================
    # distribute needs
    # ============================================================

    def _weighted_quantities(self, partners, quantity: int) -> list[int]:
        if not partners:
            return []

        if quantity <= 0:
            return [0] * len(partners)

        base = [0] * len(partners)
        remaining = quantity

        if not self.awi.allow_zero_quantity:
            for i in range(min(quantity, len(partners))):
                base[i] = 1
            remaining -= sum(base)

        if remaining <= 0:
            return base

        weights = [max(0.0, self._q_values.get(partner, 0.0)) for partner in partners]
        if sum(weights) <= 0:
            weights = [1.0] * len(partners)

        total_weight = sum(weights)
        exact = [remaining * weight / total_weight for weight in weights]
        additions = [int(value) for value in exact]
        leftover = remaining - sum(additions)

        order = sorted(
            range(len(partners)),
            key=lambda i: (exact[i] - additions[i], weights[i]),
            reverse=True,
        )
        for i in order[:leftover]:
            additions[i] += 1

        return [base[i] + additions[i] for i in range(len(partners))]

    def distribute_needs(self, t: float) -> dict[str, int]:
        """
        必要量を、q_value が高い相手に多めに分配する。
        """
        dist = dict()

        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            partners = [_ for _ in all_partners if _ in self.negotiators.keys()]
            n_partners = len(partners)

            if n_partners == 0:
                continue

            if needs <= 0:
                dist.update({p: 0 for p in partners})
                continue

            target = int(needs * (1 + self._overordering_fraction(t)))
            target = max(1, target)

            # q_value が高い相手を優先
            partners = sorted(
                partners,
                key=lambda p: self._q_values.get(p, 0.0),
                reverse=True,
            )

            # allow_zero=False の distribute で破綻しないように、
            # 選ぶ人数を target 以下にする
            n_selected = min(len(partners), target)
            selected = partners[:n_selected]
            unselected = partners[n_selected:]

            quantities = self._weighted_quantities(selected, target)

            dist.update({p: q for p, q in zip(selected, quantities)})

            for p in unselected:
                dist[p] = 0

        return dist

    # ============================================================
    # first proposals
    # ============================================================

    def first_proposals(self):
        """
        初手:
        必要量 × scatter_factor 人に、
        bestprice で数量1のオファーをばら撒く。

        例:
            needs = 5
            scatter_factor = 1.3

            int(5 * 1.3) = int(6.5) = 6

            つまり6人に bestprice で1個ずつ送る。
        """
        s = self.awi.current_step
        d = {}

        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            partners = [_ for _ in all_partners if _ in self.negotiators.keys()]

            if not partners:
                continue

            if needs <= 0:
                for p in partners:
                    d[p] = None
                continue

            n_scatter = int(needs * self.scatter_factor)
            n_scatter = max(1, min(n_scatter, len(partners)))

            selected = set(self._select_partners_by_rl(partners, n_scatter))

            for p in partners:
                if p in selected:
                    d[p] = (1, s, self._best_price(p))

                    # 探索オファーを出したので trials +1
                    self._ensure_partner(p)
                    self._trials[p] += 1
                    self._update_q_value(p)
                else:
                    d[p] = None

        return d

    # ============================================================
    # counter_all
    # ============================================================

    def counter_all(self, offers, states):
        response = dict()

        future_partners = {
            k for k, v in offers.items()
            if v[TIME] != self.awi.current_step
        }

        offers = {
            k: v for k, v in offers.items()
            if v[TIME] == self.awi.current_step
        }

        if len(states) == 0:
            t = 0.0
        else:
            t = min(_.relative_time for _ in states.values())

        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            partners = [_ for _ in all_partners if _ in offers.keys()]
            random.shuffle(partners)
            partners = set(partners)

            unneeded_response = (
                SAOResponse(ResponseType.END_NEGOTIATION, None)
                if not self.awi.allow_zero_quantity
                else SAOResponse(
                    ResponseType.REJECT_OFFER,
                    (0, self.awi.current_step, 0),
                )
            )

            # 全組み合わせを調べる
            plist = list(powerset(partners))[::-1]

            best_diff = float("inf")
            best_indx = -1
            best_q_value_sum = float("-inf")
            best_value_for_me = float("-inf")

            for i, partner_ids in enumerate(plist):
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = abs(offered - needs)

                q_value_sum = sum(
                    self._q_values.get(p, 0.0)
                    for p in partner_ids
                )

                value_for_me = 0
                for p in partner_ids:
                    q = offers[p][QUANTITY]
                    price = offers[p][UNIT_PRICE]

                    if self._is_seller_to(p):
                        value_for_me += q * price
                    else:
                        value_for_me -= q * price

                if (
                    diff < best_diff
                    or (
                        diff == best_diff
                        and q_value_sum > best_q_value_sum
                    )
                    or (
                        diff == best_diff
                        and q_value_sum == best_q_value_sum
                        and value_for_me > best_value_for_me
                    )
                ):
                    best_diff = diff
                    best_indx = i
                    best_q_value_sum = q_value_sum
                    best_value_for_me = value_for_me

            th = self._allowed_mismatch(t)

            # 良い組み合わせがあれば受諾
            if best_indx >= 0 and best_diff <= th:
                partner_ids = set(plist[best_indx])
                others = list(partners.difference(partner_ids).union(future_partners))

                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {
                    k: unneeded_response
                    for k in others
                }

                continue

            # 良い組み合わせがなければ、q_value が高い相手から分配して反対提案
            distribution = self.distribute_needs(t)

            for k, q in distribution.items():
                if k not in all_partners:
                    continue

                if q == 0:
                    response[k] = unneeded_response
                    continue

                quantity = self._quantity_for_time(k, q, t, offers.get(k))
                price = self._price_for_time(k, t)

                response[k] = SAOResponse(
                    ResponseType.REJECT_OFFER,
                    (quantity, self.awi.current_step, price),
                )

        return response

class MyClientOrientedAgent(SyncRandomOneShotAgent):
    def __init__(
        self,
        *args,
        prior_rate: float = 0.5,
        prior_strength: float = 3.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prior_rate = prior_rate
        self.prior_strength = prior_strength

    def init(self):
        super().init()

        partners = self._all_partners()
        self._good_signed_qty = {p: 0 for p in partners}
        self._first_offer_qty = {p: 0 for p in partners}
        self._client_score = {p: self.prior_rate for p in partners}
        self._first_offer_steps = {p: set() for p in partners}
        self._secured_supplies = 0
        self._secured_sales = 0
        self._needed_supplies_by_step = {}
        self._needed_sales_by_step = {}
        self._signed_supplies_by_step = {}
        self._signed_sales_by_step = {}

    # ============================================================
    # score bookkeeping
    # ============================================================

    def _all_partners(self):
        return list(
            dict.fromkeys(
                list(self.awi.my_suppliers) + list(self.awi.my_consumers)
            )
        )

    def _ensure_partner(self, partner):
        if partner is None:
            return
        self._good_signed_qty.setdefault(partner, 0)
        self._first_offer_qty.setdefault(partner, 0)
        self._client_score.setdefault(partner, self.prior_rate)
        self._first_offer_steps.setdefault(partner, set())

    def _update_client_score(self, partner):
        self._ensure_partner(partner)
        first_offer_qty = self._first_offer_qty.get(partner, 0)
        if first_offer_qty <= 0:
            self._client_score[partner] = self.prior_rate
            return
        self._client_score[partner] = (
            self._good_signed_qty.get(partner, 0) / first_offer_qty
        )

    def _is_seller_to(self, partner):
        return partner in self.awi.my_consumers

    def _issues_for(self, partner):
        if self._is_seller_to(partner):
            return self.awi.current_output_issues
        return self.awi.current_input_issues

    def _mean_price(self, partner):
        product = (
            self.awi.my_output_product
            if self._is_seller_to(partner)
            else self.awi.my_input_product
        )

        for attr in ("trading_prices", "catalog_prices"):
            prices = getattr(self.awi, attr, None)
            if prices is None:
                continue
            try:
                return prices[product]
            except Exception:
                pass

        issues = self._issues_for(partner)
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        return (pmin + pmax) / 2

    def _is_good_price(self, partner, price):
        mean_price = self._mean_price(partner)
        if self._is_seller_to(partner):
            return price > mean_price
        return price < mean_price

    def _offer_key(self, offer):
        if offer is None:
            return None
        return (offer[QUANTITY], offer[TIME], offer[UNIT_PRICE])

    def _response_type(self, response):
        return getattr(response, "response", getattr(response, "response_type", None))

    def _response_outcome(self, response):
        return getattr(response, "outcome", getattr(response, "offer", None))

    def _relative_time(self, states):
        if not states:
            return 0.0
        return min(state.relative_time for state in states.values())

    def _clamp(self, value, min_value, max_value):
        return max(min_value, min(max_value, value))

    def _counter_offer_around(self, partner, offer, t, base_quantity):
        issues = self._issues_for(partner)

        qmin = issues[QUANTITY].min_value
        qmax = issues[QUANTITY].max_value
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value

        concession = (1 - t) ** 3
        quantity_margin = round((qmax - qmin)  * 2 * concession)
        price_margin = round((pmax - pmin) * 2 * concession)

        offered_quantity = offer[QUANTITY]
        offered_price = offer[UNIT_PRICE]

        quantity_low = self._clamp(offered_quantity - quantity_margin, qmin, qmax)
        quantity_high = self._clamp(offered_quantity + quantity_margin, qmin, qmax)
        price_low = self._clamp(offered_price - price_margin, pmin, pmax)
        price_high = self._clamp(offered_price + price_margin, pmin, pmax)

        quantity = round(self._clamp(base_quantity, quantity_low, quantity_high))
        if self._is_seller_to(partner):
            price = round(price_high)
        else:
            price = round(price_low)

        return (quantity, self.awi.current_step, price)

    def _accepted_quantities_by_side(self, responses, offers):
        accepted_supplies = 0
        accepted_sales = 0

        for partner, response in responses.items():
            if self._response_type(response) != ResponseType.ACCEPT_OFFER:
                continue

            offer = offers.get(partner)
            if offer is None:
                continue

            if partner in self.awi.my_suppliers:
                accepted_supplies += offer[QUANTITY]
            elif partner in self.awi.my_consumers:
                accepted_sales += offer[QUANTITY]

        return accepted_supplies, accepted_sales

    def _remaining_needs_after_acceptances(self, partner, accepted_supplies, accepted_sales):
        if partner in self.awi.my_suppliers:
            return max(
                0,
                self.awi.needed_supplies
                - self._secured_supplies
                - accepted_supplies,
            )
        if partner in self.awi.my_consumers:
            return max(
                0,
                self.awi.needed_sales
                - self._secured_sales
                - accepted_sales,
            )
        return 0

    def _relative_client_scores(self, partners):
        scores = {
            partner: max(0.0, self._client_score.get(partner, self.prior_rate))
            for partner in partners
        }
        total = sum(scores.values())
        if total <= 0:
            return {partner: 1 / len(partners) for partner in partners}
        return {partner: score / total for partner, score in scores.items()}

    def _weighted_quantities(self, target, partners):
        if target <= 0 or not partners:
            return {partner: 0 for partner in partners}

        partners = list(partners)
        if not self.awi.allow_zero_quantity:
            partners = sorted(
                partners,
                key=lambda partner: self._client_score.get(partner, self.prior_rate),
                reverse=True,
            )
            selected = partners[: min(len(partners), target)]
            unselected = partners[len(selected):]
            quantities = {partner: 1 for partner in selected}
            quantities.update({partner: 0 for partner in unselected})
            remaining = target - len(selected)
            partners = selected
        else:
            quantities = {partner: 0 for partner in partners}
            remaining = target

        if remaining <= 0:
            return quantities

        weights = self._relative_client_scores(partners)
        raw_quantities = {
            partner: remaining * weights[partner]
            for partner in partners
        }
        extra_quantities = {
            partner: int(quantity)
            for partner, quantity in raw_quantities.items()
        }
        leftover = remaining - sum(extra_quantities.values())

        for partner in sorted(
            partners,
            key=lambda p: raw_quantities[p] - extra_quantities[p],
            reverse=True,
        )[:leftover]:
            extra_quantities[partner] += 1

        for partner, quantity in extra_quantities.items():
            quantities[partner] += quantity

        return quantities

    def _record_current_step_needs(self):
        step = self.awi.current_step
        self._needed_supplies_by_step[step] = self.awi.needed_supplies
        self._needed_sales_by_step[step] = self.awi.needed_sales

    def _recent_contract_prediction_multiplier(self, needed_by_step, signed_by_step):
        current_step = self.awi.current_step
        first_step = max(0, current_step - 10)
        recent_steps = range(first_step, current_step)

        needed_total = sum(needed_by_step.get(step, 0) for step in recent_steps)
        signed_total = sum(signed_by_step.get(step, 0) for step in recent_steps)

        if needed_total <= 0:
            return 1.0
        return needed_total / max(1, signed_total)

    def distribute_needs(self, t: float) -> dict[str, int]:
        distribution = {}

        for needs, secured, all_partners, needed_by_step, signed_by_step in [
            (
                self.awi.needed_supplies,
                self._secured_supplies,
                self.awi.my_suppliers,
                self._needed_supplies_by_step,
                self._signed_supplies_by_step,
            ),
            (
                self.awi.needed_sales,
                self._secured_sales,
                self.awi.my_consumers,
                self._needed_sales_by_step,
                self._signed_sales_by_step,
            ),
        ]:
            partners = [partner for partner in all_partners if partner in self.negotiators]
            remaining_needs = max(0, needs - secured)

            if remaining_needs <= 0:
                distribution.update({partner: 0 for partner in partners})
                continue

            prediction_multiplier = self._recent_contract_prediction_multiplier(
                needed_by_step,
                signed_by_step,
            )
            target = math.ceil(
                remaining_needs
                * prediction_multiplier
                * (1 + self._overordering_fraction(t))
            )
            distribution.update(self._weighted_quantities(target, partners))

        return distribution

    def _record_first_offer(self, partner, offer):
        if offer is None:
            return

        self._ensure_partner(partner)
        step = self.awi.current_step
        if step in self._first_offer_steps[partner]:
            return

        self._first_offer_steps[partner].add(step)
        self._first_offer_qty[partner] += offer[QUANTITY]
        self._update_client_score(partner)

    def _record_good_signed_contract(self, contract):
        partner = self._partner_from_contract(contract)
        if partner is None:
            return

        quantity = self._quantity_from_contract(contract)
        price = self._price_from_contract(contract)
        if quantity is None or price is None:
            return

        self._ensure_partner(partner)

        if not self._is_good_price(partner, price):
            return

        self._good_signed_qty[partner] += quantity
        self._update_client_score(partner)

    # ============================================================
    # contract extraction
    # ============================================================

    def _partner_from_contract(self, contract):
        try:
            for p in contract.partners:
                if p != self.id:
                    return p
        except Exception:
            pass
        return None

    def _agreement_value(self, contract, issue_index, issue_name):
        try:
            agreement = contract.agreement
            if isinstance(agreement, dict):
                if issue_index in agreement:
                    return agreement[issue_index]
                return agreement.get(issue_name, None)
            return agreement[issue_index]
        except Exception:
            return None

    def _quantity_from_contract(self, contract):
        return self._agreement_value(contract, QUANTITY, "quantity")

    def _time_from_contract(self, contract):
        value = self._agreement_value(contract, TIME, "time")
        if value is None:
            return self._agreement_value(contract, TIME, "delivery_step")
        return value

    def _price_from_contract(self, contract):
        return self._agreement_value(contract, UNIT_PRICE, "unit_price")

    # ============================================================
    # negotiation callbacks
    # ============================================================

    def first_proposals(self):
        proposals = super().first_proposals()
        for partner, offer in proposals.items():
            self._record_first_offer(partner, offer)
        return proposals

    def before_step(self):
        super().before_step()
        self._secured_supplies = 0
        self._secured_sales = 0
        self._record_current_step_needs()

    def counter_all(self, offers, states):
        t = self._relative_time(states)

        for partner, offer in offers.items():
            self._record_first_offer(partner, offer)

        responses = super().counter_all(offers, states)
        for partner, response in responses.items():
            response_type = self._response_type(response)
            outcome = self._response_outcome(response)
            offer = offers.get(partner)

            if response_type == ResponseType.ACCEPT_OFFER:
                continue

            if response_type != ResponseType.REJECT_OFFER:
                continue
            if offer is not None and outcome is not None and outcome[QUANTITY] > 0:
                accepted_supplies, accepted_sales = self._accepted_quantities_by_side(
                    responses, offers
                )
                remaining_needs = self._remaining_needs_after_acceptances(
                    partner, accepted_supplies, accepted_sales
                )
                base_quantity = min(outcome[QUANTITY], remaining_needs)

                if base_quantity <= 0:
                    responses[partner] = (
                        SAOResponse(ResponseType.END_NEGOTIATION, None)
                        if not self.awi.allow_zero_quantity
                        else SAOResponse(
                            ResponseType.REJECT_OFFER,
                            (0, self.awi.current_step, 0),
                        )
                    )
                    continue

                outcome = self._counter_offer_around(
                    partner, offer, t, base_quantity
                )
                responses[partner] = SAOResponse(ResponseType.REJECT_OFFER, outcome)
            self._record_first_offer(partner, outcome)
        return responses

    def on_negotiation_success(self, contract, mechanism):
        super().on_negotiation_success(contract, mechanism)
        partner = self._partner_from_contract(contract)
        quantity = self._quantity_from_contract(contract)
        if partner is not None and quantity is not None:
            if partner in self.awi.my_suppliers:
                self._secured_supplies += quantity
                self._signed_supplies_by_step[self.awi.current_step] = (
                    self._signed_supplies_by_step.get(self.awi.current_step, 0)
                    + quantity
                )
            elif partner in self.awi.my_consumers:
                self._secured_sales += quantity
                self._signed_sales_by_step[self.awi.current_step] = (
                    self._signed_sales_by_step.get(self.awi.current_step, 0)
                    + quantity
                )
        self._record_good_signed_contract(contract)

class MyNewAgent(EqualDistOneShotAgent):
    """EqualDistOneShotAgent that always proposes the best price for itself."""

    def _best_price(self, partner: str) -> int:
        issues = (
            self.awi.current_output_issues
            if partner in self.awi.my_consumers
            else self.awi.current_input_issues
        )
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        return pmax if partner in self.awi.my_consumers else pmin

    def first_proposals(self) -> dict[str, Outcome | None]:
        s = self.awi.current_step
        distribution = self.distribute_needs(t=0)
        return {
            k: (q, s, self._best_price(k))
            if q > 0 or self.awi.allow_zero_quantity
            else None
            for k, q in distribution.items()
        }

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        response = {}
        future_partners = {
            k for k, v in offers.items() if v[TIME] != self.awi.current_step
        }
        offers = {
            k: v for k, v in offers.items() if v[TIME] == self.awi.current_step
        }
        t = min(_.relative_time for _ in states.values())

        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            partners = [_ for _ in all_partners if _ in offers]
            random.shuffle(partners)
            partners = set(partners)

            plist = list(powerset(partners))[::-1]
            best_diff, best_indx = float("inf"), -1
            for i, partner_ids in enumerate(plist):
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = abs(offered - needs)
                if diff < best_diff:
                    best_diff, best_indx = diff, i
                if diff == 0:
                    break

            unneeded_response = (
                SAOResponse(ResponseType.END_NEGOTIATION, None)
                if not self.awi.allow_zero_quantity
                else SAOResponse(
                    ResponseType.REJECT_OFFER, (0, self.awi.current_step, 0)
                )
            )

            if best_diff <= self._allowed_mismatch(t):
                partner_ids = plist[best_indx]
                others = list(partners.difference(partner_ids).union(future_partners))
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | dict.fromkeys(others, unneeded_response)
                continue

            distribution = self.distribute_needs(t)
            response.update(
                {
                    k: (
                        unneeded_response
                        if q == 0
                        else SAOResponse(
                            ResponseType.REJECT_OFFER,
                            (q, self.awi.current_step, self._best_price(k)),
                        )
                    )
                    for k, q in distribution.items()
                    if k in all_partners
                }
            )

        return response

class MyGenuinOneshotAgent(MyNewAgent):
    """Agent that tries to finish deals in one shot using first-offer learning."""

    def init(self):
        super().init()
        self._first_offer_attempts: dict[str, int] = {}
        self._first_offer_successes: dict[str, int] = {}
        self._partner_revenue: dict[str, float] = {}
        self._last_day_success: dict[str, bool] = {}
        self._first_offers_by_day: dict[int, dict[str, Outcome]] = {}
        for partner in [*self.awi.my_suppliers, *self.awi.my_consumers]:
            self._ensure_partner(partner)

    def _ensure_partner(self, partner: str | None) -> None:
        if partner is None:
            return
        self._first_offer_attempts.setdefault(partner, 0)
        self._first_offer_successes.setdefault(partner, 0)
        self._partner_revenue.setdefault(partner, 0.0)
        self._last_day_success.setdefault(partner, False)

    def _partner_from_contract(self, contract: Contract) -> str | None:
        for partner in contract.partners:
            if partner != self.id:
                return partner
        return None

    def _partner_from_partners(self, partners) -> str | None:
        for partner in partners:
            if partner != self.id:
                return partner
        return None

    def _low_price(self, partner: str) -> int:
        issues = (
            self.awi.current_output_issues
            if partner in self.awi.my_consumers
            else self.awi.current_input_issues
        )
        return issues[UNIT_PRICE].min_value

    def _offer_price(self, partner: str) -> int:
        self._ensure_partner(partner)
        return (
            self._best_price(partner)
            if self._last_day_success.get(partner, False)
            else self._low_price(partner)
        )

    def _agreement_outcome(self, contract: Contract) -> Outcome:
        agreement = contract.agreement
        if isinstance(agreement, dict):
            return (
                agreement.get("quantity", agreement.get(QUANTITY, 0)),
                agreement.get("time", agreement.get(TIME, self.awi.current_step)),
                agreement.get("unit_price", agreement.get(UNIT_PRICE, 0)),
            )
        return agreement

    def _same_outcome(self, left: Outcome, right: Outcome) -> bool:
        return (
            left is not None
            and right is not None
            and tuple(left) == tuple(right)
        )

    def _first_offer_rate(self, partner: str) -> float:
        self._ensure_partner(partner)
        return (self._first_offer_successes[partner] + 1) / (
            self._first_offer_attempts[partner] + 2
        )

    def _partner_score(self, partner: str) -> float:
        self._ensure_partner(partner)
        return self._first_offer_rate(partner) * max(1.0, self._partner_revenue[partner])

    def _weighted_quantities(self, partners: list[str], target: int) -> dict[str, int]:
        if not partners:
            return {}
        if target <= 0:
            return {partner: 0 for partner in partners}

        selected = partners
        if not self.awi.allow_zero_quantity and target < len(partners):
            selected = sorted(partners, key=self._partner_score, reverse=True)[:target]

        scores = {partner: self._partner_score(partner) for partner in selected}
        total_score = sum(scores.values())
        if total_score <= 0:
            raw = {partner: target / len(selected) for partner in selected}
        else:
            raw = {partner: target * scores[partner] / total_score for partner in selected}

        quantities = {partner: int(raw[partner]) for partner in selected}
        if not self.awi.allow_zero_quantity:
            quantities = {partner: max(1, quantity) for partner, quantity in quantities.items()}

        remaining = target - sum(quantities.values())
        order = sorted(
            selected,
            key=lambda partner: raw[partner] - int(raw[partner]),
            reverse=True,
        )
        while remaining > 0 and order:
            for partner in order:
                if remaining <= 0:
                    break
                quantities[partner] += 1
                remaining -= 1
        while remaining < 0:
            removable = [
                partner
                for partner in sorted(selected, key=self._partner_score)
                if quantities[partner] > (0 if self.awi.allow_zero_quantity else 1)
            ]
            if not removable:
                break
            quantities[removable[0]] -= 1
            remaining += 1

        for partner in partners:
            quantities.setdefault(partner, 0)
        return quantities

    def distribute_needs(self, t: float) -> dict[str, int]:
        dist = {}
        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            partners = [_ for _ in all_partners if _ in self.negotiators]
            if needs <= 0:
                dist.update({partner: 0 for partner in partners})
                continue
            dist.update(self._weighted_quantities(partners, max(1, int(needs))))
        return dist

    def first_proposals(self) -> dict[str, Outcome | None]:
        step = self.awi.current_step
        distribution = self.distribute_needs(t=0)
        proposals = {
            partner: (quantity, step, self._offer_price(partner))
            if quantity > 0 or self.awi.allow_zero_quantity
            else None
            for partner, quantity in distribution.items()
        }
        self._first_offers_by_day[step] = {
            partner: offer for partner, offer in proposals.items() if offer is not None
        }
        for partner in self._first_offers_by_day[step]:
            self._ensure_partner(partner)
            self._first_offer_attempts[partner] += 1
        return proposals

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        response = {}
        future_partners = {
            partner
            for partner, offer in offers.items()
            if offer[TIME] != self.awi.current_step
        }
        offers = {
            partner: offer
            for partner, offer in offers.items()
            if offer[TIME] == self.awi.current_step
        }

        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            partners = {partner for partner in all_partners if partner in offers}
            if not partners:
                continue

            plist = list(powerset(partners))[::-1]
            best_diff, best_index = float("inf"), -1
            for index, partner_ids in enumerate(plist):
                offered = sum(offers[partner][QUANTITY] for partner in partner_ids)
                diff = abs(offered - needs)
                if diff < best_diff:
                    best_diff, best_index = diff, index
                if diff == 0:
                    break

            unneeded_response = (
                SAOResponse(ResponseType.END_NEGOTIATION, None)
                if not self.awi.allow_zero_quantity
                else SAOResponse(
                    ResponseType.REJECT_OFFER,
                    (0, self.awi.current_step, self._low_price(next(iter(partners)))),
                )
            )

            if best_diff == 0:
                accepted = set(plist[best_index])
                rejected = partners.difference(accepted).union(future_partners)
                response.update(
                    {
                        partner: SAOResponse(ResponseType.ACCEPT_OFFER, offers[partner])
                        for partner in accepted
                    }
                )
                response.update(dict.fromkeys(rejected, unneeded_response))
                continue

            acceptable_partners = [
                partner
                for partner in partners
                if needs * 0.7 <= offers[partner][QUANTITY] <= needs
            ]
            if acceptable_partners:
                best_partner = max(
                    acceptable_partners,
                    key=lambda partner: offers[partner][QUANTITY],
                )
                rejected = partners.difference({best_partner}).union(future_partners)
                response[best_partner] = SAOResponse(
                    ResponseType.ACCEPT_OFFER, offers[best_partner]
                )
                response.update(dict.fromkeys(rejected, unneeded_response))
                continue

            distribution = self.distribute_needs(t=min((_.relative_time for _ in states.values()), default=0.0))
            for partner in all_partners:
                if partner in response:
                    continue
                quantity = distribution.get(partner, 0)
                if quantity <= 0:
                    response[partner] = unneeded_response
                else:
                    response[partner] = SAOResponse(
                        ResponseType.REJECT_OFFER,
                        (quantity, self.awi.current_step, self._offer_price(partner)),
                    )

        return response

    def on_negotiation_success(self, contract: Contract, mechanism) -> None:
        super().on_negotiation_success(contract, mechanism)
        partner = self._partner_from_contract(contract)
        self._ensure_partner(partner)
        if partner is None:
            return
        agreement = self._agreement_outcome(contract)
        quantity = agreement[QUANTITY]
        unit_price = agreement[UNIT_PRICE]
        self._partner_revenue[partner] += quantity * unit_price
        self._last_day_success[partner] = True
        first_offer = self._first_offers_by_day.get(self.awi.current_step, {}).get(partner)
        if self._same_outcome(first_offer, agreement):
            self._first_offer_successes[partner] += 1

    def on_negotiation_failure(self, partners, annotation, mechanism, state) -> None:
        super().on_negotiation_failure(partners, annotation, mechanism, state)
        partner = self._partner_from_partners(partners)
        self._ensure_partner(partner)
        if partner is not None:
            self._last_day_success[partner] = False

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
            return 1.0
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

    def before_step(self):
        for partner in self._first_offers:
            self._ensure_partner(partner)
            self._half_offer_history[partner].append(False)
        super().before_step()
        self._first_offers = {}
        self._half_logic_used_partners = set()

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

            selected = set(self._select_first_offer_partners(partners))
            selected_in_offer_order = [partner for partner in partners if partner in selected]
            selected_quantities = dict(
                zip(
                    selected_in_offer_order,
                    self._half_quantities(needs, all_partners, len(selected_in_offer_order)),
                    strict=False,
                )
            )
            price = self._best_price(all_partners)

            for partner in partners:
                if partner in selected_quantities:
                    offer = (selected_quantities[partner], step, price)
                    proposals[partner] = offer
                    self._first_offers[partner] = offer
                    self._half_logic_used_partners.add(partner)
                else:
                    proposals[partner] = None

        return proposals

    def counter_all(self, offers, states):
        responses = super().counter_all(offers, states)
        current_offers = {
            partner: offer
            for partner, offer in offers.items()
            if offer is not None and offer[TIME] == self.awi.current_step
        }

        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            partners = [
                partner
                for partner in all_partners
                if partner in current_offers
                and partner not in self._half_logic_used_partners
            ]
            if not partners or needs <= 0:
                continue

            price = self._best_price(all_partners)
            quantities = self._half_quantities(needs, all_partners, len(partners))
            for partner, quantity in zip(partners, quantities, strict=False):
                offer = (quantity, self.awi.current_step, price)
                responses[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)
                self._first_offers[partner] = offer
                self._half_logic_used_partners.add(partner)

        return responses

    def on_negotiation_success(self, contract, mechanism):
        super().on_negotiation_success(contract, mechanism)
        partner = self._partner_from_contract(contract)
        if partner is None:
            return

        agreement = self._contract_offer(contract)
        self._ensure_partner(partner)
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

class PenaltyAwareDelayAgent(SyncRandomOneShotAgent):
    """Profit-based OneShot agent that explicitly prices quantity penalties."""

    def init(self) -> None:
        super().init()
        self.best_offer_so_far: dict[str, Outcome] = {}
        self.best_offer_profit_so_far: dict[str, float] = {}
        self.received_offers: dict[str, list[Outcome]] = defaultdict(list)
        self.accepted_contracts: list[tuple[str, Outcome]] = []
        self.secured_contracts = self.accepted_contracts
        self._pending_acceptances: dict[str, Outcome] = {}
        self._round_active_partners: set[str] = set()
        self.partner_count = 0

    def before_step(self) -> None:
        super().before_step()
        self.best_offer_so_far = {}
        self.best_offer_profit_so_far = {}
        self.received_offers = defaultdict(list)
        self.accepted_contracts = []
        self.secured_contracts = self.accepted_contracts
        self._pending_acceptances = {}
        self._round_active_partners = set()
        self.partner_count = len(getattr(self, "negotiators", {}) or {})

    # ------------------------------------------------------------------
    # Basic market helpers
    # ------------------------------------------------------------------
    def _partners(self) -> list[str]:
        return list(dict.fromkeys(list(self.awi.my_suppliers) + list(self.awi.my_consumers)))

    def _is_seller_to(self, partner_id: str | None) -> bool:
        return partner_id in set(self.awi.my_consumers)

    def _issues_for(self, partner_id: str | None):
        if self._is_seller_to(partner_id):
            return self.awi.current_output_issues
        return self.awi.current_input_issues

    def _issue_bounds(self, partner_id: str | None, issue: int) -> tuple[int, int]:
        try:
            item = self._issues_for(partner_id)[issue]
            return int(item.min_value), int(item.max_value)
        except Exception:
            return 1, 1

    def _clamp_issue(self, partner_id: str | None, issue: int, value: float) -> int:
        low, high = self._issue_bounds(partner_id, issue)
        return max(low, min(high, int(round(value))))

    def _get_round(self, state: SAOState | None = None) -> int:
        for name in ("step", "current_offer_index", "n_acceptances"):
            value = getattr(state, name, None)
            if isinstance(value, int):
                return value
        return 0

    def _offer_tuple(self, offer: Outcome | None) -> Outcome | None:
        if offer is None:
            return None
        try:
            return (int(offer[QUANTITY]), int(offer[TIME]), int(offer[UNIT_PRICE]))
        except Exception:
            return None

    def _quantity(self, offer: Outcome | None) -> int:
        try:
            return max(0, int(offer[QUANTITY]))  # type: ignore[index]
        except Exception:
            return 0

    def _price(self, offer: Outcome | None) -> float:
        try:
            return float(offer[UNIT_PRICE])  # type: ignore[index]
        except Exception:
            return 0.0

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
        for name in ("profile", "current_production_cost"):
            value = getattr(self.awi, name, None)
            try:
                if name == "profile":
                    return float(value.cost)
                return float(value)
            except Exception:
                pass
        return 0.0

    def _max_lines(self) -> int:
        return max(0, int(getattr(self.awi, "n_lines", getattr(self.awi, "max_n_lines", 0)) or 0))

    # ------------------------------------------------------------------
    # Profit and offer evaluation
    # ------------------------------------------------------------------
    def _contract_side(self, partner_id: str) -> bool:
        """Return True for sales, False for supplies."""
        return self._is_seller_to(partner_id)

    def estimate_profit(self, contracts: list[tuple[str, Outcome]] | None = None) -> float:
        """Estimate daily profit including production, disposal and shortfall."""
        if contracts is None:
            contracts = self._current_contracts()
        total_input = total_output = 0
        buy_cost = sell_revenue = 0.0

        for partner_id, offer in contracts:
            q = self._quantity(offer)
            p = self._price(offer)
            if self._contract_side(partner_id):
                total_output += q
                sell_revenue += q * p
            else:
                total_input += q
                buy_cost += q * p

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

    def evaluate_offer(self, offer: Outcome | None, partner_id: str) -> tuple[float, float, float]:
        offer = self._offer_tuple(offer)
        if offer is None:
            return -math.inf, self.estimate_profit(), -math.inf
        contracts = self._current_contracts()
        now = self.estimate_profit()
        after = self.estimate_profit(contracts + [(partner_id, offer)])
        return after - now, after, now

    def _current_contracts(self) -> list[tuple[str, Outcome]]:
        return self.accepted_contracts + list(self._pending_acceptances.items())

    def accept_threshold(self, round_: int) -> float:
        price_scale = max(self._trading_price(False), self._trading_price(True), 1.0)
        if round_ < 3:
            return 1.5 * price_scale
        if round_ < 10:
            return 0.75 * price_scale
        if round_ < 18:
            return 0.2 * price_scale
        return -0.1 * price_scale

    def remaining_need(self, partner_id: str | None = None) -> int:
        """Remaining quantity needed on the side of this negotiation."""
        if self._is_seller_to(partner_id):
            base_need = min(getattr(self.awi, "needed_sales", 0) or 0, self._max_lines())
            return max(0, int(base_need))
        base_need = min(getattr(self.awi, "needed_supplies", 0) or 0, self._max_lines())
        return max(0, int(base_need))

    def _side_need(self, seller_side: bool) -> int:
        if seller_side:
            return max(0, int(min(getattr(self.awi, "needed_sales", 0) or 0, self._max_lines())))
        return max(0, int(min(getattr(self.awi, "needed_supplies", 0) or 0, self._max_lines())))

    def _side_partner_count(self, partner_id: str) -> int:
        """Count active partners on the same side as this negotiation."""
        negotiators = self._current_active_partner_ids()
        partners = self.awi.my_consumers if self._is_seller_to(partner_id) else self.awi.my_suppliers
        return len([partner for partner in partners if partner in negotiators])

    def _current_active_partner_ids(self) -> set[str]:
        """Partners still negotiating now; ended partners are intentionally excluded."""
        if self._round_active_partners:
            return set(self._round_active_partners)
        active = getattr(self, "active_negotiators", None)
        if active is not None:
            return set(active.keys() if hasattr(active, "keys") else active)
        negotiators = getattr(self, "negotiators", {})
        return set(negotiators.keys() if hasattr(negotiators, "keys") else negotiators)

    def distribute_needs(self, round_: int = 0) -> dict[str, int]:
        """Distribute current needs almost equally over active partners."""
        quantities: dict[str, int] = {}
        negotiators = self._current_active_partner_ids()

        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            partners = [partner for partner in all_partners if partner in negotiators]
            if not partners:
                continue

            need = max(0, int(min(needs or 0, self._max_lines())))
            if need <= 0:
                quantities.update({partner: 0 for partner in partners})
                continue

            _, qmax = self._issue_bounds(partners[0], QUANTITY)
            target = self._distribution_target(need, round_)
            shares = distribute(
                target,
                len(partners),
                mx=qmax,
                equal=True,
                allow_zero=self.awi.allow_zero_quantity,
            )
            quantities.update(dict(zip(partners, shares, strict=False)))

        return quantities

    def _distribution_target(self, need: int, round_: int) -> int:
        """Offer a little more early, then converge back to the true need."""
        if need <= 0:
            return 0
        if round_ < 3:
            multiplier = 1.35
        elif round_ < 10:
            multiplier = 1.20
        elif round_ < 18:
            multiplier = 1.08
        else:
            multiplier = 1.0
        return max(1, math.ceil(need * multiplier))

    def _quantity_fit_bonus(self, offer: Outcome | None, partner_id: str) -> float:
        need = self.remaining_need(partner_id)
        q = self._quantity(offer)
        if need <= 0:
            return -q
        return -abs(need - q) / max(need, 1)

    def _helps_quantity_adjustment(self, offer: Outcome | None, partner_id: str) -> bool:
        need = self.remaining_need(partner_id)
        q = self._quantity(offer)
        if need <= 0:
            return q == 0
        tolerance = max(1, math.ceil(0.25 * need))
        return q <= need and abs(need - q) <= tolerance

    def reduces_penalty(self, offer: Outcome | None, partner_id: str) -> bool:
        offer = self._offer_tuple(offer)
        if offer is None:
            return False
        contracts = self._current_contracts()
        before = self._imbalance(contracts)
        after = self._imbalance(contracts + [(partner_id, offer)])
        return after < before

    def _imbalance(self, contracts: list[tuple[str, Outcome]]) -> float:
        inputs = sum(self._quantity(o) for p, o in contracts if not self._contract_side(p))
        outputs = sum(self._quantity(o) for p, o in contracts if self._contract_side(p))
        produced = min(inputs, outputs, self._max_lines())
        disposal = float(getattr(self.awi, "current_disposal_cost", 0.0) or 0.0)
        shortfall = float(getattr(self.awi, "current_shortfall_penalty", 0.0) or 0.0)
        return (
            max(0, inputs - produced) * disposal * self._trading_price(False)
            + max(0, outputs - produced) * shortfall * self._trading_price(True)
        )

    def update_best_offer(self, partner_id: str, offer: Outcome | None) -> None:
        offer = self._offer_tuple(offer)
        if offer is None:
            return
        self.received_offers[partner_id].append(offer)
        _, profit_after, _ = self.evaluate_offer(offer, partner_id)
        if profit_after > self.best_offer_profit_so_far.get(partner_id, -math.inf):
            self.best_offer_so_far[partner_id] = offer
            self.best_offer_profit_so_far[partner_id] = profit_after

    def should_delay(
        self,
        partner_count: int,
        round_: int,
        offer_quality: float,
        remaining_need: int,
        offer: Outcome | None,
        partner_id: str,
    ) -> bool:
        if partner_count > 2 or round_ < 3 or round_ >= 18:
            return False
        if offer_quality >= 2.0 * max(self._trading_price(False), self._trading_price(True), 1.0):
            return False
        if self.reduces_penalty(offer, partner_id):
            return False
        if remaining_need > 0 and self._helps_quantity_adjustment(offer, partner_id):
            return False
        return True

    def should_accept(self, offer: Outcome | None, partner_id: str, state: SAOState | None) -> bool:
        offer = self._offer_tuple(offer)
        if offer is None:
            return False
        round_ = self._get_round(state)
        gain, profit_after, profit_now = self.evaluate_offer(offer, partner_id)
        partner_count = self._side_partner_count(partner_id)

        if round_ >= 18:
            best = self.best_offer_so_far.get(partner_id, offer)
            best_profit = self.best_offer_profit_so_far.get(partner_id, profit_after)
            emergency_limit = profit_now - 0.5 * max(self._trading_price(False), self._trading_price(True), 1.0)
            return offer == best and best_profit >= emergency_limit

        if self.should_delay(partner_count, round_, gain, self.remaining_need(partner_id), offer, partner_id):
            return False

        threshold = self.accept_threshold(round_)
        if self.reduces_penalty(offer, partner_id):
            threshold -= 0.5 * max(self._trading_price(False), self._trading_price(True), 1.0)

        if self._quantity_fit_bonus(offer, partner_id) < -0.8 and not self.reduces_penalty(offer, partner_id):
            threshold += 0.4 * max(self._trading_price(False), self._trading_price(True), 1.0)

        return gain >= threshold

    # ------------------------------------------------------------------
    # Proposals and responses
    # ------------------------------------------------------------------
    def make_counter_offer(self, partner_id: str, round_: int) -> Outcome | None:
        if round_ >= 18 and partner_id in self.best_offer_so_far:
            return self.best_offer_so_far[partner_id]

        quantity = self._counter_quantity(partner_id, round_)
        qmin, qmax = self._issue_bounds(partner_id, QUANTITY)
        quantity = max(qmin, min(qmax, quantity))

        pmin, pmax = self._issue_bounds(partner_id, UNIT_PRICE)
        concession = min(max(round_, 0), 20) / 20.0
        if self._is_seller_to(partner_id):
            target = pmax - concession * 0.45 * (pmax - pmin)
        else:
            target = pmin + concession * 0.45 * (pmax - pmin)
        price = max(pmin, min(pmax, int(round(target))))
        return (quantity, int(self.awi.current_step), price)

    def _counter_quantity(self, partner_id: str, round_: int) -> int:
        """Start from equal distribution, then move toward the partner's quantity."""
        distributed = self.distribute_needs(round_).get(partner_id, 0)
        qmin, qmax = self._issue_bounds(partner_id, QUANTITY)
        if distributed <= 0:
            distributed = 0 if self.awi.allow_zero_quantity else qmin

        concession = min(max(round_, 0), 20) / 20.0
        quantity_weight = 0.15 + 0.65 * concession
        recent_offer = self.received_offers.get(partner_id, [])[-1:] or []
        if recent_offer:
            partner_quantity = self._quantity(recent_offer[0])
            target = (1.0 - quantity_weight) * distributed + quantity_weight * partner_quantity
        else:
            target = distributed

        need = self.remaining_need(partner_id)
        if need > 0:
            # Keep the quantity useful for our remaining need even when conceding.
            target = min(target, need)
        return max(qmin, min(qmax, int(round(target))))

    def first_proposals(self) -> dict[str, Outcome | None]:
        self.partner_count = len(getattr(self, "active_negotiators", {}) or getattr(self, "negotiators", {}) or {})
        proposals: dict[str, Outcome | None] = {}
        distribution = self.distribute_needs(0)
        for partner_id, quantity in distribution.items():
            if quantity <= 0 and not self.awi.allow_zero_quantity:
                proposals[partner_id] = None
                continue
            proposals[partner_id] = self.make_counter_offer(partner_id, 0)
        return proposals

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        return self.make_counter_offer(negotiator_id, self._get_round(state))

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        offer = getattr(state, "current_offer", None)
        self.update_best_offer(negotiator_id, offer)
        if self.should_accept(offer, negotiator_id, state):
            accepted = self._offer_tuple(offer)
            if accepted is not None:
                self._pending_acceptances[negotiator_id] = accepted
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        responses: dict[str, SAOResponse] = {}
        self.partner_count = len(offers)
        self._round_active_partners = set(offers)

        for partner_id, offer in offers.items():
            self.update_best_offer(partner_id, offer)

        accepted_partners = self._select_synchronized_acceptances(offers, states)
        for partner_id, offer in offers.items():
            if partner_id in accepted_partners:
                accepted = self._offer_tuple(offer)
                if accepted is not None:
                    self._pending_acceptances[partner_id] = accepted
                responses[partner_id] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
            else:
                state = states.get(partner_id)
                counter = self.make_counter_offer(partner_id, self._get_round(state))
                responses[partner_id] = SAOResponse(ResponseType.REJECT_OFFER, counter)
        return responses

    def _select_synchronized_acceptances(
        self,
        offers: dict[str, Outcome],
        states: dict[str, SAOState],
    ) -> set[str]:
        """Choose a same-round set of acceptances without overshooting needs."""
        accepted: set[str] = set()
        base_contracts = self.accepted_contracts + list(self._pending_acceptances.items())
        for seller_side in (False, True):
            partners = [
                partner
                for partner, offer in offers.items()
                if self._is_seller_to(partner) == seller_side
                and self._offer_tuple(offer) is not None
                and self._offer_tuple(offer)[TIME] == self.awi.current_step  # type: ignore[index]
                and self.should_accept(offer, partner, states.get(partner))
            ]
            if not partners:
                continue

            need = self._side_need(seller_side)
            if need <= 0:
                continue
            best_subset: tuple[str, ...] = tuple()
            best_profit = self.estimate_profit(base_contracts)
            best_quantity_gap = need
            for subset in self._candidate_subsets(partners, offers):
                quantity = sum(self._quantity(offers[partner]) for partner in subset)
                if need > 0 and quantity > need:
                    continue
                contracts = base_contracts + [
                    (partner, self._offer_tuple(offers[partner]))  # type: ignore[arg-type]
                    for partner in subset
                ]
                profit = self.estimate_profit(contracts)
                quantity_gap = abs(need - quantity)
                if (
                    profit > best_profit
                    or (math.isclose(profit, best_profit) and quantity_gap < best_quantity_gap)
                ):
                    best_subset = tuple(subset)
                    best_profit = profit
                    best_quantity_gap = quantity_gap

            for partner in best_subset:
                accepted.add(partner)
                offer = self._offer_tuple(offers[partner])
                if offer is not None:
                    base_contracts.append((partner, offer))
        return accepted

    def _candidate_subsets(self, partners: list[str], offers: dict[str, Outcome]):
        """Enumerate all subsets for small sets, good greedy prefixes for large sets."""
        if len(partners) <= 12:
            for size in range(1, len(partners) + 1):
                yield from itertools.combinations(partners, size)
            return

        ranked = sorted(
            partners,
            key=lambda partner: self.evaluate_offer(offers.get(partner), partner)[0],
            reverse=True,
        )
        for size in range(1, min(len(ranked), 12) + 1):
            yield tuple(ranked[:size])

    # ------------------------------------------------------------------
    # Contract bookkeeping
    # ------------------------------------------------------------------
    def _partner_from_contract(self, contract: Contract) -> str | None:
        try:
            for partner in contract.partners:
                if partner != self.id:
                    return partner
        except Exception:
            return None
        return None

    def _offer_from_contract(self, contract: Contract) -> Outcome | None:
        try:
            agreement = contract.agreement
            if isinstance(agreement, dict):
                return (
                    int(agreement.get(QUANTITY, agreement.get("quantity"))),
                    int(agreement.get(TIME, agreement.get("time", self.awi.current_step))),
                    int(agreement.get(UNIT_PRICE, agreement.get("unit_price"))),
                )
            return (int(agreement[QUANTITY]), int(agreement[TIME]), int(agreement[UNIT_PRICE]))
        except Exception:
            return None

    def on_negotiation_success(self, contract: Contract, mechanism: Any) -> None:
        super().on_negotiation_success(contract, mechanism)
        partner = self._partner_from_contract(contract)
        offer = self._offer_from_contract(contract)
        if partner is None or offer is None:
            return
        self.accepted_contracts.append((partner, offer))
        self._pending_acceptances.pop(partner, None)

class BayesianSyncRandomAgent(SyncRandomOneShotAgent):
    """
    SyncRandomOneShotAgent with a light Bayesian opponent classifier.

    The classifier keeps one posterior distribution per partner over:

    - RandomOneShotAgent
    - SyncRandomDistOneShotAgent (SyncRandomOneShotAgent and RandDistOneShotAgent)
    - EqualDistOneShotAgent
    - GreedyOneShotAgent
    - Other

    It is intentionally heuristic: the real agents use randomization, and in a
    OneShot game we only see a small sample.  The important part is to update in
    log-space from stable behavioral clues, then change strategy only when the
    posterior is confident enough.
    """

    OPPONENT_TYPES = (
        "RandomOneShotAgent",
        "SyncRandomDistOneShotAgent",
        "EqualDistOneShotAgent",
        "GreedyOneShotAgent",
        "Other",
    )

    def __init__(
        self,
        *args,
        classification_threshold: float = 0.65,
        min_observations: int = 4,
        greedy_time_concession: float = 0.15,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.classification_threshold = classification_threshold
        self.min_observations = min_observations
        self.greedy_time_concession = greedy_time_concession
        self.exploration_days = 5
        self.max_exploration_days = 10
        self.unknown_exploration_ratio = 0.5
        self.min_strategy_classifications = 3
        self.sync_equal_classification_threshold = 0.50
        self.strategy_random_threshold = 0.55
        self.exploration_quantity_multiplier = 1.2
        self.small_dist_early_quantity_multiplier = 1.3
        self.small_dist_midpoint = 0.5
        self.classification_log_path = os.environ.get("BAYES_CLASSIFICATION_LOG")

    # ---------------------------------------------------------------------
    # Initialization and small utilities
    # ---------------------------------------------------------------------

    def init(self):
        super().init()
        self._opponent_logp: dict[str, dict[str, float]] = {}
        self._opponent_observations = defaultdict(int)
        self._opponent_offer_history = defaultdict(list)
        self._sent_offer_history = defaultdict(list)
        self._own_offer_result_history = defaultdict(list)
        for partner in self._all_partners():
            self._ensure_partner(partner)

    def before_step(self):
        super().before_step()
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
        self._sent_offer_history.setdefault(partner, [])
        self._own_offer_result_history.setdefault(partner, [])

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
        """Public helper for debugging/analysis."""
        self._ensure_partner(partner)
        logp = self._opponent_logp[partner]
        base = max(logp.values())
        weights = {name: math.exp(value - base) for name, value in logp.items()}
        total = sum(weights.values())
        if total <= 0:
            return {name: 1.0 / len(self.OPPONENT_TYPES) for name in self.OPPONENT_TYPES}
        return {name: value / total for name, value in weights.items()}

    def opponent_type(self, partner) -> str:
        posteriors = self.opponent_posteriors(partner)
        best = max(posteriors, key=posteriors.get)
        if (
            self._opponent_observations[partner] >= self.min_observations
            and posteriors[best] >= self.classification_threshold
        ):
            return best
        if self._opponent_observations[partner] >= self.min_observations:
            for type_name in ("SyncRandomDistOneShotAgent", "EqualDistOneShotAgent"):
                if posteriors.get(type_name, 0.0) >= self.sync_equal_classification_threshold:
                    return type_name
        if (
            self._opponent_observations[partner] >= self.min_observations
            and posteriors.get("Other", 0.0) >= 0.45
        ):
            return "Other"
        return "Unknown"

    def _best_behavior_type(self, partner) -> tuple[str, float]:
        posteriors = self.opponent_posteriors(partner)
        candidates = {
            name: value
            for name, value in posteriors.items()
            if name != "Other"
        }
        best = max(candidates, key=candidates.get)
        if best == "RandomOneShotAgent" and candidates[best] < self.strategy_random_threshold:
            structured = {
                name: candidates[name]
                for name in (
                    "SyncRandomDistOneShotAgent",
                    "EqualDistOneShotAgent",
                    "GreedyOneShotAgent",
                )
            }
            structured_best = max(structured, key=structured.get)
            if structured[structured_best] >= candidates[best] * 0.75:
                best = structured_best
        return best, candidates[best]

    def _strategy_opponent_types(self, partners: list[str]) -> dict[str, str]:
        types = {partner: self.opponent_type(partner) for partner in partners}
        target = min(self.min_strategy_classifications, len(partners))
        classified = [
            partner
            for partner, opponent_type in types.items()
            if opponent_type not in {"Unknown", "Other"}
        ]
        if len(classified) >= target:
            return types

        candidates = []
        for partner in partners:
            if partner in classified:
                continue
            best_type, confidence = self._best_behavior_type(partner)
            candidates.append(
                (
                    self._opponent_observations[partner] > 0,
                    confidence,
                    self._opponent_observations[partner],
                    partner,
                    best_type,
                )
            )
        candidates.sort(reverse=True)
        for has_observation, _confidence, _observations, partner, best_type in candidates:
            if len(classified) >= target:
                break
            if not has_observation and classified:
                continue
            types[partner] = best_type
            classified.append(partner)
        return types

    def _add_evidence(self, partner, weights: dict[str, float], strength: float = 1.0):
        self._ensure_partner(partner)
        for name, value in weights.items():
            if name in self._opponent_logp[partner]:
                self._opponent_logp[partner][name] += strength * value
        self._opponent_observations[partner] += 1
        self._renormalize_logp(partner)

    def _record_sent_offers(self, proposals):
        for partner, offer in proposals.items():
            self._record_sent_offer(partner, offer)

    def _record_response_offers(self, responses):
        for partner, response in responses.items():
            if response is None or response.response != ResponseType.REJECT_OFFER:
                continue
            self._record_sent_offer(partner, response.outcome)

    def _record_sent_offer(self, partner, offer):
        if partner is None or offer is None or len(offer) <= UNIT_PRICE:
            return
        self._ensure_partner(partner)
        self._sent_offer_history[partner].append(
            {
                "step": self.awi.current_step,
                "offer": tuple(offer),
                "price_good": self._price_good_for_opponent(partner, offer[UNIT_PRICE]),
            }
        )
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

    def _observe_own_offer_result(self, partner, sent_offer, accepted: bool):
        if sent_offer is None:
            return
        price_good = bool(sent_offer["price_good"])
        evidence = {name: 0.0 for name in self.OPPONENT_TYPES}
        if accepted and price_good:
            evidence["GreedyOneShotAgent"] += 0.90
            evidence["RandomOneShotAgent"] -= 0.10
        elif not accepted and not price_good:
            evidence["GreedyOneShotAgent"] += 0.15
            evidence["RandomOneShotAgent"] -= 0.05
        elif accepted and not price_good:
            evidence["GreedyOneShotAgent"] -= 0.90
            evidence["RandomOneShotAgent"] += 0.45
            evidence["SyncRandomDistOneShotAgent"] += 0.20
            evidence["EqualDistOneShotAgent"] += 0.15
        elif not accepted and price_good:
            evidence["GreedyOneShotAgent"] -= 0.65
            evidence["Other"] += 0.10
        self._add_evidence(partner, evidence, strength=0.6)
        self._own_offer_result_history[partner].append(
            {
                "step": self.awi.current_step,
                "price_good": price_good,
                "accepted": bool(accepted),
            }
        )
        if len(self._own_offer_result_history[partner]) > 30:
            del self._own_offer_result_history[partner][:-30]
        self._observe_own_offer_result_pattern(partner)

    def _observe_own_offer_result_pattern(self, partner):
        history = self._own_offer_result_history[partner][-12:]
        good = [item for item in history if item["price_good"]]
        bad = [item for item in history if not item["price_good"]]
        if len(good) < 2 or len(bad) < 2:
            return
        good_accept_rate = sum(item["accepted"] for item in good) / len(good)
        bad_accept_rate = sum(item["accepted"] for item in bad) / len(bad)
        evidence = {name: 0.0 for name in self.OPPONENT_TYPES}
        if good_accept_rate >= 0.65 and bad_accept_rate <= 0.35:
            evidence["GreedyOneShotAgent"] += 1.65
            evidence["RandomOneShotAgent"] -= 0.25
            evidence["SyncRandomDistOneShotAgent"] -= 0.45
            evidence["EqualDistOneShotAgent"] -= 0.15
        elif bad_accept_rate >= 0.45:
            evidence["GreedyOneShotAgent"] -= 0.70
            evidence["RandomOneShotAgent"] += 0.35
            evidence["SyncRandomDistOneShotAgent"] += 0.10
        if any(abs(value) > 0 for value in evidence.values()):
            self._add_evidence(partner, evidence, strength=0.8)

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
        issues = self._issues_for(partner)
        qmin = int(issues[QUANTITY].min_value)
        qmax = max(qmin, int(issues[QUANTITY].max_value))
        q_span = max(1, qmax - qmin)
        q_ratio = (quantity - qmin) / q_span

        evidence = {name: -0.02 for name in self.OPPONENT_TYPES}
        evidence["Other"] = -0.01

        evidence["SyncRandomDistOneShotAgent"] += 0.08
        evidence["EqualDistOneShotAgent"] += 0.10
        evidence["GreedyOneShotAgent"] += 0.10

        first_offer = round_index <= 0

        if first_offer:
            if price_good:
                evidence["EqualDistOneShotAgent"] += 0.70
                evidence["SyncRandomDistOneShotAgent"] += 0.65
                evidence["GreedyOneShotAgent"] += 0.65
                evidence["RandomOneShotAgent"] -= 0.20
            else:
                evidence["RandomOneShotAgent"] += 0.35
                evidence["Other"] += 0.20
                evidence["EqualDistOneShotAgent"] -= 0.15
                evidence["SyncRandomDistOneShotAgent"] -= 0.10
                evidence["GreedyOneShotAgent"] -= 0.25
            if quantity > 3:
                evidence["EqualDistOneShotAgent"] -= 1.60
                evidence["SyncRandomDistOneShotAgent"] += 0.20
                evidence["GreedyOneShotAgent"] += 0.25
                evidence["RandomOneShotAgent"] += 0.35
        else:
            if price_good:
                evidence["GreedyOneShotAgent"] += 0.35
            else:
                # SyncRandomDist/EqualDist counter prices are random. Random also
                # has no price structure. Greedy should move away from ideal only
                # as time passes, which is handled by the history trend below.
                evidence["SyncRandomDistOneShotAgent"] += 0.42
                evidence["EqualDistOneShotAgent"] += 0.25
                evidence["RandomOneShotAgent"] += 0.12
                if t < 0.35:
                    evidence["GreedyOneShotAgent"] -= 0.25

        if history:
            previous = history[-1]
            previous_price_good = bool(previous["price_good"])
            # Greedy tends to concede: good-for-opponent prices can turn bad
            # later, while SyncRandomDist/EqualDist counter prices simply flip
            # randomly.
            if t >= previous["time"] and previous_price_good and not price_good:
                evidence["GreedyOneShotAgent"] += 0.65
                evidence["SyncRandomDistOneShotAgent"] += 0.05
                evidence["RandomOneShotAgent"] += 0.10
            elif t >= previous["time"] and not previous_price_good and price_good:
                evidence["RandomOneShotAgent"] += 0.15
                evidence["SyncRandomDistOneShotAgent"] += 0.38
                evidence["EqualDistOneShotAgent"] += 0.15
                evidence["GreedyOneShotAgent"] -= 0.45
            elif t >= previous["time"] and previous_price_good == price_good:
                evidence["SyncRandomDistOneShotAgent"] += 0.12
                evidence["EqualDistOneShotAgent"] += 0.08

        if quantity <= 3:
            evidence["EqualDistOneShotAgent"] += 0.12
            evidence["SyncRandomDistOneShotAgent"] += 0.10
            if first_offer:
                evidence["EqualDistOneShotAgent"] += 0.18
        elif q_ratio >= 0.80:
            evidence["RandomOneShotAgent"] += 0.70
            evidence["GreedyOneShotAgent"] += 0.10
            evidence["SyncRandomDistOneShotAgent"] -= 0.35
            evidence["EqualDistOneShotAgent"] -= 0.45
        elif q_ratio >= 0.45:
            evidence["GreedyOneShotAgent"] += 0.05
            evidence["RandomOneShotAgent"] += 0.15
            evidence["EqualDistOneShotAgent"] -= 0.25
        else:
            evidence["SyncRandomDistOneShotAgent"] += 0.15

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
        self._add_evidence(partner, evidence)
        self._observe_history_pattern(partner)

    def _observe_history_pattern(self, partner):
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
        if small_ratio >= 0.70 and mean <= 3.5 and (coefficient <= 0.25 or quantity_range <= 1):
            if good_ratio >= 0.75 and price_flip_ratio <= 0.20:
                evidence["EqualDistOneShotAgent"] += 0.75
                evidence["RandomOneShotAgent"] -= 0.20
            else:
                evidence["EqualDistOneShotAgent"] += 1.25
                evidence["SyncRandomDistOneShotAgent"] -= 0.20
                evidence["GreedyOneShotAgent"] -= 0.10
                evidence["RandomOneShotAgent"] -= 0.20
        elif small_ratio >= 0.70 and (coefficient >= 0.40 or quantity_range >= 3):
            evidence["SyncRandomDistOneShotAgent"] += 1.00
            evidence["EqualDistOneShotAgent"] -= 0.45
        elif extreme_ratio >= 0.25:
            evidence["RandomOneShotAgent"] += 1.10
            evidence["GreedyOneShotAgent"] += 0.05
            evidence["EqualDistOneShotAgent"] -= 0.50
            evidence["SyncRandomDistOneShotAgent"] -= 0.55
        elif mean_ratio <= 0.65:
            evidence["SyncRandomDistOneShotAgent"] += 0.35

        if coefficient >= 0.55:
            evidence["SyncRandomDistOneShotAgent"] += 0.65 if good_ratio < 0.75 else 0.20
            evidence["GreedyOneShotAgent"] += 0.10
            evidence["EqualDistOneShotAgent"] -= 0.40
            if extreme_ratio >= 0.20 and 0.20 <= good_ratio <= 0.80:
                evidence["RandomOneShotAgent"] += 1.00
                evidence["SyncRandomDistOneShotAgent"] -= 0.45

        quantity_unstable = coefficient >= 0.45 or quantity_range >= 3
        if (
            good_to_bad > 0
            and bad_to_good > 0
            and 0.20 <= good_ratio <= 0.80
            and quantity_unstable
        ):
            if extreme_ratio >= 0.20:
                evidence["RandomOneShotAgent"] += 0.95
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
            evidence["SyncRandomDistOneShotAgent"] += 0.50 if good_ratio < 0.75 else 0.15
            evidence["EqualDistOneShotAgent"] += 0.25
            evidence["RandomOneShotAgent"] += 0.08
            evidence["GreedyOneShotAgent"] -= 0.20
        if good_ratio >= 0.75 and price_flip_ratio <= 0.20:
            evidence["GreedyOneShotAgent"] += 1.25
            evidence["EqualDistOneShotAgent"] -= 0.35
            evidence["RandomOneShotAgent"] -= 0.20
            evidence["SyncRandomDistOneShotAgent"] -= 0.35
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
        base_proposals = super().first_proposals()
        if self._exploration_enabled():
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
            if needs <= 0:
                proposals.update({partner: None for partner in partners})
                continue
            proposals.update(self._type_based_first_proposals(int(needs), partners, base_proposals))

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
            self._observe_offer(partner, offer, states)
        responses = super().counter_all(offers, states)
        t = self._relative_time(states)
        responses = self._adapt_responses(responses, offers, states, t)
        self._record_response_offers(responses)
        return responses

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
            greedy_target = math.floor(needs * 0.8)
            if len(selected_greedy) == 1:
                quantities = [min(8, greedy_target)]
            else:
                high = math.ceil(greedy_target / 2)
                low = math.floor(greedy_target / 2)
                quantities = [high, low]
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
        quantity = self._clamp_quantity(partner, quantity)
        if quantity <= 0 and not self.awi.allow_zero_quantity:
            return None
        return (quantity, self.awi.current_step, int(price))

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
            self._observe_own_offer_result(partner, sent_offer, accepted=True)

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        super().on_negotiation_failure(partners, annotation, mechanism, state)
        try:
            partner = next(partner for partner in partners if partner != self.id)
        except StopIteration:
            return
        sent_offer = self._latest_sent_offer(partner)
        if sent_offer is not None:
            self._observe_own_offer_result(partner, sent_offer, accepted=False)

class BayesianAgent2(RDVOOneShotAgent):
    """
    Greedy / non-greedy classifier using hard vetoes and softmax logits.

    Hard vetoes handle behavior GreedyOneShotAgent should never show.  All
    softer clues are accumulated as logit evidence and converted to a two-class
    probability distribution with softmax.
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
        return list(dict.fromkeys(list(self.awi.my_suppliers) + list(self.awi.my_consumers)))

    def _max_lines(self) -> int:
        return max(1, int(getattr(self.awi, "n_lines", getattr(self.awi, "max_n_lines", 1)) or 1))

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
        """Returns the softmax-normalized greedy/non-greedy logits."""
        self._ensure_partner(partner)
        if partner in self._non_greedy_veto:
            return {"GreedyOneShotAgent": 0.0, "NonGreedy": 1.0}
        logits = self._opponent_logits[partner]
        scaled = {
            name: value / self.softmax_temperature
            for name, value in logits.items()
        }
        center = max(scaled.values())
        weights = {name: math.exp(value - center) for name, value in scaled.items()}
        total = sum(weights.values())
        if total <= 0:
            return {name: 1.0 / len(self.OPPONENT_TYPES) for name in self.OPPONENT_TYPES}
        return {name: weights[name] / total for name in self.OPPONENT_TYPES}

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
        self._add_logit_evidence(partner, non_greedy=8.0, reason=reason)

    def _record_first_offer(self, partner, offer):
        if partner is None or offer is None or len(offer) <= UNIT_PRICE:
            return
        self._ensure_partner(partner)
        self._sent_first_offers[partner].append(
            {
                "step": int(self.awi.current_step),
                "offer": tuple(offer),
                "price_label": self._opponent_price_label(partner, offer[UNIT_PRICE]),
                "observed": False,
            }
        )
        if len(self._sent_first_offers[partner]) > 20:
            del self._sent_first_offers[partner][:-20]

    def _matching_first_offer(self, partner, outcome):
        for item in reversed(self._sent_first_offers.get(partner, [])):
            if item["step"] != self.awi.current_step or item.get("observed", False):
                continue
            if self._same_offer(item["offer"], outcome):
                return item
        return None

    def _latest_unobserved_first_offer(self, partner):
        for item in reversed(self._sent_first_offers.get(partner, [])):
            if item["step"] == self.awi.current_step and not item.get("observed", False):
                return item
        return None

    def _observe_own_first_offer_result(self, partner, sent_offer, accepted: bool):
        if sent_offer is None or sent_offer.get("observed", False):
            return
        sent_offer["observed"] = True

        price_label = sent_offer.get("price_label", "neutral")
        accepted = bool(accepted)
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
            partners = [partner for partner in all_partners if partner in self.negotiators]
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
            partners = [partner for partner in all_partners if partner in self.negotiators]
            if not partners:
                continue
            if needs <= 0:
                proposals.update({partner: None for partner in partners})
                continue

            greedy_partners = self._ranked_greedy_partners(partners)
            if not greedy_partners:
                proposals.update(self._rdvo_side_proposals(int(needs), partners, t=0.0))
                continue

            proposals.update(
                self._bayesian_agent_first_side_proposals(int(needs), partners)
            )
        return proposals

    def _record_first_offers(self, proposals):
        for partner, offer in proposals.items():
            self._record_first_offer(partner, offer)

    def counter_all(self, offers, states):
        for partner, offer in offers.items():
            if offer is None or offer[TIME] != self.awi.current_step:
                continue
            sent_offer = self._latest_unobserved_first_offer(partner)
            if sent_offer is not None:
                self._observe_own_first_offer_result(partner, sent_offer, accepted=False)
            self._observe_received_first_offer(partner, offer)
        return super().counter_all(offers, states)

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

    def _bayesian_agent_first_side_proposals(self, needs: int, partners: list[str]):
        proposals = {partner: None for partner in partners}
        if needs <= 0:
            return proposals

        opponent_types = {partner: self.opponent_type(partner) for partner in partners}
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
            greedy_quantities = [math.ceil(needs / 2), math.floor(needs / 2)]
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
            scaled_target = max(int(scaled_target), min(2, len(success_scaled_partners)))
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

    def _success_adjusted_quantity(self, target_quantity: float, success_rate: float) -> int:
        success_rate = max(0.05, min(1.0, float(success_rate)))
        return math.ceil(float(target_quantity) / success_rate)

    def _half_quantity_caps(self, needs: int, count: int):
        if needs <= 0 or count <= 0:
            return []
        low = int(needs) // 2
        high = int(needs) - low
        if count == 1:
            return [high]
        return [low if index % 2 == 0 else high for index in range(count)]

    def _assign_equal_quantities(self, proposals, partners, target_quantity, price_getter):
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
            proposals[partner] = self._offer(partner, quantity, price_getter(partner))

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
            quantity_caps = [max(0, int(cap)) for cap in list(quantity_caps)[: len(partners)]]
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

        raw_quantities = [target_quantity * weight / total_weight for weight in weights]
        quantities = []
        for index, quantity in enumerate(raw_quantities):
            assigned = math.floor(quantity)
            if quantity_caps is not None:
                assigned = min(assigned, quantity_caps[index])
            quantities.append(assigned)

        remainder = target_quantity - sum(quantities)
        order = sorted(
            range(len(partners)),
            key=lambda index: (raw_quantities[index] - quantities[index], weights[index]),
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
            proposals[partner] = self._offer(partner, quantity, price_getter(partner))

    def _add_equal_quantities(self, proposals, partners, target_quantity, price_getter):
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

    def _non_greedy_initial_offer_success_rate(self) -> float:
        results = []
        for partner, history in self._own_first_offer_result_history.items():
            if self.opponent_type(partner) == "GreedyOneShotAgent":
                continue
            results.extend(bool(item.get("accepted", False)) for item in history)
        if not results:
            return self.non_greedy_success_default
        return sum(results) / len(results)

    def _partner_non_greedy_initial_offer_success_rate(self, partner) -> float:
        history = self._own_first_offer_result_history.get(partner, [])
        if not history:
            return self._non_greedy_initial_offer_success_rate()
        return sum(bool(item.get("accepted", False)) for item in history) / len(history)

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

    def _is_process_one_agent(self) -> bool:
        return str(getattr(self, "id", "")).endswith("@1")

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

    def _cautious_first_side_proposals(self, needs: int, partners: list[str]):
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
            partner: self._offer(partner, quantity, self._best_price_for_me(partner))
            if quantity > 0 or self.awi.allow_zero_quantity
            else None
            for partner, quantity in distribution.items()
        }

    def _cautious_distribution(self, needs: int, partners: list[str], t: float):
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

    def _cautious_counter_all(self, offers, states):
        responses = {}
        current_offers = {
            partner: offer
            for partner, offer in offers.items()
            if offer is not None and offer[TIME] == self.awi.current_step
        }
        future_partners = {
            partner
            for partner, offer in offers.items()
            if offer is not None and offer[TIME] != self.awi.current_step
        }
        relative_time = self._relative_time(states)

        for needs, all_partners, issues in (
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
        ):
            side_partners = [
                partner for partner in all_partners if partner in current_offers
            ]
            side_future_partners = {
                partner for partner in all_partners if partner in future_partners
            }
            if not side_partners and not side_future_partners:
                continue
            if needs <= 0:
                for partner in set(side_partners).union(side_future_partners):
                    responses[partner] = self._unneeded_response()
                continue

            price = int(issues[UNIT_PRICE].rand())
            random.shuffle(side_partners)
            partner_set = set(side_partners)
            is_selling = all_partners == self.awi.my_consumers
            unneeded_response = self._unneeded_response()

            best = self._best_offer_subset(
                partner_set,
                current_offers,
                int(needs),
                is_selling,
                relative_time,
            )
            if best is not None:
                best_diff, accepted_partners = best
                side_future = set(side_future_partners)
                others = list(partner_set.difference(accepted_partners).union(side_future))
                responses.update(
                    {
                        partner: SAOResponse(
                            ResponseType.ACCEPT_OFFER,
                            current_offers[partner],
                        )
                        for partner in accepted_partners
                    }
                )
                responses.update({partner: unneeded_response for partner in others})

                if best_diff < 0 and others:
                    responses.update(
                        self._cautious_shortage_counter_responses(
                            shortage=-best_diff,
                            partners=others,
                            states=states,
                            unneeded_response=unneeded_response,
                        )
                    )
                continue

            partners = list(partner_set.union(side_future_partners))
            distribution = self._cautious_distribution(
                int(needs),
                partners,
                t=relative_time,
            )
            for partner, quantity in distribution.items():
                if quantity <= 0 and not self.awi.allow_zero_quantity:
                    responses[partner] = unneeded_response
                else:
                    responses[partner] = SAOResponse(
                        ResponseType.REJECT_OFFER,
                        self._offer(partner, quantity, price),
                    )
        return responses

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
            offered = sum(int(offers[partner][QUANTITY]) for partner in partner_ids)
            diff = offered - needs
            price_sum = sum(float(offers[partner][UNIT_PRICE]) for partner in partner_ids)
            size = len(partner_ids)
            partner_ids = tuple(partner_ids)

            if diff >= 0:
                candidate = (diff, self._price_tiebreaker(price_sum, is_selling), partner_ids)
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
                return (minus_diff, set(minus_partners)) if is_selling else (plus_diff, set(plus_partners))
            if -minus_diff < plus_diff:
                return minus_diff, set(minus_partners)
            return plus_diff, set(plus_partners)
        if plus_allowed:
            return plus_diff, set(plus_partners)
        return minus_diff, set(minus_partners)

    def _cautious_shortage_counter_responses(
        self,
        shortage: int,
        partners: list[str],
        states,
        unneeded_response,
    ):
        if shortage <= 0 or not partners:
            return {}
        t = min(
            (
                float(getattr(states[partner], "relative_time", 0.0))
                for partner in partners
                if partner in states
            ),
            default=0.0,
        )
        distribution = self._cautious_distribution(int(shortage), partners, t=t)
        responses = {}
        for partner, q in distribution.items():
            if q <= 0 and not self.awi.allow_zero_quantity:
                responses[partner] = unneeded_response
            else:
                responses[partner] = SAOResponse(
                    ResponseType.REJECT_OFFER,
                    self._offer(partner, q, self._best_price_for_me(partner)),
                )
        return responses

    def _price_tiebreaker(self, price_sum: float, is_selling: bool) -> float:
        return -price_sum if is_selling else price_sum

    def _powerset(self, iterable):
        items = list(iterable)
        return chain.from_iterable(
            combinations(items, size) for size in range(len(items) + 1)
        )

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
            indices += [index for index in range(count) if index not in indices]
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

    def _allowed_mismatch(self, relative_time: float, n_others: int, is_selling: bool):
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
            self.total_agreed_quantity.get(partner, 0) + int(outcome[QUANTITY])
        )
        sent_offer = self._matching_first_offer(partner, outcome)
        if sent_offer is not None:
            self._observe_own_first_offer_result(partner, sent_offer, accepted=True)

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        super().on_negotiation_failure(partners, annotation, mechanism, state)
        try:
            partner = next(partner for partner in partners if partner != self.id)
        except StopIteration:
            return
        sent_offer = self._latest_unobserved_first_offer(partner)
        if sent_offer is not None:
            self._observe_own_first_offer_result(partner, sent_offer, accepted=False)

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
