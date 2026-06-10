from __future__ import annotations

import random

from negmas import Contract, Outcome, SAOResponse, SAOState, ResponseType
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from scml.oneshot.agents.rand import SyncRandomOneShotAgent, powerset


class GreedyBanditAgent(SyncRandomOneShotAgent):
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