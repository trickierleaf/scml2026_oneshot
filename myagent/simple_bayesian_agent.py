from __future__ import annotations

import math

from negmas import Outcome, SAOResponse, ResponseType
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from scml.oneshot.agents.rand import SyncRandomOneShotAgent, powerset

from .bayesian_agent import BayesianAgent


class SimpleBayesianAgent(BayesianAgent):
    """BayesianAgent のベイズ分類器をそのまま再利用した、より単純な戦略エージェント。

    - Greedy/NonGreedy の判定は親 (``BayesianAgent.opponent_type``) と同じ。
      探索フェーズ・成功率学習・evidence 記録もすべて親の仕組みを利用する。
    - オファー生成 (:meth:`first_proposals`) と受諾ロジック
      (:meth:`_current_offer_responses`) のみを、明示的に指定された単純な規則で
      上書きする。``counter_all`` は親のまま呼ばれるため、相手の挙動観測
      (分類のための evidence) は維持される。

    共通の挙動:
      - Greedy へのオファーは常に Greedy にとって good price (= 自分の worst price)。
      - 必要量ちょうどの組み合わせがあれば受諾。
      - 同じ数量でカウンターするなら、現在の同じ数量のオファーを受ける。
      - 受諾判断は相手人数・需給関係から決める。閾値は「誤差 ÷ 残り必要数」で
        正規化した値を用いる (時間は使わない)。詳細は
        :meth:`_acceptance_threshold` を参照。
      - t > 0.9 で利益を最大化する組み合わせを強制受諾 (利益が出ないなら受けない)。
    """

    def __init__(
        self,
        *args,
        accept_base_tolerance: float = 0.20,
        accept_favorable_factor: float = 0.85,
        accept_unfavorable_factor: float = 1.25,
        single_partner_acceptance_factor: float = 2.50,
        neutral_market_ratio: float = 1.06,
        market_band: float = 0.13,
        forced_accept_time: float = 0.925,
        # カウンター数量の時間譲歩を開始する t (これ以降、相手量へ寄せる)。
        quantity_concession_start: float = 0.5,
        # 不足方向 (売り手の過剰契約) の受諾を禁止する。
        forbid_shortfall_accept: bool = True,
        eighty_success_threshold: float = 0.80,
        eighty_main_ratio: float = 0.65,
        greedy_single_offer_cap: int = 7,
        greedy_first_offer_cap: int = 6,
        greedy_second_offer_cap: int = 4,
        min_success_rate: float = 0.05,
        # NonGreedy 買い手の過剰調達抑制。
        #   target 算出に使う成功率の下限 (target ブランチ)。
        target_success_floor: float = 0.45,
        #   提案総量キャップ = ceil(needs / max(success_cap_floor, 平均成功率))。
        #   成功率の下限 (これ未満だとキャップが過大になり過剰調達を許すため)。
        success_cap_floor: float = 0.45,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # 受諾閾値 (誤差÷残り必要数) の基準値と需給による補正係数。
        self.accept_base_tolerance = float(accept_base_tolerance)
        self.accept_favorable_factor = float(accept_favorable_factor)
        self.accept_unfavorable_factor = float(accept_unfavorable_factor)
        self.single_partner_acceptance_factor = float(single_partner_acceptance_factor)
        # この市場は構造的に供給>需要 (実測の供給÷需要 ≈ 1.06、供給超過が約68%)。
        # そのため需給の「中立」を 1.0 ではなく実測平均 ratio に置き、その上下に
        # band 幅の帯を作って有利/不利を判定する。
        self.neutral_market_ratio = float(neutral_market_ratio)
        self.market_band = float(market_band)
        self.favorable_market_ratio = self.neutral_market_ratio * (1.0 + self.market_band)
        self.unfavorable_market_ratio = self.neutral_market_ratio * (1.0 - self.market_band)
        # t がこの値を超えたら利益最大化の組み合わせを強制受諾する。
        self.forced_accept_time = float(forced_accept_time)
        self.quantity_concession_start = float(quantity_concession_start)
        self.forbid_shortfall_accept = bool(forbid_shortfall_accept)
        # 初手オファー契約成功率がこの値以上の相手を「主軸」とみなす。
        self.eighty_success_threshold = float(eighty_success_threshold)
        self.eighty_main_ratio = float(eighty_main_ratio)
        # Greedy 環境でのオファー数量上限。
        self.greedy_single_offer_cap = int(greedy_single_offer_cap)
        self.greedy_first_offer_cap = int(greedy_first_offer_cap)
        self.greedy_second_offer_cap = int(greedy_second_offer_cap)
        self.min_success_rate = float(min_success_rate)
        # target ブランチの過剰調達抑制 (案A: 成功率下限 / 案B: 総量キャップ)。
        self.target_success_floor = float(target_success_floor)
        self.success_cap_floor = float(success_cap_floor)

    # ------------------------------------------------------------------
    # 補助
    # ------------------------------------------------------------------

    def _is_greedy(self, partner) -> bool:
        return self.opponent_type(partner) == "GreedyOneShotAgent"

    def _greedy_partners_sorted(self, partners) -> list[str]:
        greedy = [partner for partner in partners if self._is_greedy(partner)]
        greedy.sort(
            key=lambda partner: self.opponent_posteriors(partner).get(
                "GreedyOneShotAgent",
                0.0,
            ),
            reverse=True,
        )
        return greedy

    def _success_rate(self, partner) -> float:
        return self._partner_non_greedy_initial_offer_success_rate(partner)

    def _average_success_rate(self, partners) -> float:
        rates = [self._success_rate(partner) for partner in partners]
        if not rates:
            return self._non_greedy_initial_offer_success_rate()
        return sum(rates) / len(rates)

    def _good_price_for(self, partner, greedy_partners) -> int:
        # Greedy には常に good price (= 自分の worst price)、それ以外は best price。
        if partner in greedy_partners:
            return self._worst_price_for_me(partner)
        return self._best_price_for_me(partner)

    # ------------------------------------------------------------------
    # Greedy 分類の強化 (実測の数量別受諾率に基づく)
    # ------------------------------------------------------------------

    # 親が good価格受諾に与える既定の greedy 加点 (この値からの差分で補正する)。
    _PARENT_GOOD_ACCEPT_WEIGHT = 1.20

    def _good_accept_greedy_weight(self, quantity: int) -> float:
        """good価格オファーの受諾に対する greedy 加点を数量で重み付け。

        実測 (good価格・数量別の初手受諾率) の対数尤度比に基づく:
          - 小口(<=2): Greedy/NonGreedy 共に受けやすく非識別的 → 弱
          - 中数量(4〜7): NonGreedy はほぼ受けず Greedy のみ受ける → 強
          - 大数量(>=8): Greedy 自身も needs 超過で断り始める → 弱
        """
        q = int(quantity)
        if q <= 2:
            return 0.5
        if q == 3:
            return 0.9
        if q <= 7:
            return 1.3
        return 0.5

    def _observe_first_offer_classification_result(self, partner, sent_offer, accepted):
        # 親の分類ロジックをそのまま走らせた上で、good価格受諾の加点だけ数量補正する。
        if sent_offer is None:
            return super()._observe_first_offer_classification_result(
                partner, sent_offer, accepted
            )
        already = sent_offer.get("first_result_observed", False)
        price_label = sent_offer.get("price_label", "neutral")
        offer = sent_offer.get("offer")
        quantity = int(offer[QUANTITY]) if offer is not None and len(offer) > QUANTITY else 0

        super()._observe_first_offer_classification_result(partner, sent_offer, accepted)

        if (not already) and bool(accepted) and price_label == "good":
            adjusted = self._good_accept_greedy_weight(quantity)
            # 親は +_PARENT_GOOD_ACCEPT_WEIGHT を加点済み → 差分で目標重みに補正。
            self._ensure_partner(partner)
            self._opponent_logits[partner]["GreedyOneShotAgent"] += (
                adjusted - self._PARENT_GOOD_ACCEPT_WEIGHT
            )

    # ------------------------------------------------------------------
    # 初手提案
    # ------------------------------------------------------------------

    def first_proposals(self):
        # 分類のための探索フェーズは親の挙動を踏襲する。
        if self._exploration_enabled():
            base_proposals = SyncRandomOneShotAgent.first_proposals(self)
            proposals = self._exploration_first_proposals(base_proposals)
            self._record_sent_offers(proposals)
            return proposals

        proposals = {}
        for needs, all_partners, is_buy in (
            (self.awi.needed_supplies, self.awi.my_suppliers, True),
            (self.awi.needed_sales, self.awi.my_consumers, False),
        ):
            partners = [partner for partner in all_partners if partner in self.negotiators]
            if not partners:
                continue
            if int(needs) <= 0:
                proposals.update({partner: None for partner in partners})
                continue
            proposals.update(self._simple_first_proposals(int(needs), partners, is_buy))

        self._record_sent_offers(proposals)
        return proposals

    def _simple_first_proposals(self, needs: int, partners: list[str], is_buy: bool):
        greedy_partners = self._greedy_partners_sorted(partners)
        if greedy_partners:
            return self._greedy_env_first_proposals(needs, partners, greedy_partners)
        return self._nongreedy_env_first_proposals(needs, partners, is_buy)

    def _greedy_env_first_proposals(self, needs, partners, greedy_partners):
        proposals = {partner: None for partner in partners}
        non_greedy = [partner for partner in partners if partner not in greedy_partners]

        if len(greedy_partners) >= 2:
            # 最も Greedy らしい 1 人に min(6, needs)、次に min(4, 残り) をオファー。
            # (B022式の「0.8分割+0.2保険」も試したが、SB は信頼できる Greedy 2人で
            #  ちょうど needs に着地するため、保険は過剰調達になり逆効果だった。)
            first = greedy_partners[0]
            second = greedy_partners[1]
            first_quantity = min(self.greedy_first_offer_cap, int(needs))
            proposals[first] = self._raw_offer(
                first, first_quantity, self._worst_price_for_me(first)
            )
            remaining = max(0, int(needs) - first_quantity)
            second_quantity = min(self.greedy_second_offer_cap, remaining)
            if second_quantity > 0:
                proposals[second] = self._raw_offer(
                    second, second_quantity, self._worst_price_for_me(second)
                )
            return proposals

        # Greedy が 1 人: min(7, needs) を Greedy に。残りを Greedy 以外に
        # 「成功率で割った個数」で分配する。
        greedy_partner = greedy_partners[0]
        greedy_quantity = min(self.greedy_single_offer_cap, int(needs))
        proposals[greedy_partner] = self._raw_offer(
            greedy_partner,
            greedy_quantity,
            self._worst_price_for_me(greedy_partner),
        )
        remaining = max(0, int(needs) - greedy_quantity)
        if remaining > 0 and non_greedy:
            scaled = self._success_scaled_quantity(remaining, non_greedy)
            self._fill_even(proposals, non_greedy, scaled, self._best_price_for_me)
        return proposals

    def _nongreedy_env_first_proposals(self, needs, partners, is_buy: bool = True):
        proposals = {partner: None for partner in partners}

        # 初手オファー契約成功率が閾値 (80%) 以上の相手がいるか。
        strong = [
            partner
            for partner in partners
            if self._success_rate(partner) >= self.eighty_success_threshold
        ]
        if strong:
            main = max(strong, key=self._success_rate)
            main_quantity = max(1, min(int(needs), round(int(needs) * self.eighty_main_ratio)))
            proposals[main] = self._raw_offer(
                main,
                main_quantity,
                self._best_price_for_me(main),
            )
            remaining = max(0, int(needs) - main_quantity)
            others = [partner for partner in partners if partner != main]
            if remaining > 0 and others:
                scaled = self._success_scaled_quantity(remaining, others)
                self._fill_even(proposals, others, scaled, self._best_price_for_me)
        else:
            # 80% の相手がいない場合: 何人で契約成立を狙うかを
            # target = round(人数 × 平均成功率) で決め、必要量を target で割った量を
            # 全員に配る (= 期待充足量がちょうど needs になる)。
            target = self._close_target_count(partners)
            self._fill_by_target(proposals, partners, int(needs), target, self._best_price_for_me)

        # 適応キャップ: 提案総量を ceil(needs / 平均初手成功率) に頭打ちする。
        # 期待充足量がちょうど needs になる総量で、成功率が高い環境ほど自動的に
        # 締まる。両ブランチに適用して過剰調達を抑える。
        avg = max(self.success_cap_floor, self._average_success_rate(partners))
        self._cap_total_offer(proposals, partners, math.ceil(int(needs) / avg))
        return proposals

    def _close_target_count(self, partners) -> int:
        """何人で契約成立を狙うか。

        各相手に必要量を配って期待成立件数 = 人数 × 平均成功率 とし、それを
        狙う成立人数とする。例 (下限なしのとき):
          - 4 人 / 平均 50% → round(2.0) = 2 人
          - 6 人 / 平均 30% → round(1.8) = 2 人
        案A: 平均成功率が実測より低めに出ると target が小さくなり総量が膨張する
        ため、target_success_floor で下限を設ける (過剰調達抑制)。
        """
        avg = max(self.target_success_floor, self._average_success_rate(partners))
        return max(1, min(len(partners), round(len(partners) * avg)))

    def _cap_total_offer(self, proposals, partners, max_total: int):
        """提案総量が max_total を超える分を、成功率の低い相手から削る。"""
        max_total = int(max_total)
        active = [p for p in partners if proposals.get(p) is not None]
        excess = sum(int(proposals[p][QUANTITY]) for p in active) - max_total
        if excess <= 0:
            return
        # 成功率の低い相手から削る (充足への悪影響を最小化)。
        for partner in sorted(active, key=lambda p: (self._success_rate(p), str(p))):
            if excess <= 0:
                break
            offer = proposals[partner]
            quantity = int(offer[QUANTITY])
            cut = min(quantity, excess)
            excess -= cut
            new_quantity = quantity - cut
            if new_quantity <= 0:
                proposals[partner] = None
            else:
                proposals[partner] = self._raw_offer(
                    partner, new_quantity, int(offer[UNIT_PRICE])
                )

    def _success_scaled_quantity(self, quantity: int, partners) -> int:
        """quantity を相手の平均成功率で割って (= 期待充足量が quantity になる) 量に拡大。"""
        avg = self._average_success_rate(partners)
        return max(1, math.ceil(quantity / max(self.min_success_rate, avg)))

    def _fill_even(self, proposals, partners, total: int, price_getter):
        """total を partners に均等配分する。"""
        partners = list(partners)
        n = len(partners)
        if n <= 0 or total <= 0:
            return
        base = total // n
        remainder = total - base * n
        for index, partner in enumerate(partners):
            quantity = base + (1 if index < remainder else 0)
            if quantity <= 0:
                continue
            offer = self._raw_offer(partner, quantity, price_getter(partner))
            if offer is not None:
                proposals[partner] = offer

    def _fill_by_target(self, proposals, partners, needs: int, target: int, price_getter):
        """各相手に ceil/floor(needs/target) を配る (target 人で needs を満たす配分)。

        例 (needs=7, target=3, 6 人) → (3,2,2,3,2,2)。
        """
        partners = list(partners)
        target = max(1, int(target))
        if needs <= 0 or not partners:
            return
        base = needs // target
        remainder = needs % target
        for index, partner in enumerate(partners):
            quantity = base + (1 if (index % target) < remainder else 0)
            if quantity <= 0:
                continue
            offer = self._raw_offer(partner, quantity, price_getter(partner))
            if offer is not None:
                proposals[partner] = offer

    # ------------------------------------------------------------------
    # 受諾・カウンター
    # ------------------------------------------------------------------

    def _current_offer_responses(self, offers, states):
        # 親の counter_all が相手挙動の観測を済ませた上でこのメソッドを呼ぶ。
        t = self._relative_time(states)
        current_offers = {
            partner: offer
            for partner, offer in offers.items()
            if offer is not None
            and len(offer) > UNIT_PRICE
            and offer[TIME] == self.awi.current_step
        }

        # 既定では全相手に「不要」を返し、各サイドで上書きする。
        responses = {partner: self._unneeded_response() for partner in offers}

        for needs, all_partners, is_sell in (
            (self.awi.needed_supplies, self.awi.my_suppliers, False),
            (self.awi.needed_sales, self.awi.my_consumers, True),
        ):
            side_partners = [
                partner for partner in all_partners if partner in current_offers
            ]
            if not side_partners:
                continue
            if int(needs) <= 0:
                for partner in side_partners:
                    responses[partner] = self._unneeded_response()
                continue
            responses.update(
                self._simple_side_responses(
                    int(needs),
                    side_partners,
                    current_offers,
                    t,
                    is_sell,
                )
            )

        self._record_response_offers(responses, relative_time=t)
        return responses

    def _simple_side_responses(self, needs, partners, offers, t, is_sell):
        greedy_partners = self._greedy_partners_sorted(partners)

        # 不足方向 (= 売り手の過剰契約) の受諾は禁止する。売り手は受諾合計が
        # 必要量を超えないよう、受諾候補を needs 以下に制限する。
        accept_cap = int(needs) if (is_sell and self.forbid_shortfall_accept) else None

        # 1. 最良の組み合わせ (誤差最小) を求める。誤差 0 なら必要量ちょうど → 受諾。
        best_subset, best_error = self._best_subset(partners, offers, needs)
        if best_error == 0 and best_subset:
            return self._accept_subset_and_counter(
                needs, partners, offers, best_subset, greedy_partners, t
            )

        # 2. Greedy 環境 & 売り手 & ちょうどの組み合わせなし:
        #    最も Greedy らしい 1 人を除き、必要量以下で最大の組み合わせを受諾し、
        #    残りを Greedy にオファー。
        if greedy_partners and is_sell:
            plan = self._seller_greedy_fill_plan(partners, offers, int(needs))
            if plan is not None:
                accepted_partners, greedy_partner, remaining_needs = plan
                responses = {}
                for partner in partners:
                    if partner in accepted_partners:
                        responses[partner] = SAOResponse(
                            ResponseType.ACCEPT_OFFER, offers[partner]
                        )
                    elif partner == greedy_partner and remaining_needs > 0:
                        counter = self._raw_offer(
                            partner,
                            remaining_needs,
                            self._worst_price_for_me(partner),
                        )
                        responses[partner] = (
                            self._unneeded_response()
                            if counter is None
                            else SAOResponse(ResponseType.REJECT_OFFER, counter)
                        )
                    else:
                        responses[partner] = self._unneeded_response()
                return responses

        # 3. t > forced_accept_time(0.925): 利益を最大化する組み合わせを強制受諾。
        #    売り手は needs を超える組み合わせを除外 (過剰契約=不足を防ぐ)。
        if t > self.forced_accept_time:
            return self._forced_profit_max_responses(partners, offers, max_total=accept_cap)

        # 4. 受諾閾値 (誤差÷残り必要数) 判定。売り手は needs 以下の最良部分集合で判定。
        accept_subset, accept_error = (
            (best_subset, best_error)
            if accept_cap is None
            else self._best_subset(partners, offers, needs, max_total=accept_cap)
        )
        relative_error = accept_error / max(1, int(needs))
        threshold = self._acceptance_threshold(len(partners), is_sell)
        if accept_subset and relative_error <= threshold:
            return self._accept_subset_and_counter(
                needs, partners, offers, accept_subset, greedy_partners, t
            )

        # 5. 完全受諾には至らないが、オファー内の「良い部分」だけは受諾する
        #    (部分受諾)。残りはカウンター (同じ数量なら受諾)。
        partial = self._partial_accept_set(int(needs), partners, offers)
        return self._counter_side(needs, partners, offers, partial, greedy_partners, t)

    def _acceptance_threshold(self, n_partners: int, is_sell: bool) -> float:
        """受諾閾値 = 許容する「誤差÷残り必要数」。

        - 相手人数が多いほど厳しく (まだ良い相手を探せるため小さく)、少ないほど
          緩く (取りこぼしを避けるため大きく) する: ``base * 2 / (n + 1)``。
        - 需給: 自分に有利な市場なら厳しめ (選り好み)、不利なら緩め (取りに行く)。
          買い手は供給過多 (sell/buy 比が高い) が有利、売り手は需要過多
          (sell/buy 比が低い) が有利。
          ただしこの市場は通常 sell/buy ≈ 1.06 で供給超過が常態のため、中立帯を
          実測平均 (neutral_market_ratio) の上下 band% に置いている。例えば
          neutral=1.06, band=0.13 なら、買い手有利は ratio>=1.20、買い手不利は
          ratio<=0.92。供給が需要を下回りかける (比が常態 1.06 を割り込む) と
          買い手は早めに「取りに行く」側へ振れる。
        時間 t はここでは使わない。
        """
        threshold = self.accept_base_tolerance * (2.0 / (n_partners + 1.0))
        if n_partners == 1:
            threshold *= self.single_partner_acceptance_factor

        ratio = self._input_market_sell_buy_ratio()
        if ratio is not None:
            if not is_sell:  # 買い手
                if ratio >= self.favorable_market_ratio:
                    threshold *= self.accept_favorable_factor
                elif ratio <= self.unfavorable_market_ratio:
                    threshold *= self.accept_unfavorable_factor
            else:  # 売り手
                if ratio <= self.unfavorable_market_ratio:
                    threshold *= self.accept_favorable_factor
                elif ratio >= self.favorable_market_ratio:
                    threshold *= self.accept_unfavorable_factor
        return threshold

    def _best_subset(self, partners, offers, needs: int, max_total: int | None = None):
        """|提供量合計 - needs| を最小化する組み合わせを返す。

        同点では自分にとっての価値が高く、人数が少ない方を優先する。
        ``max_total`` 指定時は合計がそれを超える組み合わせを除外する
        (不足方向＝過剰契約の受諾を禁止するために使用)。
        返り値は ``(partner の set, 誤差)``。
        """
        candidates = [
            partner for partner in partners if int(offers[partner][QUANTITY]) > 0
        ]
        if not candidates:
            return set(), int(needs)

        best_key = None
        best_set: set[str] = set()
        for subset in powerset(candidates):
            offered = sum(int(offers[partner][QUANTITY]) for partner in subset)
            if max_total is not None and offered > int(max_total):
                continue
            error = abs(offered - int(needs))
            value = self._subset_value_for_me(subset, offers)
            key = (error, -value, len(subset))
            if best_key is None or key < best_key:
                best_key = key
                best_set = set(subset)
        return best_set, (best_key[0] if best_key is not None else int(needs))

    def _partial_accept_set(self, needs: int, partners, offers):
        """部分受諾する組み合わせ (合計 <= needs) を返す。

        取引は完了しなくても、オファーの中で「割のよい部分」だけは受け取る。
        各相手の公平分 ``floor(needs / 人数)`` を 1 以上上回る量
        (= ``floor(needs / 人数) + 1`` 以上) のオファーを「良い部分」とみなし、
        その中から合計が needs を超えない範囲で最大になる組み合わせを受諾する。
        残りはカウンターで取りに行く。

        例: needs=10, 4 人, オファー (3,3,3,3) のとき
            閾値 = floor(10/4)+1 = 3 → 4 件とも該当。
            合計 <= 10 で最大は (3,3,3)=9 を受諾し、残り 1 をカウンター。
        """
        n = len(partners)
        if n <= 0 or int(needs) <= 0:
            return set()

        quality_threshold = int(needs) // n + 1
        good = [
            partner
            for partner in partners
            if int(offers[partner][QUANTITY]) >= quality_threshold
            and int(offers[partner][QUANTITY]) > 0
        ]
        if not good:
            return set()

        best_key = None
        best_set: set[str] = set()
        for subset in powerset(good):
            total = sum(int(offers[partner][QUANTITY]) for partner in subset)
            if total > int(needs):
                continue
            # 合計を最大化 (= 残りを最小化)。同点では自分の価値が高く、人数の
            # 少ない方を優先。
            key = (total, self._subset_value_for_me(subset, offers), -len(subset))
            if best_key is None or key > best_key:
                best_key = key
                best_set = set(subset)
        return best_set

    def _subset_value_for_me(self, partners, offers) -> float:
        value = 0.0
        for partner in partners:
            offer = offers[partner]
            quantity = int(offer[QUANTITY])
            price = float(offer[UNIT_PRICE])
            if self._is_seller_to(partner):
                value += quantity * price
            else:
                value -= quantity * price
        return value

    def _subset_utility(self, offers_subset) -> float:
        try:
            return float(self.ufun.from_offers(offers_subset))
        except Exception:
            total = 0.0
            for partner, offer in offers_subset.items():
                quantity = int(offer[QUANTITY])
                price = float(offer[UNIT_PRICE])
                if self._is_seller_to(partner):
                    total += quantity * price
                else:
                    total -= quantity * price
            return total

    def _forced_profit_max_responses(self, partners, offers, max_total: int | None = None):
        # 何も受けない (空集合) を基準に、効用を最大化する組み合わせを選ぶ。
        # max_total 指定時は合計がそれを超える組み合わせを除外 (過剰契約=不足の禁止)。
        best_set: set[str] = set()
        best_utility = self._subset_utility({})
        for subset in powerset(partners):
            if max_total is not None and sum(
                int(offers[p][QUANTITY]) for p in subset
            ) > int(max_total):
                continue
            subset_offers = {partner: offers[partner] for partner in subset}
            utility = self._subset_utility(subset_offers)
            if utility > best_utility:
                best_utility = utility
                best_set = set(subset)

        responses = {}
        for partner in partners:
            if partner in best_set:
                responses[partner] = SAOResponse(
                    ResponseType.ACCEPT_OFFER, offers[partner]
                )
            else:
                responses[partner] = self._unneeded_response()
        return responses

    def _accept_subset_and_counter(self, needs, partners, offers, accept_set, greedy_partners, t=0.0):
        return self._counter_side(needs, partners, offers, set(accept_set), greedy_partners, t)

    def _counter_side(self, needs, partners, offers, accepted, greedy_partners, t=0.0):
        responses = {}
        for partner in accepted:
            responses[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offers[partner])

        accepted_quantity = sum(int(offers[partner][QUANTITY]) for partner in accepted)
        remaining = max(0, int(needs) - accepted_quantity)
        counter_partners = [partner for partner in partners if partner not in accepted]

        if remaining <= 0:
            for partner in counter_partners:
                responses[partner] = self._unneeded_response()
            return responses

        accepted_offers = {partner: offers[partner] for partner in accepted}
        quantities = self._equal_counter_quantities(remaining, counter_partners)
        for partner in counter_partners:
            quantity = quantities.get(partner, 0)
            if quantity <= 0:
                responses[partner] = self._unneeded_response()
                continue
            price = self._good_price_for(partner, greedy_partners)
            # 時間譲歩: t に応じてカウンター数量を相手のオファー量へ寄せる
            # (損益分岐点ガード付き)。
            quantity = self._time_conceded_quantity(
                partner, quantity, offers[partner], t, price, accepted_offers
            )
            if quantity <= 0:
                responses[partner] = self._unneeded_response()
                continue
            counter = self._raw_offer(partner, quantity, price)
            # 同じ数量でカウンターするなら現在のオファーを受諾。
            responses[partner] = self._counter_or_accept_response(
                partner,
                offers[partner],
                counter,
            )
        return responses

    def _time_conceded_quantity(self, partner, desired_q, offer, t, price, accepted_offers):
        """カウンター数量を t に応じて相手のオファー量へ寄せる (B022 同様)。

        ただし **損益分岐点を守る**: 親の ``_max_floor_quantity`` で、
        その量を約定しても効用が無契約 (disagreement) 効用を下回らない範囲に
        制限する。売り手は需要超過 (= 不足ペナルティ) になる増量を許さない。
        """
        desired_q = int(desired_q)
        if offer is None or t <= self.quantity_concession_start or len(offer) <= QUANTITY:
            return desired_q
        opponent_q = int(offer[QUANTITY])
        if opponent_q == desired_q:
            return desired_q
        span = max(1e-9, 0.95 - self.quantity_concession_start)
        concession = max(0.0, min(1.0, (float(t) - self.quantity_concession_start) / span))
        target = int(round(desired_q + (opponent_q - desired_q) * concession))
        # 損益分岐点 (無契約効用) を下回らない量に制限。
        return self._max_floor_quantity(
            partner, desired_q, target, int(price), accepted_offers
        )
