# シミュレーション分析レポート

このプロジェクトでは、シミュレーションの実行と、結果の確認を分けています。

分析コマンドの出力は `report.html` に一本化しています。

## 最新の結果を分析する

```bash
.venv/bin/python -m myagent.helpers.analyzer
```

結果フォルダを指定しない場合、最新の `tmp_tournament_test/*-stage-*` が自動で選ばれます。

出力先:

```text
tmp_tournament_test/最新の-stage-フォルダ/analysis/report.html
```

## 特定の結果を分析する

```bash
.venv/bin/python -m myagent.helpers.analyzer tmp_tournament_test/20260505H180536836398yUf-stage-0001
```

## 最初に表示するエージェントを指定する

特定の名前や型を最初に追跡対象として表示したい場合は `--agent` を使います。

```bash
.venv/bin/python -m myagent.helpers.analyzer --agent My
```

```bash
.venv/bin/python -m myagent.helpers.analyzer --agent Random
```

`--agent` を指定しても、最終スコアランキングは全エージェントを表示します。

指定しない場合は、ランキング先頭のエージェントが最初に選ばれます。

## 出力先を変える

```bash
.venv/bin/python -m myagent.helpers.analyzer --out analysis_latest
```

この場合は以下に出力されます。

```text
analysis_latest/report.html
```

## report.html で見られるもの

### 最終スコアランキング

全エージェントの最終スコアをランキング形式で表示します。

`表示する world` で、全 world のランキングと world ごとのランキングを切り替えられます。

主な表示項目:

- `agent`
- `type`
- `process`
- `final score`
- `balance`
- `avg productivity`
- `shortfall penalty total`
- `disposal total`

ランキングの `agent` 名をクリックすると、そのエージェントを追跡対象として選択できます。

### エージェント追跡

画面右側で、追跡するエージェントと表示内容を選べます。

表示内容:

- `summary`
- `step 0`
- `step 1`
- `step 2`
- ...

### summary

選択したエージェントについて、日ごとの主要指標と、相手ごとの契約集計を表示します。

主な指標:

- `score`
- `balance`
- `productivity`
- `shortfall_quantity`
- `shortfall_penalty`
- `disposal_cost`

### step 表示

特定の step を選ぶと、その日の取引状況を確認できます。

表示されるのは、その step での対象エージェントのロールだけです。

対象エージェントが売り手側の場合:

- 仕入れ値と個数
- 成立した売値と個数
- 売るべき個数と値段
- 売り手側の成立契約

対象エージェントが買い手側の場合:

- 売るべき個数と値段
- 成立した買値と個数
- 外生仕入れ
- 買い手側の成立契約

取引履歴:

- 相手エージェント
- 自分が送った提案か、受け取った提案か
- 状態
- 数量
- 単価
- 価格範囲
- negotiation id
- agreement の有無

ここには、成立したものだけでなく、未成立・終了した交渉も含まれます。

### Codexおすすめ指標

選択した step の状況から、改善時に注目すべき点を文章で表示します。

例:

- 不足ペナルティが出ている
- 廃棄コストが出ている
- productivity が低い
- 終了した交渉が成立交渉より多い
- 交渉による契約が少ない

## 見る順番

まずは以下の順番で見るのがおすすめです。

1. 最終スコアランキング
2. 改善したいエージェントを選択
3. `summary` で全体傾向を見る
4. スコアが悪い step を選ぶ
5. 売り手側・買い手側の契約量と単価を見る
6. 取引履歴で、どの相手との交渉が成立していないかを見る
7. Codexおすすめ指標を参考に改善方針を決める

## HTML の開き方

Mac のターミナルから開く場合:

```bash
open tmp_tournament_test/20260505H180536836398yUf-stage-0001/analysis/report.html
```

VS Code で見る場合は、`report.html` を右クリックして Live Server で開くのがおすすめです。
