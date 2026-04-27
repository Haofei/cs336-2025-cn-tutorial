# Stanford CS336 第4回チュートリアル：Mixture of Experts（MoE）

> 適応注記：このチュートリアルは中国語版チュートリアルを日本語学習者向けに翻訳・適応したものです。元の構成、数式、コード、主要な技術用語を保ちつつ、自然で教育的な日本語になるように説明を調整しています。

## 学習目標

この講義を終えると、次のことができるようになります。

1. Mixture of Experts（MoE、専門家混合）が dense model（稠密モデル）に対して持つ中心的な利点、すなわちほぼ同じ activated computation（実際に使う計算量）でより多くの total parameters（総パラメータ）を持てることを説明できる。
2. router / gating（ルーター／ゲーティング）、expert（専門家）、top-k routing（Top-K ルーティング）、load balancing（負荷分散）などの重要語を理解できる。
3. MoE の forward pass（順伝播）の基本式と擬似コードを書ける。
4. 学習時・推論時の MoE のコスト、すなわち FLOPs、メモリ、通信、負荷の偏り、token dropping を分析できる。
5. DeepSeek、Mixtral、Grok、Llama 4 などの現代的な MoE システムでよく見られる工学的トレードオフを理解できる。

## 前提知識

この講義では、Transformer の基本構造、特に self-attention と FFN/MLP、softmax、top-k、residual connection、言語モデル学習における next-token prediction、そして GPU/TPU による並列学習の基礎をすでに理解していることを前提とします。

## この講義の地図

MoE の主線は一文でまとめられます。Transformer の高価な FFN 層を複数の「expert」FFN に置き換え、各 token にはそのうち少数だけを活性化させる、という考え方です。

この講義では順に次を扱います。

- なぜ MoE は同じ FLOPs で dense model より良くなりやすいのか。
- router はどのように token に対して expert を選ぶのか。
- expert はどれくらい大きく、何個にすべきか。shared expert は必要か。
- なぜ load balancing が MoE 学習の鍵なのか。
- MoE のシステム上の代償：通信、メモリ、並列化、token dropping。
- DeepSeek 系列がこれらの考えをどのように組み合わせて現代的な大規模 MoE アーキテクチャにしているか。

## 中核概念

### 1. Dense model と MoE の違い

通常の Transformer では、各層は attention と FFN を含みます。dense model の FFN は固定された大きな MLP であり、すべての token が同じ FFN を通ります。

MoE はこの FFN を複数の expert に置き換えます。各 expert も通常は FFN ですが、各 token はすべての expert を通るわけではありません。router がそのうち K 個を選びます。各 token が 1 個または 2 個の expert だけを活性化するなら、計算量は主に activated parameters に依存し、モデルの total parameters には直接比例しません。

したがって MoE の利点は次の通りです。

- 総パラメータを増やせる：モデル容量が大きくなり、より多くのパターンを記憶・表現できる。
- 活性化パラメータは少ない：各 token は一部のパラメータだけを使うため、expert の総数に対して FLOPs が線形には増えない。
- expert parallelism（専門家並列）に自然に向く：異なる expert を異なるデバイスに置ける。

一方で複雑さも増えます。ルーティングは離散的な選択なので最適化しにくく、expert 間の負荷が大きく偏ることがあり、デバイス間で token を送る通信コストも発生します。

### 2. Expert は必ずしも「意味的な専門家」ではない

“Mixture of Experts” という名前は誤解を招きやすいです。ある expert がコード担当、別の expert が数学担当、別の expert が日本語担当になることを保証するものではありません。より正確には、expert は疎に活性化されるサブネットワークです。何らかの specialization（専門化）が生じることはありますが、それは通常、人間が解釈しやすい領域分割ではありません。

より良いメンタルモデルは、MoE が各層に複数の選択可能な非線形変換経路を用意している、というものです。router は現在の hidden state に基づいてそのうち数本を選びます。hidden state にはすでに文脈、位置、前の層の計算結果が含まれているため、表面上は同じ token でも文脈によって異なる expert に送られます。たとえば “Python” はプログラミング文脈と動物文脈で異なるルーティングになるかもしれませんが、それはある expert を単純に「プログラミング専門家」と名付けられるという意味ではありません。

### 3. Router / Gating

router は軽量なモジュールです。token の hidden state を入力として受け取り、その token が各 expert にどれだけ合うかを示す affinity score（親和スコア）を出力します。よくある実装は、線形射影の後に softmax または sigmoid を適用し、スコアが最も高い K 個の expert を選ぶものです。

router の出力は gating weights（ゲーティング重み）とも呼ばれ、複数 expert の出力を重み付きで合成するために使われます。

### 4. Top-K Routing

現代の大規模 MoE は、ほぼ token choice top-k routing に収束しています。つまり各 token が自分でスコア上位 K 個の expert を選びます。

他にも方式はあります。

- expert choice：各 expert が処理したい token を選ぶ。負荷分散は自然にできるが、token にとって最良の expert とは限らない。
- global assignment：大域的なマッチング／最適輸送問題を解く。洗練されているが計算コストが高い。
- hashing routing：ハッシュ関数で token を固定的に割り当てる。意外にも改善をもたらすことがあるが、学習されるルーティングほど柔軟ではない。
- RL routing：強化学習で離散選択を扱う。原理的には妥当だが、実践ではコストと分散が高すぎる。

## ステップ別チュートリアル

### ステップ1：FFN を expert pool に置き換える

通常の FFN は次のように書けます。

```text
h_out = h + FFN(h)
```

MoE 層は次のように書けます。

```text
h_out = h + sum_{i in TopK(router(h))} g_i(h) * Expert_i(h)
```

ここで：

- `h` は現在の token の hidden state。
- `Expert_i` は i 番目の FFN。
- `router(h)` は token から見た全 expert のスコアを与える。
- `TopK` は K 個の expert だけを残す。
- `g_i(h)` はゲーティング重みで、実装によって正規化される場合も、完全には正規化されない場合もある。

K=1 なら計算量は dense FFN 1 個に近くなります。K=2 ならおおよそ FFN 2 個を活性化するのに相当し、FLOPs は約 2 倍になります。しかしモデルの total parameters は dense model よりはるかに大きくできます。

### ステップ2：expert の数と大きさを理解する

初期の MoE では、dense FFN を同じ大きさの expert として複数コピーすることがよくありました。その後、DeepSeek などのシステムは fine-grained experts（細粒度 expert）が非常に有効であることを示しました。つまり、各 expert を小さくし、その代わり数を増やします。

たとえば、元の FFN の中間次元が hidden size の 4 倍だとします。細粒度 MoE では、各 expert の中間次元を元の 1/2、1/4、またはそれ以下にし、より多くの expert を活性化できます。これにより FLOPs を抑えながら、ルーティングの組み合わせの柔軟性を高められます。

別の設計として shared expert（共有 expert）があります。router が何を選んでも、各 token は 1 個または数個の共有 FFN を必ず通ります。動機は、すべての計算を疎なルーティングに依存させるのではなく、モデルに汎用的な処理能力を残すことです。DeepSeek 系列では shared expert が使われたことがありますが、他モデルの ablation では利益が常に安定するわけではないため、これは工学的選択です。

### ステップ3：なぜ load balancing が不可欠か

制約がないと、router は悪い局所最適に陥りやすくなります。すべての token が少数の expert に送られ、他の expert はほとんど学習されず dead experts（死んだ expert）になります。これはメモリを浪費するだけでなく、MoE をはるかに小さいモデルへ退化させます。

そのため MoE の学習では通常、auxiliary load balancing loss（補助的な負荷分散損失）を加えます。Switch Transformer でよく使われる形は次です。

```text
L_balance = alpha * N * sum_i f_i * p_i
```

ここで：

- `N` は expert の数。
- `f_i` は実際に expert i にルーティングされた token の割合。
- `p_i` は router が expert i に割り当てた平均確率。
- `alpha` は損失の重み。

直感的には、ある expert がすでに多すぎる token を受け取っているなら、その expert へのルーティング確率を下げ、token を他の expert へ分散させます。

負荷分散はシステム最適化であるだけでなく、モデリング上の最適化でもあります。GPU 利用率を無視しても、expert collapse を避けるために必要です。

### ステップ4：学習時の安定性問題

MoE が学習しにくい主な理由は 3 つあります。

1. top-k は離散選択であり、選ばれなかった expert には勾配が流れない。
2. router が早い段階で少数の expert に偏ることがある。
3. softmax router は低精度学習で数値不安定性を引き起こすことがある。

よく使われる安定化手法は次の通りです。

- router の計算を float32 で行う。
- router logits に z-loss を加え、softmax normalizer を制約する。
- load balancing loss を加える。
- 探索を促すため、ノイズや jitter を加えることがある。
- fine-tuning ではより多くのデータを使い、総パラメータが巨大であることによる過学習を避ける。

DeepSeek V3 は auxiliary-loss-free balancing を提案しました。各 expert にバイアス `b_i` を持たせ、expert i が最近受け取った token が少なすぎれば `b_i` を増やし、多すぎれば `b_i` を減らします。このバイアスはルーティング選択にだけ使われ、最終的な gating weight には使われません。実践上、DeepSeek V3 は単一シーケンス内の負荷偏りを抑えるために sequence-wise auxiliary loss も残しています。

### ステップ5：学習と推論のコスト

MoE の FLOPs は魅力的に見えますが、実際のシステムコストは FLOPs だけではありません。

学習コストには次が含まれます。

- 活性化された expert の行列乗算コスト。
- すべての expert パラメータを保存するメモリコスト。
- router と load balancing の追加計算。
- token dispatch と combine の通信コスト。
- 負荷不均衡によるデバイスの待ち時間。

推論コストも似ています。各 token は少数の expert しか活性化しませんが、すべての expert の重みはどこかに保存されている必要があります。expert が複数 GPU に分散している場合、all-to-all communication が必要です。まず token を選ばれた expert のあるデバイスへ送り、計算後に結果を戻します。

ある expert が多すぎる token を受け取ると、システムは token dropping を発動することがあります。容量を超えた token はその expert を通らず、residual connection だけで次へ伝わります。これにより学習や推論の結果が同じ batch 内の他リクエストに依存し、一見奇妙な非決定性が生じることがあります。

### ステップ6：Expert Parallelism

expert parallelism は MoE の重要なシステム上の利点です。expert は自然に分割されているため、異なる expert を異なるデバイスに配置できます。流れはおおよそ次の通りです。

1. 各デバイスが一部の token hidden states を持つ。
2. router が各 token をどの expert に送るか決める。
3. all-to-all で token を対応するデバイスへ送る。
4. expert が FFN を実行する。
5. all-to-all で出力を元の位置へ戻す。
6. gating weight に従って結果を合成する。

これにより、大規模モデル並列化にもう 1 つの軸が加わります。data parallelism、tensor/model parallelism、pipeline parallelism に加えて、expert parallelism を使えます。ただし通信トポロジー、batch size、expert 数、capacity factor はすべて効率に影響します。

重要な判断基準は、expert の計算が all-to-all 通信コストを隠せるほど「厚い」必要がある、ということです。各 expert が小さすぎ、毎回少数の token しか受け取らないなら、GPU 上では小さな行列乗算と細切れの通信が大量に発生し、実際の wall-clock time は良くならないかもしれません。そのため現代の実装では fused kernels、block-sparse matrix multiplication、MegaBlocks などのライブラリを使い、複数 expert の計算をハードウェアに適したバッチ処理として整理します。

### ステップ7：DeepSeek 系列の進化

DeepSeek MoE はこの講義で繰り返し参照する現代的な事例です。初期の DeepSeek MoE はすでに 2 つの重要設計、fine-grained experts と shared experts を採用し、top-k routing と補助負荷分散も使っていました。DeepSeek V2 は全体構造を大きく変えずに規模を数百億 activated parameters へ拡大し、top-M device selection を導入しました。これは token がアクセスできるデバイス集合を先に制限し、その中で expert を選ぶことで、デバイス間通信を減らす方法です。

DeepSeek V3 は MoE 本体を維持しつつ router を調整しました。より穏やかな sigmoid 風スコアを使い、expert 負荷に応じてオンライン更新される bias を導入して、いわゆる auxiliary-loss-free balancing を実現しました。同時に、推論時に単一の異常なシーケンスが少数 expert を圧迫しないよう、sequence-wise auxiliary loss も残しています。この進化は、MoE の基本アーキテクチャ自体は複雑ではなく、本当の進歩は学習安定性、通信制御、負荷分散の細部から生まれることが多いと示しています。

## 数式と擬似コード

### Top-K MoE forward pass

```python
# h: [tokens, d_model]
# W_router: [d_model, n_experts]
# experts: list of FFN modules
# k: number of active experts

scores = h @ W_router              # [tokens, n_experts]
probs = softmax(scores, dim=-1)    # or sigmoid + normalization
indices = topk(probs, k)           # selected experts per token

output = zeros_like(h)
for token t:
    for expert i in indices[t]:
        weight = probs[t, i]
        output[t] += weight * experts[i](h[t])

h_next = h + output
```

### Load balancing loss

```text
f_i = fraction of tokens actually routed to expert i
p_i = average router probability assigned to expert i
L_balance = alpha * N * sum_i f_i p_i
```

### DeepSeek V3 風の bias update

```text
if load_i < target_load:
    b_i = b_i + gamma
else:
    b_i = b_i - gamma

routing_score_i = router_score_i + b_i
```

注意：`b_i` は top-k を決めるために使われますが、最終的に expert 出力を合成するときは通常、元の gating score が使われます。

## よくある誤解

1. 「Expert は人間が解釈できる領域専門家である。」
   必ずしもそうではありません。expert は疎に活性化されるサブネットワークであり、専門化することはありますが、通常は明確なコード／数学／日本語 expert ではありません。

2. 「MoE はパラメータが多いので必ず高価である。」
   総パラメータは多いですが、各 token は少数のパラメータしか活性化しません。FLOPs は activated parameters に依存し、メモリは total parameters に依存します。MoE の学習やデプロイでは、総パラメータ、活性化パラメータ、token ごとの活性化 expert 数を同時に報告しないと、モデル規模を誤読しやすくなります。

3. 「expert が多ければ多いほど良い。」
   expert が多いほどメモリと通信の複雑さは増えます。ルーティングが偏れば多くの expert が死にます。expert が細かすぎれば通信の断片化も深刻になります。本当に意味があるのは「より多くの、利用可能で十分に学習された expert」であり、設定ファイル上の expert 数ではありません。

4. 「softmax の後の top-k は取り除ける。」
   安易に取り除くことはできません。すべての expert が計算に参加すると、MoE は疎計算の利点を失い、学習・推論コストが爆発します。softmax は比較可能で重み付け可能なスコアを作る役割、top-k は疎活性化を強制する役割を持ち、両者は別の問題を解いています。

5. 「Load balancing は GPU 利用率のためだけである。」
   それだけではありません。expert collapse を防ぎ、すべてのパラメータが学習されるようにします。負荷分散がないと、loss は下がっているように見えても、実際には少数の expert しか学習されず、大半の容量が浪費される可能性があります。

6. 「MoE 推論は必ず dense より速い。」
   必ずしもそうではありません。モデルが大きすぎて重みが多ノード・多 GPU に分散している場合、通信とスケジューリングが疎計算の利益を打ち消すことがあります。MoE は十分大規模で成熟した推論システムにおいて利点を発揮しやすく、単一 GPU の小モデルでは dense model の方が単純で安定なことがあります。

## 実践演習

1. toy MoE 層を手で実装する：2 次元ベクトルを入力、4 個の expert、毎回 top-2 を活性化し、異なる token のルーティングを観察する。
2. load balancing loss をオフにし、各 expert の token 数を記録して、expert collapse が起きるか確認する。
3. K=1 と K=2 を比較する：training loss、計算量、expert 利用率はどう変わるか。
4. capacity factor を実装する：各 expert が受け取れる token 数を固定し、超えた分を drop する。出力が batch 構成の影響を受けるか観察する。
5. DeepSeek V3 技術報告を読み、MoE 以外の 2 つの重要設計、MLA（Multi-head Latent Attention）と MTP（Multi-token Prediction）を見つける。

## まとめ

MoE は現代の高性能言語モデルにおける重要なアーキテクチャです。中核的な考えは単純です。router を使って各 token に少数の expert を選ばせ、total parameters と activated parameters を異なる比率で切り離します。これにより、同じ学習 FLOPs でより低い loss を得たり、同程度の推論計算量でより大容量のモデルを使ったりできます。

本当に難しいのは工学と最適化です。離散的な top-k routing は学習しにくく、expert の負荷は偏り、デバイス間通信はボトルネックになり、推論時の token dropping は非決定性をもたらすことがあります。現代のシステムは top-k routing、fine-grained experts、shared experts、load balancing loss、expert parallelism、auxiliary-loss-free balancing などの技術により、MoE を大規模モデルの学習・デプロイにおける主流選択肢にしています。

## 次回講義への接続

この講義では主に MoE アーキテクチャと学習を扱いました。次に必要なのは、大規模モデルのシステムレベル並列戦略をより深く理解することです。data parallelism、tensor parallelism、pipeline parallelism、expert parallelism がどのように組み合わさり、通信帯域、メモリ、batch size が実際の学習スループットをどう決めるのかを学ぶ必要があります。これらのシステム詳細を理解して初めて、あるハードウェアクラスタ上で MoE が本当に「割に合う」か判断できます。システム部分を学び続けるなら、常に「数学上の FLOPs」と「実機上の時間」を分けて考えてください。MoE 論文の曲線は前者を示すことが多い一方で、工学的成否はしばしば後者に依存します。
