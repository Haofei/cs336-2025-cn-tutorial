# Stanford CS336 第7回チュートリアル：Parallelism 1

> 翻案メモ: この日本語チュートリアルは中国語版講義ノートをもとに、技術用語・Markdown構造・数式・コードを保ったまま、日本語で自然に学べるように説明を整えたものです。

## 学習目標

この講義を読み終えると、次のことができるようになるはずです。

1. 大規模言語モデル（LLM）の学習を、なぜ単一 GPU から複数 GPU、複数ノード、さらにはデータセンター全体へ拡張する必要があるのか説明できる。
2. data parallelism、model parallelism、tensor parallelism、pipeline parallelism、activation/sequence parallelism の中心的な考え方を区別できる。
3. all-reduce、reduce-scatter、all-gather などの collective communication が分散学習で果たす役割を理解できる。
4. 並列化戦略ごとに、計算、GPU メモリ、通信帯域、batch size のあいだのトレードオフを判断できる。
5. ハードウェアトポロジに応じて、LLM 学習の妥当な並列構成を考えられる。

## この講義の地図

この講義で扱うのは「単一 kernel をどう速くするか」ではなく、「モデルが大きすぎる、学習が遅すぎるとき、学習タスクを多くのマシンへどう分割するか」です。流れは次の通りです。

- まず動機を見る：単一 GPU の FLOPs とメモリでは足りないため、multi-machine parallelism が必要になる。
- 次に通信の基礎を見る：GPU 間の接続は同じではない。ノード内の NVLink/NVSwitch は速く、ノード間の InfiniBand や Ethernet は遅い。
- data parallelism を導入する：モデルを複製し、データを分割し、勾配を同期する。さらに ZeRO/FSDP でメモリ使用量を減らす。
- model parallelism を導入する：完全なモデルを複製せず、モデル自体を分割する。pipeline parallelism と tensor parallelism を含む。
- 最後に activation memory、sequence parallelism、通信と計算のトレードオフ、実際の大規模モデル学習での組み合わせ規則を議論する。

## 主要概念

### 1. なぜ LLM 学習には並列化が必要なのか

大規模モデルの学習には、compute と memory という 2 つの厳しい制約があります。

Compute は学習に必要な総浮動小数点演算量を指します。モデルが大きく、token 数が多いほど、必要な FLOPs は増えます。GPU は世代ごとに速くなっていますが、最先端モデルの学習需要を単一 GPU だけで満たすことはできません。

Memory は GPU メモリを指します。パラメータ自体はメモリ使用量の一部にすぎません。学習時にはさらに次を保存します。

- parameters：モデルパラメータ。通常 BF16/FP16 で、約 2 bytes/parameter。
- gradients：勾配。約 2 bytes/parameter。
- optimizer states：Adam の一次・二次モーメント、および master weights。しばしばさらに大きい。
- activations：backpropagation のために forward pass 中に保存する中間結果。

よく使われる概算では、Adam で学習すると 1 パラメータあたり約 16 bytes の学習状態が必要になります。そのため、7B、70B、さらに大きいモデルを 1 枚の GPU に単純に載せることはできません。

### 2. 通信プリミティブ：collective communication

分散学習では collective communication に大きく依存します。

- all-reduce：各 rank がテンソルを持ち、まず和や平均などの reduce を行い、その結果を全 rank に返す。データ並列の勾配同期でよく使われる。
- reduce-scatter：まず reduce し、その結果を shard ごとに異なる rank へ配る。
- all-gather：各 rank が shard を持ち、すべての shard を集めて各 rank が完全なテンソルを得る。
- broadcast：1 つの rank のデータを全 rank にコピーする。

重要な等価関係：

```text
all-reduce ≈ reduce-scatter + all-gather
```

bandwidth-bound な場面では、この 2 つの通信量はほぼ同等です。この事実により、ZeRO が通信量を明らかに増やさずにメモリを節約できる理由が説明できます。

### 3. ハードウェアトポロジが並列戦略を決める

並列アルゴリズムはハードウェアから切り離せません。典型的な NVIDIA 学習ノードでは、1 台のマシンに 8 枚の GPU があり、ノード内は NVLink/NVSwitch で高速接続されます。一方、マシン間は InfiniBand などのネットワークで接続され、帯域とレイテンシは明らかに悪くなります。

したがって経験則は次の通りです。

- 帯域を大量に使う tensor parallelism は通常ノード内に置く。
- data parallelism はより遅いネットワークをまたいでもよい。各 step で同期するのは 1 batch 分の勾配またはパラメータ shard だからである。
- pipeline parallelism は activation を通信し、多くは point-to-point なので、ノード間や比較的遅いリンクにも適する場合がある。

## ステップ別チュートリアル

並列化案を設計するときは、まず 3 つの質問を考えます。第一に、1 枚の GPU にモデルパラメータ、勾配、optimizer state、ピーク時の activation が入るか。入らないなら、まずモデルまたは学習状態を分割しなければなりません。第二に、通信リンクはどこが最も速いか。ノード内の高速インターコネクトは頻繁な同期に適し、ノード間の遅いリンクでは同期回数を減らすか point-to-point 転送を中心にするべきです。第三に、現在の学習で許される effective batch size はどれくらいか。batch がすでに critical batch size に近いなら、data parallelism で GPU を増やしても通信が増えるだけで収束速度は上がらないかもしれません。

実用的には、並列戦略を「異なる次元の分割」と見るとよいです。Data parallelism はサンプル次元を分割し、主にスループットを上げます。Tensor parallelism は各層の行列の幅を分割し、単一層が広すぎる、あるいはパラメータが大きすぎる問題を解きます。Pipeline parallelism はネットワークの深さを分割し、層数が多すぎて全体が入らない問題を解きます。Sequence parallelism は系列次元を分割し、tensor parallelism では減らしにくい activation を扱います。実際のシステムではどれか 1 つを選ぶのではなく、3D/4D parallelism と呼ばれる形で重ね合わせるのが普通です。

## Step 1: Data Parallelism：モデルを複製し、データを分割する

Data parallelism（データ並列）は最も自然な並列化です。各 GPU が完全なモデルパラメータを保持し、異なるデータサンプルを処理します。

global batch size を B、GPU 数を M とすると、各 GPU は B/M 個のサンプルを処理します。各 GPU は独立に forward/backward を実行してローカル勾配を得たあと、all-reduce で勾配を平均し、最後に各 GPU が同じ optimizer step を実行します。

SGD 更新は次のように書けます。

```text
θ_{t+1} = θ_t - η · (1/B) · Σ_{i=1}^B ∇ℓ(x_i; θ_t)
```

データ並列は、この総和を複数 GPU に分担させているだけです。

利点：

- 計算のスケーリングがよい：batch が十分大きければ、GPU を増やすことでより多くのサンプルを処理できる。
- 実装が簡単で、モデル構造にあまり依存しない。

欠点：

- メモリは節約できない：各 GPU が完全なパラメータ、勾配、optimizer state を持つ。
- 通信量はパラメータ数に依存する：各 step で勾配同期が必要。
- batch size に制限される：GPU 数を effective batch size を超えて無限に増やすことはできない。

### Batch size はリソースである

batch size が小さいと、各 GPU に最低限の有用なサンプルが必要なため、データ並列はそれ以上スケールできません。batch を増やせるとしても、最適化には critical batch size があります。ある点を超えると、batch を増やしても学習速度への利益は逓減します。ボトルネックが「勾配ノイズ」から「勾配更新ステップ数」に移るためです。

したがって batch size は自由に使える無限のリソースではありません。data parallelism、pipeline parallelism、gradient accumulation のあいだで配分する必要があります。

## Step 2: ZeRO/FSDP：データ並列でもメモリを節約する

素朴なデータ並列は、各 GPU が完全な学習状態を複製するためメモリを浪費します。ZeRO（Zero Redundancy Optimizer）はこれらの状態を段階的に分割します。

### ZeRO Stage 1：optimizer state を分割する

各 GPU は引き続き完全な parameters と gradients を保持しますが、Adam の optimizer states は shard に分割されます。各 GPU は自分が持つパラメータ shard の更新だけを担当します。

手順：

1. 各 GPU が完全な勾配を計算する。
2. reduce-scatter を使い、勾配をパラメータ shard ごとに対応する GPU へ集約する。
3. 各 GPU が自分の optimizer state で対応するパラメータ shard を更新する。
4. all-gather で更新後のパラメータ shard を各 GPU に集め戻す。

通信の観点では、reduce-scatter + all-gather は元の all-reduce とほぼ等価なので、Stage 1 はほとんど「無料」のメモリ最適化です。

### ZeRO Stage 2：さらに gradients を分割する

Stage 2 では完全な勾配を保持しません。backpropagation 中、ある層の勾配が計算されるとすぐに、そのパラメータ shard を担当する GPU へ reduce し、ローカルの一時勾配を解放します。

これにより、完全な gradient vector を GPU メモリ上に実体化せずに済みます。総通信量は元のデータ並列に近いままですが、スケジューリングはより複雑です。

### ZeRO Stage 3 / FSDP：parameters も分割する

FSDP（Fully Sharded Data Parallel）は基本的に ZeRO Stage 3 に対応します。parameters、gradients、optimizer states がすべて sharded されます。

中心的な考え方は「必要なときだけパラメータを all-gather する」ことです。

1. ある層の forward 前に、その層のパラメータを all-gather する。
2. その層の計算が終わったらパラメータを解放する。
3. backward 時にも必要に応じて再びパラメータを all-gather する。
4. 勾配を計算したら、その shard を担当する GPU へ reduce-scatter する。

通信量は約 2×parameters から約 3×parameters へ増えますが、overlap communication and computation（通信と計算の重ね合わせ）や prefetch（先読み）などにより、実際のオーバーヘッドは許容できることがあります。

FSDP の強みは汎用性です。Transformer 構造を深く理解しなくても使え、大規模モデル学習の標準的なメモリ最適化手段としてよく使われます。

## Step 3: Model Parallelism：モデル自体を分割する

モデルや activation がまだ入らない場合は、model parallelism（モデル並列）が必要です。FSDP と違い、model parallelism の目的は「一時的に完全なパラメータを集める」ことではなく、モデルの異なる部分を固定的に異なる GPU に置くことです。主な通信対象は activations になります。

この講義では pipeline parallelism と tensor parallelism の 2 つに注目します。

## Step 4: Pipeline Parallelism：層ごとにモデルを分割する

Pipeline parallelism（パイプライン並列）は、深さ方向にモデルを分割します。たとえば GPU0 が前半の層、GPU1 が中間の層、GPU2 が後半の層を持ちます。forward では activation が前から後ろへ流れ、backward では activation gradients が後ろから前へ流れます。

素朴な方法では深刻な bubble が発生します。ある時点で 1 枚の GPU だけが働き、他の GPU は空いてしまうからです。bubble を減らすため、通常は batch を複数の micro-batches に分け、異なる micro-batch が工場の流れ作業のように異なる stage に同時に存在するようにします。

bubble オーバーヘッドのよくある近似は次です。

```text
bubble_ratio ≈ (pipeline_stages - 1) / micro_batches
```

したがって micro-batch が多いほど pipeline は埋まります。ただし、これは batch size というリソースを消費します。

利点：

- パラメータと一部の activation が層ごとに分散され、メモリのスケーリングがよい。
- 通信の多くは隣接 stage 間の point-to-point activation 転送である。
- 比較的遅いネットワークリンクをまたぐ場合にも適することがある。

欠点：

- スケジューリングが複雑。特に 1F1B、interleaved pipeline、zero-bubble pipeline などの高度な方式では難しい。
- bubble により GPU 利用率が下がる。
- 実装が非常に難しく、autograd や runtime scheduling に深く介入する必要があることが多い。

Zero-bubble pipeline の 1 つの工夫は、backward を 2 種類の仕事に分けることです。

- B：activation gradient を逆伝播する。厳密な依存関係がある。
- W：weight gradient を計算する。依存が少ないため bubble の中へ移動できる。

これにより本来なら空いていた時間でパラメータ勾配を計算し、利用率を高められます。

## Step 5: Tensor Parallelism：行列の幅方向にモデルを分割する

Tensor parallelism（テンソル並列）は、幅方向に行列積を分割します。Transformer の主な計算は大きな行列積なので、重み行列を複数の部分行列に分け、複数 GPU が partial results を計算し、collective communication で結合できます。

たとえば MLP では：

```text
Y = GeLU(XA)
Z = YB
```

A と B を A1/A2、B1/B2 に分割できます。各 GPU が行列積の一部を処理し、必要に応じて all-reduce で結果を結合します。

利点：

- batch size を消費しない。
- pipeline bubble がない。
- Transformer のように行列積中心のモデルに自然に合う。

欠点：

- 各層に同期バリアが入る可能性がある。
- 通信対象は activation であり、頻度が高く、帯域要求が大きい。
- 通常は 8 GPU の NVLink/NVSwitch 環境など、ノード内高速インターコネクトに限って適している。

経験則として、tensor parallel size はノード内 GPU 数、たとえば 8 に設定されることが多いです。ノードをまたいで遅いリンクで tensor parallel を行うと、スループットが大きく落ちる可能性があります。

## Step 6: Activation と Sequence Parallelism

ここまでは主にパラメータ、勾配、optimizer state を扱ってきましたが、activation memory も大きな問題になります。そのピークは通常 backpropagation の初期に現れます。この時点では多くの forward activation がまだ解放されておらず、同時に勾配も蓄積され始めるからです。系列が長く、batch が大きいほどこの問題は顕著で、長コンテキスト学習では特に activation がボトルネックになりやすいです。

Transformer の各層の activation は、おおよそ次の 2 種類の項を含みます。

```text
activation_memory_per_layer ≈ S · B · H · 34 + 5 · A · S² · B
```

ここで S は sequence length、B は batch size、H は hidden size、A は attention heads です。右側の S² 項は attention softmax などの二次複雑度部分から来ており、FlashAttention や recomputation によって大きく減らせます。

Tensor parallelism は行列積に関連する多くの activation を分割できますが、layer norm、dropout、残差ストリーム入力などの pointwise operations では完全な activation が残ることがあります。Sequence parallelism は、これらの pointwise activation を sequence dimension に沿って分割します。異なる GPU が異なる token position を担当します。

これにより追加の all-gather / reduce-scatter が発生しますが、activation memory をさらに減らせます。activation recomputation と組み合わせると、計算量を増やす代わりにメモリを下げ、より大きな batch やモデルを可能にすることがよくあります。

## 数式と擬似コードの早見表

### データ並列学習の擬似コード

```python
for batch in data:
    local_batch = shard(batch, rank)
    loss = model(local_batch)
    loss.backward()
    all_reduce(model.gradients)   # すべての rank の勾配を同期する
    optimizer.step()
    optimizer.zero_grad()
```

### FSDP の考え方の擬似コード

```python
for layer in model.layers:
    weights = all_gather(layer.weight_shards)
    activation = layer.forward(activation, weights)
    free(weights)

for layer in reversed(model.layers):
    weights = all_gather(layer.weight_shards)
    grad = layer.backward(grad, weights)
    reduce_scatter(layer.grad_shards)
    free(weights)

optimizer.step_on_local_shards()
```

### 並列戦略を組み合わせる規則

```text
まずモデルがメモリに入ることを保証する：
    1. ノード内では tensor parallelism を優先する
    2. それでも入らない場合は FSDP/ZeRO-3 または pipeline parallelism を使う

モデルが入るようになったら：
    3. 残りの GPU を data parallelism に使ってスループットを伸ばす
    4. 通信が頻繁すぎる場合は gradient accumulation で effective batch を大きくする
```

## よくある誤解

1. 「GPU が多いほど速い。」
   誤りです。通信、同期、pipeline bubble、batch size の制限により、スケーリング効率は下がります。

2. 「Data parallelism はメモリ問題を解決できる。」
   素朴な DDP ではできません。ZeRO/FSDP のような sharding 技術だけが、パラメータ関連メモリを大きく減らせます。

3. 「FSDP は model parallelism である。」
   完全にはそうではありません。FSDP はパラメータを分割しますが、計算時には必要に応じて all-gather します。model parallelism はパラメータの固定配置を重視し、主に activation を送ります。

4. 「Tensor parallelism はノードをまたいで自由に拡張できる。」
   通常はできません。各層で通信するため、帯域とレイテンシに非常に敏感で、ノード内高速インターコネクト上に置くのが最適です。

5. 「Pipeline parallelism は概念が簡単だから実装も簡単である。」
   実際は逆です。効率的な pipeline scheduling、micro-batch、1F1B、zero-bubble、autograd への介入はどれも複雑です。

6. 「Activation memory は重要ではない。」
   長系列・大 batch・大規模モデルの学習では、activation が主要なメモリボトルネックになることがあります。

## 演習

1. P 個のパラメータを持つモデルを Adam で学習し、各パラメータの学習状態が約 16 bytes だとします。7B パラメータモデルについて、パラメータ関連の学習状態だけで必要なメモリを見積もってください。

2. 自分の言葉で、なぜ次が成り立つのか説明してください。

```text
all-reduce ≈ reduce-scatter + all-gather
```

また、この等価関係が ZeRO Stage 1 にどう役立つか説明してください。

3. 8 GPU ノード内の NVLink は速く、ノード間ネットワークは遅いとします。tensor parallelism、pipeline parallelism、data parallelism をそれぞれどの範囲に配置しますか。理由も述べてください。

4. pipeline stages = 8、micro-batches = 32 の場合、bubble_ratio を見積もってください。micro-batches を半分にすると何が起きますか。

5. FSDP と tensor parallelism の主な違いを説明してください。それぞれ何を通信し、どのようなハードウェア特性に依存しますか。

## まとめ

LLM 学習の並列化の本質は、GPU メモリ、計算、通信帯域/レイテンシ、batch size という 4 種類の希少資源を同時に管理することです。Data parallelism は単純でスループット拡張に適していますが、batch size とメモリ複製に制限されます。ZeRO/FSDP は optimizer states、gradients、parameters を sharding することで、データ並列でもメモリを節約します。Pipeline parallelism は層方向にモデルを分割し、通信は比較的穏やかですが bubble と複雑なスケジューリングがあります。Tensor parallelism は行列幅方向にモデルを分割し、batch size を消費しませんが高速インターコネクトを必要とします。Sequence parallelism と activation recomputation は activation memory をさらに扱います。

実際の学習に単一の最適解はありません。よくある方法は、ノード内で tensor parallelism を使い、必要なら sequence parallelism を組み合わせることです。それでもモデルが入らなければ FSDP または pipeline parallelism を加え、最後に残りの GPU を data parallelism で使い切り、gradient accumulation で通信頻度を調整します。各戦略が何を通信し、どのハードウェアを必要とするかを理解することが、効率的な LLM 学習システム設計の鍵です。

エンジニアリングの観点では、よい並列化案とは特定の用語を追うことではなく、高価な GPU をなるべく待たせないことです。通信は prefetch や overlap でき、メモリは sharding や recomputation と交換でき、batch size は同期コストをならすために使えます。しかしどの選択も別のリソースに圧力を移します。Lecture 7 の核心的な結論は、大規模学習はシステム設計問題であり、アルゴリズム、ハードウェアトポロジ、optimizer state、activation のライフサイクル、スケジューリング戦略を一緒に考える必要がある、ということです。
