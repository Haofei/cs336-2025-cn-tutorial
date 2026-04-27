# Stanford CS336 第10回チュートリアル：LLM Inference

> 翻案メモ: この日本語チュートリアルは中国語版講義ノートをもとに、元の構成、技術用語、数式、コード風表記を保ちながら、日本語で自然に学べるように説明を整えたものです。

この講義では **inference（推論、サービス時の生成）** を扱います。すでに学習済みで固定されたモデルに対して、ユーザーの prompt を入力し、response を生成する処理です。学習は通常、一度きりの大きなコストですが、推論はチャット、コード補完、バッチ処理、モデル評価、test-time compute、RL のサンプリングなどで繰り返し呼ばれます。そのため推論効率は、プロダクトのコストとユーザー体験を直接左右します。

## 1. 推論では何を最適化するのか

LLM inference を測る主要な指標は 3 つあります。

- **TTFT（Time To First Token、最初の token までの時間）**：ユーザーが prompt を送信してから、最初の出力 token が見えるまでの待ち時間です。主に prompt の処理時間で決まり、対話型アプリケーションでは非常に重要です。
- **Latency（レイテンシ）**：生成開始後に token が届く速度です。通常は各 token にどれくらい時間がかかるか、またはユーザーが感じるストリーミング出力の速さとして理解されます。
- **Throughput（スループット）**：システムが単位時間あたりに生成できる token 数で、通常 tokens/s で表されます。バッチ処理は throughput を重視しますが、チャット製品では低 latency との両立が必要です。

低 latency と高 throughput はしばしば衝突します。throughput を上げるため、システムは多くのリクエストを大きな batch にまとめようとします。しかし batch が大きいほど、個々のユーザーはその計算ステップが終わるまで長く待つ可能性があります。

## 2. なぜ推論は学習と違うのか

Transformer の学習では、系列全体の token が既知なので、sequence 次元に沿って並列計算できます。行列積も十分に大きく、GPU/TPU の計算能力を使い切りやすくなります。

推論、とくに **autoregressive generation（自己回帰生成）** では、`t` 番目の token はそれまでに生成された token に依存します。そのため decode 段階は 1 ステップずつ進むしかありません。各ステップでは通常、各系列につき 1 token だけを生成するので、多くの計算が細い行列積や行列ベクトル積になり、ハードウェアの計算能力を十分に使いにくくなります。

ここで重要な概念が **arithmetic intensity（算術強度）** です。これはメモリから 1 byte 読み書きするごとに何 FLOPs の計算を行えるかを表します。算術強度が高ければ通常 compute-bound（計算能力律速）で、低ければ memory-bound（メモリ帯域律速）です。H100 のような GPU は非常に強い計算能力を持ちますが、HBM memory bandwidth には限界があります。大量のパラメータや KV cache を読む一方で計算量が少ないと、GPU は「メモリ待ち」になります。

学習や prefill では `batch size × sequence length` が十分大きく、算術強度が高くなります。一方、token ごとの decode では `T=1` なので算術強度が大きく下がり、推論は memory bandwidth bottleneck になりやすいのです。

## 3. Prefill と Decode：推論の 2 つの段階

LLM 推論は 2 つの段階に分けられます。

### 3.1 Prefill（prompt の事前計算）

prompt が与えられると、モデルはすべての prompt token を一度に処理し、各層の attention に必要な key/value を計算し、次 token の logits を得ます。この段階は学習時の forward pass に似ています。sequence 次元で並列化でき、通常 compute-bound で比較的高速です。TTFT の大部分は prefill に由来し、特に prompt が長い場合に顕著です。

### 3.2 Decode / Generation（逐次 token 生成）

モデルは現在の文脈から新しい token を 1 つ生成し、それを文脈に追加して次のステップへ進みます。この段階は逐次性が強く、`T=1` で、通常 memory-bound です。推論システムで最も最適化が難しい部分です。

## 4. KV Cache：重複計算を避ける中核機構

最も素朴な生成方法は、1 token 生成するたびに完全な文脈を transformer に再入力することです。これでは過去すべての token の key/value を毎回再計算するため、計算量が非常に悪くなります。

**KV cache（Key-Value cache）** の考え方は、causal transformer では過去 token の key/value は新しい token が来ても変わらないため、キャッシュできるというものです。Prefill 時に prompt の KV cache を作り、decode 時には新しい token の K/V だけを計算して cache に追加します。各ステップで完全な prefix を再計算するのではなく、過去の KV cache を読んで attention を行います。

KV cache の大きさは、おおよそ次の要素に比例します。

- batch size：同時に処理する系列数。
- sequence length：各系列がすでに持つ token 数。
- number of layers：各層で保存が必要。
- number of KV heads：key/value head の数。
- head dimension：各 head の次元。
- K と V の 2 種類のデータ、および BF16 などの数値精度。

そのため KV cache は GPU メモリを大量に消費しやすい要素です。長い文脈、多数の同時リクエスト、大きな batch は、メモリ使用量を急速に増やします。

## 5. なぜ Memory Bandwidth がボトルネックになるのか

MLP 層では、異なるリクエストが同じモデル重みを共有します。batch が大きいほど、重みを 1 回読んでより多くの token を処理できるため、batch によって算術強度を上げられます。

しかし attention の decode はより厄介です。各系列はそれぞれ固有の KV cache を持ちます。batch が大きくなっても、各リクエストは自分専用の過去 K/V を読む必要があり、MLP 重みのように batch 内で大量に再利用できません。そのため attention decode の算術強度はほぼ定数に近く、しばしば memory-bound になるほど低くなります。

ここから推論最適化の中心原則が見えてきます。**HBM から読み書きするデータ量を減らすことは、単に FLOPs を減らすことより重要な場合が多い** ということです。

## 6. Batch Size、Latency、Throughput のトレードオフ

batch size を大きくすると throughput は向上します。1 回の decode step で `B` 個のリクエストそれぞれに 1 token ずつ生成できるからです。しかし代償もあります。

1. 各ステップでより多くの系列を処理するため、1 ステップあたりの latency が上がる可能性がある。
2. 各系列が KV cache を保持する必要があり、メモリ使用量は batch とともに増える。
3. throughput の向上には限界効用の逓減があり、最終的にはメモリ容量に制限される。

1 人のユーザーだけを処理するなら latency は低くできますが、GPU 利用率は悪くなります。多くのユーザーをまとめれば throughput はよくなりますが、ユーザーの待ち時間は長くなり得ます。これが LLM serving の基本的な緊張関係です。

単純で有効な拡張方法の 1 つが **replication（モデル複製）** です。複数 GPU にそれぞれモデルのコピーを置きます。学習時のような複雑な勾配同期は不要で、latency はほぼ変わらず、総 throughput はレプリカ数にほぼ比例して増えます。モデルが大きすぎる場合には、tensor parallelism、pipeline parallelism、KV cache sharding などのより複雑な戦略が必要になります。

## 7. KV Cache を小さくするアーキテクチャ上の工夫

decode は主に KV cache の読み書きに制限されるため、現代的なアーキテクチャ変更の多くは「KV cache を小さくする」方法として理解できます。

### 7.1 GQA：Grouped-Query Attention

従来の multi-head attention では、query heads、key heads、value heads の数は同じです。**MQA（Multi-Query Attention）** は極端に、すべての query が 1 組の K/V を共有しますが、表現力が不足する可能性があります。**GQA（Grouped-Query Attention）** はその中間で、複数の query heads がより少数の KV heads を共有します。

これにより query の表現力を大きく損なわずに、KV cache に保存する head 数を大幅に減らせます。KV cache が小さくなると、メモリ使用量が減り、memory transfer も減り、latency と throughput の両方が改善します。同時に、より大きな batch size も使いやすくなります。

### 7.2 MLA：Multi-head Latent Attention

DeepSeek 系列が提案した **MLA（Multi-head Latent Attention）** は、必ずしも KV head 数を減らすわけではありません。代わりに K/V をより低次元の latent space に射影し、それを cache します。つまり完全な高次元 K/V を保存するのではなく、圧縮表現を保存し、必要なときに復元したり計算に参加させたりします。これは「次元」の方向から KV cache を小さくし、同じくメモリと帯域の圧力を下げることを狙います。

### 7.3 CLA：Cross-Layer Attention

**CLA（Cross-Layer Attention）** は層の間で K/V 表現を共有します。GQA は head 間の共有であり、CLA は layer 間の共有です。KV cache は本来、各層ごとに 1 つ保存するため、層をまたいだ共有によって cache size をさらに減らせます。ただし、モデル品質と効率の間でトレードオフが必要です。

### 7.4 Local / Sliding Window Attention

**Local attention（局所注意）** または **sliding window attention（スライディングウィンドウ注意）** は、直近 `K` token だけに attention します。非常に長い系列を生成するとき、ウィンドウ外の KV は捨てられるので、cache は総系列長に比例して増え続けるのではなく、ほぼウィンドウサイズで固定されます。

問題は、純粋な局所注意では長距離依存を扱う能力が損なわれることです。そのため実際のモデルでは hybrid design がよく使われます。多くの層では local attention を使い、少数の層では full/global attention を残す、あるいは KV sharing や GQA などと組み合わせます。

## 8. より大胆な方向：Transformer を変える

full attention の KV cache が根本的なボトルネックなら、full attention 層を減らす、あるいは transformer 構造の一部を置き換えるという方向があります。

- **State Space Models（SSM）/ Mamba**：系列長とともに増える KV cache の代わりに RNN のような状態表現を使い、推論状態をほぼ定数サイズにする。課題は言語モデリング能力、とくに遠くの情報を正確に検索する associative recall のようなタスクの能力を保つこと。
- **Linear Attention（線形注意）**：kernel 関数や特徴写像で attention を書き換え、計算量を二次から線形にし、recurrent state に近い実装形を持たせる。現代のモデルでは linear/local/full attention を混合することが多い。
- **Diffusion language models（拡散型言語モデル）**：厳密な自己回帰的 token-by-token 生成をやめ、テキストの一部を並列に生成して繰り返し refine する。ハードウェアを使い切りやすい一方、テキスト品質と汎用性はまだ研究課題である。

これらの方法は、推論最適化が単なるシステム工学ではなく、モデルアーキテクチャ設計を逆に押し動かすことを示しています。

## 9. Quantization、Pruning、Distillation

**Quantization（量子化）** は数値精度を下げることでメモリ使用量と帯域を減らします。たとえば BF16 から FP8、INT8、さらには INT4 に下げます。推論は memory-bound になりやすいため、パラメータや KV 要素あたりの byte 数を減らすと、速度と容量を直接改善できます。ただし低精度は誤差を生みます。特に大モデルには outliers（異常に大きい activation や重み）が存在するため注意が必要です。一般的な方法には post-training quantization、outliers だけ高精度で保持する方法、activation-aware quantization などがあります。

**Pruning（枝刈り）** は重要でない layer、head、hidden dimensions を削除し、モデル構造自体を小さくします。枝刈り後のモデルは通常性能が落ちるため、しばしば **distillation（蒸留）** と組み合わせます。元の大モデルを teacher とし、枝刈り済みまたは小型の student model に能力を移します。

これらの方法は通常 lossy です。速度とコストは改善しますが、品質が許容できるかを検証する必要があります。

## 10. Speculative Decoding：小モデルで大モデルを高速化する

**Speculative decoding / speculative sampling（投機的デコード/投機的サンプリング）** の重要な観察は、与えられた token 列を検証するほうが、それらを 1 つずつ生成するより速いということです。検証は prefill のように並列化できますが、生成は自己回帰的でなければなりません。

手順は次の通りです。

1. 安価な **draft model（草稿モデル）** を使って、`K` 個の候補 token を自己回帰的に生成する。
2. 高価な **target model（対象モデル）** を使って、これらの token の確率を並列に計算する。
3. accept-reject ルールに従って、どれだけの draft token を残すか決める。
4. ある token が拒否された場合は、target model の補正分布からサンプリングして続ける。

数学的には、accept-reject ステップが正しく実装されていれば、この方法は target model から直接サンプリングした場合と同じ出力分布、つまり “exact sampling from target model” を保証できます。高速化の効果は draft model が target model にどれだけ近いかに依存します。draft が正確なほど受理率は高くなり、速度も上がります。Medusa、EAGLE などの方法は、よりよい draft や並列的な草稿生成を中心に発展しています。

## 11. Serving Systems：実トラフィック下のシステム問題

学習時の batch は通常、整った dense token block です。一方、サービス時のリクエストは動的です。到着時刻、prompt 長、生成長が異なり、prefix を共有する場合もあれば、すぐ終わる場合もあります。そのため serving system には動的スケジューリングが必要です。

### 11.1 Continuous Batching

**Continuous batching（連続バッチ処理）** は、batch 全体が完了するまで新しいリクエストを待たせるのではありません。各 decode step の後に制御を scheduler に戻し、完了したリクエストは退出し、新しいリクエストは参加します。これにより GPU の空転が減り、throughput が向上します。

### 11.2 Selective Batching

リクエストの長さはそれぞれ異なるため、attention 部分を完全にきれいに batch 化するのは難しいです。しかし MLP 部分は系列間の相互作用に依存しないため、異なる長さの token を batch 次元へ flatten して一緒に計算できます。これが **selective batching（選択的バッチ処理）** の考え方です。

### 11.3 PagedAttention と vLLM

KV cache が動的に増えると GPU メモリの断片化が起きます。リクエストの生成長は事前にわからないため、先に確保すると無駄が出ます。リクエスト終了後には不連続な空き領域が残ります。**PagedAttention** は OS の仮想メモリの考え方を借り、KV cache を固定サイズの block/page に分割します。リクエストの論理的に連続した文脈を、物理的には不連続な GPU メモリブロックへ対応付けられるようにします。これにより断片化を減らし、メモリ利用率を高めます。

複数のリクエストが prefix を共有する場合は、**copy-on-write（書き込み時コピー）** も使えます。同じ KV block を共有し、その後の生成が分岐したときだけコピーすることで、さらにメモリを節約できます。

## 12. サンプリングとデコード戦略

ここまでの内容は主に「次 token の logits をどう速く計算するか」を扱いました。しかし実際にテキストを生成するには、logits または確率分布から token を選ぶ **decoding strategy（デコード戦略）** が必要です。

最も単純なのは **greedy decoding（貪欲デコード）** で、各ステップで最も確率の高い token を選びます。安定していて、安価で、再現可能ですが、テンプレート的な文章を出しやすく、多段推論や創作では早い段階で経路を固定しすぎることがあります。**beam search（ビームサーチ）** は複数の候補系列を同時に保持する方法で、従来の機械翻訳ではよく使われました。しかしオープンエンドな LLM 対話では、反復的で保守的な答えを生成しやすく、計算とメモリの負担も増えます。

より一般的なのはランダムサンプリングです。**temperature（温度）** は logits をスケールします。温度が低いと分布は鋭くなり、出力はより決定的になります。温度が高いと分布は平坦になり、出力は多様になります。**top-k sampling** は確率上位 `k` 個の token だけからサンプリングし、低確率の長い尾を切り落とします。**top-p / nucleus sampling（核サンプリング）** は累積確率が `p` に達する最小の token 集合を選び、候補集合の大きさを動的に決めます。実際のサービスでは repetition penalty、frequency penalty、stop sequences、最大出力長なども加え、反復、終了、コストを制御します。

これらの戦略自体は transformer の主要な計算ボトルネックを変えませんが、生成長、受理率、ユーザーが感じる品質に影響します。たとえば speculative decoding の速度は、draft token が target model に受理される確率に依存します。temperature が高くサンプリングがランダムになるほど、小さな draft model が大モデルを予測するのは難しくなり、受理率が下がる可能性があります。したがって推論システムでは、モデル、サンプリングパラメータ、サービス目標をまとめて調整する必要があります。

## 13. まとめ

LLM inference の中心的な難しさは autoregressive decode にあります。毎回 1 token しか生成せず、並列化しにくく、モデル重みと各系列専用の KV cache を繰り返し読む必要があるため、通常 memory bandwidth に制限されます。Prefill は比較的並列化しやすく、本当の主要ボトルネックは decode です。

最適化の道筋は大きくいくつかに分けられます。

- システム面：continuous batching、selective batching、PagedAttention、モデル複製と並列化。
- アーキテクチャ面：GQA、MLA、CLA、local attention、SSM、linear attention、diffusion models。
- モデル圧縮：quantization、pruning、distillation。
- デコードアルゴリズム：speculative decoding。小モデルの草稿と大モデルの検証を組み合わせ、分布を変えずに高速化する。

最終目標は、単に「固定された transformer を速く動かす」ことではありません。latency、throughput、メモリ、コストの制約のもとで、できるだけ高品質なモデル出力を提供することです。推論効率は、現代 LLM のアーキテクチャ、アルゴリズム、システムを共同設計する中心的な駆動力になっています。
