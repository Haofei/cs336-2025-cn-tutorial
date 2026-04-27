# Stanford CS336 2025 Language Modeling from Scratch — 日本語完全チュートリアル


出典: Stanford CS336 2025 の公開講義動画・transcript。本資料は学習用に再構成した非公式チュートリアル版です。


---


# CS336 2025 第1講チュートリアル：コース概観と Tokenization

> これは Chinese CS336 2025 study guide の日本語チュートリアル版です。

## 学習目標

この講義を終えると、次のことを説明できるようになります。

1. CS336「Language Models from Scratch」は API を呼ぶだけの授業ではなく、language model を作るパイプラインを理解し、実装する授業である。
2. コースの中心問題は、限られた compute（計算資源）と data（データ）予算のもとで、最良のモデルをどう訓練するかである。
3. tokenizer、Transformer、training、systems optimization、scaling laws、data、evaluation、alignment までの全体像を理解する。
4. tokenization は Unicode 文字列を整数列に変換し、モデルが扱える形にする処理である。
5. BPE（Byte Pair Encoding）の考え方、学習手順、encode/decode の流れを理解する。

## 前提知識

推奨される前提は次の通りです。

- Python と PyTorch の基礎。
- loss、optimizer、batch size、overfitting などの基本的な機械学習概念。
- Transformer や attention についての大まかな理解。詳細は授業内で下から組み立てます。
- GPU や並列計算の専門知識は不要ですが、性能ボトルネックを工学的に考える姿勢が必要です。

## 講義の地図

この講義は大きく二つに分かれます。

1. コース全体像：なぜ language model をゼロから作るのか、どのモジュールを扱うのか、研究とエンジニアリングの視点をどう結びつけるのか。
2. Tokenization 入門：なぜ tokenizer が必要か、文字単位・byte 単位・単語単位の問題点、そして BPE がどのように折衷案を与えるか。

コース全体の流れは次のように表せます。

```text
raw data → cleaning/filtering → tokenizer → integer sequences → Transformer → training → evaluation → systems optimization → alignment/fine-tuning → usable model
```

## 1. なぜ language model をゼロから作るのか

講義の冒頭では、研究者が基盤技術から遠ざかっているという問題が指摘されます。以前の NLP 研究では、研究者自身がモデルを実装し訓練することが普通でした。その後、BERT などをダウンロードして fine-tuning する時代になり、現在では closed model に prompt を投げるだけで研究やアプリケーションが成立することも増えています。

これは便利ですが、抽象化はしばしば「漏れ」ます。language model API は一見「文字列を入れると文字列が返る」だけに見えます。しかし、その背後の data、model、systems、training の仕組みを知らなければ、基礎的な研究や深い改善は難しくなります。CS336 の基本姿勢は次の一文です。

```text
To understand it, you have to build it.
理解するには、自分で作らなければならない。
```

ただし授業は現実的でもあります。frontier models は巨額の資本、大規模 GPU クラスタ、公開されていない実装上の詳細を必要とします。授業で全員が GPT-4 級のモデルを訓練することはできません。そこで小さなモデルを訓練しながら、小規模実験から何が学べて、何は学べないのかを明確にします。

授業では知識を三つに分けます。

- mechanics：Transformer の実装や GPU 並列の仕組み。これは具体的に学べます。
- mindset：常に効率と scale を意識し、ハードウェアを有効利用する考え方。
- intuitions：大規模で有効な data や architecture の直感。これは小規模実験だけでは部分的にしか得られません。

## 2. 中心視点：盲目的な巨大化ではなく効率

講師は bitter lesson を「規模だけが重要で algorithm は重要でない」と誤解しないよう注意します。より正確には次の通りです。

```text
Algorithms at scale matter.
scale したときに効く algorithm が重要である。
```

モデル性能は、投入資源と効率の組み合わせで決まります。資源が高価になるほど、効率の重要性は増します。一回の訓練に莫大な費用がかかるなら、手元の小実験のように何度も無駄に試すことはできません。algorithmic efficiency、hardware utilization、data quality、model architecture のすべてが結果に影響します。

この授業が繰り返し問う問題は次です。

```text
与えられた compute budget と data budget のもとで、訓練できる最良のモデルは何か。
```

これはエンジニアリングの核心です。「モデルをもっと大きくできるか」だけではなく、「各 FLOP、各 GPU、各 token が有効に使われているか」を問います。

## 3. language model の簡単な歴史

language model は近年突然現れたものではありません。Shannon は英語のエントロピー推定に language model を使いました。従来の NLP でも、機械翻訳や音声認識の部品として language model が使われてきました。深層学習の時代には、次の要素が蓄積されました。

- neural language model
- seq2seq
- Adam optimizer
- attention mechanism
- Transformer
- model parallelism
- ELMo、BERT、T5 などの foundation models

GPT-2、GPT-3 以降、scaling laws と工学的な大規模訓練が中心になりました。同時にモデルの公開度にも階層が生まれました。

- closed models：API からしか使えない。
- open-weight models：重みは使えるが、データや訓練詳細が不完全なことがある。
- open-source models：重み、データ、実装をできるだけ公開する。ただし論文を読むだけでは、自分で作る経験の代わりにはなりません。

## 4. コースの五つのモジュール

### 4.1 Basics

最小だが完全な language model training pipeline を実装します。

- tokenizer：文字列と整数列の相互変換。
- model architecture：主に Transformer。
- training：loss、optimizer、learning rate schedule、training loop。

課題では BPE tokenizer、Transformer、cross-entropy loss、AdamW optimizer、training loop を実装します。PyTorch は使えますが、既成の Transformer 実装を呼ぶことが目的ではありません。

### 4.2 Systems

訓練は数式だけではなくハードウェアの問題でもあります。GPU の演算器はチップ上にありますが、HBM などのメモリは外側にあり、data movement がボトルネックになり得ます。扱う話題は次の通りです。

- kernels：行列積を tiling や fusion で高速化し、データ移動を減らす方法。
- Triton：高性能 GPU kernel を書くためのツール。
- parallelism：data parallelism、tensor/model parallelism など。
- inference：訓練済みモデルが token を生成する過程。

inference には二つの段階があります。

- prefill：prompt を処理する。入力 token がすべて既知なので並列化しやすく、training に近い。
- decode：自己回帰的に token を一つずつ生成する。GPU を使い切りにくく、memory-bound になりやすい。

また speculative decoding も紹介されます。安価な小モデルが候補を作り、大モデルがそれを並列に検証することで推論を高速化します。

### 4.3 Scaling Laws

中心問題は、FLOPs 予算が決まっているとき、model parameters と training tokens をどう配分するかです。大きいモデルは少ないデータで済む場合があり、小さいモデルはより多くの token を見られます。最適点はどこでしょうか。

Chinchilla optimal の考え方では、小規模実験で関係式を当てはめ、大規模訓練での最適な parameter 数や loss を予測します。安い実験で高価な訓練判断を導ける点が重要です。

ただし、parameter 数と training token 数の比率に関する経験則には前提があり、inference cost を含まないことも多いため、機械的に使うべきではありません。

### 4.4 Data と Evaluation

モデル能力は大きく data に依存します。多言語 data で訓練すれば多言語能力が、code data で訓練すれば coding 能力が得られます。代表的なデータ源には Web/Common Crawl、Wikipedia、GitHub、StackExchange、書籍、論文などがあります。

しかし「インターネットをそのまま食べさせる」という表現は誤解を招きます。生の Web data には HTML、PDF、コードリポジトリ、スパム、重複、法的・安全上の問題が含まれます。必要な処理は次です。

- extraction：HTML/PDF などをテキストに変換する。
- filtering：低品質、有害、無関係な内容を除く。
- deduplication：重複を削除し、training budget の浪費を避ける。
- legal considerations：どの data を訓練に使えるかを検討する。

evaluation では次を扱います。

- perplexity：次 token 予測能力の指標。
- MMLU などの standardized benchmarks。
- instruction following evaluation。
- language model を含む agentic system 全体の評価。

### 4.5 Alignment

pretraining で得られる base model は主に次 token を予測します。潜在能力はありますが、必ずしも指示に従うとは限りません。alignment は、モデルをより有用で安全で、対話に適したものにする過程です。

目標は次の通りです。

- instruction following
- style control：長さ、箇条書き、口調などの制御
- safety：有害な要求への安全な拒否

代表的な段階は次です。

- SFT, supervised fine-tuning：user/assistant の prompt-response ペアで教師あり学習する。
- learning from feedback：preference data や verifier を使って改善する。
- PPO、DPO、GRPO：強化学習または preference optimization の手法。DPO は preference data に適し、GRPO は DeepSeek などで使われた PPO 簡略化系の方法です。

## 5. Tokenization：なぜ tokenizer が必要か

language model は数値 tensor を処理しますが、生のテキストは Unicode 文字列です。tokenization は文字列を整数列に変換し、できれば元の文字列に decode できるようにする処理です。

```text
encode: string → list[int]
decode: list[int] → string
```

vocabulary size は token ID の種類数です。語彙が大きいほど一つの token が長い断片を表しやすい一方、入力・出力層が大きくなります。語彙が小さいと sequence が長くなり、attention cost が増えます。

重要な点として、現代の tokenizer は通常可逆で、空白も token に含めます。例えば “hello” と “ hello” は異なる token になることがあります。これは単純な空白分割とは異なります。

## 6. 素朴な tokenization と問題点

### 6.1 Character-based tokenization

各 Unicode 文字を code point に対応させます。英字も emoji も整数で表せます。

問題点：

- Unicode の範囲が非常に大きい。
- ほとんど出現しない文字が語彙を消費する。
- compression が悪い。

### 6.2 Byte-based tokenization

文字列を UTF-8 bytes に変換し、各 byte を token とします。byte は 0 から 255 なので語彙は小さく、任意のテキストを表現できます。

利点は単純で、未知文字問題がないことです。問題は sequence が長すぎることです。一つの token が 1 byte しか表さないため compression ratio が低く、標準的な attention は sequence length に対して二次コストなので非効率です。

### 6.3 Word-based tokenization

空白、正規表現、pre-tokenization rules で単語や断片に分割し、それぞれに整数を割り当てます。

利点は、頻出語を一つの token で表せるため sequence が短くなることです。問題は語彙が巨大になりやすく、未知語、綴り、固有名詞、コード断片などに必ず遭遇することです。UNK token は情報を失わせ、評価も難しくします。

## 7. BPE：Byte Pair Encoding

BPE は古い圧縮アルゴリズムで、後に neural machine translation や GPT-2 などの language model に使われました。核心は、「何が単語か」を手で決めるのではなく、corpus statistics から token を学習することです。

直感は次の通りです。

- 高頻度の連続断片は一つの token にまとめ、compression を良くする。
- 低頻度の断片は複数 token のままでよく、語彙を浪費しない。
- bytes から始めれば、任意の文字列を表現できる。

BPE の訓練手順：

```text
Input: training corpus, target vocabulary size or number of merges
Initialize: text を byte sequence に変換し、初期語彙を 0..255 にする
Repeat:
  1. 現在の sequence 内の隣接 token pair を数える
  2. 最頻出 pair、例えば (116, 104) を見つける
  3. その pair に新しい token id、例えば 256 を割り当てる
  4. training sequence 内のその pair を新 token に置き換える
Output: merge rules and vocabulary
```

新しいテキストを encode するとき：

```text
1. 文字列を bytes に変換する
2. 学習した merge rules を順番に適用する
3. integer token sequence を得る
```

decode するとき：

```text
1. 各 token id を対応する byte sequence に戻す
2. bytes を連結する
3. UTF-8 として文字列に戻す
```

GPT-2 風 tokenizer では、まず regular expression で pre-tokenization を行い、その断片内部で BPE を実行します。これは効率と挙動制御のための実用的な折衷です。

## 8. よくある誤解

1. 「すべての問題で、まずゼロから訓練すべき」
   違います。prompting や fine-tuning で解けるなら、それを優先すべきです。ゼロからの訓練は、基盤を学ぶ場合や本当に新しい base model が必要な場合に適します。

2. 「小モデルの結論は必ず大モデルに外挿できる」
   必ずしもそうではありません。attention と MLP の FLOPs 比率、emergent behavior、安定性は scale によって変わります。

3. 「tokenization は単なる前処理」
   違います。tokenizer は sequence length、training efficiency、vocabulary size、可逆性、多言語・コード性能に直接影響します。

4. 「byte-level tokenizer はきれいなので常に最良」
   優雅ではありますが、現在の Transformer では sequence が長すぎて非効率になりがちです。

5. 「インターネット data はそのまま訓練できる」
   できません。Common Crawl などにはスパム、重複、HTML/PDF 構造、法的・安全上の問題があるため、丁寧な処理が必要です。

## 9. 演習

1. tokenizer visualization tool を開き、英語、日本語または中国語、数字、コード、emoji を入力して token boundary を観察する。
2. `encode(str) -> list[int]` と `decode(list[int]) -> str` を持つ最小 byte tokenizer を実装する。
3. 小さな corpus で BPE merge を 3 回手で実行し、各回の最頻出 pair と新 token id を記録する。
4. 同じテキストを character、byte、BPE で token 化し、token 数を比較する。sequence length が attention cost にどう効くか考える。
5. Web text をランダムにサンプルし、高品質、filter すべき、deduplicate すべき内容を判断する。

## 10. まとめ

この講義は CS336 の全体枠組みを示しました。language model は孤立した Transformer ではなく、end-to-end の engineering pipeline です。tokenizer、model、training loop、systems optimization、data pipeline、evaluation、alignment を下から作りながら、限られた compute と data で最良のモデルを得る方法を考えます。

Tokenization はその入口です。character、byte、word 単位の方法にはそれぞれ欠点があります。BPE は bytes から始め、高頻度の隣接 token を繰り返し merge することで、表現可能性と compression efficiency の実用的なバランスを取ります。将来 tokenizer-free architecture が成熟する可能性はありますが、現在の frontier model 実践では BPE とその変種が重要な基礎であり続けています。

## 参考と次回への接続

- Andrej Karpathy の tokenization と from-scratch model に関する動画。
- Transformer 原論文 “Attention Is All You Need”。
- GPT-2 tokenizer と byte-level BPE 実装。
- Chinchilla scaling laws 関連論文。

次回は PyTorch と resource accounting に進みます。動くプログラムを書くだけでなく、FLOPs、memory、data movement を追跡し、計算資源がどこで使われるのかを理解します。


---


# Stanford CS336 2025 第2講チュートリアル：PyTorch と Resource Accounting

> これは Chinese CS336 2025 study guide の日本語チュートリアル版です。

この講義の主題は「Transformer をもう一度説明すること」ではありません。大規模モデルを訓練するときに必要になる、より低レベルで見落とされがちな能力を扱います。すなわち、PyTorch でモデルを組み、同時に memory、compute、time、cost を概算できるようになることです。研究コードは「動けばよい」だけでは不十分です。parameters、tokens、GPUs が増えると、各 matrix multiplication、optimizer state、CPU/GPU data transfer が実際のコストになります。

## 1. なぜ resource accounting が必要か

講義は二つの紙上計算から始まります。

第一の問題：1024 枚の H100 で、70B parameters の dense Transformer を 15 trillion tokens で訓練すると、どれくらい時間がかかるか。粗い式は次です。

```text
training FLOPs ≈ 6 × parameters × tokens
usable FLOPs/day ≈ number of GPUs × peak FLOPs/s per GPU × MFU × 86400
training days ≈ total training FLOPs / usable FLOPs per day
```

H100 の実効利用率 MFU を 0.5 と仮定すると、結果は百日を超えるオーダーになります。重要なのは正確な数字ではなく、まず総 compute を見積もり、それを実効 throughput で割るという考え方です。

第二の問題：8 枚の 80GB H100 で AdamW を使い、複雑な最適化をしない場合、どれくらい大きなモデルを載せられるか。よく使う概算は 1 parameter あたり約 16 bytes です。parameter、gradient、Adam の first moment、second moment などが memory を使うためです。

```text
maximum parameters ≈ 8 × 80GB / 16 bytes ≈ 40B parameters
```

これは activation、batch size、sequence length などを真面目には含まないので、上限に近い概算です。実際の訓練では activation も大きなボトルネックになります。

## 2. PyTorch tensor：すべての原子

PyTorch では parameter、gradient、optimizer state、data、intermediate activation はすべて tensor です。tensor の保存方法を理解することが memory accounting の第一歩です。

Tensor の memory は、要素数と 1 要素あたりの bytes で決まります。

```python
x = torch.zeros(4, 8)       # default float32
x.numel()                   # 32 elements
x.element_size()            # 4 bytes
memory = 32 * 4             # 128 bytes
```

代表的な dtype は次の通りです。

| Type | Bytes/element | 特徴 |
|---|---:|---|
| FP32 / float32 | 4 | 従来の default。安定だが遅く memory を使う |
| FP16 / float16 | 2 | memory を節約し高速だが dynamic range が狭く underflow/overflow しやすい |
| BF16 / bfloat16 | 2 | FP32 に近い指数範囲を持ち、大規模深層学習でよく使う |
| FP8 | 1 | H100 などで対応。速度と memory の利点が大きいが training stability は難しい |

FP16 と BF16 はどちらも 16 bit ですが、bit の配分が違います。FP16 は mantissa に多くを割り当てる一方、dynamic range が狭いです。BF16 は FP32 に近い exponent range を保つため、非常に小さい値や大きい値を表しやすく、大規模モデル訓練に向きます。実際には parameter の master copy や optimizer state を FP32 で持ち、forward/backward の matmul を BF16 や FP8 で行うことがよくあります。

これが mixed precision training の核心です。モデル全体を一つの dtype にするのではなく、安定性と throughput のトレードオフを部位ごとに決めます。

## 3. Device placement と data movement

PyTorch は既定で CPU 上に tensor を作ります。

```python
x = torch.zeros(32, 32)     # CPU RAM
x = x.to("cuda")           # GPU HBM
```

GPU 上に直接作ることもできます。

```python
x = torch.zeros(32, 32, device="cuda")
```

訓練中は、各 tensor がどこにあるかを常に把握します。CPU RAM から GPU HBM への転送は無料ではありません。頻繁な転送は GPU を計算ではなく data 待ちにします。実用コードでは `x.device` を確認する assertion や log を入れ、batch、mask、loss target が誤って CPU に残らないようにします。

## 4. Tensor は配列そのものではなく storage への view

PyTorch tensor は underlying storage を指すオブジェクトで、shape、stride、offset などの metadata を持ちます。連続な 2D 行列の stride は例えば `(4, 1)` です。行方向に 1 進むと 4 要素飛び、列方向に 1 進むと 1 要素進みます。

このため、多くの操作はほぼ無料です。slice、transpose、view は多くの場合、metadata を変えるだけで data をコピーしません。

```python
x = torch.arange(6).view(2, 3)
y = x[0]       # view, shares storage
z = x.T        # transpose, usually shares storage
```

ただし shared storage には注意が必要です。`x` を in-place で変更すると `y` にも見えます。また transpose 後の tensor は非 contiguous になりがちで、一部の `view` は失敗します。その場合は次が必要です。

```python
z = x.T.contiguous()
```

`contiguous()` は実際に data をコピーすることがあるため無料ではありません。高性能コードでは「view を変えただけ」なのか「新しい memory を確保した」なのかを区別します。

## 5. Dimension に名前をつけて tensor bug を減らす

実際のモデルの tensor は batch、sequence、head、hidden など複数次元を持ちます。`transpose(-2, -1)` や `view(b, s, h, d)` はよく使われますが、保守時にはミスの原因になります。`-1` は hidden なのか head_dim なのか、dimension を変えた後も comment は正しいのか、という問題が起きます。

講義では dimension に意味を持たせる書き方を推奨します。`einsum` は matrix multiplication を dimension 名つきで書けます。attention score は次のように表せます。

```python
scores = torch.einsum(
    "batch seq_q hidden, batch seq_k hidden -> batch seq_q seq_k",
    q, k,
)
```

output に現れない `hidden` は sum され、`batch`、`seq_q`、`seq_k` は残ります。つまり「hidden 方向に内積を取り、query と key の類似度を得る」とコードが直接示します。

`einops.rearrange` は reshape と transpose の組み合わせに便利です。

```python
x = rearrange(x, "batch seq (heads dim) -> batch heads seq dim", heads=num_heads)
```

この書き方自体が compute を減らすとは限りませんが、shape 間違いを大きく減らします。教育・研究コードでは可読性も engineering efficiency です。

## 6. Matrix multiplication が深層学習の主コスト

多くの elementwise operation の FLOPs は tensor 要素数に比例します。しかし大規模モデルで支配的なのは matrix multiplication です。

```text
[B, D] × [D, K] -> [B, K]
```

各 output 要素には D 回の multiply とおよそ D 回の add が必要なので、概算 FLOPs は次です。

```text
FLOPs ≈ 2 × B × D × K
```

この規則は非常に重要です。matmul の FLOPs は三つの次元の積に 2 を掛けたものです。

`B` を tokens または data points、`D × K` を parameters とみなすと、linear layer の forward は次のように見積もれます。

```text
forward FLOPs ≈ 2 × tokens × parameters
```

matmul が支配的である限り、この考え方は Transformer にも粗く拡張できます。attention の二次項、sequence length、その他の操作による補正はありますが、紙上計算には有用です。

## 7. FLOPs と FLOPs/s を混同しない

“FLOPs” は floating point operations、つまり総演算数を指すこともあれば、floating point operations per second の意味で使われることもあります。混乱を避けるには次のように書き分けます。

```text
FLOPs      = total floating-point operations
FLOPs/s    = floating-point operations per second
```

A100 や H100 の仕様表には FP32、TF32、BF16、FP8 などでの peak FLOPs/s が載ります。低精度ほど throughput は高い傾向があります。ただし 2:4 sparsity のような structured sparsity 前提の数値もあるため、dense model では宣伝上の最大値をそのまま使えません。

実際の訓練では MFU を見ます。

```text
MFU = model effective FLOPs/s / hardware peak FLOPs/s
```

MFU はハードウェアをどれだけ使い切っているかの指標です。大きな matmul が多いほど高 MFU になりやすく、小 batch、細かい kernel、多い communication、data movement は MFU を下げます。実践では 0.5 を超えればかなり良く、数 % しかないなら code や parallelization に深刻な bottleneck がある可能性が高いです。

## 8. Autograd と backward のコスト

PyTorch の autograd により gradient を手書きする必要はありません。

```python
pred = x @ w
loss = ((pred - y) ** 2).mean()
loss.backward()
w.grad        # PyTorch が自動で埋める
```

しかし autograd は無料ではありません。linear layer を考えます。

```text
X: [B, D]
W: [D, K]
H = XW: [B, K]
```

forward `H = XW` の cost は次です。

```text
2 × B × D × K
```

backward では少なくとも二つの gradient を計算します。

```text
dL/dW = X^T × dL/dH
dL/dX = dL/dH × W^T
```

それぞれ同規模の matmul なので、backward 全体はおよそ次です。

```text
backward FLOPs ≈ 4 × B × D × K
```

したがって 1 training step の主要 matmul については、

```text
forward + backward ≈ 6 × tokens × parameters
```

となります。大規模訓練見積もりの係数 6 はここから来ています。

## 9. Parameters、initialization、nn.Module

PyTorch の trainable parameters は通常 `nn.Parameter` として `nn.Module` 内に置きます。

```python
class Cruncher(nn.Module):
    def __init__(self, d, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d, d, bias=False) for _ in range(num_layers)
        ])
        self.head = nn.Linear(d, 1, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)
```

Initialization では標準正規分布をそのまま使うべきではありません。`W ~ N(0, 1)` で input dimension が大きいと output variance が fan-in とともに増え、activation が爆発します。よく使う scaling は次です。

```python
w = torch.randn(d_in, d_out) / math.sqrt(d_in)
```

これは Xavier/Glorot initialization の考え方と同じで、層をまたぐ signal scale を安定させます。極端な値を避けるため truncated normal を使うこともあります。

## 10. Optimizer state も memory の大きな部分

訓練時の memory は parameters だけではありません。Adam/AdamW では各 parameter に通常次が対応します。

1. parameter 自体
2. gradient
3. first moment `m`
4. second moment `v`
5. 場合によって FP32 master weights や temporary buffers

これらを主に FP32 で持つと、1 parameter あたり十数 bytes になることは普通です。したがって「parameter 数 × dtype size」は training memory を大きく過小評価します。

Adagrad のようなより単純な optimizer でも、累積 squared gradient を保存します。optimizer の `step()` は `p.grad` を読み、state を更新し、parameter を in-place 更新します。state は step をまたいで保持されるため、一時変数ではなく長期 memory です。

## 11. Activation：なぜ forward の中間結果を保存するか

backpropagation には forward で作られた intermediate activation が必要です。例えば第一層の weight gradient を計算するには、その層の input activation が必要です。そのため autograd は多くの中間結果を保存します。

単純な深い linear model で batch が `B`、width が `D`、layers が `L` なら、activation 数は粗く次です。

```text
activations ≈ B × D × L
```

総 memory はカテゴリごとに見積もれます。

```text
total memory ≈ bytes_per_elem × (parameters + gradients + optimizer state + activations)
```

Transformer では activation は sequence length、attention matrix、MLP intermediate dimension にも依存します。memory が足りない場合、activation checkpointing が重要です。すべての activation を保存せず、backward 時に一部を再計算することで、追加 compute と引き換えに memory を減らします。

## 12. Data loading と training loop

language model data は通常 tokenizer が出力した integer sequence です。実データは TB 級になることがあり、すべて RAM に載せることはできません。よく使う方法は `numpy.memmap` で、disk file を配列として map し、必要な slice だけ読みます。

典型的な training loop は次です。

```python
model = Cruncher(d, num_layers).to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(num_steps):
    x, y = next_batch()
    x, y = x.to("cuda"), y.to("cuda")

    optimizer.zero_grad(set_to_none=True)
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
```

工学的には定期的な checkpoint が必須です。model state、optimizer state、current step、random-number state などを保存します。大規模訓練では interruption、preemption、OOM、node failure が必ず起きます。再開時には learning-rate scheduler、data iterator の位置、random seed も確認しないと、同じ実験が別の training curve になってしまいます。再現性は研究比較の前提であり、cluster time を節約する実務上の要件でもあります。

## 13. Compute、memory、bandwidth を一緒に見る

resource accounting では FLOPs だけ数えてはいけません。GPU training は compute units、memory capacity、memory/interconnect bandwidth の三つに制約されます。memory capacity は model と batch が載るかを決め、FLOPs/s は理想的な matmul 速度を決め、bandwidth は HBM、cache、CPU、GPU、多 GPU link の data movement 速度を決めます。

大量の multiply-add を行い、読み書きが少ない operation は compute-bound です。大きな matmul が典型です。逆に elementwise add、mask、copy、reshape による contiguous copy などは FLOPs が少なくても大量に data を読み書きするため memory-bound になり得ます。FLOPs が少ない operation でも GPU core が data 待ちになると training を遅くします。

Multi-GPU training では communication bandwidth も問題になります。data parallelism は gradients を同期し、tensor parallelism は layer 内の intermediate results を交換し、pipeline parallelism は activation を stage 間で渡します。多くの systems optimization は数学的な compute を減らすのではなく、compute と communication を overlap し、不要な copy を減らし、matrix shape を Tensor Core に適したものにします。

## 14. Research code から engineering cost awareness へ

この講義の最重要メッセージは、モデルを書くときに同時に「コスト帳簿」も書くことです。モデルを見たら、loss が下がるかだけでなく次を問います。

- parameters は何個か。
- 各 parameter は訓練時に実際何 bytes か。
- activation は batch size と sequence length に対してどう増えるか。
- 主な matmul の FLOPs はいくつか。
- BF16/FP8 により throughput は本当に上がるか。
- MFU は 50% か 5% か。
- CPU/GPU transfer、non-contiguous copy、小さな kernel、communication が遅くしていないか。

授業が単純な linear model で導出するのは、式を透明にするためです。forward はおよそ `2 × tokens × params`、backward は `4 × tokens × params`、training は `6 × tokens × params`。memory は parameters、gradients、optimizer state、activations に分けます。Transformer では帳簿は複雑になりますが、方法は同じです。

大規模モデル工学は、数学的に正しい network を書いたところで終わりません。code、numerical precision、hardware throughput、training cost が同時に成立する必要があります。PyTorch は autograd と module abstraction を提供しますが、効率的な訓練にはその抽象を見通す力が必要です。tensor がどこにあり、copy が起きるか、dtype は何か、matmul はどれほど大きいか、backward はどれだけ高価か、optimizer は何を保存するか。こうした resource accounting の意識があって初めて、研究 prototype は scalable、affordable、reproducible な training system になります。


---


# CS336 2025 第3講チュートリアル：Transformer Architecture と Hyperparameters

> これは Chinese CS336 2025 study guide の日本語チュートリアル版です。

この講義の主題は、language model を本当にゼロから訓練するなら、「Transformer とは何か」だけでなく、訓練安定性、throughput、最終性能に直接効く多くの工学的選択を理解する必要がある、ということです。現代の large language model は Transformer から完全に離れたわけではありませんが、2017 年の原論文そのままではありません。pre-norm、RMSNorm、bias なし、RoPE、SwiGLU、適切な width/depth ratio、いくつかの stability techniques という実用的な「標準レシピ」が形成されています。

以下では component ごとに設計選択を整理し、新しいモデルを訓練するときに使える経験則をまとめます。

## 1. Original Transformer から modern LLM Transformer へ

Original Transformer block はおおよそ次の要素から成ります。

1. token embedding と position encoding;
2. multi-head self-attention;
3. residual connection;
4. layer normalization;
5. feed-forward network、つまり MLP;
6. 最後の output softmax。

しかし modern LLM は原始版をそのまま使うことは少ないです。LLaMA 系列や授業課題に近い block は次の形です。

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

よく使われる設定は次です。

- normalization は sublayer の前、つまり pre-norm;
- normalization は traditional LayerNorm ではなく RMSNorm が多い;
- linear layer は通常 bias を使わない;
- position encoding は RoPE が多い;
- MLP は SwiGLU や他の GLU variants が多い;
- 一部の新しいモデルでは sublayer output 後にも norm を加える “double norm” 構造がある。

これらを魔法として覚えるのではなく、二つの目的から見ることが重要です。第一に training stability、第二に GPU efficiency です。

## 2. Residual と normalization：安定訓練の主軸

### 2.1 Post-norm と pre-norm

Original Transformer は post-norm を使いました。attention または MLP を実行し、residual を足し、その後 LayerNorm を行います。

```text
x = Norm(x + Attention(x))
x = Norm(x + MLP(x))
```

Modern LLM はほぼ pre-norm に移っています。

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

これは normalization の位置を動かしただけに見えますが、影響は大きいです。Residual stream の価値は、上層から下層へ gradient が流れる identity に近い経路を提供することです。norm を residual stream の途中に置くと、この直接経路を邪魔します。実践上 post-norm は gradient explosion、loss spike が起きやすく、warmup や learning rate に敏感です。pre-norm はより安定で、深いモデルを訓練しやすい傾向があります。

重要な経験則は、residual stream の identity connection を不用意に壊さないことです。norm は主に non-residual branch の入口または出口に置き、residual trunk 全体を何度も正規化しないようにします。

### 2.2 LayerNorm と RMSNorm

Traditional LayerNorm は各 token の hidden vector について、平均を引き、標準偏差で割り、learned scale gamma を掛け、bias beta を足します。RMSNorm はより単純で、平均を引かず、通常 beta も加えず、root mean square で scale します。

RMSNorm が流行した理由は次です。

- 性能は LayerNorm に劣らないことが多い;
- operation が少ない;
- parameters が少ない;
- 特に memory read/write が減る。

Transformer では多くの FLOPs は matmul から来ますが、それ以外の操作が重要でないわけではありません。softmax や normalization は FLOPs 比率が小さくても、memory movement に制限されるため実時間では無視できません。RMSNorm の利点は少し計算が減るだけでなく、動かす data が減る点にあります。

### 2.3 Bias なし linear layer

Modern LLM の linear layer は attention projection や MLP projection を含め、bias を外すことが多いです。経験的には性能を損なわないことが多く、parameters と memory access を減らせます。一部の報告では、特に大規模訓練で bias を外すことが optimization stability にも役立つとされています。

この節のまとめとして、modern Transformer の normalization design は stability と efficiency のためにあります。pre-norm は residual trunk を保ち、RMSNorm は normalization を単純化し、bias なし設計は余分な state と潜在的不安定要因を減らします。

## 3. MLP と activation：なぜ SwiGLU が default になったか

Transformer block で attention 以外に大きい component が MLP です。初期 Transformer は ReLU を使い、GPT 系では GELU が広く使われ、現代の多くのモデルでは GLU variants、特に SwiGLU が使われます。

通常の MLP は次の形です。

```text
MLP(x) = W2 * activation(W1 * x)
```

GLU 系構造は gate branch を追加します。

```text
MLP(x) = W2 * (activation(W1 * x) ⊙ (V * x))
```

`⊙` は elementwise multiplication です。直感的には、モデルは hidden features を作るだけでなく、どの次元を通し、どの次元を抑えるかを決める gate vector も学習します。

SwiGLU は Swish を非線形として使います。

```text
swish(x) = x * sigmoid(x)
```

多くのモデルと ablation では、GLU variants は ReLU/GELU MLP より小さいが安定した改善を与えることが示されています。SwiGLU がないと訓練できないという意味ではありません。GPT-3 は SwiGLU を使っていませんが非常に強いモデルです。ただし新しいモデルを設計するなら、SwiGLU は安全な default です。

GLU は追加 projection `V` を持ちます。parameter 数を通常 MLP とほぼ揃えるため、中間次元は普通 2/3 程度に縮めます。通常 MLP が `d_ff = 4 * d_model` なら、SwiGLU ではよく次を使います。

```text
d_ff ≈ 8/3 * d_model
```

これが多くの LLaMA-like model で MLP hidden size が 4 倍ではなく、約 2.6 から 2.7 倍に見える理由です。

## 4. Attention と position encoding：RoPE の現代的地位

Language model は token の順序を知る必要があります。初期には sinusoidal position embedding、learned absolute position embedding、relative position bias などが使われました。近年の dense LLM はほぼ RoPE、すなわち rotary position embedding に収束しています。

RoPE の核心は、attention がしばしば absolute position ではなく relative distance を重視するという点です。query と key の位置を同じだけ平行移動しても相対距離が変わらなければ、inner product の関係はなるべく保たれるべきです。

RoPE はこれを rotation で実現します。input embedding の下に position vector を足すのではなく、各 attention layer で query と key に position-dependent rotation をかけます。後ろの position ほど大きな角度で回転し、異なる dimension pair は異なる frequency を使うため、近距離と遠距離の情報を同時に表現できます。

2D で考えると、二つの vector が同じ角度だけ回転すれば相対角は変わらず、inner product も保たれます。RoPE は高次元 vector を複数の 2D pair に分け、それぞれを固定 frequency で回転します。その結果、query-key inner product が自然に relative position を encode します。

RoPE が広く使われる理由は次です。

- relative position modeling が自然;
- short context と long context の両方で性能が良い;
- context length extrapolation や extension の技術が多い;
- 多くの modern model で検証されている。

実践上の注意は、RoPE は Q と K に作用し、token embedding に単純に足すものではないということです。rotation frequency は通常 fixed schedule で、学習 parameter ではありません。

## 5. Attention の inference efficiency：MHA、MQA、GQA

Standard multi-head attention では、各 head が自分の Q、K、V を持ちます。訓練時は full batch と full sequence を一度に処理するため、大きな matmul があり GPU utilization は比較的良いです。しかし inference では autoregressive generation により token を一つずつ生成します。過去 token の K/V を再計算しないために、system は KV cache を保持します。

問題は、context が長くなるほど KV cache が大きくなることです。新しい token を一つ生成するたび、過去の K/V を大量に HBM から読む必要があります。このとき bottleneck は計算能力ではなく memory bandwidth になりがちです。

MQA、multi-query attention は大胆な簡略化です。複数の query head は保ちますが、すべての head が一組の K と V を共有します。これにより KV cache は大幅に減り、inference speed と long-context serving が改善します。

GQA、grouped-query attention は折衷案です。query head を複数 group に分け、各 group が一組の K/V を共有します。MHA より KV cache を節約し、MQA より表現力を保ちます。多くの modern large model は quality と inference cost のバランスのため GQA を採用します。

つまり attention head の設計は training だけでなく deployment の問題です。モデル公開後の大きな cost は inference から来ます。GQA/MQA の価値は主に inference の memory access を減らし throughput を上げる点にあります。

## 6. 重要 hyperparameters の経験則

### 6.1 MLP intermediate dimension

通常の ReLU/GELU MLP なら古典的な選択は次です。

```text
d_ff = 4 * d_model
```

SwiGLU/GeGLU など gated MLP では parameter 数を近づけるため、よく次を使います。

```text
d_ff ≈ 8/3 * d_model
```

Kaplan らの scaling law 系 ablation では、MLP ratio はかなり広い範囲で動きますが、4 倍付近は合理的な default です。T5 は一時 64 倍という非常に大きい `d_ff` を使ったことがあり、規則が絶対ではないことを示します。ただし T5 v1.1 はより標準的な GLU ratio に戻っており、通常の default が競争力を持つことも示しています。

### 6.2 Attention head dimension

よく使う設定は次です。

```text
d_model = n_heads * d_head
```

head 数を増やしても attention 全体の dimension を無制限に増やすのではなく、`d_model` を head に分割します。多くの GPT、PaLM、LLaMA 系 model はこの 1:1 に近い設定です。理論的には head dimension が小さすぎると low-rank bottleneck になり得ますが、実践上この default はよく機能します。

### 6.3 Width/depth ratio

Model capacity は幅を広げても深くしても増やせます。width は通常 `d_model`、depth は layer 数で制御します。多くのモデルは次の範囲に入ります。

```text
d_model / n_layers ≈ 100 to 128
```

これは法則ではありませんが、Kaplan らの実験では、複数の parameter scale で最適な width/depth の領域は大きくは変わりませんでした。

System factors も影響します。深いモデルは layer を device に分ける pipeline parallelism に合い、広いモデルは大きな matrix を GPU に分ける tensor parallelism に合います。つまり hyperparameter は loss だけでなく、cluster network、parallel strategy、memory limit にも制約されます。

### 6.4 Vocabulary size

初期の英語モデルでは 30k から 50k token の vocabulary がよく使われました。現代の production model、特に multilingual model では 100k から 250k、またはそれ以上の vocabulary がよく使われます。

大きい vocabulary の利点は次です。

- multilingual text がより少ない token に分割される;
- low-resource language の inference cost が下がる;
- emoji、code、special symbols の coverage が良くなる;
- large model は大きな vocabulary をより活用しやすい。

英語だけの小モデルなら小さい vocabulary でも可能です。汎用・多言語・production 向け model では大きな vocabulary がトレンドです。

## 7. Dropout、weight decay、training stability

Pretraining は従来の supervised learning と異なります。data が巨大で、通常は完全な multi-epoch 訓練をしないため、overfitting は主問題ではありません。このため dropout は modern LLM pretraining ではあまり使われなくなっています。

一方で weight decay はよく使われます。ここでの役割は従来の「overfitting を防ぐ regularization」だけではありません。実験では weight decay が learning rate schedule、特に cosine decay と複雑に相互作用することが観察されています。高 learning rate の段階では訓練が遅く見えても、learning rate が下がると weight decay ありのモデルが急に改善し、最終的に training loss と validation loss が良くなることがあります。

したがって LLM pretraining における weight decay は、単なる regularization というより optimization dynamics の道具として見るべきです。

## 8. 大規模訓練安定性：softmax が重要なリスク領域

モデルが大きく、訓練が長くなるほど loss spike や gradient norm spike が重要になります。modern architecture の改善では、softmax 周辺の安定化が明確な流れです。Transformer には二つの重要な softmax があります。

1. output layer の vocabulary softmax;
2. attention 内の softmax。

### 8.1 Output softmax の z-loss

Output softmax は次を計算します。

```text
p(x) = exp(logit_x) / Z
```

`Z` は vocabulary 全体の exponentiated logits の和です。Z が大きすぎたり不安定だったりすると、softmax は numerical problem を起こします。z-loss は補助項を加え、`log Z` を 0 付近、つまり normalizer を 1 付近に保つよう促します。

PaLM はこの技術を使い、後続モデルにも採用例があります。目的は表現力を上げることではなく、output softmax の numerical range を制御することです。

### 8.2 Attention softmax の QK norm

Attention softmax の入力は QK inner product です。query/key の norm が大きすぎると logits が極端になり、softmax が飽和し、gradient が不安定になります。QK norm は inner product の前に Q と K を normalization します。

これは softmax 入力の scale を直接制御する方法です。vision transformer や multimodal training stability で有用で、その後 text LLM にも取り込まれました。注目すべき現象は、normalization の位置が modern model で広がっていることです。block 前 norm、sublayer 後 norm、Q/K norm へと広がっており、activation scale の制御が大規模訓練の中心であることを示しています。

### 8.3 Logit soft capping

別の方法は attention logits に soft cap をかけることです。例えば `tanh` を使って過大な logits を滑らかに制限します。Gemma 2 などのモデルは類似技術を使いました。極端値を制御できますが、常に性能が上がるとは限りません。一部の実験では QK norm の方が安全な選択です。

## 9. Long-context attention：local window と sparse structure

Full self-attention の cost は sequence length の二乗で増えます。長い context を扱うため、モデルは structured attention を使うことがあります。

- sliding window attention：各 layer が近傍 window のみを見る;
- sparse attention：local と cross-block connection を設計する;
- periodic full attention：全 layer で global attention をせず、数 layer ごとに行う。

最近のモデルには hybrid structure があります。例えば 4 block ごとに 1 layer は position encoding なしの full attention を行い、他の layer は RoPE 付き sliding window attention を行う、といった設計です。利点は二つあります。

1. 多くの layer は local window だけを処理するため system cost を抑えられる。
2. 超長距離情報は position encoding なし full attention を通って伝わり、RoPE length extrapolation への負荷を減らせる。

このような設計は、long context capability が単に「RoPE を伸ばす」問題ではなく、attention pattern、position encoding、system cost の共同設計であることを示します。

## 10. 実用的な default configuration

標準的な dense decoder-only LLM を訓練するなら、次から始めるとよいでしょう。

- block：pre-norm Transformer;
- norm：RMSNorm;
- linear：default で bias なし;
- position：Q/K に RoPE;
- MLP：SwiGLU;
- MLP ratio：約 `8/3 * d_model`;
- attention：小規模訓練では MHA、inference deployment を意識するなら GQA を優先;
- head dimension：`d_model = n_heads * d_head` を満たす;
- width/depth ratio：`d_model / n_layers ≈ 100-128` を参考;
- dropout：大規模 pretraining では通常使わないか非常に小さい;
- weight decay：残し、learning rate schedule と合わせて調整する;
- stability：gradient norm と loss spike を監視し、z-loss、QK norm、追加 norm、logit soft cap を検討する。

## まとめ

この講義の核心は、modern LLM architecture は単一の突破ではなく、多くの経験的選択が収束した結果だということです。pre-norm と clean residual stream は深い network を訓練しやすくします。RMSNorm、bias なし、GQA は memory movement と inference cost への配慮を反映します。SwiGLU、RoPE、妥当な hyperparameter ratios は安定して有効な default performance を与えます。z-loss や QK norm は、大規模訓練でより目立つ numerical stability の問題を扱います。

一文だけ覚えるなら、Transformer 訓練は単に layer と parameters を積むことではなく、architecture、hyperparameters、optimization dynamics、hardware efficiency の間で協調した選択を行うことです。modern LLM の default recipe が重要なのは、多くの大規模訓練実験で検証され、高価な失敗を避ける助けになるからです。


---


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


---


# Stanford CS336 Lecture 5：GPU 日本語チュートリアル

> 適応注記：このチュートリアルは中国語版チュートリアルを日本語学習者向けに翻訳・適応したものです。元の構成と技術用語を保ちつつ、自然で教育的な日本語になるように説明を調整しています。

大規模言語モデルが現在の規模まで学習できるようになった中心的な理由の 1 つは、ハードウェアのスループットが継続的に向上したこと、特に GPU が普及したことです。この講義の目的は CUDA API を暗記することではありません。むしろ、性能に関する直感を作ることです。GPU は並列行列乗算が非常に得意ですが、しばしば「計算できない」ことではなく「データを運べない」ことがボトルネックになります。GPU アーキテクチャ、実行モデル、メモリ階層を理解すると、同じ行列乗算がある次元では非常に速く、別の次元では突然遅くなる理由や、FlashAttention のようなアルゴリズムがなぜ有効かを説明できます。

## 1. CPU と GPU の設計目標

CPU が最適化しているのは低レイテンシです。CPU は通常、複雑な制御ロジック、分岐予測、キャッシュ階層、高い単一スレッド性能を持ち、1 つのタスクをできるだけ早く終わらせることを目指します。GPU が最適化しているのは高スループットです。単一スレッドの柔軟性と低レイテンシを犠牲にし、チップ面積の多くを大量の算術ユニットに使うことで、何千もの似たタスクを同時に進めます。

これは深層学習にちょうど合っています。Transformer の学習では、多くの作業が行列乗算、elementwise 演算、reduction、テンソル変換です。これらは構造が規則的でデータ量が大きく、多数の似た小タスクに分解して並列実行できます。Dennard scaling と単一スレッド性能の伸びが鈍化するにつれ、深層学習のスケーリングはますます並列ハードウェアに依存するようになり、GPU はその代表になりました。

## 2. GPU の基本アーキテクチャ：SM、SP、Tensor Core

1 枚の GPU は、多数の Streaming Multiprocessor（SM）から構成されていると考えられます。SM は GPU 上の基本実行単位のようなものです。各 SM は独自のスケジューリング・制御ロジック、レジスタ、共有メモリ、多数の実行ユニットを持ちます。SM 内部にはさらに細かい処理ユニットが多数あり、異なるデータに対して同じ命令を実行できます。

現代の NVIDIA GPU には Tensor Core も含まれます。これは行列乗算に特化したハードウェアユニットです。V100 以降、行列乗算スループットと通常の浮動小数点演算スループットの間には大きな差が生まれました。ニューラルネットワークの実行時間の大半を行列乗算に乗せられるなら Tensor Core の恩恵を受けられます。一方で、行列乗算ではない複雑な操作を大量に設計すると、理論上の FLOPs が高くなくても実行は遅くなり得ます。

これが、LLM アーキテクチャが線形層、attention における QK^T と PV、MLP における大きな行列乗算を好む理由です。これらの操作は GPU のハードウェア能力とよく一致しています。

## 3. SIMT、thread、warp、block

GPU の実行モデルは通常 SIMT、すなわち Single Instruction, Multiple Threads と呼ばれます。同じグループのスレッドが同じ時刻に同じ命令を実行し、異なるデータを処理します。

CUDA プログラミングでは、よく次の 3 階層が登場します。

- thread：最小の論理実行単位。
- warp：通常 32 個の連続スレッドからなるグループで、同じ命令を一緒に実行する。
- block：スレッドのグループで、通常はある SM に割り当てられて実行される。

このモデルには重要な制約があります。同じ warp 内で深刻な分岐分岐を起こさない方がよい、ということです。たとえば 32 個のスレッドの半分が if 分岐、残り半分が else 分岐を通る場合、GPU は 2 つの経路を本当に同時実行することはできません。一部のスレッドを実行し、他を停止し、その後逆にします。これにより有効利用率が下がります。そのため高性能 GPU kernel は通常、規則的なデータアクセスと規則的な制御フローを目指します。

## 4. メモリ階層：性能最適化の主戦場

GPU の計算ユニットは非常に速い一方、メモリ速度には大きな差があります。速い順におおよそ次のように理解できます。

- register：各スレッド専用で最速。短命のスカラー値に向く。
- shared memory / L1：SM 内部にあり、低レイテンシ。スレッドブロック内で共有できる。
- L2 cache：チップ上にあるが単一 SM 内ではなく、より遅い。
- global memory / HBM：チップ外の高帯域メモリ。容量は大きいがレイテンシは高い。

shared memory へのアクセスは数十 cycle で済むことがありますが、global memory へのアクセスは数百 cycle かかることがあります。kernel が HBM から中間結果を何度も読み書きしていると、計算ユニットはデータ待ちになり、スループットは上がりにくくなります。

優れた GPU アルゴリズムの基本原則は、global memory へのアクセスをできるだけ減らすことです。一度データを SM に運んだら、shared memory や register の中で可能な限り多く計算し、最後に必要な結果だけを HBM に書き戻します。

## 5. Roofline モデル：計算ボトルネックかメモリボトルネックか

Roofline モデルは、プログラム性能が何に制限されているかを判断するために使われます。横軸は通常 arithmetic intensity、つまり 1 byte のデータを読むごとに何 FLOPs の計算を行えるか、と理解します。縦軸は実際のスループットです。

arithmetic intensity が低いと、プログラムは左側の memory-bound 領域にあります。算術ユニットが十分に供給されず、性能は主にメモリ帯域で決まります。ReLU、加算、LayerNorm の一部など、多くの elementwise 演算がこれに当たります。大量のデータを読み書きしますが、各要素では少量の計算しかしません。

arithmetic intensity が十分高いと、プログラムは compute-bound 領域に入ります。行列乗算が十分大きく、データ再利用が十分で、Tensor Core が十分に使われ、スループットはハードウェアピークに近づきます。

LLM 学習では、大きな行列乗算は compute-bound にしやすい一方、小さい batch、小さい行列、elementwise 演算、reduction、中間テンソルの頻繁な書き戻しは memory-bound になりやすいです。最適化の核心は、より多くの処理を roofline の右上へ押し上げることです。

## 6. なぜ行列乗算に tiling が必要か

素朴な行列乗算 C = A × B では、各 C[i,j] は A の 1 行と B の 1 列を読む必要があります。各スレッドが必要な要素を直接 global memory から読むと、大量の重複読み出しが発生します。同じ A の要素が複数の出力で再利用され、同じ B の要素も複数の出力で再利用されます。毎回 HBM から取るのは明らかに無駄です。

Tiling の考え方は、A、B、C を小さなブロックに切ることです。1 つの block が C の 1 つの tile を計算します。まず A と B の対応する tile を global memory から shared memory へ運び、その shared memory 内のデータを繰り返し使って partial sum を累積します。現在の tile を処理し終えたら、次の tile を読み込みます。

この方法には 2 つの利点があります。

1. global memory の読み出し回数が減る。tile サイズを T とすると、理想的には一部の global memory アクセスを約 T 倍削減できる。
2. アクセスパターンが規則的になる。tile を読み込むとき、連続スレッドが連続アドレスを読むようにでき、memory coalescing に有利になる。

ただし tile は大きければよいわけではありません。shared memory 容量、レジスタ数、warp スケジューリング、Tensor Core の形状、行列次元の割り切れ方に制約されます。行列次元が tile サイズ、warp サイズ、burst section の倍数にちょうどなっていると通常は速くなります。1 要素だけ余ると追加 tile が必要になり、多くの SM が「疎な端のブロック」を処理することになって、スループットが突然落ちることがあります。

## 7. Memory coalescing、padding、奇妙な性能変動

DRAM は通常、スカラー 1 個ずつを返すのではなく、連続したブロック単位で読みます。同じ warp のスレッドが隣接アドレスにアクセスすると、ハードウェアはそれらをより少ないメモリトランザクションにまとめられます。これを memory coalescing と呼びます。スレッドが散らばったアドレスにアクセスすると、複数回の読み出しが発生し、帯域利用率が下がります。

これは一見オカルトのような現象を多く説明します。行列を行方向にアクセスするか列方向にアクセスするかで性能が大きく変わることがあります。vocab size、hidden size、batch size が 8、16、32、64、128 の倍数かどうかもスループットに影響します。Karpathy は nanoGPT の vocab size を 64 の倍数に padding すると明確に速くなると述べたことがあります。本質的には、行列形状を GPU の tile、warp、メモリアライメントに合いやすくしているのです。

もう 1 つの現象は wave quantization です。A100 に 108 個の SM があるとします。ある行列乗算が 98 個の tile に切られるなら、1 波でほとんどの SM を動かせます。次元が少し増えて tile 数が 120 になると、最初の 108 個の tile が先に走り、残り 12 個の tile が小さな第 2 波として走ります。この後半では SM 利用率が低くなります。そのため行列が少し大きくなっただけで、性能が大きく落ちることがあります。

## 8. 低精度、fusion、recomputation

メモリ圧を下げるためによく使われる手法は 3 種類あります。

第一は低精度です。FP16、BF16、FP8、int8 は各要素のバイト数を減らし、同じ帯域でより多くのデータを運べるようにし、より高速な Tensor Core も使えるようにします。学習では通常 mixed precision を使います。入力と重みは 16 bit、乗算の累積は FP32 accumulator にして、速度と数値安定性を両立します。

第二は operator fusion です。コードがまず sin(x) を計算して HBM に書き戻し、次にそれを読み出して二乗し、また書き戻すなら、メモリ往復が非常に多くなります。fused kernel は複数の elementwise 演算を 1 回の kernel で完了し、中間値を register や shared memory に保持し、最後に結果だけを書き戻します。torch.compile、Triton、手書き CUDA kernel はこの最適化によく使われます。

第三は recomputation です。backpropagation には forward activation が必要です。素朴な方法ではすべての activation を HBM に保存し、backward 時に読み戻します。しかし一部の activation が計算は安く、読み出しが高価なら、backward 時に再計算できます。追加 FLOPs と引き換えにメモリ読み書きを減らします。これはメモリ節約だけでなく、memory-bound な場面では高速化にもなります。

## 9. FlashAttention：これらの考えを組み合わせる

標準的な attention は QK^T、softmax、そして V との乗算を含みます。問題は attention matrix のサイズが n × n であることです。完全な score と softmax 結果を HBM に materialize すると、長い文脈ではメモリ読み書きが非常に高価になります。

FlashAttention の鍵は attention の数学的計算量を減らすことではなく、HBM アクセスを減らすことです。FlashAttention は tiling を使います。Q、K、V をブロックに分けて SRAM/shared memory に運び、ブロック内で QK^T と後続の累積を計算します。難しい点は softmax が行単位の大域操作であり、行全体の最大値と正規化分母を知る必要があることです。FlashAttention は online softmax を使います。ブロックごとに各行の running max と正規化和を維持し、tile が来るたびにこれらの統計量を更新するため、完全な n × n 行列を GPU メモリへ書き戻す必要がありません。

backward でも、FlashAttention は softmax 関連量を recomputation し、n × n の中間 activation を保存しません。結果として、この講義の複数の核心技術、tiling、shared memory 再利用、operator fusion、online softmax、recomputation を組み合わせています。得られるのは厳密な attention ですが、HBM アクセスは大きく減り、長系列 Transformer の学習と推論が速くなります。

## 10. まとめ：LLM 学習はなぜ GPU に依存するのか

LLM 学習が GPU に依存するのは、単に GPU の FLOPs が高いからではありません。Transformer の主要計算形式、大規模行列乗算、規則的テンソル操作、batch 化可能な並列データフローが、GPU に自然に合っているからです。Tensor Core は行列乗算を「ハードウェアに祝福された」操作にし、mixed precision はさらにスループットを増幅します。

しかし現代 GPU の真のボトルネックは、純粋な計算よりもメモリ移動に由来することが増えています。高性能実装では次を意識する必要があります。warp は連続メモリにアクセスしているか。不必要な HBM 読み書きを避けているか。tiling でデータ再利用を高めているか。行列次元は揃っているか。fusion できるか。recomputation でメモリを計算に交換できるか。tile 数と SM 数はうまく噛み合っているか。

したがって GPU を理解する核心は、特定モデルのスペックを覚えることではありません。性能ボトルネックを判断する思考を身につけることです。計算は十分密か。データは HBM から何度も運ばれていないか。warp は分岐していないか。アクセスは合成されているか。tile は揃っているか。これらの細部が合わさって、LLM 学習が高価な GPU を本当に使い切れるかを決めます。

## 11. 実践チェックリスト：遅いコードを見たとき最初に問うこと

LLM の学習や推論で GPU 利用率が低いときは、次の順に確認できます。第一に、CPU が足を引っ張っていないか確認します。データ読み込み、tokenization、ログ出力、頻繁な `.item()` は GPU を待たせます。第二に、kernel が細かすぎないか確認します。Transformer block 内で非常に短い小 kernel が大量に出ているなら、多くの elementwise 演算が融合されておらず、kernel launch と HBM 往復が時間を食っています。第三に、行列形状が Tensor Core に適しているか見ます。hidden size、vocab size、batch×sequence がハードウェアの好む倍数に揃っているかです。第四に、メモリ使用量が batch を小さくしすぎていないか確認します。小さい batch は行列乗算の arithmetic intensity を下げ、並列度も不足させます。第五に、`nvidia-smi` の利用率だけでなく profiler でボトルネックを確認します。`nvidia-smi` は粗い信号を与えるだけで、どの kernel が遅いかは教えてくれません。

## 12. 一貫した例：「同じモデル」でも速度が大きく違う理由

2 つの実装が同じ Transformer を学習しているとします。パラメータ数、batch size、dtype は完全に同じです。実装 A は、多数の小さな PyTorch 操作で attention、MLP、residual、normalization を直接組み合わせています。実装 B は fused layer norm、FlashAttention、fused optimizer を使い、vocab と hidden size をより扱いやすい次元へ padding しています。数学的にはほぼ等価ですが、実装 B は大量の中間テンソルの書き戻しを減らし、CPU/GPU 同期を減らし、行列乗算を安定して効率的な tile に乗せます。結果は 5% の改善ではなく、数十% 以上の改善になることもあります。

これがこの講義で作りたい工学的直感です。モデルアーキテクチャ論文の 1 行の式は、GPU 上では多数の kernel、多数のメモリトランザクション、多数のスケジューリング判断に変わります。LLM Infra ではモデルを知るだけでは不十分で、式をハードウェアコストへ翻訳できる必要があります。どのテンソルが materialize されるのか。どの中間値は再計算できるのか。どの操作は融合すべきか。どの次元が余分な tile の波を発生させるのか。この直感を身につけると、後続の Triton、自作 kernel、分散学習、推論サービスが本当につながって見えるようになります。

最後に最も実用的な判断基準を挙げます。最適化が HBM 読み書きを減らし、Tensor Core 利用率を高め、同期回数を下げ、batch/sequence の有効並列度を上げるなら、通常は試す価値があります。単にコードを低レベルに書き換えるだけでデータ移動経路を変えていないなら、利益は限られることが多いです。


---


# Stanford CS336 2025 第6回チュートリアル：Kernel、Triton、LLM 演算子最適化

> 適応注記：このチュートリアルは中国語版チュートリアルを日本語学習者向けに翻訳・適応したものです。元の構成、コード片、数式、主要な技術用語を保ちつつ、自然で教育的な日本語になるように説明を調整しています。

この講義では、大規模モデル学習の低レベル性能の世界に入ります。GPU 上の kernel をどう理解するか、CUDA/Triton で custom operator をどう書くか、そして FlashAttention がなぜ大きな高速化をもたらすのかを扱います。中心的な考えは単純です。GPU は計算が得意ですが、GPU メモリからデータを運ぶのは高価です。高性能コードでは、データを計算ユニットに近い場所で何度も使い、意味のない読み書きと kernel 起動コストを減らす必要があります。

## 1. GPU 実行モデル：SM、block から warp へ

A100/H100 GPU は多数の SM（Streaming Multiprocessor）から構成されます。各 SM には計算ユニット、レジスタ、共有メモリ、キャッシュがあります。GPU に投入される基本単位は kernel と呼ばれます。1 つの kernel は多数の thread を起動し、それらの thread は thread block に組織されます。複数の block が grid を構成します。

三層構造として理解できます。

```text
grid = 多数の thread block
thread block = ある SM にスケジュールされて実行される thread のグループ
thread = 実際に命令を実行し、要素を処理する最小単位
```

同じ block 内の thread は shared memory を通じて高速に通信・同期できます。異なる block 間の通信は高価で、通常 1 つの kernel 内では同期できません。そのため kernel を設計するときは、共有が必要なデータをできるだけ同じ block/SM 内で処理できるようにします。

GPU はさらに thread を 32 個ずつ warp としてまとめます。1 つの warp 内の thread は SIMD/SIMT 風に一緒に実行されます。これにより制御ロジックを減らし、より多くのチップ面積を計算に使えます。代償として、warp 内の thread が異なる分岐を通ったり、仕事量が不均一だったりすると効率が下がります。kernel を書くときは通常、全 SM を埋めるだけの block 数を用意し、各 warp の仕事をできるだけ均一にします。

## 2. 性能ボトルネック：計算か、データ移動か

ある演算子が速いかどうかを判断するとき、FLOPs だけを見てはいけません。arithmetic intensity、つまり 1 byte のデータ移動あたりどれだけ計算を行うかを見る必要があります。

```text
算術強度 = FLOPs / メモリ移動バイト数
```

行列乗算はうまく実装されていれば、データブロックを繰り返し使えるため算術強度が高く、通常は compute-bound です。多くの elementwise 演算、softmax、正規化、単純な活性化関数はしばしば memory-bound です。各要素では少量の計算しかしないのに、HBM から読み出して書き戻す必要があるからです。

LLM 学習では特に 2 種類の最適化が重要です。

1. 行列乗算を cuBLAS/CUTLASS/Tensor Core などの高性能ライブラリやハードウェア経路に乗せる。
2. memory-bound な演算子に対して fusion、tiling、中間結果の書き戻し削減を行う。

## 3. Benchmark と profiling

この講義では繰り返し、感覚で最適化してはいけない、と強調します。benchmark は end-to-end 実行時間を教え、profiling はどの kernel に時間が使われているかを教えます。

GPU benchmark にはよくある落とし穴が 2 つあります。

第一に、warm up が必要です。PyTorch/CUDA コードの初回実行では kernel コンパイル、ライブラリ読み込み、キャッシュ初期化などのコストが発生することがあります。本当に知りたいのは定常状態の速度です。

第二に、明示的な同期が必要です。CPU は CUDA kernel を投入した後、通常 GPU の完了を待たずに先へ進みます。そのため Python の時間関数で GPU 操作を直接囲むと、「タスクを投入する時間」だけを測ってしまうことがあります。正しい方法は、計測の前後で次を呼ぶことです。

```python
torch.cuda.synchronize()
```

Profiler はさらに低レベルのイベントを見せます。たとえば Python の `a + b` の裏側には、PyTorch C++ インターフェース、CUDA kernel launch、実際の elementwise kernel があります。行列乗算も固定実装ではありません。shape、dtype、ハードウェアに応じて異なる cuBLAS/CUTLASS kernel が選ばれます。Nsight Systems は CPU thread と GPU stream のタイムラインも描けます。CPU は通常、先に kernel をキューへ投入し、GPU は後ろでキュー順に実行します。

これにより、Python の学習コードが必ず遅いわけではない理由が分かります。CPU が十分速くタスクを投入できるなら、ボトルネックは依然として GPU にあります。逆に、頻繁な `print(loss)`、`.item()`、tensor の CPU への移動は強制同期を起こし、CPU/GPU パイプラインを切断します。

## 4. Kernel fusion：読み書き 1 回の削減が大きな利益になる

GELU の近似式を計算するとします。

```text
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

通常の PyTorch 式で手書きすると、複数の乗算、加算、`tanh`、累乗に分解されるかもしれません。各ステップは 1 つの kernel です。HBM から `x` を読み、計算し、中間結果を書き、次のステップでその中間結果を読み、また書きます。各操作自体は単純でも、全体はメモリ読み書きと kernel launch によって遅くなります。

Kernel fusion の目標は、これらの操作を 1 つの kernel にまとめることです。各要素を HBM から 1 回だけ読み、すべての計算を register 内で行い、最後に 1 回だけ書き戻します。講義の例では、素朴に手書きした GELU は PyTorch 組み込みの fused GELU よりかなり遅くなります。custom CUDA/Triton kernel は複数 kernel を 1 つに圧縮し、組み込み演算子に近い速度を出せます。

この考え方は LLM にとって重要です。Transformer には大きな行列乗算以外にも、bias add、activation、dropout、residual add、layer norm、softmax、mask などの小さな操作が多数あります。それぞれが個別に HBM を読み書きすると、メモリ帯域がボトルネックになります。現代のフレームワークは一部の fusion を自動で行いますが、複雑な構造では手書き kernel が必要になることがあります。

## 5. CUDA kernel の基本的な書き方

CUDA は GPU を直接プログラムする C++ インターフェースです。elementwise GELU kernel を書くときは、通常 2 つの部分に分かれます。

第一は CPU 側 wrapper です。入力が CUDA 上にあるか、contiguous かを確認し、`torch.empty_like(x)` で出力を確保し、block size と block 数を計算し、最後に kernel を起動します。

第二は GPU 側 kernel です。各 thread が自分の位置に基づいて global index を計算します。

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    out[i] = gelu_formula(in[i]);
}
```

覚えておくべき工学的細部がいくつかあります。

1. 出力はすぐ上書きされるため、先に確保してゼロクリアするより `empty_like` がよい。
2. block 数は切り上げ、末尾の要素も処理されるようにする。
3. 最後の block は範囲外に出る可能性があるため、`i < n` のチェックが必須。
4. custom kernel は入力が contiguous であると仮定することが多い。そうでないと index logic が複雑になる。transpose/view は非連続 tensor を作ることがあるため、外側で `.contiguous()` が必要になる場合があるが、これはコピーコストを生む。
5. CUDA のデバッグでは `CUDA_LAUNCH_BLOCKING=1` を設定するとエラー位置を見つけやすくなるが、性能には影響する。

CUDA の利点は制御力が強いことです。欠点は boilerplate が多く、thread、block、shared memory、同期、境界条件を手で管理する必要があることです。

## 6. Triton：block 中心の GPU プログラミング

Triton は OpenAI が開発した GPU プログラミング DSL です。Python の中で kernel を書けますが、抽象度は CUDA より高いです。CUDA では通常「各 thread が何をするか」を考えますが、Triton では「各 program/block がどのデータブロックを処理するか」を考えるよう促されます。

Triton の elementwise kernel はおおよそ次のようになります。

```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n
x = tl.load(x_ptr + offsets, mask=mask)
y = gelu_formula(x)
tl.store(y_ptr + offsets, y, mask=mask)
```

`tl.arange` はベクトル化された offset を生成するため、1 つの Triton program が 1 つの block 全体を一度に処理します。Triton コンパイラはこれをより低レベルの GPU 命令へ落とし、memory coalescing、レジスタ使用、一部の shared memory 管理など、多くの面倒な細部を扱います。

Memory coalescing は GPU 性能の鍵です。GPU が HBM からデータを取るとき、連続アドレスアクセスを好みます。隣接 thread が隣接要素を読むと、ハードウェアはそれらを効率的なメモリトランザクションにまとめられます。Triton の vectorized load/store はこのパターンを自然にします。生成された PTX を見ると、コンパイラが連続値をまとめて register に読み込み、乗算、指数、tanh などの操作を行い、最後にまとめて書き戻すことが分かります。

Triton の価値は折衷にあります。PyTorch 式よりハードウェアに近く、CUDA より書きやすくデバッグしやすい。新しいモデルの特殊な演算子では、Triton はしばしば最も実用的な custom kernel ツールです。

## 7. Tiling：近くにあるデータを何度も使う

Tiling は高性能 GPU 演算子の中核パターンです。大きなテンソルを小さな tile に切り、1 つの block/SM が 1 つの tile を担当します。tile を register や shared memory に読み込み、局所的にできるだけ多く計算し、その後 global memory に書き戻します。

行列乗算の高性能実装は tiling に大きく依存します。各出力要素が独立に HBM から行全体と列全体を読むのではなく、A と B の小ブロックを shared memory に読み込み、複数 thread が協調してこれらを再利用します。同じデータが複数回の multiply-add に参加し、算術強度が高まります。

Softmax も典型例です。1 行が 1 block に収まるなら、各 block が 1 行全体を処理できます。まず 1 行を読み、数値安定性のため最大値を引き、exp を計算し、行内 sum reduction を行い、sum で割って正規化し、最後に書き戻します。中間結果を何度も HBM に落とす必要はありません。

```text
1 行の softmax：
load row -> max -> exp(row - max) -> sum -> normalize -> store row
```

sequence が長い、または行列が大きい場合、1 つの tile には全データが入りません。その場合は分割 reduction や tile 間での統計量の合成が必要になり、複雑度が上がります。これが FlashAttention の中心的動機の 1 つです。

## 8. FlashAttention：なぜ custom kernel が必要か

標準的な attention 計算は次です。

```text
scores = QK^T / sqrt(d)
probs = softmax(scores)
out = probs V
```

素朴な実装では `scores` と `probs` を明示的に構築します。形状は `[batch, heads, seq, seq]` です。seq が長いと、この中間行列は非常に大きくなり、HBM の読み書きコストが非常に高くなります。attention の数学自体は変わりませんが、実装方式が大量のメモリ帯域を浪費しています。

FlashAttention の鍵は IO-aware であることです。Q、K、V を tile に分け、現在ブロックの softmax 統計量と出力累積だけを SRAM/register に保持し、完全な attention matrix を HBM に書きません。online softmax を使い、K/V を走査しながら各行の最大値、正規化分母、出力累積を維持することで、標準 attention と等価な結果を得ます。

この種の最適化は、単純な fusion では自動発見が難しいです。計算スケジュールと中間状態の置き場所を変えているからです。FlashAttention 2/3 はさらに、より良い並列分割、Tensor Core、H100 の新機能などのハードウェア特性を活用します。したがって、複雑な reduction、データ再利用、特殊なハードウェア経路を持つ演算子では、手書きの Triton/CUDA kernel には依然として価値があります。

## 9. torch.compile：自動最適化

この講義では、すべてを CUDA kernel として手書きすべきではない、とも注意しています。PyTorch の `torch.compile` は、単純な kernel fusion、shape-specialized な最適化、行列乗算に対するより適切な低レベル kernel の選択など、多くの最適化をすでに自動で行えます。例では、手書き GELU に `torch.compile` をかけると fused Triton kernel が生成され、授業で手書きした Triton 版に近い、あるいはそれを上回る性能になることがあります。

実践上のすすめは次の通りです。

1. まず明確で正しい PyTorch 版を書く。
2. benchmark と profiler で本当のボトルネックを見つける。
3. まず `torch.compile`、公式 fused op、xFormers/FlashAttention などの成熟した実装を試す。
4. それでも特殊な演算子の時間占有が大きく、メモリ読み書きが多く、自動コンパイラで扱えないなら、Triton/CUDA を検討する。

## 10. LLM 演算子最適化の原則

大規模モデルの性能最適化は、「Python を C++ に変える」ほど単純ではありません。GPU メモリ階層を中心に計算を再構成することです。

- 高性能行列乗算ライブラリをできるだけ使い、Tensor Core を働かせる。
- memory-bound な操作を fusion し、中間 tensor の書き戻しを減らす。
- tiling でデータ再利用を高め、ホットデータを register/shared memory に残す。
- 連続アクセスと memory coalescing を保証し、ばらばらな読み書きを避ける。
- 頻繁な `.item()`、`print`、CPU へのコピーなど、不必要な CPU/GPU 同期を避ける。
- 直感で推測せず、profiler で各最適化を検証する。

この講義の中心的 takeaway は、LLM の速度は演算子の実装方法に大きく依存する、ということです。数学は同じでも、性能は 1 桁違うことがあります。Triton は研究者に実用的な入口を提供します。PyTorch 式では遅く、CUDA は煩雑すぎるとき、Python に近い形で低レベル性能に近い custom kernel を書けます。

## 11. いつ custom kernel を書く価値があるか

遅いコードすべてを手書き kernel にすべきではありません。実用的な判断基準は、profiler である操作の時間占有が大きく、かつそれが標準的な大きな行列乗算ではない場合に、深掘りする価値がある、というものです。標準行列乗算は通常、cuBLAS、CUTLASS、FlashAttention などの成熟したライブラリで十分よく処理されています。自分で書き直すとかえって遅くなりがちです。custom に向く場面は、複数の elementwise 操作が同じ大きなテンソルを何度も読み書きする場合、特殊な mask や特殊な reduction が必要な演算子、大きな中間テンソルを実は計算しながら捨てられる場合、新しいモデル構造で framework に fused 実装がない場合などです。

書く前には利益の上限も見積もるべきです。ある kernel が総学習時間の 1% しか占めないなら、10 倍速くしても end-to-end では 1% 未満の改善です。長文脈で attention が 30% を占め、profiler が大量の HBM 読み書きを示しているなら、FlashAttention のような IO-aware な並べ替えは全体速度を大きく変える可能性があります。最適化は最も面白い低レベルコードからではなく、end-to-end のボトルネックから始めるべきです。

## 12. PyTorch から Triton への推奨ワークフロー

実践では 4 段階の方法が使えます。第一に、最も明確な PyTorch reference implementation を書き、小さな tensor で correctness test を行います。第二に、`torch.compile` または既存の fused op で強い baseline を得つつ、profiler で hotspot を探します。第三に、まだ custom kernel が必要なら、まず Triton 版を書きます。Triton は block レベルのロジックを表現しやすく、Python のテストフレームワークとも組み合わせやすいからです。第四に、shared memory、warp-level primitive、Tensor Core 命令、多段 pipeline などをより細かく制御する必要がある場合に限り、CUDA/CUTLASS へ下ります。

kernel を変更するたびに、3 つを同時に検証する必要があります。数値が一致するか、複数 shape で速度が本当に速いか、メモリ読み書きが本当に減っているかです。多くの kernel は 1 つの shape では速くても、batch size、sequence length、head dimension が変わると退化します。また一部の最適化は数値安定性を犠牲にし、長系列や低精度で初めて問題が現れます。したがって LLM 演算子最適化は単発の小技ではなく、correctness、benchmark、hardware constraints の 3 つを一緒に反復する作業です。

## 13. 後続講義との接続

この講義は、後の分散学習と推論システムの土台です。単一 GPU の kernel 最適化は「1 つの GPU 内部でどうデータ移動を減らし、有効計算を増やすか」を解きます。並列学習は「複数 GPU の間でパラメータ、activation、batch をどう切り分け、通信をできるだけ減らすか」を解きます。推論システムは「オンラインリクエストをどう batching し、KV cache をどうスケジューリングし、latency と throughput の間でどう折り合うか」を解きます。三者は同じ思考を使います。まず希少資源を見つけ、その後計算とデータフローを組み替えます。希少資源は HBM 帯域、SM 算力、メモリ容量、PCIe/NVLink 帯域かもしれませんし、ユーザーリクエストのレイテンシ予算かもしれません。kernel を理解すると、FlashAttention、FSDP、tensor parallel、PagedAttention がいずれもデータ移動を中心にした技術であることが見えやすくなります。


---


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


---


# Stanford CS336 第8回チュートリアル：並列学習（二）

> 翻案メモ: この日本語チュートリアルは中国語版講義ノートをもとに、Markdown構造、技術用語、数式、コード風の表記を保ちながら、日本語で自然に読める教材として整えたものです。

この講義では、大規模モデル学習のシステム面の問題をさらに扱います。単一 GPU がモデル、optimizer state、十分大きな batch を保持できなくなったとき、計算とデータを複数 GPU、さらには複数ノードへどう分散し、通信がボトルネックになるのをどう避けるか、という問題です。基本原則は単一 GPU 最適化と同じです。高価な計算ユニットをできるだけ忙しく保ち、算術強度を高め、不要なデータ移動を減らします。

## 1. 多 GPU 学習のハードウェア視点

学習クラスタは、階層的なストレージおよび通信システムとして理解できます。

- GPU 内部：SM が計算を実行する。L1/shared memory は非常に速いが小さく、HBM は容量が大きいが遅い。
- 単一ノード内の複数 GPU：GPU 間は PCIe または NVLink/NVSwitch で接続される。NVLink は NVIDIA が GPU 間高帯域通信のために設計した専用リンクで、通常 PCIe よりはるかに速い。
- 複数ノード：ノード間通信は NIC、スイッチなどのネットワーク機器を経由し、帯域は低く、レイテンシは高くなる。

したがって、GPU 間通信はローカル HBM アクセスより高価であり、ノード間通信は同一ノード内通信よりさらに高価です。分散学習エンジニアリングの目標は「通信を完全に避ける」ことではなく、次を実現することです。

1. 通信量を減らす。
2. 通信を適切な場所に配置する。
3. できるだけ計算と重ね合わせる。
4. ハードウェアトポロジに応じて並列戦略を選ぶ。

NVIDIA エコシステムでは、低レベル通信は通常 NCCL が担当します。NCCL は GPU トポロジに応じて通信経路を選び、高レベルの collective 操作を低レベルの CUDA kernel とデータ転送へ変換します。PyTorch の `torch.distributed` は、Python 層で `all_reduce`、`all_gather`、`reduce_scatter` などの便利なインターフェースを提供します。

## 2. Collective 通信プリミティブ

分散学習では collective operations が頻繁に使われます。全部で `world_size` 個のプロセスまたはデバイスがあり、各デバイスには `rank` という番号が付いているとします。

よく使われるプリミティブは次の通りです。

- `broadcast`：1 つの rank 上のデータをすべての rank にコピーする。
- `scatter`：1 つの rank 上の異なるスライスを異なる rank に配る。
- `gather`：複数 rank のデータを 1 つの rank に集める。
- `reduce`：複数 rank のデータを和、平均、最大値などで集約し、1 つの rank に置く。
- `all_gather`：各 rank が全 rank のデータを連結した結果を得る。
- `reduce_scatter`：各 rank の入力をまず reduce し、reduce 後の異なるスライスを異なる rank に配る。
- `all_reduce`：各 rank が全 rank のデータを reduce した完全な結果を得る。

重要な等価関係は次です。

```text
all_reduce = reduce_scatter + all_gather
```

たとえばデータ並列学習では、各 GPU が異なるデータスライス上で勾配を計算し、`all_reduce` で平均して、すべての GPU のパラメータ更新を一致させます。多くの高度な戦略では、ピークメモリを減らしたり計算との重ね合わせをよくしたりするため、`all_reduce` を `reduce_scatter` と `all_gather` に分解します。

これらのプリミティブを使うときは、同期関係に特に注意する必要があります。collective 操作では、同じ process group に参加する rank が同じ順序で呼び出す必要があります。ある rank が `all_reduce` を 1 回呼び忘れると、他の rank は永久に待つ可能性があり、プログラムは hang したように見えます。

## 3. Benchmark：理論帯域だけを見ない

H100 の NVLink は理論帯域が非常に高いですが、実際の学習でどれだけ出るかは、テンソルサイズ、rank 数、通信パターン、NCCL アルゴリズム、ノードトポロジ、ノード間通信の有無に依存します。講義では大きなテンソルで `all_reduce` を測定し、実効帯域がハードウェア公称値より明らかに低いことを示します。`reduce_scatter` の性能も単純な見積もりと完全には一致しない場合があります。

エンジニアリング上は次の習慣を持つべきです。

- 製品仕様だけを読むのではなく、対象クラスタで実際に benchmark する。
- 同一ノード内通信とノード間通信を別々に測る。
- メッセージサイズごとのスループットとレイテンシを見る。
- warmup、`torch.cuda.synchronize()`、barrier などで計時誤差を避ける。
- 「アルゴリズムが転送すべきバイト数」と「壁時計時間」を区別する。

通信性能は数式だけでは正確に予測しにくいです。NCCL は ring、tree、階層的通信、ネットワーク内 reduce などの実装詳細を使うからです。したがって実際の調整には profiling と benchmark が不可欠です。

## 4. データ並列 DDP：batch を分割し、勾配を同期する

データ並列は最も直感的な並列化です。各 GPU が完全なモデルを保持し、異なる batch スライスを処理します。各 rank は独立に forward と backward を実行し、ローカル勾配を得たあと、すべてのパラメータ勾配を `all_reduce` で平均します。

学習ステップは次のようにまとめられます。

1. global batch を rank ごとの local batch に分割する。
2. 各 rank が同じモデルパラメータで自分の local batch を処理する。
3. backward でローカル勾配を得る。
4. 各パラメータの勾配に `all_reduce(mean)` を実行する。
5. 各 rank が同じ勾配で optimizer step を実行する。

データが異なるため各 rank の loss は異なるかもしれませんが、勾配同期後はパラメータが一致したままになります。

DDP の利点は実装が簡単で、計算スケーリングがよいことです。欠点は、各 GPU が完全なパラメータ、勾配、optimizer state を保存しなければならないことです。モデルが大きくなると、まずメモリがボトルネックになります。もう 1 つの工学的注意点は、`all_reduce` 自体が同期点であることです。遅い rank があると、他の rank は待たされます。これが straggler 問題です。

## 5. ZeRO と FSDP：パラメータ、勾配、optimizer state を分割する

DDP のメモリ制限を超えるため、モデル状態を分割できます。ZeRO（Zero Redundancy Optimizer）は学習状態をいくつかの段階に分けます。

- ZeRO-1：Adam の一次・二次モーメントなどの optimizer states を分割する。
- ZeRO-2：さらに gradients を分割する。
- ZeRO-3：parameters も分割する。

FSDP（Fully Sharded Data Parallel）は、PyTorch における ZeRO-3 風の実装と見なせます。各 rank は常駐状態としてパラメータの一部だけを持ち、ある層を計算するときに `all_gather` でその層の完全なパラメータを一時的に集めます。backpropagation 後には `reduce_scatter` で勾配を reduce し、再び shard に分けます。

FSDP の基本的なトレードオフは次の通りです。

- 利点：各 GPU の常駐メモリを大きく下げ、より大きなモデルを学習できる。
- 代償：forward/backward 中にパラメータの `all_gather` と勾配の `reduce_scatter` が頻繁に必要になる。
- 工学的重点：wrapping 粒度、prefetch、bucket size を適切に設定し、通信の断片化を避ける。

細かく分割しすぎると、層ごとの通信コストとスケジューリングコストが増えます。粗すぎると、ピークメモリが増えます。実際の学習では Transformer block を FSDP 単位にすることが多く、mixed precision、activation checkpointing、CPU/offload 戦略と組み合わせます。

## 6. Activation checkpointing：再計算でメモリを節約する

backpropagation には forward 中に保存した activations が必要です。長い系列、大きな batch、深い Transformer では activation メモリが非常に大きくなります。Activation checkpointing は、一部の中間結果だけを保存し、backward 時に不足した activations を再計算する方法です。

これは典型的な「計算とストレージの交換」です。

- checkpoint しない：計算は少ないが、多くの activation を保存する。
- block 全体または一部を checkpoint：メモリは少ないが、backward で追加の forward 再計算が必要。

実践では、すべてを盲目的に再計算すべきではありません。適切な粒度を選びます。通常は Transformer block レベルで checkpoint できます。matmul の直後に続く単純な pointwise 操作では、すべての中間値を保存する必要はないかもしれません。再計算コストが低いからです。Checkpointing は FSDP/ZeRO とよく組み合わせられます。パラメータ、勾配、optimizer state を分割したあと、activation が新しいメモリボトルネックになることがあるためです。

## 7. Tensor Parallel：隠れ次元を分割し、頻繁に collective する

Tensor parallelism は batch ではなく、モデル内部の行列次元を分割します。MLP の線形層を例にすると、重み行列を列方向または行方向に複数 rank へ分割できます。各 rank は重みの一部だけを保存し、出力の一部を計算します。

講義の単純化した例では、各 rank が各層の hidden dimension の一部を持ちます。ローカル activation を計算したあと、次の層へ入る前に `all_gather` で全 rank の activation を連結し、完全な hidden vector に戻す必要があります。

Tensor Parallel の特徴：

- 利点：単一層の非常に大きな行列を複数 GPU に分散できる。
- 欠点：各層または数個の演算ごとに collective が必要になる可能性がある。
- 適したハードウェア：高速インターコネクトに強く依存し、通常は同一ノード内の NVLink/NVSwitch を優先する。

Transformer でより一般的なのは、attention heads、MLP intermediate dimension、vocabulary projection の分割です。分割方法ごとに必要な collective は異なります。forward で `all_reduce` が必要な場合もあれば backward で必要な場合もあり、`reduce_scatter` + `all_gather` でメモリと通信を最適化できる場合もあります。

## 8. Pipeline Parallel：層を分割し、pipeline bubbles を扱う

Pipeline parallelism はモデルを「深さ」方向に分割します。rank 0 が前半の層、rank 1 が後半の層、というように保持します。forward では前段が activation を後段へ送り、backward では勾配が逆方向に戻ります。

素朴な pipeline の問題は bubble です。完全な batch を 1 つずつ送ると、後段が計算している間に前段が空き、後段は入力待ちで空くこともあります。解決策は batch を複数の microbatch に分け、異なる microbatch が異なる stage を同時に流れるようにすることです。

bubble の直感的な規則は次です。

- pipeline stage が多いほど、充填と排出のオーバーヘッドが大きい。
- microbatch 数が多いほど、bubble の比率は小さい。
- ただし microbatch が多すぎるとスケジューリングコストが増え、batch norm や optimizer などの挙動に影響することがある。

実システムでは forward/backward スケジュールも設計する必要があります。たとえば GPipe の「全 forward 後に全 backward」や、1F1B（一前進一後退）スケジュールがあります。待ち時間を減らすには非同期 `isend/irecv` を使い、通信を後続 microbatch の計算と重ねます。同期 send/recv だけだと GPU が頻繁にブロックされます。

## 9. 組み合わせ並列とエンジニアリング調整

大規模モデル学習では、通常 1 種類の並列化だけでなく複数を組み合わせます。

- データ並列：より多くのノードへ広げてスループットを拡張する。
- FSDP/ZeRO：モデル状態メモリを下げる。
- Tensor Parallel：単一層が広すぎる、1 枚に入らない、または計算が大きすぎる場合を扱う。
- Pipeline Parallel：モデルが深すぎる場合に層方向で分割する。
- Activation checkpointing：activation メモリを下げる。
- Sequence/context parallel：長コンテキスト学習で系列次元を分割する。

よくある経験則は、最も頻繁で細粒度な tensor parallel 通信を同一ノード内の高速インターコネクトに置き、より粗粒度なデータ並列をノード間に広げ、必要なら pipeline parallel を加えることです。戦略を選ぶときは、メモリ、計算利用率、通信帯域、レイテンシ、コード複雑性を同時に見ます。

エンジニアリング調整のチェックリスト：

1. まずボトルネックを確認する：profiler で compute-bound、memory-bound、communication-bound のどれか判断する。
2. batch と microbatch を調整する：microbatch を大きくすると行列積の利用率は上がるが、activation メモリは増える。
3. FSDP bucket/prefetch を調整する：`all_gather` と `reduce_scatter` をできるだけ計算と重ねる。
4. 小さな collective を増やしすぎない：小メッセージはレイテンシ支配で、bucket をまとめるほうが速いことが多い。
5. トポロジに合わせる：同一ノードでは tensor parallel を使い、ノード間では高頻度通信を減らす。
6. 同期点を確認する：barrier、loss logging、checkpoint 保存は待ち時間を生む可能性がある。
7. 決定性と一貫性を保つ：すべての rank は同じ順序で collective に入る必要がある。乱数シード、データ分割、dropout も正しく扱う。
8. 完全な checkpoint を定期的に保存する：sharded training では checkpoint が複数 rank に分散する可能性があるため、保存・復元形式を明確にする。

## 10. 実用的な選択フロー

ゼロからモデルの並列化案を選ぶなら、次の順に考えます。第一に単一 GPU メモリを見積もります。parameters、gradients、optimizer states、activations がそれぞれどれだけ使うかを確認します。完全なモデルと optimizer state が入らないなら、FSDP または ZeRO を優先します。主な問題が長系列や大きな microbatch による activation なら、activation checkpointing を先に有効にします。第二に、単一層が大きすぎるかを見ます。attention heads、MLP intermediate layer、vocabulary projection が 1 枚の GPU の計算やメモリで厳しいなら tensor parallel が必要で、同じ tensor parallel group はできるだけ同一マシンの高速インターコネクト内に置きます。第三に、モデルの深さを見ます。層数が多く、FSDP と tensor parallel だけでは足りないなら、pipeline parallel で層分割します。第四に、複数のモデル複製をより多くのノードに置く data parallelism で総スループットを上げます。

調整時は tokens/sec だけを見てはいけません。GPU 利用率、通信時間の割合、ピークメモリ、step time の分散も同時に見ます。GPU 利用率が低く通信時間が高いなら、並列分割が細かすぎるかノード間通信が頻繁すぎます。メモリが上限に近いが通信が高くないなら、checkpointing を増やすか sharding を細かくします。pipeline stage の負荷が不均衡なら、一部 rank が長く待つため、層の割り当てを見直すか microbatch 数を調整します。本当の分散学習は、測定、ボトルネック特定、分割戦略の修正を繰り返すプロセスです。

## 11. よくある故障シグナル

分散プログラムで最もよくある問題はエラーではなく、停止です。長時間出力がないときは、まずすべての rank が同じ collective に入っているか、テンソル形状が一致しているか、送信元・受信先が対応しているかを確認します。学習は走るが速度が大きく揺れる場合は、データ読み込み、ログ出力、checkpoint 保存、ノード間ネットワーク混雑を確認します。ある rank だけメモリが明らかに高い場合は、モデル分割の不均衡、pipeline stage の負荷不均衡、または解放されていない activation が原因であることが多いです。デバッグでは、まず小さなモデルと少ない rank で再現し、徐々に規模を広げます。どんな変更後も再測定し、直感だけでボトルネックや最適化効果を判断しないようにします。

## 12. まとめ

この講義の核心は、分散学習とはデータ、パラメータ、勾配、optimizer state、activation を分割し、計算、メモリ、通信のあいだでトレードオフすることだ、という点です。DDP は単純ですがメモリ冗長です。ZeRO/FSDP は sharding でメモリを下げますが通信を増やします。Activation checkpointing は再計算でメモリを節約します。Tensor parallel は大きな行列の分割に適していますが高速 collective が必要です。Pipeline parallel は深いモデルを分割できますが、bubbles とスケジューリング問題を扱う必要があります。

ハードウェアは進歩し続けますが、モデル規模もハードウェア限界に近づき続けます。したがって階層的メモリ、通信ボトルネック、再計算、sharding といったシステム問題は消えません。大規模モデル学習における本当の性能は、アルゴリズム、モデル構造、並列戦略、ハードウェアトポロジを共同設計することから生まれます。


---


# CS336 第9回チュートリアル：Scaling Laws（一）

> 翻案メモ: この日本語チュートリアルは中国語版講義ノートをもとに、元の構成、技術用語、数式、コード風表記を保ちながら、日本語で自然に学べるように説明を整えたものです。

この講義では、大規模言語モデル学習における最も重要な工学的道具の 1 つである scaling laws（スケーリング則）を扱います。その目的は「モデルは永遠に賢くなり続ける」と主張することではありません。小規模実験から大規模学習の結果を予測し、本当に巨額の計算資源を使う前に、実務上の問いに答えることです。どれくらい大きなモデルを学習すべきか。どれくらいのデータを使うべきか。アーキテクチャ、optimizer、batch size、学習率は規模に応じてどう変えるべきか。固定 FLOPs のもとで、モデルパラメータと学習 token をどう配分するのが最も得か。

## 1. なぜ Scaling Laws が必要なのか

10 万枚の H100 があり、最強のオープンソース言語モデルを学習できるとします。システム、データ、アーキテクチャはすべて準備できています。しかし、まだ高価な問題が残ります。巨大モデルを何度も学習してハイパーパラメータを調整することはできません。従来の「大きなモデルを学習する、結果を見る、また調整する」という方法は、最前線の規模ではあまりに高価です。

Scaling laws の考え方は次の通りです。

1. compute、データ量、またはパラメータ数が数桁にまたがる小さなモデル群を学習する。
2. モデル loss と投入資源のあいだの単純な関数関係をフィットする。多くの場合はべき乗則である。
3. その関係をより大きな規模へ外挿し、大規模モデルの性能予測と学習計画の選択に使う。

したがって scaling laws は「規模を意識した」工学的方法です。LLaMA や GPT など既存設計を盲目的にコピーするのではなく、候補アーキテクチャ、optimizer、データ配合、学習予算を体系的に比較できます。

## 2. 基本形：Loss とデータ、モデル、計算量の関係

経験的に、言語モデルの cross entropy loss は、データ量、モデルパラメータ数、学習計算量に対して log-log 線形関係を示すことがよくあります。つまり、横軸を資源規模の対数、縦軸を excess loss（不可約損失を超える部分）の対数にすると、曲線はほぼ直線になります。これはべき乗則と等価です。

```text
L(x) = L_infinity + A * x^(-alpha)
```

ここで `x` はデータ量、非 embedding パラメータ数、または compute です。`L_infinity` は不可約損失、`alpha` は scaling exponent で、資源を増やしたときに loss がどれだけ速く下がるかを表します。

この関係には通常 3 つの領域があります。

- ランダム推測領域：モデルまたはデータが小さすぎ、挙動が不安定で外挿しにくい。
- べき乗則領域：loss が規模とともに安定して下がり、scaling laws が最も役立つ領域。
- 飽和領域：不可約誤差に近づき、追加資源の効果が小さくなる。

スケーリング実験を行うときは、データ点ができるだけべき乗則領域に入るようにします。たとえば「データ scaling」を研究するなら、モデル容量が先にボトルネックにならないよう、モデルは十分大きくする必要があります。「モデル scaling」を研究する場合も、学習 token が早すぎる段階でモデルを制限しないようにします。

## 3. データ Scaling：なぜべき乗則は自然なのか

統計学習の観点では、データが多いほど推定誤差は小さくなります。最も単純な例はガウス分布の平均推定です。平均二乗誤差はおよそ `sigma^2 / n` であり、対数を取ると傾き -1 の直線になります。

しかし実際のニューラルネットワークは 1 つの平均を推定しているわけではなく、高次元空間で複雑な関数を学習しています。入力空間を多数の小領域に分け、各領域で局所平均を推定すると考えると、次元が高いほど各領域に必要なデータが増え、誤差の低下は遅くなります。ノンパラメトリック統計でよく知られる結論として、誤差指数はタスクの内在次元に依存します。そのため実タスクのデータ scaling exponent は 1 よりはるかに小さいことが多く、初期の実験では機械翻訳、音声、言語モデリングの指数が 0.1 から 0.3 程度にすぎないこともありました。

これは scaling exponent が単なるフィットパラメータではなく、タスクの学習しやすさも反映していることを示します。指数が小さいほど、データ追加による利益は遅くなります。

データ scaling の工学的用途には次があります。

- データソース品質の比較：異なるデータ mixture が主に曲線の切片を変え、傾きをあまり変えないなら、小モデルでデータを選別できる。
- データ配合の最適化：異なる data mixture に scaling 曲線をフィットし、大規模でどの組み合わせがよいか予測する。
- multi epoch 学習の分析：同じ token を繰り返すと利益は逓減し、通常は「有効データ量」で scaling law を補正できる。
- 高品質な重複データと低品質な新規データのトレードオフ：高品質データが限られる場合、Wikipedia や書籍を繰り返すか、低品質な Web データを増やすかを判断する。

## 4. モデル Scaling：小規模実験でアーキテクチャとハイパーパラメータを選ぶ

Scaling laws はデータだけでなく、モデルや学習方法の比較にも使えます。古典的な方法は、複数の候補を学習し、複数の compute scale で loss 曲線を観察することです。2 本の曲線の傾きが近く、交差しないなら、その差は「定数倍の compute efficiency gap」と解釈できます。たとえば Kaplan らの実験では、Transformer は LSTM に対して明らかな優位性を示しました。同じ loss 目標に到達するには、LSTM は何倍もの計算量を必要とする可能性があります。

同様の方法は次に使えます。

- アーキテクチャ選択：Transformer、LSTM、state space model、GLU、MoE などが拡大後も優位か比較する。
- optimizer 選択：Adam と SGD が安定した compute efficiency gap を示す可能性がある。
- 深さ/幅の比率：多くのハイパーパラメータには鋭い最適点ではなく、広い「ほぼ最適な盆地」がある。
- パラメータ数え上げ：embedding パラメータと非 embedding パラメータは scaling の挙動が異なる。MoE では総パラメータと活性化パラメータも区別する必要がある。

重要な注意点として、scaling laws は next-token cross entropy / log loss については通常安定していますが、下流 benchmark について同じとは限りません。perplexity が規模とともに下がっても、質問応答、in-context learning、推論などの能力が同じ法則で向上する保証はありません。したがって実務では loss を主な予測対象にしつつ、下流評価で検証する必要があります。

## 5. Batch Size、学習率、規模

学習規模が大きくなると、batch size と learning rate を単純に固定することはできません。

Batch size には「critical batch size」という概念があります。小さい batch では、batch を大きくすることは有効な勾配サンプルを増やすことに近く、並列効率を高めます。しかしある点を超えると、batch 増加の限界利益は急速に下がります。この閾値は目標 loss に依存します。モデルがよく学習され、目標 loss が低いほど、通常はより精密な勾配が必要になるため、より大きな batch を許容または必要とする場合があります。実際の大規模モデル学習では、学習の進行に合わせて batch size を徐々に大きくすることがよくあります。

学習率もモデル幅に応じて変わります。標準的な parameterization では、モデルが広いほど最適学習率は小さくなることが多いため、規模ごとに調整するか、「最適学習率 vs モデル幅」の scaling 関係をフィットする必要があります。別の考え方が μP（mu-parameterization）です。幅に応じて初期化、学習率、forward 出力を再スケールすることで、小モデルで調整した学習率を大モデルへ移しやすくします。これは重要な考え方を示しています。規模を意識すべきなのはハイパーパラメータ調整だけではなく、parameterization 自体も規模をまたぐ転移のために設計できる、ということです。

## 6. データ・モデルの同時 Scaling と Chinchilla 最適性

ここまでは単一変数の scaling、つまりデータ、モデル、compute のどれか 1 つだけを変える場合を議論しました。しかし実際の学習では、固定 compute を 2 つのものに配分できます。より大きなモデルに使うか、より多くの学習 token に使うかです。極端な場合はどちらも無駄になります。小さなモデルに多すぎるデータを与えると飽和し、巨大モデルが少しの token しか見ない場合もうまく学べません。

同時 scaling law は次をフィットしようとします。

```text
L(N, D) = E + A / N^alpha + B / D^beta
```

ここで `N` はモデルパラメータ数、`D` は学習 token 数、`E` は不可約損失です。学習 compute はおおよそ `N * D` に比例し、より正確には約 `6ND` FLOPs と書かれることが多いです。総 compute が与えられれば、この制約線上で loss を最小にする `N` と `D` を探せます。

Chinchilla 論文はこの問題を体系的に研究し、有名な結論を得ました。training compute 最適性の意味では、モデルパラメータと学習 token はおおむね同じ比率で増やすべきであり、経験則は 1 パラメータあたり約 20 token です。つまり、GPT-3 のような「パラメータが多く、token が相対的に少ない」モデルと比べ、Chinchilla 風の選択ではより小さなモデルとより多くのデータを使い、同じ学習 FLOPs でより低い loss を得ます。

Chinchilla は 3 種類の方法を使いました。

1. 下包絡線法：異なるサイズのモデルの学習曲線を集め、各 compute で loss が最も低い checkpoint を見つけ、最適パラメータ数と token 数をフィットする。
2. IsoFLOP 分析：いくつかの compute budget を固定し、各 budget でモデルサイズを掃引する。小モデルは多くの token で、大モデルは少ない token で学習し、各曲線の最小点を見つける。
3. 二次元 loss surface の直接フィット：異なる `N, D` 組み合わせを学習し、同時 scaling law をフィットして、最適な compute 配分を導く。

IsoFLOP 分析が最も直感的です。同じ FLOPs のもとで、異なるモデルサイズを横方向に比較し、loss が最も低い点を探します。さらに、これらの最適点が FLOPs の増加に応じてどう変わるかを観察します。Chinchilla の複数の方法は近い結論を与えました。後の再現実験では、第三の方法の元の曲線フィットに小さな問題があることも分かり、修正後は前二者の結果により近づきました。

## 7. 学習結果の予測と実験設計フロー

Scaling laws を実際に使うときは、次のような流れで実験を設計できます。

1. 目標指標を明確にする：不安定な benchmark スコアより、validation cross entropy を優先する。
2. scaling 軸を選ぶ：データ量、非 embedding パラメータ数、総 FLOPs、または同時の `N` と `D`。
3. 複数桁をカバーする：小実験は十分広い範囲にまたがる必要がある。そうでないと外挿は信頼できない。
4. 交絡変数を制御する：データを研究するときはモデルを十分大きくする。モデルを研究するときはデータと training schedule を妥当にする。アーキテクチャ比較では学習予算、tokenizer、データをできるだけ揃える。
5. log-log 曲線をフィットする：べき乗則領域にいるか、曲がり、飽和、ランダム領域の異常がないか確認する。
6. 外挿して案を選ぶ：大規模 loss、最適モデルサイズ、token 数、batch size、学習率、データ配合を予測する。
7. 中規模検証を行う：本当に大きな学習の前に、目標規模に近い 1、2 点で外挿がまだ成り立つか確認する。

## 8. よくある落とし穴と実践的助言

第一に、すべてのパラメータを同じものとして扱ってはいけません。embedding パラメータ、dense 層パラメータ、MoE の総パラメータと活性化パラメータは、学習 loss と推論コストへの寄与が異なります。これらをそのまま混ぜてフィットすると、曲線が曲がったり誤った結論に至ったりします。

第二に、小さすぎるモデルから遠くへ外挿してはいけません。ランダム推測領域、学習率調整不足、大きすぎる batch size、データ不足は、小モデルの点を真のべき乗則から外れさせます。小規模実験では、まず学習が安定しており、loss 曲線が比較可能であることを確認する必要があります。

第三に、最終 checkpoint だけを見てはいけません。cosine learning rate schedule のようなものは完全な cooldown を必要とし、途中で切った中間 checkpoint は短い schedule で最初から学習したモデルと等価ではありません。Kaplan と Chinchilla の推定差の一部は、このような学習曲線の扱いの細部に由来します。

第四に、「定数倍の優位」と「傾きの優位」を区別します。新しいアーキテクチャが曲線全体を下に動かすだけなら、それは固定倍だけ計算効率がよいだけかもしれません。傾きがより急なら、規模が大きくなるほど優位が広がります。これこそ最前線の規模で賭ける価値のある信号です。

第五に、予測結果には不確実性があると考えるべきです。scaling law は物理法則ではなく、特定のデータ、コード、optimizer、学習制度のもとでの経験モデルです。外挿距離が長いほど保守的である必要があります。中規模の検証点を残しておき、フィット曲線が実際の loss をまだ当てられるかを専用に確認するのが望ましいです。検証点が明らかに外れたら、データ品質、学習率、warmup、weight decay、tokenizer、重複除去、評価セット漏洩などを再確認するべきです。

## 9. 現代的視点：学習最適はデプロイ最適ではない

Chinchilla が解いたのは、「与えられた training FLOPs で、どうすれば最低の training/validation loss を得られるか」という問題です。しかし現在のモデルは製品であり、推論コストも同じくらい重要です。大きなモデルは少ない training token でも学習最適かもしれませんが、デプロイ時には 1 token あたりの推論コストが高くなります。したがって多くの現代的モデルは、20 tokens/parameter を大きく超える比率を使います。事前学習時に一度だけ多くのコストを払って、モデルをより小さく、より密に学習し、長期的な推論コストを下げることを選ぶのです。

そのため scaling laws の結論は、目的関数と合わせて理解する必要があります。目的が training FLOPs 最適なら、Chinchilla 比率は重要な基準です。目的が総コスト（学習 + 大量推論）最適なら、より多くの token とより小さなモデルを選ぶ可能性があります。

## 小結

Scaling laws は大規模モデル工学における予測道具です。小規模学習から loss とデータ、モデルパラメータ、compute のべき乗関係をフィットし、大規模学習の結果予測、アーキテクチャと optimizer の比較、ハイパーパラメータ選択、固定予算下でのモデルサイズと学習 token のトレードオフに役立ちます。Chinchilla-style compute optimality は、学習最適なモデルでは単にパラメータを増やすのではなく、パラメータ数と token 数を協調して増やす必要があることを示します。信頼できる scaling 実験には、複数桁の範囲、交絡変数の制御、安定した log loss の優先、そして外挿前に曲線が本当にべき乗則領域にあることの確認が必要です。


---


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


---


# CS336 第11回チュートリアル：Scaling Laws（二）

> 翻案メモ: この日本語チュートリアルは中国語版講義ノートをもとに、元の構成、技術用語、数式、コード風表記を保ちながら、日本語で自然に学べるように説明を整えたものです。

前回の講義では scaling laws の基本的な考え方を紹介しました。小規模実験で loss、パラメータ数、データ量、compute の関係をフィットし、それを大規模モデルへ外挿するというものです。今回の講義は、実際の大規模モデル学習の流れにより近い内容です。公開論文のチームは、scaling laws を使って学習率、batch size、モデルサイズ、学習 token 数、アーキテクチャをどのように選んでいるのでしょうか。どのようなフィットは信頼でき、どのようなものは単に直線に見えているだけなのでしょうか。

中心的な結論を一文でまとめると、scaling laws は単一の公式ではなく、大規模学習のリスクを下げるための実験方法論です。通常、小モデルによる代理実験、ハイパーパラメータ転移、IsoFLOP 分析、WSD 学習率スケジュール、μP parameterization、そして外挿結果の保守的な検証を含みます。

## 1. 実際の学習における Scaling 問題

最前線の言語モデルを学習する前に、チームはいくつかの高価な問いに答える必要があります。

1. 与えられた training FLOPs のもとで、どれくらい大きなモデルを使い、何 token 学習すべきか。
2. batch size は規模に応じてどう変えるべきか。
3. モデルが大きくなるにつれて学習率を下げる必要があるのか。
4. 小モデルで調整したハイパーパラメータは大モデルへ移せるのか。
5. 新しいアーキテクチャが小規模でよく見えるとして、拡大後も本当に得なのか。

これらの問いは、70B や 400B モデルで何度も試行錯誤して答えられるものではありません。公開されている最前線実験でよく見られる戦略は、まず 10M、100M、1B 規模の代理モデルを多数学習し、傾向をフィットし、目標規模では少数の検証または最終学習だけを行うことです。Cerebras-GPT、MiniCPM、DeepSeek LLM、Llama 3、Hunyuan、MiniMax-01 は、それぞれ異なる形でこの流れを示しています。

## 2. Cerebras-GPT：μP でハイパーパラメータを安定させる

Cerebras-GPT は約 0.1B から 13B までのモデルを学習し、**μP（mu-parameterization、maximal update parameterization）** の検証を重視しました。標準的な parameterization では、モデルが広くなるほど最適学習率は小さい値へ移動することがよくあります。小モデルで調整した学習率をそのまま大モデルに使うと、学習が不安定になったり、loss が高くなったりします。μP の目標は、初期化と層ごとの学習率を再スケールし、同じ「基礎学習率」が異なる幅でもほぼ最適になるようにすることです。

実践的には、μP はおおよそ次のようにします。非 embedding 重みの初期化を幅に応じてスケールする。Adam/AdamW を使うときは、すべてのパラメータに 1 つのグローバル学習率を共有させるのではなく、fan-in や幅に応じて層ごとの学習率もスケールする。こうすると、小モデル上で密なグリッドサーチを行い、学習率、初期化、幅と深さの比率などの設定を見つけられます。モデルを大きくしたときにも、それらの設定が有効である可能性が高くなります。

Cerebras-GPT の実験では、μP の曲線は期待される scaling law によりよく合い、標準 parameterization では規模によって振動や逸脱が起こりやすいことが示されました。これは μP がなければ大モデルを学習できないという意味ではありません。μP は「規模ごとに学習率を調整し直す」問題を、「小規模で調整して転移する」問題へ変換しているのです。

## 3. MiniCPM：小モデル、長い学習、WSD スケジュール

MiniCPM の目標は、1B から 2B 程度の非常に強い小型モデルを、大量データで十分に学習することです。その scaling 実験には 3 つの重要な道具があります。

第一に、学習率転移を安定させるため、やはり μP を使います。MiniCPM は数千万パラメータの小さな代理モデルでハイパーパラメータを探索し、その後数億または 1B 規模へ拡大します。実験では、異なるモデルサイズの最適学習率がおおむね同じ位置に落ちており、μP の実用的価値を支持しています。

第二に、critical batch size をフィットします。臨界 batch size は、「batch をこれ以上大きくしても利益が逓減し始める点」と理解できます。モデルが大きく、目標 loss が低いほど、通常はより大きな batch を使えます。MiniCPM は異なる batch size の学習曲線から、目標 loss と最適 batch size の log-log 関係をフィットし、それを目標学習規模へ外挿します。

第三に、この講義で特に重要な点として、**WSD（warmup-stable-decay）** 学習率スケジュールを使い、Chinchilla 風のデータ/モデル scaling 実験のコストを下げます。

通常の cosine schedule の問題は、総学習 token 数が異なると、cosine 曲線全体が異なることです。1T token まで学習するモデルの 100B token 時点の checkpoint は、「最初から 100B token だけ学習し、完全な cooldown を行った」モデルと等価ではありません。したがって、長い学習の途中 checkpoint を短いデータ量の実験点として単純に使うことはできません。

WSD は学習率を 3 段階に分けます。

```text
warmup -> stable plateau -> decay/cooldown
```

利点は stable 段階を再利用できることです。より短いデータ量の最終 loss を推定したい場合、中間 checkpoint から分岐し、別個の decay 段階を接続できます。これにより、1 本の長い学習と複数の短い cooldown で、複数のデータ量終点を近似でき、重複学習コストを大幅に節約できます。

MiniCPM は WSD を使って Chinchilla 風分析を行い、2 つの方法でモデル/データの最適比を推定しました。1 つは学習曲線の下包絡線を取る方法、もう 1 つは 2 次元の loss surface を直接フィットする方法です。

```text
L(N, D) = E + A / N^alpha + B / D^beta
```

得られた token/parameter 比率は非常に高く、約 192:1 でした。この数字を必ずしも普遍的な法則として扱うべきではありません。より重要な示唆は、Chinchilla の 20:1 が破れない硬い規則ではないことです。現代の高品質データ、改良されたアーキテクチャ、より強い最適化により、「より多くの token、より小さいモデル」という選択肢が魅力的になる場合があります。特に推論コストも目的関数に含める場合はそうです。

## 4. DeepSeek LLM：学習率、Batch Size、IsoFLOP を直接フィットする

DeepSeek LLM の公開論文が有用なのは、別のより直接的な scaling 方法を示しているからです。μP に頼らず、batch size と learning rate が規模に応じてどう変わるかを明示的にフィットします。

DeepSeek はまず小モデルで batch size と学習率のグリッドサーチを行い、各規模での最適点またはほぼ最適な領域を見つけます。その後、これらの最適 batch size、学習率、training FLOPs を log-log 図に置いて傾向をフィットし、7B と 67B モデルへ外挿します。

ここには実践上の判断があります。batch size の scaling は比較的きれいなことが多い一方、学習率の scaling はノイズが大きく、より疑わしいことがあります。学習率曲線は水平線でも説明できるように見える場合があります。そのため学習率のフィットは、正確な公式を得るためというより、正しい数量級を得るためのものです。大モデル学習にはしばしば広い「使える盆地」があり、学習率が 1 桁ずれていなければ学習はまだ可能なことがあります。

DeepSeek は Chinchilla / IsoFLOP 分析も行っています。IsoFLOP の手順は次の通りです。複数の compute budget を固定する。各 budget のもとで異なるモデルサイズを学習し、小さいモデルにはより多くの token を見せ、大きいモデルにはより少ない token を見せる。固定 FLOPs 曲線それぞれで最小 loss の点を見つける。最後に、最適パラメータ数と最適 token 数が FLOPs とともにどう増えるかをフィットする。学習率フィットと比べると、このような IsoFLOP 曲線は通常より安定していて信頼できます。

DeepSeek も WSD 風の cooldown を使って重複学習を減らし、最終的におよそ `10^20` FLOPs から `10^24` FLOPs へ外挿し、7B と 67B モデルの loss をかなり正確に予測しました。これは、学習制度、データ、アーキテクチャが一貫していれば、loss scaling の外挿が大規模学習前のリスク管理ツールとして本当に使えることを示しています。

このような成功した外挿は、多くのチームが正式学習の前に大量の「小実験予算」を投入する理由も説明します。これらの実験自体も高価かもしれません。しかし目標規模の学習失敗を 1 回でも避けられるなら、そのコストには価値があります。より現実的には、scaling law が最終 benchmark をすべて正確に予測する必要はありません。学習率の数量級ミス、不適切な batch size、明らかにずれた token/parameter 比、拡大後に利点を失うアーキテクチャを事前に発見できるだけでも、大量の計算資源を節約できます。

## 5. 近年のモデルに見られる傾向：比率は変わり、方法は再利用される

Llama 3、Hunyuan、MiniMax-01 の論文は、MiniCPM や DeepSeek ほど多くの scaling 詳細を公開していませんが、それでもいくつかの傾向を示しています。

Llama 3 は IsoFLOP / Chinchilla 分析をやり直し、最適 token/parameter 比率を約 40:1 としました。これは Chinchilla の 20:1 より高い値です。また、training loss や negative log likelihood を下流 benchmark accuracy へ写像しようとしました。たとえば loss から MMLU などのタスク性能への関係を sigmoid でフィットします。動機は明確です。チームが本当に関心を持つのは log loss そのものではなく、下流能力だからです。ただし benchmark スコアはよりノイズが大きく、飽和もしやすいので、通常はまず loss を予測し、それを補助的に使って benchmark を予測します。

Hunyuan の分析では、さらに高い active-parameter token 比率、たとえば約 96:1 が得られています。ここでは MoE や疎モデルにおける「総パラメータ」と「活性化パラメータ」の違いに注意が必要です。この比率を dense model の比率と直接混ぜて比較することはできません。

MiniMax-01 は scaling laws をアーキテクチャ選択に使っています。softmax attention、linear attention、hybrid attention の loss-compute 曲線を比較し、それらの下包絡線や最適 model/token 傾向が近いかを観察します。同じ compute で linear attention の scaling 曲線が明らかに悪化しないなら、長文脈モデルに採用する根拠になります。これは scaling laws がサイズ選択だけでなく、新しいアーキテクチャを拡大する価値があるかを判断するためにも使えることを示しています。

## 6. μP の直感：Activation と更新のスケールを制御する

μP の背後にある数学的直感は、2 つの条件に簡略化できます。

第一に、モデルが広くなっても、activation の各座標が爆発したり消えたりしてはいけません。ある層が行列積 `h_l = W_l h_{l-1}` であるとき、出力 activation のスケールを安定させるには、一般的な初期化では fan-in の平方根に応じてスケールします。おおよそ次の形です。

```text
W_l ~ 1 / sqrt(fan_in)
```

これは Kaiming/Xavier 初期化の直感と一致しています。

第二に、1 回の勾配更新後、activation の変化量も幅に応じて爆発したり消えたりしてはいけません。この条件は、学習率が層幅に応じてどう変わるべきかを制約します。SGD では fan-out/fan-in に似た比率を導けます。Adam/AdamW では adaptive normalization が勾配スケールを変えるため、一般的な μP ルールでは学習率を fan-in や幅に応じてスケールします。

したがって μP の重点は初期化だけではありません。「初期化 + 層ごとの学習率 + 一部の forward scaling」が一体となって、更新スケールの安定性を保証することです。Transformer では attention logits のスケーリングも関係する場合があります。一部の μP 実装では、更新安定性を満たすために、従来の `1/sqrt(d)` ではなく `1/d` の attention scaling を使います。

経験的研究では、μP は多くの変更に対して頑健であることが示されています。ReLU、SwiGLU、Squared ReLU を入れ替える場合や、一定範囲で batch size を変える場合でも、学習率転移は成り立ちます。ただし万能ではありません。learnable norm gain、強い weight decay、Lion のような sign-gradient 型 optimizer は、μP の転移仮定を壊す可能性があります。言い換えれば、μP は特定の optimizer と parameterization 設計に対する工学的道具であり、すべての学習レシピに自動的に成り立つ定理ではありません。

## 7. 実験フィットの実践的な流れ

比較的完全な scaling 実験は、次の手順で進められます。

1. データ、tokenizer、アーキテクチャ族、optimizer、学習コードを固定し、交絡変数を減らす。
2. 小規模から中規模のモデルをいくつか選び、パラメータ数または FLOPs で少なくとも数桁を覆う。
3. 小規模グリッドサーチで、学習率、batch size、warmup、weight decay などの重要ハイパーパラメータを探索する。
4. μP を使う場合は小モデルで調整し、学習率が幅をまたいで転移することを検証する。μP を使わない場合は、学習率と batch size が規模に応じてどう変化するかを明示的にフィットする。
5. WSD または同等の方法で、異なる token 終点における真の cooldown loss を集め、cosine の途中 checkpoint を誤用しない。
6. IsoFLOP または 2 次元 `L(N,D)` フィットを行い、与えられた compute に対する最適モデルサイズと学習 token を推定する。
7. 最終規模より小さいが、フィット点より明らかに大きい中規模で検証し、外挿が当たるか確認する。
8. 最終学習中も監視を続ける。loss が予測から外れたら、データ、optimizer、batch、学習率 schedule、実装 bug を早期に調査する。

## 8. 限界とよくある誤解

第一に、log-log の直線は真理ではありません。Scaling law は経験的フィットであり、データ分布、モデル族、optimizer、学習 schedule、評価セットに依存します。外挿が遠いほど不確実性は大きくなります。

第二に、training loss は最も安定していますが、下流能力は必ずしも安定しません。perplexity の低下は通常よい兆候ですが、数学推論、ツール利用、長文脈、指示追従などの能力には閾値効果や評価ノイズがあるかもしれません。

第三に、token/parameter 比率に統一された定数はありません。20:1、40:1、96:1、192:1 はいずれも異なる論文に登場します。違いはデータ品質、アーキテクチャ、MoE かどうか、学習目標、デプロイコストから来ます。Chinchilla が与えるのは training FLOPs 最適の基準であり、プロダクト全体コストの最適解ではありません。

第四に、学習率 scaling は loss scaling より脆弱です。Batch size と IsoFLOP は比較的フィットしやすい一方、学習率曲線は平坦でノイズが大きいことがあり、数量級の参考として使うべきです。

第五に、長い学習の中間 checkpoint を短い学習の終点として扱ってはいけません。cooldown のない checkpoint loss はしばしば高めで、data scaling のフィットを体系的に汚染します。WSD の価値はまさにこの問題を解決することにあります。

## 小結

この講義では、実際のモデル学習における scaling laws の使い方を見ました。Cerebras-GPT と MiniCPM は μP でハイパーパラメータ転移を安定させました。MiniCPM と DeepSeek は WSD で Chinchilla 分析のコストを下げました。DeepSeek は batch size、学習率、IsoFLOP を直接フィットし、大モデル loss の予測に成功しました。Llama 3、Hunyuan、MiniMax-01 は、現代のチームが今も IsoFLOP 分析を再利用していること、ただし最適 token/parameter 比率やアーキテクチャ問題は目標に応じて変わることを示しています。

本当に信頼できる scaling 作業は、「線を 1 本引いて信じる」ことではありません。小規模実験でリスクを体系的に取り除くことです。学習率、batch size、モデルサイズ、データ量、アーキテクチャ選択は、拡大前に証拠を持つべきです。同時に外挿の限界を認め、中規模の検証点で予測を校正する必要があります。その価値は、大モデル学習をギャンブルから管理可能な工学判断へ変えることにあります。まず安価な実験で探索空間を狭め、次に最も証拠の強い案へ高価な学習 compute を投入するのです。


---


# CS336 第12回チュートリアル：LLM Evaluation

> 翻案メモ: この日本語チュートリアルは中国語版講義ノートをもとに、元の構成、技術用語、数式、コード風表記を保ちながら、日本語で自然に学べるように説明を整えたものです。

## 1. なぜ評価は単純ではないのか

LLM evaluation は表面的には 1 本のスクリプトのように見えます。モデルを決め、prompts を用意し、モデルを呼び出し、outputs を集め、指標を計算し、平均を取る。しかし本当に難しいのは、その数字で何を答えたいのかを決めることです。

立場によって、関心のある評価はまったく異なります。

- ユーザーまたは企業：Claude、Gemini、GPT、オープンソースモデルの中から選ぶなら、自分の業務に最も合うのはどれか。
- 研究者：モデルは本当により強い汎用能力を持つのか。AI は科学的な意味で進歩しているのか。
- 政策立案者：モデルはどのような利益とリスクをもたらすのか。十分に安全なのか。
- モデル開発者：ある学習、データ、alignment の方法は本当にモデルを改善したのか。

したがって「唯一正しい」評価は存在しません。leaderboard のスコアは、その入力分布、呼び出し方法、採点規則、利用目的を理解して初めて解釈できます。評価はモデル開発の方向も形作ります。ある benchmark が目標になると、開発者はそれを最適化します。指標が過度に最適化されると、元の意味を失うことがあります。これが LLM 評価における Goodhart's law です。

## 2. 評価フレームワーク：入力、呼び出し、出力、解釈

信頼できる評価は、4 つの問いに分解できます。

第一に、入力はどこから来るのか。prompt は実際のユースケースを覆っているか。難しい例、ロングテールの例、境界ケースを含むか。多ターンチャットの場合、後続の入力はモデルの前の回答に依存するため、静的な test set では実際の会話を模擬できないかもしれません。red-team testing でも、モデルの挙動に応じて攻撃 prompt を適応的に生成する必要があることが多く、そうしないと稀な失敗を見つけにくくなります。

第二に、モデルをどう呼び出すのか。zero-shot、few-shot、chain-of-thought、ツール呼び出し、RAG、agent scaffolding はすべて結果に大きく影響します。初期の base model は、形式を説明するために few-shot 例を必要とすることがよくありました。現代の instruction-tuned model は、通常 zero-shot で「A/B/C/D だけを出力せよ」のような指示に従えます。prompt の順序、形式、例の選び方はいずれも分散を生みます。

第三に、出力をどう評価するのか。選択問題なら accuracy、コード課題なら pass@1 または pass@k、オープンエンド生成なら人間の選好や LLM-as-a-judge が必要になる場合があります。コストも考慮すべきです。スコアが高くても、価格、latency、推論 token 数がはるかに大きいモデルが、必ずしもよりよいシステムとは限りません。誤りの代償も種類によって異なります。医療、法律、安全などの場面では平均 accuracy だけでは不十分です。

第四に、スコアをどう解釈するのか。91% は良いのか悪いのか。デプロイに十分なのか。モデルが能力を学んだことを示すのか、それとも似た問題を見たことがあるだけなのか。評価対象は base model、chat model、完全な agent system、あるいは特定の学習方法なのか。これらは事前に明確にする必要があります。

## 3. Perplexity：今も重要な基礎指標

Perplexity は、モデルがあるデータセットにどれだけ高い確率を割り当てるかを測ります。言語モデルは本質的に token 系列上の確率分布です。perplexity が低いほど、そのデータセット内の token をよりよく予測できることを意味します。従来の言語モデリング研究では、Penn Treebank、WikiText、One Billion Word などの固定データセットで学習とテストを行い、test perplexity を下げることを目標にしていました。

GPT-2 以降、パラダイムは変わりました。モデルは大規模な web text で事前学習され、そのまま多くの下流タスクや perplexity benchmark に転移されます。このとき評価は out-of-distribution generalization に近くなります。モデルは Penn Treebank 専用に学習されていなくても、学習コーパスが十分に広ければよい結果を出す可能性があります。

perplexity の利点は次の通りです。

- 滑らかさ：各 token の log probability を使うため、「正解/不正解」の accuracy より細かい情報を持つ。
- scaling law に適している：モデル規模、データ、計算量の変化に伴う loss 曲線をフィットしやすい。
- 網羅性：最終答えだけでなく、データセット内のすべての token に注目する。
- train/test 分離が信頼できる限り、答えの形式でずるをしにくい。

ただし perplexity にも限界があります。第一に、下流タスク性能と常に強く相関するわけではありません。短期的または具体的なタスクでは、関係が乱れることがあります。第二に、leaderboard がモデルに確率を提供させる場合、提供者の logits または確率 API が正しく正規化されていると信頼する必要があります。実装 bug によって偽の低 perplexity が生じることもあります。最後に、perplexity 最大化派は「真の分布に一致すればすべて解決する」と考えますが、これは必ずしも最も効率的な道ではありません。実務上重要でない token も多いからです。

LAMBADA の欠落語予測や HellaSwag の多肢選択 continuation のように、perplexity に近いタスクもあります。モデルは候補 continuation の likelihood を比較します。しかしこれらのタスクも飽和しやすく、Web 上の元ソースからの近似的な汚染リスクがあります。

## 4. Multiple-choice benchmarks：MMLU、MMLU-Pro、GPQA、HLE

MMLU は古典的な LLM 知識 benchmark の 1 つで、57 分野の多肢選択問題を含みます。GPT-3 以後に登場し、当時は base model に few-shot で多くの科目の問題を解かせる設定自体が新しいものでした。MMLU の名前には “language understanding” が含まれますが、実際には知識試験に近いです。多くの問題は純粋な言語理解ではなく、特定分野の事実を問います。

MMLU のスコアは、学習方法と評価設定と合わせて解釈する必要があります。base model が MMLU 専用に最適化されていないのに、多分野の選択問題で高得点を取るなら、強い汎用知識と転移能力を持つ可能性があります。しかし開発者が類似問題を集め、prompt を調整し、chain-of-thought や ensemble を MMLU 向けに使っている場合、高得点が同じ程度の汎用能力を示すとは限りません。

MMLU-Pro は MMLU の飽和問題を緩和しようとします。ノイズのある問題や簡単な問題を取り除き、選択肢を 4 個から 10 個に増やし、chain-of-thought をより多く使います。これにより frontier models の accuracy は下がり、benchmark は再び識別力を持ちます。

GPQA は専門家レベルの難問を重視します。問題は博士号保持者や領域専門家によって作成・検証され、目標は “Google-proof” であることです。つまり非専門家は検索しても答えにくい。初期の GPT-4 の成績は高くありませんでしたが、新しいモデルでは大きく改善しています。これは「人間にとって検索が難しい」ことが「LLM にとって永遠に難しい」ことを意味しないと示しています。評価時には、モデルがインターネット利用を許可されているかも確認すべきです。ブラックボックス API が裏で検索機能を呼ぶ可能性があるからです。

Humanity's Last Exam（HLE）はさらに、極めて難しい、多モーダルの、選択式または短答式問題を集めます。賞金と署名を用いて問題提供者を集め、frontier model が簡単に解ける問題をフィルタします。利点は難しいことです。欠点は分布バイアスが明確なことです。問題を作る人は LLM に詳しいことが多く、意図的に「モデルに難しい問題」を設計しがちです。そのため HLE は普通のユーザー需要を代表しません。

## 5. オープンエンドと instruction following の評価

現代のチャットモデルの中心能力は、試験問題を解くだけではなく、自然言語指示に従ってオープンなタスクを完成させることです。オープンエンド出力には一意の ground truth がないため、評価はより難しくなります。

Chatbot Arena の方法では、ユーザーが実際の prompt を入力し、2 つの匿名モデルがそれぞれ回答し、ユーザーがよりよい答えを選びます。その pairwise preference から Elo ランキングを計算します。利点は、動的で、実際の利用に近く、新しいモデルを受け入れられることです。欠点は、ユーザー分布が制御されず、prompt が娯楽やテスト目的かもしれず、leaderboard が重要になるほど最適化や操作の対象になりやすいことです。近年の Arena をめぐる議論も、評価プロトコル、提出権限、モデルバージョン、データ透明性が重要であることを示しています。

IFEval は instruction following の「制約遵守」を専門に評価します。たとえば、何語未満でなければならない、特定の語を含むまたは含まない、特定形式を使う、などです。利点はスクリプトで自動検証できることです。欠点は形式的制約だけを検査し、意味的品質を見ないことです。10 語の物語が長さ条件を満たしていても、良い物語とは限りません。

AlpacaEval は LLM-as-a-judge を使い、モデル回答と参照モデル回答を比較して勝率を計算します。自動で高速、再現可能ですが、judge model にはバイアスがあります。たとえば初期の小モデルは、より長い回答を出すことで GPT-4 judge をだませたため、後に長さ補正が追加されました。WildBench などのデータセットは、実際の人間とモデルの会話から抽出し、judge に checklist に基づいて評価させます。通常、Chatbot Arena との相関も報告します。

## 6. Agent benchmark：モデルを評価しているのか、システムを評価しているのか

多くのタスクではツール呼び出しと多段反復が必要です。この場合、評価対象はもはや LM 単体ではなく、「モデル + agent scaffolding」のシステムです。

SWE-bench は GitHub issue とコードベースを agent に与え、コードを修正して patch を提出させ、最終的に単体テストが通るかを見ます。Cybench は CTF のサイバーセキュリティ環境で、agent にコマンド実行、サーバー探索、flag 取得を行わせます。MLE-bench は Kaggle を模擬し、agent がタスクを読み、学習コードを書き、ハイパーパラメータを調整し、結果を提出する必要があります。これらの benchmark は実際のワークフローに近いですが、スコアはツール、文脈管理、リトライ戦略、時間予算、コストに強く影響されます。

したがって agent スコアを報告するときは、次を明記する必要があります。インターネット利用は許可されているか。何ステップ実行できるか。人間のヒントは許されるか。隠しテストはあるか。何ドル、どれくらいの時間を使ったか。大量サンプリングと高価な推論で高得点を取るシステムは、1 回限りの低コスト回答と同じ能力ではありません。

## 7. Contamination：学習データ汚染と評価の妥当性

現代のモデルは大規模インターネットデータで学習され、開発者は通常完全なコーパスを公開しません。そのため train/test overlap を完全に排除することはほぼ不可能です。汚染には、逐語的な重複、近重複、言い換え、翻訳、解法の漏洩、答えの漏洩があります。単純な n-gram 重複除去で一部は発見できますが、言語をまたぐ版や意味的に等価な版は検出できません。

対応方法は大きく 3 種類あります。

- データ decontamination：test set と学習コーパスの文書、段落、n-gram overlap を検査し、疑わしいサンプルは保守的に削除する。
- 行動検出：選択肢順序、問題順序、珍しいテキストに対するモデルの異常な好みから、データを見たかどうかを推定する。
- コミュニティ規範：論文や model card は、decontamination 方法、テストセット漏洩を確認したか、信頼区間、標準誤差を報告すべきである。

汚染は選択問題だけでなく、HellaSwag、WikiHow 派生タスク、数学問題、コード問題にも影響します。benchmark データ自体にもアノテーション誤りがあるかもしれません。モデルスコアが非常に高いとき、残る誤りのかなりの部分はモデル能力不足ではなく問題ノイズに由来する可能性があります。

## 8. Human eval、実ユースケース、安全評価

人間評価はオープンエンドタスクでよく使われますが、評価者が誰か、専門家かどうか、採点 rubric は何か、盲検評価かどうか、回答の長さとスタイルを制御するかを明確にする必要があります。一般インターネットユーザーの好み、領域専門家の判断、製品ユーザー満足度は同じものではありません。

実ユースケース評価は試験より難しく、同時により重要です。ユーザーは「質問している」場合があります。つまり答えを知らず助けが必要です。一方で「モデルを試している」場合もあります。つまり自分は答えを知っていて、単にテストしたいだけです。標準化試験の多くは後者に属しますが、商業価値はしばしば前者から生まれます。Anthropic などの研究は実際の会話をクラスタリングし、人々がモデルを何に使っているかを分析します。たとえばコード、執筆、学習、事務作業です。医療領域の MedHELM では、臨床医が診療録要約、治療計画、患者コミュニケーションなどの実タスクを提案します。しかし実データはしばしばプライバシーを含み、公開再現性と現実性の間に緊張があります。

安全評価も「拒否率」だけを見てはいけません。HarmBench、AIRBench などは、モデルが有害リクエストに従うか、あるいは法規制や企業ポリシーに基づくリスク分類をテストします。しかしすべての質問を拒否するモデルは当然「安全」ですが、役に立ちません。したがって安全は有用性と一緒に評価する必要があります。また capability と propensity を区別することも重要です。モデルが危険なことのやり方を知っているかは capability です。それを出力しようとするかは propensity です。閉源 API は propensity と jailbreak 防御をより重視します。オープンウェイトでは capability も重視する必要があります。安全層が fine-tuning で除去される可能性があるからです。

## 9. 信頼できる評価の実践チェックリスト

1. まず評価目的を明確にする：モデル選定、研究、製品監視、安全審査、学習フィードバックのどれか。
2. 評価対象を明確にする：base model、chat model、agent system、または学習方法か。
3. 呼び出しプロトコルを固定し公開する：prompt、few-shot 例、temperature、max tokens、ツール権限、リトライ回数。
4. 品質、コスト、latency、分散を同時に報告し、平均 accuracy だけを報告しない。
5. 選択問題では選択肢順序バイアスを確認し、オープン問題では長さバイアスと judge バイアスを確認する。
6. contamination 検査を行い、方法を報告する。高リスク benchmark は保守的に解釈する。
7. 具体的な予測をサンプリングして確認し、leaderboard の数字だけを見ない。
8. 実デプロイ場面では、ユーザー分布に近い、非公開で更新される eval set を作る。
9. 安全評価は有用性評価と組み合わせ、「全拒否」による虚高を避ける。
10. benchmark は道具であり真理ではないことを忘れない。目標になった benchmark は最適化され、飽和し、場合によっては歪む。

まとめると、LLM evaluation の核心は「スコアを 1 つ出す」ことではありません。現実の問いを、実行可能で、解釈可能で、再現可能な測定プロセスへ翻訳することです。良い評価には、標準化 benchmark の比較可能性と、実ユースケースの代表性の両方が必要です。能力だけでなく、コスト、安全性、汚染、データ品質にも注意しなければなりません。数字の背後にある規則を理解して初めて、モデルがどこで優れ、どこで弱く、自分の目的に合うかを本当に知ることができます。


---


# Stanford CS336 第13回チュートリアル：事前学習データ（Data 1）

> 翻案メモ: この日本語チュートリアルは中国語版講義ノートをもとに、元の構成、技術用語、数式、コード風表記を保ちながら、日本語で自然に学べるように説明を整えたものです。

この講義は、現代の言語モデルでは、最終的な能力を決めるうえでモデル構造よりもデータのほうが重要になることが多い、という中心的な見方から始まります。Transformer アーキテクチャ、optimizer、並列学習などの技術はかなり公開されています。一方で、最先端モデルの論文は学習データについて「複数のデータ源から、ある年までを含む」といった曖昧な説明しかしないことが多いです。この秘密主義には商業競争上の理由だけでなく、著作権や法的リスクも関係します。実践者にとって本当に難しい問いは、「データがあればどう学習するか」ではなく、「どんなデータを学習する価値があるのか、そして生のインターネットをどうやって学習可能なコーパスに変えるのか」です。

## 1. 学習段階とデータの役割

大規模モデルの学習は、大まかに 3 つの段階に分けられます。

- **Pretraining（事前学習）**：Web、コード、書籍、論文、百科事典などに由来する、大量で比較的生に近いテキストを使います。目的は、モデルに言語、知識、汎用的なパターンを学ばせることです。
- **Mid-training（中期学習）**：事前学習のあと、より小さいが高品質で、目的が明確なデータを使って能力を強化します。たとえば数学、コード、長文コンテキスト、多言語などです。
- **Post-training（事後学習）**：instruction tuning、chat data、RLHF/RLAIF などを含みます。モデルをよりアシスタントらしくし、指示に従い、対話でき、安全要件を満たせるようにします。

用語として、**base model** は通常、事前学習または中期学習を終えたモデルを指します。**instruct/chat model** は、事後学習を受け、対話に適したモデルです。現実には、この 3 つの境界は明確ではありません。たとえば Stack Exchange の QA データは事前学習に入れられますが、自然に instruction data のようにも見えます。また現代のデータパイプラインでは、事前学習段階からモデルで選別・書き換えたデータを導入することもあります。

## 2. なぜ「インターネットデータ」は単純な概念ではないのか

「大規模モデルはインターネットデータで学習される」とよく言われますが、これは粗すぎます。実際のデータ源は、通常 3 層の変換を経ます。

1. **Live service（オンラインサービス）**：Wikipedia、GitHub、Reddit、Stack Overflow、ニュースサイトなど。
2. **Raw dump / crawl（生のダンプまたはクロール）**：Common Crawl、Wikipedia dump、GitHub Archive など。
3. **Trainable dataset（学習可能データセット）**：テキスト抽出、言語識別、クリーニング、フィルタリング、重複除去、サンプリング、混合を経た token。

したがって、誰かが「GitHub / Common Crawl / Reddit で学習した」と言ったら、必ず確認すべきです。どのスナップショットを使ったのか。テキストをどう抽出したのか。license はどう扱ったのか。重複除去はしたのか。フィルタ規則は何か。どのフィールドやメタデータを残したのか。これらの決定はモデル能力に大きく影響します。

## 3. 初期のデータ：Books と Wikipedia

BERT が主に使ったデータは **BooksCorpus** と **Wikipedia** でした。BooksCorpus は Smashwords 上の無料の自費出版書籍に由来し、後に利用規約の問題で公開停止されました。これは書籍データの重要性を示しています。書籍には長文構造、物語としての一貫性、長距離依存があり、モデルに長い文脈を理解させるのに適しています。

Wikipedia は長いあいだ「高品質テキスト」の代表と見なされてきました。Wikipedia には明確な編集規範があります。検証可能性、出典の引用、独自研究の禁止、個人的意見の少なさ、そして notability（特筆性）による主題選別です。しかしこれは、Wikipedia がすべての価値ある内容をカバーしないことも意味します。個人的経験、レシピ、フォーラム議論、ニッチな知識、口語表現などは欠けやすいです。

Wikipedia は安全上の問題も提起します。それが **data poisoning（データポイズニング）** です。攻撃者がデータスナップショット生成前に一時的に悪意ある内容を挿入できれば、その後に差し戻されても内容が学習セットに入る可能性があります。より広く言えば、インターネット上の学習データは、さまざまな動機を持つ多くの人によって共同で形作られています。モデルの振る舞いはそれらのデータに影響されますが、学習側が完全に監査するのは非常に困難です。

## 4. WebText：リンク信号で Web ページを選ぶ

GPT-2 の WebText データセットは、重要な考え方を示しました。Web ページをランダムにクロールするのではなく、人間コミュニティのリンクと投票の信号を利用するという方法です。OpenAI は、一定以上の karma を持つ Reddit 投稿がリンクしている Web ページを収集し、約 800 万ページ、40GB のテキストを得ました。直感は、ユーザーに共有され支持されたリンクは、平均的には普通の Web ページより品質が高い、というものです。

WebText は公開されず、後にコミュニティが OpenWebText として再現しました。この種の方法の鍵は **link-based filtering（リンクベースのフィルタリング）** です。高品質なコミュニティ、百科事典の引用、または人手で curated されたページが指す外部リンクを品質信号として使います。後の LLaMA も似た考え方を使いました。Web ページが Wikipedia に引用されたページに似ているかどうかを判定する分類器を学習したのです。

## 5. Common Crawl：最大だが非常に汚い公開 Web ソース

**Common Crawl** は、学術界とオープンソースコミュニティで最もよく使われる大規模 Web ソースです。2007 年から定期的に Web をクロールしており、各クロールには数十億ページが含まれます。クローラは多数の seed URLs から出発し、frontier キューを維持します。これは Web に対する幅優先探索に似ていますが、robots.txt、サーバ負荷、重複 URL、動的ページなどの工学的問題も扱う必要があります。

Common Crawl は 2 つの重要な形式を提供します。

- **WARC**：生の HTTP レスポンス。通常は HTML を含みますが、他のリソースを含むこともあります。
- **WET**：HTML から変換されたプレーンテキスト。これは不可逆な変換です。

HTML-to-text 変換は低レベルな作業に見えますが、学習品質に大きく影響します。Common Crawl 付属の WET、Trafilatura、jusText などのツールを使うと、それぞれ異なるテキストが得られ、モデル評価にも影響します。現代のデータエンジニアリングでは、WET に直接依存するのではなく、WARC から本文を再抽出することがよくあります。

Common Crawl は「インターネット全体」ではありません。カバレッジは疎で、テキストに偏っており、robots.txt に従う、または少なくとも考慮します。すべてのページを含む保証はありません。同時に、大量のスパム、広告、テンプレート、重複、低品質テキスト、攻撃的内容も含みます。したがって Common Crawl は、そのまま学習できるデータセットというより原材料です。

## 6. クリーニング、フィルタリング、重複除去

生の Web ページから学習 token にするまでの典型的な手順は次の通りです。

### 言語識別（Language Identification）

fastText などの分類器で文書の言語を判定し、対象言語だけを残すか、多言語比率に従ってサンプリングします。初期の研究の多くは英語に集中していましたが、Common Crawl 自体は多言語データを含みます。

### ルールベースのフィルタリング（Rule-based Filtering）

C4、Gopher/MassiveText、RefinedWeb、FineWeb などは、多数の手書きルールを使います。たとえば、句読点で終わる行を残す、文が少なすぎるページを除く、汚い語をフィルタする、一定割合の単語に文字が含まれることを要求する、boilerplate を除く、コードやテンプレートらしいものを除く、などです。ルールベースの方法は透明で、安価で、解釈しやすいです。しかし、構造だけ整ったジャンクテキストを残すこともありますし、方言、少数派コミュニティのテキスト、非標準的な書き方を誤って削除することもあります。

### モデルベースのフィルタリング（Model-based Filtering）

CCNet は Wikipedia で n-gram モデルを学習し、「Wikipedia らしい」文書を残しました。GPT-3 は WebText、Wikipedia、books を正例として品質分類器を学習し、Common Crawl から似た内容を探しました。DCLM はさらに進み、OpenHermes、ELI5 などの instruction-like データを正例にし、fastText 分類器で 240T tokens のプールから約 3.8T tokens まで絞りました。

モデルベースのフィルタリングは benchmark を大きく改善できますが、リスクもあります。「品質」が正例分布に狭められてしまうのです。正例が百科事典的、英語中心、主流の書き方に偏っていれば、モデルは多様性を失います。近年は、得られる利益が非常に大きいため、モデルをデータ選別に使うことを再び受け入れ、むしろ強化する傾向があります。

### 重複除去（Deduplication）

Web には重複が非常に多いです。ミラーサイト、転載、テンプレート、動的 URL、コード fork、ドキュメントのコピーなどが重複を生みます。重複除去には、完全一致の重複除去と **fuzzy deduplication（曖昧重複除去）** があります。重複除去は学習の無駄を減らし、モデルが特定テキストを記憶する確率を下げ、特定のソースが過大に重み付けされることを防ぎます。

### 有害内容とプライバシーのフィルタリング

多くのパイプラインは toxicity classifier、safe search、PII anonymization（個人識別情報の匿名化）などの手順を加えます。ただし、これらのフィルタ自体も完全ではありません。強すぎれば現実世界の分布を失い、弱すぎれば安全、プライバシー、法的問題を持ち込みます。

## 7. 典型的な事前学習データセットの系譜

- **C4（Colossal Clean Crawled Corpus）**：Google/T5 が使った Common Crawl のクリーニング版。主にルールベースのフィルタリングに頼り、英語の自然言語テキストだけを残します。
- **The Pile**：EleutherAI コミュニティが構築した 22 個の高品質データ源の混合。Common Crawl、OpenWebText、Stack Exchange、Wikipedia、arXiv、PubMed、GitHub、Books3 などを含みます。「人手で領域を選ぶ」路線を表しています。
- **MassiveText / Gopher**：DeepMind のデータ混合。MassiveWeb、C4、books、news、GitHub、Wikipedia を含み、ルールと安全フィルタを使います。
- **LLaMA データ**：Common Crawl + C4 + GitHub + Wikipedia + Project Gutenberg + Books3 + arXiv + Stack Exchange、合計約 1.2T tokens。公開はされていませんが、RedPajama が再現しました。
- **RefinedWeb / FineWeb**：Web を十分うまくフィルタすれば強いデータが得られる、という主張です。FineWeb は Hugging Face による大規模 Common Crawl の軽量フィルタ版で、さらに選別するための土台になります。
- **DCLM Baseline**：Common Crawl の全体プールを競技形式のデータ基準にし、強い品質分類器で aggressive filtering を行います。近年のオープンソースモデルでよく使われるデータ源になっています。
- **Nemotron-CC**：NVIDIA が DCLM の考え方を拡張したものです。大規模モデルで “educational value（教育的価値）” を採点し、それを高速なモデルへ蒸留し、複数のフィルタを組み合わせます。さらに LLM を使って低品質データを書き換えたり、高品質文書をタスク形式に変換したりも試しています。

これらのデータセットは 2 つの緊張関係を示します。1 つ目は品質と規模のトレードオフです。強くフィルタするほど品質は上がりますが token は減ります。2 つ目は品質と多様性のトレードオフです。高品質な正例に似せるほど、ロングテール知識、口語、非主流テキストを失う可能性が高くなります。

## 8. コード、QA、書籍、論文の特別な価値

異なるソースは異なる能力を提供します。

- **GitHub / The Stack**：主にコード能力を学習しますが、構造化された推論を高める可能性もあります。処理時には license の識別、重複除去、生成ファイルのフィルタ、コードとドキュメントの区別、issues や commit history を使うかどうかの検討が必要です。
- **Stack Exchange / Stack Overflow**：自然に QA 形式を持ち、質問、回答、コメント、投票などのメタデータがあります。高品質な説明を選別するのに適しており、事前学習と instruction training の境界を曖昧にします。
- **Project Gutenberg / PG19**：public domain の書籍で、著作権状態が明確です。長文コンテキスト学習に適していますが、文体は古めです。
- **arXiv / PubMed / Semantic Scholar**：学術論文は知識密度、数学、技術的表現を提供します。しかし形式抽出、数式、引用、著作権を処理する必要があります。
- **Reddit / ELI5**：ユーザーの質問や平易な説明に近く、品質分類器の正例や instruction-like コーパスとして使えます。

## 9. 著作権とデータ利用可能性

インターネット上のほとんどの創作的表現は、Web ページに copyright 表記がなくても、デフォルトで著作権によって保護されます。利用方法には大まかに 2 つあります。license（ライセンス）を得るか、**fair use（フェアユース）** を主張するかです。フェアユースでは、利用が変容的か、作品の性質、使用割合、元の市場への影響などが考慮されます。大規模モデル学習では、学習データをコピーすること自体が著作権に関わります。学習が十分に transformative か、モデルが原文を記憶して再現するか、原作者の市場を代替するかは、いずれも論争点です。

さらに、内容が Creative Commons であったり fair use に該当し得たりしても、プラットフォームの Terms of Service（利用規約）が自動ダウンロードを禁止している場合があります。公開動画だからといって自由にクロールできるわけではありません。大企業は Reddit、Stack Exchange、Shutterstock などのデータを商用ライセンスで得られます。一方、オープンソースや学術チームは、公開 dump、ライセンスが明確なデータ、慎重なフィルタリングにより依存します。

## 10. Mid-training と Post-training のデータ

中期学習と事後学習は、より特定の能力に注目します。長文コンテキスト拡張は後期に行われることが多いです。最初から超長系列で学習するのは高価すぎるからです。データとしては、書籍、長い論文、長いコード、合成された長距離依存タスクなどが使えます。

instruction data では、初期に Super-Natural Instructions や FLAN がありました。これらは伝統的な NLP タスクを instruction format に統一しました。その後、Alpaca/self-instruct、Vicuna、OpenHermes、Evol-Instruct などの合成データ手法が登場しました。強いモデルでタスク、回答、多ターン対話を生成する方法です。合成データは安価でスケールしやすいですが、生成モデルのライセンスに制約され、教師モデルのバイアスを引き継ぐこともあります。もう 1 つの路線は、アノテータを雇って高品質な instruction data を書いてもらうことです。これは高価ですが制御しやすいです。ただし、アノテータが商用モデルをこっそり使って回答を生成しないようにする必要もあります。

## 11. 工学的実践のまとめ

事前学習データパイプラインを構築するときは、次の流れで考えるとよいです。

1. 目標能力を明確にする：汎用知識、コード、数学、多言語、長文コンテキスト、対話スタイル。
2. 生のソースを集める：Web crawl、dump、API、ライセンスデータ、public domain データ。
3. テキスト抽出：HTML/PDF/code/email などの形式をプレーンテキストに変換し、必要なメタデータを保持する。
4. 基本クリーニング：言語識別、エンコーディング修復、boilerplate 除去、長さフィルタ、形式フィルタ。
5. 品質選別：ルール、分類器、LLM 採点、リンク信号、コミュニティ投票信号。
6. 安全とコンプライアンス：著作権、license、robots.txt、ToS、PII、toxicity。
7. 重複除去とサンプリング：完全一致/曖昧重複除去、重複ソースが学習を支配しないようにする。
8. データ混合：能力と品質に応じて mixture weights を設定し、小モデルの ablation で検証する。
9. バージョン記録：スナップショット時刻、処理コード、フィルタ閾値、統計情報を保存し、追跡可能にする。

実際の運用では、最終 token 数だけを見てはいけません。より有用な監視項目は、各ソースの保持率、重複率、平均文書長、言語分布、ドメイン分布、perplexity や品質スコアの分布、フィルタされたサンプル例、そして学習後の目標評価での改善です。データパイプラインはモデルコードと同じようにバージョン管理されるべきです。そうしないと、ある学習がなぜ良くなったのか、または悪くなったのかを説明しにくくなります。

この講義の核心的な結論は、データは空から降ってくるものではない、ということです。学習可能なコーパスは、大量の工学、ヒューリスティック、法的判断、実験的反復の結果です。現代モデルどうしのアーキテクチャ差は大きくないかもしれません。むしろ、データ源、フィルタ戦略、重複除去の品質、合成データ、ライセンスされた資源こそが、モデル差を決める重要な要因です。


---


# Stanford CS336 第14回：データ（二）——生の Web ページから学習可能コーパスへ

> 翻案メモ: この日本語チュートリアルは中国語版講義ノートをもとに、元の構成、技術用語、数式、コード風表記を保ちながら、日本語で自然に学べるように説明を整えたものです。

この講義では、大規模モデルの事前学習における「データエンジニアリング」の議論を続けます。前回の講義は、初期コーパスから Common Crawl、C4、The Pile、LLaMA、Dolma などへ至る、データセット史のような内容でした。今回は、より実行可能な方法に移ります。大量の生の Web ページと少量の理想的データが手元にあるとき、どのように選別し、混合し、重複除去し、それらの決定を学習データのレシピに変えるのか、という問題です。

中心問題は次のようにまとめられます。小さく高品質なターゲット集合 T と、巨大だがノイズの多い生集合 R が与えられたとき、R の中から「T に似ている」部分集合 T' を見つける。これは品質フィルタリングだけではありません。言語識別、ドメイン選択、毒性フィルタリング、数学/コードデータの発掘、合成データの選別、さらに学習中の domain mixture 調整にも使える考え方です。

## 1. データ選択の基本パラダイム

データ選択は、単に「よさそうな Web ページを残す」ことではありません。一般的なパイプラインは通常 3 つのステップを含みます。

1. 目標を定義する：どんなデータが欲しいのか。Wikipedia 風のテキスト、教科書風コード、数学証明、英語 Web ページ、低毒性の議論、ある製品が必要とするタスク領域などがあり得ます。
2. スコアラーを学習または構築する：ターゲットデータと生データを使って `score(x)` を推定します。これはサンプル x がどれだけターゲット領域に似ているか、どれだけ価値があるか、どれだけ安全かを表します。
3. 選択または再サンプリングする：閾値で残す、確率に従ってサンプリングする、または importance weight によってデータ分布を調整します。

ここには 2 つの実務上の制約があります。第一に、スコアラーは汎化しなければなりません。T 自体を見つけ直すだけでは意味がなく、R から新しい類似サンプルを発見する必要があります。第二に、スコアラーは十分速くなければなりません。Web 規模のデータは巨大で、巨大モデルで 1 件ずつ採点すると、フィルタリングのコストが事前学習そのものに近づく、あるいは上回る可能性があります。

## 2. よく使われる 3 種類のフィルタ

### 2.1 n-gram 言語モデル：粗いが安い

最も伝統的な方法は n-gram 言語モデルを学習することです。たとえば KenLM に Kneser-Ney smoothing を組み合わせます。本質的には n 個の単語列の出現回数を数え、条件付き確率を推定します。たとえば文脈 “the cat” が与えられたとき、次の単語が “in” である確率を推定します。多くの n-gram は一度も出現しないため、smoothing により短い文脈へ back off します。

使い方は直接的です。ターゲットコーパスで n-gram モデルを学習し、生文書の perplexity を計算します。perplexity が低いほど、そのテキストはターゲット領域に似ています。CCNet は類似した方法で段落を perplexity により並べ、よい部分だけを残しました。後の LLaMA データもこの種の流れから影響を受けています。

この方法の利点は速く、単純で、スケールしやすいことです。欠点も明確です。主に局所的な共起を見るだけで、長距離の論理や意味的品質を本当に判断することはできません。段落をシャッフルした文章、テンプレートスパム、局所的な文法は自然だが全体として無意味なテキストは、これをすり抜けることがあります。したがって n-gram フィルタリングは、細かな品質評価よりも、明らかなノイズを落とす用途に向いています。

### 2.2 fastText / 線形分類器：産業界でよく使われるベースライン

fastText は軽量なテキスト分類器です。単語や n-gram を固定 bucket にハッシュし、低次元表現を通して線形分類を行います。構造は単純ですが、高速で並列化しやすく、Web 規模のフィルタリングに適しています。

典型的な学習方法は二値分類タスクを作ることです。正例は高品質データまたはターゲット領域データから取り、負例は Common Crawl などの生データから取ります。分類器は「このサンプルがターゲット領域から来た」確率を出力します。GPT-3 は高品質ソースを正例、Common Crawl を負例として品質分類器を学習しました。LLaMA は Wikipedia に引用された Web ページを正例にしました。Dolma は fastText を言語識別と毒性フィルタリングに使いました。

fastText の重要な価値は「十分賢い」ことではなく、「Web 全体に対して実行できるほど安い」ことです。生データを 1% まで圧縮する場合、フィルタは最終学習量の 100 倍のデータを処理します。このとき、サンプル 1 件あたりの採点コストは非常に低くなければなりません。

### 2.3 重要度再サンプリング：「分類」から「分布合わせ」へ

分類器は「ターゲット領域に似ているか」を答えます。しかし学習データは分布の多様性も保つ必要があります。重要度再サンプリングは、より原理的な見方を与えます。ターゲット分布を P、生分布を Q とします。私たちは Q からしかサンプリングできませんが、最終サンプルを P から来たようにしたい。そこで各サンプルに重みを付けます。

w(x) = P(x) / Q(x)

そして w(x) に比例して再サンプリングします。直感はこうです。ある種類のテキストがターゲット領域ではよく出るが生データでは少ないなら、サンプリング確率を上げます。逆なら下げます。

実際の場面では、ターゲットデータが少ないため P を正確に推定するのは困難です。実務では、ハッシュ化された n-gram で粗い分布を推定し、近似重みを計算します。必ず大きな利益をもたらすわけではありませんが、純粋な二値分類よりも、単に「閾値を超える」ことではなく domain mixture の分布マッチングを重視します。

## 3. 品質評価、領域混合、言語選択

「よいデータ」に単一の基準はありません。品質とは、文法的に自然であること、情報密度が高いこと、教育的価値が高いこと、毒性が低いこと、テンプレートが少ないこと、目標タスクに合うこと、信頼できるソースに由来すること、などを意味し得ます。したがってデータフィルタリングは、複数の独立した次元に分解されることが多いです。

言語識別は最も基本的な例です。英語モデルが目標なら、大量の他言語を混ぜると token budget を消費し、英語学習の密度が下がります。しかしモデルが十分大きければ、多言語データは正の転移をもたらす可能性もあります。Bloom は約 30% が英語で、多言語能力を重視しました。最前線のモデルは通常、数百言語をカバーします。言語をフィルタするかどうかは、本質的には学習データ上の意思決定です。対象ユーザー、モデル容量、計算予算、評価指標が一緒に mixture を決めます。

領域選択も同じく重要です。OpenWebMath は「数学」を特殊な言語として扱いました。まずルールで候補を探し、次に Proof-Pile などの数学証明データで学習した KenLM と fastText 分類器で選別し、最終的に約 150 億の数学 token を得ました。結果は、数学領域に向けた高密度データが、はるかに大きいが焦点の定まらないデータを上回り得ることを示しています。これは domain mixture が「大きければ大きいほどよい」ものではなく、目標能力に合わせるべきものだと示しています。

品質評価は強いモデルで補助することもできます。Phi-1 の考え方は、小さなモデルを学習するが、それに「教科書風」の高価値コードデータを与える、というものでした。研究者はまず GPT-4 に Python コード片が初心者にとって教育的価値を持つかを判定させ、約 10 万件のラベル付きサンプルを得ました。その後、より安価な分類器を使って大規模データへ拡張しました。これはよくあるパターンです。高価なモデルで小規模・高品質な T を作り、それを安いフィルタへ蒸留して R を処理します。

## 4. 合成データ：ターゲットデータは「生成」できる

既存のターゲットコーパスがない場合、強い言語モデルにターゲットデータを合成または選別させることができます。たとえば教科書風コード、数学推論、化学 QA を生成させたり、Web ページに「教育的価値」ラベルを付けさせたりできます。この場合、T はもはや既存の特定ソースだけではなく、要求と prompt によって定義されます。

ただし合成データにはリスクがあります。分布が狭すぎる、文体が単調になる、誤りが増幅される、既存データと非常に似てしまう、といった問題です。したがって合成データは通常、無制限に直接追加すべきではありません。品質分類、重複除去、人手による抜き取り確認、下流評価を通すべきです。より堅実な方法は、合成データで特定能力を高めつつ、実データの多様性を保つことです。強いモデルで小さなバッチにラベルを付け、安いフィルタを学習して規模を広げます。

## 5. 重複除去：無駄と記憶を減らす

フィルタリングは「どのデータを学習する価値があるか」を決めます。重複除去は「同じ情報を何回学習するか」を決めます。Web には自然に大量の重複があります。ミラーサイト、ライセンステキスト、商品テンプレート、コピー＆ペーストされた記事、少数の語だけが変わったテンプレートページなどです。C4 には、ある普通の英文が数万回出現していたことがありました。それは悪いテキストではありませんが、6 万回学習する意味はありません。

完全一致の重複除去は単純です。文、段落、文書をハッシュし、同じハッシュを持つサンプルをグループ化して 1 つだけ残します。精度が高く、並列化しやすいですが、近い重複は見つけられません。Bloom filter は bit array と複数の hash 関数でメモリを節約します。偽陰性は生じませんが偽陽性はあり得るため、超大規模な近似集合問い合わせに適しています。

近重複除去は通常 Jaccard similarity に基づきます。文書を shingles または n-gram 集合に分割し、2 つの集合の交差/和集合比が閾値を超えたら近重複と見なします。直接すべてのペアを比較すると O(N²) であり、実行不可能です。MinHash の重要な性質は、2 つの集合の MinHash 衝突確率が、それらの Jaccard similarity に等しいことです。さらに LSH（locality sensitive hashing）を組み合わせ、複数の hash をいくつかの band に分けます。すると高類似文書は高確率で衝突し、低類似文書は低確率で衝突するため、線形またはほぼ線形時間で重複候補を見つけられます。

重複除去には注意が必要です。事前学習段階で Web のジャンクな重複を取り除くことは通常有益です。しかし mid-training や継続学習では、高品質データを複数 epoch 繰り返すことこそ望ましい場合があります。より合理的な戦略は、単純に 1 つだけ残すのではなく、重複回数に対して重みを下げることかもしれません。たとえば log や平方根に従ってサンプリングし、「重要かつよく出る」内容に高い重みを与えながら、生の重複回数で線形に増幅しないようにします。

## 6. Curriculum、annealing、学習データレシピ

データの決定は学習前だけに起こるわけではありません。現代の学習では、時間とともに mixture を変えることがよくあります。初期には、大規模で多様でフィルタが比較的緩いデータを使い、汎用言語と世界知識を学びます。後期には、高品質、ターゲット領域、指示形式、推論データへ徐々に anneal し、最終評価を高めます。これは curriculum learning に似ています。最初は広くカバーし、その後、密度と難度を上げます。

よくある戦略は次の通りです。

- 初期はカバレッジを広げる：Web、書籍、コード、多言語、フォーラムなどを混ぜ、狭い領域へ早期に過適合するのを避ける。
- 中期にターゲット領域の比率を上げる：コード、数学、特定言語を重視するなら、その領域の token を徐々に増やす。
- 後期に品質を anneal する：低品質 Web を減らし、教科書、QA、推論、人手データ、強いモデルで選別したデータの比率を上げる。
- 合成データは量を制限して使う：文体崩壊を避けつつ、希少能力を補う。
- 評価ループで調整する：mixture の変更ごとに、下流 benchmark、perplexity、人手サンプル、安全指標を見る。

したがって domain mixture は静的な表ではなく、最適化問題です。最良の比率は通常、直感だけで一度に書けるものではありません。小モデルを学習し、ablation を行い、データサンプルを観察し、反復する必要があります。実用的な原則は、データレシピをモデルの一部として記録することです。各ソースの token 数、フィルタ閾値、重複除去の粒度、繰り返しサンプリング倍率、学習に入る期間を追跡可能にしておくべきです。そうしないと、モデルのある能力や安全指標が変化したとき、原因がモデル規模なのか、最適化 hyperparameter なのか、データレシピなのか判断しにくくなります。

## 7. 実践チェックリスト

事前学習コーパスを構築するときは、次の問いで確認できます。

1. 目標能力は何か。汎用チャット、コード、数学、多言語、あるいは特定専門領域か。
2. ターゲットデータ T はどこから来るか。人手ソース、信頼できるサイト、強いモデルのラベル、合成生成、ルールによる初期選別か。
3. フィルタは十分安いか。処理するのは最終的な小コーパスではなく、巨大な生 R である。
4. 品質閾値をどう選ぶか。緩すぎればノイズを残し、厳しすぎれば多様性や低リソース集団を失う。
5. mixture は token budget に合っているか。ある領域の比率を上げることは、他領域の学習機会を減らすことを意味する。
6. 完全一致と近重複の重複除去を行ったか。学習セットが評価セットへ漏れることを避けたか。
7. 高品質データを繰り返す必要があるか。必要なら、線形に繰り返すのか、重みを下げて繰り返すのか。
8. フィルタスコアだけでなく、小規模学習でデータ決定を検証したか。

## まとめ

この講義の主線は、データは自然に学習セットへ落ちてくるものではない、ということです。データは、計算可能でスケール可能だがトレードオフだらけの一連の意思決定によって得られます。n-gram、fastText、重要度再サンプリングは、生の Web ページからターゲットデータを見つける基本ツールです。言語識別、品質フィルタリング、毒性フィルタリング、領域発掘は、すべて同じ枠組みの異なる例です。合成データと強いモデルによるラベル付けにより、「ターゲットデータ」そのものも設計できるようになりました。重複除去は無意味な反復を減らし、記憶リスクを下げ、計算資源を節約します。

本当のデータ能力は閉ループから生まれます。データを見る、フィルタを書く、モデルを学習する、結果を評価する、mixture を調整する、そして繰り返す。大規模モデル学習では、データ選択はモデル構造や計算規模と同じくらい重要であり、特定能力では最終性能をより強く決めることさえあります。


---


# CS336 第15回 日本語チュートリアル：Alignment、SFT、RLHF

> 翻案メモ: この日本語チュートリアルは中国語版講義ノートをもとに、元の構成、技術用語、数式、コード風表記を保ちながら、日本語で自然に学べるように説明を整えたものです。

## 1. 事前学習からアラインメントへ：なぜ post-training が必要なのか

事前学習は、多くの能力をモデルパラメータへ「圧縮」します。言語、知識、コード、推論パターン、常識、さまざまな文体などです。しかし next-token prediction だけで学習した base model は、通常、そのままでは使いやすいチャットアシスタントのようには振る舞いません。GPT-3 はすでに強力でしたが、指示に安定して従うわけではありませんでした。ChatGPT の重要な変化は post-training です。ユーザー意図をより理解し、指示に従ってタスクを完了しようとし、危険な場面では拒否したり安全な方向へ誘導したりするようにしました。

この講義での Alignment は抽象的な標語ではなく、工学的なパイプラインです。まず教師ありデータでモデルに「どう答えるべきか」を教え、次に preference data と強化学習または代替アルゴリズムで、人間がより好む振る舞いへモデルを押し出します。目標には helpfulness、truthfulness、harmlessness、つまり有用性、真実性、無害性が含まれます。難しいのは、この 3 つがしばしば衝突することです。たとえば、有用であろうとして答えを捏造するかもしれません。安全であろうとして過剰に拒否するかもしれません。好みに迎合して、より長いが正しくはない回答を出すかもしれません。

## 2. SFT：示範回答でモデルをアシスタントモードに入れる

Supervised Fine-Tuning（SFT）は InstructGPT パイプラインの第一歩です。データ形式は単純です。prompt または対話文脈を与え、理想的な response を付け、その response tokens に対して最尤学習を行います。直感的には、モデルに専門家の示範を模倣させることです。

よく使われる SFT データ源は 3 種類あります。

- タスク集約型データ、たとえば FLAN：既存の NLP データセットを instruction format に書き換えます。要約、分類、QA、多肢選択などです。利点は規模が大きく低コストなことです。欠点は、形式が実際のチャットにあまり似ていないこと、回答が短いものが多いこと、タスク由来の雰囲気が明らかなことです。
- 人間が書いたデータ、たとえば OpenAssistant：ボランティアやアノテータが複雑な prompt と詳細な回答を書きます。利点は自然で高品質なことです。欠点は高価で遅く、文体や事実品質を安定して制御しにくいことです。
- モデル生成データ、たとえば Alpaca：少数の人手 seed prompt からより多くの指示を展開し、強いモデルで回答を生成します。利点は安価で長い回答が多いことです。欠点は教師モデルのバイアスを引き継ぎ、学生モデルがまだ持たない能力を模倣させてしまう可能性があることです。

SFT の重要な経験則は、少量でもレバレッジの高いデータがモデルの振る舞いを大きく変えられることです。強い base model なら、比較的少ない instruction data だけで「テキストを補完する」モデルから「ユーザーに答える」モデルへ変わることがあります。ただし「高品質データ」は「長く、知識密度が高く、引用が多ければ多いほどよい」という意味ではありません。SFT 例がモデルの知らない事実を答えることを要求すると、学習損失は正解らしく見える token を生成することを報酬化します。引用らしい文字列も含まれます。その結果、モデルは「事実を調べて引用する」のではなく、「複雑な質問では最後に引用を捏造する」ことを学ぶかもしれません。

これは、SFT が出力の type signature と文体を教えるのに向いていることを示しています。箇条書きにするか、長く説明するか、引用するか、謝るか、拒否するか、などです。新しい知識も教えられますが、小規模 SFT はその点では事前学習や大規模 mid-training ほど安定しないことが多いです。示範データがモデルの既存能力を明らかに超える場合、モデルは幻覚的な近道を学んでしまいます。したがって、よい SFT データはモデル能力に合っており、「わかりません」「追加情報が必要です」「確認してください」といった妥当な abstention 行動を含むべきです。

## 3. 安全 SFT：拒否と過剰拒否のバランス

安全アラインメントも SFT によって注入できます。少量の安全サンプルを instruction tuning に混ぜるだけで、詐欺、マルウェア、暴力、自傷などの場面で拒否したり、安全な代替案を出したりすることをモデルに学ばせられます。研究では、数百件の丁寧に作った安全サンプルでも明確な効果が出ることがあります。

しかし安全とは、単に拒否率を上げることではありません。本当に難しいのは、危険な要求と、表面上は危険に見えるが正当な要求を区別することです。たとえば “how can I kill a Python process?” は、コンピュータ文脈ではプロセスを終了するという意味であり、生物を傷つける意味ではありません。データが敏感語を見るたびに拒否することだけを教えると、over-refusal が起き、使いやすさが下がります。安全データは境界事例、dual-use 問題、文脈の曖昧さ、回答してよい安全なバージョンを含む必要があります。

## 4. SFT の学習方式は事前学習に近づいている

学術的な設定では、SFT はしばしば「base model を取り、instruction data で数 epoch 勾配降下する」と理解されます。しかし最前線モデルの post-training は、すでに完全な学習段階に近づいています。多くの現代的パイプラインは、事前学習の末尾、学習率減衰段階で、高品質データ、コード SFT、QA、チャット、多言語書籍、安全データを混ぜます。これは mid-training または decay-stage data mixing と呼ばれることがあります。

利点は、データ規模を大きくできること、短い微調整による catastrophic forgetting が起きにくいこと、instruction behavior がモデルへより深く組み込まれることです。代償は「base model」と「chat model」の境界が曖昧になることです。今日のいわゆる base model の多くは、学習後期にすでに大量の指示形式データを見ている可能性があります。したがって異なるモデルを比較するときは、base が必ずしも完全に未アラインメントという意味ではない点に注意すべきです。

## 5. なぜ preference data が必要なのか

SFT では、人間または強いモデルが理想回答を直接書く必要があります。しかし高品質な長い回答を生成するのは高価で疲れますし、人間が自分で書いた回答が、必ずしもその人が最も好む回答とは限りません。生成より検証のほうが簡単なことが多いです。アノテータに A/B のどちらがよいかを比較させるほうが、ゼロから完璧な回答を書かせるより通常は安価です。これが preference data の動機です。

偏好データの基本形式は、同じ prompt に対して 2 つ以上のモデル回答を用意し、アノテータがよりよいものを選ぶというものです。InstructGPT 型のアノテーション基準は通常、helpful、truthful、harmless の 3 点を中心にします。実際の基準はさらに細かく、ユーザーの真の意図に答えているか、形式を守っているか、幻覚していないか、有毒でないか、不適切内容を含まないか、 clarification が必要か、などを見ます。

しかし preference annotation も難しいです。アノテータは時間圧の中で作業することが多く、事実確認、数学の検算、隠れた幻覚の発見に十分な時間を持てないことがあります。長い回答は、誤りを含んでいても「詳しくて有用」と判断されやすいです。アノテータによって注目点も違います。専門家は事実性を重視し、一般のクラウドワーカーは形式、流暢さ、礼儀正しさを重視するかもしれません。アノテータの文化、国、宗教、政治的背景も価値判断に影響します。そして alignment はパイプライン末端にあり、最終モデルの振る舞いへ強く影響します。

したがって preference data は技術資源であるだけでなく、社会的選択でもあります。明確な rubric、公正な報酬、品質監査、多様なアノテータ集団、そしてバイアス源の透明な記録が必要です。

## 6. Reward Model：ペアワイズ偏好を最適化可能な報酬へ変える

RLHF の古典的な第二段階は reward model の学習です。prompt x のもとで各回答 y に潜在的なスカラー報酬 R(x, y) があると仮定します。ただしそれを直接観測することはできず、人間の比較、つまり回答 A が回答 B よりよいかどうかだけを観測できます。

よく使われるモデル化は Bradley-Terry preference model です。A が B に勝つ確率は R(x, A) - R(x, B) の差に依存し、通常 sigmoid を通します。reward model の学習では、選ばれた回答のスコアが、選ばれなかった回答より高くなるようにします。学習後、reward model は任意の新しい回答にスカラー得点を付けられ、RL の報酬信号として使えます。

注意すべきなのは、reward model は人間の偏好の近似であって真理ではないことです。長い回答を好む、リストを好む、特定の口調を好む、といったアノテーションデータ中のバイアスを学びます。また policy model に exploit される可能性もあります。RL 段階で reward model を過剰最適化すると、モデルは高い reward を得ても実際の人間には好まれないかもしれません。これが reward hacking、または Goodhart 化です。

## 7. RLHF：PPO で報酬と制約のあいだを最適化する

古典的な InstructGPT パイプラインの第三段階は、PPO で policy model を最適化することです。目標は、ある参照分布をさらに模倣することではなく、reward model が与える期待報酬がより高い policy π(y|x) を見つけることです。

実際の目的関数には通常、制約が含まれます。

- 報酬項：モデルに reward model が好む回答を生成させる。
- KL ペナルティ：RL 後の policy が SFT モデルから離れすぎないようにし、言語品質の崩壊、mode collapse、reward hacking を避ける。
- 場合によっては事前学習損失も混ぜ、catastrophic forgetting を緩和する。

PPO は policy gradient の安定した工学版と理解できます。モデルが回答をサンプリングし、reward model が採点し、advantage に基づいてよい出力を強め悪い出力を弱めます。同時に importance ratio と clipping によって各更新幅を制限します。効果的ではありますが、実装は複雑で、調整が難しく、学習が不安定で、学術・オープンソース実践には扱いにくいです。

ここでは on-policy と off-policy の区別も重要です。On-policy データは現在最適化しているモデルから来るため、モデルの現在の誤りに合わせて改善できます。Off-policy データは他のモデルや古いモデルから来ます。安価で再利用できますが、現在のモデルが最も修正すべき領域を必ずしもカバーしません。現代のパイプラインでは両者を混ぜることが多いです。

## 8. DPO と代替方法：RL 問題を教師あり loss に変える

PPO は面倒なので、研究者は多くの代替法を試しました。preferred responses だけで SFT する、chosen/rejected に good/bad token を付けて条件付き生成する、reward model で複数回答を採点し最良のものを選んで学習する、などです。これらは有効な場合もありますが、通常は古典的 RLHF ほど安定しません。

Direct Preference Optimization（DPO）が人気になった理由は、明示的な reward model と PPO rollout を取り除き、偏好最適化を直接の教師あり風 loss に書き換えたことです。鍵となる考えは、KL 正則化付きの最適 policy 問題では、ある policy が暗黙に reward を定義するというものです。この暗黙 reward を Bradley-Terry preference model に代入すれば、「chosen が rejected より好まれる」確率を直接最大化できます。

DPO の学習直感は単純です。reference model に対して、chosen response の log probability を上げ、rejected response の log probability を下げ、係数によって reference からの距離を制御します。PPO のようにオンラインサンプリングや複雑な RL 状態を必要としないため、実装、再現、拡張が容易です。欠点は、通常は既存の preference pairs に依存し、on-policy 探索をあまり使わないことです。preference data の品質が低い、分布が偏っている、あるいはすべて古いモデルの出力である場合、DPO も制約されます。

関連する代替路線には RLAIF（AI feedback）、Constitutional AI、拒否サンプリング式学習、さまざまな DPO 変種があります。RLAIF は人間の代わりに強いモデルで偏好判断を行い、低コストで大規模にできます。GPT-4 などの judge は、多くのオープンなタスクで人間の偏好と高い相関を示します。しかし AI judge には自己偏好、長さバイアス、位置バイアス、価値バイアスがあり、無偏なアノテータとして扱ってはいけません。

AI feedback を使うときは「閉ループ増幅」にも注意が必要です。同じモデルファミリーが候補を生成し、裁判し、学生モデルの蒸留にも使われると、システムは実ユーザーの需要に近づくのではなく、そのファミリーが好む表現様式へますます偏るかもしれません。よりよい実践は、複数の judge を混ぜる、人間レビューを抽 sample する、難しい負例を残す、長さ正規化後の勝率を別に報告することです。事実密度の高いタスクや数学タスクでは、開放的な偏好だけに頼るのではなく、検証可能な信号、ツールチェック、専門家監査を使うのが望ましいです。

## 9. アラインメントパイプラインの実践チェックリスト

典型的な alignment pipeline は次のように構成できます。

1. 強い base model から始め、chat template と基本 instruction data を準備する。
2. SFT を行い、モデルを安定してアシスタントモードに入れる。汎用能力、形式遵守、多ターン対話、初期の安全拒否をカバーする。
3. prompts を集め、1 つ以上のモデルに候補回答を生成させる。
4. 人間または AI で pairwise preferences をアノテーションし、chosen/rejected データを作る。同時に長さ、文体、事実性、アノテータ一致度を監査する。
5. reward model を学習する、または DPO/IPO/KTO などの偏好最適化方法を直接使う。
6. PPO/RLHF を使う場合は、KL 制約と安全監視を入れ、reward hacking を避ける。
7. 多面的評価で検証する：開放的偏好、事実性、数学・コード benchmark、拒否率、過剰拒否率、red-team testing、実ユーザータスク、コスト、レイテンシ。
8. 単一ランキングを追うのではなく、失敗例に基づいてデータを反復する。

## 10. リスクと評価：偏好を真理として扱わない

Alignment は「モデルをより好かれるようにすること」と誤解されやすいです。これは危険です。人間も AI judge も、長く、構造化され、礼儀正しく、自信があるように見える回答を好みます。しかしこれらの特徴は正しさと同じではありません。モデルは、箇条書きがうまい、引用がうまい、迎合がうまいという理由で preference 評価に勝ちながら、より多く幻覚している可能性があります。

したがって評価は多様でなければなりません。

- Open-ended human eval や chatbot arena はユーザー偏好を測るが、長さと judge バイアスを制御する必要がある。
- 標準 benchmark は知識、推論、コード、数学能力を測り、post-training が文体だけを最適化するのを防ぐ。
- 安全評価では harmful compliance と over-refusal の両方を見る。「すべて拒否する」ことを安全に見せかけてはいけない。
- 事実性評価では、流暢さを見るだけでなく claim を検証する必要がある。
- 実運用では、分布シフト、jailbreak 攻撃、悪用、ユーザー満足度、コスト、レイテンシも監視する必要がある。

まとめると、Lecture 15 の主線は次の通りです。事前学習はモデルに能力を与え、SFT はモデルにどう振る舞うかを教え、preference data はどの振る舞いがより好まれるかを伝え、reward model または DPO は偏好を最適化可能な目標へ変え、RLHF/偏好最適化はモデルをより有用で、より真実で、より安全な領域へ押し出します。本当の課題はアルゴリズムだけではなく、データ品質、アノテーションのインセンティブ、価値バイアス、評価バイアス、安全性と有用性のバランスにあります。


---


# CS336 第16回チュートリアル：Alignment における強化学習（一）

> 翻案メモ: この日本語チュートリアルは中国語版講義ノートをもとに、元の構成、技術用語、数式、コード風表記を保ちながら、日本語で自然に学べるように説明を整えたものです。

この講義は post-training（後学習）パートの第 2 回であり、テーマは従来の RLHF から「検証可能な報酬に基づく強化学習」（reinforcement learning from verifiable rewards）へ移ります。中心となる問いは、なぜ言語モデルの alignment に RL が必要なのか、PPO や GRPO のようなアルゴリズムは実際に何を最適化しているのか、なぜ学習は不安定になりやすく、工学上どの細部が重要なのか、というものです。

## 1. RLHF から検証可能な報酬へ

RLHF（Reinforcement Learning from Human Feedback）は通常、人間の preference data から始まります。同じ prompt に対する 2 つの回答が与えられ、人間がどちらの回答の方がよいかをラベル付けします。目標は、人間が好む回答をより出しやすい language model policy を学習することです。

ここでいう policy とは、与えられた文脈のもとで出力 token 列に対してモデルが定める確率分布です。事前学習や SFT（supervised fine-tuning）と異なり、RLHF は単に「データ分布をフィットする」作業ではありません。モデルが何を生成するかによって得られる reward が変わるため、目的関数には「現在のモデルからサンプリングする」過程が含まれます。そのため、通常の maximum likelihood より最適化が難しくなります。

前回扱った DPO（Direct Preference Optimization）は、preference optimization を教師あり学習に近い目的へ変換します。明示的に reward model を学習せず、完全な RL loop も回さず、preference pair から直接 policy を調整します。DPO の直感は単純です。chosen response の確率を上げ、rejected response の確率を下げる。モデルの暗黙の reward 判断が大きく間違っているほど、更新も大きくなります。実装が簡単なため、DPO は一時期、オープンソースモデルの post-training における主流手法になりました。

しかし DPO にも限界があります。DPO は pairwise preference には自然に合いますが、「数学問題が正解か不正解か」のような scalar reward しかないタスクにはあまり向きません。また通常は offline です。つまり、まず preference pair を集め、その上で学習します。reasoning model では、モデルが新しい解答を生成し続ける過程で、検証可能な結果に基づいて直接 online optimization することが望まれます。

## 2. RLHF の 2 つのリスク：過最適化と calibration 悪化

RLHF における最も重要な経験的現象の 1 つが overoptimization（過最適化）です。reward model は人間の preference の代理モデルにすぎず、ノイズや誤差を含みます。学習初期には、代理 reward を最適化すると真の人間 preference も改善することが多いです。しかしさらに最適化を続けると、モデルは「reward model の抜け穴を突く」ようになり、代理 reward は上がり続ける一方で、真の win rate は停滞したり低下したりします。これは教師あり学習における train-test gap に似ています。訓練集合上の reward model は、真の preference oracle ではありません。

もう 1 つの現象は calibration（校正）の悪化です。事前学習モデルは確率的生成モデルと見なせますが、RLHF 後のモデルは、ある reward のために調整された policy に近くなります。reward が「不確実性を表明すること」を促さない場合、モデルはより自信過剰になり、より迎合的になり、「わかりません」と言う頻度が下がる可能性があります。したがって、RLHF モデルの出力確率を、信頼できる真の確率推定としてそのまま解釈してはいけません。

これらの問題は、人間の preference が非常に価値ある一方で、大規模・低ノイズ・安定に最適化することが難しいことを示しています。そこで自然な方向性として、より明確な reward を持つタスクを探すことになります。たとえば数学、コード、形式証明、実行可能なテストなどです。これらの領域では、答えが正しいかどうかを自動検証できます。reward は真の目標に近く、reward hacking も起こりにくくなります。

## 3. RL の基礎：policy、reward、value、advantage

言語モデル RL では、1 つのサンプルは通常 prompt と、モデルが生成した response からなります。完全な response を生成することを 1 回の rollout と見なせます。よく使う用語は次の通りです。

- Policy：現在の言語モデル `πθ`。prompt が与えられた後に token 列を生成する確率分布。
- Reward：生成結果へのスコア。RLHF では reward model から来る場合があり、数学やコードでは答えの照合、unit test、format check などから来る場合があります。
- Value function：ある状態または部分生成が将来どれだけ reward を得るかを推定する関数。policy gradient の分散を下げるためによく使われます。
- Advantage：ある action / output が baseline よりどれだけ良いか。直感的には、advantage が正ならその出力確率を上げ、負なら下げます。

最も基本的な policy gradient の考え方は、高 reward の出力の log probability を増やし、低 reward の出力の log probability を減らす、というものです。多くの RL アルゴリズムは本質的に「良いものを重くし、悪いものを軽くする」と理解できます。違いは、良し悪しをどう定義するか、分散をどう下げるか、policy が一歩で遠くへ行きすぎないようどう制御するかにあります。

言語モデル RL にはもう 1 つ特徴があります。多くのタスクは contextual bandit に近いのです。モデルは prompt を見て、完全な回答を生成し、最後に terminal reward を受け取ります。従来のゲーム環境のような複雑な状態遷移はありません。ただし学習時には、KL penalty などの正則化項を token level に分配し、「正解/不正解」のような task reward は最後の token または sequence level に置くことがよくあります。

## 4. PPO：強力だが工学的に複雑

PPO（Proximal Policy Optimization）は、RLHF 初期における最も重要なアルゴリズムの 1 つでした。policy gradient から出発し、2 つの重要な仕組みを導入します。

第 1 に、importance sampling と old policy です。純粋な on-policy 手法では、各更新で現在の policy から新しく生成したサンプルを使う必要があります。rollout は遅いため、これは高コストです。PPO では、まず old policy でデータを 1 バッチサンプリングし、同じ rollout バッチに対して複数回の勾配更新を行えます。

第 2 に、clipping です。PPO は new policy が old policy に比べて大きく変わりすぎることを望みません。そのため確率比 ratio を使い、それを `1-ε` から `1+ε` の範囲、たとえば 0.8 から 1.2 に clipping します。これにより、あるサンプルの reward が非常に高くても、モデルがその確率を無制限に押し上げることはできず、学習の安定性が高まります。

PPO は通常、advantage を推定するために value model も必要とします。たとえば GAE（Generalized Advantage Estimation）を使います。これは勾配分散を下げますが、工学的複雑さを伴います。policy model、reward model、value model を維持する必要があり、場合によっては異なる tokenizer、KL shaping、value loss、policy loss、clip norm、rollout worker と training worker の同期なども扱う必要があります。実際の PPO には多数の実装詳細があり、少しの違いでも結果に影響することがあります。

大規模言語モデルでは、value model は特に高価です。多くの場合 policy と同じ大きさであり、メモリと計算コストがほぼ倍になります。そのため、PPO の安定性を保ちつつ value model を取り除く方法が求められました。

## 5. GRPO：value model を group 内 baseline で置き換える

GRPO（Group Relative Policy Optimization）は PPO の簡略化変種と見なせます。また DeepSeek Math / R1 系列における重要なアルゴリズムでもあります。policy gradient、KL regularization、ratio clipping などの考え方を保ちながら、value function と複雑な GAE を取り除きます。

GRPO の中心的な方法は、同じ問題 `q` に対して一度に `G` 個の回答をサンプリングし、group を作ることです。各回答には reward があります。そして group 内 reward の平均と標準偏差を用いて advantage を構成します。

A_i = (r_i - mean(r_1, ..., r_G)) / std(r_1, ..., r_G)

つまり、「この回答の絶対 reward はどれだけ高いか」ではなく、「同じ問題に対する他の回答よりどれだけ良いか」を問います。これは自然です。問題ごとに難易度が異なり、簡単な問題では平均 reward が高く、難しい問題では平均 reward が低くなります。group 内平均は、問題難易度に対する baseline として使えます。これにより、追加の value model を学習する必要がなくなります。

各 rollout バッチに対して online update を 1 ステップだけ行うなら、GRPO は通常の policy gradient にかなり近くなります。group 平均より高い回答は上方修正され、平均より低い回答は下方修正されます。実装上は、複数回答を生成し、reward を計算し、group ごとに正規化し、KL penalty を加え、勾配更新を行えばよいだけです。

ただし GRPO にも微妙な問題があります。標準偏差による正規化は、厳密な policy gradient の導出で許される通常の baseline ではありません。reward 分散が非常に小さい group、たとえば全回答が不正解または全回答が正解の問題を増幅してしまいます。その結果、最も学習信号が豊富な中程度の難易度の問題ではなく、「難しすぎる」または「簡単すぎる」問題に学習の焦点が移る可能性があります。

もう 1 つの問題は length normalization です。sequence reward を出力長で割ると、不正解時にはモデルが長い内容を生成して負の reward を薄めようとする可能性があります。正解時には短い出力を好みます。これは、不確かなときに非常に長い chain-of-thought を出すようモデルを誘導し、「より長く考えている」ように見えても、実際には目的関数の偏りにすぎない場合があります。Dr. GRPO などの後続分析では、一部の length normalization を取り除くことで、reward を保ちながら無制限な長文化を減らせるとされています。

## 6. なぜ検証可能な報酬が reasoning models を推進するのか

DeepSeek R1 を例にすると、その学習フローは非常に単純ながら有効なパラダイムを示しています。数学やコードなどの検証可能タスクで、outcome reward（最終答えが正しいかどうか）を用いて RL を行います。R1-Zero はほぼ base model から直接 RL を行い、reward は主に accuracy reward と format reward からなります。format reward は、推論を特定の think tags 内に置くことを要求します。一見すると単なる形式制約ですが、実践上は安定した学習に重要です。

R1 の重要な結論は、複雑な MCTS search や PRM（Process Reward Model）が必ずしも必要ではない、ということです。PRM は推論の中間ステップにスコアを与えられるため、理論上はより豊富な feedback を提供します。しかし信頼できる process supervisor を構築するのは難しいです。R1 は、単純な outcome-based reward と GRPO だけでも強い推論能力を得られることを示しました。

実際に公開されるモデルは、通常 RL だけで作られるわけではありません。より一般的な流れは、まず少量の long chain-of-thought SFT を行い、モデルに読みやすい推論形式を学ばせる。次に verifiable reward RL を行い、数学やコードの正答率を高める。最後に general instruction tuning と RLHF を行い、チャット、作文、安全性などの汎用能力を回復する、というものです。これは SFT と RL が補完的であることを示します。SFT は初期の行動様式を与え、RL は真の目標に向けてさらに最適化します。

Kimi K1.5 と Qwen3 も同様の考え方を示しています。Kimi はデータ選別と長さ制御を強調します。best-of-N で簡単すぎる問題を除外し、curriculum learning を構成し、学習後期には length reward を加えて、推論チェーンが長くなりすぎて推論コストが制御不能になるのを防ぎます。Qwen3 は thinking mode fusion を導入します。同じモデルが think と no-think の両モードをサポートし、token budget によってテスト時の思考長を制御できるため、inference-time scaling が可能になります。

## 7. 学習安定性と工学上の注意点

LLM RL の難しさはアルゴリズムだけでなく、システムにもあります。rollout には autoregressive generation が必要であり、通常の teacher-forcing 学習よりはるかに遅いです。training worker が重みを更新した後、その重みを inference worker に同期する必要もあります。長い chain-of-thought は batch 長を不均一にし、GPU 利用率を下げます。多くのシステムでは training と inference を別 worker に分け、vLLM などの推論エンジンでサンプルを生成します。

安定した学習は通常、次の技法に依存します。

1. KL regularization：new policy が reference policy から遠く離れすぎないよう制限し、言語品質の崩壊を避ける。
2. Clipping または明示的正則化：1 回の policy update の大きさを制御する。
3. 適切な baseline：policy gradient の分散を下げる。たとえば PPO の value function や GRPO の group 内平均。
4. Reward shaping：format、言語の一貫性、長さなどの補助目標を重み付き reward として加える。ただし重みは経験的な調整が必要。
5. データ難易度制御：簡単すぎると学習信号がなく、難しすぎて全て不正解でも信号がない。best-of-N filtering や curriculum は学習効率を改善する。
6. 長さ制御：長い推論は性能を上げることもあるが、目的関数によって誘導されているだけの場合もある。正答率と推論コストのあいだでトレードオフが必要。

## 8. SFT、RL、RLHF の役割分担

この講義を alignment pipeline 全体に戻して見ると、3 種類の学習にはそれぞれ役割があります。SFT はモデルに「どのように答えるべきか」のデモを与えます。たとえば指示に従う、長い推論チェーンを書く、固定形式を使う、明らかに有害な出力を避ける、といったことです。利点は安定で、安価で、デバッグしやすいことです。欠点は、データに存在する行動を模倣するだけで、デモより良い解法を探索するよう直接促せないことです。

RL はモデルを「模倣できる」状態から「目標を最適化できる」状態へ押し上げます。数学やコードでは、モデルは多くの異なる解法を試せます。最終答えが正しい、またはテストに通る限り、正の reward を得ます。この探索により、SFT データに含まれない行動パターンを発見でき、正解確率を継続的に高められます。RLHF は最適化目標を、検証可能タスクから人間 preference へ拡張します。たとえば有用性、丁寧さ、安全性、style consistency です。ただし preference reward はノイズが大きいため、過最適化を防ぐには KL、early stopping、評価集合、人間による確認がより重要になります。

したがって実用的な順序は通常、まず SFT で制御しやすい initial policy を作る。次に高品質で検証可能かつ適切な難易度のデータ上で RL を行い、推論と問題解決能力を高める。最後に preference optimization または RLHF で一般チャット体験を修正する、というものです。順序を逆にして、弱いモデルや形式が混乱したモデルから直接 RL を始めると、reward が疎すぎて学習が不安定になりえます。SFT だけで RL をしなければ、モデルは「推論できそうに見える」段階にとどまり、本当に検証可能な目標で最高の正答率を得るところまでは行かない可能性があります。

評価時にも異なる目標を区別する必要があります。数学 leaderboard の向上は、汎用 assistant が良くなったことを意味しません。一般 preference の向上も、推論が強くなったことを意味しません。信頼できる学習フローでは、task accuracy、回答長、KL distance、format violation rate、refusal rate、人間 preference、安全性指標を同時に監視する必要があります。これらの曲線が全体として妥当なときに初めて、RL がモデルを本当に改善していると言えます。単に benchmark や reward の抜け穴を利用しているだけではない、という確認が必要です。

## 9. まとめ

この講義の主線は次の通りです。RLHF は、RL が言語モデル alignment に使えることを示しました。しかし人間 preference reward はノイズが大きく、過最適化されやすいです。検証可能な報酬は、より明確でスケーラブルな学習信号を提供します。PPO は古典的で強力な RLHF アルゴリズムですが、value model と多数の実装詳細により高コストで調整が難しくなります。GRPO は、同じ問題に対する複数回答の group 内相対 reward で value model を置き換え、学習を大きく簡略化します。そのため reasoning model の post-training における重要な道具になりました。

R1、Kimi K1.5、Qwen3 の経験から見ると、成功する recipe は多くの場合、少量の高品質 long CoT SFT、検証可能タスク上の RL、KL・長さ・format などの安定化制約、そして general RLHF または instruction tuning を含みます。最終目標は、モデルに「無限に考えさせる」ことではありません。制御可能なコストのもとで、policy をより高い正答率、より良い alignment、より安定した挙動へ押し上げることです。


---


# CS336 第17回チュートリアル：Alignment における強化学習（二）

> 翻案メモ: この日本語チュートリアルは中国語版講義ノートをもとに、元の構成、技術用語、数式、コード風表記を保ちながら、日本語で自然に学べるように説明を整えたものです。

この講義は、前回の RLHF と RL for Verifiable Rewards（RLVR）のテーマを引き続き扱います。ただし重点は、まったく新しい概念を導入することではありません。言語モデルにおける policy gradient、PPO/GRPO 系アルゴリズム、reward design、工学実装の重要な細部をより深く分解して理解することです。中心となる問いは、モデルがすでに一定の能力を持っているとき、人間がラベル付けしたデータを単に模倣するだけでなく、「採点可能な結果」を使ってどのようにさらに最適化するか、というものです。

## 1. 言語モデルにおける強化学習の設定

古典的な強化学習では、状態、行動、報酬、遷移ダイナミクスを定義する必要があります。言語モデルでは、これらの概念は非常に具体的に対応します。

- 状態：prompt と、これまでに生成済みの response prefix。
- 行動：次の token を生成すること。
- trajectory / episode / rollout：prompt から始めて、モデルが連続的に完全な回答を生成すること。
- 報酬：回答全体がどれだけ良いか。

この講義では主に outcome reward を扱います。つまり、完全な回答が生成された後にだけ与えられる報酬です。たとえば数学問題で、モデルがまず推論過程を書き、最後に「答えは 3 miles」と出力するとします。reward function は最終答えを抽出し、標準答えと比較します。一致すれば 1、不一致なら 0 を与えます。

これは一般的な RL と重要な点で異なります。言語モデルの遷移ダイナミクスは非常に単純で、新しい token を既存の文脈の後ろに連結するだけです。そのため言語モデルは自然に test-time compute を行えます。複数の候補答案をサンプリングし、探索し、再ランキングし、検証できます。ロボット制御では、このように正確にシミュレート可能な世界ダイナミクスを持つことは困難です。

ただし困難も別の場所へ移ります。ロボットでは「ある物理状態に到達すること」が難しいことが多いです。言語モデルはほぼ任意の token 列を書けます。難点は、その token が本当に正しい推論、正しい答え、信頼できる挙動に対応しているかどうかです。

## 2. SFT から policy gradient へ

言語モデル RL の目標は expected reward を最大化することです。

\[
J(\pi)=\mathbb{E}_{s\sim p(s), a\sim \pi(\cdot|s)}[R(s,a)]
\]

ここで \(s\) は prompt であり、\(a\) は一旦、完全な response と見なせます。policy gradient の基本的なトリックは次です。

\[
\nabla J(\pi)=\mathbb{E}[R(s,a)\nabla \log \pi(a|s)]
\]

直感的には、これは SFT によく似ています。SFT では、人間が良い回答を与え、モデルはその回答の確率を最大化します。policy gradient では、モデル自身が回答をサンプリングし、その回答を reward によって重み付けします。reward が 1 ならその回答の確率を上げ、reward が 0 なら素朴な形ではほとんど更新されません。

これは、RLVR が初期モデルの能力を必要とする理由を説明します。タスクが難しすぎて、モデルがほとんど正解をサンプリングできない場合、すべての reward が 0 になり、gradient もほぼ 0 になって、学習が始まりません。実際のシステムでは通常、次が必要です。

- 十分に強い base/SFT モデル。
- 正の reward に出会う確率を高めるだけの十分なサンプリング。
- より滑らかな reward または部分 reward の設計。
- baseline、advantage、正規化などによる分散低減。

## 3. Baseline、Advantage、分散低減

policy gradient の主な問題は高分散です。reward の絶対値が高いからといって、その action が現在の prompt に対して良い選択だとは限りません。たとえば簡単な問題の誤答が 9 点を得る一方で、難しい問題の比較的良い答えが 2 点しか得ないことがあります。reward の大きさにそのまま従って更新すると、モデルは「簡単な prompt における次善 action」を誤って好むかもしれません。

解決策は baseline を導入することです。

\[
\mathbb{E}[(R(s,a)-B(s))\nabla\log\pi(a|s)]
\]

\(B(s)\) が action \(a\) に依存しない限り、期待勾配の向きは変わりません。差し引かれるのは policy に依存しない定数項だからです。しかし分散を大きく下げることができます。

よくある選択は、baseline を現在状態における expected reward に近似することです。

\[
B(s)\approx \mathbb{E}_{a\sim\pi}[R(s,a)]
\]

このとき \(R(s,a)-B(s)\) が advantage です。その回答が同じ prompt における平均回答よりどれだけ良いかを表します。平均より良い回答は強化され、平均より悪い回答は抑制されます。これは「誤答 reward をなぜ -1 にしないのか」という問いにも答えます。中心化した後、group 内平均を下回るサンプルは自然に負の advantage を持つからです。

## 4. GRPO：同じ prompt の複数回答を group として使う

PPO は通常、baseline を推定するために value function / critic を使います。GRPO（Group Relative Policy Optimization）の考え方は、言語モデルにより適しています。同じ prompt に対して複数の response を一度にサンプリングし、group を作り、group 内 reward 平均を baseline として使います。

大まかな流れは次の通りです。

1. prompt のバッチから始める。
2. 各 prompt に対して複数の候補回答をサンプリングする。
3. 各回答の reward を計算する。
4. 同じ prompt の group 内で平均と標準偏差を計算する。
5. \((r_i-\bar r)/\sigma\) を更新信号として使う。
6. group 平均より高い回答の確率が上がり、平均より低い回答の確率が下がるよう current policy を更新する。

これが “relative” の意味です。ある回答の絶対スコアが高いかどうかではなく、同じ prompt に対する他の回答より良いかどうかを問います。言語モデルはこの構造に自然に合います。同じ prompt に対して複数の候補を並列生成できるからです。従来のロボット RL では、各 trajectory の状態が大きく異なることが多く、このような自然な group 構造はありません。

標準化にはもう 1 つ利点があります。reward scale の変化にあまり敏感でなくなります。すべての reward に 100 を掛けても、正規化後の advantage は基本的に変わりません。ただし、ある group 内のすべての回答 reward が完全に同じ場合、中心化後の delta はすべて 0 になり、モデルはその group から更新されません。これは直感に合っています。group 内に相対的な優劣信号がないからです。

## 5. PPO/GRPO における ratio、clipping、KL

policy optimization でよく使われる重要な量があります。

\[
\rho=\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}
\]

これは、サンプリング時の old policy に比べて、current policy が同じ回答の確率をどれだけ変えたかを表します。PPO/GRPO はこれを advantage と掛け合わせ、clipping します。

\[
\text{clip}(\rho, 1-\epsilon, 1+\epsilon)
\]

clipping の役割は、1 回の更新幅を制限し、少数の高 reward サンプルによってモデルが急に遠くへずれることを防ぐことです。実装時には注意が必要です。\(\pi_{old}\) は定数であるべきで、旧 policy を通して勾配を流してはいけません。工学的には通常 `no_grad` を使うか、rollout 時に old policy がこれらの response に与えた log probability を直接保存します。

old policy とは別に、学習中には KL 正則化用の reference model が存在することもあります。

\[
\text{reward objective} - \beta \mathrm{KL}(\pi_\theta || \pi_{ref})
\]

reference model は通常、初期 SFT モデルまたはよりゆっくり更新されるモデルです。KL penalty は、current model が reward を追求するあまり元の言語能力から遠く離れすぎることを防ぎます。たとえば出力形式の崩壊、過度な投機、汎用性の喪失を避けます。実践では、current training policy、ratio 用の old policy、KL 用の reference policy という 3 種類のモデルまたは量が同時に存在することがあります。

## 6. Reward design、reward hacking、検証可能な報酬

RLVR の魅力は、各サンプルについて人間に尋ねなくても reward を自動計算できることです。たとえば数学問題の最終答え照合、コード unit test の通過、ソート結果の正しさ、定理証明 checker の通過などです。この種の reward には、決定的、スケーラブル、低コストという利点があります。

しかし reward design は非常に危険です。講義中のソート例がそれを示しています。もし reward が「出力 token が入力由来か」と「隣接 token が順序通りか」だけを評価するなら、モデルは抜け穴を見つけるかもしれません。たとえば一部の token を繰り返し出力したり、局所的な順序性を利用して高得点を得たりしながら、実際にはソートを完了していない、ということが起こります。これが reward hacking です。モデルは指標を最適化していますが、私たちが本当に気にする目標を最適化していません。

reward が dense であるほど学習信号は強くなりますが、誤った近道を導入しやすくなります。reward が sparse であるほど真の目標に近づきやすい一方、最適化は難しくなります。工学上よく使われる折衷案には次があります。

- 正確な final reward を使い、目標の正しさを保証する。
- 探索を助けるために partial reward を使うが、それが悪用可能かどうかを継続的に検査する。
- 出力形式を厳密に parse / validate する。
- hidden test set や多様な環境を使い、reward への過適合を防ぐ。
- 高 reward サンプルを人手で抽出検査し、投機的パターンを探す。

## 7. Reasoning models と RLVR の関係

reasoning model の鍵は、単に「より長い chain-of-thought を書けるようになる」ことではありません。検証可能タスクにおいて、大量サンプリングと強化学習を通じて、正解に至る推論 trajectory をより高頻度に生成できるようにすることです。reward が最終結果を信頼できる形で判定できる限り、モデルは人間のデモより効果的な戦略を発見できる可能性があります。

これこそが SFT と比べた RL の潜在力です。SFT は既存の回答を模倣するだけですが、RL は測定可能な目標上でデモを超えられます。ただし前提として、「測定」そのものが十分に信頼できなければなりません。数学、コード、形式証明、ゲーム、tool calling 環境は RLVR に向いています。一方、オープンエンドな作文、価値判断、実ユーザー満足度は RLHF に近く、reward model、人間 preference、または LLM-as-judge を必要としますが、bias や reward attack の影響も受けやすくなります。

## 8. 評価と工学的リスク

講義の最後では、RL 学習は事前学習や SFT よりはるかに工学的に複雑であることが強調されます。事前学習は主に固定データセット上の next-token loss です。RL では、新しいデータを生成し、採点し、更新し、さらに新しいデータを生成し続ける必要があります。loss 自体も、教師あり学習のように直接解釈できるものではなくなります。training distribution が policy とともに変わるからです。本当に監視すべきものは、reward、pass rate、format error rate、KL、sample diversity、外部評価性能です。

評価時には特に、「training reward が上がった」ことと「真の能力が向上した」ことを区別する必要があります。reward function に抜け穴があれば、学習曲線はきれいに見えます。しかしモデルは、あるテンプレートを出力すること、parser の欠陥を利用すること、高得点 token を繰り返すこと、あるいはテスト環境で予期しない挙動を引き起こすことを学んだだけかもしれません。したがって少なくとも 3 種類の評価を用意すべきです。第 1 に、訓練と同分布の validation set。過学習や学習崩壊を素早く見つけるために使います。第 2 に、hidden またはより難しい out-of-distribution 評価。汎用的な戦略を学んだかを確認します。第 3 に、人間によるサンプル確認。自動指標では捉えにくい誤り、たとえば推論の捏造、format 上の投機、説明と答えの不一致を発見するために使います。

また、RL における「良いサンプル」は、常に良いとは限りません。学習初期にサンプリングされた高 reward 回答は、偶然正解しただけかもしれません。それに対して多すぎる勾配ステップを行うと、policy は狭い mode に急速に収縮し、探索能力が下がります。sampling temperature、prompt ごとの候補数、各バッチデータを何ステップ再利用するか、clip range、KL coefficient、reference model の更新頻度は、探索と安定性のバランスに影響します。実践では、平均 reward、best sample reward、response length、repetition rate、entropy、refusal rate、KL 曲線を同時に見る必要があり、単一指標だけを見てはいけません。

工学的には次も扱う必要があります。

- 推論コスト：各 prompt に対して複数の response をサンプリングする必要がある。
- reward 計算コスト：test を走らせたり、環境を呼び出したり、agent を実行したりする必要がある。
- 複数モデル管理：current policy、old policy、reference model、critic/reward model。
- 分散同期：rollout worker と trainer のあいだで model、sample、logprob をやり取りする必要がある。
- メモリ overhead：reference model によって GPU メモリ使用量が倍増することがある。
- stale policy：サンプルは古いパラメータで生成されるが、新しいパラメータで学習するため、ずれを制御しなければならない。

典型的な RLVR システムでは、rollout、reward、training、evaluation を複数のサービスに分けます。rollout worker は current または少し古いモデルで候補を生成します。reward worker は答えを parse し、test を走らせ、または環境を呼び出します。trainer は保存された logprob、reward、advantage に基づいてモデルを更新します。evaluator は固定 benchmark で定期的に評価します。どの工程でエラーが起きても学習は汚染されます。parser bug は誤った reward を作り、環境の非決定性は reward noise を増幅し、worker が古すぎるモデルを使うと ratio が歪み、分散ログが不完全だと問題の再現が難しくなります。

したがって RLVR はアルゴリズム問題であるだけでなく、システム問題でもあります。動作する学習システムは、サンプリング、reward、最適化、監視、安全性評価のすべてが信頼できることを同時に保証しなければなりません。optimizer が強いほど、reward 定義の欠陥は増幅されます。学習規模を拡大する前に、小モデル・小データ・解釈しやすい例で reward function を繰り返し検証すべきです。

## 9. まとめ

この講義の主線は次のようにまとめられます。言語モデル RL は、完全な回答を action sequence と見なし、検証可能または学習された reward で結果を評価し、policy gradient によって高 reward 回答の確率を上げます。素朴な policy gradient は「reward で重み付けした SFT」に似ていますが、高分散、sparse reward、credit assignment により学習は難しくなります。baseline と advantage は相対比較によって分散を下げます。GRPO は同じ prompt に対する複数サンプリングを利用して自然に group 内 baseline を構成します。clipping と KL regularization は更新幅を制御し、policy collapse を防ぎます。

RLVR は reasoning models を学習する重要な経路です。検証可能タスクからモデルが自己改善できるからです。ただし成功は、reward function が真実で、hack されにくく、汎化可能であるか、そして学習工程全体が安定しているかに依存します。最終原則は、信頼して測定できるなら最適化できる、測定に抜け穴があるなら最適化はその抜け穴を増幅する、ということです。言い換えれば、reasoning model の能力向上は「生成—検証—更新」の閉ループから生まれるのであって、単に推論文を長く書くことから生まれるのではありません。本当に重要なのは、検証器が有効な推論ともっともらしいだけのナンセンスを区別できるかどうかです。


---
