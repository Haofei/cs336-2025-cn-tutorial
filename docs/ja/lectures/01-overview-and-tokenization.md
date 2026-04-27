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
