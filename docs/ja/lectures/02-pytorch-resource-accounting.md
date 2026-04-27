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
