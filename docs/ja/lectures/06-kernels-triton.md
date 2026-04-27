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
