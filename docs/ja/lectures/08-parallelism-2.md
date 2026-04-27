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
