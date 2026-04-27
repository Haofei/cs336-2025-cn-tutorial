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
