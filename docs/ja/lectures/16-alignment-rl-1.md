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
