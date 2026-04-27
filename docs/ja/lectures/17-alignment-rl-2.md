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
