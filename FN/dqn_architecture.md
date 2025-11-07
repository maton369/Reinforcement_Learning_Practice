# Deep Q-Network (DQN) アーキテクチャと仕組み

## 1. 全体システムアーキテクチャ

```mermaid
graph TB
    subgraph "Environment"
        ENV[Gym Environment<br/>Catcher-v0 / CartPole-v0]
    end

    subgraph "Observer Layer"
        OBS[CatcherObserver<br/>画像前処理・フレームスタック]
    end

    subgraph "Agent Layer"
        AGENT[DeepQNetworkAgent<br/>行動選択・Q値推定]
        MODEL[Online Network<br/>CNN/MLP]
        TEACHER[Target Network<br/>CNN/MLP]
    end

    subgraph "Training Layer"
        TRAINER[DeepQNetworkTrainer<br/>学習ループ制御]
        BUFFER[Experience Replay Buffer<br/>size=50000]
        LOGGER[TensorBoard Logger]
    end

    ENV -->|state| OBS
    OBS -->|processed state| AGENT
    AGENT -->|action| ENV
    ENV -->|reward, done| AGENT

    AGENT -->|experience| BUFFER
    BUFFER -->|batch sampling| TRAINER
    TRAINER -->|update| MODEL
    TRAINER -->|periodic copy| TEACHER

    MODEL -->|Q-values| AGENT
    TEACHER -->|target Q-values| TRAINER

    TRAINER -->|metrics| LOGGER
```

## 2. DQNアルゴリズムのフロー

```mermaid
sequenceDiagram
    participant E as Environment
    participant O as Observer
    participant A as Agent
    participant M as Online Network
    participant T as Target Network
    participant B as Replay Buffer
    participant Tr as Trainer

    Note over E,Tr: Episode開始

    E->>O: state (生画像)
    O->>O: グレースケール化<br/>リサイズ(80x80)<br/>正規化(0-1)<br/>4フレームスタック
    O->>A: processed state

    A->>M: estimate(state)
    M->>A: Q-values for all actions
    A->>A: ε-greedy選択
    A->>E: action

    E->>A: next_state, reward, done
    A->>B: store(s, a, r, s', done)

    alt 学習フェーズ & バッファ十分
        B->>Tr: sample batch (32経験)
        Tr->>M: predict(states)
        Tr->>T: predict(next_states)

        Tr->>Tr: 計算 TD Target<br/>y = r + γ * max(Q_target(s'))
        Tr->>M: train_on_batch(states, targets)
        M->>Tr: loss

        alt 一定周期 (3エピソードごと)
            Tr->>T: copy weights from Online
        end

        Tr->>A: update epsilon (線形減衰)
    end

    Note over E,Tr: Episode終了後
    Tr->>Tr: 報酬合計を計算
    alt 最高報酬更新
        Tr->>M: save model (dqn_agent.h5)
    end
```

## 3. ニューラルネットワーク構造

### 3.1 Catcher用CNN (DeepQNetworkAgent)

```mermaid
graph LR
    INPUT[入力<br/>80x80x4<br/>height×width×frames]

    subgraph "特徴抽出層"
        CONV1[Conv2D<br/>filters=32<br/>kernel=8x8<br/>stride=4<br/>ReLU]
        CONV2[Conv2D<br/>filters=64<br/>kernel=4x4<br/>stride=2<br/>ReLU]
        CONV3[Conv2D<br/>filters=64<br/>kernel=3x3<br/>stride=1<br/>ReLU]
    end

    FLAT[Flatten]

    subgraph "決定層"
        FC1[Dense<br/>units=256<br/>ReLU]
        FC2[Dense<br/>units=3<br/>行動数<br/>線形出力]
    end

    OUTPUT[出力<br/>Q値ベクトル<br/>各行動のQ値]

    INPUT --> CONV1 --> CONV2 --> CONV3 --> FLAT --> FC1 --> FC2 --> OUTPUT
```

### 3.2 CartPole用MLP (DeepQNetworkAgentTest)

```mermaid
graph LR
    INPUT2[入力<br/>状態ベクトル<br/>ex: 4次元]

    FC1_TEST[Dense<br/>units=64<br/>ReLU]
    FC2_TEST[Dense<br/>units=2<br/>行動数<br/>ReLU]

    OUTPUT2[出力<br/>Q値ベクトル]

    INPUT2 --> FC1_TEST --> FC2_TEST --> OUTPUT2
```

## 4. Experience Replay Buffer

```mermaid
graph TB
    subgraph "Experience Structure"
        EXP[Experience<br/>s: 状態<br/>a: 行動<br/>r: 報酬<br/>n_s: 次状態<br/>d: 終了フラグ]
    end

    subgraph "Replay Buffer (deque)"
        direction LR
        E1[exp_1] --> E2[exp_2] --> E3[exp_3] --> E4[...] --> E5[exp_50000]
    end

    subgraph "Sampling"
        RANDOM[ランダムサンプリング<br/>batch_size=32]
    end

    EXP -->|store| E1
    E1 -.->|oldest removed<br/>when full| E5
    E3 --> RANDOM
    E5 --> RANDOM
    E2 --> RANDOM

    RANDOM --> UPDATE[Update Network]
```

## 5. Q学習の更新メカニズム

```mermaid
graph TB
    subgraph "入力"
        BATCH[Batch of 32 experiences<br/>s, a, r, s', done]
    end

    subgraph "予測"
        ONLINE[Online Network<br/>Q_online状態s]
        TARGET[Target Network<br/>Q_target次状態s']
    end

    subgraph "TD Target計算"
        TD[TD Target<br/>y_i = r_i + γ * max_a' Q_target次状態s', a'<br/>if done: y_i = r_i]
    end

    subgraph "損失計算"
        LOSS[MSE Loss<br/>L = 1/N Σ Q_online状態s, a - y_i²]
    end

    subgraph "更新"
        BACKPROP[Backpropagation<br/>Adam Optimizer<br/>lr=1e-4 Catcher<br/>gradient clipping=1.0]
    end

    BATCH --> ONLINE
    BATCH --> TARGET
    ONLINE --> TD
    TARGET --> TD
    TD --> LOSS
    LOSS --> BACKPROP
    BACKPROP --> ONLINE
```

## 6. ε-greedy探索戦略

```mermaid
graph TB
    START[行動選択開始<br/>現在状態 s]

    ESTIMATE[Online Network<br/>Q値推定]

    RAND{乱数 < ε?}

    EXPLORE[探索<br/>ランダム行動選択]
    EXPLOIT[活用<br/>argmax_a Qs, a]

    ACTION[行動実行]

    DECAY[ε減衰<br/>ε = ε - ε_initial - ε_final / episodes<br/>ε_initial=0.5 → ε_final=0.001]

    START --> ESTIMATE
    ESTIMATE --> RAND
    RAND -->|Yes| EXPLORE
    RAND -->|No| EXPLOIT
    EXPLORE --> ACTION
    EXPLOIT --> ACTION
    ACTION --> DECAY
```

## 7. Target Network更新メカニズム

```mermaid
graph LR
    subgraph "Training Progress"
        E1[Episode 1-2]
        E3[Episode 3]
        E4[Episode 4-5]
        E6[Episode 6]
    end

    subgraph "Online Network"
        O1[θ_online]
        O2[θ_online']
        O3[θ_online'']
        O4[θ_online''']
    end

    subgraph "Target Network"
        T1[θ_target]
        T2[θ_target]
        T3[θ_target θ_online']
        T4[θ_target θ_online']
    end

    E1 --> O1
    E3 --> O2
    E4 --> O3
    E6 --> O4

    O1 -.->|毎ステップ更新| O2
    O2 -.->|毎ステップ更新| O3
    O3 -.->|毎ステップ更新| O4

    T1 -->|固定| T2
    T2 -->|3エピソード毎コピー| T3
    T3 -->|固定| T4

    style T3 fill:#f96
    style O2 fill:#9cf
```

## 8. 学習パラメータと設定

```mermaid
graph TB
    subgraph "ハイパーパラメータ"
        HP1[バッファサイズ: 50000]
        HP2[バッチサイズ: 32]
        HP3[割引率 γ: 0.99]
        HP4[初期ε: 0.5]
        HP5[最終ε: 0.001]
        HP6[学習率: 1e-4 Catcher<br/>1e-3 CartPole]
        HP7[Target更新頻度: 3エピソード]
        HP8[最適化: Adam + clipping1.0]
    end

    subgraph "学習フェーズ"
        INIT[初期化フェーズ<br/>200エピソード<br/>経験蓄積のみ]
        TRAIN[訓練フェーズ<br/>1000エピソード<br/>学習+探索]
    end

    subgraph "保存条件"
        SAVE[ベスト報酬更新時<br/>モデル保存<br/>dqn_agent.h5]
    end

    HP1 --> INIT
    HP2 --> TRAIN
    HP3 --> TRAIN
    HP4 --> INIT
    HP5 --> TRAIN
    HP6 --> TRAIN
    HP7 --> TRAIN
    HP8 --> TRAIN

    INIT --> TRAIN
    TRAIN --> SAVE
```

## 9. CatcherObserver前処理パイプライン

```mermaid
graph TB
    RAW[生画像<br/>PLEからの出力<br/>RGB配列]

    GRAY[グレースケール変換<br/>PIL Image.convert_L]

    RESIZE[リサイズ<br/>80 × 80]

    NORM[正規化<br/>0.0 - 1.0<br/>÷ 255.0]

    STACK[フレームスタック<br/>deque maxlen=4<br/>最新4フレーム保持]

    TRANS[軸転置<br/>frame, width, height<br/>→ height, width, frame<br/>80, 80, 4]

    OUTPUT[CNNへの入力<br/>80×80×4 テンソル]

    RAW --> GRAY --> RESIZE --> NORM --> STACK --> TRANS --> OUTPUT
```

## 10. 強化学習としての成立要素

```mermaid
mindmap
    root((DQN))
        価値ベース強化学習
            Q学習の拡張
            状態行動価値関数の近似
            Bellman方程式に基づく更新

        関数近似
            ニューラルネットワーク
                CNN: 画像から特徴抽出
                MLP: 状態ベクトル処理
            非線形近似能力
            高次元状態空間への対応

        安定化技術
            Experience Replay
                相関の除去
                データ効率の向上
                過去経験の再利用
            Target Network
                学習の安定化
                moving targetの回避
                周期的な重み同期

        探索と活用
            ε-greedy戦略
                初期: 高探索 ε0.5
                終盤: 高活用 ε0.001
                線形減衰

        学習プロセス
            初期化: 200エピソード
                経験蓄積
                更新なし
            訓練: 1000エピソード
                ミニバッチ更新
                ε減衰
                Target更新

        最適化
            Adam Optimizer
            勾配クリッピング
            MSE損失関数
            バッチ学習
```

## 11. DQNが強化学習として成り立つ理由

### 11.1 マルコフ決定過程 (MDP) の要素

| MDP要素 | DQN実装 |
|---------|---------|
| **状態 S** | Catcher: 80×80×4画像テンソル<br/>CartPole: 4次元状態ベクトル |
| **行動 A** | Catcher: 3行動 (左/停止/右)<br/>CartPole: 2行動 (左/右) |
| **報酬 R** | 環境から各ステップで返される即時報酬 |
| **状態遷移 P** | 環境のstep関数により決定論的/確率的に遷移 |
| **割引率 γ** | 0.99 (未来報酬の重み付け) |
| **方策 π** | ε-greedy方策 (Q値最大化 + ランダム探索) |

### 11.2 Q学習の実装

**Bellman最適方程式:**
```
Q*(s, a) = E[r + γ * max_a' Q*(s', a')]
```

**DQNでの近似:**
```python
# update() メソッド内 (lines 94-110)
reward = e.r
if not e.d:
    reward += gamma * np.max(future[i])  # γ * max Q_target(s', a')
estimateds[i][e.a] = reward  # TD target
loss = model.train_on_batch(states, estimateds)  # MSE最小化
```

### 11.3 時間的差分 (TD) 学習

- **TD誤差**: `δ = [r + γ * max Q_target(s', a')] - Q_online(s, a)`
- **MSE損失**: `L = δ²`
- **勾配降下**: Adam optimizerによる重み更新

### 11.4 off-policy学習

- 経験再生バッファから過去の経験をサンプリング
- 現在の方策とは異なる過去の方策で得た経験から学習可能
- データ効率の向上とサンプル相関の除去

## まとめ

このDQN実装は以下の要素により強化学習として成立しています:

1. **価値関数近似**: Q値をニューラルネットで近似
2. **TD学習**: Bellman方程式に基づく逐次更新
3. **探索と活用のバランス**: ε-greedy戦略
4. **安定化技術**: Experience Replay + Target Network
5. **最適化**: 勾配降下法による方策改善
6. **報酬最大化**: 累積割引報酬の最大化を目指す

これらが統合され、環境との相互作用を通じて最適な行動方策を学習する強化学習システムとして機能します。
