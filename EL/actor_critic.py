# ================================================================
# Actor-Critic （方策勾配＋TD学習）による FrozenLake 学習デモ
# ---------------------------------------------------------------
# ・Actor  : 離散状態×行動のソフトマックス方策（好み=preference）
# ・Critic : 状態価値 V(s) を TD(0) で推定（ベースラインとして利用）
# ・更新式:
#     δ = r + γ V(s') − V(s)
#     V(s) ← V(s) + α_v δ
#     preference(s,a) ← preference(s,a) + α_p δ ( 1_{a=a_t} − π(a|s) )
#   併せて Q(s,a) も TD 誤差で更新し、可視化に利用する。
# ・学習途中の報酬推移は 50 エピソードごとに平均値・標準偏差を表示。
# ・学習後は Q 値ヒートマップと報酬履歴を可視化する。
# ================================================================

from __future__ import annotations

from typing import Callable, Sequence, Tuple, Type
import numpy as np
import gym

from frozen_lake_util import show_q_value


def _reset_env(env: gym.Env):
    """Gym API の差分（旧/新）を吸収して状態のみを返す。"""
    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs
    return obs


def _step_env(env: gym.Env, action: int) -> Tuple[int, float, bool, dict]:
    """Gym API の差分を吸収して (next_state, reward, done, info) を返す。"""
    outcome = env.step(action)
    if len(outcome) == 5:
        n_state, reward, terminated, truncated, info = outcome
        done = bool(terminated or truncated)
    else:
        n_state, reward, done, info = outcome
    return int(n_state), float(reward), bool(done), info


class Actor:
    """
    ソフトマックス方策を持つ actor。

    Attributes
    ----------
    preferences : ndarray, shape=(n_states, n_actions)
        方策勾配で更新する「好み」（logit）。softmax(preference[s]) が方策 π(a|s)。
    Q : ndarray, shape=(n_states, n_actions)
        TD 誤差を同じ係数で積算した行動価値。可視化に利用する。
    reward_log : list[float]
        各エピソードの総報酬。
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        q_learning_rate: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.learning_rate = float(learning_rate)
        self.q_learning_rate = float(q_learning_rate)
        self.temperature = float(max(temperature, 1e-6))

        self.preferences = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self._q_table = np.zeros_like(self.preferences)
        self.reward_log = []

    @property
    def Q(self) -> np.ndarray:
        """行動価値テーブル（可視化用）。"""
        return self._q_table

    def _policy(self, state: int) -> np.ndarray:
        """ソフトマックス方策 π(a|s) を返す。"""
        prefs = self.preferences[state] / self.temperature
        prefs = prefs - np.max(prefs)  # 数値安定化
        exp_prefs = np.exp(prefs)
        probs = exp_prefs / np.sum(exp_prefs)
        return probs

    def act(self, state: int) -> int:
        """方策に従って行動をサンプルする。"""
        probs = self._policy(state)
        action = np.random.choice(self.n_actions, p=probs)
        return int(action)

    def update(self, state: int, action: int, td_error: float) -> None:
        """
        方策勾配と Q テーブルを更新する。

        preference[s,a] ← preference[s,a] + α δ (1_{a=a_t} − π(a|s))
        Q(s,a) ← Q(s,a) + α_q δ
        """
        probs = self._policy(state)
        one_hot = np.zeros_like(probs)
        one_hot[action] = 1.0
        grad = one_hot - probs
        self.preferences[state] += self.learning_rate * td_error * grad
        self._q_table[state, action] += self.q_learning_rate * td_error

    # --- 報酬ログ周り（ELAgent と同等のインタフェース） ---
    def init_log(self) -> None:
        self.reward_log = []

    def log_reward(self, reward: float) -> None:
        self.reward_log.append(float(reward))

    def show_reward_log(self, interval: int = 50, episode: int = -1) -> None:
        """
        報酬ログを統計表示／可視化する。
        episode>0 のときは末尾 interval の平均・標準偏差を print。
        """
        if interval <= 0:
            raise ValueError("interval must be positive.")

        T = len(self.reward_log)
        if T == 0:
            print("No rewards to show.")
            return

        if episode > 0:
            window = min(interval, T)
            rewards = self.reward_log[-window:]
            mean = float(np.round(np.mean(rewards), 3))
            std = float(np.round(np.std(rewards), 3))
            print(
                f"At Episode {episode} average reward over last {window} steps is {mean} (+/-{std})."
            )
            return

        indices = list(range(0, T, interval))
        means, stds = [], []
        for i in indices:
            chunk = self.reward_log[i : i + interval]
            means.append(float(np.mean(chunk)))
            stds.append(float(np.std(chunk)))

        means = np.asarray(means)
        stds = np.asarray(stds)

        import matplotlib.pyplot as plt

        plt.figure()
        plt.title("Reward History")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.fill_between(
            indices,
            means - stds,
            means + stds,
            alpha=0.15,
            color="b",
            label="mean ± std",
        )
        plt.plot(indices, means, "o-", color="b", label=f"mean per {interval} steps")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


class Critic:
    """状態価値 V(s) を TD(0) で更新する。"""

    def __init__(self, n_states: int, gamma: float = 0.99, learning_rate: float = 0.1):
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.V = np.zeros(int(n_states), dtype=np.float64)

    def update(self, state: int, reward: float, next_state: int, done: bool) -> float:
        """TD 誤差 δ を計算し、V を更新して δ を返す。"""
        target = reward
        if not done:
            target += self.gamma * self.V[next_state]
        td_error = target - self.V[state]
        self.V[state] += self.learning_rate * td_error
        return td_error


class ActorCritic:
    """Actor と Critic を組み合わせて学習を進めるトレーナ。"""

    def __init__(
        self,
        actor_cls: Type[Actor],
        critic_cls: Type[Critic],
        gamma: float = 0.99,
        actor_lr: float = 0.1,
        critic_lr: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        self.actor_cls = actor_cls
        self.critic_cls = critic_cls
        self.gamma = float(gamma)
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.temperature = float(temperature)

    def _build_modules(self, env: gym.Env) -> Tuple[Actor, Critic]:
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        actor = self.actor_cls(
            n_states=n_states,
            n_actions=n_actions,
            learning_rate=self.actor_lr,
            q_learning_rate=self.actor_lr,
            temperature=self.temperature,
        )
        critic = self.critic_cls(
            n_states=n_states,
            gamma=self.gamma,
            learning_rate=self.critic_lr,
        )
        return actor, critic

    def train(
        self,
        env: gym.Env,
        episode_count: int = 3000,
        max_steps_per_episode: int = 200,
        report_interval: int = 50,
        seed: int | None = None,
    ) -> Tuple[Actor, Critic]:
        """
        Actor-Critic 学習を実行し、学習済み Actor / Critic を返す。
        """
        if seed is not None:
            np.random.seed(seed)
            try:
                env.reset(seed=seed)
            except TypeError:
                # 古い Gym では reset(seed=...) が無い場合がある
                pass

        actor, critic = self._build_modules(env)
        actor.init_log()

        for episode in range(1, episode_count + 1):
            state = _reset_env(env)
            total_reward = 0.0

            for _ in range(max_steps_per_episode):
                action = actor.act(state)
                next_state, reward, done, _ = _step_env(env, action)

                td_error = critic.update(state, reward, next_state, done)
                actor.update(state, action, td_error)

                total_reward += reward
                state = next_state

                if done:
                    break

            actor.log_reward(total_reward)

            if report_interval > 0 and episode % report_interval == 0:
                actor.show_reward_log(interval=report_interval, episode=episode)

        return actor, critic


def train() -> None:
    """
    FrozenLakeEasy-v0（is_slippery=False）で Actor-Critic 学習を実行するデモ。
    """
    env = gym.make("FrozenLakeEasy-v0")
    trainer = ActorCritic(
        Actor,
        Critic,
        gamma=0.99,
        actor_lr=0.1,
        critic_lr=0.1,
        temperature=1.0,
    )

    actor, critic = trainer.train(env, episode_count=3000, report_interval=50)

    show_q_value(actor.Q, env_id="FrozenLakeEasy-v0")
    actor.show_reward_log()
    env.close()


if __name__ == "__main__":
    train()
