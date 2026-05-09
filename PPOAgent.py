import numpy as np
import gymnasium as gym
import torch
import torch.optim as optim
from torch.distributions import Categorical

from Networks import PolicyNetwork, ValueNetwork


class RolloutBuffer:
    def __init__(self, n_steps, num_envs, obs_dim, device, gamma, gae_lambda):
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # pre-allocate storage on CPU
        self.states   = torch.zeros(n_steps, num_envs, obs_dim)
        self.actions  = torch.zeros(n_steps, num_envs, dtype=torch.long)
        self.rewards  = torch.zeros(n_steps, num_envs)
        self.dones    = torch.zeros(n_steps, num_envs)
        self.values   = torch.zeros(n_steps, num_envs)
        self.log_probs = torch.zeros(n_steps, num_envs)

        self.advantages = None
        self.returns    = None
        self.step = 0

    def add(self, state, action, reward, done, value, log_prob):
        self.states[self.step]    = state.cpu()
        self.actions[self.step]   = action.cpu()
        self.rewards[self.step]   = torch.tensor(reward, dtype=torch.float32)
        self.dones[self.step]     = torch.tensor(done,   dtype=torch.float32)
        self.values[self.step]    = value.cpu().detach()
        self.log_probs[self.step] = log_prob.cpu().detach()
        self.step += 1

    def compute_advantages(self, last_value):
        last_value = last_value.cpu().detach()
        advantages = torch.zeros_like(self.rewards)
        last_adv   = torch.zeros(self.num_envs)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value        = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value        = self.values[t + 1]

            delta    = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_adv = delta + self.gamma * self.gae_lambda * next_non_terminal * last_adv
            advantages[t] = last_adv

        self.advantages = advantages
        self.returns    = advantages + self.values

    def get_minibatches(self, minibatch_size):
        N = self.n_steps * self.num_envs

        states    = self.states.view(N, -1).to(self.device)
        actions   = self.actions.view(N).to(self.device)
        log_probs = self.log_probs.view(N).to(self.device)
        returns   = self.returns.view(N).to(self.device)

        # normalise advantages over the whole buffer for training stability
        adv = self.advantages.view(N).to(self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        indices = torch.randperm(N, device=self.device)
        for start in range(0, N, minibatch_size):
            idx = indices[start : start + minibatch_size]
            yield (states[idx], actions[idx], log_probs[idx], returns[idx], adv[idx])

    def reset(self):
        self.step = 0


class PPOAgent:
    def __init__(self, state_size, action_size, device, config):
        self.device = device
        self.config = config

        self.policy_net = PolicyNetwork(state_size, action_size, config.hidden_size).to(device)
        self.value_net  = ValueNetwork(state_size, config.hidden_size).to(device)
        self.optimizer  = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=config.lr,
        )

        self.steps_done = 0

    def select_action(self, state, greedy=False):
        probs = self.policy_net(state)
        dist  = Categorical(probs)

        if greedy:
            action = probs.argmax(dim=-1)
            return action, None, None

        action   = dist.sample()
        log_prob = dist.log_prob(action)
        value    = self.value_net(state)
        return action, log_prob, value

    def update(self, buffer):
        cfg = self.config

        for _ in range(cfg.n_epochs):
            for states, actions, old_log_probs, returns, advantages in \
                    buffer.get_minibatches(cfg.minibatch_size):

                probs     = self.policy_net(states)
                dist      = Categorical(probs)
                log_probs = dist.log_prob(actions)
                entropy   = dist.entropy().mean()
                values    = self.value_net(states)

                # clipped surrogate loss
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # value loss
                critic_loss = torch.nn.functional.mse_loss(values, returns)

                # combined loss
                loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

    def evaluate(self, eval_env, eval_episodes=10):
        self.policy_net.eval()
        returns = []

        for _ in range(eval_episodes):
            result = eval_env.reset()
            obs    = result[0] if isinstance(result, tuple) else result
            state  = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            episode_return = 0.0
            terminated, truncated = False, False

            with torch.no_grad():
                while not (terminated or truncated):
                    action, _, _ = self.select_action(state, greedy=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action.item())
                    episode_return += reward
                    state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            returns.append(episode_return)

        self.policy_net.train()
        return np.mean(returns)


def train_PPO(params, device):
    env = gym.vector.SyncVectorEnv([
        lambda: gym.make("CartPole-v1") for _ in range(params.num_envs)
    ])
    eval_env = gym.make("CartPole-v1")

    n_actions      = env.single_action_space.n
    n_observations = env.single_observation_space.shape[0]

    agent  = PPOAgent(n_observations, n_actions, device, params)
    buffer = RolloutBuffer(
        n_steps    = params.n_steps,
        num_envs   = params.num_envs,
        obs_dim    = n_observations,
        device     = device,
        gamma      = params.gamma,
        gae_lambda = params.gae_lambda,
    )

    eval_returns = []

    result = env.reset()
    obs    = result[0] if isinstance(result, tuple) else result
    states = torch.tensor(obs, dtype=torch.float32, device=device)

    while agent.steps_done < params.total_steps:

        # collect one rollout
        buffer.reset()
        for _ in range(params.n_steps):
            with torch.no_grad():
                actions, log_probs, values = agent.select_action(states)

            obs, rewards, terminated, truncated, _ = env.step(actions.cpu().numpy())
            dones = terminated | truncated
            buffer.add(states, actions, rewards, dones, values, log_probs)

            agent.steps_done += params.num_envs
            states = torch.tensor(obs, dtype=torch.float32, device=device)  # FIX: was torch.floar32

            if agent.steps_done % params.evaluate_every < params.num_envs:
                ret = agent.evaluate(eval_env, eval_episodes=params.eval_episodes)
                eval_returns.append(ret)
                print(f"Steps: {agent.steps_done} | Reward: {ret:.1f}")

        # bootstrap value for the last state
        with torch.no_grad():
            last_value = agent.value_net(states)

        # GAE + PPO update
        buffer.compute_advantages(last_value)
        agent.update(buffer)

    env.close()
    eval_env.close()                                        # FIX: was eval_env.close (missing call parens)
    return eval_returns