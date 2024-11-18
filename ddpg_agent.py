from pyDOE import lhs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import environment
import utils

# Define the Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()  # Ensures actions are in [-1, 1]

    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        action = self.output_activation(self.fc3(x))
        return action

# Define the Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.activation = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # Concatenate state and action
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# Replay Buffer Class
class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.buffer = {
            "state": np.zeros((max_size, state_dim)),
            "action": np.zeros((max_size, action_dim)),
            "reward": np.zeros((max_size, 1)),
            "next_state": np.zeros((max_size, state_dim)),
            "done": np.zeros((max_size, 1)),
        }

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        self.buffer["state"][self.ptr] = state
        self.buffer["action"][self.ptr] = action
        self.buffer["reward"][self.ptr] = reward
        self.buffer["next_state"][self.ptr] = next_state
        self.buffer["done"][self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """Sample a batch of experiences for training."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = {
            key: torch.tensor(self.buffer[key][indices], dtype=torch.float32)
            for key in self.buffer
        }
        return batch

class LHSActionSampler:
    def __init__(self, num_clients, num_servers):
        self.num_clients = num_clients
        self.num_servers = num_servers

    def sample(self, num_samples):
        """Generate LHS samples for client-server pairings."""
        samples = lhs(self.num_clients, samples=num_samples)
        actions = []

        for sample in samples:
            action = []
            for client_idx, proportion in enumerate(sample):
                server_idx = int(proportion * self.num_servers)  # Map proportion to server index
                action.append((client_idx, server_idx))
            actions.append(action)

        return actions

class DDPGAgent:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, gamma, tau, buffer_size, batch_size, num_clients, num_servers):
        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.tau = tau  # Soft update rate
        self.batch_size = batch_size

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

        # LHS Action Sampler
        self.action_sampler = LHSActionSampler(num_clients, num_servers)

        # Actor Networks
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic Networks
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state, noise_scale=0.1):
        """Select an action using the actor network."""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        action += noise_scale * np.random.normal(size=action.shape)  # Add exploration noise
        return np.clip(action, -1, 1)  # Ensure actions are within valid range

    def train(self):
        """Train the actor and critic networks."""
        if self.replay_buffer.size < self.batch_size:
            return  # Wait until the replay buffer has enough samples

        # Sample a batch of experiences
        batch = self.replay_buffer.sample(self.batch_size)

        # Extract batch components
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]

        # Critic loss
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_value = reward + (1 - done) * self.gamma * target_q
        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_value)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        predicted_action = self.actor(state)
        actor_loss = -self.critic(state, predicted_action).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)

    def soft_update(self, target, source, tau):
        """Perform a soft update of the target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

    def initialize_replay_buffer(self, env, num_samples):
        """
        Use LHS to populate the replay buffer with diverse initial experiences.
        Args:
            env: The environment instance.
            num_samples: Number of initial samples to generate.
        """
        # Generate LHS actions
        lhs_actions = self.action_sampler.sample(num_samples)

        for action in lhs_actions:
            # Retrieve the initial state
            state = env.get_environment_state()

            # Apply the action to the environment
            next_state, reward, done = env.apply_action(action)

            # Add the experience to the replay buffer
            self.replay_buffer.add(state, action, reward, next_state, done)

            # Reset the environment if episode ends
            if done:
                env.reset_environment()

    def train_agent(self, env, num_episodes, max_steps_per_episode, noise_scale_start, noise_scale_decay, log_interval=10):
        """
        Train the DDPG agent by interacting with the environment.
        Args:
            env: The environment instance.
            num_episodes: Number of training episodes.
            max_steps_per_episode: Maximum steps per episode.
            noise_scale_start: Initial exploration noise scale.
            noise_scale_decay: Decay factor for exploration noise.
            log_interval: Interval for logging progress.
        """
        noise_scale = noise_scale_start

        for episode in range(1, num_episodes + 1):
            # Reset the environment at the start of each episode
            state = env.reset_environment()

            episode_reward = 0

            for step in range(max_steps_per_episode):
                # Select an action using the actor (with exploration noise)
                action = self.select_action(state, noise_scale)

                # Apply the action in the environment
                next_state, reward, done = env.apply_action(action)

                # Store the experience in the replay buffer
                self.replay_buffer.add(state, action, reward, next_state, done)

                # Train the agent
                self.train()

                # Update state and accumulate rewards
                state = next_state
                episode_reward += reward

                if done:
                    break

            # Decay the noise scale
            noise_scale *= noise_scale_decay

            # Log progress at specified intervals
            if episode % log_interval == 0:
                print(f"Episode {episode}/{num_episodes}, Total Reward: {episode_reward:.2f}")

        print("Training complete.")

    def evaluate_agent(self, env, num_episodes):
        """
        Evaluate the agent's performance in the environment.
        Args:
            env: The environment instance.
            num_episodes: Number of episodes to evaluate.
        Returns:
            Average reward across evaluation episodes.
        """
        total_reward = 0

        for episode in range(num_episodes):
            state = env.reset_environment()
            episode_reward = 0
            done = False

            while not done:
                # Select an action without exploration noise
                action = self.select_action(state, noise_scale=0.0)
                state, reward, done = env.apply_action(action)
                episode_reward += reward

            total_reward += episode_reward

        avg_reward = total_reward / num_episodes
        print(f"Evaluation over {num_episodes} episodes: Avg Reward = {avg_reward:.2f}")
        return avg_reward
