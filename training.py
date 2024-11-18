import numpy as np
import torch
from environment import Environment  # Assuming you have this class
from ddpg_agent import DDPGAgent  # Assuming DDPG agent is already implemented

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # Environment parameters
    num_clients = 10  # Example: Number of clients in the environment
    num_servers = 5   # Example: Number of servers in the environment
    state_dim = num_clients * 2 + num_servers * 2  # Adjust dimensions based on state features
    action_dim = num_clients  # Number of client-server pairings (one action per client)

    # Hyperparameters for DDPG agent
    actor_lr = 1e-4
    critic_lr = 1e-3
    gamma = 0.99  # Discount factor
    tau = 0.001  # Soft update rate
    buffer_size = 100000
    batch_size = 64

    # Create the environment and DDPG agent
    env = Environment(num_clients=num_clients, num_servers=num_servers)
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        tau=tau,
        buffer_size=buffer_size,
        batch_size=batch_size,
        num_clients=num_clients,
        num_servers=num_servers
    )

    # Set the random seed for reproducibility
    set_random_seeds(seed=42)

    # Initialize the replay buffer with diverse actions from LHS
    num_initial_samples = 1000
    agent.initialize_replay_buffer(env, num_samples=num_initial_samples)

    # Train the agent
    num_episodes = 500  # Total number of episodes for training
    max_steps_per_episode = 200  # Max steps per episode
    noise_scale_start = 1.0  # Initial exploration noise
    noise_scale_decay = 0.995  # Decay factor for noise scale

    print("Starting Training...")
    agent.train_agent(
        env,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        noise_scale_start=noise_scale_start,
        noise_scale_decay=noise_scale_decay,
        log_interval=10  # Log every 10 episodes
    )

    # Evaluate the agent after training
    num_eval_episodes = 100
    print("\nEvaluating the agent after training...")
    avg_reward = agent.evaluate_agent(env, num_episodes=num_eval_episodes)
    print(f"Average Reward after {num_eval_episodes} evaluation episodes: {avg_reward:.2f}")

if __name__ == "__main__":
    main()
