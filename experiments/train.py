"""
Unified training entry point for GRSN.

Supports:
- Model types: RNN (GRU/LSTM), SNN (LIF/RecurrentLIF/GRSNwoTAP), MLP
- Algorithms: TD3, SAC, SAC-Discrete
- Environments: POMDP, Meta-RL, Credit Assignment, etc.

Example usage:
    python experiments/train.py --env Pendulum-V-v0 --model_type rnn --encoder gru --algo sac --seed 0
    python experiments/train.py --env Catch-5-v0 --model_type snn --snn_type RecurrentLIF --algo sacd --seed 0
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gym
import numpy as np
import torch
from ruamel.yaml import YAML
from torch.nn import functional as F

import grsn.torchkit.pytorch_utils as ptu
from grsn import Policy_RNN, Policy_SNN, Policy_MLP
from grsn.buffers import SeqReplayBuffer, SimpleReplayBuffer
from grsn.utils import helpers as utl
from grsn.utils import system
import math
import time


@torch.no_grad()
def collect_rollouts(agent, env, policy_storage, max_trajectory_len, act_dim, act_continuous,
                     num_rollouts, random_actions=False, deterministic=False, train_mode=True):
    """Collect rollouts for recurrent policies (RNN/SNN).

    :param agent: Policy agent
    :param env: Gym environment
    :param policy_storage: Replay buffer
    :param max_trajectory_len: Maximum trajectory length
    :param act_dim: Action dimension
    :param act_continuous: Whether action space is continuous
    :param num_rollouts: Number of rollouts to collect
    :param random_actions: Whether to use random actions
    :param deterministic: Whether to use deterministic action selection
    :param train_mode: Whether in training mode (store to buffer)
    """
    if not train_mode:
        assert random_actions is False and deterministic is True

    total_steps = 0
    total_rewards = 0.0

    for idx in range(num_rollouts):
        steps = 0
        rewards = 0.0
        obs = ptu.from_numpy(env.reset())
        obs = obs.reshape(1, obs.shape[-1])
        done_rollout = False

        # Get hidden state at timestep=0
        action, reward, internal_state = agent.get_initial_info()

        if train_mode:
            # Temporary storage
            obs_list, act_list, rew_list, next_obs_list, term_list = [], [], [], [], []

        while not done_rollout:
            if random_actions:
                action = ptu.FloatTensor([env.action_space.sample()])  # (1, A)
                if not act_continuous:
                    action = F.one_hot(action.long(), num_classes=act_dim).float()  # (1, A)
            else:
                # Policy takes hidden state as input for rnn/snn
                (action, _, _, _), internal_state = agent.act(
                    prev_internal_state=internal_state,
                    prev_action=action,
                    reward=reward,
                    obs=obs,
                    deterministic=deterministic,
                )

            # Observe reward and next obs (B=1, dim)
            next_obs, reward, done, info = utl.env_step(env, action.squeeze(dim=0))
            done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

            # Update statistics
            steps += 1
            rewards += reward.item()

            # Early stopping env: term ignores timeout
            term = (
                False
                if "TimeLimit.truncated" in info or steps >= max_trajectory_len
                else done_rollout
            )

            if train_mode:
                # Append tensors to temporary storage
                obs_list.append(obs)  # (1, dim)
                act_list.append(action)  # (1, dim)
                rew_list.append(reward)  # (1, dim)
                term_list.append(term)  # bool
                next_obs_list.append(next_obs)  # (1, dim)

            # Set: obs <- next_obs
            obs = next_obs.clone()

        if train_mode:
            # Add collected sequence to buffer
            act_buffer = torch.cat(act_list, dim=0)
            if not act_continuous:
                act_buffer = torch.argmax(act_buffer, dim=-1, keepdims=True)
            policy_storage.add_episode(
                observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                actions=ptu.get_numpy(act_buffer),  # (L, dim)
                rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                next_observations=ptu.get_numpy(torch.cat(next_obs_list, dim=0)),  # (L, dim)
            )

        if train_mode:
            print(f"Mode: Train, Steps: {steps}, Rewards: {rewards:.2f}")
        else:
            print(f"Mode: Test, Steps: {steps}, Rewards: {rewards:.2f}")

        total_steps += steps
        total_rewards += rewards

    if train_mode:
        return total_steps
    else:
        return total_rewards / num_rollouts


@torch.no_grad()
def collect_rollouts_mlp(agent, env, policy_storage, max_trajectory_len, act_dim, act_continuous,
                         num_rollouts, random_actions=False, deterministic=False, train_mode=True):
    """Collect rollouts for MLP policies.

    :param agent: Policy agent
    :param env: Gym environment
    :param policy_storage: Replay buffer
    :param max_trajectory_len: Maximum trajectory length
    :param act_dim: Action dimension
    :param act_continuous: Whether action space is continuous
    :param num_rollouts: Number of rollouts to collect
    :param random_actions: Whether to use random actions
    :param deterministic: Whether to use deterministic action selection
    :param train_mode: Whether in training mode (store to buffer)
    """
    if not train_mode:
        assert random_actions is False and deterministic is True

    total_steps = 0
    total_rewards = 0.0

    for idx in range(num_rollouts):
        steps = 0
        rewards = 0.0
        obs = ptu.from_numpy(env.reset())
        obs = obs.reshape(1, obs.shape[-1])
        done_rollout = False

        while not done_rollout:
            if random_actions:
                action = ptu.FloatTensor([env.action_space.sample()])  # (1, A)
                if not act_continuous:
                    action = F.one_hot(action.long(), num_classes=act_dim).float()  # (1, A)
            else:
                # MLP policy takes obs directly
                action, _, _, _ = agent.act(obs, deterministic=deterministic)

            # Observe reward and next obs (B=1, dim)
            next_obs, reward, done, info = utl.env_step(env, action.squeeze(dim=0))
            done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

            # Update statistics
            steps += 1
            rewards += reward.item()

            # Early stopping env: term ignores timeout
            term = (
                False
                if "TimeLimit.truncated" in info or steps >= max_trajectory_len
                else done_rollout
            )

            if train_mode:
                # Add to buffer
                policy_storage.add_sample(
                    observation=ptu.get_numpy(obs.squeeze(dim=0)),
                    action=ptu.get_numpy(
                        action.squeeze(dim=0)
                        if act_continuous
                        else torch.argmax(action.squeeze(dim=0), dim=-1, keepdims=True)
                    ),
                    reward=ptu.get_numpy(reward.squeeze(dim=0)),
                    terminal=np.array([term], dtype=float),
                    next_observation=ptu.get_numpy(next_obs.squeeze(dim=0))
                )

            # Set: obs <- next_obs
            obs = next_obs.clone()

        if train_mode:
            print(f"Mode: Train, Steps: {steps}, Rewards: {rewards:.2f}")
        else:
            print(f"Mode: Test, Steps: {steps}, Rewards: {rewards:.2f}")

        total_steps += steps
        total_rewards += rewards

    if train_mode:
        return total_steps
    else:
        return total_rewards / num_rollouts


def update(agent, policy_storage, batch_size, model_type, num_updates):
    """Update the agent using collected experience.

    :param agent: Policy agent
    :param policy_storage: Replay buffer
    :param batch_size: Batch size for training
    :param model_type: Type of model (mlp, rnn, snn)
    :param num_updates: Number of updates to perform
    :return: Dictionary of training statistics
    """
    rl_losses_agg = {}

    for update in range(num_updates):
        # Sample random RL batch
        if model_type == "mlp":
            batch = ptu.np_to_pytorch_batch(policy_storage.random_batch(batch_size))
        else:
            batch = ptu.np_to_pytorch_batch(policy_storage.random_episodes(batch_size))

        # RL update
        rl_losses = agent.update(batch)

        for k, v in rl_losses.items():
            if update == 0:  # First iterate - create list
                rl_losses_agg[k] = [v]
            else:  # Append values
                rl_losses_agg[k].append(v)

    # Statistics
    for k in rl_losses_agg:
        rl_losses_agg[k] = np.mean(rl_losses_agg[k])

    return rl_losses_agg


def main():
    parser = argparse.ArgumentParser(description='GRSN Training')
    parser.add_argument('--env', type=str, required=True,
                        help='Environment name (e.g., Pendulum-V-v0, Catch-5-v0)')
    parser.add_argument('--model_type', type=str, default='rnn',
                        choices=['mlp', 'rnn', 'snn'],
                        help='Type of model to use')
    parser.add_argument('--snn_type', type=str, default='RecurrentLIF',
                        choices=['LIF', 'RecurrentLIF', 'GRSNwoTAP', 'AdaptiveLIF', 'LIFwoTAP'],
                        help='Type of SNN neuron (only for model_type=snn)')
    parser.add_argument('--encoder', type=str, default='gru',
                        choices=['gru', 'lstm'],
                        help='Type of RNN encoder (only for model_type=rnn)')
    parser.add_argument('--algo', type=str, default='sac',
                        choices=['td3', 'sac', 'sacd'],
                        help='RL algorithm to use')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--cuda', type=int, default=0,
                        help='CUDA device ID (-1 for CPU)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional, overrides auto-detection)')
    parser.add_argument('--save_model', action='store_true',
                        help='Save trained model')

    args = parser.parse_args()

    # Set device
    ptu.set_gpu_mode(torch.cuda.is_available() and args.cuda >= 0, args.cuda)

    # Set random seed
    system.reproduce(args.seed)

    # Import environments
    import grsn.envs.pomdp
    import grsn.envs.meta
    import grsn.envs.credit_assign

    # Create environment
    env = gym.make(args.env)
    max_trajectory_len = env._max_episode_steps

    # Determine action space type
    if env.action_space.__class__.__name__ == "Box":
        act_dim = env.action_space.shape[0]
        act_continuous = True
    else:
        assert env.action_space.__class__.__name__ == "Discrete"
        act_dim = env.action_space.n
        act_continuous = False
        args.algo = 'sacd'  # Force SACD for discrete actions

    obs_dim = env.observation_space.shape[0]
    print(f"Environment: {args.env}")
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}, Max steps: {max_trajectory_len}")

    # Create experiment name
    if args.model_type == "rnn":
        exp_name = f"{args.model_type}_{args.encoder}_{args.algo}_seed{args.seed}"
    elif args.model_type == "snn":
        exp_name = f"{args.snn_type}_{args.algo}_seed{args.seed}"
    else:
        exp_name = f"{args.model_type}_{args.algo}_seed{args.seed}"

    # Create result directories
    result_path = f'./results/{args.env}'
    model_path = f'./models/{args.env}'
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # Load config
    if args.config is not None:
        cfg_path = args.config
    else:
        # Auto-detect config based on environment and model type
        env_parts = args.env.lower().split('-')
        env_name = env_parts[0]

        # Handle BLT suffix
        if 'blt' in env_name:
            env_name = env_name.replace('blt', '_blt')

        # Get obs type (p, v, f, etc.)
        if len(env_parts) > 1:
            obs_type = env_parts[1]
        else:
            obs_type = 'f'  # Default to full observation

        if args.model_type == 'mlp':
            cfg_path = f'./configs/pomdp/{env_name}/{obs_type}/mlp.yml'
        else:
            cfg_path = f'./configs/pomdp/{env_name}/{obs_type}/rnn.yml'

    # Load YAML config
    yaml = YAML()
    try:
        v = yaml.load(open(cfg_path))
        print(f"Loaded config from: {cfg_path}")
    except FileNotFoundError:
        print(f"Config file not found: {cfg_path}")
        print("Using default parameters...")
        v = {
            'train': {
                'num_updates_per_iter': 1,
                'buffer_size': 10000,
                'batch_size': 32,
                'num_iters': 1000,
                'num_init_rollouts_pool': 10,
                'num_rollouts_per_iter': 1,
                'sampled_seq_len': 50,
            },
            'policy': {
                'action_embedding_size': 8,
                'observ_embedding_size': 32,
                'reward_embedding_size': 8,
                'rnn_hidden_size': 128,
                'dqn_layers': [128, 128],
                'policy_layers': [128, 128],
                'lr': 3e-4,
                'gamma': 0.99,
                'tau': 5e-3,
                'sac': {
                    'entropy_alpha': 0.1,
                    'automatic_entropy_tuning': True,
                    'target_entropy': None,
                    'alpha_lr': 3e-4,
                },
                'td3': {},
                'sacd': {
                    'entropy_alpha': 0.1,
                    'automatic_entropy_tuning': True,
                    'target_entropy': None,
                    'alpha_lr': 3e-4,
                },
            }
        }

    # Extract parameters
    buffer_size = v['train']['buffer_size']
    batch_size = v['train']['batch_size']
    num_iters = v['train']['num_iters']
    num_init_rollouts_pool = v['train']['num_init_rollouts_pool']
    num_rollouts_per_iter = v['train']['num_rollouts_per_iter']

    dqn_layers = v['policy']['dqn_layers']
    policy_layers = v['policy']['policy_layers']
    lr = v['policy']['lr']
    gamma = v['policy']['gamma']
    tau = v['policy']['tau']

    # Create agent
    if args.model_type != 'mlp':
        num_updates_per_iter = v['train']['num_updates_per_iter']
        sampled_seq_len = v['train']['sampled_seq_len']
        action_embedding_size = v['policy']['action_embedding_size']
        observ_embedding_size = v['policy']['observ_embedding_size']
        reward_embedding_size = v['policy']['reward_embedding_size']
        rnn_hidden_size = v['policy']['rnn_hidden_size']

    if args.model_type == 'rnn':
        agent = Policy_RNN(
            obs_dim=obs_dim,
            action_dim=act_dim,
            encoder=args.encoder,
            algo_name=args.algo,
            action_embedding_size=action_embedding_size,
            observ_embedding_size=observ_embedding_size,
            reward_embedding_size=reward_embedding_size,
            rnn_hidden_size=rnn_hidden_size,
            dqn_layers=dqn_layers,
            policy_layers=policy_layers,
            lr=lr,
            gamma=gamma,
            tau=tau,
            image_encoder_fn=lambda: None,
            kwargs=v['policy'][args.algo],
        ).to(ptu.device)

        policy_storage = SeqReplayBuffer(
            max_replay_buffer_size=int(buffer_size),
            observation_dim=obs_dim,
            action_dim=act_dim if act_continuous else 1,
            sampled_seq_len=sampled_seq_len,
            sample_weight_baseline=0.0,
        )
        collect_fn = collect_rollouts

    elif args.model_type == 'snn':
        agent = Policy_SNN(
            obs_dim=obs_dim,
            action_dim=act_dim,
            snn_type=args.snn_type,
            algo_name=args.algo,
            action_embedding_size=action_embedding_size,
            observ_embedding_size=observ_embedding_size,
            reward_embedding_size=reward_embedding_size,
            rnn_hidden_size=rnn_hidden_size,
            dqn_layers=dqn_layers,
            policy_layers=policy_layers,
            lr=lr,
            gamma=gamma,
            tau=tau,
            image_encoder_fn=lambda: None,
            kwargs=v['policy'][args.algo],
        ).to(ptu.device)

        policy_storage = SeqReplayBuffer(
            max_replay_buffer_size=int(buffer_size),
            observation_dim=obs_dim,
            action_dim=act_dim if act_continuous else 1,
            sampled_seq_len=sampled_seq_len,
            sample_weight_baseline=0.0,
        )
        collect_fn = collect_rollouts

    else:  # mlp
        agent = Policy_MLP(
            obs_dim=obs_dim,
            action_dim=act_dim,
            algo_name=args.algo,
            dqn_layers=dqn_layers,
            policy_layers=policy_layers,
            lr=lr,
            gamma=gamma,
            tau=tau,
            kwargs=v['policy'][args.algo],
        ).to(ptu.device)

        policy_storage = SimpleReplayBuffer(
            max_replay_buffer_size=int(buffer_size),
            observation_dim=obs_dim,
            action_dim=act_dim if act_continuous else 1,
            max_trajectory_len=max_trajectory_len,
            add_timeout=False,
        )
        collect_fn = collect_rollouts_mlp
        num_updates_per_iter = 1.0

    print(f"Created {args.model_type.upper()} agent with {args.algo.upper()} algorithm")

    # Calculate total steps
    total_rollouts = num_init_rollouts_pool + num_iters * num_rollouts_per_iter
    n_env_steps_total = max_trajectory_len * total_rollouts
    _n_env_steps_total = 0

    print(f"Total episodes: {total_rollouts}, Total steps: {n_env_steps_total}")

    # Initial data collection
    print("Collecting initial rollouts...")
    env_steps = collect_fn(
        agent, env, policy_storage, max_trajectory_len, act_dim, act_continuous,
        num_rollouts=num_init_rollouts_pool, random_actions=True, train_mode=True
    )
    _n_env_steps_total += env_steps

    # Evaluation parameters
    last_eval_num_iters = 0
    log_interval = 5
    eval_num_rollouts = 10
    learning_curve = {"x": [], "y": []}

    print("Starting training...")
    start_time = time.time()

    # Main training loop
    while _n_env_steps_total < n_env_steps_total:
        # Collect rollouts
        env_steps = collect_fn(
            agent, env, policy_storage, max_trajectory_len, act_dim, act_continuous,
            num_rollouts=num_rollouts_per_iter, train_mode=True
        )
        _n_env_steps_total += env_steps

        # Update agent
        num_updates = num_updates_per_iter if isinstance(num_updates_per_iter, int) else int(math.ceil(num_updates_per_iter * env_steps))
        train_stats = update(agent, policy_storage, batch_size, args.model_type, num_updates)

        # Evaluate
        current_num_iters = _n_env_steps_total // (num_rollouts_per_iter * max_trajectory_len)
        if current_num_iters != last_eval_num_iters and current_num_iters % log_interval == 0:
            last_eval_num_iters = current_num_iters
            average_returns = collect_fn(
                agent, env, policy_storage, max_trajectory_len, act_dim, act_continuous,
                num_rollouts=eval_num_rollouts, train_mode=False, random_actions=False, deterministic=True
            )
            learning_curve["x"].append(_n_env_steps_total)
            learning_curve["y"].append(average_returns)
            print(f"Step {_n_env_steps_total}: Average return = {average_returns:.2f}")

    end_time = time.time()
    print(f"Training completed! Total time: {end_time - start_time:.2f}s")

    # Save results
    result_file = os.path.join(result_path, f"{exp_name}.pth")
    torch.save(learning_curve, result_file)
    print(f"Saved learning curve to: {result_file}")

    # Save model if requested
    if args.save_model:
        model_file = os.path.join(model_path, f"{exp_name}_agent.pth")
        torch.save(agent.state_dict(), model_file)
        print(f"Saved model to: {model_file}")


if __name__ == "__main__":
    main()
