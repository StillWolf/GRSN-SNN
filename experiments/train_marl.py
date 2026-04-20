"""MARL 训练入口：QMIX + GRSN on SMAC / MockSMAC。

示例：
    # 冒烟（Mock env, CPU, 1000 env steps）
    python experiments/train_marl.py --env MockSMAC --map 8m \
        --rnn_type GRSN --seed 0 --cuda -1 --num_env_steps 1000

    # 真实 SMAC（需先装 SC2 + smac，见 docs/MARL_EXTENSION.md）
    python experiments/train_marl.py --env SMAC --map 8m \
        --rnn_type GRSN --seed 0 --cuda 0 --num_env_steps 10000000
"""
import argparse
import os
import sys
import time
from typing import Dict

import numpy as np
import torch
from ruamel.yaml import YAML

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grsn.algorithms.marl import QMIX, QMixer, AgentNetwork  # noqa: E402
from grsn.buffers.episode_buffer import EpisodeBuffer  # noqa: E402
from grsn.envs.marl import MARLEnv, MockSMAC  # noqa: E402


def make_env(name: str, map_name: str, seed: int, cfg_env: Dict) -> MARLEnv:
    if name == "MockSMAC":
        return MockSMAC(
            n_agents=cfg_env["n_agents"],
            n_actions=cfg_env["n_actions"],
            obs_dim=cfg_env["obs_dim"],
            state_dim=cfg_env["state_dim"],
            episode_limit=cfg_env["episode_limit"],
            seed=seed,
        )
    if name == "SMAC":
        from grsn.envs.marl.smac_wrapper import SMACWrapper
        return SMACWrapper(map_name=map_name, seed=seed)
    raise ValueError(f"unknown env: {name!r}")


def epsilon_schedule(step: int, eps_start: float, eps_end: float, anneal: int) -> float:
    frac = min(step / max(anneal, 1), 1.0)
    return eps_start + frac * (eps_end - eps_start)


@torch.no_grad()
def rollout_one_episode(env: MARLEnv, agent_net: AgentNetwork, epsilon: float,
                        device: torch.device, rng: np.random.RandomState) -> Dict:
    env.reset()
    info = env.get_env_info()
    n_agents = info["n_agents"]
    n_actions = info["n_actions"]
    episode_limit = info["episode_limit"]

    obs_list, state_list, actions_list = [], [], []
    avail_list, reward_list, term_list = [], [], []

    state_t = agent_net.init_hidden(n_agents, device=device)

    terminated = False
    for _ in range(episode_limit):
        obs = env.get_obs()
        state = env.get_state()
        avail = env.get_avail_actions()

        obs_tensor = torch.from_numpy(obs).to(device).unsqueeze(0)
        q, state_t = agent_net(obs_tensor, state_t)
        q = q.squeeze(0)
        q_masked = q.clone()
        q_masked[torch.from_numpy(avail).to(device) == 0] = -1e9
        greedy = q_masked.argmax(dim=-1).cpu().numpy()

        random_actions = np.array([
            rng.choice(np.flatnonzero(avail[a])) for a in range(n_agents)
        ])
        use_random = rng.rand(n_agents) < epsilon
        actions = np.where(use_random, random_actions, greedy).astype(np.int64)

        reward, terminated, _ = env.step(actions)

        obs_list.append(obs)
        state_list.append(state)
        actions_list.append(actions)
        avail_list.append(avail)
        reward_list.append([reward])
        term_list.append([1.0 if terminated else 0.0])

        if terminated:
            break

    obs_list.append(env.get_obs())
    state_list.append(env.get_state())
    avail_list.append(env.get_avail_actions())

    return {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "state": np.asarray(state_list, dtype=np.float32),
        "actions": np.asarray(actions_list, dtype=np.int64),
        "avail_actions": np.asarray(avail_list, dtype=np.int64),
        "rewards": np.asarray(reward_list, dtype=np.float32),
        "terminated": np.asarray(term_list, dtype=np.float32),
        "length": len(reward_list),
    }


def main():
    parser = argparse.ArgumentParser(description="QMIX + GRSN on SMAC/MockSMAC")
    parser.add_argument("--env", type=str, default="MockSMAC", choices=["MockSMAC", "SMAC"])
    parser.add_argument("--map", type=str, default="8m")
    parser.add_argument("--rnn_type", type=str, default="GRSN",
                        choices=["gru", "LIF", "LIFwoTAP", "GRSN", "GRSNwoTAP"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=-1, help="-1 for CPU")
    parser.add_argument("--config", type=str, default=None,
                        help="path to yaml config; default = configs/marl/smac/<map>/qmix_grsn.yml")
    parser.add_argument("--num_env_steps", type=int, default=None,
                        help="override config's train.num_env_steps (for smoke tests)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg_path = args.config or f"configs/marl/smac/{args.map}/qmix_grsn.yml"
    yaml = YAML()
    with open(cfg_path) as f:
        cfg = yaml.load(f)

    if args.num_env_steps is not None:
        cfg["train"]["num_env_steps"] = int(args.num_env_steps)

    device = torch.device("cpu") if args.cuda < 0 else torch.device(f"cuda:{args.cuda}")

    env = make_env(args.env, args.map, args.seed, cfg["env"])
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]
    obs_dim = env_info["obs_shape"]
    state_dim = env_info["state_shape"]
    episode_limit = env_info["episode_limit"]

    agent_net = AgentNetwork(
        obs_dim=obs_dim, n_actions=n_actions,
        rnn_type=args.rnn_type,
        rnn_hidden_size=cfg["agent"]["rnn_hidden_size"],
        obs_embed_size=cfg["agent"]["obs_embed_size"],
        num_layers=cfg["agent"]["num_layers"],
    )
    mixer = QMixer(
        n_agents=n_agents, state_dim=state_dim,
        embed_dim=cfg["mixer"]["embed_dim"],
        hypernet_layers=cfg["mixer"]["hypernet_layers"],
        hypernet_embed=cfg["mixer"]["hypernet_embed"],
    )
    algo = QMIX(
        agent_net, mixer,
        lr=cfg["train"]["lr"],
        gamma=cfg["train"]["gamma"],
        grad_clip=cfg["train"]["grad_clip"],
        device=device,
    )
    buffer = EpisodeBuffer(
        buffer_size=cfg["train"]["buffer_size"],
        episode_limit=episode_limit,
        n_agents=n_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=n_actions,
    )

    rng = np.random.RandomState(args.seed)
    env_steps = 0
    train_iter = 0
    target_update_interval = cfg["train"]["target_update_interval"]
    batch_size = cfg["train"]["batch_size"]
    eps_cfg = cfg["explore"]
    num_env_steps = cfg["train"]["num_env_steps"]

    print(f"[train_marl] env={args.env} map={args.map} rnn={args.rnn_type} "
          f"device={device} num_env_steps={num_env_steps}")
    t0 = time.time()

    while env_steps < num_env_steps:
        epsilon = epsilon_schedule(
            env_steps,
            eps_cfg["epsilon_start"], eps_cfg["epsilon_end"], eps_cfg["epsilon_anneal_time"],
        )
        episode = rollout_one_episode(env, algo.agent_net, epsilon, device, rng)
        ep_len = episode.pop("length")
        env_steps += ep_len
        buffer.insert(episode)

        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            info = algo.train_step(batch)
            train_iter += 1
            if train_iter % target_update_interval == 0:
                algo.target_update()
            if train_iter % 10 == 0 or env_steps >= num_env_steps:
                elapsed = time.time() - t0
                print(f"[train_marl] env_steps={env_steps} iter={train_iter} "
                      f"eps={epsilon:.3f} loss={info['loss']:.4f} "
                      f"q_tot_mean={info['q_tot_mean']:.3f} elapsed={elapsed:.1f}s")

    env.close()
    print(f"[train_marl] done. total env_steps={env_steps} train_iters={train_iter}")


if __name__ == "__main__":
    main()
