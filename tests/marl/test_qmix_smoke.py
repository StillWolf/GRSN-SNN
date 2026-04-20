"""QMIX 端到端冒烟测试：在 MockSMAC 上跑 train_marl 若干 env steps，验证无异常。

使用极简 config（n_agents=4，obs_dim=16，hidden=16，episode_limit=25），
让 GRSN Python-loop 反向传播在 CPU 上每 train_step ~1-2s，整组测试可在 120s 内完成。

MockSMAC 每 episode 20-25 env steps（randint[20, episode_limit]），
batch_size=4 → 填满 buffer 需 ~80-100 步；300 步给足多次 train_step 触发空间。
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]


SMOKE_CONFIG_YAML = textwrap.dedent("""\
    env:
      map_name: 8m
      n_agents: 4
      n_actions: 6
      obs_dim: 16
      state_dim: 32
      episode_limit: 25

    agent:
      rnn_type: GRSN
      rnn_hidden_size: 16
      obs_embed_size: 16
      num_layers: 1

    mixer:
      embed_dim: 8
      hypernet_layers: 1
      hypernet_embed: 16

    train:
      lr: 5.0e-4
      gamma: 0.99
      grad_clip: 10.0
      batch_size: 4
      buffer_size: 8
      target_update_interval: 5
      num_env_steps: 300

    explore:
      epsilon_start: 1.0
      epsilon_end: 0.05
      epsilon_anneal_time: 100

    eval:
      interval_env_steps: 100
      num_episodes: 2
""")


@pytest.fixture(scope="module")
def smoke_config_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("smoke") / "qmix_smoke.yml"
    path.write_text(SMOKE_CONFIG_YAML)
    return str(path)


def _run_train(rnn_type: str, smoke_config_path: str, num_env_steps: int,
               timeout: int = 120) -> str:
    """运行 train_marl.py 若干步，返回 stdout。"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    result = subprocess.run(
        [
            sys.executable, str(REPO / "experiments" / "train_marl.py"),
            "--env", "MockSMAC",
            "--map", "8m",
            "--rnn_type", rnn_type,
            "--seed", "0",
            "--cuda", "-1",
            "--config", smoke_config_path,
            "--num_env_steps", str(num_env_steps),
        ],
        env=env, capture_output=True, text=True, timeout=timeout,
    )
    assert result.returncode == 0, (
        f"train_marl failed (rnn={rnn_type}):\nSTDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    return result.stdout


def test_smoke_gru(smoke_config_path):
    """最快的 RNN，先验证整个训练 loop 能跑通。"""
    out = _run_train("gru", smoke_config_path, num_env_steps=300, timeout=120)
    assert "[train_marl] done" in out
    assert "loss=" in out, f"no training step occurred; output:\n{out}"


def test_smoke_grsn(smoke_config_path):
    """论文主模型。CPU 上 Python-loop SNN，单 train_step ~1-2s；300 步 + 120s 超时足够。"""
    out = _run_train("GRSN", smoke_config_path, num_env_steps=300, timeout=120)
    assert "[train_marl] done" in out
    assert "loss=" in out, f"no training step occurred; output:\n{out}"


def test_smoke_grsn_wo_tap(smoke_config_path):
    """GRSN 无 TAP 消融（time_step=4，每 MDP 步跑 4 个 SNN 子步）。与 GRSN 相近，120s 超时。"""
    out = _run_train("GRSNwoTAP", smoke_config_path, num_env_steps=300, timeout=120)
    assert "[train_marl] done" in out
    assert "loss=" in out, f"no training step occurred; output:\n{out}"
