# 环境配置指南

本文档记录在 Linux 上从零部署 `GRSN-SNN` 的完整步骤。已在 Ubuntu 22.04 + Python 3.10 + PyTorch 2.4 实测通过。

## 1. 系统要求

| 组件 | 最低 | 推荐 |
|---|---|---|
| Python | 3.10 | 3.10 / 3.11 |
| 磁盘 | ~5 GB（不含 SC2） | 10 GB+ |
| Conda / Mamba | 任意近期版本 | miniconda3 |
| GPU | 无（CPU 可跑通所有测试 + Mock 训练） | CUDA 12.x（真实训练用） |
| OS | Linux | Linux（SMAC 必须） |

## 2. 一键 conda 环境

```bash
git clone https://github.com/StillWolf/GRSN-SNN.git
cd GRSN-SNN

conda env create -f environments.yml
conda activate grsn
```

**注意：** `conda env create` 在某些版本下会**静默跳过**`pip:` 段（issue 视 conda
版本而异）。**创建后务必验证关键包**：

```bash
python -c "import torch, gym, spikingjelly, numpy, pycolab; \
    print('torch', torch.__version__); \
    print('numpy', numpy.__version__); \
    print('gym', gym.__version__)"
```

预期输出（以下版本是实测可跑通的组合）：
```
torch 2.4.1+cu121
numpy 1.26.4
gym 0.26.2
```

如果 `import torch` 失败，说明 conda 跳过了 pip 段，手动补：
```bash
pip install -r requirements.txt
pip install pycolab
```

## 3. 验证安装

仓库根目录下：

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

预期：**41 个测试全部 PASS**（耗时 CPU 约 1 分钟）。
- 15 个神经元测试（`tests/test_neurons.py`）
- 23 个 MARL 单测（`tests/marl/test_{mock_smac_api,mixer,agent_network,episode_buffer}.py`）
- 3 个 MARL 端到端冒烟（`tests/marl/test_qmix_smoke.py`，单独慢约 40s）

## 4. 跑一个最小冒烟训练

### 4.1 单智能体 POMDP（GRSN on Pendulum-V）

```bash
PYTHONPATH=. python experiments/train.py \
    --env Pendulum-V-v0 --model_type snn --snn_type GRSN \
    --algo td3 --seed 0 --cuda -1
```

`--cuda -1` 强制 CPU；config 在 `configs/pomdp/pendulum/v/rnn.yml`。完整训练
250 iters 在 CPU 上要数十分钟，可用 `Ctrl+C` 提前停。

### 4.2 多智能体 MARL（QMIX + GRSN on MockSMAC）

```bash
PYTHONPATH=. python experiments/train_marl.py \
    --env MockSMAC --map 8m --rnn_type GRSN \
    --seed 0 --cuda -1 --num_env_steps 2000
```

MockSMAC 是无意义的随机环境，只是验证 train loop 跑得动；真实 SMAC 见下节。

## 5. 真实 SMAC 训练（可选，需要额外装 SC2）

### 5.1 装 StarCraft II（Linux headless）

```bash
# Blizzard 官方 Linux 版本（4.6.2 或更高）
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
unzip -P iagreetotheeula SC2.4.10.zip -d ~/StarCraftII
echo 'export SC2PATH=~/StarCraftII' >> ~/.bashrc
source ~/.bashrc
```

参考：https://github.com/Blizzard/s2client-proto#downloads

磁盘开销约 30GB。

### 5.2 装 SMAC maps + Python 包

```bash
git clone https://github.com/oxwhirl/smac.git /tmp/smac
cp -r /tmp/smac/smac/env/starcraft2/maps/SMAC_Maps $SC2PATH/Maps/

pip install smac
```

### 5.3 验证

```bash
python -c "from smac.env import StarCraft2Env; \
    env = StarCraft2Env(map_name='8m'); \
    env.reset(); print('SMAC OK', env.get_env_info())"
```

应能打开 SC2 实例并打印 env_info。

### 5.4 跑训练

```bash
PYTHONPATH=. python experiments/train_marl.py \
    --env SMAC --map 8m --rnn_type GRSN \
    --seed 0 --cuda 0 --num_env_steps 10000000
```

论文复现要求：5 seeds × 6 maps × 10M steps，详见
[`docs/MARL_EXTENSION.md`](MARL_EXTENSION.md)。

## 6. GPU 注意事项

**这是一台共享机器。** 跑 GPU 之前必须先查占用：

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

- 任意 GPU **显存使用 > 80%** 或 **util > 80%**：视为占用，不要再起任务
- 所有 GPU 满：用 `--cuda -1`（CPU）或等空闲

## 7. 常见问题

| 症状 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` 在新装 env 后 | conda 静默跳过 pip 段 | `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'pycolab'` | KeyToDoor 环境的依赖没装 | `pip install pycolab` |
| `AttributeError: module 'collections' has no attribute 'Set'` | Python 3.10+ 把 ABC 移到 `collections.abc` | 已在 `grsn/utils/logger.py` 修过；如又出现说明引入新依赖也犯了同样问题 |
| `Gym has been unmaintained since 2022 ... NumPy 2.0` warning | gym 0.26 + numpy 2 不兼容 | 已在 `environments.yml` 锁 `numpy<2`；如手动 pip install 升级了 numpy，请 `pip install 'numpy<2'` |
| GRSN forward 在 CPU 上很慢 | spikingjelly clock_driven 没有 fast path | 训练用 GPU；冒烟测试用小网络（见 `tests/marl/test_qmix_smoke.py` 的 SMOKE_CONFIG_YAML） |
| `RuntimeError: CUDA error` 启动训练就崩 | 别人的进程把 GPU 显存占满 | 切 CPU 或换 GPU |

## 8. 开发者：从修改到提交

```bash
# 跑测试
PYTHONPATH=. python -m pytest tests/ -v

# 跑单个文件
PYTHONPATH=. python -m pytest tests/test_neurons.py::test_grsn_soft_reset -v

# 单智能体 / 多智能体冒烟
PYTHONPATH=. python experiments/train.py --env Pendulum-V-v0 \
    --model_type snn --snn_type GRSN --algo td3 --seed 0 --cuda -1
PYTHONPATH=. python experiments/train_marl.py --env MockSMAC --map 8m \
    --rnn_type GRSN --seed 0 --cuda -1 --num_env_steps 1000

# 提交（commit message 用中文，不加 Co-Authored-By）
git add <file>
git commit -m "中文描述"
```
