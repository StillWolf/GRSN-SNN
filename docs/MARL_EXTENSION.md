# MARL (SMAC + QMIX) 扩展指南

当前仓库只实现了论文 arXiv:2404.15597 (AAAI'25) 的 **POMDP 单智能体** 部分。
论文第二半的 MARL 实验（QMIX on StarCraft Multi-Agent Challenge）需要的是
在 QMIX 的 agent RNN 位置替换成 GRSN。以下是已经预留的接入点。

## 可直接复用的模块

- `grsn.policies.rlifs.GRSN` / `GRSNwoTAP` / `LIF` / `LIFwoTAP` — 与上下游
  策略无关的 RNN 单元。接口：
  ```python
  cell = GRSN(input_size, hidden_size, num_layers)
  spikes, final_state = cell(inputs, initial_state)
  # inputs: (T, B, input_size)
  # spikes: (T, B, hidden_size)
  # final_state: (num_layers, B, state_size_per_layer)
  #   - LIF / LIFwoTAP: state_size_per_layer = hidden_size
  #   - GRSN / GRSNwoTAP: state_size_per_layer = 3 * hidden_size
  #     （h, c, spike_prev 三部分拼接——推理时务必把 final_state 回传才能保留门控递归）
  ```

## 需要新增的组件

1. **SMAC 环境 wrapper**：放在 `grsn/envs/marl/smac_wrapper.py`
2. **QMIX + Mixer**：参考 PyMARL / epymarl 的实现，把 agent RNN 从 GRU
   替换为 `GRSN`；mixer 部分保持 monotonic 网络不变。代码放 `grsn/algorithms/marl/qmix_snn.py`。
3. **训练循环**：MARL 的 rollout / replay 与单智能体不同，不要复用
   `experiments/train.py`；另起 `experiments/train_marl.py`。
4. **Config**：`configs/marl/smac/{scenario}/qmix_grsn.yml`。

## 论文对齐要点（实现时查 arXiv:2404.15597）

- SMAC 场景：8m、2s3z (easy)；8m_vs_9m、3s_vs_5z (hard)；27m_vs_30m、MMM2 (super hard)
- 训练步数：10M
- Seeds：5
- SNN 仿真步：T=1（TAP 对齐）
- 使用 CTDE (centralized training decentralized execution)
