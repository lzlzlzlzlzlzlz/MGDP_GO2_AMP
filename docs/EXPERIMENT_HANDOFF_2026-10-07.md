# MGDP_GO2_AMP 实验与开发交接

更新时间：2026-10-07

## 1. 新对话接手规则

新对话开始后，应先完成以下只读检查，再修改代码或启动实验：

1. 阅读本文件、根目录 `README.md`、设计规格和实施计划。
2. 执行 `git status --short` 与 `git log -5 --oneline`，确认实际工作树和最新提交。
3. 检查云服务器上的实验目录、TensorBoard 日志和 checkpoint；这些训练产物目前未保存在本地仓库。
4. 以代码、Git 状态和训练日志为准；本文件记录的是截至 2026-10-07 的交接状态。
5. 每完成一组实验或一次关键修改，就更新本文档中的实验表、结论、下一步和最新提交。

## 2. 项目目标

研究对象为 Unitree Go2 四足机器人。项目保留 MGDP 的感知、世界模型、PPO、控制与多地形任务主体，用 WMP 风格的 AMP 奖励替换 MGDP 中手工设计的步态风格奖励。Stage 1 与 Stage 2 均使用 AMP。专家动作来自 `amp_go2/datasets/go2_motion`。

需要保持的边界：

- `amp_go2`、`MGDP`、`WMP` 是参考项目，不在其中继续开发。
- 新功能与实验配置放在 `MGDP_GO2_AMP`。
- MGDP 的任务奖励、地形、感知、世界模型和控制框架保持主体不变。
- 新任务中关闭原有手工步态风格奖励，保留任务相关奖励。
- 在独立评估完成前，不宣称 AMP 已改善运动风格或总体性能。

## 3. 已实现内容

- 新增 Go2 任务：`go2_amp_stage1`、`go2_amp_stage2`。
- 两个阶段均接入 AMP 判别器、专家回放、策略样本回放和梯度惩罚。
- 专家数据由 `amp_go2/datasets/go2_motion` 选取并复制到本项目，共 17 条轨迹。
- AMP 状态维度为 30，连续两帧组成 60 维 transition 输入。
- 自动 reset 时保留 terminal state，避免把 reset 后状态误作 transition 终点。
- checkpoint 包含策略/PPO、AMP 判别器、判别器优化器、AMP normalizer，以及世界模型配对状态。
- 已处理 Isaac Gym 旧版设备参数兼容和 AMP 默认计算 GPU 配置。
- 本机曾完成静态检查、数据检查和无需 PyTorch/Isaac Gym 的测试；GPU 训练由云端继续验证。

关键目录：

- `legged_gym/legged_gym/envs/go2_amp/`
- `legged_gym/rl/MGDP/amp/`
- `legged_gym/scripts/train_go2_amp_stage1.py`
- `legged_gym/scripts/train_go2_amp_stage2.py`
- `tests/`
- `docs/superpowers/specs/2026-09-21-mgdp-go2-amp-design.md`
- `docs/superpowers/plans/2026-09-21-mgdp-go2-amp.md`

截至交接时已知的最近提交包括：

- `4ccc917 fix: support legacy Isaac Gym device arguments`
- `fba3797 fix: default AMP training to compute GPU`
- `e68898d fix: resolve Stage 1 resume paths consistently`

新对话必须用 `git log` 重新核对，因为交接文档本身以及后续修改可能产生更新提交。

## 4. 云端运行状态

- Isaac Gym/PyTorch/CUDA 环境能够启动项目。
- Stage 1 短跑成功，200 iteration 训练也能够完成。
- 当前已比较的两组实验均使用 `seed=1`、`num_envs=64`、200 iterations。
- 云端原始 event 文件、checkpoint 和视频没有随本交接文档复制到本地；以下数字来自用户提供的 TensorBoard 图。

## 5. 已完成的系数对比

| 实验 | `amp_reward_coef` | seed | envs | iterations | 约 200 iter 的结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| no AMP baseline | 0.0 | 1 | 64 | 200 | episode length 约 520；task reward 约 `3.3e-4`；style reward 为 0 |
| AMP high coefficient | 0.01 | 1 | 64 | 200 | episode length 约 180；task reward 约 `0.8e-4`；style reward 约 `5.5e-3` |

观察与当前判断：

- 两组的 `terrain_level` 都在约 30 iteration 降到 0 并保持，因此后续差距不能归因于 AMP 组进入了更难地形。
- `0.01` 组的 style reward 明显大于 task reward，AMP 信号主导了总奖励。
- `0.01` 组的 episode length 与 task reward 都明显弱于 `0.0` 组；该系数当前过大。
- AMP 判别器相关曲线表明链路正在工作，但不同实验各自训练了判别器，其 logit 不能直接作为跨实验的风格优劣分数。
- `Train/mean_reward` 混入了不同权重的 style reward，不能直接用于不同 AMP 系数的公平比较。
- 世界模型损失整体相近；某些损失更低可能来自更窄或更简单的状态分布，不能单独解释为感知效果更好。

## 6. 下一组实验

先在 Stage 1 做同条件系数扫描：

| `amp_reward_coef` | 建议实验名 |
| ---: | --- |
| 0.0 | `stage1_coef0p0000_seed1`（已有，可复用） |
| 0.0005 | `stage1_coef0p0005_seed1` |
| 0.001 | `stage1_coef0p0010_seed1` |
| 0.0025 | `stage1_coef0p0025_seed1` |
| 0.01 | `stage1_coef0p0100_seed1`（已有，可复用） |

实验约束：

- 全部使用 `seed=1`、`num_envs=64`、相同 iteration 数（建议 200 至 300）和相同其他超参数。
- 每个系数使用新的输出目录，不覆盖旧实验。
- 系数扫描必须从头训练，不从另一系数的 checkpoint resume。
- 每次运行前同时核对配置中的 `amp_reward_coef` 和训练脚本中的输出目录/实验名。
- 记录完整启动命令、Git commit、seed、系数、环境数、iteration、checkpoint 和 TensorBoard 目录。

推荐输出目录命名：

```text
outputs/go2_amp_stage1/stage1_coef0p0005_seed1
outputs/go2_amp_stage1/stage1_coef0p0010_seed1
outputs/go2_amp_stage1/stage1_coef0p0025_seed1
```

若训练入口使用脚本内硬编码的日志目录，应逐次修改该路径；更好的后续改进是新增命令行参数，例如 `--run_name` 或 `--output_dir`，并让配置和有效参数写入运行目录。

## 7. 系数选择标准

先排除明显破坏任务能力的系数，再评估风格：

1. episode length、task-only return、命令速度跟踪和终止率达到 `coef=0.0` 基线的约 80% 至 90% 以上。
2. 固定 checkpoint 做独立 rollout，比较身体高度、姿态、足端轨迹、关节速度/加速度、足端滑移、落足节律和视频。
3. 使用冻结的独立风格评估器或 held-out 专家数据评估风格；不要只比较各训练运行自己的判别器 logit。
4. 找到候选系数后，再使用 seeds 1、2、3 重复实验并报告均值与波动。
5. Stage 2 使用与其对应的 Stage 1 系数、seed 和 checkpoint，保持配对关系。

## 8. 已知日志问题与优先代码任务

现有日志足以证明 AMP 链路在运行，但不足以严谨比较系数：

- `amp_rollout_metrics` 目前可能只记录 rollout 最后一步，而非完整 rollout 平均值。
- `AMP/policy_logit` 可能同时被 rollout 奖励计算与判别器更新写入，造成同一 tag 混入两类数据和锯齿曲线。
- 缺少 episode 级 task-only return 与 style-only return。
- 运行目录中缺少完整有效配置与 Git commit 记录。

下一项代码工作建议先修日志：

- 分成 `AMP/rollout_policy_logit` 与 `AMP/update_policy_logit`。
- 对完整 rollout 聚合 task/style/total reward。
- 记录 episode 级 task-only return、style-only return、终止原因和成功率。
- 自动保存 seed、AMP 系数、num_envs、命令行、有效配置、Git commit。
- 给训练脚本增加 `--run_name`/`--output_dir` 和 `--amp_reward_coef` 覆盖参数，避免为每个实验改源码。

完成这些修改后，应先做一次短跑，确认 tag、配置快照、checkpoint 和输出目录正确，再开始正式系数扫描。

## 9. 新对话的首个工作目标

新对话应按以下顺序推进：

1. 核对 Git 与云端实验文件，补全本交接文档中缺失的实际路径和 commit。
2. 检查当前训练脚本参数接口，实施并测试日志隔离、rollout 聚合及运行元数据记录。
3. 给出不改源码即可启动三组新系数实验的精确命令。
4. 用户在云端运行后，根据 event 文件、CSV、checkpoint 或截图更新实验表并选择候选系数。
5. 候选系数通过多 seed 后，再进入 Stage 2。

## 10. 新对话启动提示词

将下面内容粘贴到以 `MGDP_GO2_AMP` 为工作目录的新 Codex 对话：

```text
请接手 MGDP_GO2_AMP 项目。项目路径是 D:\New Beginning Help\LZAMP\MGDP_GO2_AMP；如果当前在云服务器，请使用该仓库在云端的实际绝对路径。

先阅读：
1. docs/EXPERIMENT_HANDOFF_2026-10-07.md
2. README.md
3. docs/superpowers/specs/2026-09-21-mgdp-go2-amp-design.md
4. docs/superpowers/plans/2026-09-21-mgdp-go2-amp.md

然后执行 git status --short 和 git log -5 --oneline，核对代码、分支和未提交修改。不要假设新对话继承了旧对话的上下文；以交接文档、仓库代码和云端训练日志为准。

当前阶段是 Stage 1 AMP 系数对比。已有 seed=1、num_envs=64、200 iterations 的 coef=0.0 与 coef=0.01 实验，0.01 明显损害 episode length 和 task reward。下一步先完善实验日志与命令行覆盖参数，再从头训练 coef=0.0005、0.001、0.0025，使用独立输出目录，不覆盖旧实验，不跨系数 resume。

请先汇报你核对到的当前状态和准备修改的文件，再逐步实施、运行本机可运行的测试，并给出云端精确训练命令。每完成一组实验或关键修改，都更新交接文档。
```

