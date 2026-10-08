# MGDP_GO2_AMP 当前实验交接

更新时间：2026-10-08（Asia/Shanghai）

> 本文档是当前有效的实验交接入口。`docs/EXPERIMENT_HANDOFF_2026-10-07.md` 仅保留为历史记录；若两者描述冲突，以本文档、仓库代码和云端实际训练日志为准。

## 1. 新对话接手规则

新对话开始后，先进行只读核对，不要假设继承了旧对话上下文：

1. 阅读本文档、根目录 `README.md`、设计规格和实施计划。
2. 执行 `git status --short` 和 `git log -5 --oneline`，核对分支、HEAD 和未提交修改。
3. 检查本地 `legged_gym/outputs/go2_amp/` 中的 TensorBoard event 文件；checkpoint 和实际训练状态以云服务器为准。
4. 先汇报当前状态和下一步，再决定是否修改源码或交接文档。
5. 云端曾手动修改实验系数、输出目录和可视化脚本；不要假设这些改动已经同步或提交到本地仓库。

## 2. 项目目标与边界

项目面向 Unitree Go2 四足机器人，在 MGDP 的感知、世界模型、PPO、控制和多地形任务主体上接入 WMP 风格的 AMP 奖励，用 AMP 替换手工步态风格奖励。Stage 1 用于确定可接受的 AMP 系数并形成基础运动策略；Stage 2 只能从已确认的 Stage 1 checkpoint 继续。

边界：

- `amp_go2`、`MGDP` 和 `WMP` 是参考项目，开发工作位于 `MGDP_GO2_AMP`。
- 不跨 AMP 系数 resume；每个系数只恢复自己的 policy、world model、PPO 和 AMP 状态。
- 在独立评估和多 seed 验证完成前，不宣称 AMP 已改善风格或总体性能。
- 当前优先推进实验，不为方便而扩大源码修改范围。

## 3. 当前仓库状态

截至创建本文档时，本地只读核对结果为：

```text
branch: Test
HEAD:   641e3e7 docs: add experiment handoff
git status --short: clean（创建本文档之前）
```

最近五个提交：

```text
641e3e7 docs: add experiment handoff
4ccc917 fix: support legacy Isaac Gym device arguments
fba3797 fix: default AMP training to compute GPU
e68898d fix: resolve Stage 1 resume paths consistently
9bde65f fix: pair Go2 AMP policy and world-model checkpoints
```

新对话必须重新执行 Git 检查；本文档创建后本地会出现本文档本身的未提交修改。

## 4. 本地实验资料

本地目前保存以下三组 300-iteration 实验的 TensorBoard event 文件，每组包含初始训练和 resume 产生的两个 event 文件：

```text
legged_gym/outputs/go2_amp/stage1_noamp_seed1/stage1_tb/
legged_gym/outputs/go2_amp/stage1_coef0p0005_seed1/stage1_tb/
legged_gym/outputs/go2_amp/stage1_coef0p0010_seed1/stage1_tb/
```

本地项目内另有独立 TensorBoard 环境：

```text
legged_gym/outputs/.venv-tensorboard/
```

该环境用于日志分析，不改变项目训练源码。云端 checkpoint 没有复制到本地；可视化视频也不属于仓库产物。

## 5. Stage 1 已完成实验

所有下表实验均为 `seed=1`、`num_envs=64`，并从各自目录独立训练或独立 resume：

| 系数 | 当前状态 | 结论 |
| ---: | --- | --- |
| `0.0` | 已到 300 iterations | no-AMP 基线，保留并继续延长 |
| `0.0005` | 已到 300 iterations | 当前 AMP 候选，保留并继续延长 |
| `0.001` | 已到 300 iterations | resume 后 episode length 和任务表现明显恶化，暂时排除 |
| `0.0025` | 已做短期比较 | 已排除，本地 event 文件已由用户删除 |
| `0.01` | 已做短期比较 | AMP 奖励主导并明显损害任务学习，已排除，本地 event 文件已由用户删除 |

300-iteration 最后约 10 个点的主要比较：

| 指标 | `coef=0` | `coef=0.0005` | 相对关系 |
| --- | ---: | ---: | --- |
| mean episode length | 约 524 | 约 508 | `0.0005` 约为基线的 96.9% |
| linear-velocity tracking | 约 0.1092 | 约 0.0987 | `0.0005` 约为基线的 90% |
| angular-velocity tracking | 约 0.0715 | 约 0.0549 | `0.0005` 低于基线 |

当前只能认为 `0.0005` 是“低损害候选”，不能认为它已经证明风格更好或任务性能更好。

## 6. 日志解释限制

当前训练日志存在两项已知问题：

1. `AMP/policy_logit` 混合了 rollout 奖励计算和 discriminator update 两个阶段的数据。
2. task/style reward tag 记录的是每轮 rollout 最后一步，而不是完整 rollout 均值。

这些问题影响曲线解释和最终可信度说明，但不改变已经送入 PPO 的训练奖励，不会直接改变机器人策略训练。当前可继续用 episode length、任务奖励、跟踪奖励和固定 checkpoint rollout 做筛选；正式报告前再修正日志并重新取得严谨曲线。

## 7. 可视化与静止问题的诊断结果

已使用 no-AMP 300-iteration checkpoint 做固定速度可视化，确认：

```text
policy:      stage1_noamp_seed1/stage1_nn/last.pt
world model: stage1_noamp_seed1/stage1_nn/wm_last.pt
command x:   约 0.7 m/s
```

典型 eval 数据：

```text
step 100: vel_x ≈ 0.0178, action_mean ≈ 0.5845, action_max ≈ 1.6035
step 300: vel_x ≈ -0.0014, action_mean ≈ 0.5817, action_max ≈ 1.5883
step 500: vel_x ≈ -0.0001, action_mean ≈ 0.5827, action_max ≈ 1.5954
```

结论：

- 固定前进指令已生效，policy/world-model checkpoint 配对正确。
- policy 并非没有输出动作，而是输出很快稳定为近乎恒定的关节目标，形成低伏静态支撑姿态。
- 视频中没有形成周期性抬腿、摆腿和交替支撑；离散的位置变化更像 episode reset，不是连续前进。
- `rew_tracking_lin_vel ≈ 0.1405` 与 `0.7 m/s` 指令下原地静止的理论奖励近似一致。
- 当前最合理判断是训练量不足；300 iterations、64 envs 只适合早期系数筛选，不代表最终运动能力。

Qt/xcb 报错只导致 `Logger.plot_states()` 的 Matplotlib 窗口无法弹出，不会导致机器人静止，也不影响训练。`torch.load(weights_only=False)` 是 FutureWarning，不是本次问题原因。

云端 `play_stage1.py` 曾临时加入每 50 步输出 command、velocity 和 action 统计的诊断代码；本地仓库未同步该手动修改。云端 `vis_stage1.py` 也曾手动设置任务和实验目录。后续应明确区分云端手动状态与本地 Git 状态。

## 8. 当前正在执行的下一步

暂不进入 Stage 2，也不新增 `0.001` 附近的系数。仅继续以下两组：

1. `coef=0`：从当前 300 iterations 独立 resume 700，达到总计约 1000。
2. `coef=0.0005`：从当前 300 iterations 独立 resume 700，达到总计约 1000。

项目 runner 的 resume 语义是：

```python
tot_iter = loaded_iteration + 1 + num_learning_iterations
```

因此从已完成的 300 iterations 继续到总计 1000，应传 `--max_iterations 700`，而不是 1000。

### 8.1 no-AMP 基线

云端运行前手动确认：

```python
# legged_gym/legged_gym/envs/go2_amp/config.py，Stage 1 配置
amp_reward_coef = 0.0
```

```python
# legged_gym/scripts/train_go2_amp_stage1.py
args.output_name = os.path.join(
    LEGGED_GYM_ROOT_DIR, "outputs", "go2_amp", "stage1_noamp_seed1"
)
```

命令：

```bash
cd /root/gpufree-data/MGDP_GO2_AMP

python legged_gym/scripts/train_go2_amp_stage1.py \
  --headless \
  --num_envs 64 \
  --seed 1 \
  --resume \
  --resume_name /root/gpufree-data/MGDP_GO2_AMP/legged_gym/outputs/go2_amp/stage1_noamp_seed1 \
  --checkpoint_model last.pt \
  --max_iterations 700
```

### 8.2 `coef=0.0005`

云端运行前手动确认：

```python
# legged_gym/legged_gym/envs/go2_amp/config.py，Stage 1 配置
amp_reward_coef = 0.0005
```

```python
# legged_gym/scripts/train_go2_amp_stage1.py
args.output_name = os.path.join(
    LEGGED_GYM_ROOT_DIR, "outputs", "go2_amp", "stage1_coef0p0005_seed1"
)
```

命令：

```bash
python legged_gym/scripts/train_go2_amp_stage1.py \
  --headless \
  --num_envs 64 \
  --seed 1 \
  --resume \
  --resume_name /root/gpufree-data/MGDP_GO2_AMP/legged_gym/outputs/go2_amp/stage1_coef0p0005_seed1 \
  --checkpoint_model last.pt \
  --max_iterations 700
```

重要：训练脚本当前会在 Python 代码中覆盖命令行的 `--output_name`。脚本内 `args.output_name` 必须与 `--resume_name` 指向同一实验目录，否则可能从一个目录加载、向另一个目录写入。两组实验不得交叉 resume。

预期启动信息应同时包含同一目录下的：

```text
stage1_nn/last.pt
stage1_nn/wm_last.pt
```

训练进度应从约 `Learning iteration 300/1000` 开始。

## 9. 1000 iterations 后的工作

训练完成后：

1. 将两组新生成的 event 文件分别复制到本地已有的对应 `stage1_tb/` 目录，不要删除前两段 event 文件。
2. 用全部 event 文件拼接分析 300–1000 iteration 的趋势，重点比较 episode length、task reward、linear/angular velocity tracking、world-model loss 和 AMP 稳定性。
3. 分别用各自的 `last.pt + wm_last.pt` 执行：

   ```bash
   python legged_gym/scripts/vis_stage1.py --resume --vis_env_id 0
   ```

4. 保留或重新加入 `[eval]` 输出，观察 step 100–500 的 `cmd`、`vel`、`action_mean` 和 `action_max`，并录制固定视角视频。
5. 根据以下分支决定下一步：
   - 两组都开始连续行走：比较任务保持程度和 AMP 风格表现，决定是否确立 `0.0005`。
   - `coef=0` 行走而 `0.0005` 不行走：AMP 仍抑制学习，暂不采用 `0.0005`。
   - 两组都不行走但曲线持续改善：可在确认后继续到总计 2000 iterations。
   - 两组都不行走且曲线平台化：停止盲目延长，检查奖励平衡、命令课程和静态局部最优。

在完成上述判断前，不启动 Stage 2，不开展多 seed 正式验证。

## 10. 新对话启动提示词

```text
请接手 MGDP_GO2_AMP 项目。项目路径是 D:\New Beginning Help\LZAMP\MGDP_GO2_AMP。

不要假设继承旧对话上下文。请依次阅读：

1. docs/EXPERIMENT_HANDOFF_2026-10-08.md
2. README.md
3. docs/superpowers/specs/2026-09-21-mgdp-go2-amp-design.md
4. docs/superpowers/plans/2026-09-21-mgdp-go2-amp.md

其中 2026-10-08 交接文档是当前有效状态；2026-10-07 文档仅作为历史记录。

随后执行：

git status --short
git log -5 --oneline

先汇报你核对到的代码、分支、未提交修改、实验进度和下一步，不要立即修改源码或交接文档。以新交接文档、仓库代码和云端实际训练日志为准。

当前处于 Stage 1 AMP 系数确立阶段：seed=1、num_envs=64；coef=0 与 coef=0.0005 正准备从 300 iterations 各自独立 resume 700，目标总计 1000；coef=0.001、0.0025、0.01 已暂时排除；不跨系数 resume；暂不进入 Stage 2。

训练完成后，我会把新增 TensorBoard event 文件放入各自对应目录。届时请检查完整曲线、分析 300–1000 iterations 的学习趋势，并结合云端可视化 eval 和视频判断机器人是否形成连续前进步态。

交接文档是否继续更新，由我确认后再修改。
```
