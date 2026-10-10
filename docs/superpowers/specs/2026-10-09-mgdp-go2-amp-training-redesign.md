# MGDP Go2 AMP：任务优先的多地形风格训练设计

日期：2026-10-09
修订：2026-10-10（平地锚点复用零坡地形；首轮 AMP 目标系数改为 0.01；停止门与准入改为人工流程）
状态：设计已由用户确认；实施计划已同步

## 1. 文档地位

本文定义 MGDP_GO2_AMP 下一阶段的 Stage 1 训练架构，取代
`2026-09-21-mgdp-go2-amp-design.md` 中有关奖励边界、训练分布、课程和 AMP
实验流程的决定。旧文档和旧实施计划只用于查找现有代码结构，不能覆盖本文。

本轮目标是先形成可训练、可恢复、可比较的最小闭环，不同时建设论文级诊断与完整消融系统。
现有 64、1024 和 4096 环境实验只作为问题诊断证据，不能决定新架构的最终 AMP 系数。

2026-10-10 修订只改变平地锚点的物理实现：负坡和正坡在 level 0 时已经满足
`difficulty = 0`、`slope = 0`，是完整平地，因此不再新增独立 `flat` 地形类型或第七列。
锚点仍作为显式逻辑分组保留，用于课程解锁后持续提供平地策略样本、分层 replay 和长期对照；
其必要性不再表述为“让训练场景中存在平地”。

同日第二次修订将首轮 AMP 压力测试的 `lambda_target` 从 `0.0005` 改为 `0.01`。旧架构中
`0.01` 曾直接主导奖励并损害任务学习，但新架构增加了 task-only warmup、线性 ramp、弱
`feet_air_time`、前进专用专家集和分层 replay，因此旧结果只作为高风险证据，不作为排除新实验
的依据。`0.01` 用于直接检验 AMP 能否重新形成可辨识的步态约束；`0.0005` 保留为高强度实验
损害任务时的低强度后备候选，不预设任何系数已经胜出。

## 2. 目标与职责分离

项目目标是在 Unitree Go2 上实现感知与运动控制同步训练的多地形策略：

- 感知、深度图、高度图、对比学习、World Model、PPO、控制接口和多地形任务继承 MGDP；
- 所有环境从 terrain level 0 开始，之后按每个环境的任务表现逐级升降难度；
- AMP 从 Go2 专家轨迹学习自然运动风格；
- AMP 替代 `motion_trot`、`motion_bound`、`motion_pace` 等明确规定腿序的奖励；
- 保留任务、安全、动力学约束和弱 `feet_air_time` 接触脚手架。

职责边界固定为：

- 感知、World Model、PPO 和任务奖励负责“往哪里走、能否通过”；
- 安全、动力学约束和 `feet_air_time` 负责动作可执行性与基本抬脚接触周期；
- AMP 只负责“怎样走得更自然”，不教授楼梯高度、沟槽位置或地形探索策略。

正式研究主张为：

> 在不使用显式步态模式奖励的条件下，以 AMP 学习 Go2 专家运动风格，保留最小足端接触、
> 任务、安全和动力学约束，并与 MGDP 感知和 World Model 同步训练，完成渐进式多地形运动控制。

`feet_air_time` 属于保留的手工 locomotion shaping，必须在结果说明中披露。项目不声称 AMP
替代全部手工 locomotion shaping。

## 3. 继承边界

### 3.1 保留 MGDP

- Go2 模型、PD 控制、动作空间、仿真步长和原始任务注册；
- 深度图输入、高度图目标、对比学习、World Model 及其与 PPO 的同步更新；
- Stage 1 `mix` 地形种类、行进距离课程判据、任务奖励和终止逻辑；
- 线/角速度跟踪、姿态、碰撞、关节和力矩限制、动作平滑、`feet_stumble` 等约束；
- 终止前 AMP 状态捕获、策略 checkpoint 和评估入口；
- MGDP 原有“成功超过最高 terrain level 后随机回落到较低 level”的行为。

最高级随机回落首版不修改。逻辑平地锚点复用现有零坡度 level-0 地形，负责在课程解锁后持续
提供平地风格样本；只有实际训练显示高难度 replay 样本不足时，才通过新的独立实验测试
“成功后保持最高级”。

### 3.2 本轮新增或修改

- Go2 AMP 专用的 6 列显式地形 grid；
- 15% 逻辑平地锚点环境；锚点复用负坡列的 level-0 零坡地形，不新增平地生成器或 terrain class；
- 简单前进命令、stance/forward 专家组和三段启动日程；
- 所有地形统一 AMP 系数；
- 锚点/课程双 replay 与 1:1 判别器采样；
- 必要的 rollout 日志、checkpoint 完整恢复和实验目录隔离。

### 3.3 本轮不做

- hurdle、gap、ramp、bream、new stairs down、pit 等进阶复杂地形；
- Stage 2、全向命令、push 鲁棒性和新增地形专家动作；
- terrain-conditioned discriminator 或向 AMP 输入加入 terrain class/height；
- validation AUC、logit 分位数、连续饱和报警和特征屏蔽诊断；
- 自动化 jerk、专家周期距离、盲化视频评估系统；
- 在线奖励重标定、自动调参、梯度投影或新增 `dof_error`；
- 自动停止、自动准入、自动续训、自动回退系数或自动启动下一阶段的实验编排逻辑；
- Original MGDP、Pure AMP、多 seed 等最终论文消融。

这些内容只有在最小闭环训练给出明确需要后，才进入新的设计。

## 4. Stage 1 固定配置

### 4.1 奖励

Go2 AMP 任务关闭显式步态模式奖励：

```python
motion_trot = 0.0
motion_bound = 0.0
motion_pace = 0.0
```

保留弱足端接触脚手架：

```python
feet_air_time = 0.5
```

Go2 AMP 单足原始项固定为：

```text
(min(feet_air_time, 0.75) - 0.5) * first_contact
```

仍使用现有非零平移命令 gate。该截断和权重只应用于 Go2 AMP 新任务，不改变原 MGDP 任务。

其余 Stage 1 已启用的任务、安全和动力学奖励保持不变，包括
`tracking_lin_vel`、`tracking_ang_vel`、`lin_vel_z`、`ang_vel_xy`、`orientation`、
`collision`、`feet_stumble`、`stand_still`、`torques`、`dof_acc` 和 `action_rate`。

### 4.2 命令与随机化

首版只训练基础前进，不引入全向命令课程：

```python
max_init_terrain_level = 0
commands.curriculum = False
commands.heading_command = True

lin_vel_x = [0.0, 0.8]
lin_vel_y = [0.0, 0.0]
heading = [0.0, 0.0]

new_lin_vel_x = [0.0, 0.8]
new_lin_vel_y = [0.0, 0.0]
new_heading = [0.0, 0.0]

push_robots = False
```

保留 `heading_command=True`，使 MGDP 根据相对零航向的误差生成必要的 yaw 修正；首版不采样主动
转向任务。普通和 `new_*` 前进/横向范围保持一致，避免 terrain class 改变命令分布。质量、摩擦、
动作延迟等原 MGDP domain randomization 保持原值。

当前 zero-command gate 会把小于 0.2 m/s 的平移命令置零。在 `[0.0, 0.8]` 均匀采样下，站立命令
比例约为 25%。

### 4.3 专家数据

首版只使用：

- `go2_stance.txt`；
- `go2_forward.txt`；
- `go2_forward_fast.txt`；
- `go2_forward_faster.txt`。

专家组权重固定为 stance 0.25、forward 0.75，与策略的全局站立/前进比例匹配。forward 组内部
继续使用现有 motion weight。后退、横向和转向轨迹不进入首版判别器训练。

AMP 状态和数据格式不变：单帧为 12 维关节位置、3 维机身线速度、3 维机身角速度和 12 维关节
速度；判别器输入为相邻两帧拼接的 60 维向量。AMP 输入不加入地形高度、世界位置、绝对根节点
高度、terrain class 或命令。

## 5. 地形 grid、锚点与课程

### 5.1 显式 6 列 grid

Go2 AMP Stage 1 只使用原 MGDP Stage 1 实际能够采样到的六种简单地形，共 6 列：

| 列 | 地形 |
| ---: | --- |
| 0 | 负坡度斜坡；level 0 时为零坡完整平地，也是逻辑锚点复用位置 |
| 1 | 正坡度斜坡；level 0 时同样为零坡完整平地 |
| 2 | 带粗糙度的金字塔斜坡 |
| 3 | 下楼梯 |
| 4 | 上楼梯 |
| 5 | 离散障碍 |

负坡和正坡都沿用 `slope = difficulty * 0.4`，其中负坡只改变符号；因此 level 0 的
`difficulty = 0` 会使两列都生成完整平地。新增独立 `flat` 生成器、独立平地列或新的平地
terrain class 只会重复已有几何，不属于本轮范围。锚点与普通课程环境是否相同必须由
`is_amp_anchor` 标识，不能由列号、`env_class`、几何形状或 terrain level 反推。

原 MGDP Stage 1 的累计阈值在离散障碍后已经超过 1，因此 `choice` 位于 `[0,1]` 时，后续
`hurdle`、`gap`、`ramp`、`new stairs down` 和 `pit` 分支不会被采样；`bream` 的配置权重本身为
0。新实现用列号显式复现这六种实际 Stage 1 地形，不把原本不可达的复杂分支提前引入首版。

这些复杂地形只有在本 Stage 1 达到准入标准后，才进入新的独立进阶地形训练设计。

### 5.2 逻辑锚点与环境分组

AMP 训练要求 `num_envs >= 2`。锚点数量为：

```text
min(num_envs - 1, max(1, round(num_envs * 0.15)))
```

首版按 Python `round` 语义计算。锚点始终映射到第 0 列并固定为 level 0，但它们使用的就是该列
已有的零坡地形，不是第七种地形。其余课程环境按环境顺序重复使用固定列序列
`(0, 1, 2, 2, 3, 3, 4, 4, 5, 5)`。这使负坡、正坡、粗糙金字塔、下楼梯、上楼梯、离散障碍在
课程环境中的有效比例为 `1:1:2:2:2:2`，复现原 MGDP Stage 1 十列布局，同时不再依赖未归一化
累计阈值。环境数不能整除时，按固定序列的前缀确定余数。

必须明确区分“独立平地几何”和“逻辑锚点配额”。前者已删除，后者仍占总环境的 15%。在
iteration 0--499，所有课程环境都保持 level 0，因此课程环境中的负坡和正坡两类也都是完整
平地，占课程环境的 20%。如果完全不设置锚点，4096 个课程环境按固定序列分配时已有 820 个
完整平地环境（按 20% 比例估算约为 819 个），足以否定“必须新增平地地形才能启动基础运动”
的前提。按本设计保留逻辑锚点时，锚点为 614 个，剩余 3482 个课程环境按固定序列恰有 698 个
属于 level-0 零坡列，完整平地总数为 1312，即约占全部环境的 32%。这个启动期比例是
“15% 固定锚点 + 课程地形原有 20% 零坡列”的直接结果，不得把它误写为只有 15%，也不能作为
新增独立平地列的理由。

15% 锚点的设计价值只体现在 iteration 500 以后：普通坡地环境随课程升降难度，锚点仍固定在
level 0，为平地专家轨迹提供持续、可识别的同域 policy 样本，并为双 replay 与长期平地评估
提供稳定对照。锚点比例是首版固定实验参数，不主张它是理论最优值；只有完成最小闭环后，才在
新的独立实验中比较其他比例或无锚点方案。新增独立布尔 `is_amp_anchor`，不得从 `env_class`、
列号或 `terrain_level` 推断锚点。

锚点环境参与 PPO、感知、World Model 和 AMP reward，不使用独立策略，也不冻结任何网络。

### 5.3 课程行为

所有课程环境从 level 0 开始。课程解锁后继续使用 MGDP 原有行进距离判据：成功上升一级，失败
下降一级；成功超过最高级后仍按 MGDP 原逻辑随机回落。锚点在任何 reset 或 curriculum update
后都重新映射到第 0 列的 level-0 零坡地形，不进入升降级逻辑。

## 6. AMP 启动日程

日程使用 runner 的零基 policy iteration：

1. iteration 0--99：所有课程环境保持 level 0；策略只接收任务、约束和 `feet_air_time`；
   判别器不更新，AMP 有效系数为 0；策略转移仍写入双 replay。
2. iteration 100--499：课程环境仍保持 level 0；每轮更新判别器一次；AMP 系数线性升高。
3. iteration 500 起：AMP 使用目标系数，非锚点课程环境在后续 reset 时开始执行难度升降；不强制
   在 iteration 500 统一 reset。

目标系数为 `lambda_target`，有效系数为：

```text
iteration <= 100: 0
100 < iteration < 499: lambda_target * (iteration - 100) / 399
iteration >= 499: lambda_target
```

因此 iteration 100 的 rollout 不接收 AMP reward，iteration 499 首次使用完整目标系数。判别器
从 iteration 100 的 rollout 结束后开始更新，更新后的判别器供下一轮 rollout 使用。

Scaffold-only 使用相同日程和地形解锁时间，但 `lambda_target = 0`；判别器仍从 iteration 100
开始训练，以保证日志和训练开销可比。

首轮 AMP 压力测试固定 `lambda_target = 0.01`，关键检查点的有效系数为：

| Policy iteration | `lambda_effective` |
| ---: | ---: |
| 0--100 | 0 |
| 200 | 约 0.00251 |
| 250 | 约 0.00376 |
| 300 | 约 0.00501 |
| 400 | 约 0.00752 |
| 499 起 | 0.01 |

因此 20-iteration smoke 只验证工程链路，不能判断 `0.01` 的训练效果；iteration 200 时有效系数
已经接近旧实验中被排除的 `0.0025`，必须从 200 起分阶段检查，不得直接盲跑到 1000。

## 7. AMP reward、replay 与更新

### 7.1 奖励

原始 AMP 映射保持不变：

```text
r_amp_raw = max(0, 1 - (D(s, s') - 1)^2 / 4)
r_amp_contribution = lambda_effective * r_amp_raw
r_total = r_mgdp + r_amp_contribution
```

`r_mgdp` 是 MGDP 环境交给 runner 的最终任务奖励。首版所有地形使用相同
`lambda_effective`，不使用当前按 `env_class` 区分的 1.0/0.25 gate。兼容 gate 可以保留，但默认
easy/hard 值必须相同。

### 7.2 双 replay

锚点与课程环境分别写入两个 replay pool。两个 pool 各保存最近 2 个 rollout：

```text
anchor_capacity = 2 * 24 * num_anchor_envs
course_capacity = 2 * 24 * num_course_envs
```

每次判别器更新抽取 512 条策略转移，其中锚点 256 条、课程 256 条，并配对 512 条专家转移。
pool 非空但样本少于 256 时使用有放回采样；任一 pool 为空时跳过该轮判别器更新并记录原因，
不得退化成非分层采样。

### 7.3 固定参数与更新顺序

```text
num_steps_per_env = 24
amp_updates_per_iter = 1
amp_batch_size = 512
amp_replay_rollouts = 2
amp_learning_rate = 1e-4
amp_gradient_penalty_coef = 10.0
```

正式云端训练通过 `--num_envs 4096` 指定并行环境数，不写死在配置中。仿真 `dt=0.005 s`、控制
`decimation=4`，每个策略步为 0.02 s，24 步为 0.48 s rollout。

每轮顺序固定为：

1. 用当前判别器收集 24 步 rollout、计算 reward 并写 replay；
2. PPO 使用该 rollout 的固定 reward 更新策略；
3. 从双 replay 和专家数据采样；
4. 执行一次 least-squares 判别器更新和专家梯度惩罚；
5. 新判别器只供下一轮 rollout 使用。

判别器损失保持：

```text
0.5 * mean((D(policy) + 1)^2)
+ 0.5 * mean((D(expert) - 1)^2)
+ amp_gradient_penalty_coef * gradient_penalty
```

判别器优化器不得包含 PPO policy 或 World Model 参数。单个 run 内不自动改变更新次数、batch、
学习率、replay 长度、梯度惩罚或 AMP 系数。

`amp_updates_per_iter = 1` 是首轮固定起点，不声明为跨实验最优值。如果 AMP reward 非零但
`r_amp_raw` 和 policy logit 在策略样本间接近常数，必须先在新的独立实验中检查判别器更新不足，
不能在原 run 内动态增加更新次数，也不能只靠继续提高 `lambda_target` 放大无区分度的信号。

## 8. 最低限度日志与恢复

### 8.1 日志

必须记录：

- 完整 rollout 的 `task_reward`、`amp_reward_contribution`、`total_reward` 均值；
- `r_amp_raw` 的 mean、std 和 zero fraction；
- `amp_reward_contribution` 的 mean 和 std；
- `policy_logit_rollout` 与 `policy_logit_update` 的 mean 和 std，保持两个阶段的 tag 分离；
- 判别器 loss、gradient penalty、expert logit mean/std 和实际更新次数；
- `mean(abs(amp_reward_contribution)) / max(mean(abs(task_reward)), 1e-8)`；
- `feet_air_time` episode reward；
- terrain level 均值；
- 锚点/课程 replay 样本数，以及两组分别计算的 policy logit、`r_amp_raw`、AMP contribution 和
  更新跳过次数；
- 实际锚点数、level-0 课程环境数，以及两者合计的完整平地环境占比，避免把锚点比例误当成
  启动期全部平地比例。

日志只用于确认链路、奖励尺度和课程是否工作，不在训练中自动调参。

### 8.2 Checkpoint

checkpoint 必须保存并恢复：

- 判别器、优化器和归一化器；
- 两个 replay pool 的内容、容量、游标和当前大小；
- policy iteration 与实际判别器更新计数；
- 影响兼容性的 AMP 配置：目标系数、日程边界、锚点比例、更新次数、batch、学习率、replay
  rollouts、梯度惩罚和专家组。

resume 时这些配置必须一致，不允许静默加载部分 AMP 状态。policy-only warm start 入口保留，但
不用于本轮正式实验。新架构正式训练必须从零开始；旧 checkpoint 只用于历史可视化。

## 9. 人工实验流程与准入判据

本节规定研究者实际训练时应采用的分阶段实验流程、停止门和准入判据，属于实验操作文档，
不属于训练代码的自动控制需求。代码只需提供可选择的实验配置、所需日志、checkpoint 保存/恢复
和独立输出目录；是否停止当前 run、是否延长到下一检查点、是否切换后备系数，以及是否进入
Stage 2，均由研究者结合本节指标、固定 rollout 和视频人工判断，并可与后续协作分析共同决定。

### 9.1 首轮配对实验与后备系数

在相同 Go2、`num_envs`、24 steps、地形、命令、seed 1 和训练步数下，从零训练：

1. **Scaffold-only**：`lambda_target=0`；
2. **AMP + scaffold 压力测试**：`lambda_target=0.01`。

`0.01` 的目的不是预设最终系数，而是检验新 warmup/ramp/replay 架构能否在 AMP 明确具有足够
奖励尺度时，同时形成可辨识风格变化并保住任务学习。`0.0005` 保留为后备候选：只有 `0.01`
产生风格变化但明显损害任务时，才用新的独立 run 测试 `0.0005` 或经明确批准的中间系数；不同
系数不得相互 resume。

运行时必须能明确选择系数和独立输出目录，不通过修改源码在两个实验间切换。推荐目录名包含
`task_priority`、系数和 seed，避免覆盖旧实验或相互 resume。

训练顺序：

- 20 iterations：有限值、保存和恢复 smoke；AMP 尚未启动，不评价 `0.01`；
- 100 iterations：保存 task-only warmup checkpoint，作为 ramp 前状态记录；
- 200 iterations：第一次 AMP 影响检查，`lambda_effective` 约为 `0.00251`；
- 250 iterations：奖励、判别器和任务趋势诊断，`lambda_effective` 约为 `0.00376`；
- 500 iterations：完整 `0.01` 与课程解锁检查；研究者按第 9.3--9.4 节人工判断是否继续；
- 1000 iterations：早期运动与多地形筛选；人工确认通过后，才安排至少 10000 iterations 的正式
  训练。

### 9.2 最低评估

固定 0.7 m/s 前进命令，使用相同地形种子、相机和 rollout 长度比较两组：

- 任务：行进距离/成功、线速度跟踪、跌倒和 terrain level 趋势；
- 风格：固定视频和接触期足端滑移；
- AMP：`r_amp_raw`/contribution 的均值、方差和零比例，policy/expert logit 分布，以及锚点/课程
  分组结果；
- MGDP：感知、对比学习和 World Model loss 有限，无 NaN/Inf 或明显常数坍缩。

首版不建设额外自动化风格评估框架。现有可直接取得的指标与固定视频足以判断是否值得进入下一
阶段；正式论文指标在 Stage 1 通过后再单独实现。

### 9.3 Stage 1 人工准入判据

必须同时满足：

- **运动控制有效**：固定前进评估中能在平地、坡面、楼梯及阶段内其他地形持续运动，课程 level
  有上升趋势，不能依赖 reset 位移；
- **MGDP 链路正常**：感知、World Model 和 PPO 指标有限且无明显坍缩；
- **AMP 有区分度**：AMP contribution 不长期为零，`r_amp_raw` 在策略样本间具有非退化方差，
  policy logit 不长期饱和或退化为近似常数；单纯存在一个非零的近常数 AMP bonus 不算通过；
- **AMP 确实改变风格**：相对 Scaffold-only，AMP 组在固定视频或接触期足端滑移中出现可重复的
  风格差异，且这种差异不是仅靠降低实际速度、延长站立或频繁 reset 获得；
- **AMP 不阻断任务**：AMP 组仍能完成 Scaffold-only 已经稳定完成的阶段地形；
- **工程链路正常**：CPU 测试、短 rollout、日志、checkpoint 保存与恢复通过，无 NaN/Inf。

这些条目是人工评审清单，不要求代码计算综合分数、输出 pass/fail 或阻止训练进程。人工确认
全部满足前，不进入 Stage 2、全向命令，也不宣称 AMP 已成功替代显式步态奖励。

### 9.4 人工分阶段决策

以下规则由研究者在各检查点查看日志、checkpoint、固定 rollout 和视频后执行，不编码为训练循环
中的硬停止门，也不由脚本自动启动新 run：

- 任一检查点出现 NaN/Inf、checkpoint 无法恢复或任务指标持续崩溃时，不继续延长该 run；
- `0.01` 产生可重复风格变化且任务保持时，继续到下一检查点；
- `0.01` 产生风格变化但明显损害任务时，停止延长并从零测试 `0.0005` 或批准后的中间系数；
- AMP reward 非零但策略风格与 Scaffold-only 无可辨识差异时，先检查判别器输出是否近似常数；
- AMP reward 大量为零时，先检查 policy logit 饱和、归一化器和判别器更新，不把“继续训练”当作
  默认修复；
- 不允许依据单个 TensorBoard 均值认定 AMP 有效，最终判断必须同时使用任务、AMP 分布和固定
  rollout 行为。

## 10. 实施范围

首版只修改：

- `legged_gym/legged_gym/envs/go2_amp/config.py`：奖励、命令、专家组、terrain 和 AMP 参数；
- Go2 AMP 专用地形构建与环境分配代码：6 列、锚点复用第 0 列 level 0、`is_amp_anchor`、
  课程冻结/解锁；不得新增独立 `flat` 生成器或 terrain class；
- `legged_gym/rl/MGDP/amp/replay.py`：可保存的双 replay 基础能力；
- `legged_gym/rl/MGDP/amp/session.py`：日程、统一 reward、分层采样、固定更新和 checkpoint；
- `legged_gym/rl/MGDP/amp/discriminator.py`：可配置梯度惩罚；
- `legged_gym/rl/MGDP/runners/policy_runner.py`：iteration 传递、rollout 聚合和 checkpoint；
- 训练参数解析/Stage 1 wrapper：运行时系数选择和输出目录隔离；
- 相关单元测试和 README。

实施范围明确不包含第 9 节的人工作业编排：不实现自动停止/续训、自动准入判断、自动系数回退、
自动创建下一阶段实验或根据指标修改当前 run。训练程序到达指定 iteration 时正常保存和记录即可，
由研究者决定是否再次启动或延长训练。

不得修改原 MGDP task 的默认行为。新实验使用 `go2_amp_stage1_task_priority` experiment name。
现有未跟踪文件 `legged_gym/scripts/record_go2_amp_stage1.py` 和
`tests/test_record_go2_amp_stage1.py` 不属于本轮范围，不修改、不暂存、不提交。

## 11. 必要测试与验证

CPU 单元测试至少覆盖：

- Stage 1 配置中的奖励、普通/`new_*` 命令、level 0、push 和专家组；
- Go2 AMP `feet_air_time` 的 0.75 s 截断不影响原任务；
- 6 列显式原 MGDP Stage 1 简单地形，以及课程环境 `1:1:2:2:2:2` 的原有效比例；
- 负坡/正坡在 level 0 时确实生成零坡完整平地，且没有独立 `flat` 生成器、平地列或新 terrain
  class；
- 15% 锚点、`is_amp_anchor`、锚点复用第 0 列并永久保持 level 0；
- iteration 0--499 的完整平地占比同时计入锚点和课程环境中的两类零坡列；4096 环境按固定
  序列分配时为 614 个锚点加 698 个课程零坡环境，而不是只有 15%；
- 课程解锁前冻结，解锁后使用 MGDP 原升降与最高级随机回落；
- 三段 AMP 系数和判别器更新边界，以及 `lambda_target=0.01` 时 iteration 200、250、300、400、
  499 的有效系数；
- 双 replay 容量、1:1 采样、有放回采样、空池跳过和 state round-trip；
- 统一 terrain gate、固定判别器更新次数和可配置梯度惩罚；
- rollout 指标为完整均值，`r_amp_raw`/contribution/logit 的 mean/std 与 `r_amp_raw` zero fraction
  正确聚合，且 rollout/update logit tag 分离；
- 运行时接受 `0`、`0.01` 和后备 `0.0005`，不同系数输出目录隔离且不允许相互 resume；
- checkpoint 完整恢复与不兼容配置拒绝；
- 原 MGDP task config 不变。

建议人工实验顺序为：完整 CPU 测试、compileall、20-iteration GPU smoke、100-iteration task-only
状态保存、4096×200/250 分段诊断、4096×500 完整 `0.01` 检查、4096×1000 早期筛选。研究者按
第 9 节判断前一检查点是否通过，再决定是否延长；代码测试只验证产生判断所需的数据与恢复能力，
不测试或实现自动停止门。只有后续云端长训练可以用于多地形性能判断。
