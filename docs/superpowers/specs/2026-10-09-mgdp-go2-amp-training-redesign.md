# MGDP Go2 AMP：任务优先的多地形风格训练重设计

日期：2026-10-09

## 1. 文档地位

本文定义 MGDP_GO2_AMP 的下一阶段训练架构，取代
`2026-09-21-mgdp-go2-amp-design.md` 中有关奖励边界、训练分布、课程和 AMP
实验流程的设计。旧文档继续保留，作为最初集成方案的历史记录；其中已经验证的数据解析、
30 维 AMP 状态映射、终止转移、checkpoint 和 Go2 任务隔离设计仍然有效。

本文不修改 `EXPERIMENT_HANDOFF_2026-10-08.md`。现有 64、1024 和 4096 环境实验只作为
问题诊断证据，不再用于确定新架构下的最终 AMP 系数。

## 2. 目标与研究主张

项目目标是在 Unitree Go2 上实现感知与运动控制同步训练的多地形策略：

- 感知、世界模型、深度图与高度图对比学习、PPO 主体和控制接口继承 MGDP；
- 地形种类和多地形任务定义继承 MGDP；
- 地形难度初始化与逐级课程采用 WMP 的核心做法；
- AMP 从 Go2 专家轨迹中学习运动风格；
- AMP 替代明确规定 trot、bound、pace 等腿序和步态类型的手工奖励；
- 保留最小足端接触脚手架、任务、安全和动力学约束。

系统采用明确的职责分离：感知、PPO 和任务奖励负责“往哪里走、能否通过”；安全、动力学与
最小足端脚手架负责动作可执行性；AMP 只负责“怎样走得更自然”。平地专家数据提供运动风格
先验，不承担教授楼梯高度、沟槽位置或地形探索策略的职责。

正式研究主张调整为：

> 在不使用显式步态模式奖励的条件下，以 AMP 学习 Go2 专家运动风格，保留最小足端接触、
> 任务、安全和动力学约束，并与 MGDP 感知和世界模型同步训练，完成渐进式多地形运动控制。

不再声称 AMP 替代全部手工 locomotion shaping。`feet_air_time` 必须作为保留的手工接触脚手架
明确披露，并通过独立消融量化其作用。

## 3. 继承边界

### 3.1 保留 MGDP 的内容

- Go2 机器人模型、PD 控制、动作空间和仿真步长；
- 深度图输入、高度图目标、对比学习和 world model；
- 感知、world model 与 PPO 策略在同一训练过程中同步更新；
- Stage 1 `mix` 地形生成器、地形类型与课程升降判据；
- 线速度和角速度跟踪、姿态与机身稳定、碰撞、关节与力矩限制、动作平滑；
- `feet_stumble`、`feet_edge` 等安全或地形任务约束；
- 终止状态 AMP 转移、策略回放、断点和评估入口。

### 3.2 采用 WMP 的内容

- 4096 个并行环境作为正式训练规模；
- 所有环境从 `terrain level = 0` 开始；
- 同时保留多种地形，但通过每个环境独立升降难度逐渐进入困难地形；
- 初始任务使用简单前进命令，先建立稳定的基础运动；
- 保留固定比例的平地/简单地形风格锚点，其余环境执行逐级多地形课程；
- 保留弱 `feet_air_time` 奖励作为实际抬脚和接触周期脚手架；
- 判别器更新次数、批量和 replay 容量均可配置，并依据日志调整，而不是写死为当前实现或 WMP 数值。

### 3.3 不直接照搬 WMP 的内容

- 不使用 A1 机器人、A1 默认关节姿态或 A1 特定 `dof_error`；
- 不恢复 MGDP 的 `motion_trot`、`motion_bound`、`motion_pace`；
- 不直接采用 WMP 的 `amp_reward_coef = 0.01`，因为两项目的任务奖励尺度不同；
- 不在缺少当前项目判别器曲线证据时直接采用每轮 20 次判别器更新；
- 不把 WMP 的地形比例替换到 MGDP，地形种类与任务主体仍以 MGDP 为准。

## 4. 奖励设计

### 4.1 必须关闭的显式步态模式奖励

Go2 AMP 任务中下列权重保持为零：

```python
motion_trot = 0.0
motion_bound = 0.0
motion_pace = 0.0
```

这些奖励直接规定腿之间的对称或配对关系，其职责由专家数据与 AMP 承担。

### 4.2 保留的最小步态脚手架

Stage 1 首个正式配置采用：

```python
feet_air_time = 0.5
```

`feet_air_time` 只在非零平移命令下、足端首次重新接触地面时奖励足端离地时间。它提供 AMP
30 维状态中缺失的足端接触信号，并抑制低伏静态支撑或滑动局部最优。它不指定 trot、pace
或 bound 的腿序，但仍属于手工 locomotion shaping，必须在报告和消融中明确说明。首版应把
单次离地时间截断到 0.75 s，并保持现有的非零平移命令 gate，避免通过超长腾空获取额外回报。

首版不新增 WMP 的 `dof_error`。如后续出现不可接受的关节姿态问题，必须通过新的独立假设和
消融决定是否增加，不能与本次重设计捆绑。

### 4.3 保留的任务、安全和动力学奖励

保留当前 MGDP Stage 1 已启用的：

- `tracking_lin_vel`、`tracking_ang_vel`；
- `lin_vel_z`、`ang_vel_xy`、`orientation`；
- `collision`、`feet_stumble`、`stand_still`；
- `torques`、`dof_acc`、`action_rate`；
- 已存在的关节、速度、力矩限制和终止逻辑；
- 后续 Stage 2 的 `feet_edge` 及地形任务奖励。

### 4.4 总奖励

默认不再使用当前按 `env_class` 区分 1.0/0.25 的地形类型 gate。首版配置对全部环境使用相同
AMP 系数：

```text
r_total = r_task_and_constraints + lambda_amp * r_amp
```

原 gate 可以保留为显式的兼容/消融选项，但不得作为新训练默认值。`terrain_level` 与
`env_class` 必须分别记录，不能再用地形类型代替难度等级。

AMP 是次级风格目标，不允许依靠增大系数补偿已经饱和的判别器。首版将
`style_reward / max(abs(task_reward), 1e-8)` 的 rollout 均值保持在 1%--10% 作为诊断带；
超出该范围时通过独立实验调整系数，不在单次 run 内自动重标定奖励。

## 5. Stage 1 训练课程

### 5.1 首个可验证阶段：基础前进与多地形难度课程

首版只解决基础前进步态和渐进多地形，不同时引入全向命令课程：

```text
num_envs = 4096
max_init_terrain_level = 0
lin_vel_x = [0.0, 0.8]
lin_vel_y = [0.0, 0.0]
heading = [0.0, 0.0]
```

环境固定分为两组：`round(num_envs * 0.15)` 个风格锚点始终使用平地；其余为课程环境，
保留 MGDP Stage 1 的 `mix` 地形类型、比例和行进距离升降级规则。所有课程环境初始位于
level 0，之后成功环境逐级上升、失败环境下降。锚点环境参与 PPO 与 AMP reward，不参与
terrain level 升级；它们不是单独的策略，也不冻结感知模块。

首版专家采样仅启用与命令分布匹配的：

- `go2_stance.txt`；
- `go2_forward.txt`；
- `go2_forward_fast.txt`；
- `go2_forward_faster.txt`。

专家采样权重为 stance 0.10、forward 0.90。后退、横向和转向轨迹不进入首版判别器训练，
避免命令与专家运动分布不匹配。

### 5.2 AMP 启动与地形解锁

为避免随机初始策略和困难地形让判别器立即完成分类，首版使用可复现的三段启动过程：

1. iteration 0--99：全部课程环境保持 level 0，策略只接收任务、约束和 `feet_air_time`；
   判别器不更新，AMP 系数为 0；
2. iteration 100--499：仍保持 level 0，开始更新判别器，并把 AMP 系数从 0 线性升到目标值；
3. iteration 500 起：目标 AMP 系数保持不变，85% 课程环境开始执行逐级升降难度。

该启动过程属于同一次从零训练，不加载或切换 checkpoint。Scaffold-only 基线采用完全相同的
地形解锁时间表，但 AMP 系数始终为 0；为保证诊断可比，判别器仍从 iteration 100 开始训练。

### 5.3 后续命令扩展

只有基础前进阶段通过第 9 节准入标准后，才能从其 checkpoint 创建新的独立实验，按以下顺序
扩展命令与专家组：

1. 加入转向命令和 turn 专家轨迹；
2. 加入横向命令和 left/right 专家轨迹；
3. 最后加入后退命令和 backward 专家轨迹。

每次只增加一种命令能力，不在同一次实验中同时改变 AMP 系数、判别器更新次数或奖励权重。
命令扩展不属于首版代码和实验准入范围；首版不得进入原计划 Stage 2。

## 6. AMP 判别器与更新节奏

### 6.1 数据流与 replay

判别器继续使用当前与下一 30 维 AMP 状态拼接的 60 维输入，保持现有数据集兼容性。首版：

- iteration 100 起所有环境按同一日程获得 AMP reward；
- 锚点和课程环境分别写入两个近期 replay pool；
- replay 容量按 rollout 数而不是固定 transition 数表达，两个 pool 各保留最近 2 个 rollout；
- 每次策略负样本先按锚点/课程 1:1 分层采样，再与等量专家样本组成判别器 batch；
- terrain level 只用于课程池的日志分层，不继续过滤课程池样本。

任一 pool 在启动期尚无足够样本时跳过该次判别器更新并记录原因，不允许静默改变采样比例。
这样可以让判别器持续看到接近平地专家分布的策略动作，同时仍学习当前复杂地形上的实际策略
分布，避免 4096 环境时固定 100000 容量退化为约一个 rollout。

### 6.2 AMP 表征边界

首版不改变 30 维状态或专家文件格式。必须增加离线/评估消融，分别屏蔽 `base_lin_vel_z` 与
`base_ang_vel_xy`，检查判别器是否主要利用复杂地形引起的机身扰动区分专家和策略。只有证据
表明存在这种捷径后，才设计包含基座坐标系足端位置、足端速度或接触相位的新版数据格式。
AMP 输入不得加入地形高度、世界位置、绝对根节点高度或 terrain class 标签。

### 6.3 可配置多次更新

新增正整数配置 `amp_updates_per_iter`。实现把当前单次更新拆为 `_update_once()`，每个 PPO
iteration 执行配置次数并汇总均值指标。checkpoint 中的 AMP iteration 记录实际判别器优化步数。

首个因果实验使用：

```text
amp_updates_per_iter = 1
amp_batch_size = 512
amp_replay_rollouts = 2
```

这是为了先检验地形初始化、命令匹配和 `feet_air_time`，避免同时改变判别器强度。后续按以下
固定判据决定是否实验 `amp_updates_per_iter = 4`：

- 若 expert/policy 输出很快分别接近 +1/-1，判别器已经过强，不增加更新次数；
- 若明显异常动作存在而 expert/policy 长期都接近 0，且判别器损失没有下降，才测试 4 次；
- 20 次只作为单独的 WMP 更新比例对照，不作为默认值。

每个更新次数使用独立 run，不跨更新次数 resume。

### 6.4 判别器健康边界

首版保留 least-squares 的 expert `+1`、policy `-1` 目标以及当前标准 AMP reward 映射，不在同一
实验中同时更换 GAN 损失或 reward 公式。`amp_gradient_penalty_coef` 从硬编码改为配置，默认仍为
10.0。判别器日志必须增加 balanced validation batch 上的 AUC、expert/policy 的 p10/p50/p90，
并分别报告锚点与课程策略样本。

若连续 50 个 iteration 同时出现 validation AUC 大于 0.98、expert p10 大于 0.9、policy p90
小于 -0.9，则标记为饱和。若饱和发生在 iteration 500 之前，停止该 run 的长训练资格，按以下
顺序排查：状态特征捷径、命令与专家不匹配、replay 分层是否生效，最后才用独立实验提高梯度
惩罚或降低判别器容量。不得通过提高 `lambda_amp` 掩盖饱和。非饱和 reward 映射只作为这些
措施仍失败后的新设计，不纳入首版。

## 7. 日志与诊断

必须先修复现有 AMP 日志语义：

- rollout 的 task/style/total reward 记录完整 rollout 均值，不再记录最后一步；
- `policy_logit_rollout` 与 `policy_logit_update` 使用不同 tag；
- expert/policy logit、判别器损失和梯度惩罚按实际多次更新取均值；
- 记录 validation AUC、expert/policy p10/p50/p90 和连续饱和计数；
- 记录 `style_reward / max(abs(task_reward), 1e-8)` 比例；
- 记录 `feet_air_time` 单项 episode reward；
- 记录 terrain level 的均值、分布和至少 `level 0`、`level 1-2`、`level >=3` 三组 policy logit；
- 分别记录锚点池与课程池的样本数、policy logit、style reward 和采样失败次数；
- 记录各 terrain class 的样本数和任务成功指标，但不得把 class 当作 level；
- TensorBoard tag 在 resume 前后保持一致。

分层 logit 只用于诊断，不改变首版奖励或 replay。

## 8. 实验矩阵

### 8.1 必需基线

在相同 Go2、4096 环境、地形、命令、种子和训练步数下比较：

1. **Original MGDP**：原 MGDP 显式 gait rewards，AMP 关闭；
2. **Scaffold-only**：`motion_trot/bound/pace = 0`，`feet_air_time = 0.5`；保留判别器训练和日志，
   但令 `lambda_amp = 0`，使策略不接收 AMP reward；
3. **AMP + scaffold**：与 Scaffold-only 相同，并启用 AMP；
4. **Pure AMP 消融**：显式 gait rewards 和 `feet_air_time` 都关闭，启用 AMP；该组是可选的研究消融，
   不阻塞首版训练架构验收。

AMP 的因果贡献由第 2、3 组比较；第 1、3 组比较人工 gait pattern 与 AMP 的任务表现；第 4 组
用于量化完全删除接触脚手架的难度。

另外执行一次不更新策略的 AMP 表征诊断：对相同 checkpoint 的专家、锚点、低难度和高难度
策略转移计算完整 30 维、屏蔽 `base_lin_vel_z`、屏蔽 `base_ang_vel_xy` 三组判别结果。该诊断
用于发现地形域捷径，不参与首轮训练超参数选择。

### 8.2 系数与训练长度

重设计改变了奖励与训练分布，因此旧实验对 0.0005 的筛选结论失效。新的第一轮只比较：

```text
lambda_amp = 0
lambda_amp = 0.0005
```

两组必须从零训练，不得加载旧策略、旧判别器或跨系数 resume。训练分级为：

- 20 iterations：启动和有限值 smoke test；
- 250 iterations：判别器与奖励尺度诊断，不用于最终性能结论；
- 1000 iterations：早期运动和地形课程筛选；
- 正式性能训练目标至少 10000 iterations，或在预先记录的连续 1000 iterations 窗口内任务、
  terrain level 和策略指标全部平台化后停止。

只有 0.0005 在新架构下表现为非零且不饱和的风格信号，才继续测试更高系数。新系数每次只新增
一档，并从零训练。

### 8.3 种子

开发诊断使用 seed 1。正式结论至少使用 seed 1、2、3，报告均值和离散程度。单 seed 结果只可
用于工程筛选。

### 8.4 固定评估协议

每个正式 checkpoint 在 0.7 m/s 前进命令下分别评估平地、坡面、上/下楼梯、离散障碍和
gap/pit。任务指标包括现有行进距离判据定义的成功率、线速度跟踪误差和跌倒率。风格指标至少
包括接触期足端滑移速度、动作 jerk、左右步时变异系数，以及平地上相对专家关节周期的距离。
复杂地形不使用平地专家距离作为准入标准，因为必要的抬脚和机身响应本就会偏离平地示范。

视频使用相同命令、相同地形种子、相同相机和相同长度生成，比较时隐藏实验名称。自然性结论
必须同时引用客观风格指标和盲化视频比较，不能只使用判别器分数或单个最佳视频。

## 9. Stage 1 准入标准

基础前进阶段必须同时满足：

- 固定 0.7 m/s 前进评估中形成连续、可观察的周期性步态，而非低伏静态支撑或 reset 位移；
- AMP + scaffold 相对 Scaffold-only 显示可复现的步态差异，且任务表现没有不可接受下降；
- 判别器不是长期 expert 约 +1、policy 约 -1 的完全饱和状态；
- style reward 不长期坍缩为相对 task reward 可忽略的零信号；
- terrain level 能从 0 稳定提升，楼梯、坡面等成功率随训练改善；
- 感知/world-model 指标保持有限并持续学习；
- 固定 checkpoint 视频与 TensorBoard 曲线结论一致；
- 代码测试、短 rollout 和 checkpoint 往返通过。

“任务表现没有不可接受下降”定义为：在至少 3 个相同种子上，AMP + scaffold 相对
Scaffold-only 的总体和每个关键地形组成功率下降均不超过 5 个百分点，速度跟踪得分的相对
下降不超过 5%。任何一个关键地形组越界即不满足非干扰主张，不允许用其他地形的平均提升
抵消。自然性成功还要求预先声明的风格指标中至少两项改善，且盲化视频比较多数选择 AMP 组。

在满足以上条件前：

- 不进入原计划 Stage 2；
- 不开展全向命令扩展；
- 不引入固定风格锚点；
- 不宣称 AMP 已成功替代显式 gait-pattern rewards。

## 10. 代码修改范围

首版实施计划应限制在：

- `legged_gym/legged_gym/envs/go2_amp/config.py`：奖励、level 0、简单命令、专家组和新 AMP 配置；
- Go2 AMP 环境/地形分配代码：锚点与课程环境的稳定分组、课程冻结和 level 解锁；
- `legged_gym/rl/MGDP/amp/session.py`：可配置多次更新、明确指标和必要的分层统计接口；
- AMP replay 代码：按 rollout 计量的锚点/课程双池与 1:1 分层采样；
- `legged_gym/rl/MGDP/runners/policy_runner.py`：完整 rollout 聚合、level/class 分层日志和明确 tag；
- 现有 AMP/config 测试及必要的新单元测试；
- README 中新的训练、恢复、实验隔离与结果解释说明。

不得修改 MGDP 原任务的默认行为。Go2 AMP 新实验统一使用
`go2_amp_stage1_task_priority` experiment 名称，避免与现有输出混合。
现有未跟踪的录制脚本和测试不属于本重设计，除非后续实施计划明确纳入。

## 11. 兼容性与恢复规则

- 新架构正式训练必须从零开始；
- 旧 checkpoint 只允许用于可视化和历史对比，不作为新实验 resume 起点；
- `lambda_amp`、启动/解锁日程、锚点比例、replay 分层、`amp_updates_per_iter`、奖励边界或专家组
  不同的实验不得相互 resume；
- checkpoint 必须保存判别器、优化器、归一化器、两个 replay pool 的内容/游标、策略 iteration、
  实际 AMP update 计数和连续饱和计数；
- 配置不兼容时应尽早报错，不允许静默加载部分 AMP 状态；
- policy-only warm start 保留为明确命令选项，但不用于本轮正式基线。

## 12. 测试与验证

CPU 单元测试至少覆盖：

- Go2 AMP Stage 1 的 `max_init_terrain_level == 0`；
- 显式 gait-pattern rewards 为零，`feet_air_time == 0.5`，单次离地时间上限为 0.75 s；
- 初始命令和专家文件仅包含 stance/forward；
- 默认 AMP gate 对所有 terrain class 相同；
- 锚点数量按 15% 稳定取整、始终为平地且不参与 terrain level 升级；
- iteration 0--99、100--499、500 起三段日程的 AMP 系数、判别器更新和课程解锁边界；
- 双 replay pool 各保留两个 rollout，并按锚点/课程 1:1 采样；
- pool 样本不足时跳过更新并显式记录，不静默回退为非分层采样；
- `amp_updates_per_iter` 的参数校验、优化步数和均值指标；
- `amp_gradient_penalty_coef` 的参数校验与传递；
- validation AUC、分位数和连续 50 iteration 饱和判据；
- 30 维特征诊断 mask 不改变原始数据文件且只影响评估副本；
- rollout reward/logit 聚合不是最后一步值；
- terrain level 与 terrain class 分层不混用；
- checkpoint 的双 replay、日程 iteration、多次更新计数和饱和计数往返；
- 原 MGDP task config 不发生变化。

GPU/Isaac Gym 验证顺序为：配置启动、20-iteration smoke、4096×250 配对诊断、4096×1000
早期筛选。只有用户提供的云端长训练结果可用于多地形性能判断。

## 13. 明确不在首版范围内

- 新增或合成 hop、楼梯、gap 专家动作；
- terrain-conditioned discriminator 或把 terrain class 输入判别器；
- 按具体 terrain level/class 进一步过滤课程 replay；
- 自动 terrain-specific AMP gate、在线奖励重标定或梯度投影/约束优化器；
- 自动恢复 `dof_error` 或其他新增姿态 shaping；
- 全向命令课程和原计划 Stage 2；
- 直接复用旧实验确认的 AMP 系数。

这些内容只有在首版分层日志给出明确失败证据后，才进入新的设计与消融。
