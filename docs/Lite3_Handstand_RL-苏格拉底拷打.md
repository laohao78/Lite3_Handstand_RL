# Lite3_Handstand_RL 苏格拉底拷打

> 基于 [laohao78/Lite3_Handstand_RL](https://github.com/laohao78/Lite3_Handstand_RL) 项目的深度理解测试。
> 不要直接看答案，先在脑子里回答，答不出来再去看源码。

---

## 第一层：整体架构（热身）

**Q1.1** 这个项目要解决什么问题？它的完整 pipeline 包含哪几个环节？

**Q1.2** 项目依赖了哪些核心开源框架？分别扮演什么角色？
- IsaacGym 是干什么的？
- rsl_rl 是干什么的？
- legged_gym 是干什么的？
- Lite3_rl_deploy 是干什么的？

**Q1.3** 这个项目的代码是从哪个仓库 fork/修改来的？原作者做了哪些关键修改？

<details>
<summary>点击查看答案</summary>

**A1.1** 用强化学习训练四足机器人（云深处绝影 Lite3）完成"手倒立"动作，并实现 sim2real 部署。
Pipeline：Train (IsaacGym) → Play (仿真回放+导出ONNX) → Sim2Sim (MuJoCo测试) → Sim2Real (实机部署)

**A1.2**
- **IsaacGym**：NVIDIA 的高性能物理仿真器，提供 GPU 并行仿真，同时跑 4096 个环境
- **rsl_rl**：PPO 算法的实现库，提供 Actor-Critic 网络 + PPO 训练循环
- **legged_gym**：在 IsaacGym 上搭建的四足机器人 RL 训练框架，提供了环境模板、奖励函数、训练脚本等
- **Lite3_rl_deploy**：云深处官方的部署工程，负责加载 ONNX 模型在 MuJoCo 仿真/实机上运行

**A1.3** 从 [cmjang/legged_gym_handstand](https://github.com/cmjang/legged_gym_handstand) 修改而来，关键修改：
1. 引入了 Lite3 的 URDF 模型和 mesh 文件
2. 将输入从 235 维（45 + 170地形 + 3线速度 + ...) 精简到 45 维（删除了地形高度采样 170维 + base线速度 3维）
3. 修改 play.py 支持导出 ONNX 模型
</details>

---

## 第二层：观测空间与动作空间

**Q2.1** 观测空间是 45 维。请**精确**说出这 45 维分别是什么，能画出来吗？

提示：打开 `compute_observations()` 看看 `torch.cat` 了哪些东西。

**Q2.2** 为什么删掉了 `base_lin_vel`（3维）？删掉它有什么好处和风险？

**Q2.3** 为什么要删掉地形高度测量（170维）？

**Q2.4** 动作空间是 12 维。Lite3 有 12 个关节，每个关节对应什么？动作是怎么变成关节力矩的？

<details>
<summary>点击查看答案</summary>

**A2.1** 45维 = 3 + 3 + 3 + 12 + 12 + 12：
- `base_ang_vel * obs_scales.ang_vel` → 3维（机体角速度）
- `projected_gravity` → 3维（重力在机体坐标系下的投影）
- `commands[:, :3] * commands_scale` → 3维（线速度x, 线速度y, 偏航角速度指令）
- `(dof_pos - default_dof_pos) * obs_scales.dof_pos` → 12维（关节位置偏差）
- `dof_vel * obs_scales.dof_vel` → 12维（关节速度）
- `actions`（上一帧动作）→ 12维

**A2.2** 
- 好处：减少观测维度，使模型更容易泛化，不依赖绝对速度估计（sim2real 时速度估计不准），也让模型更关注姿态信息
- 风险：失去了速度反馈，策略可能无法感知自己是否在漂移

**A2.3** 地形高度采样点是一个 17×10 的网格，170 维，用于非平坦地形行走。手倒立任务用平面地形（`mesh_type = 'plane'`），不需要地形信息，加上去反而增加冗余维度、让训练更难。

**A2.4** Lite3 每条腿 3 个关节（HipX, HipY, Knee），4条腿共 12 个。动作用的是 **P 控制（位置控制）**：
```
torques = p_gains * (action_scale * actions + default_dof_pos - dof_pos) - d_gains * dof_vel
```
即：动作 = 目标关节位置偏移量，乘以 scale 后加上默认角度作为 PD 控制器的目标位置。
</details>

---

## 第三层：奖励函数设计（核心）

**Q3.1** 项目中有哪几个和"手倒立"直接相关的奖励函数？各自的设计思路是什么？

**Q3.2** `_reward_handstand_feet_height_exp` 是整个项目最复杂的奖励函数（约180行代码）。请回答：
- 它奖励的是**哪几条腿**抬高？为什么要区分前腿和后腿？
- `LIFT_THRESHOLD = 0.025` 是什么意思？为什么需要这个阈值？
- 它设计了哪几个子奖励项（base_lift_reward, single_leg_reward, both_legs_reward, alternation_reward, target_reward）？各自的目的是什么？
- `backward_penalty` 是用来干什么的？
- 膝盖触地时会发生什么？

**Q3.3** `_reward_handstand_feet_on_air` 和 `_reward_handstand_feet_air_time` 有什么区别？为什么需要两个不同的"脚离地"奖励？

**Q3.4** `_reward_handstand_orientation_l2` 中 `target_gravity = [1, 0.0, 0.0]`，为什么不直接设为 `[0, 0, 1]` 或 `[0, 0, -1]`？这个 target_gravity 的含义是什么？

**Q3.5** `_reward_front_feet_contact` 的权重是 `-40.0`，比其他惩罚项大得多，为什么？

<details>
<summary>点击查看答案</summary>

**A3.1** 手倒立相关奖励函数（共5个）：
- `handstand_feet_height_exp`：奖励前腿脚部达到目标高度（0.6m）
- `handstand_feet_on_air`：奖励所有脚+膝盖都离地
- `handstand_feet_air_time`：奖励脚部持续悬空的时间
- `handstand_orientation_l2`：奖励躯干姿态接近目标（倒立方向）
- `front_feet_contact`：惩罚前脚接触地面

**A3.2**
- **区分前腿后腿**：倒立时前腿（FL, FR）是"朝上"的两条腿，是"手"的角色，需要高高抬起；后腿（HL, HR）是"支撑腿"，需要稳定在地面附近。代码里前腿用的 foot_indices [4, 8]（FL_FOOT, FR_FOOT），后腿用 [12, 16]（HL_FOOT, HR_FOOT）
- **LIFT_THRESHOLD**：高度低于 0.025m 认为腿"还在地上"，不算抬腿。避免因微小离地就给大量奖励，防止 reward hacking
- **子奖励**：
  - `base_lift_reward (×0.3)`：基础抬腿奖励，任意腿离地就给，鼓励"至少动一动"
  - `single_leg_reward (×0.4)`：抬得越高越好
  - `both_legs_reward (×0.5)`：双前腿都离地才给，鼓励双腿协调
  - `min_lift_reward (×0.3)`：两条腿中较低的那条也要抬高，防止一条腿拖地
  - `alternation_reward (×0.4)`：交替模式奖励，一条抬一条放，对行走式倒立有用
  - `target_reward (×0.6)`：抬离的腿高度接近目标 0.6m
  - `knee_safety_reward (×0.2)`：膝盖不能太低
- **backward_penalty**：惩罚前腿脚部 x 坐标 < 0（即腿向前伸太远，这在倒立姿态下不合理，腿应该向上伸）
- **膝盖触地**：`combined_reward[severe_knee_contact] = 0.0`——直接清零所有奖励

**A3.3**
- `feet_on_air`：**瞬时状态**奖励，检查当前时刻脚和膝盖是否接触地面，不接触就给 reward=1
- `feet_air_time`：**持续时长**奖励，计算脚从离地到再次着地之间的空中时间，超过 threshold（5s）才给奖励
- 两者互补：一个鼓励"现在离地"，一个鼓励"一直保持离地"

**A3.4** `target_gravity = [1, 0, 0]` 表示在机器人**机体坐标系**下，重力方向指向 +x 轴。因为在倒立姿态下：
- 正常站立：重力在机体坐标系下是 [0, 0, -1]（向下 = z 轴负方向）
- 完全倒立：机器人的 x 轴（前后方向）指向天空，重力在机体坐标系下变成 [1, 0, 0]（沿机器人前方向下）
- 所以 `projected_gravity` 接近 `[1, 0, 0]` 意味着机器人躯干的前后轴对着天空，即倒立成功

**A3.5** 前脚触碰地面意味着"倒立失败"——倒立的核心就是前腿离地。权重 -40 远大于其他项（如 torques=-1e-5），因为前脚落地是严重的负样本，需要给极强的负反馈来阻止策略退化到"四脚站立"。
</details>

---

## 第四层：渐进式课程学习

**Q4.1** `transition_progress`（过渡进度）是什么？它从多少变到多少？经历多长时间？

**Q4.2** `_update_progressive_targets()` 定义了一个三阶段的姿态序列，是哪三个阶段？`target_gravity_vec` 分别是什么？

**Q4.3** 过渡用了 S 曲线（smooth curve）`3*progress^2 - 2*progress^3`，为什么不直接用线性过渡？

**Q4.4** `_reward_progressive_orientation` 的 tolerance 从 30° 逐渐减小到 10°，这个设计背后的 intuition 是什么？

<details>
<summary>点击查看答案</summary>

**A4.1** `transition_progress` 是从 0（站立）到 1（完全倒立）的标量，每个环境随机分配 3~5 秒的过渡时间，过渡速度也随机（0.5~1.5倍速）。每个 episode reset 时重新随机。

**A4.2** 三阶段：
- Stage 0 (progress=0)：`target_gravity = [0, 0, 1]`，正常站立（重力向下）
- Stage 1 (0 < progress ≤ 0.5)：`[0,0,1]` → `[0.7, 0, 0.7]`，45°倾斜
- Stage 2 (0.5 < progress ≤ 1.0)：`[0.7, 0, 0.7]` → `[1, 0, 0]`，完全倒立

**A4.3** S 曲线在两端变化慢、中间变化快。这样：
- 初期给策略更多时间适应站立，不会突然要求倾斜
- 末期精度要求高时变化也放缓，策略有时间微调
- 线性过渡在两端会太突兀

**A4.4** Intuition：婴儿学步原则
- 初期（progress≈0）：策略刚开始学，容许 30° 偏差，避免过于严苛导致学不到东西
- 后期（progress≈1）：策略已经基本会倒立了，收紧到 10°，要求精确控制姿态
- 如果一开始就要求 10°，奖励信号太稀疏，策略根本学不会
</details>

---

## 第五层：训练配置与PPO

**Q5.1** 训练用了多少个并行环境？每轮迭代收集多少步经验？mini-batch 怎么划分？

**Q5.2** PPO 的 Actor 和 Critic 网络结构是什么样的？为什么用 `elu` 激活函数？

**Q5.3** `control.decimation = 4` 是什么意思？`sim.dt = 0.005`，那么策略的控制频率是多少？

**Q5.4** 训练时使用了哪些 domain randomization？分别有什么作用？

<details>
<summary>点击查看答案</summary>

**A5.1**
- 4096 个并行环境
- 每环境 24 步（`num_steps_per_env = 24`），共 4096×24 = 98,304 条 transitions
- 4 个 mini-batch（`num_mini_batches = 4`），每个 mini-batch 约 24,576 条

**A5.2**
- Actor: [512, 256, 128] → 12维输出（动作均值）
- Critic: [512, 256, 128] → 1维输出（状态价值）
- ELU 在负值区域比 ReLU 更平滑，连续可导，对 RL 训练的梯度稳定性更友好。在 legged_gym 中 ELU 是默认选择

**A5.3**
- 物理仿真频率：1/0.005 = 200 Hz
- 策略控制频率：200 / 4 = 50 Hz
- 即每 4 个物理步长（共 20ms），策略才输出一次动作，这期间用相同的 PD 目标

**A5.4**
- **摩擦系数随机化** (0.5~1.25)：让策略适应不同地面摩擦力，便于 sim2real
- **随机推搡** (`push_robots`，每15秒)：施加随机基座速度扰动，训练抗干扰能力
- **观测噪声**（关节位置、角速度、重力等）：模拟真实传感器噪声，防止过拟合仿真环境
- **动作噪声**（初始 init_noise_std=1.0，探索噪声）：PPO 的标准探索机制
</details>

---

## 第六层：终止条件与环境重置

**Q6.1** 什么情况会触发 `termination`（训练重置）？

**Q6.2** `terminate_after_contacts_on = ["TORSO"]` 和 `penalize_contacts_on = ["SHANK", "THIGH", "Knee"]` 有什么区别？为什么 TORSO 接触直接终止，而腿接触只是惩罚？

**Q6.3** `_reset_dofs` 时关节位置随机在 default 的 0.5~1.5 倍，为什么要这样随机？为什么不直接重置到 default 角度？

<details>
<summary>点击查看答案</summary>

**A6.1** 两个条件：
1. TORSO（躯干）接触地面：`torch.norm(contact_forces[termination_indices]) > 1.0`
2. 超时：`episode_length > max_episode_length`（20秒 / dt）

**A6.2**
- **TORSO 接触 = 终止**：躯干是机器人本体，碰地意味着机器人倒地/摔了。这属于不可恢复的失败状态，继续仿真没有意义，必须重置
- **SHANK/THIGH/Knee 接触 = 惩罚**：腿碰到地面是"不完美"但不是"失败"，给负奖励即可，策略可以从中学习如何抬高腿

**A6.3** 随机初始角度有以下目的：
- **防止过拟合**：如果总是从 default 角度开始，策略只会处理这一个初始状态
- **鲁棒性**：确保策略能应对各种关节配置，包括摔倒后重新站起的情况
- **探索**：增加状态分布覆盖，帮助策略学到更广泛的关节空间
</details>

---

## 第七层：Sim2Sim 与 Sim2Real

**Q7.1** 为什么不直接在 IsaacGym 里测试完就上实机，还要走 MuJoCo 做 sim2sim？

**Q7.2** ONNX 导出发生在什么时候？导出的模型文件有什么要求（输入输出的维度格式）？

**Q7.3** Lite3_rl_deploy 中提到的"危险动作导致切入 Joint Damping"是什么？为什么要在仿真测试时临时屏蔽它？

**Q7.4** `play.py` 中有一行设置 `env.commands[:, :3] = 0.0`，这行是干什么的？为什么不直接用训练时的随机 commands？

<details>
<summary>点击查看答案</summary>

**A7.1**
- IsaacGym 使用的是简化物理模型（PhysX），和真实物理有差距
- MuJoCo 的接触力学建模更精确，更接近真实世界
- Sim2sim 是一个低成本验证环节：如果模型在 MuJoCo 中也能倒立，说明策略没有过拟合 IsaacGym 的物理特性，sim2real 成功率更高
- 如果在 MuJoCo 中表现不好，可以提前发现问题再进行 sim2real 参数调节（如 PD 增益、质量分布等）

**A7.2**
- 导出时机：`play.py` 运行后会调用 `export_policy_as_onnx()` 导出模型
- 输入：45 维观测向量
- 输出：12 维动作（均值），ONNX 文件保存到 `logs/onnx/legged.onnx`
- 部署时被复制到 `Lite3_rl_deploy/policy/ppo/policy.onnx`

**A7.3**
- "Joint Damping"是实机上的安全保护机制：当状态机检测到动作可能会导致关节损坏（如关节角度过大、力矩突变），会切断 RL 控制，切换到阻尼模式让关节自然减速
- 倒立动作本身就是"异常姿态"，容易触发保护
- 仿真测试时屏蔽它是因为：MuJoCo 模型和实机有差异，倒立动作在仿真中常被误判为"危险"，导致 RL 控制被反复切断，无法完成测试
- **但上实机时必须恢复保护！**否则可能损坏真实机器人

**A7.4**
- 训练时 commands 是随机的（线速度 x/y 在 [-0.7, 0.7]），是为了训练一个能跟踪各种速度指令的策略
- 回放时设为零，因为手倒立任务不需要移动，只需要原地倒立，零指令让策略只关注姿态控制
</details>

---

## 第八层：灵魂拷问（设计哲学）

**Q8.1** 为什么这个项目不用 DDPG、SAC 等 off-policy 算法，而选择 PPO？

**Q8.2** 项目设置了 `only_positive_rewards = False`（允许负奖励）。如果改为 `True` 会有什么影响？

**Q8.3** 观测空间中**没有**基座的绝对位置 (x, y, z) 和绝对朝向（偏航角），这是有意为之吗？为什么？

**Q8.4** `_reward_progressive_orientation` 和 `_reward_smooth_transition` 的权重被设为了 0（注释说"投影重力有问题，这两项不加也没事"）。你猜"投影重力有问题"指的是什么？为什么作者放弃这两个奖励？

**Q8.5** 如果你来改进这个项目，你觉得最值得做的 3 个改进是什么？给出具体理由。

<details>
<summary>点击查看答案</summary>

**A8.1**
- PPO 是 on-policy 算法，在 legged_gym + IsaacGym 生态中已经是标准方案，稳定可靠
- 4096 个并行环境天然适合 on-policy 的数据收集效率
- PPO 的超参数（clip, KL penalty）使得训练过程对学习率不那么敏感
- Off-policy 算法在 4096 环境的大批量下样本效率优势不明显，反而调参更难
- rsl_rl 提供了开箱即用的 PPO 实现

**A8.2**
- `only_positive_rewards = True` 会把负的总奖励 clip 到 0
- 负面影响：如果某一步 reward 本来就是负的（如惩罚项很多），clip 后会变成 0，策略无法区分"稍微差"和"非常差"，梯度信息丢失
- 好处：防止早期训练时负奖励过大导致策略崩溃
- 这个项目用 `False` 是正确的，因为有很多惩罚项（front_feet_contact=-40），需要精确的负反馈

**A8.3** 是的，有意为之。原因：
- **sim2real 泛化**：绝对坐标在实机部署时无法获取（没有 GPS/动作捕捉），策略不应依赖绝对位置
- **平移/旋转不变性**：倒立这个技能是"相对于自身姿态"的，不依赖在房间的哪个位置做、面朝哪个方向做
- **观测精简**：少几个无关维度让训练更高效

**A8.4** "投影重力有问题"的最可能原因：
- `target_gravity_vec` 的初始化和更新时序和 `projected_gravity` 不对齐
- 或者在渐进过渡中，`projected_gravity` 的定义（机体坐标系下的重力投影）和 `target_gravity_vec` 的物理含义存在歧义
- `_reward_progressive_orientation` 需要计算 `acos(cos_similarity)`，当两个向量接近但夹角很大时可能出现数值问题
- `_reward_smooth_transition` 依赖 `_reward_progressive_orientation` 的中间变量，所以也受影响
- 作者发现 `handstand_orientation_l2`（简单的 L2 距离）已经足够表达姿态目标，不需要额外的渐进奖励

**A8.5** 开放题，合理即可。参考方向：
1. **添加 teacher-student 蒸馏**：先用完整观测（含地形、速度）训练 teacher，再蒸馏到 45 维 student，可能提升性能
2. **参考轨迹 + RL 混合**：先用轨迹优化生成一条倒立参考轨迹，RL 只做跟踪，降低探索难度
3. **多阶段课程**：先训练站立→再训练倾斜→最后倒立，每个阶段固定再继续，避免一次性从头学倒立
4. **更好的域随机化**：增加关节摩擦、质心偏移、负载变化等随机化，提高 sim2real 成功率
5. **RMA（Rapid Motor Adaptation）**：用 teacher-student 蒸馏环境参数到观测的隐变量，实机自适应
</details>

---

## 终极挑战：如果你能流畅回答下面这个问题，你是真的懂了

**Q9** 在 `compute_observations()` 中，注释掉的那行 `target_gravity_obs = self.target_gravity_vec * self.obs_scales.lin_vel` 被取消了（没放入 obs_buf），但 `_update_progressive_targets()` 仍然在 `compute_reward()` 中被调用。请问：

1. `target_gravity_vec` 现在通过什么途径影响了策略的学习？
2. 如果把它放回 `obs_buf`（作为观测的一部分），会有好处还是坏处？为什么？

<details>
<summary>点击查看答案</summary>

**A9**
1. `target_gravity_vec` 当前**只通过奖励函数**影响策略：
   - `_reward_progressive_orientation`（权重=0，已废弃）
   - `_update_progressive_targets()` 在 `compute_reward()` 前被调用，更新 `self.target_gravity_vec`
   - 即使 `_reward_progressive_orientation` 权重为 0，`target_gravity_vec` 的更新逻辑仍然在跑（浪费计算），但实际对策略无影响
   - 真正起作用的是 `_reward_handstand_orientation_l2`，它直接用固定 target `[1,0,0]` 而不是渐进目标

2. 如果把 `target_gravity_vec` 放入观测：
   - **好处**：策略知道当前处于渐进课程的哪个阶段（"现在要求斜着还是直立？"），可以更精准地调整行为。类似于给策略一个"进度条"，让它知道目标在哪
   - **坏处**：
     - 增加了 3 维输入（总 48 维），虽然不多
     - `target_gravity_vec` 是按 episode 动态变化的，策略需要学习解耦"观测中的目标变化"和"本体状态变化"，增加学习难度
     - 如果策略过度依赖这个信号，当实际部署时（没有渐进课程，直接要求倒立），策略可能表现不佳
   - **结论**：对于最终目标是倒立的场景，不加更好，因为部署时没有渐进课程。但如果训练困难（学不会倒立），加入可以提高训练成功率
</details>

---

> 答出来几题？诚实记录：____ / 30+
