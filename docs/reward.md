# lite3 当前生效的奖励说明

这份文档对应当前 lite3 任务的实际配置。只有 reward scale 不是 0 的项才会进入训练和评估，因此下面列出的项目才是真正起作用的奖励或惩罚。

## 1. 基础运动相关奖励

1. tracking_lin_vel，权重 10.0
	作用：奖励机器人跟随期望的平面线速度。命令速度越接近 base 的实际速度，奖励越高。

2. tracking_ang_vel，权重 5.0
	作用：奖励机器人跟随期望的偏航角速度。用于控制原地转向或旋转。

3. ang_vel_xy，权重 -0.3
	作用：惩罚机身绕 x、y 轴的角速度。它越晃，惩罚越大。

4. torques，权重 -0.00001
	作用：惩罚总关节力矩过大，鼓励控制更省力。

5. dof_acc，权重 -2.5e-7
	作用：惩罚关节加速度变化，抑制过猛的关节抖动。

6. collision，权重 -1.0
	作用：惩罚指定刚体部位发生接触。当前通过 SHANK、THIGH、HIP 这些刚体名匹配。

7. action_rate，权重 -0.03
	作用：惩罚相邻时刻动作变化太大，鼓励动作平滑。

8. stand_still，权重 -0.8
	作用：当命令接近静止时，惩罚关节偏离默认姿态，帮助站稳。

9. dof_pos_limits，权重 -10.0
	作用：惩罚关节位置接近或超过 URDF 限位，防止关节打到极限。

## 2. 手倒立相关奖励

1. handstand_feet_height_exp，权重 17.5
	作用：核心的抬腿高度奖励。会根据前脚、后脚、膝盖高度和离地状态组合成一个综合奖励，鼓励前腿抬起并逐步接近目标高度。

2. handstand_feet_on_air，权重 1.5
	作用：奖励脚部离地，同时惩罚腿部其他部位接触地面。目标是让机器人尽量保持足部悬空。

3. handstand_feet_air_time，权重 1.5
	作用：奖励脚部连续悬空的时间，鼓励动作维持得更久，而不是刚抬起来又掉下去。

4. handstand_orientation_l2，权重 0.8
	作用：用 projected_gravity 和目标重力方向做 L2 距离，奖励机器人朝向更接近倒立姿态。

5. front_feet_contact，权重 -40.0
	作用：前脚一旦接触地面就强烈惩罚，防止训练中前腿反复落地。

6. joint_smoothness，权重 2.5e-9
	作用：奖励关节运动更平滑，综合约束动作变化率、加速度和 jerk。

7. torque_smoothness，权重 0.06
	作用：奖励扭矩变化更平滑，减少控制输出的抖动。

## 3. 当前没有起作用的项

下面这些项当前 scale 是 0，所以不会进入 reward 计算：

1. termination
2. lin_vel_z
3. base_height
4. feet_air_time
5. feet_stumble
6. progressive_orientation
7. smooth_transition

另外，虽然代码里有一些历史版本的 reward 函数被保留或注释掉了，但只要对应 scale 为 0，它们就不会参与当前训练。

## 4. 计算方式

reward 的计算流程是：

1. 先读取 reward scales。
2. 过滤掉 scale 为 0 的项。
3. 对每个非 0 项调用对应的 _reward_xxx 函数。
4. 将函数输出乘以对应权重后累加到总 reward。

所以“起作用”的标准不是代码里有没有这个函数，而是配置里对应的 scale 是否为非 0。

## 5. 当前配置的直观理解

这套配置的目标很明确：

1. 用 tracking 系列 reward 保留基础的姿态和速度稳定性。
2. 用 handstand 系列 reward 把优化重心推向抬腿、悬空、倒立朝向。
3. 用 collision、front_feet_contact、stand_still 这些惩罚项约束错误接触和乱动。
4. 用 smoothness 系列项减少动作抖动，让策略更稳。

如果你后面要调参，最值得优先观察的是 handstand_feet_height_exp、front_feet_contact、handstand_orientation_l2、stand_still 这四组，因为它们最直接决定“能不能倒起来、倒起来后稳不稳”。
