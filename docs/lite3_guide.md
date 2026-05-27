# Lite3 任务入门指南

这份文档的目标很简单：帮助你快速弄懂 lite3 这个任务到底在训练什么、代码从哪里开始看、哪些地方最关键。

## 1. 先看整体结构

lite3 任务不是只看一个配置文件就够了，它由四层东西拼起来：

1. 任务配置，决定环境长什么样、奖励怎么配、观测和控制怎么定义。
2. 机器人模型，决定 URDF 里有哪些 link 和 joint，名字怎么匹配。
3. 环境逻辑，决定每一步怎么出观测、怎么算 reward、什么时候 reset。
4. 测试脚本，决定 play 或 train 时到底用哪些覆盖参数。

如果你只看 reward，容易漏掉命令、控制器、碰撞体名这些真正决定任务行为的部分。

## 2. 第一个要看的文件

建议按这个顺序读：

1. legged_gym/envs/lite3/lite3_config.py
2. legged_gym/envs/base/legged_robot_config.py
3. legged_gym/envs/base/legged_robot.py
4. legged_gym/scripts/play.py
5. resources/robots/lite3/urdf/lite3.urdf

这个顺序的好处是：先看“配了什么”，再看“怎么执行”，最后看“模型里到底有哪些部件”。

## 3. 配置文件到底看什么

在 lite3_config.py 里，你最应该先确认这几项：

1. 初始姿态。机器人一开始是怎么摆放的。
2. 控制器。是 P 控制还是别的控制，action scale 多大。
3. 资产文件。对应哪个 URDF。
4. foot_name。脚部怎么被识别。
5. penalize_contacts_on。哪些刚体碰地要惩罚。
6. rewards.scales。哪些奖励真的会生效。

这里最重要的一点是：

奖励函数有没有写出来不重要，真正决定它生不生效的是 scale 有没有非 0。

## 4. 基础配置文件的作用

在 legged_robot_config.py 里，主要看四类信息：

1. 环境数量和观测维度。
2. 地形设置。
3. 命令范围。
4. 奖励和噪声的默认值。

这个文件像“总模板”，lite3_config.py 只是对它做覆盖。

你要特别注意这几个地方：

1. num_observations。它决定策略输入维度。
2. mesh_type。它决定你是在平地还是地形上训练。
3. commands。它决定机器人学什么指令。
4. rewards.scales。它决定哪些行为被鼓励，哪些被惩罚。

## 5. 环境逻辑最核心的地方

legged_robot.py 才是任务真正“跑起来”的地方。

你至少要搞明白这几个函数：

1. _init_buffers
   这里初始化所有状态张量，比如 root state、dof state、contact forces、观测缓存。

2. step
   每次策略输出 action 后，先算力矩，再模拟，再进入 post_physics_step。

3. post_physics_step
   这里会刷新状态、检查终止、算奖励、算观测、reset。

4. compute_observations
   这里决定策略到底看到了什么。

5. compute_reward
   这里把所有非 0 的 reward term 加起来。

6. reset_idx
   这里决定环境重置时机器人怎么摆、命令怎么刷新、课程学习怎么推进。

7. _post_physics_step_callback
   这里通常处理命令重采样、push robot、测地形高度等辅助逻辑。

## 6. 奖励要怎么理解

你可以把 reward 分成三类：

1. 保稳定的。
   比如 tracking、orientation、stand_still、smoothness。

2. 保动作质量的。
   比如 torques、dof_acc、action_rate、torque_smoothness、joint_smoothness。

3. 推任务目标的。
   比如 handstand_feet_height_exp、handstand_feet_on_air、handstand_feet_air_time、handstand_orientation_l2、front_feet_contact。

如果你想训练倒立，第三类才是核心；前两类更多是在防止策略乱跑。

## 7. 命令系统要特别注意

commands 决定机器人在训练时被要求做什么。

你要重点看：

1. num_commands。命令有几维。
2. resampling_time。多久换一次指令。
3. heading_command。是用朝向误差还是角速度控制。
4. ranges。每一维命令能随机到什么范围。

如果你在 play 里把命令设成 0，只是起始状态静止，不代表环境以后不会重新采样。

## 8. URDF 里的名字为什么那么重要

碰撞惩罚和脚部索引都依赖名字匹配。

你要确认：

1. 哪些是 link 名字。
2. 哪些是 joint 名字。
3. 哪些名字对应 FOOT、SHANK、THIGH、HIP、TORSO。

一个常见错误是把 joint 名当成 link 名来写，这样 contact 惩罚会完全不生效。

## 9. 学习 lite3 的实用方法

最有效的方法不是先背所有参数，而是先回答这几个问题：

1. 它的观测输入到底是什么。
2. 它的动作输出到底怎么变成力矩。
3. 它的奖励到底在鼓励什么行为。
4. 它什么时候 reset，为什么 reset。
5. 它的命令到底是不是一直静止。

如果这 5 个问题都能说清楚，你就已经真正理解这个任务了。

## 10. 建议你实际怎么读

可以按下面这个顺序做笔记：

1. 先抄出 lite3_config.py 里所有非 0 的奖励。
2. 再去 legged_robot.py 里找这些奖励对应的函数。
3. 再看 compute_observations 里喂给策略的输入。
4. 再看 _compute_torques 看动作怎么转成力矩。
5. 最后看 reset 和 command 逻辑，确认训练分布是什么。

## 11. 你现在最值得盯住的点

如果你现在目标是把 lite3 训练成手倒立，我建议优先关注这四个东西：

1. handstand_feet_height_exp
2. front_feet_contact
3. handstand_orientation_l2
4. stand_still

这四项基本决定了机器人能不能抬腿、抬起来后会不会落地、姿态是不是朝着倒立方向走。

## 12. 一句话总结

理解 lite3 的关键，不是看它“写了多少奖励”，而是弄清楚：

1. 观测给了什么。
2. 动作怎么变成控制。
3. 奖励在推什么方向。
4. reset 和命令在改变什么训练分布。
5. URDF 名字和碰撞匹配有没有对上。

把这五件事弄清楚，lite3 基本就不再是黑盒了。

## 13. 常见问题答疑

### Q1. 它的观测输入到底是什么？

当前是 45 维观测，核心由 base_ang_vel、projected_gravity、commands 前 3 维、dof_pos 偏差、dof_vel 和 actions 组成。地形高度测量没有加入，所以它不是 232 维。

### Q2. 动作到底怎么变成力矩？

当前是 P 控制。策略输出的动作先乘 action_scale，再和默认关节角叠加成目标位姿，最后通过 PD 公式转成 torque。

### Q3. 奖励到底在鼓励什么？

它同时在做三件事：保持基础稳定、抑制乱动和错误接触、推动手倒立目标。真正主导倒立行为的是 handstand_feet_height_exp、front_feet_contact、handstand_feet_on_air、handstand_feet_air_time 和 handstand_orientation_l2。

### Q4. 它什么时候 reset，为什么 reset？

当终止条件触发或者 episode 超时后会 reset。reset 会重置关节、根节点状态、命令和一些缓存，并把环境放回新的采样点。

### Q5. 它的命令是不是一直静止？

不是。默认命令会在 reset 和固定周期内重采样。只有在测试脚本里显式开静态模式，或者把命令范围全部压成 0，才会一直是静止命令。