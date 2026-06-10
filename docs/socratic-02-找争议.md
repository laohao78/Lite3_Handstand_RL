# ⚔️ Socratic Learn — Step 2 找争议

<p align="center">
  <img src="https://img.shields.io/badge/Topic-Lite3%20Handstand%20RL-blue" alt="Topic">
  <img src="https://img.shields.io/badge/Method-Socratic%20Learn-orange" alt="Method">
  <img src="https://img.shields.io/badge/Step-2%2F3%20找争议-red" alt="Step 2/3">
  <img src="https://img.shields.io/badge/Date-2026--06--10-lightgrey" alt="Date">
</p>

---

## 📌 第二步：找争议（Find the Controversies）

> 🔑 **核心问题：** 在「四足机器人手倒立强化学习控制与 Sim2Real 部署」这个领域里，专家吵得最凶的三个问题是什么？各自站哪边？各自的核心理由是什么？

---

## 争议一：Sim2Real — 位置控制 vs 力矩控制

<p align="center">
  <img src="https://img.shields.io/badge/△-底层分歧：物理建模精度应该放在哪？-critical" alt="Core">
</p>

|  | 阵营 A：位置控制（P Control） | 阵营 B：力矩控制（T Control） |
|:--|:--|:--|
| 🏷️ **立场** | 策略输出关节目标角度，PD 转为力矩 | 策略直接输出关节力矩 |
| 🧑‍🔬 **代表** | ETH Rudin、本项目、legged_gym | MIT Cheetah、fan-ziqi/rl_sar |
| ✅ **核心论据** | PD 对建模误差鲁棒。URDF 质量不准？不慌，PD 会自己补偿。Sim2Real 时不用重新辨识物理参数 | 直接力矩控制给策略更大的表现空间。PD 把策略的"创造力"限制在一个低带宽的盒子里面 |
| 🗣️ **反驳对方** | 你力矩控制到真机上，URDF 惯量差 20%，力矩就直接错 20%，策略当场发散 | 你 PD 控制等于在策略输出后面接了个低通滤波器，高动态行为（跑酷、跳跃）PD 跟不上 |

```
位置控制 (本项目)                           力矩控制 (MIT Cheetah 系)
─────────────────                         ──────────────────────
策略 → 目标角度 → PD → 力矩 → 电机         策略 → 力矩 → 电机
       ↑                    ↑                    ↑
    PD 兜底补偿            物理                直接输出
    Sim2Real gap 小                         表达能力强
    但带宽受限                               但 Sim2Real 风险大
```

> 🔗 **关联骨架：** [[socratic-01-抓骨架.md#模型四PD-控制器与执行层low-level-control-interface|模型四 PD 控制]] + [[socratic-01-抓骨架.md#模型五sim2real-迁移链simulation-to-reality-transfer-pipeline|模型五 Sim2Real]]。本项目明显站在阵营 A。

---

## 争议二：领域随机化 vs 系统辨识

<p align="center">
  <img src="https://img.shields.io/badge/△-底层分歧：鲁棒性 vs 最优性能-critical" alt="Core">
</p>

|  | 阵营 A：Domain Randomization | 阵营 B：System Identification |
|:--|:--|:--|
| 🏷️ **立场** | 训练时随机扰动物理参数，让策略"见惯不怪" | 精确测量真机物理参数，建高保真数字孪生 |
| ✅ **核心论据** | 策略天然鲁棒，不依赖任何参数精度。摩擦 0.5 还是 1.2、质量 4kg 还是 6kg，一律通吃 | 随机化会让策略"保守"——永远为了最坏情况牺牲性能。精确模型才能训出最优策略 |
| 🗣️ **反驳对方** | 你数字孪生再准，跑三天老化了参数就变了，策略当场废 | 你随机化范围太大，策略学了一个"平均解"而不是最优解，倒立这种极限动作根本收敛不了 |

```
Domain Rand (本项目)                       System ID (数字孪生派)
─────────────────                        ─────────────────────
训练: 每轮随机摩擦/质量/推搡              训练: 用实测参数建精确仿真
结果: 策略对变化不敏感                     结果: 策略在特定参数下最优
优点: 部署即用，不挑环境                    优点: 极限性能更高
缺点: 可能偏保守，收敛慢                    缺点: 参数飘了就废
```

> 🔗 **关联骨架：** [[socratic-01-抓骨架.md#模型二大规模并行仿真与环境课程parallel-simulation--curriculum|模型二 Domain Rand]] + [[socratic-01-抓骨架.md#模型五sim2real-迁移链simulation-to-reality-transfer-pipeline|模型五 Sim2Real]]。本项目站 A，但手倒立这种极限动作需要把随机化范围收窄，不然收敛不了。

---

## 争议三：稠密奖励工程 vs 稀疏奖励 + 课程

<p align="center">
  <img src="https://img.shields.io/badge/△-底层分歧：复杂度放在奖励设计还是课程设计？-critical" alt="Core">
</p>

|  | 阵营 A：Dense Reward Engineering | 阵营 B：Sparse Reward + Curriculum |
|:--|:--|:--|
| 🏷️ **立场** | 精心设计 10+ 个奖励分量，每个调权重 | 只给最终目标奖励（如"脚离地=1"），靠课程引导 |
| ✅ **核心论据** | 手倒立这种复杂行为，光靠稀疏奖励探索永远找不到，必须每一步都告诉策略"你离目标还有多远" | 奖励分量越多，越容易训出 reward hacking——策略找到一个讨奖励函数开心但不是你想要的解。19 个分量调参玄学，换一个任务全废 |
| 🗣️ **反驳对方** | 你说稀疏奖励可以靠课程，但课程本身也要设计，课程设计错了比奖励调参更难 debug | 你这个项目 `feet_height_exp=17.5`、`front_feet_contact=-40`、抬腿阈值=0.025——全是手调的工程魔法，换到翻滚、跳跃任务全部推倒重来 |

```
Dense Reward (本项目)                      Sparse Reward (OpenAI 系)
─────────────────                         ────────────────────────
10+ 奖励分量                               1-2 个最终目标奖励
+ 精细权重调参                             + 课程学习引导探索
+ 多级奖励系统 (抬腿/高度/姿态)              + 自动域随机化
优点: 训练信号密集稳定                      优点: 泛化性好，少 reward hacking
缺点: 调参量大，任务迁移需重做               缺点: 探索困难，可能永远学不到
```

> 🔗 **关联骨架：** [[socratic-01-抓骨架.md#模型一observation-action-reward-三元组rl-环境设计|模型一 Reward 设计]] + [[socratic-01-抓骨架.md#模型二大规模并行仿真与环境课程parallel-simulation--curriculum|模型二 渐进课程]]。本项目是个**混合体**——既有 10+ 稠密奖励（站 A），又有渐进姿态课程（站 B）。

---

## 🗺️ 争议 vs 骨架 对照

```
争议一 (位置 vs 力矩)  ──→  模型四 (PD)     + 模型五 (Sim2Real)
争议二 (随机化 vs 辨识) ──→  模型二 (Rand)    + 模型五 (Sim2Real)
争议三 (稠密 vs 稀疏)   ──→  模型一 (Reward)  + 模型二 (课程)
```

```
                    模型一 (Reward)
                        │
                   争议三 🔥
                        │
        模型二 (Rand/课程) ───── 模型三 (PPO) ───── 模型四 (PD)
              │                                         │
         争议二 🔥                                 争议一 🔥
              │                                         │
              └────────── 模型五 (Sim2Real) ─────────────┘
                              ↑
                    三个争议的交汇点
```

---

## ❓ 反问

这三个争议里，**你最站哪边？为什么？**
