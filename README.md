# 🐕 Lite3_Handstand_RL

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/PyTorch-1.10-orange" alt="PyTorch 1.10">
  <img src="https://img.shields.io/badge/CUDA-11.3-76B900" alt="CUDA 11.3">
  <img src="https://img.shields.io/badge/Platform-Ubuntu-lightgrey" alt="Platform Ubuntu">
  <img src="https://img.shields.io/badge/IsaacGym-Preview3-6f42c1" alt="IsaacGym Preview 3">
  <img src="https://img.shields.io/badge/MuJoCo-2.x-00bcd4" alt="MuJoCo 2.x">
  <img src="https://img.shields.io/badge/PPO-rsl__rl-ff6f00" alt="PPO rsl_rl">
  <a href="https://github.com/XiaoBaiBZS/Legged_gym_handstand-for-DeepRobotics-JueYing-Lite3"><img src="https://img.shields.io/badge/Original%20Author-XiaoBaiBZS%2FLegged__gym__handstand--for--DeepRobotics--JueYing--Lite3-blue" alt="Original Author Repo"></a>
</p>

## 📖 01. 关于本文 

### 🌟 介绍 
**Legged_gym_handstand-for-DeepRobotics-JueYing-Lite3** 是希望通过强化学习的方式训练一种可以让四足机器人完成“手倒立”动作的方法。本文的最终环节是将训练好的模型部署到云深处绝影 Lite3 机器人上（sim2real）。

本文不会过于注重艰深的算法公式介绍，而是更倾向于**工程流程的跑通**：从模型训练 (train) ➡️ 初步仿真 (play) ➡️ 仿真测试 (sim2sim test) ➡️ 实机部署 (sim2real)。

原作者仓库链接：https://github.com/XiaoBaiBZS/Legged_gym_handstand-for-DeepRobotics-JueYing-Lite3

此仓库基于 `cmjang` 的手倒立仓库修改而来，主要为其适配了 **DeepRobotics JueYing Lite3** 机器狗。笔者使用 RTX 4060 显卡的笔记本电脑，能够正常跑通上述流程。
*(💡 提示：本文涉及的诸多依赖需要良好的网络环境，建议在网络畅通的条件下实验。)*

### ✨ 修改内容概述
1. 🐶 **增加 Lite3 模型**：引入了 DeepRobotics JueYing Lite3 的 URDF 模型及 Mesh 文件。
2. 📐 **精简输入维度**：修改了训练中的输入维度，删除了地形高度测量维度（170维）和基座线速度维度（3维）。目前采用 45维输入、12维输出，使模型更容易接入官方的 `Lite3_rl_deploy` 部署工程进行 sim2sim 与 sim2real 测试。
3. 💾 **导出适配**：修改了 `play.py` 等相关脚本，支持在回放时直接导出 `onnx` 格式的模型文件。

### 🙏 参考与引用
本文的部分内容基于以下优秀开源工作修改或生成，在此表示诚挚感谢：
- [1] [cmjang/legged_gym_handstand](https://github.com/cmjang/legged_gym_handstand)
- [2] [飞书教程参考](https://za8k8pe2ezm.feishu.cn/wiki/N5hFwIrC3isrVckQRRPcx6cHnPs)
- [3] [DeepRoboticsLab/gamepad](https://github.com/DeepRoboticsLab/gamepad)
- [4] [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [5] [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym)
- [6] [fan-ziqi/rl_sar](https://github.com/fan-ziqi/rl_sar)
- [7] [fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab)
- [8] [DeepRoboticsLab/Lite3_rl_deploy](https://github.com/DeepRoboticsLab/Lite3_rl_deploy)

---

## 🛠️ 02. 环境准备 (初步)

### 1️⃣ 创建 Python 虚拟环境 (推荐 3.8)
```bash
conda create -n myenv python=3.8
conda activate myenv
```

### 2️⃣ 安装 PyTorch (以 CUDA 11.3 为例)
*注：如果环境有报错，可基于自身显卡驱动尝试更新版本的 CUDA。*
```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### 3️⃣ 安装 IsaacGym
由于 legged_gym 依赖于 IsaacGym Preview 3：
```bash
cd isaacgym/python && pip install -e .
```
> 🔧 **Troubleshooting:**
> 若运行示例 `python 1080_balls_of_solitude.py` 时提示找不到 `libpython3.8.so.1.0`，可设置环境变量：
> `export LD_LIBRARY_PATH=/opt/conda/envs/myenv/lib:$LD_LIBRARY_PATH`

### 4️⃣ 安装 rsl_rl (PPO 实现)
```bash
cd rsl_rl && git checkout v1.0.2 && pip install -e .
```

### 5️⃣ 安装 legged_gym 与本项目
```bash
# 克隆本项目
git clone https://github.com/XiaoBaiBZS/Legged_gym_handstand-for-DeepRobotics-JueYing-Lite3.git
cd Legged_gym_handstand-for-DeepRobotics-JueYing-Lite3
pip install -e .
```

### 6️⃣ 准备部署环境 Lite3_rl_deploy (推荐在非 conda 环境下编译)
此步骤涉及 MuJoCo 仿真和 C++ 编译，不建议在 conda 虚拟环境内执行，请先 `conda deactivate` 退为系统默认。
```bash
conda deactivate
sudo apt-get install libdw-dev
wget https://raw.githubusercontent.com/bombela/backward-cpp/master/backward.hpp
sudo mv backward.hpp /usr/include

# 安装必要的 Python 依赖
pip install pybullet "numpy < 2.0" mujoco

# 克隆并编译云深处官方部署仓库
git clone --recurse-submodule https://github.com/DeepRoboticsLab/Lite3_rl_deploy.git
cd Lite3_rl_deploy
mkdir build && cd build
cmake .. -DBUILD_PLATFORM=x86 -DBUILD_SIM=ON -DSEND_REMOTE=OFF
make -j
```

---

## 📂 03. 核心文件说明
进入项目 `Legged_gym_handstand-for-DeepRobotics-JueYing-Lite3` 后：

* 🤖 **机器人模型**:
  * `resources/robots/lite3/urdf/lite3.urdf` - Lite3 机器人 URDF 模型与关节/连杆定义。
* ⚙️ **配置与任务**:
  * `legged_gym/envs/lite3/lite3_config.py` - Lite3 任务配置入口，包含初始位置 (`0.42m` 或更低)、动作惩罚和奖励函数设计。
  * `legged_gym/envs/base/legged_robot_config.py` - 通用四足配置基类，其中定义了 `num_observations=45` 和 `num_actions=12`。
* 🌍 **环境实现**:
  * `legged_gym/envs/base/legged_robot.py` - 核心环境逻辑与由于维度裁剪而作的修改位置。
* 🎯 **训练与部署**:
  * `legged_gym/scripts/train.py` - 训练脚本。
  * `legged_gym/scripts/play.py` - 仿真回放并自动导出最新模型的 `onnx` 参数文件。

---

## 🚀 04. 训练与使用流程

### 🏋️‍♂️ 1. 启动训练 (Train)
```bash
cd legged_gym/scripts
export LD_LIBRARY_PATH=/opt/conda/envs/myenv/lib:$LD_LIBRARY_PATH

# CPU 训练测试 (100 次迭代)
python train.py --task=lite3 --max_iterations=100 --headless --sim_device=cpu --rl_device=cpu

# GPU 训练 (500 次迭代)
python train.py --task=lite3 --max_iterations=500 --headless

# 从断点恢复训练 (Resume-train)
python train.py --task=lite3 --resume --load_run May05_14-18-12_ --checkpoint 2500 --headless --max_iterations=1000
```

### 🎮 2. 模型回放与导出 (Play)
该命令不仅会渲染画面，还会在 `logs/onnx/` 路径下生成最新的 ONNX 权重文件：
```bash
mkdir -p ../../logs/onnx
cd legged_gym/scripts
export LD_LIBRARY_PATH=/opt/conda/envs/myenv/lib:$LD_LIBRARY_PATH

# 默认加载最新训练的模型
python play.py --task=lite3

# 指定特定的日志文件夹与检查点
python play.py --task=lite3 --load_run=May04_15-43-59_ --checkpoint=0
```

### 📈 3. 查看 TensorBoard
在本地浏览器中观测训练曲线：
```bash
conda activate myenv
cd /root/gpufree-data/Legged_gym_handstand-for-DeepRobotics-JueYing-Lite3
tensorboard --logdir=logs --port=6006
```

### 🕹️ 4. Sim2Sim 仿真测试 (配合 Lite3_rl_deploy)
导出 `onnx` 并进行 MuJoCo 测试：
```bash
# 复制生成的 onnx 文件到部署库内
cp /root/gpufree-data/Legged_gym_handstand-for-DeepRobotics-JueYing-Lite3/logs/onnx/legged.onnx    /root/gpufree-data/Lite3_rl_deploy/policy/ppo/policy.onnx
```

启动测试（需开启两个终端且都要停用 conda）：
```bash
# 终端 1：启动 MuJoCo 环境
conda deactivate
cd Lite3_rl_deploy/interface/robot/simulation
python3 mujoco_simulation.py

# 终端 2：启动控制核心
conda deactivate
cd Lite3_rl_deploy/build
./rl_deploy
```

**⌨️ 操作指南：**
- 按 `z` 键：机器人进入待机站立模式。
- 按 `c` 键：激活强化学习 (RL) 模式（加载你的模型）。
- 按 `w/a/s/d` 键：控制前后左右平移。
- 按 `q/e` 键：顺/逆时针旋转。

### ⚠️ 5. 危险及兜底策略提示 (Sim2Real 注意)
如果在回放动作（手倒立倒角过大等）时，状态机将其误判为**危险动作**导致切入**关节阻尼状态 (Joint Damping)**：
你可以临时在仿真中屏蔽该安全判断（修改 `Lite3_rl_deploy/state_machine/state_machine.hpp` 约 190 行）：
```cpp
// if(current_controller_->LoseControlJudge()) next_state_name_ = StateName::kJointDamping;
// else next_state_name_ = current_controller_ -> GetNextStateName();

// 直接使用控制器的状态切换 (纯仿真测试用)
next_state_name_ = current_controller_ -> GetNextStateName();
```
修改后重新进入 `build` 目录 `make -j` 编译。

🚨 **郑重警告：**
请**务必**在纯仿真条件下确保其不会炸机再上实机！如果是接入真实的机器狗，必须保留保护程序避免设备损坏和人身伤害！
