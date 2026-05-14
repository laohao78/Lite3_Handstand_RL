# 00.一些重要的文件
## 机器人模型
- [lite3.urdf](./resources/robots/lite3/urdf/lite3.urdf) - Lite3 机器人 URDF 模型与关节/连杆定义。

## 配置与任务
- [lite3_config.py](./legged_gym/envs/lite3/lite3_config.py) - Lite3 任务配置入口，覆盖训练/仿真参数。
- [legged_robot_config.py](./legged_gym/envs/base/legged_robot_config.py) - 通用四足机器人配置基类，其他配置通常继承它。

## 环境实现
- [legged_robot.py](./legged_gym/envs/base/legged_robot.py) - 主要环境逻辑与步进流程实现。
- [base_task.py](./legged_gym/envs/base/base_task.py) - 环境基类与生命周期管理。

## 训练与回放
- [train.py](./legged_gym/scripts/train.py) - 训练入口脚本。
- [play.py](./legged_gym/scripts/play.py) - 加载模型并在仿真中回放。

## 注册与工具
- [task_registry.py](./legged_gym/utils/task_registry.py) - 任务注册与配置加载入口。
- [helpers.py](./legged_gym/utils/helpers.py) - 常用工具函数（含导出策略相关）。

# 01.train
```sh
cd Legged_gym_handstand-for-DeepRobotics-JueYing-Lite3/legged_gym/scripts
export LD_LIBRARY_PATH=/opt/conda/envs/myenv/lib:$LD_LIBRARY_PATH
# CPU-train 100 headless
python train.py --task=lite3 --max_iterations=100 --headless --sim_device=cpu --rl_device=cpu
# GPU-train 500 headless
python train.py --task=lite3 --max_iterations=500 --headless
# Resume-train
python train.py --task=lite3 --resume --load_run May05_14-18-12_ --checkpoint 2500 --headless --max_iterations=1000
```
# 02.play
```sh
# onnx 保存路径 [](./legged_gym/utils/helpers.py) 184行左右
mkdir -p /root/gpufree-data/Legged_gym_handstand-for-DeepRobotics-JueYing-Lite3/logs/onnx
# 会自动加载最新训练的模型
cd Legged_gym_handstand-for-DeepRobotics-JueYing-Lite3/legged_gym/scripts
export LD_LIBRARY_PATH=/opt/conda/envs/myenv/lib:$LD_LIBRARY_PATH
python play.py --task=lite3
# 格式：指定 task + 运行日志文件夹 + 检查点编号
python play.py --task=lite3 --load_run=May04_15-43-59_ --checkpoint=0
```
# 03.tensorboard
```sh
# 在 PowerShell 或 CMD 中建立隧道
ssh -L 6006:localhost:6006 root@183.147.142.40 -p 31112
# 激活环境（如果还没激活）
conda activate myenv
# 进入项目目录
cd /root/gpufree-data/Legged_gym_handstand-for-DeepRobotics-JueYing-Lite3
# 启动 TensorBoard（指向 logs 目录）
tensorboard --logdir=logs --port=6006
# 本地浏览器打开
http://localhost:6006
```
# 04.sim2sim test
```sh
# 复制 onnx
cp /root/gpufree-data/Legged_gym_handstand-for-DeepRobotics-JueYing-Lite3/logs/onnx/legged.onnx \
   /root/gpufree-data/Lite3_rl_deploy/policy/ppo/policy.onnx
```
```sh
# 新开终端1 (mujoco)
conda deactivate
cd Lite3_rl_deploy/interface/robot/simulation
python3 mujoco_simulation.py
# 新开终端2 (control)
conda deactivate
cd Lite3_rl_deploy/build
./rl_deploy
```
```py
- `使用键盘 `z` 键来让 mujoco 中的四足机器人站立
- `使用 `c` 键来切换四足机器人为 rl 模式(你的模型参数)
- `使用 `w/a/s/d` 来控制前后左右平移
- `使用 `q/e` 来顺逆时针旋转。
通过 mujoco 即可看到四足机器人的 sim2sim 效果，如果效果比较不错，可以尝试进行下一步实机部署。
```

若发现当四足机器人执行关键动作的时候由于系统误认为其处在危险动作而进入关节阻尼状态，可以再仿真中关闭此功能，即任何状态下都为力控状态(尽管这比较危险)，具体操作是打开文件Lite3_rl_deploy/state_machine/state_machine.hpp，190 行附近执行y以下操作，注释掉条件判断，直接使用控制器的状态切换。修改完成后 cd 进入 build 目录重新 make -j 编译即可。需要确保在仿真条件下无误后再进行 sim2real 的部署，否则会在实机出现问题(危险状况下)仍然力控，这样很用可能损坏设备甚至伤人。
```cpp
// if(current_controller_->LoseControlJudge()) next_state_name_ = StateName::kJointDamping;
// else next_state_name_ = current_controller_ -> GetNextStateName();

// 直接使用控制器的状态切换
next_state_name_ = current_controller_ -> GetNextStateName();
```
```sh
cd Lite3_rl_deploy
rm -rf ./build
mkdir build && cd build # 编译
cmake .. -DBUILD_PLATFORM=x86 -DBUILD_SIM=ON -DSEND_REMOTE=OFF
# 指令解释
# -DBUILD_PLATFORM：电脑平台，Ubuntu为x86，机器狗运动主机为arm
# -DBUILD_SIM：是否使用仿真器，如果在实机上部署设为OFF 
make -j
```
