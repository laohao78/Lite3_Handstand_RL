# 一些重要的文件
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
# 运行 play.py（会自动加载最新训练的模型）
# onnx 保存路径 [](./legged_gym/utils/helpers.py) 184行左右
mkdir -p /root/gpufree-data/Legged_gym_handstand-for-DeepRobotics-JueYing-Lite3/logs/onnx
python play.py --task=lite3
# 格式：指定 task + 运行日志文件夹 + 检查点编号
python play.py --task=lite3 --load_run=May04_15-43-59_ --checkpoint=0
```
# 02.tensorboard
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
# 03.sim2sim test
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
- `使用 `c` 键来切换四足机器人为 rl 模式
- `使用 `w/a/s/d` 来控制前后左右平移
- `使用 `q/e` 来顺逆时针旋转。
通过 mujoco 即可看到四足机器人的 sim2sim 效果，如果效果比较不错，可以尝试进行下一步实机部署。
```












# train、play、sim2sim test and sim2real #
### train ###
```bash
cd legged_gym_handstand/legged_gym/script
python train.py --task=lite3 --headless
```
首先先 cd 进入代码文件夹，然后运行 train.py 即可，传入参数 task=lite3 ，可选使用无头模式运行，使用无头模式运行能够显著提高训练效率，推荐加上 --headless 。

若运行中出现了下面的报错不妨试试：
```bash
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
conda activate env_isaacgym
```
1. 找到conda环境中的libpython库路径
	```bash
	find ~/miniconda3/envs/env_isaacgym -name "libpython3.8.so.1.0" 2>/dev/null
	```
2. 设置LD_LIBRARY_PATH（根据实际路径调整）
	```bash
	export LD_LIBRARY_PATH=~/miniconda3/envs/env_isaacgym/lib:$LD_LIBRARY_PATH
	```

若运行中出现了找不到 lite3 任务的情况，不妨把前文提到的关于 lite3 的修改同步复制到 legged_gym/resources 和legged_gym/legged_gym/env 里面。笔者遇到了类似的问题，如此修改便解决，但是不清楚是否由于此引起。

如果训练不小心中断，不要担心，你可以根据检查点恢复训练，首先先让我们打开 logs 目录，这个目录下面存放着训练时候产生的检查点 pt 文件，让我们找到训练的实验名称文件夹(比如：rough_lite3)，然后找到你上次训练的日期的文件夹，这里面存放着检查点，如果你没有对仓库进行修改，这里应该是每50次训练存储一次。我们可以使用如下指令恢复训练，但是需要注意的是恢复训练指的是“在指定检查点的基础上重新运行多少轮”，所以如果你不改变训练轮数，他依然会接续训练相同次数。
```bash
python train.py --task=lite3 --resume  --load_run Oct31_22-14-50_ --checkpoint 18600 --headless
```
其中 --load_run 后面需要加上 logs 下面你之前训练存储的文件目录名称； --checkpoint 指的是从哪个检查点接续训练。
### play ###
```bash
cd legged_gym_handstand/legged_gym/script
python play.py --task=lite3
```
首先先 cd 进入代码文件夹，然后运行 play.py 即可，传入参数 task=lite3 ，运行完成后即可看到训练的效果，然后就可以在对应目录下找到生成的 .onnx 文件。

可以通过 tensorboard 来帮助你进行训练状况的查看。如果 tensorboard 的使用中出现了安装问题，不妨让 ai 帮帮你。
```bash
cd ~/legged_gym_handstand/
tensorboard --logdir=logs/rough_lite3

# 然后浏览器访问 http://localhost:6006/ 查看
```

![[Pasted image 20251101074410.png]]
### sim2sim test ###
如果在上一步骤中获得到的效果比较理想，可以进行 sim2sim test ，将直接训练好的模型部署到实机上往往容易失败，而且比较危险，很有可能弄坏设备，因此我们先进行 sim2sim test ，mujoco 是一个与真实环境比较类似的仿真环境，我们需要使用由云深处提供的 repo ，进行 sim2sim test ，这一步的操作推荐在非 conda 环境内进行，前文已经提及，笔者在 conda 环境中执行 sim2sim test 会报错。我们将得到的 .onnx 文件复制到 Lite3_rl_deploy/policy/ppo 文件夹中，然后修改 .onnx 文件的名称为 policy.onnx ，然后我们打开两个终端，一个终端启动仿真 mujoco ，另一个启用键盘控制。
```bash
cd Lite3_rl_deploy

# 终端1 (mujoco)
cd interface/robot/simulation
python3 mujoco_simulation.py

# 终端2 
cd build
./rl_deploy
```
使用键盘 `z` 键来让 mujoco 中的四足机器人站立，然后使用 `c` 键来切换四足机器人为 rl 模式，使用 `w/a/s/d` 来控制前后左右平移，使用 `q/e` 来顺逆时针旋转。通过 mujoco 即可看到四足机器人的 sim2sim 效果，如果效果比较不错，可以尝试进行下一步实机部署。

若发现当四足机器人执行关键动作的时候由于系统误认为其处在危险动作而进入关节阻尼状态，可以再仿真中关闭此功能，即任何状态下都为力控状态(尽管这比较危险)，具体操作是打开文件Lite3_rl_deploy/state_machine/state_machine.hpp，190 行附近执行y以下操作，注释掉条件判断，直接使用控制器的状态切换。修改完成后 cd 进入 build 目录重新 make -j 编译即可。需要确保在仿真条件下无误后再进行 sim2real 的部署，否则会在实机出现问题(危险状况下)仍然力控，这样很用可能损坏设备甚至伤人。
```cpp
// if(current_controller_->LoseControlJudge()) next_state_name_ = StateName::kJointDamping;
// else next_state_name_ = current_controller_ -> GetNextStateName();

// 直接使用控制器的状态切换
next_state_name_ = current_controller_ -> GetNextStateName();
```
### sim2real ###
首先先让我们将电脑与四足机器人连接到同一个网络，即电脑连接四足机器人的 wifi ，通常是 ysc- 开头的网络。密码通常是 12345678 。然后我们使用 scp 传输数据。
```bash
# scp传输文件 (打开本地电脑终端)
scp -r ~/Lite3_rl_deploy ysc@192.168.2.1:~/
```
然后我们使用 ssh 连接机器狗，密码通常是英文单引号 `'` 。
```bash
# ssh连接机器狗运动主机以远程开发，密码有以下三种组合
#Username	Password
#ysc		' (a single quote)
#user		123456 (推荐)
#firefly	firefly
ssh ysc@192.168.2.1
# 输入密码后会进入远程开发模式
```
然后 cd 到 Lite3_rl_deploy 的文件夹。并按照下面的操作进行编译，推荐先将 build 目录删除，重新编译。否则可能会报错。
```bash
# 编译
cd Lite3_rl_deploy

# 删除 build 文件夹（可选）
rm -rf ./build

# 编译
mkdir build && cd build
cmake .. -DBUILD_PLATFORM=arm -DBUILD_SIM=OFF -DSEND_REMOTE=OFF
# 指令解释
# -DBUILD_PLATFORM：电脑平台，Ubuntu为x86，机器狗运动主机为arm
# -DBUILD_SIM：是否使用仿真器，如果在实机上部署设为OFF 
make -j 
./rl_deploy
```
执行完上述指令后四足机器人应当关节回零然后我们拿出手柄，连接机器狗的 wifi ，然后掌机上需要提前安装好云深处为强化学习测试开发的遥控 App “云深处科技”，应用图标是蓝色的 Flutter 图标。![[Pasted image 20251031115100.png]]然后我们进入 App ，点击左上角更多按钮，然后修改端口号为：192.168.1.120:12121 ，然后点击右上角保存退出重新进入应用。![[Pasted image 20251031115136.png]]
然后我们回到电脑，通过 ssh 连接四足机器人，cd 进入并使用 vim 打开 jy_exe/conf/network.toml 文件，修改数据上报的 ip ，我们希望其返回关节信息、传感器信息到我们的电脑，修改此处的 ip 为 192.168.2.1 。然后我们试着使用手柄让四足机器人起立，我们可以按下手柄右侧字母 Y 键，可以看到四足机器人起立，为平地站立静止状态。然后我们可以在确保安全的条件下按下手柄右侧字母 A 键，此时机器人将会进入到 rl 力控状态。若不希望继续进行，可以 ctrl+c 终止 ./rl_deploy 进行。

