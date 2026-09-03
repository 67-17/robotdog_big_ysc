# Lite3机械臂部署说明

本目录用于部署到云深处绝影Lite3机载Ubuntu，只包含机械臂相关运行文件，不包含已经单独部署过的仪表盘识别代码。

机械臂侧负责红色长条识别、末端小范围视觉对准、夹取、保持和放置动作。机器狗导航负责把机械臂移动到纸箱边缘附近，机械臂不承担整张纸箱的大范围搜索。

## 1. 拷贝到机器狗

建议放到机载Ubuntu的固定目录：

```bash
mkdir -p ~/competition
cp -r lite3_arm_runtime ~/competition/
cd ~/competition/lite3_arm_runtime
```

第一次使用时复制环境配置：

```bash
cp lite3_arm.env.example lite3_arm.env
```

如果串口或相机不是默认设备，修改 `lite3_arm.env`：

```bash
ARM_PORT=/dev/ttyUSB0
ARM_CAMERA=/dev/video0
ARM_WIDTH=1280
ARM_HEIGHT=720
ARM_FPS=25
ARM_BAUD=115200
ARM_TIMEOUT=2
```

## 2. 安装依赖

优先使用Ubuntu系统包：

```bash
bash scripts/install_ubuntu_dependencies.sh
```

如需使用虚拟环境，再执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

注意不要混用NumPy 2.x和旧版OpenCV。本包的 `requirements.txt` 已将 `numpy` 限制为 `<2`。

## 3. 通信接口

机械臂串口接口：

- 设备：默认 `/dev/ttyUSB0`，通常为CH340串口。
- 波特率：默认 `115200`。
- 底层工具：`test.py`，通过PySerial发送JSON指令。
- 常用底层命令：`T=105` 查询状态，`T=122` 发送关节/夹爪目标，`T=114` 控制指示灯。

相机接口：

- 设备：默认 `/dev/video0`。
- 分辨率：默认 `1280x720`。
- 帧率：默认 `25`。
- 视觉配置：`strip_detector_grasp_config.json`。
- 标定参数：`camera_calibration.json`。

机器狗与机械臂任务接口：

- 入口文件：`arm_task.py`。
- 调用方式：命令行调用并读取JSON结果。
- 推荐由机器狗导航程序调用脚本，而不是直接拼长命令。
- 结果文件：`logs/last_preflight_result.json`、`logs/last_grasp_result.json`、`logs/last_place_A.json` 等。

抓取结果中需要重点读取：

- `ok`：任务是否成功。
- `stage`：当前或结束阶段。
- `feedback`：给机器狗的协同提示。
- `object_held`：是否认为已经夹住红条。

常见 `feedback` 含义：

- `target_left`：目标偏左，机器狗或机械臂需要向左侧调整。
- `target_right`：目标偏右。
- `target_too_far`：目标太远，需要机器狗靠近纸箱边缘。
- `target_too_near`：目标太近，需要机器狗后退。
- `target_in_grasp_window`：目标已进入可抓窗口。
- `target_lost`：目标丢失，需要重新定位。
- `arm_control_failed`：机械臂动作失败，需要停止本次抓取并检查状态。
- `fallback_cargo_pose`：抓取闭合后已进入兜底货运姿态，需结合现场确认夹持效果。

## 4. 上电和设备检查

接好机械臂12V供电、USB串口和机械臂相机后执行：

```bash
bash scripts/check_devices.sh
```

应确认：

- `/dev/ttyUSB0` 或配置中的串口设备存在；
- `/dev/video0` 或配置中的相机设备存在；
- `python3 test.py ... status` 能返回 `T=1051` 状态。

如果串口权限不足：

```bash
sudo usermod -aG dialout "$USER"
```

然后重新登录机载Ubuntu。

## 5. 视觉调试

只运行红条识别，不让机械臂运动：

```bash
bash scripts/run_vision_debug.sh
```

默认使用无窗口模式，适合机器狗无桌面环境。接显示器时可以手动运行带窗口版本：

```bash
python3 strip_detector.py \
  --device /dev/video0 \
  --config strip_detector_grasp_config.json \
  --calibration camera_calibration.json \
  --width 1280 \
  --height 720 \
  --fps 25
```

## 6. 抓取前预检

```bash
bash scripts/run_preflight.sh
```

预检结果会写入：

```bash
logs/last_preflight_result.json
```

只有预检通过后再执行真实抓取。

## 7. 执行抓取

机器狗导航到纸箱边缘附近后，调用：

```bash
bash scripts/run_grasp.sh
```

抓取结果写入：

```bash
logs/last_grasp_result.json
```

完整过程日志写入：

```bash
grasp_runs/
```

机器狗侧建议逻辑：

```text
运行 run_grasp.sh
读取 logs/last_grasp_result.json
ok=true 且 object_held=true：进入运输或放置流程
feedback=target_left/right/too_far/too_near：导航小幅调整后重试
feedback=target_lost：重新巡检或重新靠近纸箱边缘
feedback=arm_control_failed：停止并人工检查机械臂状态
```

## 8. 执行放置

巡检模块给出异常区域字母后，调用对应放置槽：

```bash
bash scripts/run_place.sh A
```

`A` 可替换为 `B`、`C` 或 `D`。放置结果会写入：

```bash
logs/last_place_A.json
```

当前 `place_reference.json` 中的A/B/C/D放置姿态需要在机器狗实装状态下复核。

## 9. 常用维护命令

查询机械臂状态：

```bash
python3 test.py --port /dev/ttyUSB0 status
```

回零：

```bash
bash scripts/run_home.sh
```

只打开或闭合夹爪：

```bash
python3 test.py --port /dev/ttyUSB0 gripper open --execute
python3 test.py --port /dev/ttyUSB0 gripper close --execute
```

查看最近一次抓取日志：

```bash
python3 arm_task.py diagnose-run
python3 arm_task.py validate-run
```

## 10. 上线前确认

- 机器狗停稳后再允许机械臂动作。
- 机械臂、狗体和线缆之间不能互相拉扯。
- 红条应先进入相机可抓窗口，再让机械臂执行夹取。
- 不要让机械臂独自覆盖整张纸箱搜索范围。
- e肘关节曾出现不到位和 `move=1` 粘住记录，上场前需在当前供电和安装姿态下重新跑一遍 `check_devices`、`run_preflight`、`run_grasp` 和 `run_place`。
