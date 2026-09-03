# 云深处机器狗比赛

这是一个围绕 2026 四足大型组任务整理的仓库。仓库里同时保留了原始资料、示例代码和当前主要可运行代码。

## 目录说明

- `mission_lite3/`：当前主要任务代码，包含任务状态机、运动控制、视觉识别、机械臂适配和运行入口。
- `26比赛资料/`：比赛原始资料和历史示例代码，包含 `DeepRobotDog-main/` 及地图、题目、素材文件。
- `lite3资料/`：Lite3 相关说明文档。
- `tests/`：烟测和基础测试。

## 主要入口

### 任务程序

```bash
python -m mission_lite3.run_mission --dry-run --skip-arm
```

机器人回到比赛起点、夹爪清空且现场急停就绪后，一键启动完整实机任务：

```bash
scripts/run_full_mission.sh
```

该入口会加载 ROS2 Foxy，并执行巡检、两轮抓取投放和最终收纳。

任务起步后的 `PASS_OBSTACLE` 已集成 `startup_avoidance`：使用
`/dev/video0` 识别红色锥桶，结合前向超声波和 `/leg_odom2`
穿越障碍区。第一次横移后必须向前走满 `1.50 m` 才回到原行进线；如果这段前进中
遇到第二个障碍，则再次向左横移并清零前一段累计，从最后一次横移完成点重新走满
`1.50 m`，最后一次性回到第一次绕障前的行进线。如果没有障碍则直接巡航 `1.50 m`。
该过程与主任务共用唯一的运动控制器，不会额外发布一套 `/cmd_vel`。
详细参数见 `mission_lite3/config/robot.yaml`，每次运行的逐帧决策日志写入
`startup_avoidance_runs/`。

实机任务启用有界 `FAULT_HOLD`：相机、超声波、里程计、机器人状态、运动接口或
后续任务出现异常时，程序持续发送零速度并尝试恢复，但单次等待最多 `30 s`，每个
任务状态最多重试 `2` 次。启动检查、站立与机械臂准备、开场避障、结果汇报和最终
收纳在安全状态连续稳定 5 次后自动重试；巡检移动、抓取和放置可在 30 秒窗口内由
操作员确认现场状态后执行：

```bash
touch /tmp/lite3_fault_resume
```

确认信号会保留到安全检查稳定后再恢复。等待超时或重试次数用尽后，程序执行停车、
释放相机和运动接口等清理并以失败码退出，不再无限占用进程。`Ctrl+C`、进程被系统
终止、主机掉电或操作系统故障仍会终止任务。

### 视觉调试

```bash
python -m mission_lite3.run_mission --vision-test --headless --vision-frames 120
python -m mission_lite3.tools.vision_debug --hsv-key red_hsv_1
```

### 单站点巡检

```bash
python3 run_live_inspection.py --no-window --json-terminal --exit-after-stable
```

稳定识别后会写入 `latest_stop_result.json` 和 `round_result.json`，供主任务读取异常区域。

### 机械臂调试

机械臂代码已经整合到 `mission_lite3/arm/runtime/`，默认通过 `mission_lite3/arm/lite_arm.py` 调用 runtime 后端。串口和机械臂相机设备名以 `mission_lite3/config/robot.yaml` 为准，也可以现场临时覆盖：

```bash
scripts/check_arm_devices.sh
python3 tools/test_arm_standalone.py
python3 tools/test_arm_standalone.py --scenario preflight --real
python3 tools/test_arm_standalone.py --scenario transport --real --yes
scripts/run_arm_task.sh --dry-run grasp
scripts/run_arm_task.sh preflight
ARM_PORT=/dev/ttyUSB1 ARM_CAMERA=/dev/video2 scripts/run_arm_task.sh preflight
```

`tools/test_arm_standalone.py` 只测试机械臂 runtime，不启动机器狗任务状态机。默认是 dry-run；带 `--real` 后才连接真实硬件，涉及机械臂运动的场景还必须加 `--yes`。

抓取视觉、相机标定、抓取参考和 A/B/C/D 放置姿态在 `mission_lite3/arm/runtime/` 下集中调参。

### 音频探测

先只测声音，不让机器狗运动：

```bash
python3 run_speech_probe.py
```

脚本默认向运动主机 `192.168.1.120:43910` 发送 UDP 播报命令，由狗身扬声器播放 `/opt/robot_competition/inspection_audio_test/` 中的 WAV。运动主机需先启动服务：

```bash
scripts/install_remote_audio_service.sh
```

狗身扬声器默认应用 `audio.remote_gain_db=3.0` 的数字增益。升级任务代码后必须重新执行安装脚本升级运动主机服务。可临时比较音量：

```bash
python3 run_speech_probe.py --remote-gain-db 0
python3 run_speech_probe.py --remote-gain-db 3
```

需要排查感知主机本机音频时再运行：

```bash
python3 run_speech_probe.py --local-audio
python3 run_speech_probe.py --local-audio --diagnose --skip-beep --skip-speech
python3 run_speech_probe.py --local-audio --test-alsa-devices
```

### 超广角巡检

视觉默认使用从 `../robot_runtime` 移植的 `runtime_meter_anchor` 后端，适配超广角画面：先定位仪表盘圆心，再推导字母和仪表盘 ROI。主任务和 `run_live_inspection.py` 共用同一套 `VisionPipeline` 接口。
正式任务不会把第 1/2/3/4 个停车点固定为 A/B/C/D；A/B/C/D 由相机识别结果决定，异常区域再用于后续抓取红色物块并投放到对应字母区域。

### 运动标定

```bash
python -m mission_lite3.tools.calibrate_motion --mode forward --speed 0.15 --duration 2
```

检查脚本路线积分后的近似坐标和航向：

```bash
python3 -m mission_lite3.tools.validate_route
```

### 测试

```bash
bash check_robot_runtime.sh
```

该检查会运行全部 `unittest` 用例，不依赖 pytest 或 ROS pytest 插件。

### 实机测试流程

详细步骤见 `mission_lite3/README.md` 的“实机测试流程”章节。推荐顺序是：

1. `bash check_robot_runtime.sh`
2. `python3 -m mission_lite3.run_mission --dry-run --skip-arm`
3. `scripts/test_box_centers.sh`，人工完成放置区和抓取区两处只读识别
4. `scripts/test_box_center_step.sh --robot --yes --target-letter D`，再对 A 箱重复一次限幅横移和自动回撤测试
5. `python3 run_live_inspection.py --no-window --json-terminal --reset-round --no-speak --max-frames 120`
6. `scripts/run_full_mission.sh --skip-arm`
7. `scripts/check_arm_devices.sh` 和 `scripts/run_arm_task.sh preflight`
8. 人工将机器人放在抓取区纸箱中心、前向距离约 `0.80 m`，先运行 `scripts/run_pickup_transfer_mission.sh --targets A D` 干跑
9. 急停人员就位后运行 `scripts/run_pickup_transfer_mission.sh --robot --yes --targets A D`，专项验证两次抓取、运输和 A/D 投放
10. 确认机器人已回到起点且夹爪清空后，运行 `scripts/run_full_mission.sh`

第 4、6、9、10 步会发送真实机器狗运动指令；第 7、9、10 步还可能驱动机械臂。详细的起始姿态、通过标准和日志判读见 `mission_lite3/README.md`。

当前路线已按 `26比赛资料/比赛具体位置图.png` 重规划，详细拓扑和可调距离见 `mission_lite3/config/field.yaml` 的 `scripted_route`。

抓取成功后的投放流程为：后退到抓取箱前约 `0.80 m`，依次完成抓取区偏航和中心修正，
再旋转 `180°` 面向投放区。机器狗以 `0.08 m/s` 连续前进，前向超声波达到
`0.80 m` 后停车并首次启用字母识别。若未识别到目标字母，则以该停车点为原点先向左
搜索 `1.00 m`，再反向越过原点搜索到右侧 `1.00 m`；目标字母在画面任意水平位置首次
出现时立即停车并开始放置。完整左右搜索的最坏实际横移累计为 `3.00 m`。

## 依赖

`mission_lite3/requirements.txt` 中列出了当前 Python 依赖。一般先安装：

```bash
pip install -r mission_lite3/requirements.txt
```

## 文档入口

- `mission_lite3/README.md`：当前任务代码的详细说明
- `26比赛资料/DeepRobotDog-main/README.md`：原始示例工程说明
- `lite3资料/`：Lite3 设备和接口资料

## 说明

仓库中保留了比赛所需的大体量图片、PDF、Word 和示例资源。`.gitignore` 已排除压缩包、缓存和 `__pycache__` 等临时文件。
