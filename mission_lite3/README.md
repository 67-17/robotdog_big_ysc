# Lite3 四足大型组实赛工程

本目录是 2026 四足大型组任务的实赛代码工程。资料目录 `lite3资料/` 和 `26比赛资料/` 只作为文档、地图、样例代码来源；正式运行代码集中放在 `mission_lite3/`，避免污染原始资料。

代码默认部署在 Lite3 感知主机 Jetson Xavier NX 上，优先使用 ROS2 `/cmd_vel` 控制运动，并订阅运动主机状态话题；当 ROS2 链路不可用时，可切换到 Lite3 运动主机 UDP 控制。比赛规则禁止激光雷达，因此本工程不调用 Lite3 激光 SLAM、雷达定位导航或智能避障链路，只使用相机、腿部里程计、IMU、超声波和机械臂。

## 目录结构

```text
mission_lite3/
  config/
    field.yaml          # 场地尺寸、区域、航点配置
    robot.yaml          # Lite3 IP、ROS2 topic、相机、速度、安全、HSV、机械臂配置
  arm/
    lite_arm.py         # 机械臂适配层，默认调用包内 runtime 后端
    runtime/            # TLS 串口控制、红条视觉伺服、固定放置序列和示教工具
  tools/
    calibrate_motion.py # 运动标定工具
    vision_debug.py     # HSV 视觉阈值调试工具
  vision/
    color.py            # 锥桶、红条、绿条 HSV 检测
    letters.py          # A/B/C/D 字母识别
    dashboard.py        # 仪表盘指针识别
    pipeline.py         # 巡检视觉总流程
  camera.py             # OpenCV 相机/RTSP 封装
  state_reader.py       # ROS2 里程计、IMU、超声波状态读取
  lite3_motion.py       # ROS2 / UDP 运动控制统一接口
  mission.py            # 大型四足任务状态机
  run_mission.py        # 主入口
  requirements.txt      # Python 依赖
```

## 运行环境

推荐在 Lite3 感知主机上运行：

- Python 3.8 或更高版本
- ROS2 环境和 `transfer_ros2` 服务
- OpenCV、NumPy、PyYAML、pyserial
- 可访问 Lite3 运动主机，默认 `192.168.1.120:43893`
- 可访问前向相机，默认 `rtsp://192.168.1.120:8554/test`

安装 Python 依赖：

```bash
pip install -r mission_lite3/requirements.txt
```

如果在 Lite3 原厂系统中运行，ROS2 相关包通常由系统环境提供，不建议用 `pip` 安装 ROS2。

## 快速启动

开发电脑干跑，不发送任何硬件指令：

```bash
python -m mission_lite3.run_mission --dry-run --skip-arm
```

在 Lite3 上使用 ROS2 `/cmd_vel` 运行：

```bash
scripts/run_full_mission.sh
```

ROS2 不稳定时，改用 UDP 速度指令：

```bash
scripts/run_full_mission.sh --udp-fallback
```

最保守的 UDP 轴指令 fallback：

```bash
scripts/run_full_mission.sh --udp-fallback --axis-fallback
```

禁用机械臂，只测试巡检、运动、播报流程：

```bash
scripts/run_full_mission.sh --skip-arm
```

需要在巡检停车时弹窗显示前向相机画面和识别结果时加：

```bash
scripts/run_full_mission.sh --skip-arm --inspection-window
```

## 视觉调试

单张图片检测：

```bash
python -m mission_lite3.run_mission --vision-test path/to/image.png
```

使用配置中的前向相机预览 300 帧：

```bash
python -m mission_lite3.run_mission --vision-test --vision-frames 300
```

无图形界面环境只打印检测结果：

```bash
python -m mission_lite3.run_mission --vision-test --headless --vision-frames 120
```

HSV 阈值交互调试，例如调红色长条：

```bash
python -m mission_lite3.tools.vision_debug --hsv-key red_hsv_1
```

可选的 `--hsv-key`：

- `cone_hsv`：橙色锥桶
- `red_hsv_1`：红色低 Hue 段
- `red_hsv_2`：红色高 Hue 段
- `green_hsv`：绿色长条

在调试窗口中按 `p` 打印当前 HSV 范围，按 `q` 退出。

## 单站点巡检结果

`run_live_inspection.py` 是轻量巡检入口，复用 `mission_lite3.vision` 的字母、仪表盘和时序稳定逻辑：

```bash
python3 run_live_inspection.py --no-window --json-terminal --exit-after-stable
```

上机无 `DISPLAY` 时会自动进入无窗口模式，避免 OpenCV 窗口崩溃。常用参数：

- `--reset-round`：先原子写入空的 `round_result.json`，本轮重新累计。
- `--max-read-failures 30`：连续读帧失败达到阈值后写入 `block_reason=camera_failed` 并非零退出。
- `--no-speak-on-repeat`：默认启用，同一区域同一结果不重复播报。

稳定结果写入：

- `latest_stop_result.json`：单站点最新结果。
- `round_result.json`：四站点累计结果，保留 `abnormal_areas`、`unknown_areas`、`count_check`、`ready`，并补充 `timestamp`、`source_camera`、`stability_votes`、`evidence_image`、`block_reason`。

主任务在抓取前会读取 `round_result.json`；如果该文件存在，只有 `ready=true` 且 `unknown_areas=[]` 才进入抓取投放，否则跳过抓取并打印阻塞原因。

结果同时带有 `schema_version` 和本轮唯一 `run_id`。主任务只接受当前启动轮次的结果，旧文件即使 `ready=true` 也不能触发抓取。

## 运动标定

先干跑确认命令格式：

```bash
python -m mission_lite3.tools.calibrate_motion --mode forward --speed 0.15 --duration 2
```

在空旷区域让机器人实际前进 2 秒：

```bash
python -m mission_lite3.tools.calibrate_motion --robot --yes --mode forward --speed 0.15 --duration 2
```

支持的 `--mode`：

- `forward`：前进
- `backward`：后退
- `left`：左平移
- `right`：右平移
- `turn-left`：左转
- `turn-right`：右转

标定时记录实际距离或角度，回填到 `config/robot.yaml` 的速度、限幅和路线参数。

正式任务默认用腿部里程计闭环结束直行/横移、用 IMU 闭环结束转向，并在每个速度控制周期检查姿态和传感器新鲜度。`--allow-open-loop` 只把动作终止方式降级为计时，仍保留状态安全检查，仅限受控标定使用。

路线参数修改后先查看积分结果：

```bash
python3 -m mission_lite3.tools.validate_route
```

## 关键配置

`config/robot.yaml` 中常用配置：

- `network.motion_ip`：Lite3 运动主机 IP，默认 `192.168.1.120`
- `network.motion_port`：Lite3 UDP 端口，默认 `43893`
- `ros2.cmd_vel_topic`：速度控制 topic，默认 `/cmd_vel`
- `ros2.odom_topic`：腿部里程计 topic，默认 `/leg_odom2`（`nav_msgs/msg/Odometry`）
- `ros2.imu_topic`：IMU topic，默认 `/imu/data`
- `ros2.ultrasound_topic`：前向超声波 topic，当前为 `/us_publisher/front_distance`；抓取后退不读取后向超声波
- `navigation.*_tolerance`：闭环距离和航向容差；`action_timeout_scale` 控制动作超时保护
- `camera.front`：前向相机源，可以是 RTSP、视频文件或设备编号
- `camera.arm`：旧抓取相机源配置；启用 runtime 后优先使用 `arm.camera_device`
- `camera.digital_zoom`：前向巡检相机数字放大倍数，当前为 `1.0`，保留完整 `1280×720` 广角画面
- `inspection.use_wide_undistortion`：巡检识别前应用 `camera.wide_calibration` 去畸变，当前启用
- `camera.flush_grab_frames`：每次读帧前丢弃的旧缓冲帧数，当前默认 `4`，用于降低 RTSP 延迟
- `vision.inspection_backend`：巡检识别后端，默认 `runtime_meter_anchor`；加载失败会阻止任务，只有明确配置 `legacy` 或 `allow_legacy_fallback=true` 才会降级
- `motion.max_vx/max_vy/max_wz`：速度限幅
- `safety.front_stop_distance_m`：前向超声波急停距离，当前为 `0.35 m`
- `safety.use_vision_obstacle`：视觉锥桶避障开关，当前为 `false`，避免橙黄色背景误触发；前向超声波避障仍启用
- `inspection.front_stop_distance_m`：仅在左右巡检任务阶段覆盖前向急停距离，当前为 `0.28 m`
- `pregrasp_red_align.ultrasound_gate_enabled/min_m/max_m`：抓取前要求新鲜的前向超声波样本，默认作为 `0.10–2.00 m` 的宽范围安全门；正式抓取流程先连续前进到 `28 cm`，再开始红块横向搜索
- `pregrasp_red_align.final_distance_max_m`：抓取前横移后的距离通过上限，当前为严格小于 `0.30 m`；超过或等于上限时根据当前距离补前进到 `28 cm`，每轮只执行一步，不做连续小脉冲微调
- `pregrasp_red_align.final_distance_attempts`：最终距离判断机会总数，当前为 `3`；对应第一次测距以及最多两次按最新距离补前进到 `28 cm` 后的复测
- `pickup_transfer.pre_retreat_yaw_alignment_enabled=false`：取消抓取前及后退前的纸箱偏航矫正；抓取前的红块横移仍执行
- `pickup_transfer.post_retreat_yaw_alignment_enabled=true`：抓取后先后退到约 `0.80 m`，再使用抓取区纸箱做偏航矫正；识别失败或视觉异常只告警并继续抓取区中心横移
- `pickup_transfer.retreat_stuck_front_*`：前向超声波连续至少 `5` 帧卡在 `0.28±0.01 m` 时，允许以腿部里程计 `0.44 m` 作为后退完成条件；`retreat_max_odom_m=0.55` 仍是硬停止上限
- `pickup_transfer.departure_tolerance_fraction/arrival_tolerance_fraction`：抓取后离开时纸箱中心误差为画面宽度 `±3%`，第二次返回抓取区时放宽为 `±6%`
- `placement_yaw_alignment.enabled=false`：暂时取消旋转 `180°` 后基于四个放置箱的偏航矫正；路线动作保留，但只打印 `disabled` 后继续箱区横移
- 横移脉冲按物块中心到目标 ROI 边界的水平像素差线性调整：`min_pulse_seconds` 对应最小误差，误差接近画面边缘时增大到 `max_pulse_seconds`；当前 `0.15–1.00 s` 对应名义横移约 `1.2–8.0 cm`
- 横移搜索期间以 `max_vx_correction_mps=0.04 m/s` 主动补偿前后方向漂移；搜索脉冲不再因累计漂移刚超过 `0.15 m` 直接退出。进入红块最终精对准后仍保留 `max_forward_drift_m=0.15 m` 保护
- 单轮累计名义横移上限 `max_strafe_distance_m` 当前为 `0.50 m`；达到上限仍未对齐时停车
- 连续 `no_red_frame_limit=5` 帧未锁定红块时会清除临时轨迹并重连相机，最多重试 `target_not_found_retries=3` 轮
- 到达 `28 cm` 后，三次重连仍未找到时以 `target_search_speed_mps=0.08 m/s`、每步 `1.00 s` 单向持续向左搜索；搜索步之间不额外等待，不反向右扫，也不再使用旧的 `0.50 m` 无目标死区
- 抓取专用检测只保留高色相红色 `HSV H=168–179, S=45–255, V=50–255`，排除现场低色相红色障碍物；该修改不影响开场避障检测。搜索只接受当前帧尺寸仍达到参考值 `70%`、位于下半画面并由严格跟踪器判定稳定的目标；宽松候选和画面边缘疑似目标不能锁轨或结束搜索。候选进入画面水平中央 `1/3` 后停车，重新建立视觉轨迹并精对准 `roi=[0.42,0.55,0.58,0.85]`，连续确认 `3` 帧后才允许抓取
- `target_search_until_found=true` 时，单向左移不再因名义 `1.00 m` 或对准总时长 `90 s` 返回目标未找到；只有取得合格红块后才结束该搜索阶段
- `pregrasp_red_align.loose_motion_*`：宽松红色候选只有满足近场尺寸、纵向位置和连续帧稳定条件后才允许驱动侧移；背景标牌和小反光区域只能记录，不能触发运动
- `box_center_alignment.enabled`：放置区/第二次抓取的箱体中心自动横移总开关；默认关闭，只有两处人工识别都稳定且 ROI 已按现场标注图校准后才能改为 `true`
- `box_center_alignment.placement_roi`：放置区四箱中心的有效范围，使用归一化 `[x0,y0,x1,y1]`；主检测直接使用下半画面的连续纸箱顶边和三条竖直分隔边界，不依赖现场不存在的 A-D 标签，白色标签识别只作为兼容回退
- `box_center_alignment.frames_per_measurement/min_valid_frames/max_center_range_fraction`：每次采集 `7` 帧，至少 `4` 帧有效，任一中心横坐标范围不得超过画面宽度 `3%`
- `box_center_alignment.tolerance_fraction/max_corrections`：目标误差不超过画面宽度 `5%` 即成功，否则最多横移修正 `3` 次
- `box_center_alignment.max_single_strafe_m/max_total_strafe_m`：视觉修正单次不超过 `0.25 m`、累计不超过 `0.75 m`；放置区按相邻箱实际间距 `0.30 m` 动态换算像素比例
- `pickup_transfer.enabled=true` 时，放置区字母视觉失败会停车并中止，不能回退到固定 A/B/C/D 距离；固定偏移只保留给关闭抓取转运模式的旧流程
- `vision.*_hsv`：锥桶、红条、绿条颜色阈值
- `arm.backend`：机械臂后端，默认 `runtime`；如需回退旧示例控制器可改为 `legacy`
- `arm.port`：机械臂串口，Jetson 常见为 `/dev/ttyUSB0`，现场可用 `ARM_PORT=/dev/ttyUSB1` 临时覆盖
- `arm.camera_device`：机械臂抓取相机，当前实机默认 `/dev/video4`，现场可用 `ARM_CAMERA=/dev/video2` 临时覆盖
- `arm.runtime_config`、`arm.calibration`、`arm.grasp_reference`、`arm.place_reference`：机械臂视觉、标定、抓取和放置调参文件
- `audio.remote_gain_db`：狗身远端 WAV 数字增益，默认 `3.0 dB`，允许 `-12` 至 `+6 dB`

`config/field.yaml` 中配置场地、区域和航点。正式场地测试时，应以比赛地图和现场测量值为准。

启动时会严格检查两个配置文件、路线动作和数值范围。当前 `field.yaml` 使用 YAML 注释语法，运行环境必须安装 `PyYAML`；缺文件或解析失败会在发送硬件命令前退出。

## 机械臂 Runtime 调试

机械臂默认使用 `arm.backend=runtime`。`mission_lite3/arm/runtime/` 来自
`../lite3_arm_runtime/`；JSON 参数、shell 脚本和任务状态机保持一致。代码只增加
了预检读取 JSON `image_size` 的格式兼容，项目层另行适配设备路径、结果文件和
任务状态机调用。源包的完整部署说明见 `arm/runtime/README_DEPLOY.md`。

常用调试命令：

```bash
scripts/check_arm_devices.sh
python3 tools/test_arm_standalone.py
scripts/run_arm_task.sh --dry-run grasp
scripts/run_arm_task.sh --dry-run place --slot A --object-held
scripts/run_arm_task.sh preflight
scripts/run_pregrasp_align.sh --max-pulses 1
scripts/run_arm_task.sh grasp
scripts/run_arm_task.sh place --slot A --object-held
scripts/run_arm_task.sh home
```

包装入口支持的任务与源 `arm_task.py` 一致：`grasp`、`grasp-ready`、
`transport`、`close`、`hold`、`place`、`home`、`status`、`abort`、
`preflight`、`diagnose-run` 和 `validate-run`。底层串口调试继续直接使用
`arm/runtime/test.py`。

`run_pregrasp_align.sh` 默认是无硬件输出的仿真。逐步实机验证必须显式加
`--robot`；`--max-pulses 1` 将每次运行限制为一个
`0.08 m/s × 0.25 s` 横移脉冲。入口会先检查新鲜的里程计、IMU 和超声波，
临时切换自动模式，结束时连续停止并恢复手动模式：

```bash
scripts/run_pregrasp_align.sh --robot --max-pulses 1
```

源抓取状态机会按 `grasp_reference_square_face.json` 执行初始红条搜索、
`b/s/e/w` 视觉对准、最终姿态、闭合夹爪和货运姿态。项目层不改写这些步骤。
主任务已提前进入抓取准备姿态时，仅使用源入口已有的
`--skip-grasp-ready`，避免重复下发相同准备动作。

单独测试机械臂时优先使用 `tools/test_arm_standalone.py`。这个脚本不启动机器狗运动、巡检或任务状态机，只调用 `mission_lite3.arm.run_arm_task` 和包内 runtime。默认不碰真实硬件：

```bash
python3 tools/test_arm_standalone.py
python3 tools/test_arm_standalone.py --scenario status
python3 tools/test_arm_standalone.py --scenario transport
python3 tools/test_arm_standalone.py --scenario grasp
python3 tools/test_arm_standalone.py --scenario place --slot A
```

连接真实机械臂时使用 `--real`。`preflight` 只检查设备、配置、相机标定和串口状态；会移动机械臂的场景必须额外加 `--yes`：

```bash
python3 tools/test_arm_standalone.py --scenario preflight --real
python3 tools/test_arm_standalone.py --scenario transport --real --yes
python3 tools/test_arm_standalone.py --scenario grasp --real --yes --show-vision
python3 tools/test_arm_standalone.py --scenario place --real --yes --slot A
python3 tools/test_arm_standalone.py --scenario full --real --yes --slot A --show-vision
```

现场设备号变化时可临时覆盖，不需要改代码：

```bash
ARM_RESULT_DIR=/tmp/arm_results python3 tools/test_arm_standalone.py
python3 tools/test_arm_standalone.py --scenario preflight --real --port /dev/ttyUSB1 --camera /dev/video2
```

现场设备号变化时，不建议改代码。优先修改 `config/robot.yaml`：

```json
"arm": {
  "port": "/dev/ttyUSB0",
  "camera_device": "/dev/video4",
  "camera_width": 1280,
  "camera_height": 720,
  "camera_fps": 25
}
```

临时试错可用环境变量覆盖：

```bash
ARM_PORT=/dev/ttyUSB1 ARM_CAMERA=/dev/video2 scripts/run_arm_task.sh preflight
ARM_WIDTH=1280 ARM_HEIGHT=720 ARM_FPS=25 scripts/run_arm_task.sh --dry-run grasp
```

集中调参文件：

- `arm.runtime_config`：红条识别和抓取阶段视觉参数，默认 `mission_lite3/arm/runtime/strip_detector_grasp_config.json`；当前抓取专用红色范围为高色相 `H=168–179`，不会把低色相红色锥桶作为抓取目标
- `arm.calibration`：机械臂相机标定，默认 `mission_lite3/arm/runtime/camera_calibration.json`
- `arm.grasp_reference`：抓取参考姿态和视觉伺服参考，默认 `mission_lite3/arm/runtime/grasp_reference_square_face.json`
- `arm.moving_pose`：源 `test.py moving-pose` 和 `scripts/run_startup_pose.sh` 使用的移动姿态
- `arm.place_reference`：源 A/B/C/D 放置序列；每个槽位保留各自的关节目标

参数默认值完全来自源代码：抓取任务 `spd/acc` 默认 `10/10`，放置控制器
默认 `40/40`，单关节测试的最大步长为 `20` 度。需要覆盖时使用源入口已有的
`--spd`、`--acc`、`--final-spd` 和 `--final-acc`。

源工具示例：

```bash
# 保存当前姿态为 moving_pose.json
python3 mission_lite3/arm/runtime/test.py teach-moving-pose
# 预览/执行保存的移动姿态
python3 mission_lite3/arm/runtime/test.py moving-pose
python3 mission_lite3/arm/runtime/test.py moving-pose --execute

# 采集抓取参考；参数以 --help 为准
scripts/teach_grasp_pose.sh --help
```

注意：源 `arm_task.py` 的 `transport` 实际复用 `grasp_ready()`，所以任务层
`stow()` 不使用 `transport`。任务层直接复用源 `test.py` 的姿态读取、反馈坐标
转换和运动校验，将 `moving_pose.json` 作为所有非夹持底盘移动时的机械臂姿态。
紧急情况仍应物理切断机械臂电源。

运行结果默认写入 `logs/last_*_result.json`，真实抓取过程日志写入 `grasp_runs/`。这两个目录只作现场诊断，不纳入版本管理。

## 任务流程

主状态机在 `mission.py` 中，顺序如下：

1. `BOOT_CHECK`：检查运动、状态读取、相机、机械臂、播报等基础链路。
2. `STAND_AND_ARM`：默认认为狗已经站立，不重复发送起立命令；切自主模式并收纳机械臂。
3. `PASS_OBSTACLE`：使用 `/dev/video0` 红色锥桶视觉、前向超声波和
   `/leg_odom2` 穿越障碍区；遇到几个锥桶就逐个避开几个，每次绕行后回到原行进线。
   前向投影里程到达障碍区出口后停车并交给原巡检路线，障碍数量可以是 0、1 或多个。
4. `INSPECT_LEFT_OBJECT`：巡检左侧对象，识别字母和仪表盘状态。
5. `INSPECT_RIGHT_OBJECT`：巡检右侧对象，识别字母和仪表盘状态。
6. `REPORT_RESULTS`：只汇总打印本轮已识别结果，不再按 A/B/C/D 排序补播。
7. `PICK_RED_BAR`：机械臂保持移动姿态进入抓取区，先让前向超声波连续 `3` 帧确认
   接近到 `28 cm`，再原地检测并最多重连相机 3 次；仍看不到合格红块时以 `0.08 m/s`
   单向持续向左搜索，横移时主动修正前后漂移，不因名义 `1.00 m` 或 `90 s` 超时退出。
   严格红块中心进入画面水平中央 `1/3` 后停车，重新打开相机、清空旧轨迹，进入窄
   ROI 精对准并连续确认 `3` 帧，最后确认距离 `<30 cm`；抓取成功后
   后退到约 `0.80 m`，使用抓取区纸箱做偏航矫正，
   再以画面宽度 `±3%` 误差对准纸箱中心。
8. `PLACE_TO_LETTER_BOX`：旋转 `180°` 后暂不执行四箱偏航矫正；先以 `0.08 m/s`
   连续前进到前向超声波 `<=0.80 m` 并停车，此前不启用字母相机识别。停车后若未看到
   目标字母，先利用当前可见字母、A/B/C/D 顺序和 `0.50 m` 格距粗定位；完全无字母时
   才以停车点为原点先向左搜索 `1.00 m`，再反向越过原点搜索到右侧 `1.00 m`。目标
   进入中央 `1/3` 并连续确认 `3` 帧后前进 `0.30 m`，前进后重新采图，在中央
   `45%–55%` 再确认 `3` 帧才开始放置。程序记录第一次视觉导航的实测净横移和字母
   位置供第二次投放复用。以源码货运姿态携带红条进入放置区；释放后先单独
   回收肩关节到总回收行程的 50% 并确认到位，再让肩关节与肘关节共同回到
   移动姿态，最后整理底座和腕关节。
9. `SECOND_PICK_PLACE`：返回时使用第一次记录的实测净横移距离回到抓取区，以
   `±6%` 误差对准纸箱中心，再重复抓取、后退、偏航、中心横移和投放。
10. `FINISH_OR_SAFE_STOP`：停止运动、机械臂收纳，等待人工接管。

干跑模式下默认生成 A、C 异常记录，用于快速验证完整流程。

## 比赛路线规划

当前路线按 `26比赛资料/比赛具体位置图.png` 重新规划。地图拓扑如下：

```text
                         y+ / 场地内部
                              ↑

                    A      B      C      D
              ┌──────────────────────────────┐
              │        投放区 / 字母箱         │
              └──────────────────────────────┘

        上巡检箱
        inspect_upper              │
             │                     │
             │                     │
        下巡检箱              物块抓取区 pickup
        inspect_lower              │
             │                     │
             └──── 随机障碍区 ──────┘
                              │
                         出发区 start

                   x- / 地图左      x+ / 地图右
```

实机初始位姿：

- 狗放在右下角 `出发区` 原中心向地图上方 `0.40 m` 的位置（靠近上边缘）。
- 狗头朝向地图左方，也就是从出发区直接指向随机障碍区的方向。
- 5m x 6m 的有效场地只按路线图中上下黑色横线之间计算，不包含页眉、页脚和横线外区域。
- 抓取前的脚本路线中，`forward` 表示地图向左，正 `strafe` 表示地图向下。
- 从上巡检区去物块区时，程序会右转 90 度；右转后 `forward` 表示地图向上，用于从物块区前往投放区。

主路线顺序：

1. 从右下出发区正向前进，也就是朝地图左方进入随机障碍区。
2. 穿越下方随机障碍区，遇到障碍时按配置侧移绕行。
3. 保持面向地图左方，依次停靠 4 个识别图点，每个点停车、稳定识别并现场语音播报。
4. 从第 4 个识别点侧移回物块区方向，再后退到右中物块抓取区。
5. 右转 90 度面向地图上方，抓取异常红条后前进到右上投放区。
6. 根据异常字母 A/B/C/D 做横向对位并投放。
7. 返回物块抓取区，抓取第二个异常红条，再次前往投放区投放。

路线距离集中配置在 `mission_lite3/config/field.yaml` 的 `scripted_route`：

- `pass_obstacle`：出发区到随机障碍区并穿越障碍。
- `inspect_stop_1_arrive`：从障碍区到第 1 个识别图点；当前现场调参为地图向上横移 `0.65 m`，再向地图左方前进 `0.40 m`。
- `inspect_stop_2_arrive`：第 1 个识别图点到第 2 个识别图点。
- `inspect_stop_3_arrive`：第 2 个识别图点到第 3 个识别图点。
- `inspect_stop_4_arrive`：第 3 个识别图点到第 4 个识别图点。
- `pickup_from_upper_inspection`：第 4 个识别图点到物块区。
- `place_from_pickup`：物块区到投放区。
- `pickup_from_place`：投放区返回物块区。
- `placement_letter_strafe_m`：旧工具和关闭抓取转运模式时使用的固定横移偏移；完整任务的
  抓取转运流程不再把它作为视觉失败后的回退。

抓取区位置主要调 `pickup_from_upper_inspection`：

- 第一段 `turn`：从第 4 个识别点转向地图下方。
- 第二段 `forward`：向物块区所在行前进；若连续 3 个新鲜超声样本确认已到达
  `0.35 m` 前方边界，则把本段视为完成并继续，不把正常到达误判为任务失败。
- 第三段 `strafe`：向地图右方进入物块抓取通道。

如果到抓取区时偏上/偏下，优先调第二段 `forward`；如果偏左/偏右，优先调第三段 `strafe`。

正常启用 `startup_avoidance.enabled=true` 时，`pass_obstacle` 的旧固定步进参数
不执行。启动避障参数集中在 `mission_lite3/config/robot.yaml`：

`startup_avoidance` 只在 `PASS_OBSTACLE` 状态运行，即从起点到第一个仪表盘之前；
达到结束条件后立即退出该控制器，后续巡检、抓取和放置阶段不会再次调用侧移绕障状态机。

- 相机：`/dev/video0`、`1280 x 720`、`25 FPS`。
- 触发距离：正前方和侧方均为 `0.40 m`；`0.15 m` 为不可绕过的紧急停车下限。
- 运动速度：巡航 `0.08 m/s`、向左绕行 `0.08 m/s`、越障前进 `0.102 m/s`。
- 目标必须连续确认 5 帧；确认期间保持当前前进阶段的速度，不再为确认主动停车。
- 第一次横移完成时记录原始横向线；总前进进度仍从避障入口累计。目标即使很快离开画面，也必须让全局前进达到 `finish_forward_m=2.40 m` 才开始回线。
- 如果这段 `1.50 m` 前进中检测到第二个障碍，重新确认并再次向左横移，旧的 `pass_progress_m` 清零；从最后一次横移完成点重新前进 `1.50 m`。所有连续绕障完成后只回线一次，目标仍是第一次绕障前的原始横向线。
- 回线容差 `0.02 m`，连续稳定 5 帧；不再用 PASS 时间倍率、总避障超时或回线超时提前结束任务。
- 没有遇到障碍时，仍以避障开始位置和初始朝向为基准，将 `/leg_odom2` 位移投影到前向，达到 `finish_forward_m=2.40 m` 后输出零速度；遇到一个或多个障碍时，绕行动作不会清零全局前进进度，达到同一 `2.40 m` 终点并回到原行进线后输出零速度，随后执行
  `inspect_stop_1_arrive`：横移 `-0.65 m`，前进 `0.40 m` 到第一个巡检点。
- 图像、超声、里程计最大允许数据年龄分别为 `0.30 s`、`0.20 s`、`0.20 s`，
  数据过期、跟踪歧义、未知近障、`0.15 m` 紧急距离或相机故障会进入 `HOLD`，
  持续输出零速度；数据连续稳定 5 帧后从原避障状态恢复，已保存的 PASS 起点、
  `pass_progress_m`、原始回线和避障次数不会清零。相机或运动接口异常时同一进程会持续
  重连，不会因此退出整项任务。
- 逐帧状态、速度、传感器年龄、跟踪 ID 和回线误差写入
  `startup_avoidance_runs/*.jsonl`；日志同时记录前向投影里程
  `forward_progress_m`、当前 PASS 独立里程 `pass_progress_m` 和实际避障次数。

避障发生故障时不会静默退回固定横移路线。只有显式关闭
`startup_avoidance.enabled`，或使用 `--ignore-obstacles` /
`--ignore-ultrasound-obstacle` 调试参数时，才保留旧 `pass_obstacle` 路线。
第一次实机测试必须先核对 `/dev/video0` 的实际画面和左右方向，并使用
`--skip-arm` 分段观察；静态测试不能证明真实运动已安全通过。

### 主任务故障保持与恢复

实机配置 `fault_hold.enabled=true`。任务状态抛出异常后进入 `FAULT_HOLD`，持续尝试
发送零速度并检查里程计、IMU、超声波、姿态、电量和机器人健康状态。单次故障等待上限
为 `max_wait_s=30 s`，每个任务状态最多重试 `max_retries_per_state=2` 次；超时或重试
用尽后执行清理并以失败码退出，不再无限保持进程。

- `BOOT_CHECK`、`STAND_AND_ARM`、`PASS_OBSTACLE`、`REPORT_RESULTS` 和
  `FINISH_OR_SAFE_STOP`：安全状态连续稳定 5 次后自动重试当前状态。
- 两段巡检、首次抓取、首次放置和第二轮抓放：为避免重复移动、重复夹取或重复释放，
  需要操作员确认机器人和夹爪实际状态后执行 `touch /tmp/lite3_fault_resume`。信号即使在
  数据尚未恢复时发出也会被锁存，待安全状态连续稳定 5 次后重试当前状态；30 秒内没有
  确认信号则退出。
- 第二轮抓放一次失败只需要一次新的恢复信号，不会重复消费两次确认；最终已完成至少
  两次放置后才进入收纳状态。

开场避障的连续 `HOLD`、相机重连或运动接口恢复同样受
`startup_avoidance.fault_hold_max_s=30 s` 限制。`Ctrl+C` 会进入统一清理；系统直接
终止、主机掉电或操作系统故障仍可能跳过部分软件清理。

巡检停车时间配置在 `mission_lite3/config/robot.yaml`：

```json
"inspection": {
  "reset_round_result_on_mission_start": true,
  "stop_dwell_seconds": 8.0,
  "speak_at_inspection_stop": true,
  "evidence_dir": "evidence",
  "place_pause_seconds": 3.0
}
```

到达每个识别图点后程序会先发送停止速度，最多在 `stop_dwell_seconds` 时间内等待识别，默认 8 秒。字母定位、字母置信度和指针支撑均可靠时，单帧即可结束；其余结果继续走多帧投票。超时时优先采用等待期间质量最高的真实识别结果，仍无有效结果才使用预设值。巡检使用完整 `1280×720` 广角画面并在识别前去畸变。正式国赛流程中，停车点只代表第几个拍摄位置，A/B/C/D 必须从画面上方字母识别出来，仪表盘状态从同一帧识别出来。每个识别点识别一次、播报一次，共 4 次；播报字母按实际识别结果，不按 A/B/C/D 固定顺序。

每个停车点会保存证据图到 `evidence/`。真实模式下，相机有画面但在时限内没有稳定结果时，会先使用符合字母锚点和仪表有效性条件的最佳真实候选；没有有效候选时才使用该停车点的预设值。若该字母已被真实识别或先前回退占用，则改用尚未记录的 A/B/C/D 字母，避免覆盖已有结果。后续真实识别与先前回退字母冲突时，真实结果优先，旧回退会迁移到尚未记录的字母。回退记录会标记 `source_camera=default_fallback`、`confidence=0` 和 `stability_votes.default_fallback=1`。相机完全无帧时仍以 `camera_failed` 安全终止。

语音默认通过 UDP 下发到运动主机 `192.168.1.120:43910`，由运动主机本地播放持久目录 `/opt/robot_competition/inspection_audio_test/` 中的预录 WAV。狗身声卡设备为 `plughw:CARD=rockchipes8388c,DEV=0`，默认数字增益为 `+3 dB`。正常日志会出现 `[audio] remote_audio_ok: A_low.wav`；如果失败，先确认 `lite3-remote-audio.service` 已升级并启动，且 12 条 WAV 已同步。

机械臂投放后会等待 `place_pause_seconds`，给放物品和机械臂稳定留时间。

## 控制链路

运动控制采用三层降级：

1. ROS2 `/cmd_vel`：首选方式，速度指令频率按 `motion.command_hz` 发送。
2. UDP 复杂速度指令：使用 `0x0140/0x0145/0x0141` 控制前后、横移、转向。
3. UDP 轴指令：使用 `0x21010130/31/35`，按 20Hz 连续发送，兼容资料中的 `move_lite3.py`。

UDP 心跳命令为 `0x21040001`，发送频率不低于 2Hz。轴指令超时约 250ms，因此不能只发一次速度命令。

## 测试

编译检查：

```bash
python -m compileall mission_lite3 tests
```

烟测：

```bash
python -m unittest discover -s tests
```

完整干跑：

```bash
python -m mission_lite3.run_mission --dry-run --skip-arm
```

通过标准：

- 配置可以正常加载
- UDP 简单命令和速度命令打包格式正确
- 干跑流程能完成巡检、播报、两次异常投放

## 上机前检查

正式运行前建议按顺序确认：

1. Lite3 电量不低于 75%。
2. 机器人周围 2 米内无人员和障碍物。
3. 感知主机与运动主机网络连通。
4. `transfer_ros2` 已启动，`/cmd_vel`、`/leg_odom2`、`/imu/data`、超声波 topic 可用。
5. 前向相机 RTSP 可以打开。
6. 机械臂串口、抓取相机、抓取参考和固定放置序列已通过 `scripts/check_arm_devices.sh` 与 `scripts/run_arm_task.sh preflight` 检查。
7. HSV 阈值已按现场光照重新调过。
8. 先运行 `--dry-run`，再运行 `scripts/run_full_mission.sh --skip-arm`，最后再启用机械臂。

## 实机测试流程

实机测试按“软件自检、干跑、相机巡检、运动、机械臂、全流程”的顺序推进。不要一开始直接运行完整比赛流程。

### 1. 进入工程并确认依赖

```bash
cd /home/ysc/my_sw/robot_competition
pip install -r mission_lite3/requirements.txt
```

检查关键配置：

```bash
sed -n '1,180p' mission_lite3/config/robot.yaml
```

重点确认：

- `camera.front`：前向相机 RTSP 或设备号是否正确。
- `network.motion_ip`、`network.motion_port`：Lite3 运动主机地址是否正确。
- `ros2.cmd_vel_topic`、`ros2.odom_topic`、`ros2.imu_topic`、`ros2.ultrasound_topic`：是否和实机 ROS2 topic 一致。
- `arm.port`、`arm.camera_device`：机械臂串口和抓取相机是否为实际设备。
- `arm.runtime_config`、`arm.calibration`、`arm.grasp_reference`、`arm.place_reference`：调参文件是否指向本场实机版本。

### 2. 软件自检

```bash
bash check_robot_runtime.sh
```

测试入口统一使用 `unittest`，可发现包括巡检绑定在内的全部用例，不受 ROS Foxy pytest 插件版本影响。

通过标准：

- Python 编译检查通过。
- 单元测试全部通过。
- `run_live_inspection.py` 命令行入口可正常加载。

### 2.1 音频出声探测

先只测声音，不让机器狗运动。默认 `run_speech_probe.py` 走运动主机 UDP 播放；先安装开机自启服务：

```bash
scripts/install_remote_audio_service.sh
scripts/check_remote_audio_service.sh
```

然后在感知主机运行：

```bash
python3 run_speech_probe.py
python3 run_speech_probe.py --remote-gain-db 3
```

可用 `--remote-gain-db 0` 对比原始音量。任务端要求服务响应确认实际增益，因此更新代码后必须重新安装远端服务。

听到狗身扬声器播报后，再继续跑巡检或全流程。需要单独测试感知主机 TTS 时使用 `python3 run_speech_probe.py --tts-only`。

如果需要排查感知主机本机音频链路，再加 `--local-audio`：

```bash
python3 run_speech_probe.py --local-audio
python3 run_speech_probe.py --local-audio --diagnose --skip-beep --skip-speech
python3 run_speech_probe.py --local-audio --test-alsa-devices
```

若某个 ALSA 设备能响，终端会打印 `working_alsa_device=...`。这说明扬声器本身可用，下一步再把 PulseAudio 的 sink、card profile 或 port 切到对应物理输出。

### 3. 主任务干跑

干跑不会发送真实运动指令，也不会控制真实机械臂：

```bash
python3 -m mission_lite3.run_mission --dry-run --skip-arm
```

通过标准是最后能看到类似输出：

```text
[mission] placed anomaly bars: ['A', 'C']
```

这表示状态机、默认巡检结果、播报调用、两次模拟抓取投放流程已经跑通。

### 4. 相机和巡检识别测试

先测试箱体中心识别。该入口只创建前向相机和去畸变器，不创建运动控制器，也不发送机器狗命令：

```bash
scripts/test_box_centers.sh
scripts/test_box_centers.sh --scene placement --target-letter D
scripts/test_box_centers.sh --scene pickup
```

脚本依次提示人工把机器狗移到放置区和抓取区，每处采集 7 帧。结果写入
`box_recognition_runs/<timestamp>/result.json`，并保存每帧原图和标注图；通过时
`motion_command_count` 必须为 `0`，两处均至少 4 帧有效且中心波动不超过画面宽度
的 3%。先检查标注图没有把画面上方标牌当作箱子，再按现场画面调整
`box_center_alignment.placement_roi`。

在两处结果均为 `ok=true` 之前，保持
`config/robot.yaml` 的 `box_center_alignment.enabled=false`。校准完成后才能改为
`true` 接入自动横移；第一次实机启用仍应使用空载、低速和随时可急停的受控测试。

识别通过后，用独立单步入口验证横移方向。默认 dry-run；实机命令固定为最多
一次 `0.05 m` 修正，复测后自动回撤，并再次采集 7 帧验证视觉位置和里程计均返回：

```bash
scripts/test_box_center_step.sh
scripts/test_box_center_step.sh --robot --yes --target-letter D
scripts/test_box_center_step.sh --robot --yes --target-letter A
```

单步方向和回撤均通过后，再运行空载完整对中。完整模式恢复正式任务的
`3 次 / 单次 0.25 m / 累计 0.75 m` 上限；视觉进入 ±5% 后会主动返回公共基准，
并执行第三次 7 帧复测：

```bash
scripts/test_box_center_step.sh --profile full --target-letter D
scripts/test_box_center_step.sh --profile full --robot --yes --target-letter D
```

无图形界面或 SSH 环境使用：

```bash
python3 run_live_inspection.py \
  --no-window \
  --json-terminal \
  --reset-round \
  --no-speak \
  --max-frames 120
```

如果需要持续识别直到稳定结果：

```bash
python3 run_live_inspection.py \
  --no-window \
  --json-terminal \
  --reset-round \
  --no-speak
```

如果当前终端有图形界面，需要弹窗实时显示 1.5 倍放大后的相机画面和识别结果，去掉 `--no-window`，不要加 `--exit-after-stable`，按 `q` 退出：

```bash
python3 run_live_inspection.py \
  --json-terminal \
  --reset-round \
  --no-speak
```

检查输出文件：

```bash
cat latest_stop_result.json
cat round_result.json
```

`round_result.json` 的关键字段：

- `ready`：四站点结果是否可供抓取投放阶段使用。
- `abnormal_areas`：异常区域列表。
- `unknown_areas`：仍未稳定识别的区域。
- `count_check`：是否满足 2 正常、2 异常。
- `block_reason`：`ready=false` 时的阻塞原因。

四站点巡检完成后的理想状态：

```json
{
  "ready": true,
  "abnormal_areas": ["A", "C"],
  "unknown_areas": [],
  "count_check": {
    "normal": 2,
    "abnormal": 2,
    "passed": true
  }
}
```

如果巡检脚本读不到稳定结果，先单独确认前向相机能读帧：

```bash
python3 -m mission_lite3.run_mission --vision-test --headless --vision-frames 60
```

### 5. 低风险实机运动测试

场地清空，人工准备急停。第一次实机运动必须禁用机械臂：

```bash
scripts/run_full_mission.sh --skip-arm
```

如果 ROS2 `/cmd_vel` 控制不稳定，改用 UDP fallback：

```bash
scripts/run_full_mission.sh --skip-arm --udp-fallback
```

如果 UDP 复杂速度指令仍不稳定，使用最保守的轴指令 fallback：

```bash
scripts/run_full_mission.sh --skip-arm --udp-fallback --axis-fallback
```

如果机器人在空旷场地仍反复打印 `obstacle close`，先看括号中的触发原因。测试路线时可以临时忽略超声波障碍触发：

```bash
scripts/run_full_mission.sh --skip-arm --ignore-ultrasound-obstacle
```

如果需要只验证运动路线、完全绕过障碍判断：

```bash
scripts/run_full_mission.sh --skip-arm --ignore-obstacles
```

如果需要同时弹窗看狗看到的画面和识别结果：

```bash
scripts/run_full_mission.sh --skip-arm --ignore-obstacles --inspection-window
```

通过标准：

- 机器人能按状态机前进、停止。
- 人工急停可随时介入。
- 无异常姿态、无持续打滑、无不可控速度。
- 终端没有连续相机、ROS2 或安全检查错误。

### 6. 巡检结果 gate 检查

主任务在启动时会重置本轮 `round_result.json`，巡检过程中逐站写入实际识别到的 A/B/C/D 和仪表盘状态。抓取前优先使用本轮内存中的巡检结果；只有满足下面条件才进入抓取投放：

```json
{
  "ready": true,
  "unknown_areas": []
}
```

如果结果不可用，主任务会跳过抓取并打印阻塞原因，例如：

```text
[mission] round_result blocks pickup: block_reason=unknown_area unknown_areas=['B']
```

正式启用机械臂前建议先确认：

```bash
cat round_result.json
```

### 7. 机械臂和完整流程

先单独检查机械臂串口、相机和预检：

```bash
scripts/check_arm_devices.sh
scripts/run_arm_task.sh --dry-run grasp
scripts/run_arm_task.sh preflight
```

如现场设备号变化，可临时覆盖：

```bash
ARM_PORT=/dev/ttyUSB1 ARM_CAMERA=/dev/video2 scripts/run_arm_task.sh preflight
```

先运行抓取、运输和投放专项入口的干跑。该入口默认测试 A、D 两个区域，不经过巡检路线：

```bash
scripts/run_pickup_transfer_mission.sh --targets A D
```

实机专项测试前必须满足：机器人由人工放在抓取区纸箱中心线上，面向纸箱，前向
超声波约为 `0.80 m`；夹爪为空且机械臂可正常进入移动姿态；A、D 投放通道无人员
和障碍物；一名操作人员全程持有急停。确认后执行：

```bash
scripts/run_pickup_transfer_mission.sh --robot --yes --targets A D
```

专项脚本在`--robot`模式下也持有`/tmp/lite3_motion_test.lock`，与完整任务互斥；
锁路径和默认`5 s`等待时间使用相同的`LITE3_MOTION_LOCK`、
`LITE3_MOTION_LOCK_WAIT_S`环境变量。

该命令会真实驱动机器狗和机械臂。每一轮的预期顺序为：先前移到 `28 cm`、持续向左
搜索并获取红块、红块最终横移精对准、抓取、后退到约 `0.80 m`、抓取区纸箱偏航、纸箱中心横移、旋转 `180°`、
视觉搜索并对准目标字母、前往放置区并释放。第二轮先按第一轮记录的实测净横移距离
返回抓取区，然后重新运行一次字母视觉导航。

关键正常日志包括：

```text
[pickup-transfer] retreat ok=True reason=target_reached ...
[pickup-transfer] retreat ok=True reason=target_reached_odom_fallback ...
[pregrasp] wide-camera-only alignment complete
[box-center] pickup stage=pickup_departure ok=True ... tolerance=0.030 ...
[placement-yaw] disabled
[pickup-transfer] restore pickup lane physical_recorded=... command=... ok=True ...
```

两种 `retreat` 成功原因二选一即可：前向超声波正常增长时为 `target_reached`；传感器
持续卡在 `0.28 m`、腿部里程计达到 `0.44 m` 时为
`target_reached_odom_fallback`。偏航识别不到纸箱会打印 warning 并直接继续中心横移；
后退达到 `0.55 m`、中心横移失败、里程计/IMU 陈旧或姿态越限仍会安全中止。
专项结果保存在 `pickup_transfer_runs/<timestamp>/result.json`。

确认相机、巡检、运动都单独通过后，再启用机械臂完整流程：

```bash
scripts/run_full_mission.sh
```

脚本会自动切换到项目根目录、加载 `/opt/ros/foxy/setup.bash`，然后以
`python3 -u -m mission_lite3.run_mission --robot` 启动任务。完整进程会一直持有
`/tmp/lite3_motion_test.lock`，默认等待锁 `5 s`；可分别通过 `LITE3_MOTION_LOCK`
和 `LITE3_MOTION_LOCK_WAIT_S` 覆盖。需要使用其他 ROS2 环境时可通过 `ROS_SETUP`
指定其 `setup.bash`。等待超时会打印 `[full-mission] motion lock timeout` 并以状态码
`3` 退出，此时 Python 任务尚未启动，也不会发送运动指令。

### 放置区字母视觉导航

- 字母牌必须是白纸上的黑色大写 A、B、C、D，现场从左到右依次排列 A、B、C、D。
- 抓取成功后先后退到抓取箱前约 `0.80 m`，完成抓取区纸箱偏航和中心修正，再旋转
  `180°` 面向投放区。
- 旋转后机器狗以 `0.08 m/s` 连续前进。只有经过滤波的前向超声波达到 `<=0.80 m`
  并明确停车后才采集第一帧字母画面；远处提前出现的字母不会跳过该接近阶段。
- 停车后的搜索原点记为 `0 m`。没有识别到目标字母时，先向物理左侧分步搜索到
  `+1.00 m`；仍未找到则反向横移，越过原点继续到物理右侧 `-1.00 m`。
- 看到任意可靠的非目标字母时，按 A/B/C/D 顺序和 `letter_spacing_m=0.50 m` 计算目标
  方向，单次粗横移最多 `max_anchor_jump_m=0.50 m`；只有完全无字母时才盲扫。
- 左侧搜索最多走 `1.00 m`，完整右扫从左端到右端最多再走 `2.00 m`，所以全程未找到
  字母时的实际横移累计为 `3.00 m`，最终净横移为原点右侧 `1.00 m`。
- 目标字母置信度必须达到 `0.60`。搜索阶段中心进入画面中央 `1/3` 并连续确认 `3` 帧
  后，执行独立的 `0.30 m` 前进；前进里程必须由里程计确认，不能只按时间估算。
- 前进后重新采集新画面，目标中心位于 `45%–55%` 且再次连续确认 `3` 帧才进入机械臂
  放置；前进造成横向漂移时先小幅横移修正。
- 第一次成功投放缓存目标字母的实测净横移。第二次按固定 `0.50 m` 格距预测目标横向
  位置并先做粗定位，最终仍必须通过相机确认；预测超出左右 `1.00 m` 边界时弃用。
- 横向累计硬上限为 `3.10 m`，单轮视觉导航总时限为 `90 s`。固定
  `lane_offsets_m` 和 `placement_letter_strafe_m` 只保留给旧工具或关闭抓取转运模式的
  兼容流程，不是完整任务的视觉失败回退。
- 原图、去畸变图、标注图和 JSONL 过程记录保存在
  `placement_letter_navigation_runs/`。相机、超声波、识别、运动或清理任一环节失败时，
  程序停车并中止本次放置，不会继续调用机械臂。
- 当前代码尚未在机器狗上验证 `physical_left_strafe_sign=1` 的真实横移方向，也没有用
  现场 A、B、C、D 相机样本完成全组合测试。启用机械臂前必须先做不超过 `0.10 m` 的
  左右方向验证，再依次验证四个目标。

ROS2 不稳定时：

```bash
scripts/run_full_mission.sh --udp-fallback
```

最保守模式：

```bash
scripts/run_full_mission.sh --udp-fallback --axis-fallback
```

完整流程通过标准：

- 完成巡检播报。
- `round_result.json` 满足 `ready=true` 且 `unknown_areas=[]`。
- 两个异常区域被依次选为抓取投放目标。
- 机械臂抓取和放置动作无卡滞。
- 最后进入 `FINISH_OR_SAFE_STOP`，机器人停止，机械臂收纳。

### 8. 常见测试结论

- 干跑通过但相机巡检失败：优先检查 `camera.front`、RTSP 网络、光照和识别阈值。
- 日志出现 `use default area=...`：该站点未形成稳定识别，任务已使用预设值继续；赛后应检查对应证据图并调整仪表盘、字母识别参数。
- `ready=false` 且 `block_reason=count_check_fail`：四站点已识别，但数量不是 2 正常、2 异常，先不要进入抓取。
- `block_reason=camera_failed`：相机连续读帧失败，检查相机电源、网络、RTSP 地址。
- 运动链路不稳定：按 ROS2、UDP、UDP 轴指令的顺序降级测试。
- 日志出现 `sample is stale`：对应 ROS2 话题没有在 `state_max_age_s` 内更新，任务会停止而不是继续开环运动。

## 故障排查

ROS2 不可用：

```bash
scripts/run_full_mission.sh --udp-fallback
```

UDP 速度不稳定：

```bash
scripts/run_full_mission.sh --udp-fallback --axis-fallback
```

相机打不开：

- 检查 `config/robot.yaml` 中 `camera.front`
- 抓取相机检查 `config/robot.yaml` 中 `arm.camera_device`，当前实机应为 `/dev/video4`
- 用 `--vision-test --headless` 看是否能读帧
- 确认 RTSP 地址和网络在 Jetson 上可访问

机械臂不可用：

- 运行 `scripts/check_arm_devices.sh`，确认 `arm.port` 或 `ARM_PORT` 是否为实际串口
- 运行 `scripts/run_arm_task.sh --dry-run grasp` 和 `scripts/run_arm_task.sh preflight` 分别检查配置与硬件
- 可先加 `--skip-arm` 跑通导航和识别
- 如必须使用旧样例控制器，将 `arm.backend` 改为 `legacy` 后再检查 `26比赛资料/DeepRobotDog-main/utils/ArmController.py`

识别不稳定：

- 使用 `vision_debug.py` 重新调 HSV
- 采集现场图片后用 `--vision-test image.png` 离线验证
- 仪表盘和字母识别仍需基于现场素材继续标定

## 安全说明

- 默认不加 `--robot` 时，程序不会发送真实运动指令。
- 真实任务需要新鲜的里程计、IMU 和启用后的超声波数据；动作期间任一数据过期都会发送零速度并进入 `ABORT_SAFE`。
- 投放失败时保持“仍持物”状态并等待人工接管，不再自动开始第二次抓取或收臂。
- 上机运行前必须确认场地安全，并安排人工急停。
- 本工程不会使用激光雷达、激光 SLAM、雷达导航或智能避障。
- 任何姿态异常、超声波距离过近、机械臂卡滞等情况，应立即停止流程并人工接管。


### robot_runtime 功能移植说明

- 巡检运行时统一放在 `mission_lite3.inspection_runtime`，不再依赖根目录 `inspection` 包。
- `run_live_inspection.py` 作为单站巡检入口，支持 `--list-cameras`、`--camera-name` 和 `--debug-frame-dir`。
- `run_speech_probe.py` 复用 `mission_lite3.audio.AudioReporter`，并可发送 Lite3 扬声器开关命令；使用 `--skip-lite3-speaker` 可跳过。
- 仪表盘摄像头调试入口为 `python3 -m mission_lite3.meter_recognition.scripts.camera_meter_recognition --help`。
