# Lite3机械臂部署文件清单

本目录只整理机械臂相关代码和配置，不包含已单独部署的仪表盘识别代码。

## 主入口

- `arm_task.py`：比赛任务入口，封装抓取、抓取准备、运输姿态、夹爪闭合、保持、放置、回零、预检、状态查询和日志诊断。
- `test.py`：TLS机械臂串口控制工具，负责状态查询、单关节微动、末端小位移、夹爪、原始JSON指令和回零。
- `strip_detector.py`：红绿长条实时识别调试入口。

## 机械臂任务模块

- `arm_grasp.py`：红色长条视觉伺服抓取状态机，输出 `target_left`、`target_right`、`target_too_far`、`target_too_near`、`target_in_grasp_window`、`target_lost`、`arm_control_failed` 等机器狗协同反馈。
- `place_controller.py`：A/B/C/D放置动作接口。
- `final_grasp_matcher.py`：夹爪闭合前最终视图匹配。
- `strip_detection.py`：红绿长条颜色、几何、角度、尺寸和轨迹稳定识别。
- `camera_calibration.py`：相机标定参数读取和去畸变。

## 示教与调参工具

- `capture_grasp_samples.py`：采集抓取参考图像。
- `save_grasp_reference_image.py`：保存可抓取参考图。
- `teach_grasp_pose.py`：示教抓取姿态与参考特征。
- `build_final_view_references.py`：生成最终抓取视图参考。

## 配置文件

- `lite3_arm.env.example`：机器狗端环境变量模板。
- `requirements.txt`：Python依赖，主要为OpenCV、NumPy和PySerial。
- `strip_detector_grasp_config.json`：抓取阶段使用的红条识别配置。
- `strip_detector_config.json`：通用红绿长条识别配置。
- `camera_calibration.json`：当前机械臂相机标定参数。
- `grasp_reference_square_face.json`：夹爪闭合前的可抓取参考。
- `grasp_final_view_references.json`：最终视图匹配参考。
- `place_reference.json`：A/B/C/D放置动作参考。
- `moving_pose.json`：机器狗移动时机械臂安全姿态参考。
- `ost.yaml`：ROS2标定程序导出的相机参数备份。

## 运行脚本

- `scripts/common.sh`：读取 `lite3_arm.env` 并设置默认设备、相机、配置和日志目录。
- `scripts/install_ubuntu_dependencies.sh`：安装Ubuntu依赖。
- `scripts/check_devices.sh`：检查串口、相机和当前机械臂状态。
- `scripts/run_vision_debug.sh`：只运行红条视觉识别，不控制机械臂。
- `scripts/run_preflight.sh`：执行抓取前预检，并写入 `logs/last_preflight_result.json`。
- `scripts/run_grasp.sh`：执行红条识别抓取，并写入 `logs/last_grasp_result.json` 和 `grasp_runs/`。
- `scripts/run_place.sh`：执行A/B/C/D放置，并写入对应 `logs/last_place_*.json`。
- `scripts/run_home.sh`：机械臂回零。

## 对外调用顺序

机器狗导航程序建议按以下顺序调用机械臂包：

```bash
bash scripts/check_devices.sh
bash scripts/run_preflight.sh
bash scripts/run_grasp.sh
bash scripts/run_place.sh A
bash scripts/run_home.sh
```

其中 `run_place.sh` 的参数由巡检任务识别到的异常区域字母决定。
