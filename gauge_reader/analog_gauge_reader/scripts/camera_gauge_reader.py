import argparse
import tempfile
import time
import os
import cv2
import subprocess
import sys
import numpy as np

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)


def classify_status(value, vmin, vmax, low_threshold, high_threshold):
    if low_threshold is not None and high_threshold is not None:
        if value < low_threshold:
            return "偏低（异常）"
        if value > high_threshold:
            return "偏高（异常）"
        return "正常"
    if vmin is None or vmax is None or vmax <= vmin:
        return None
    r = (value - vmin) / (vmax - vmin)
    if r < 0.3:
        return "偏低（异常）"
    if r > 0.7:
        return "偏高（异常）"
    return "正常"


def speak(text):
    try:
        subprocess.Popen(["espeak-ng", "-v", "zh", text],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except Exception:
        pass


def try_open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def default_jetson_gst_pipeline(width, height, fps):
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1, format=NV12 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )


def try_open_gst_camera(pipeline):
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap.release()
        return None
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


class RealSenseCapture:
    def __init__(self, width, height, fps):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8,
                                  fps)
        self.pipeline.start(self.config)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return False, None
        import numpy as np
        color_image = np.asanyarray(color_frame.get_data())
        return True, color_image

    def release(self):
        self.pipeline.stop()


def find_available_camera(max_index):
    for cam_index in range(max_index + 1):
        cap = try_open_camera(cam_index)
        if cap is not None:
            return cam_index, cap
    return None, None


def _resolve_model_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PARENT_DIR, path)


def _is_lfs_pointer_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        return first_line == "version https://git-lfs.github.com/spec/v1"
    except Exception:
        return False


def crop_image(img, box):
    cropped_img = img[box[1]:box[3], box[0]:box[2], :]
    height = int(box[3] - box[1])
    width = int(box[2] - box[0])
    if height > width:
        delta = height - width
        left, right = delta // 2, delta - (delta // 2)
        top = bottom = 0
    else:
        delta = width - height
        top, bottom = delta // 2, delta - (delta // 2)
        left = right = 0
    return cv2.copyMakeBorder(cropped_img,
                              top,
                              bottom,
                              left,
                              right,
                              cv2.BORDER_CONSTANT,
                              value=[0, 0, 0])


def fast_process_image(image, detection_model_path, key_point_inferencer,
                       segmentation_model_path, range_min, range_max,
                       reverse_scale):
    from gauge_detection.detection_inference import detection_gauge_face
    from geometry.ellipse import fit_ellipse, cart_to_pol, get_polar_angle, \
        get_line_ellipse_point
    from key_point_detection.key_point_inference import detect_key_points
    from segmentation.segmenation_inference import segment_gauge_needle, \
        get_fitted_line, get_start_end_line, cut_off_line
    from angle_reading_fit.angle_converter import AngleConverter

    resolution = (448, 448)
    box, _ = detection_gauge_face(image, detection_model_path)
    cropped_img = crop_image(image, box)
    cropped_resized_img = cv2.resize(cropped_img,
                                     dsize=resolution,
                                     interpolation=cv2.INTER_CUBIC)
    heatmaps = key_point_inferencer.predict_heatmaps(cropped_resized_img)
    key_point_list = detect_key_points(heatmaps)
    key_points = key_point_list[1]
    start_point = key_point_list[0]
    end_point = key_point_list[2]
    coeffs = fit_ellipse(key_points[:, 0], key_points[:, 1])
    ellipse_params = cart_to_pol(coeffs)
    if start_point.shape != (1, 2) or end_point.shape != (1, 2):
        raise Exception("Key points incomplete")
    needle_mask_x, needle_mask_y = segment_gauge_needle(cropped_resized_img,
                                                        segmentation_model_path)
    needle_line_coeffs, _ = get_fitted_line(needle_mask_x, needle_mask_y)
    needle_line_start_x, needle_line_end_x = get_start_end_line(needle_mask_x)
    needle_line_start_y, needle_line_end_y = get_start_end_line(needle_mask_y)
    needle_line_start_x, needle_line_end_x = cut_off_line(
        [needle_line_start_x, needle_line_end_x], needle_line_start_y,
        needle_line_end_y, needle_line_coeffs)
    point_needle_ellipse = get_line_ellipse_point(
        needle_line_coeffs, (needle_line_start_x, needle_line_end_x),
        ellipse_params)
    if point_needle_ellipse.shape[0] == 0:
        raise Exception("Needle and ellipse do not intersect")
    theta_start = get_polar_angle(start_point.flatten(), ellipse_params)
    theta_end = get_polar_angle(end_point.flatten(), ellipse_params)
    theta_needle = get_polar_angle(point_needle_ellipse, ellipse_params)
    converter = AngleConverter(theta_start)
    end_conv = converter.convert_angle(theta_end)
    needle_conv = converter.convert_angle(theta_needle)
    if end_conv <= 1e-8:
        raise Exception("Invalid scale endpoints")
    ratio = needle_conv / end_conv
    ratio = max(0.0, min(1.0, ratio))
    if reverse_scale:
        ratio = 1.0 - ratio
    value = range_min + ratio * (range_max - range_min)
    return {"value": float(value), "unit": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--detection_model",
                        type=str,
                        default="models/gauge_detection_model.pt")
    parser.add_argument("--key_point_model",
                        type=str,
                        default="models/key_point_model.pt")
    parser.add_argument("--segmentation_model",
                        type=str,
                        default="models/segmentation_model.pt")
    parser.add_argument("--range_min", type=float, default=0.0)
    parser.add_argument("--range_max", type=float, default=1.0)
    parser.add_argument("--low_threshold", type=float, default=0.3)
    parser.add_argument("--high_threshold", type=float, default=0.7)
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--auto_cam", action="store_true")
    parser.add_argument("--cam_scan_max", type=int, default=9)
    parser.add_argument("--use_jetson_csi", action="store_true")
    parser.add_argument("--gst_pipeline", type=str, default=None)
    parser.add_argument("--use_realsense", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no_gui", action="store_true")
    parser.add_argument("--mode",
                        type=str,
                        default="fast",
                        choices=["fast", "full"])
    parser.add_argument("--reverse_scale", action="store_true")
    args = parser.parse_args()
    use_gui = (not args.no_gui) and bool(os.environ.get("DISPLAY"))
    if not use_gui:
        print("无图形界面模式运行（不调用 cv2.imshow）")

    detection_model_path = _resolve_model_path(args.detection_model)
    key_point_model_path = _resolve_model_path(args.key_point_model)
    segmentation_model_path = _resolve_model_path(args.segmentation_model)
    missing_or_lfs = []
    for model_path in [detection_model_path, key_point_model_path,
                       segmentation_model_path]:
        if not os.path.exists(model_path):
            missing_or_lfs.append(f"{model_path}（文件不存在）")
        elif _is_lfs_pointer_file(model_path):
            missing_or_lfs.append(f"{model_path}（Git LFS 指针文件）")
    if missing_or_lfs:
        msg = "模型文件不可用：\n" + "\n".join(missing_or_lfs) + \
              "\n请安装 git-lfs 后执行：git lfs pull"
        raise SystemExit(msg)
    try:
        import torch.distributed as torch_dist
        if not hasattr(torch_dist, "ReduceOp"):
            class _ReduceOp:
                SUM = "sum"
                PRODUCT = "product"
                MIN = "min"
                MAX = "max"
                BAND = "band"
                BOR = "bor"
                BXOR = "bxor"
            torch_dist.ReduceOp = _ReduceOp
    except Exception:
        pass
    if args.mode == "full":
        from pipeline import process_image
        key_point_inferencer = None
    else:
        from key_point_detection.key_point_inference import KeyPointInference
        key_point_inferencer = KeyPointInference(key_point_model_path)

    selected_cam = args.cam
    cap = None
    if args.use_realsense:
        try:
            cap = RealSenseCapture(args.width, args.height, args.fps)
        except Exception as e:
            raise SystemExit(f"RealSense 打开失败：{e}")
        print("使用 RealSense 摄像头")
    elif args.gst_pipeline is not None:
        cap = try_open_gst_camera(args.gst_pipeline)
        if cap is None:
            raise SystemExit("GStreamer 管道打开失败")
        print("使用 GStreamer 管道摄像头")
    elif args.use_jetson_csi:
        jetson_pipeline = default_jetson_gst_pipeline(args.width, args.height,
                                                      args.fps)
        cap = try_open_gst_camera(jetson_pipeline)
        if cap is None:
            raise SystemExit("Jetson CSI 摄像头打开失败")
        print("使用 Jetson CSI 摄像头")
    elif args.auto_cam:
        selected_cam, cap = find_available_camera(args.cam_scan_max)
        if cap is None:
            raise SystemExit("未找到可用摄像头")
        print(f"自动选择摄像头：{selected_cam}")
    else:
        cap = try_open_camera(args.cam)
        if cap is None:
            selected_cam, cap = find_available_camera(args.cam_scan_max)
            if cap is None:
                raise SystemExit("无法打开摄像头")
            print(f"指定摄像头 {args.cam} 不可用，自动切换到：{selected_cam}")

    last_time = 0.0
    reading_text = ""
    state_text = ""

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            show = frame.copy()
            h, w = show.shape[:2]
            now = time.time()
            trigger = (now - last_time) >= args.interval
            if use_gui:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    trigger = True
            if trigger:
                last_time = now
                with tempfile.TemporaryDirectory() as run_path:
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if args.mode == "full":
                            res = process_image(image=rgb,
                                                image_is_raw=True,
                                                detection_model_path=
                                                detection_model_path,
                                                key_point_model_path=
                                                key_point_model_path,
                                                segmentation_model_path=
                                                segmentation_model_path,
                                                run_path=run_path,
                                                debug=False,
                                                eval_mode=False)
                        else:
                            res = fast_process_image(
                                image=rgb,
                                detection_model_path=detection_model_path,
                                key_point_inferencer=key_point_inferencer,
                                segmentation_model_path=segmentation_model_path,
                                range_min=args.range_min,
                                range_max=args.range_max,
                                reverse_scale=args.reverse_scale)
                        val = res.get("value", None)
                        unit = res.get("unit", None) or ""
                        if val is not None:
                            reading_text = f"{val:.2f}{unit}"
                            st = classify_status(val, args.range_min,
                                                 args.range_max,
                                                 args.low_threshold,
                                                 args.high_threshold)
                            state_text = st or ""
                            print(f"当前读数：{reading_text}" +
                                  (f"，状态：{state_text}" if state_text else ""))
                            if args.speak and state_text:
                                speak(f"当前读数{reading_text}，状态{state_text}")
                        else:
                            reading_text = "读取失败"
                            state_text = ""
                            print("读取失败")
                    except Exception as e:
                        reading_text = "读取失败"
                        state_text = ""
                        print(f"读取失败：{e}")

            cv2.putText(show,
                        f"Reading: {reading_text}",
                        (10, max(30, h - 40)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0),
                        2,
                        lineType=cv2.LINE_AA)
            if state_text:
                cv2.putText(show,
                            f"State: {state_text}",
                            (10, max(60, h - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255),
                            2,
                            lineType=cv2.LINE_AA)

            if use_gui:
                cv2.imshow("Gauge Reader", show)
    finally:
        cap.release()
        if use_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
