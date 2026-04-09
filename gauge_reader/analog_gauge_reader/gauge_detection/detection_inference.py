from ultralytics import YOLO

MODEL_CACHE = {}


def detection_gauge_face(img, model_path='best.pt'):
    '''
    uses yolo v8 to get bounding box of gauge face
    :param img: numpy image
    :param model_path: path to yolov8 detection model
    :return: highest confidence box for further processing and list of all boxes for visualization
    '''
    if model_path not in MODEL_CACHE:
        MODEL_CACHE[model_path] = YOLO(model_path)
    model = MODEL_CACHE[model_path]

    results = model(img)  # run inference, detects gauge face and needle

    # get list of detected boxes, already sorted by confidence
    boxes = results[0].boxes

    if len(boxes) == 0:
        raise Exception("No gauge detected in image")

    # get highest confidence box which is of a gauge face
    gauge_face_box = boxes[0]

    box_list = []
    for box in boxes:
        box_list.append(box.xyxy[0].int())

    return gauge_face_box.xyxy[0].int(), box_list
