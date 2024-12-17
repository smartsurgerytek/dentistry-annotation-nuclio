import io, base64, cv2, json
import numpy as np
from ultralytics import YOLO
from skimage.measure import find_contours, approximate_polygon

def init_context(context):
    context.logger.info("Init context...  0%")
    context.user_data.model_handler = YOLO('/opt/nuclio/model.pt')
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run sst model")
    data = event.body
    image_bytes = io.BytesIO(base64.b64decode(data["image"]))
    nparr  = np.frombuffer(image_bytes.getvalue(),  np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rt = transform(image, context.user_data.model_handler)
    return context.Response(body=json.dumps(rt), headers={},
        content_type='application/json', status_code=200)

def transform(image, model):
    rt = []
    image= cv2.resize(image, (1280,960))
    class_names = model.names

    results = model(image)
    boxes = results[0].boxes
    confs = boxes.conf.tolist()

    masks = results[0].masks  # Masks object for segmentation masks outputs

    for box, conf, mask in zip(boxes, confs, masks.data):
        mask_np = mask.cpu().numpy()
        mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
        mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
        contours = find_contours(mask_binary, 0.5)
        contour = contours[0]
        contour = np.flip(contour, axis=1)
        polygons = approximate_polygon(contour, tolerance=2.5)
        xyxy = box.xyxy.tolist()
        xtl = int(xyxy[0][0])
        ytl = int(xyxy[0][1])
        xbr = int(xyxy[0][2])
        ybr = int(xyxy[0][3])
        cvat_mask = to_cvat_mask((xtl, ytl, xbr, ybr), mask_binary)
        rt.append({
            "confidence": conf,
            "label": class_names[int(box.cls)],
            "type": "mask",
            "points": polygons.ravel().tolist(),
            "mask": cvat_mask
        })
    return rt

def to_cvat_mask(box: list, mask):
    xtl, ytl, xbr, ybr = box
    flattened = mask[ytl:ybr + 1, xtl:xbr + 1].flat[:].tolist()
    flattened.extend([xtl, ytl, xbr, ybr])
    return flattened