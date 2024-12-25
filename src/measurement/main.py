import io, base64, cv2, json
import numpy as np
from ultralytics import YOLO
from utils import calculate_dental_pairs, clean_mask, extract_largest_component,AREA_THRESHOLD,SHORT_SIDE,get_rotation_angle,is_within_range,locate_points_with_dental_crown,get_mid_point,locate_points_with_gum,locate_points_with_dentin,int_processing
def init_context(context):
    context.logger.info("Init context... 0%")
    context.user_data.segmentation_model = YOLO('/opt/nuclio/segmentation-model.pt')
    context.user_data.contour_model = YOLO('/opt/nuclio/contour-model.pt')
    context.logger.info('Init context... 100%')

def handler(context, event):
    context.logger.info('Run sst model')
    data = event.body

    image_bytes = io.BytesIO(base64.b64decode(data['image']))
    nparr = np.frombuffer(image_bytes.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    segmentation_model = context.user_data.segmentation_model
    contour_model = context.user_data.contour_model

    prediction = inference(image, segmentation_model, contour_model)
    rt = inferenced_result_to_nuclio_result(prediction)
    return context.Response(body=json.dumps(rt), headers={}, content_type='application/json', status_code=200)


def inference(image: cv2.typing.MatLike, segmentation_model: YOLO, contour_model: YOLO):
    rt = []
    scale= (31/960,41/1080)
    components_model_masks_dict=get_mask_dict_from_model(segmentation_model, image, method='semantic')
    contours_model_masks_dict=get_mask_dict_from_model(contour_model, image, method='instance')
    denti_measure_names_map={
        'Alveolar_bone': 'gum',
        'Dentin': 'dentin',
        'Enamel': 'dental_crown',
        'Crown': 'crown' ### In fact it is enamel (why labeling man so stupid)
    }
    components_model_masks_dict = {denti_measure_names_map.get(k, k): v for k, v in components_model_masks_dict.items()}

    required_components = {
        'dentin': "No dental instance detected",
        'dental_crown': "No dental_crown detected",
        'gum': "No gum detected"
    }
    for component, _ in required_components.items():
        if components_model_masks_dict.get(component) is None:
            return []

    if contours_model_masks_dict.get('dental_contour') is None:
        return []

    for mask in contours_model_masks_dict['dental_contour']:
        if not 'dental_contour' in components_model_masks_dict.keys():
            components_model_masks_dict['dental_contour']=mask
        else:
            components_model_masks_dict['dental_contour']=cv2.bitwise_or(components_model_masks_dict['dental_contour'], mask) # find the union
    crown_or_enamal_mask=np.zeros_like(components_model_masks_dict['dentin'])
    for key in ['dental_crown','crown']:
        if components_model_masks_dict.get(key) is not None:
            crown_or_enamal_mask=cv2.bitwise_or(crown_or_enamal_mask, components_model_masks_dict[key])
    components_model_masks_dict['dentin']=components_model_masks_dict['dental_contour']-cv2.bitwise_and(components_model_masks_dict['dental_contour'], crown_or_enamal_mask)
    overlay = extract_features(components_model_masks_dict, image) # 處理繪圖用圖片等特徵處理後圖片

    predictions = []
    image_for_drawing=image.copy()

    for i, component_mask in enumerate(contours_model_masks_dict['dental_contour']):
        prediction = locate_points(image_for_drawing, component_mask, components_model_masks_dict, i+1, overlay)
        if len(prediction) == 0:
            continue
        dental_pair_list=calculate_dental_pairs(prediction, scale)
        if len(dental_pair_list) == 0:
            continue
        rt.append(dental_pair_list)
        if dental_pair_list:
            predictions.append(prediction)

    return rt

def extract_features(masks_dict, original_img):
    overlay = original_img.copy()
    kernel = np.ones((3, 3), np.uint8)

    masks_dict['dental_crown'] = clean_mask(masks_dict['dental_crown'])
    masks_dict['dentin'] = clean_mask(masks_dict['dentin'], kernel_size=(30, 1), iterations=1)
    masks_dict['gum'] = clean_mask(masks_dict['gum'], kernel_size=(30, 1), iterations=2)

    masks_dict['gum'] = extract_largest_component(masks_dict['gum'])

    masks_dict['gum'] = cv2.dilate(masks_dict['gum'], kernel, iterations=10)

    dental_contours=np.maximum(masks_dict['dentin'], masks_dict['dental_crown'])
    kernel = np.ones((23,23), np.uint8)
    filled = cv2.morphologyEx(dental_contours, cv2.MORPH_CLOSE, kernel)
    filled=cv2.bitwise_and(filled, cv2.bitwise_not(masks_dict['dental_crown']))
    masks_dict['dentin']=filled

    key_color_mapping={
        'dental_crown': (163, 118, 158),
        'dentin':(117, 122, 152),
        'gum': (0, 177, 177),
        'crown': (255, 0, 128),
    }
    for key in key_color_mapping.keys():
        if masks_dict.get(key) is not None:
            overlay[masks_dict[key] > 0] = key_color_mapping[key]

    return overlay

def locate_points(image, component_mask, binary_images, idx, overlay):
    def less_than_area_threshold(component_mask, area_threshold):
        area = cv2.countNonZero(component_mask)
        if area < area_threshold:
            return True
        return False

    prediction = {}
    if less_than_area_threshold(component_mask, AREA_THRESHOLD):
        return prediction
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0]) # 最小區域長方形
    box = cv2.boxPoints(rect) # 取得長方形座標
    box = np.int32(box) # 整數化
    width = rect[1][0]  # 寬度
    height = rect[1][1]  # 高度
    short_side = min(width, height)  # 短邊
    long_side = max(width, height)
    if short_side < SHORT_SIDE:
       return prediction

    angle = get_rotation_angle(component_mask)
    if is_within_range(short_side, long_side, 30):
        angle = 0

    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(component_mask, kernel, iterations=7)

    mid_y, mid_x = get_mid_point(image, dilated_mask, idx)

    enamel_left_x, enamel_left_y, enamel_right_x, enamel_right_y = locate_points_with_dental_crown(binary_images["dental_crown"], dilated_mask, mid_x, mid_y, overlay)
    gum_left_x, gum_left_y, gum_right_x, gum_right_y = locate_points_with_gum(binary_images["gum"], dilated_mask, mid_x, mid_y, overlay)
    dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y = locate_points_with_dentin(binary_images["gum"], dilated_mask, mid_x, mid_y, angle, short_side, image, component_mask)

    prediction = {"teeth_center": (mid_x, mid_y),
                "enamel_left": (enamel_left_x, enamel_left_y), "enamel_right":(enamel_right_x, enamel_right_y),
                "gum_left":(gum_left_x, gum_left_y), "gum_right": (gum_right_x, gum_right_y),
                "dentin_left":(dentin_left_x, dentin_left_y), "dentin_right":(dentin_right_x, dentin_right_y),
                }

    for key, (x, y) in prediction.items():
        prediction[key] = (int_processing(x), int_processing(y))

    return prediction

def get_mask_dict_from_model(model, image, method='semantic'):
    results=model.predict(image)
    result=results[0]
    class_names = result.names
    boxes = result.boxes
    masks = result.masks

    if masks is None:
        return {}

    masks_dict={}
    for mask, box in zip(masks.data, boxes):
        class_id = int(box.cls)
        class_name = class_names[class_id]
        mask_np = mask.cpu().numpy()
        mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
        mask_binary = (mask_np > 0.5).astype(np.uint8) * 255

        if np.sum(mask_binary) == 0:
            continue

        if class_name not in masks_dict.keys():
            if method=='semantic':
                masks_dict[class_name]=mask_binary
            else:
                masks_dict[class_name]=[mask_binary]
            continue

        if method=='semantic':
            masks_dict[class_name]=cv2.bitwise_or(masks_dict[class_name], mask_binary)
        else:
            masks_dict[class_name].append(mask_binary)


    return masks_dict

def inferenced_result_to_nuclio_result(pairs):
    rt = []
    point_names = ['CEJ', 'ALC', 'APEX']
    line_specs = [
        { "name": 'CAL', "points": ['CEJ', 'ALC']},
        { "name": 'TRL', "points": ['CEJ', 'APEX']}]

    for pair in pairs:
        for i, element in enumerate(pair):
            for point_name in point_names:
                [x, y] = element[point_name]
                rt.append({
                    "label": point_name,
                    "type": "points",
                    "points": [x, y]
                })
            for line_spec in line_specs:
                points = [*element[line_spec['points'][0]], *element[line_spec['points'][1]]]
                length = element[line_spec['name']]
                rt.append({
                    "label": line_spec['name'],
                    "type": "polyline",
                    "points": points,
                    "attributes": [{"name": "length", "input_type": "number", "value": str(length)}]
                })
            rt.append({
                "label": "metadata",
                "type": "tag",
                "attributes": [
                    {"name": "ABLD", "input_type": "number", "value": element["ABLD"]},
                    {"name": "stage", "input_type": "number", "value": element["stage"]}
                ]
            })
    return rt

if __name__ == "__main__":
    image = cv2.imread('/opt/nuclio/tests/test.jpeg')
    segmentation_model = YOLO('/opt/nuclio/segmentation-model.pt')
    contour_model = YOLO('/opt/nuclio/contour-model.pt')
    prediction = inference(image, segmentation_model, contour_model)
    rt = inferenced_result_to_nuclio_result(prediction)
    print(rt)