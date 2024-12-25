import cv2
import numpy as np
import os

PIXEL_THRESHOLD = 2000  # 設定閾值，僅保留像素數大於該值的區域
AREA_THRESHOLD = 500 # 設定閾值，避免過小的分割區域
DISTANCE_THRESHOLD = 250 # 定義距離閾值（例如：設定 10 為最大可接受距離）
SHORT_SIDE = 120 # 轉動短邊判斷閾值
TWO_POINT_TEETH_THRESHOLD = 259 # 初判單雙牙尖使用
RANGE_FOR_TOOTH_TIP_LEFT = 80 # 強迫判斷雙牙尖，中心區域定義使用(左)
RANGE_FOR_TOOTH_TIP_RIGHT = 40 # 強迫判斷雙牙尖，中心區域定義使用(右)

def show_two(img1, img2, title1="Image 1", title2="Image 2"):
    """
    顯示兩張圖像並設定標題

    參數:
    - img1: 第一張圖像
    - img2: 第二張圖像
    - title1: 第一張圖像的標題
    - title2: 第二張圖像的標題
    """
    # 檢查圖像是否為灰階，若是則轉換成 RGB 以便 matplotlib 正確顯示
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

def calculate_distance(row_true, row_cleaned):
    true_values = np.array([row_true['enamel_x'], row_true['enamel_y']])
    cleaned_values = np.array([row_cleaned['enamel_x_predicted'], row_cleaned['enamel_y_predicted']])
    return np.linalg.norm(true_values - cleaned_values)

def calculate_distance_with_scale(p1,p2,scale_x,scale_y):
    p1_array = np.array([p1[0]*scale_x, p1[1]*scale_y])
    p2_array = np.array([p2[0]*scale_x, p2[1]*scale_y])
    return np.linalg.norm(p1_array - p2_array)

# 根據 mask 判斷轉正使用的旋轉角度
def get_rotation_angle(mask):
    # 找輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0
    cnt = contours[0]

    # 根據輪廓取得方框
    rect = cv2.minAreaRect(cnt)
    angle = rect[2]

    # 牙齒長邊在左右兩側，確保長邊是垂直於水平線
    if rect[1][0] > rect[1][1]:
        angle += 90
    if angle > 90:
        angle += 180
    return angle

# 利用 cv2 進行轉動
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 建立轉移角度用矩陣
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 利用建立的矩陣進行轉移轉動
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

# 把指定座標轉移回去原圖角度
def convert_coord_back(coord, angle, image):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # 建立轉移回原角度用矩陣
    M_inv = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

    # 因為只轉動單座標，該轉移矩陣要有另一項進行矩陣運算，這裡放單位矩陣做運算配合用
    coord_ones = np.array([coord[0], coord[1], 1.0])

    # 轉回原先角度
    original_coord_back = M_inv.dot(coord_ones)

    # 取得原圖座標，並且轉為整數
    original_coord_back = original_coord_back[:2].astype(int)

    return original_coord_back

# 檢查數值是否在指定範圍中
def is_within_range(value, target, range_size=50):
    return target - range_size <= value <= target + range_size

def assign_non_none_values(dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y):
    # 檢查 dentin_left_x 和 dentin_right_x 是否只找到其中一邊，是則判定為單牙尖
    if dentin_left_x is None and dentin_right_x is not None:
        dentin_left_x = dentin_right_x
    elif dentin_right_x is None and dentin_left_x is not None:
        dentin_right_x = dentin_left_x

    # 檢查 dentin_left_y 和 dentin_right_y 是否存在，是則判定為單牙尖
    if dentin_left_y is None and dentin_right_y is not None:
        dentin_left_y = dentin_right_y
    elif dentin_right_y is None and dentin_left_y is not None:
        dentin_right_y = dentin_left_y

    return dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y

# 根據指定順序，照點座標高度(y座標)排序
def get_top_points(contours, reverse=True):
    all_points = []
    # 取得輪廓中所有點，並且放到同一 list
    for contour in contours:
        sorted_points = sorted(contour, key=lambda x: x[0][1], reverse=reverse)
        top_points = sorted_points
        all_points.extend(top_points)
    # 排序
    all_points = sorted(all_points, key=lambda x: x[0][1], reverse=reverse)
    return all_points

# 計算 percentage 和期數，預測資料使用
def calculate_predicted_stage(row):
    enamel_x, enamel_y = row['enamel_x_predicted'], row['enamel_y_predicted']
    gum_x, gum_y = row['gum_x_predicted'], row['gum_y_predicted']
    dentin_x, dentin_y = row['dentin_x_predicted'], row['dentin_y_predicted']

    # 計算 A, B, C 點之間的距離
    AB = np.sqrt((enamel_x - gum_x) ** 2 + (enamel_y - gum_y) ** 2)
    AC = np.sqrt((enamel_x - dentin_x) ** 2 + (enamel_y - dentin_y) ** 2)
    
    # 計算 percentage
    percentage = (AB / AC) * 100
    
    # 判斷期數
    if percentage < 15:
        stage = "1"
    elif 15 <= percentage <= 33:
        stage = "2"
    else:
        stage = "3"
    
    return percentage, stage


# ---------- 影像處理與遮罩相關函式 ------------ #

def load_images_and_masks(dir_path, target_dir):
    """載入影像及其對應的遮罩"""
    paths = {
        'gum': f"gum_{target_dir}.png",
        'teeth': f"teeth_{target_dir}.png",
        'dental_crown': f"dentalcrown_{target_dir}.png",
        'crown': f"crown_{target_dir}.png",
        'dentin': f"dentin_{target_dir}.png",
        'original': f"raw_{target_dir}.png"
    }
    images = {name: cv2.imread(os.path.join(dir_path, path), cv2.IMREAD_GRAYSCALE if name != 'original' else cv2.IMREAD_COLOR) 
              for name, path in paths.items()}
    return images

def threshold_images(images):
    """將影像轉為二值圖"""
    binary_images = {}
    for key, img in images.items():
        if key != 'original':
            _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
            binary_images[key] = binary_img
    return binary_images

def clean_mask(mask, kernel_size=(3, 3), iterations=5):
    """清理影像中的雜點"""
    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.erode(mask, kernel, iterations=iterations)
    mask = cv2.dilate(mask, kernel, iterations=iterations)
    return mask

def filter_large_components(mask, pixel_threshold):
    """過濾掉像素數量小於閾值的區域"""
    num_labels, labels = cv2.connectedComponents(mask)
    label_counts = np.bincount(labels.flatten())
    filtered_image = np.zeros_like(mask)
    for label in range(1, num_labels):
        if label_counts[label] > pixel_threshold:
            filtered_image[labels == label] = 255
    return filtered_image

# ---------- 影像分析與特徵提取相關函式 ------------ #
def get_mid_point(image, dilated_mask, idx):
    """取得物件中點，並且繪製點及idx標籤於 image"""
    non_zero_points = np.column_stack(np.where(dilated_mask > 0))
    if len(non_zero_points) > 0:
        mid_point = np.mean(non_zero_points, axis=0).astype(int)
        mid_y, mid_x = mid_point
        # 在繪製用圖片標註 dentin 中心點及對應數字標籤
        #cv2.putText(image, str(idx), (mid_x-5, mid_y-5), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(image, (mid_x, mid_y), 5, (255, 255, 0), -1)  # 黃色圓點
    return mid_y, mid_x

def locate_points_with_dental_crown(dental_crown_bin, dilated_mask, mid_x, mid_y, overlay):
    intersection = cv2.bitwise_and(dental_crown_bin, dilated_mask)

    overlay[intersection > 0] = (255, 0, 0)  # 將 dentin 顯示
    # 取得交集區域的 contour 作為交點
    contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None, None
    # 將交點進行排序
    corners = get_top_points(contours, reverse=True)
    # 確認排序成功成功
    if corners is not None:
        # 整數化
        corners = np.int32(corners)
        # 初始化取得座標，預設左右兩邊皆有
        enamel_left_x = None
        enamel_left_y = None
        enamel_right_x = None
        enamel_right_y = None
        # 針對每個點進行處理
        for corner in corners:
            # 取得交點座標
            x, y = corner.ravel()
            # 判斷左右
            if x < mid_x:
                # 後續判斷
                if enamel_left_x is not None:
                    # 找到 y 最大者
                    if y > enamel_left_y:
                        enamel_left_x = x
                        enamel_left_y = y
                        continue
                    # 因以排序，看到 x 座標過接近的交點就不要重複看
                    elif is_within_range(x, enamel_left_x):
                        continue
                # 初判
                else:
                    enamel_left_x = x
                    enamel_left_y = y
            elif x > mid_x:
                # 後續判斷
                if enamel_right_x is not None:
                    # 找到 y 最大者
                    if y > enamel_right_y:
                        enamel_right_x = x
                        enamel_right_y = y
                        continue
                    # 因以排序，看到 x 座標過接近的交點就不要重複看
                    elif is_within_range(x, enamel_right_x):
                        continue
                # 初判
                else:
                    enamel_right_x = x
                    enamel_right_y = y
    return enamel_left_x, enamel_left_y, enamel_right_x, enamel_right_y

def locate_points_with_gum(gum_bin, dilated_mask, mid_x, mid_y, overlay):
    """處理與 gum 之交點 (Alveolar_bone的頂端)"""
    # 獲取每個獨立 mask 與原始 mask 的交集區域
    intersection = cv2.bitwise_and(gum_bin, dilated_mask)
    overlay[intersection > 0] = (0, 255, 0)  # 將 dentin 顯示
    # 取得交集區域的 contour 作為交點
    contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None, None
    # 反向排序
    corners = get_top_points(contours, reverse=False) #  
    # 確認排序成功
    if corners is not None:
        # 整數化
        top_corners = np.int32(corners)  
        # 初始化取得座標，預設左右兩邊皆有
        gum_left_x = None
        gum_left_y = None
        gum_right_x = None
        gum_right_y = None
        # 針對每個點進行處理
        for corner in top_corners:
            # 取得交點座標
            x, y = corner.ravel()
            # 如果該交點超出中心點太多，就過濾掉
            if x >= mid_x-40 and x <= mid_x+40:
                continue
            # 判斷左右
            if x < mid_x:
                # 後續判斷
                if gum_left_x is not None:
                    # 找到 y 最小者
                    if y < gum_left_y:
                        gum_left_x = x
                        gum_left_y = y
                        continue
                    # 因以排序，看到 x 座標過接近的交點就不要重複看
                    elif is_within_range(x, gum_left_x):
                        continue
                # 初判
                else:
                    gum_left_x = x
                    gum_left_y = y
            elif x > mid_x:
                # 後續判斷
                if gum_right_x is not None:
                    # 找到 y 最小者
                    if y < gum_right_y:
                        gum_right_x = x
                        gum_right_y = y
                        continue
                    # 因以排序，看到 x 座標過接近的交點就不要重複看
                    elif is_within_range(x, gum_right_x):
                        continue
                # 初判
                else:
                    gum_right_x = x
                    gum_right_y = y
    return gum_left_x, gum_left_y, gum_right_x, gum_right_y

def locate_points_with_dentin(gum_bin, dilated_mask, mid_x, mid_y, angle ,short_side, image, component_mask):
    # 取得 dentin 與 gum 交集
    intersection = cv2.bitwise_and(gum_bin, dilated_mask)
    # 由於希望取得 dentin 與 gum 交集的底部，所以需要轉正
    # 建立旋轉後的 mask 和 繪製用圖片
    c_image = rotate_image(image, angle)
    c_intersection = rotate_image(intersection, angle)
    c_dilated_mask = rotate_image(dilated_mask, angle)
    # 取得旋轉後區域的中點
    non_zero_points = np.column_stack(np.where(c_dilated_mask > 0))
    if len(non_zero_points) > 0:
        mid_point = np.mean(non_zero_points, axis=0).astype(int)
        c_mid_y, c_mid_x = mid_point
    # 初始化取得座標，預設左右兩邊皆有        
    dentin_left_x = None
    dentin_left_y = None
    dentin_right_x = None
    dentin_right_y = None
    # 根據短邊大小(寬度)，初步判斷單牙尖或雙牙尖
    if short_side > TWO_POINT_TEETH_THRESHOLD:
        # 較寬者，強迫判斷為雙牙尖
        contours, _ = cv2.findContours(c_intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, None, None, None
        top_points = get_top_points(contours, reverse=True)
        bottom_corners = top_points
        # 初始化左右兩邊的點列表
        left_corners = []
        right_corners = []
        # 根據 c_mid_x 分配點到左右兩邊
        for point in bottom_corners:
            x, y = point.ravel() # 取得左右兩邊
            if x < c_mid_x-RANGE_FOR_TOOTH_TIP_LEFT:
                left_corners.append(point)
            # 雙牙尖，太中間的點不可能是牙尖
            elif x >= c_mid_x-RANGE_FOR_TOOTH_TIP_LEFT and x <= c_mid_x+RANGE_FOR_TOOTH_TIP_RIGHT:
                continue
            else:
                right_corners.append(point)
        # 左牙尖判斷
        for corner in left_corners:
            # 取得點座標
            x, y = corner.ravel()
            # 確定為左邊
            if x < c_mid_x:
                # 後續判斷
                if dentin_left_x is not None:
                    # 取得 y 最大者
                    if y > dentin_left_y:
                        dentin_left_x = x
                        dentin_left_y = y
                        continue
                # 初判
                else:
                    dentin_left_x = x
                    dentin_left_y = y
        # 右牙尖判斷
        for corner in right_corners:
            # 取得點座標
            x, y = corner.ravel()
            # 確定為右邊
            if x > c_mid_x:
                # 後續判斷
                if dentin_right_x is not None:
                    # 取得 y 最大者
                    if y > dentin_right_y:
                        dentin_right_x = x
                        dentin_right_y = y
                        continue
                # 初判
                else:
                    dentin_right_x = x
                    dentin_right_y = y
        # 避免 None 存在，因有可能在上述流程誤判為雙牙尖
        dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y = assign_non_none_values(dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y)
        #---- 例外狀況 ---- 左右牙尖高度落差較大
        print("Debugging ::")
        if all(v is None for v in [dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y]):
            print("All variables are None.")
        else:
            if not is_within_range(dentin_left_y, dentin_right_y, 200):
                if dentin_right_y > dentin_left_y:
                    dentin_left_y = dentin_right_y
                    dentin_left_x = dentin_right_x
                else:
                    dentin_right_y = dentin_left_y
                    dentin_right_x = dentin_left_x
    else:
        # 進行交點搜尋
        contours, _ = cv2.findContours(c_intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, None, None, None
        # 排序
        bottom_corners = get_top_points(contours, reverse=True)
        for corner in bottom_corners:
            x, y = corner.ravel()
            # 判斷左右邊(預設是雙牙尖)
            if x < c_mid_x:
                # 後續判斷
                if dentin_left_x is not None:
                    # 取得 y 最大者
                    if y > dentin_left_y:
                        dentin_left_x = x
                        dentin_left_y = y
                        continue
                # 初判
                else:
                    dentin_left_x = x
                    dentin_left_y = y
            elif x > c_mid_x:
                # 後續判斷
                if dentin_right_x is not None:
                    # 取得 y 最大者
                    if y > dentin_right_y:
                        dentin_right_x = x
                        dentin_right_y = y
                        continue
                # 初判
                else:
                    dentin_right_x = x
                    dentin_right_y = y
         # 避免 None 存在，因有可能在上述流程誤判為雙牙尖
        dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y = assign_non_none_values(dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y)
        # 如果判斷出來的雙邊牙尖過於接近，確定應為單牙尖狀況，故指定最小者為牙尖
        if is_within_range(dentin_left_x, dentin_right_x, 80):
            bottom_corner = bottom_corners[:1]
            for corner in bottom_corner:
                x, y = corner.ravel()
                dentin_left_x = x
                dentin_left_y = y
                dentin_right_x = x
                dentin_right_y = y
                cv2.circle(c_image, (x, y), 5, (255, 0, 0), -1)
    
    # 將旋轉後的座標，轉回原角度
    dentin_left_coord = [dentin_left_x, dentin_left_y]
    dentin_right_coord = [dentin_right_x, dentin_right_y]
    dentin_left_x, dentin_left_y = convert_coord_back(dentin_left_coord, angle, component_mask)
    dentin_right_x, dentin_right_y = convert_coord_back(dentin_right_coord, angle, component_mask)
    
    # 膨脹避免資訊被截斷(旋轉回去有可能截到)
    kernel = np.ones((3, 3), np.uint8)
    c_dilated_mask = cv2.dilate(c_dilated_mask, kernel, iterations=5)
    c_dilated_mask_rotated_back = rotate_image(c_dilated_mask, -angle)
    c_image_rotated_back = rotate_image(c_image, -angle)
    mask = c_dilated_mask_rotated_back > 0
    # 把對應區域放回去原圖
    image[mask] = c_image_rotated_back[mask]
    
    return dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y

def extract_largest_component(mask):
    """提取遮罩中的最大連通區域"""
    num_labels, labels = cv2.connectedComponents(mask)
    label_counts = np.bincount(labels.flatten())
    max_label = np.argmax(label_counts[1:]) + 1
    largest_component = np.zeros_like(mask)
    largest_component[labels == max_label] = 255
    return largest_component

def combine_masks(binary_images):
    """合併所有遮罩，形成完整的合併遮罩"""
    combined_mask = None
    for key in ['dental_crown', 'dentin', 'binary_images']:
        if key in binary_images:
            combined_mask = cv2.bitwise_or(combined_mask, binary_images[key]) if combined_mask is not None else binary_images[key]
    return combined_mask

def determine_stage(length, length2):
    ratio=length/length2
    if ratio < 0.15 and length < 2:
        return 0
    elif ratio < 0.15:
        return 1
    elif ratio <= 0.33:
        return 2
    return 3

def calculate_dental_pairs(prediction, scale):
    scale_w, scale_h = scale
    line_pairs = [
        (("enamel_left", "dentin_left", (0, 0, 255)), ("enamel_left", "gum_left", (0, 255, 255))),
        (("enamel_right", "dentin_right", (0, 0, 255)), ("enamel_right", "gum_right", (0, 255, 255)))
    ]
    dental_pair_list=[]
    # Iterate through the line pairs and draw lines
    for i, ((start, end, _), (start2, end2, _)) in enumerate(line_pairs):
        # Fetch the coordinates from the prediction
        p1, p2, p3, p4 = (prediction.get(start), prediction.get(end),
                          prediction.get(start2), prediction.get(end2))
        if any(pt is None or None in pt for pt in [p1, p2, p3, p4]):
            continue

        cal_length=calculate_distance_with_scale(p3, p4, scale_h, scale_w)
        trl_length=calculate_distance_with_scale(p1, p2, scale_h, scale_w)
        stage=determine_stage(cal_length, trl_length)
        dental_pair_list.append(
            {
            'side_id': i,
            'CEJ': (int(p1[0]),int(p1[1])),
            'ALC': (int(p4[0]),int(p4[1])),
            'APEX': (int(p2[0]),int(p2[1])),
            'TRL': float(trl_length),
            'CAL': float(cal_length),
            'ABLD': float(cal_length/trl_length),
            'stage': stage,
            }
        )
    return dental_pair_list

def int_processing(val):
    if val is None:
        return None  # 或者設為 0，根據需求決定
    return int(val)


