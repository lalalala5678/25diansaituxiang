import cv2
import numpy as np
import get_image

# ================== 配置参数 ==================
IMAGE_PATH = "original.jpg"          # 读取的源图像
CROP_COORDS = (0, 3040, 508, 3548)   # (y1, y2, x1, x2) 根据需要修改
THRESHOLD_VALUE = 70                # 灰度阈值 (0‑255)
DARK_THRESHOLD = 80                  # RGB 三通道都 ≤ 此值 → 认为是真黑色
MORPH_KERNEL_SIZE = 5                # 形态学核尺寸 (像素)
MORPH_ITERATIONS = 1                 # 形态学运算迭代次数
MIN_CONTOUR_AREA = 10000             # 轮廓面积阈值 (像素²)
DEBUG_PREFIX = "step_"               # 输出文件名前缀（自动加步骤编号）
A4_RATIO = 297 / 210        # ≈ 1.414
RATIO_TOL = 0.25            # 可调
EXPECTED_BORDER_PX = 20     # 期望黑边宽度（像素）
BORDER_TOL_PX = 8           # 可调

# ============================================


def read_and_crop(image_path: str, crop_coords: tuple) -> np.ndarray:
    """读取图像并裁剪指定 ROI。"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    y1, y2, x1, x2 = crop_coords
    return img[y1:y2, x1:x2]


def binarize_image(img: np.ndarray, thresh: int, dark_thresh: int) -> np.ndarray:
    """灰度阈值反转 + RGB 再过滤：仅保留真正的黑色像素。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    b, g, r = cv2.split(img)
    dark_mask = (b <= dark_thresh) & (g <= dark_thresh) & (r <= dark_thresh)
    binary[(binary == 255) & (~dark_mask)] = 0
    return binary


def morphology_clean(binary: np.ndarray, ksize: int, iterations: int = 1) -> np.ndarray:
    """闭运算连通断裂 + 开运算去除小噪声。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opened

# ===== 新增 Step‑4: 轮廓提取 =====

def find_candidate_contours(binary: np.ndarray, min_area: int = MIN_CONTOUR_AREA):
    """\
    提取满足面积阈值的**全部**轮廓（外部 + 内部），按面积从大到小排序。

    参数:
        binary (np.ndarray): 形态学清理后的二值图。
        min_area (int): 保留轮廓的最小面积阈值。

    返回:
        List[np.ndarray]: 经过面积过滤的轮廓列表，包含嵌套的内部轮廓。
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
    filtered.sort(key=cv2.contourArea, reverse=True)
    return filtered

def _rect_info(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) != 4:
        return False, None, None
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    if w == 0 or h == 0:
        return False, rect, None
    ratio = max(w, h) / min(w, h)
    ratio_err = abs(ratio - A4_RATIO)
    return ratio_err <= RATIO_TOL, rect, ratio_err


def select_nested_rectangles(contours, debug=0):
    """
    在给定轮廓列表中选出最符合要求的外框和内框轮廓对。
    评分标准包括:
    1. 单个轮廓形状接近矩形 (近似4条直边, 4个直角), 长宽比接近 A4_RATIO。
    2. 两个轮廓中心点接近重合。
    3. 两个轮廓边缘方向平行 (朝向差异小)。
    4. 大轮廓完全包含小轮廓。
    5. 黑边像素分布: 大轮廓略内侧的像素偏黑 (在黑边上), 略外侧的像素偏亮; 小轮廓略外侧像素偏黑 (在黑边上), 略内侧的像素偏亮。
    6. 上下左右黑边宽度一致性: 大轮廓与小轮廓对应边之间的距离基本一致。
    始终返回一个 (outer, inner) 轮廓对; 若无完美匹配, 返回评分最高的矩形对。
    参数:
        debug: 默认为0。当非0时，将详细日志输出到文本文件。
    """
    log = None
    if debug:
        log = open("select_rectangles_log.txt", "w")
        log.write("=== select_nested_rectangles Debug Log ===\n")
    # 如果轮廓不足两个, 直接返回面积最大的两个
    if len(contours) < 2:
        if log:
            log.write(f"Contours count {len(contours)} < 2, returning top 2 by area without scoring.\n")
            log.close()
        return contours[0], contours[1]
    if log:
        log.write(f"Total input contours: {len(contours)}\n")
    best_pair = None
    best_score = -1.0  # 初始化最佳分数
    rect_candidates = []
    # 尝试获取全局原始裁剪图像 (用于像素黑度判断)
    img = None
    try:
        global cropped
        img = cropped
    except NameError:
        img = None
    dark_mask = None
    if img is not None:
        # 计算暗像素掩码 (像素RGB各通道 <= DARK_THRESHOLD 则视为黑)
        b, g, r = cv2.split(img)
        dark_mask = (b <= DARK_THRESHOLD) & (g <= DARK_THRESHOLD) & (r <= DARK_THRESHOLD)
    # 预选矩形候选轮廓
    for idx, c in enumerate(contours):
        # 近似多边形轮廓，确保是4边形
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            if log:
                log.write(f"Contour {idx}: approximated vertices = {len(approx)}, not 4, skipped.\n")
            continue  # 轮廓不是4个顶点，跳过
        # 计算每个角的角度误差
        pts = [pt[0] for pt in approx]
        angles = []
        for i in range(4):
            p0 = pts[i]
            p1 = pts[(i - 1) % 4]
            p2 = pts[(i + 1) % 4]
            v1 = p1 - p0
            v2 = p2 - p0
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                continue
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # 修正浮点误差范围
            angle = np.degrees(np.arccos(cos_angle))
            if angle > 180:
                angle = 360 - angle  # 取内角
            angles.append(angle)
        if len(angles) < 4:
            if log:
                log.write(f"Contour {idx}: angle calculation incomplete, skipped.\n")
            continue
        angle_errors = [abs(a - 90) for a in angles]
        max_angle_err = max(angle_errors)
        avg_angle_err = sum(angle_errors) 
        # 角度评分: 平均偏差在阈值内则线性评分, 否则为0
        angle_score = 0.0
        angle_threshold = 20.0  # 平均角度误差容差 (度)
        if max_angle_err < 45:  # 若某角偏差超过45度，则视为无效矩形
            angle_score = max(0.0, 1.0 - avg_angle_err / angle_threshold)
        # 长宽比评分
        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        if w == 0 or h == 0:
            if log:
                log.write(f"Contour {idx}: zero width/height in minAreaRect, skipped.\n")
            continue
        ratio = max(w, h) / min(w, h)
        ratio_err = abs(ratio - A4_RATIO)
        ratio_score = 0.0
        if ratio_err <= RATIO_TOL:
            ratio_score = 1.0 - (ratio_err / RATIO_TOL)
        else:
            ratio_score = 0.0
        # 单轮廓形状总评分 (角度 + 长宽比*4)
        shape_score = angle_score + ratio_score * 4.0
        # 规范化方向角度 (将角度归一化到 [-45, 45] 范围用于比较)
        angle_rect = rect[2]
        if angle_rect < -45.0:
            angle_rect += 90.0
        center = rect[0]
        rect_candidates.append((c, center, angle_rect, shape_score, rect))
        if log:
            log.write(f"Candidate {len(rect_candidates)-1}: area={cv2.contourArea(c):.0f}, shape_score={shape_score:.2f}, angle_score={angle_score:.2f}, ratio_score={ratio_score:.2f}, center={center}, angle_rect={angle_rect:.1f}\n")
    # 候选不足两个时，退而返回面积最大的两个轮廓
    if len(rect_candidates) < 2:
        if log:
            log.write(f"Rectangular candidates count = {len(rect_candidates)} < 2, returning top 2 by area.\n")
            log.close()
        return contours[0], contours[1]
    # 按轮廓面积从大到小排序候选列表
    rect_candidates.sort(key=lambda x: cv2.contourArea(x[0]), reverse=True)
    if log:
        log.write(f"Filtered rect_candidates count: {len(rect_candidates)} (sorted by area)\n")
    # 遍历所有候选对计算评分
    for i in range(len(rect_candidates)):
        c_out, center_out, angle_out, shape_score_out, rect_out = rect_candidates[i]
        for j in range(i + 1, len(rect_candidates)):
            c_in, center_in, angle_in, shape_score_in, rect_in = rect_candidates[j]
            # 确保外框面积大于内框面积，如不是则交换
            if cv2.contourArea(c_in) > cv2.contourArea(c_out):
                c_out, center_out, angle_out, shape_score_out, rect_out, \
                c_in, center_in, angle_in, shape_score_in, rect_in = \
                c_in, center_in, angle_in, shape_score_in, rect_in, \
                c_out, center_out, angle_out, shape_score_out, rect_out
            # 内外包含关系检测
            inside = True
            for pt in c_in:
                if cv2.pointPolygonTest(c_out, (float(pt[0][0]), float(pt[0][1])), False) < 0:
                    inside = False
                    break
            if not inside:
                if log:
                    log.write(f"Pair (outer={i}, inner={j}): inner contour not fully inside outer, skip.\n")
                continue
            # 中心点距离评分 (距离越小分数越高)
            dx = center_out[0] - center_in[0]
            dy = center_out[1] - center_in[1]
            center_dist = np.hypot(dx, dy)
            center_score = max(0.0, 1.0 - center_dist / 100.0)  # 超过100像素则趋向0分
            # 方向平行评分 (角度差异越小分数越高)
            angle_diff = abs(angle_out - angle_in)
            if angle_diff > 90.0:
                angle_diff = abs(angle_diff - 90.0)
            orientation_score = max(0.0, 1.0 - angle_diff / 10.0)
            # 黑边像素分布评分
            pixel_score = 0.0
            if dark_mask is not None:
                offset = 10  # 像素偏移距离 
                # 定义辅助函数计算矩形边线的暗像素占比
                def line_dark_fraction(rect):
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    mask = np.zeros(dark_mask.shape, dtype=np.uint8)
                    cv2.polylines(mask, [box], isClosed=True, color=255, thickness=5)
                    line_pixels = mask.astype(bool)
                    total = np.count_nonzero(line_pixels)
                    if total == 0:
                        return 0.0
                    dark_count = np.count_nonzero(dark_mask & line_pixels)
                    return dark_count / float(total)
                # 计算外框偏移内侧和外侧的黑像素比例
                outer_rect_inner = (center_out, (rect_out[1][0] - 2*offset, rect_out[1][1] - 2*offset), angle_out)
                outer_rect_outer = (center_out, (rect_out[1][0] + 2*offset, rect_out[1][1] + 2*offset), angle_out)
                outer_inner_frac = line_dark_fraction(outer_rect_inner)
                outer_outer_frac = line_dark_fraction(outer_rect_outer)
                # 计算内框偏移内侧和外侧的黑像素比例
                inner_rect_inner = (center_in, (rect_in[1][0] - 2*offset, rect_in[1][1] - 2*offset), angle_in)
                inner_rect_outer = (center_in, (rect_in[1][0] + 2*offset, rect_in[1][1] + 2*offset), angle_in)
                inner_inner_frac = line_dark_fraction(inner_rect_inner)
                inner_outer_frac = line_dark_fraction(inner_rect_outer)
                # 外框黑边评分: 内侧线黑度 - 外侧线黑度 (黑边内侧应更黑)
                outer_line_score = outer_inner_frac - outer_outer_frac
                # 内框黑边评分: 外侧线黑度 - 内侧线黑度 (黑边外侧应更黑)
                inner_line_score = inner_outer_frac - inner_inner_frac
                # 若出现负值则置0，表示不符合预期
                if outer_line_score < 0:
                    outer_line_score = 0.0
                if inner_line_score < 0:
                    inner_line_score = 0.0
                # 综合像素黑度评分 (可取平均或加权，这里取平均)
                pixel_score = (outer_line_score + inner_line_score) / 2.0
            # 黑边宽度一致性评分
            border_consistency_score = 0.0
            # 计算外框和内框的四边距离
            box_out = cv2.boxPoints(rect_out)
            box_in = cv2.boxPoints(rect_in)
            # 将4个顶点按 TL-TR-BR-BL 顺时针排序
            def order_corners(pts):
                pts = np.array(pts, dtype=np.float32)
                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1)
                tl = np.argmin(s); br = np.argmax(s)
                tr = np.argmin(diff); bl = np.argmax(diff)
                return np.array([pts[tl], pts[tr], pts[br], pts[bl]], dtype=np.float32)
            outer_pts = order_corners(box_out)
            inner_pts = order_corners(box_in)
            # 点到线的距离计算函数
            def point_line_distance(P, A, B):
                AB = B - A
                if AB[0] == 0 and AB[1] == 0:
                    return 0.0
                AP = P - A
                cross_val = AB[0]*AP[1] - AB[1]*AP[0]
                return abs(cross_val) / np.linalg.norm(AB)
            # 计算对应边之间的距离（平均）
            border_dists = []
            edge_pairs = [(0,1), (1,2), (2,3), (3,0)]  # Top, Right, Bottom, Left
            for (a, b) in edge_pairs:
                A = outer_pts[a]; B = outer_pts[b]
                P = inner_pts[a]; Q = inner_pts[b]
                d1 = point_line_distance(P, A, B)
                d2 = point_line_distance(Q, A, B)
                border_dists.append((d1 + d2) / 2.0)
            if border_dists:
                max_bd = max(border_dists); min_bd = min(border_dists)
                diff_bd = max_bd - min_bd
                if diff_bd <= BORDER_TOL_PX:
                    border_consistency_score = 1.0 - (diff_bd / BORDER_TOL_PX)
                else:
                    border_consistency_score = 0.0
            # 组合总评分: 形状、中心、方向、像素、黑边宽度一致性 (像素和一致性分数适当加权)
            total_score = shape_score_out + shape_score_in + center_score \
                          + orientation_score + pixel_score * 2.0 + border_consistency_score * 2.0
            # 日志记录评分细节
            if log:
                log.write(f"Pair (outer={i}, inner={j}): shape_out={shape_score_out:.2f}, "
                          f"shape_in={shape_score_in:.2f}, center_score={center_score:.2f}, "
                          f"orient_score={orientation_score:.2f}, pixel_score={pixel_score:.2f}, "
                          f"border_score={border_consistency_score:.2f}, total={total_score:.2f}\n")
                if border_dists:
                    log.write(f"    border_dists(px): top={border_dists[0]:.1f}, right={border_dists[1]:.1f}, "
                              f"bottom={border_dists[2]:.1f}, left={border_dists[3]:.1f}, range={diff_bd:.1f}\n")
            # 如果当前组合评分更高，则更新最佳匹配
            if total_score > best_score:
                best_score = total_score
                best_pair = (c_out, c_in)
                if log:
                    log.write(f"    -> New best pair (outer={i}, inner={j}) with score {best_score:.2f}\n")

                    

    # 返回最佳轮廓对（若未找到匹配则返回面积最大的两个矩形候选）
    if best_pair is not None:
        if log:
            log.write(f"Best pair found with score {best_score:.2f}. Returning best outer/inner contours.\n")
            log.close()
        return best_pair
    else:
        if log:
            log.write("No ideal nested pair found. Returning top 2 candidates by area as fallback.\n")
            log.close()
        return rect_candidates[0][0], rect_candidates[1][0]



# 编写函数，将rect_candidates[0][0], rect_candidates[1][0]优化为理想的外/内矩形对，理想指的是线绝对是直的，角是90度。
def optimize_rectangles(cnt_a: np.ndarray, cnt_b: np.ndarray):
    """
    ⽤最⼩外接矩形（minAreaRect）将两个任意四边形轮廓
    优化为“理想矩形”外/内框：
        • 四条边完全直且两两垂直（90°）。
        • 返回顺序为 (outer_rect_cnt, inner_rect_cnt)，即外框在前、内框在后。
    
    参数
    ----
    cnt_a, cnt_b : np.ndarray
        原始轮廓（findContours 返回的格式）。
    
    返回
    ----
    outer_cnt, inner_cnt : np.ndarray
        形状均为 (4, 1, 2) 的整型顶点坐标轮廓，满足外框包含内框。
    """
    def contour_to_box(contour):
        """把任意轮廓拟合为 4×1×2 的理想矩形顶点数组。"""
        rect = cv2.minAreaRect(contour)            # (center, (w,h), angle)
        box  = cv2.boxPoints(rect)                 # 4×2 float
        box  = np.intp(box).reshape(-1, 1, 2)      # 4×1×2 int
        return box, rect

    # 将两个轮廓拟合为矩形
    box_a, rect_a = contour_to_box(cnt_a)
    box_b, rect_b = contour_to_box(cnt_b)

    # 按面积确定外框 / 内框
    area_a = cv2.contourArea(box_a)
    area_b = cv2.contourArea(box_b)
    if area_a >= area_b:
        outer_cnt, inner_cnt = box_a, box_b
    else:
        outer_cnt, inner_cnt = box_b, box_a

    # 可选：强制内框整体向内微缩 1 像素，避免数值误差导致“不完全包含”
    shrink_px = 1
    inner_cnt = inner_cnt.astype(np.float32)
    inner_cnt[:, 0, 0] = np.where(inner_cnt[:, 0, 0] > outer_cnt[:, 0, 0].min(),
                                  inner_cnt[:, 0, 0] - shrink_px,
                                  inner_cnt[:, 0, 0] + shrink_px)
    inner_cnt[:, 0, 1] = np.where(inner_cnt[:, 0, 1] > outer_cnt[:, 0, 1].min(),
                                  inner_cnt[:, 0, 1] - shrink_px,
                                  inner_cnt[:, 0, 1] + shrink_px)
    inner_cnt = np.intp(inner_cnt)

    return outer_cnt, inner_cnt

# 返回结果，结果为较大矩形四个角的像素坐标（原始坐标，在original.jpg中的坐标）和较大矩形宽度，长度以及黑色边框的像素宽度
# 黑色边框像素宽度计算较小矩形四个角的像素坐标和较大矩形四个角的像素坐标的距离的1/根号2，取平均值
def calculate_dimensions(outer_cnt: np.ndarray,
                          inner_cnt: np.ndarray,
                          crop_coords: tuple):
    """
    计算外框四角绝对坐标、宽/高以及黑色边框像素宽度。

    参数
    ----
    outer_cnt, inner_cnt : (4,1,2) np.ndarray
        optimize_rectangles 输出的外/内矩形四点轮廓（相对裁剪图）。
    crop_coords : tuple
        CROP_COORDS = (y1, y2, x1, x2)，用于把局部坐标还原为原图坐标。

    返回
    ----
    corners_global : (4,2) np.ndarray[int]
        外框四角在 original.jpg 中的坐标，按 TL-TR-BR-BL 顺序。
    width_px  : float
        外框短边像素长度（像素单位）。
    height_px : float
        外框长边像素长度（像素单位）。
    border_px : float
        黑色边框平均像素宽度（像素单位）。
    """
    # ---------- 1. 工具函数：四点排序 ----------
    def order_corners(pts: np.ndarray):
        """将任意 4×2 点按 TL-TR-BR-BL 顺时针排序"""
        pts = pts.astype(np.float32)
        s = pts.sum(axis=1)            # x+y
        diff = np.diff(pts, axis=1)    # x-y
        tl = np.argmin(s)
        br = np.argmax(s)
        tr = np.argmin(diff)
        bl = np.argmax(diff)
        return np.array([pts[tl], pts[tr], pts[br], pts[bl]], dtype=np.float32)

    # ---------- 2. 坐标还原到原图 ----------
    y1, _, x1, _ = crop_coords
    offset = np.array([x1, y1], dtype=np.int32)
    outer_glb = (outer_cnt.reshape(-1, 2) + offset).astype(np.int32)
    inner_glb = (inner_cnt.reshape(-1, 2) + offset).astype(np.int32)

    outer_ord = order_corners(outer_glb)   # (4,2) float32
    inner_ord = order_corners(inner_glb)

    # ---------- 3. 外框宽高 ----------
    # 连续点两两距离：0-1, 1-2, 2-3, 3-0
    dists = [np.linalg.norm(outer_ord[i] - outer_ord[(i + 1) % 4]) for i in range(4)]
    dists_sorted = sorted(dists)  # 前两条为短边，后两条为长边
    width_px  = (dists_sorted[0] + dists_sorted[1]) / 2.0
    height_px = (dists_sorted[2] + dists_sorted[3]) / 2.0

    # ---------- 4. 黑框宽度 ----------
    borders = [
        np.linalg.norm(outer_ord[i] - inner_ord[i]) / np.sqrt(2.0)
        for i in range(4)
    ]
    border_px = float(np.mean(borders))

    return outer_ord.astype(np.int32), float(width_px), float(height_px), border_px


# 计算透视变换矩阵，将原图裁剪为 A4 纸的视角，像素精度不变
def calculate_perspective_transform(outer_cnt: np.ndarray,
                                    crop_coords: tuple,
                                    image_path: str,
                                    save_path: str = "warped_a4.jpg"):
    """
    计算透视变换矩阵，并把 original.jpg 矫正裁剪成 A4 纸俯视图。

    参数
    ----
    outer_cnt : (4,1,2) np.ndarray
        optimize_rectangles 输出的外框（相对裁剪图坐标）。
    crop_coords : tuple
        CROP_COORDS = (y1, y2, x1, x2)，用于还原到原图坐标。
    image_path : str
        原始图像路径 (original.jpg)。
    save_path : str
        矫正后图片保存路径。

    返回
    ----
    M : (3,3) np.ndarray[float]
        透视变换矩阵，使得 dst = cv2.warpPerspective(src, M, (W, H))。
    size : tuple[int, int]
        (W, H) —— 矫正图像的宽、高像素尺寸。
    """
    # ---------- 1. 将四点还原到原图坐标 ----------
    y1, _, x1, _ = crop_coords
    offset = np.array([x1, y1], dtype=np.float32)
    src_pts = outer_cnt.reshape(-1, 2).astype(np.float32) + offset  # (4,2)

    # ---------- 2. 统一顶点顺序 (TL-TR-BR-BL) ----------
    def order_corners(pts):
        s = pts.sum(axis=1)           # x+y
        diff = np.diff(pts, axis=1)   # x-y
        tl = np.argmin(s)
        br = np.argmax(s)
        tr = np.argmin(diff)
        bl = np.argmax(diff)
        return np.array([pts[tl], pts[tr], pts[br], pts[bl]], dtype=np.float32)

    src_ordered = order_corners(src_pts)

    # ---------- 3. 计算目标平面大小 ----------
    width_top  = np.linalg.norm(src_ordered[1] - src_ordered[0])
    width_bot  = np.linalg.norm(src_ordered[2] - src_ordered[3])
    height_lft = np.linalg.norm(src_ordered[3] - src_ordered[0])
    height_rgt = np.linalg.norm(src_ordered[2] - src_ordered[1])
    max_w = int(round(max(width_top,  width_bot)))
    max_h = int(round(max(height_lft, height_rgt)))

    # ---------- 4. 构造目标四点 ----------
    dst_ordered = np.array([
        [0,      0],          # TL
        [max_w-1, 0],         # TR
        [max_w-1, max_h-1],   # BR
        [0,      max_h-1]     # BL
    ], dtype=np.float32)

    # ---------- 5. 计算透视矩阵并矫正 ----------
    M = cv2.getPerspectiveTransform(src_ordered, dst_ordered)
    img = cv2.imread(image_path)
    warped = cv2.warpPerspective(img, M, (max_w, max_h))
    cv2.imwrite(save_path, warped)
    print(f"[Step-8] 已保存透视矫正图 → {save_path}")

    return M, (max_w, max_h)

# 对透视后得到的A4进行预处理，首先计算整个图像的灰度平均值，根据该平均值进行2值化处理
# ================== 透视后 A4 预处理 ==================
def preprocess_a4_image(a4_img: np.ndarray,
                        save_prefix: str = f"{DEBUG_PREFIX}9_a4_",
                        auto_invert: bool = True,
                        morph_kernel: int = 3,
                        morph_iter: int = 1):
    """
    对透视矫正后的 A4 图像做灰度均值阈值二值化，并可选进行形态学清理。

    步骤
    ----
    1. 计算整幅图的灰度平均值 `mean_gray` 作为阈值。
    2. 根据 `auto_invert` 自动选择 THRESH_BINARY 或 THRESH_BINARY_INV：
       • 若 `mean_gray` < 128，说明整体偏暗，以 BLACK=0, WHITE=255 的正向阈值更合适；
       • 否则取反阈值，使黑色内容为白、背景为黑，方便后续分析。
    3. 可选：用矩形核( `morph_kernel`) 做一次闭运算 + 开运算去噪。
    4. 保存中间结果（灰度、二值、morph）到硬盘，返回最终二值图及阈值。

    参数
    ----
    a4_img        : np.ndarray  —— cv2 读取的 BGR 图像 (透视矫正后的 A4)。
    save_prefix   : str         —— 调试图像文件名前缀。
    auto_invert   : bool        —— 是否根据平均灰度自动决定阈值类型。
    morph_kernel  : int         —— 形态学核尺寸(像素)；<=0 表示跳过形态学。
    morph_iter    : int         —— 形态学迭代次数。

    返回
    ----
    binary_final  : np.ndarray  —— 处理后的二值化结果 (单通道 0/255)。
    mean_gray     : float       —— 全图灰度平均值，用作阈值。
    """
    # 1. 灰度图 & 平均值
    gray = cv2.cvtColor(a4_img, cv2.COLOR_BGR2GRAY)
    mean_gray = np.mean(gray)
    cv2.imwrite(f"{save_prefix}gray.jpg", gray)

    # 2. 根据平均值自动选择阈值方式
    if auto_invert:
        if mean_gray < 128:
            thresh_type = cv2.THRESH_BINARY        # 背景白，内容黑
        else:
            thresh_type = cv2.THRESH_BINARY_INV    # 背景黑，内容白
    else:
        thresh_type = cv2.THRESH_BINARY

    _, binary = cv2.threshold(gray,
                              int(mean_gray),
                              255,
                              thresh_type)
    cv2.imwrite(f"{save_prefix}binary_raw.jpg", binary)

    # 3. 形态学清理（可选）
    if morph_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (morph_kernel, morph_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                  kernel, iterations=morph_iter)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                  kernel, iterations=morph_iter)
        cv2.imwrite(f"{save_prefix}binary_morph.jpg", binary)

    return binary, mean_gray


# 封装整个检测过程，调用函数时，自动拍摄一张图片然后返回被外框四角，黑边平均宽度，校正后的 A4 图像信息
def detect_rectangles(crop_coords: tuple = CROP_COORDS,
                      debug_prefix: str = DEBUG_PREFIX,
                      save_original: bool = True):
    """
    一键完成：拍照 → 检测 A4 黑边矩形 → 透视矫正 → 二值预处理，
    并返回核心几何信息与路径。

    参数
    ----
    crop_coords   : tuple  (y1, y2, x1, x2) – ROI 裁剪区域。
    debug_prefix  : str    – 输出调试文件名前缀。
    save_original : bool   – 是否把原始拍摄存盘到 `IMAGE_PATH`。

    返回
    ----
    result : dict
        {
          "outer_corners" : 4×2 np.ndarray[int],  # 外框四角(原图坐标，TL‑TR‑BR‑BL)
          "width_px"      : float,                # 外框短边像素
          "height_px"     : float,                # 外框长边像素
          "border_px"     : float,                # 黑边平均像素宽度
          "warped_path"   : str,                  # 透视矫正后的 A4 图像路径
          "binary_path"   : str,                  # A4 二值图（可选形态学）路径
          "mean_gray"     : float,                # A4 灰度均值阈值
          "M"             : 3×3 np.ndarray[float] # 透视矩阵
        }
    """
    # ---------- 0. 拍照并保存 ----------
    original = get_image.capture_image()
    if save_original:
        cv2.imwrite(IMAGE_PATH, original)

    # ---------- 1. ROI 裁剪 + 预处理 ----------
    cropped = read_and_crop(IMAGE_PATH, crop_coords)
    binary  = binarize_image(cropped, THRESHOLD_VALUE, DARK_THRESHOLD)
    binary_clean = morphology_clean(binary, MORPH_KERNEL_SIZE, MORPH_ITERATIONS)

    # ---------- 2. 轮廓 & 嵌套矩形 ----------
    contours = find_candidate_contours(binary_clean)
    outer, inner = select_nested_rectangles(contours,debug=0)
    outer_opt, inner_opt = optimize_rectangles(outer, inner)

    # ---------- 3. 尺寸 / 黑边 ----------
    outer_corners, w_px, h_px, border_px = calculate_dimensions(
        outer_opt, inner_opt, crop_coords
    )

    # ---------- 4. 透视矫正 ----------
    warped_path = f"{debug_prefix}8_warped_a4.jpg"
    M, (W, H) = calculate_perspective_transform(
        outer_opt, crop_coords, IMAGE_PATH, warped_path
    )

    # ---------- 5. A4 二值预处理 ----------
    warped_img = cv2.imread(warped_path)
    binary_a4, mean_g = preprocess_a4_image(
        warped_img,
        save_prefix=f"{debug_prefix}9_a4_",
        auto_invert=True,
        morph_kernel=3,
        morph_iter=1
    )
    binary_path = f"{debug_prefix}9_a4_binary_morph.jpg"  # 与 preprocess 保存一致

    # ---------- 6. 汇总结果 ----------
    result = {
        "outer_corners": outer_corners,  # np.ndarray (4,2)
        "width_px"     : w_px,
        "height_px"    : h_px,
        "border_px"    : border_px,
        "warped_path"  : warped_path,
        "binary_path"  : binary_path,
        "mean_gray"    : mean_g,
        "M"            : M
    }
    return result



# ================== 主流程 ==================

def main():
    cropped = read_and_crop(IMAGE_PATH, CROP_COORDS)
    cv2.imwrite(f"{DEBUG_PREFIX}1_cropped.jpg", cropped)

    binary = binarize_image(cropped, THRESHOLD_VALUE, DARK_THRESHOLD)
    cv2.imwrite(f"{DEBUG_PREFIX}2_binary_rgb_filtered.jpg", binary)
    print("[Step‑2] 二值化处理完成 → step_2_binary_rgb_filtered.jpg")

    binary_clean = morphology_clean(binary, MORPH_KERNEL_SIZE, MORPH_ITERATIONS)
    cv2.imwrite(f"{DEBUG_PREFIX}3_binary_clean.jpg", binary_clean)
    print("[Step‑3] 形态学处理完成 → step_3_binary_clean.jpg")

    # Step‑4: 提取候选轮廓并保存可视化（调试可选）
    contours = find_candidate_contours(binary_clean)
    vis = cropped.copy()
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(f"{DEBUG_PREFIX}4_contours.jpg", vis)
    print(f"[Step‑4] 提取到 {len(contours)} 个候选轮廓 → step_4_contours.jpg")


    # Step‑5: 选择嵌套外/内矩形并一次性绘制到同一张图
    outer, inner = select_nested_rectangles(contours,debug=1)
    vis_pair = cropped.copy()
    cv2.drawContours(vis_pair, [outer], -1, (255, 0, 0), 3)  # 外矩形：蓝色
    cv2.drawContours(vis_pair, [inner], -1, (0, 0, 255), 3)  # 内矩形：红色
    cv2.imwrite(f"{DEBUG_PREFIX}5_selected_pair.jpg", vis_pair)
    print("[Step‑5] 已绘制外/内矩形 → step_5_selected_pair.jpg")


    # Step‑6: 优化外/内矩形为理想矩形
    outer_opt, inner_opt = optimize_rectangles(outer, inner)
    vis_opt = cropped.copy()
    cv2.drawContours(vis_opt, [outer_opt], -1, (255, 0, 0), 3)  # 外矩形：蓝色
    cv2.drawContours(vis_opt, [inner_opt], -1, (0, 0, 255), 3)  # 内矩形：红色
    cv2.imwrite(f"{DEBUG_PREFIX}6_optimized_pair.jpg", vis_opt)
    print("[Step‑6] 已优化外/内矩形为理想矩形 → step_6_optimized_pair.jpg")
    

    # Step-7: 计算尺寸与黑边宽度，绘制在IMAGE_PATH后输出
    outer_corners, w_px, h_px, border_px = calculate_dimensions(
        outer_opt, inner_opt, CROP_COORDS
    )
    print("外框四角 (原图坐标):", outer_corners.tolist())
    print(f"外框宽度:  {w_px:.2f} px")
    print(f"外框高度:  {h_px:.2f} px")
    print(f"黑边平均宽度: {border_px:.2f} px")
    # 在原图上绘制外框四角
    img_with_corners = cv2.imread(IMAGE_PATH)
    for i, corner in enumerate(outer_corners):
        cv2.circle(img_with_corners, tuple(corner), 10, (0, 255, 0), -1)
        cv2.putText(img_with_corners, f"Corner {i+1}", tuple(corner + [10, 10]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(f"{DEBUG_PREFIX}7_final_output.jpg", img_with_corners)
    print("[Step‑7] 已绘制外框四角并保存 → step_7_final_output.jpg")

    # Step-8: 透视矫正，输出 A4 俯视图
    M, (W, H) = calculate_perspective_transform(
        outer_opt, CROP_COORDS, IMAGE_PATH, f"{DEBUG_PREFIX}8_warped_a4.jpg"
    )
    print("透视矩阵:\n", M)
    print(f"矫正后尺寸: {W} × {H} px")

    # Step-9: 预处理 A4 图像
    # Step-9: 对矫正后的 A4 做二值化预处理
    warped_img = cv2.imread(f"{DEBUG_PREFIX}8_warped_a4.jpg")
    binary_a4, mean_g = preprocess_a4_image(
        warped_img,
        save_prefix=f"{DEBUG_PREFIX}9_a4_",
        auto_invert=True,
        morph_kernel=3,
        morph_iter=1
    )
    print(f"Step-9: A4 预处理完成, 灰度均值阈值 = {mean_g:.2f}")



    


if __name__ == "__main__":
    main()

    # info = detect_rectangles()
    # print("检测结果：")
    # print("外框四角:", info["outer_corners"].tolist())
    # print("宽/高(px):", info["width_px"], "/", info["height_px"])
    # print("黑边平均宽度(px):", info["border_px"])
    # print("A4 灰度均值阈值:", info["mean_gray"])
    # print("透视矩阵:\n", info["M"])
    # print("矫正图保存于:", info["warped_path"])
