import os
import math
import test_detect_rect

# 定义全局常量：A4纸实际尺寸 (毫米)
A4_WIDTH_MM = 210.0   # A4宽度 21.0 cm
A4_HEIGHT_MM = 297.0  # A4长度 29.7 cm
known_distance_mm = 1327.0  # 校准时使用的已知距离（毫米），默认1米

# 校准文件路径
CALIB_FILE = "camera_calibration.txt"

def calibrate_distance(known_distance_mm):
    """
    利用已知距离的A4纸进行摄像头焦距标定。
    参数:
        known_distance_mm: 校准物体距离（毫米），默认1000mm即1米。
    返回:
        focal_px: 计算得到的摄像头等效焦距（像素）。
    """
    # 获取检测信息
    info = test_detect_rect.detect_rectangles()
    if info is None:
        raise RuntimeError("检测失败：未能获得A4纸角点信息")
    width_px = info["width_px"]
    height_px = info["height_px"]
    border_px = info.get("border_px", None)
    # 确保 width_px 对应物理短边，height_px 对应长边（如有必要交换）
    if width_px > height_px:
        width_px, height_px = height_px, width_px  # 交换使宽对应短边

    # 计算基于宽和高的焦距（像素）
    focal_from_width = (width_px * known_distance_mm) / A4_WIDTH_MM
    focal_from_height = (height_px * known_distance_mm) / A4_HEIGHT_MM
    # 取平均提高稳定性
    focal_px = (focal_from_width + focal_from_height) / 2.0

    # 将焦距参数保存到文件
    try:
        with open(CALIB_FILE, "w") as f:
            f.write(f"{focal_px:.6f}")
    except Exception as e:
        print("保存校准文件失败:", e)
        raise

    # 打印标定结果
    print("=== 相机标定完成 ===")
    print(f"检测像素宽度: {width_px:.3f}px, 像素高度: {height_px:.3f}px")
    if border_px is not None:
        print(f"检测黑边平均宽度: {border_px:.3f}px (物理宽度2cm)")
    print(f"标定距离: {known_distance_mm/10:.1f}cm")
    print(f"计算得到焦距F: {focal_px:.2f}px")
    print(f"参数已保存至: {CALIB_FILE}")
    return focal_px

def estimate_distance():
    """
    利用之前标定的焦距参数，测量当前图像中A4纸的距离。
    返回:
        distance_mm: 测量的距离（毫米）。
    """
    # 检查校准文件是否存在
    if not os.path.exists(CALIB_FILE):
        raise RuntimeError("未找到校准参数文件，请先运行校准。")
    # 读取保存的焦距（像素）
    with open(CALIB_FILE, "r") as f:
        focal_px_str = f.readline().strip()
        if not focal_px_str:
            raise RuntimeError("校准文件内容无效！")
        focal_px = float(focal_px_str)

    # 获取检测信息
    info = test_detect_rect.detect_rectangles()
    if info is None:
        raise RuntimeError("检测失败：未能获得A4纸角点信息")
    width_px = info["width_px"]
    height_px = info["height_px"]
    # 确保对应正确的物理尺寸
    if width_px > height_px:
        width_px, height_px = height_px, width_px

    # 根据宽度和高度分别计算距离
    distance_from_width = (A4_WIDTH_MM * focal_px) / width_px   # mm
    distance_from_height = (A4_HEIGHT_MM * focal_px) / height_px  # mm
    distance_mm = (distance_from_width + distance_from_height) / 2.0

    # 保存infor返回的图像
    info["warped_path"] = "warped_a4.jpg"

    # 输出结果
    print("=== 测距结果 ===")
    print(f"像素宽度: {width_px:.3f}px, 像素高度: {height_px:.3f}px")
    # 转换为厘米输出，保留1位小数
    distance_cm = distance_mm / 10.0
    print(f"距离: {distance_cm:.1f} cm")
    return distance_mm

# 主程序：如果直接运行本模块，则执行标定或测距
if __name__ == "__main__":
    # 如果不存在校准文件，则进行标定
    if not os.path.exists(CALIB_FILE):
        print("未找到校准文件，开始执行标定...")
        calibrate_distance(known_distance_mm= known_distance_mm)
    # 若已标定，则直接测距
    else:
        print("检测到已有校准参数，直接测量当前距离...")
        estimate_distance()
