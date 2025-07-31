import os
import math
import test_detect_rect

# 定义全局常量：A4纸实际尺寸 (毫米)
A4_WIDTH_MM = 210.0   # A4宽度 21.0 cm
A4_HEIGHT_MM = 297.0  # A4长度 29.7 cm
known_distance_mm = 1850.0  # 校准时使用的已知距离（毫米），默认1米

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

    # 将焦距参数和黑边厚度一起保存到校准文件（追加模式）
    try:
        content = f"{focal_px:.6f}"
        if border_px is not None:
            content += f" {border_px:.6f}"
        # 以追加模式写入（若文件不存在则创建，存在则在末尾添加）
        with open(CALIB_FILE, "a") as f:
            # 每条记录写在新行
            f.write(content + "\n")
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
    # 读取所有保存的焦距参数及对应黑边厚度
    calibrations = []
    with open(CALIB_FILE, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # 如果此行同时包含焦距和黑边厚度
            if len(parts) >= 2:
                try:
                    focal_val = float(parts[0])
                    border_val = float(parts[1])
                    calibrations.append((border_val, focal_val))
                except:
                    continue  # 跳过无法解析的行
            elif len(parts) == 1:
                # 仅包含焦距（旧格式）
                try:
                    focal_val = float(parts[0])
                    # 没有黑边厚度信息，暂存但不用于选择最近参数
                    calibrations.append((None, focal_val))
                except:
                    continue
    if not calibrations:
        raise RuntimeError("校准文件内容无效！")
    # 获取当前图像的检测信息
    info = test_detect_rect.detect_rectangles()
    if info is None:
        raise RuntimeError("检测失败：未能获得A4纸角点信息")
    width_px = info["width_px"]
    height_px = info["height_px"]
    border_px = info.get("border_px", None)
    # 确保对应正确的物理尺寸
    if width_px > height_px:
        width_px, height_px = height_px, width_px

    # 根据当前黑边厚度，选择最接近的标定参数
    if border_px is not None:
        closest_focal = None
        min_diff = float('inf')
        for (border_val, focal_val) in calibrations:
            if border_val is None:
                continue  # 跳过没有黑边厚度的旧数据
            diff = abs(border_val - border_px)
            if diff < min_diff:
                min_diff = diff
                closest_focal = focal_val
        if closest_focal is None:
            # 如果所有记录都没有黑边厚度（不应出现此情况）
            raise RuntimeError("校准参数缺少黑边厚度信息，无法选择最近参数！")
        focal_px = closest_focal
    else:
        # 未检测到黑边厚度，无法进行匹配选择
        raise RuntimeError("检测失败：未能获得黑边厚度，无法选择最近参数进行测距")

    # 使用选定的焦距计算距离
    distance_from_width = (A4_WIDTH_MM * focal_px) / width_px   # mm
    distance_from_height = (A4_HEIGHT_MM * focal_px) / height_px  # mm
    distance_mm = (distance_from_width + distance_from_height) / 2.0

    # 保存info返回的图像路径
    info["warped_path"] = "warped_a4.jpg"

    # 输出结果
    print("=== 测距结果 ===")
    print(f"像素宽度: {width_px:.3f}px, 像素高度: {height_px:.3f}px")
    # 转换为厘米输出，保留1位小数
    distance_cm = distance_mm / 10.0
    print(f"距离: {distance_cm:.1f} cm")
    return distance_mm

# 主程序：如果直接运行本模块，则执行标定
if __name__ == "__main__":
    # 总是执行标定（多点标定模式）
    # print("开始执行标定...")
    # calibrate_distance(known_distance_mm= known_distance_mm)
    # 测距
    print("开始测距...")
    distance_mm = estimate_distance()
    print(f"测距结果: {distance_mm:.2f} mm")

    # 注意：标定至少一次后可调用 estimate_distance() 进行测距。
