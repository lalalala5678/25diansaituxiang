import cv2
import numpy as np
import test_detect_rect

# 识别图形并计算实际尺寸函数
def recognize_shapes_and_measure():

    print("开始运行 recognize_shapes_and_measure() ...")
        
        # 获取校正后的图像和信息
    print("正在调用 test_detect_rect.detect_rectangles()...")
    result = test_detect_rect.detect_rectangles()
    print(f"检测结果: {result}")
        
    warped_img_path = result["warped_path"]
    width_px, height_px = result["width_px"], result["height_px"]
        
        # 检测图像读取情况
    warped_img = cv2.imread(warped_img_path)
    if warped_img is None:
        print(f"无法读取图像文件: {warped_img_path}")
        return

    print("图像读取成功，开始处理...")

    # 获取校正后的图像和信息
    result = test_detect_rect.detect_rectangles()
    warped_img_path = result["warped_path"]
    width_px, height_px = result["width_px"], result["height_px"]

    # 读取校正图像
    warped_img = cv2.imread(warped_img_path)
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 计算像素到实际尺寸的比例（单位：像素/cm）
    scale = ((width_px / 21.0) + (height_px / 29.7)) / 2.0

    # 轮廓检测
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 1000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        vertices = approx.reshape(-1, 2)
        num_vertices = len(vertices)

        shape_type, real_size_cm = None, None

        if num_vertices == 3:
            sides = [np.linalg.norm(vertices[i] - vertices[(i+1)%3]) for i in range(3)]
            if max(sides) - min(sides) < 0.1 * np.mean(sides):
                shape_type = "Equilateral Triangle"
                real_size_cm = np.mean(sides) / scale

        elif num_vertices == 4:
            sides = [np.linalg.norm(vertices[i] - vertices[(i+1)%4]) for i in range(4)]
            side_diff = max(sides) - min(sides)
            angles = []
            for i in range(4):
                vec1 = vertices[i] - vertices[i-1]
                vec2 = vertices[(i+1)%4] - vertices[i]
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                angles.append(np.arccos(cos_angle) * 180 / np.pi)
            if side_diff < 0.1 * np.mean(sides) and all(abs(a-90)<10 for a in angles):
                shape_type = "Square"
                real_size_cm = np.mean(sides) / scale

        else:
            perimeter = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > 0.8:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                shape_type = "Circle"
                real_size_cm = (2 * radius) / scale

        if shape_type and real_size_cm:
            results.append({"shape": shape_type, "size_cm": round(real_size_cm, 2)})
            cv2.drawContours(warped_img, [cnt], -1, (0, 255, 0), 2)
            cx, cy = np.mean(vertices, axis=0).astype(int)
            cv2.putText(warped_img, f"{shape_type}: {real_size_cm:.2f}cm", (cx-50, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imwrite("measured_shapes.jpg", warped_img)

    # 输出结果
    for res in results:
        print(f"Detected {res['shape']} with size: {res['size_cm']} cm")

    return results

# 调用函数
if __name__ == "__main__":
    print(">>> 开始执行图形识别程序...")
    recognize_shapes_and_measure()
    print(">>> 程序执行结束")
