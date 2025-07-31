import test_detect_rect
import cv2

info = test_detect_rect.detect_rectangles()

print("检测结果：")
print("外框四角:", info["outer_corners"].tolist())
print("宽/高(px):", info["width_px"], "/", info["height_px"])
print("黑边平均宽度(px):", info["border_px"])
print("A4 灰度均值阈值:", info["mean_gray"])
print("透视矩阵:\n", info["M"])
print("矫正图保存于:", info["warped_path"])