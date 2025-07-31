import cv2
import logging
import time
import math
import numpy as np
from picamera2 import Picamera2
# 导入自定义模块
import detectrect
import detect_laser
import xy_to_plus

# ==== 可配置参数 ====
FRAME_RATE = 15          # 帧率（FPS）
PID_KP = 0.5             # PID 比例系数
PID_KI = 0.1             # PID 积分系数
PID_KD = 0.05            # PID 微分系数
PREVIEW_RES = (640, 480) # 连续跟踪使用的分辨率
CALIBRATION_RES = (2592, 2592)  # 标定用高分辨率照片

# 卡尔曼滤波参数
KF_PROCESS_VAR = 50.0    # 过程噪声方差（加速度不确定性）
KF_MEASURE_VAR = 4.0     # 测量噪声方差

logging.basicConfig(level=logging.INFO, format='%(message)s')

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = None
        self.integral = 0.0

    def reset(self):
        self.prev_error = None
        self.integral = 0.0

    def compute(self, error: float, dt: float) -> float:
        self.integral += error * dt
        derivative = 0.0
        if self.prev_error is not None:
            derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class KalmanFilter1D:
    def __init__(self, initial_pos: float, initial_vel: float = 0.0,
                 process_var: float = 1.0, meas_var: float = 1.0):
        self.x = np.array([[initial_pos], [initial_vel]], dtype=float)
        self.P = np.eye(2, dtype=float) * 1e-3
        self.H = np.array([[1.0, 0.0]], dtype=float)
        self.R = np.array([[meas_var]], dtype=float)
        self.process_var = process_var

    def predict(self, dt: float):
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
        Q = np.array([[ (dt**4)/4.0, (dt**3)/2.0 ],
                      [ (dt**3)/2.0, (dt**2)      ]], dtype=float) * self.process_var
        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T) + Q

    def update(self, measured_pos: float):
        z = np.array([[measured_pos]], dtype=float)
        y = z - self.H.dot(self.x)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        I = np.eye(2, dtype=float)
        self.P = (I - K.dot(self.H)).dot(self.P)

def main():
    # 标定：拍摄高分辨率图像以获取透视变换
    try:
        cam_calib = Picamera2()
        cam_calib.configure(cam_calib.create_still_configuration(
            main={"format": "RGB888", "size": CALIBRATION_RES}
        ))
        cam_calib.start()
        time.sleep(0.5)
        frame_rgb = cam_calib.capture_array()
        cam_calib.close()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"Failed to capture calibration frame: {e}")
        return

    # 检测标定矩形获取透视变换
    corners, M = detectrect.detect_rectangle(frame_bgr)
    if M is None:
        logging.warning("Calibration rectangle not found. Using identity transform.")
        M = np.eye(3, dtype=np.float32)
    else:
        logging.info(f"Calibration successful, perspective transform matrix obtained.")

    # 打开相机预览模式获取图像
    try:
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(
            main={"format": "RGB888", "size": PREVIEW_RES}
        ))
        cam.start()
        time.sleep(0.2)
    except Exception as e:
        logging.error(f"Failed to open camera for tracking: {e}")
        return

    # 初始检测两个激光点（红色目标和绿色指示）
    green_pos = None
    red_pos = None
    for _ in range(10):
        try:
            frame_rgb = cam.capture_array()
        except Exception as e:
            logging.warning(f"Camera capture error during initial detection: {e}")
            continue
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        g = detect_laser.detect_green(frame_bgr, M)
        r = detect_laser.detect_red(frame_bgr, M)
        if g is not None:
            green_pos = g
        if r is not None:
            red_pos = r
        if green_pos is not None and red_pos is not None:
            break
        time.sleep(0.05)
    if green_pos is None or red_pos is None:
        logging.error("Initial laser positions not found (Green or Red missing).")
        cam.close()
        return

    # 将舵机初始位置设为绿光初始坐标 
    current_x, current_y = green_pos
    target_x, target_y = red_pos
    servo_x_cmd = current_x
    servo_y_cmd = current_y
    # 移动舵机到当前绿光位置（若绿光本就在当前舵机指向则无需移动）
    xy_to_plus.set_xy(servo_x_cmd, servo_y_cmd, 50)
    # 初始化 PID 控制器
    pid_x = PIDController(PID_KP, PID_KI, PID_KD)
    pid_y = PIDController(PID_KP, PID_KI, PID_KD)
    # 初始化绿光和红光位置的卡尔曼滤波器
    kf_green_x = KalmanFilter1D(initial_pos=current_x, initial_vel=0.0,
                                process_var=KF_PROCESS_VAR, meas_var=KF_MEASURE_VAR)
    kf_green_y = KalmanFilter1D(initial_pos=current_y, initial_vel=0.0,
                                process_var=KF_PROCESS_VAR, meas_var=KF_MEASURE_VAR)
    kf_red_x = KalmanFilter1D(initial_pos=target_x, initial_vel=0.0,
                              process_var=KF_PROCESS_VAR, meas_var=KF_MEASURE_VAR)
    kf_red_y = KalmanFilter1D(initial_pos=target_y, initial_vel=0.0,
                              process_var=KF_PROCESS_VAR, meas_var=KF_MEASURE_VAR)

    start_time = time.time()
    last_time = None
    lost_red_frames = 0
    lost_green_frames = 0

    # 主控制循环（30 秒）
    while time.time() - start_time < 30:
        try:
            frame_rgb = cam.capture_array()
        except Exception as e:
            logging.warning(f"Camera capture error: {e}")
            break
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        current_time = time.time()
        if last_time is None:
            dt = 1.0 / FRAME_RATE
        else:
            dt = current_time - last_time
        if dt <= 0 or dt > 0.2:
            dt = 1.0 / FRAME_RATE

        pos_green = detect_laser.detect_green(frame_bgr, M)
        pos_red = detect_laser.detect_red(frame_bgr, M)

        # 更新丢失帧计数器
        if pos_red is None:
            lost_red_frames += 1
        else:
            lost_red_frames = 0
        if pos_green is None:
            lost_green_frames += 1
        else:
            lost_green_frames = 0

        # 处理单个或两个激光点丢失的情况
        if pos_green is None:
            # 绿色激光点（当前指示点）丢失 - 无法在未知当前指针位置时控制
            logging.warning("Green laser spot lost.")
            # 滤波器状态预测
            kf_green_x.predict(dt)
            kf_green_y.predict(dt)
            if pos_red is not None:
                # 更新红光滤波器（因为有红光测量值）
                kf_red_x.predict(dt)
                kf_red_y.predict(dt)
                kf_red_x.update(pos_red[0])
                kf_red_y.update(pos_red[1])
            else:
                # 两者都丢失，则仅预测红光
                kf_red_x.predict(dt)
                kf_red_y.predict(dt)
            # 跳过此循环且不改变舵机指令
            last_time = current_time
            continue

        # 现在已知绿色光点位置（pos_green 非空）
        current_x, current_y = pos_green

        if pos_red is None:
            # 红色目标丢失，短时间使用预测继续跟踪
            logging.warning("Red laser spot lost, predicting target position.")
            # 预测目标位置前进
            kf_red_x.predict(dt)
            kf_red_y.predict(dt)
            # 红光无测量更新
            # 用测量值更新绿光滤波器
            kf_green_x.predict(dt)
            kf_green_y.predict(dt)
            kf_green_x.update(current_x)
            kf_green_y.update(current_y)
            # 获取滤波估计值
            filtered_rx = float(kf_red_x.x[0])
            filtered_ry = float(kf_red_y.x[0])
            filtered_gx = float(kf_green_x.x[0])
            filtered_gy = float(kf_green_y.x[0])
            # 用预测的目标位置计算误差
            err_x = filtered_rx - filtered_gx
            err_y = filtered_ry - filtered_gy
            # 如目标长时间丢失，则停止移动以避免发散
            if lost_red_frames > 5:
                # 保持舵机位置，等待目标重新出现
                last_time = current_time
                continue
        else:
            # 两个激光点都可见，用测量值更新两者滤波器
            red_x, red_y = pos_red
            kf_red_x.predict(dt)
            kf_red_y.predict(dt)
            kf_green_x.predict(dt)
            kf_green_y.predict(dt)
            kf_red_x.update(red_x)
            kf_red_y.update(red_y)
            kf_green_x.update(current_x)
            kf_green_y.update(current_y)
            # 滤波后的位置
            filtered_rx = float(kf_red_x.x[0])
            filtered_ry = float(kf_red_y.x[0])
            filtered_gx = float(kf_green_x.x[0])
            filtered_gy = float(kf_green_y.x[0])
            # 位置误差（绿点追踪红点）
            err_x = filtered_rx - filtered_gx
            err_y = filtered_ry - filtered_gy

        # 计算 PID 控制输出
        output_x = pid_x.compute(err_x, dt)
        output_y = pid_y.compute(err_y, dt)
        # 更新舵机命令（并限制在 0-100 范围）
        servo_x_cmd += output_x
        servo_y_cmd += output_y
        if servo_x_cmd < 0.0: servo_x_cmd = 0.0
        if servo_x_cmd > 100.0: servo_x_cmd = 100.0
        if servo_y_cmd < 0.0: servo_y_cmd = 0.0
        if servo_y_cmd > 100.0: servo_y_cmd = 100.0

        # 计算舵机移动时间
        step = math.hypot(output_x, output_y)
        move_ms = int(max(30.0, min(60.0, 4.5 * step)))
        xy_to_plus.set_xy(servo_x_cmd, servo_y_cmd, move_ms)

        # 日志输出当前状态
        logging.info(f"Err: ({err_x:.1f}, {err_y:.1f}), "
                     f"Target: ({filtered_rx:.1f}, {filtered_ry:.1f}), "
                     f"Pos: ({filtered_gx:.1f}, {filtered_gy:.1f}), "
                     f"ServoCmd: ({servo_x_cmd:.1f}, {servo_y_cmd:.1f})")

        # 更新时间
        last_time = current_time

        # 帧率控制
        loop_time = time.time() - current_time
        sleep_time = (1.0 / FRAME_RATE) - loop_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    cam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
