import cv2
import numpy as np
from collections import deque
import math

# Настройки
POINT_RADIUS = 8
TRAJECTORY_COLOR = (0, 255, 0)
CURSOR_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
MIN_AREA = 100
INFO_PANEL_WIDTH = 300
SPEED_WINDOW_SECONDS = 2  # окно усреднения скорости (сек)

trajectory = deque()

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def main():
    cap = cv2.VideoCapture("v.mp4")
    if not cap.isOpened():
        print("Ошибка открытия видео!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # fallback
    frame_time = 1.0 / fps
    speed_window_size = int(SPEED_WINDOW_SECONDS * fps)

    cv2.namedWindow("Object Tracking", cv2.WINDOW_NORMAL)

    ret, prev_frame = cap.read()
    if not ret:
        print("Не удалось прочитать первый кадр.")
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    last_center = None
    total_distance = 0
    elapsed_time = 0

    recent_points = deque(maxlen=speed_window_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        center = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > MIN_AREA:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    center = (cx, cy)

        if center is not None:
            if last_center:
                d = distance(last_center, center)
                total_distance += d
            last_center = center
            trajectory.append(center)
            recent_points.append(center)
        elif last_center:
            trajectory.append(last_center)
            recent_points.append(last_center)

        # Рисуем траекторию
        for i in range(1, len(trajectory)):
            if trajectory[i - 1] and trajectory[i]:
                cv2.line(frame, trajectory[i - 1], trajectory[i], TRAJECTORY_COLOR, 2)

        # Рисуем текущую позицию
        if last_center:
            cv2.circle(frame, last_center, POINT_RADIUS, CURSOR_COLOR, 2)

        # Расчёт времени
        elapsed_time += frame_time

        # Средняя скорость за N кадров
        average_speed = 0
        if len(recent_points) >= 2:
            dist_sum = sum(
                distance(recent_points[i - 1], recent_points[i])
                for i in range(1, len(recent_points))
            )
            duration = frame_time * (len(recent_points) - 1)
            average_speed = dist_sum / duration if duration > 0 else 0

        # Создание панели с информацией
        info_panel = np.zeros((frame.shape[0], INFO_PANEL_WIDTH, 3), dtype=np.uint8)

        def put_info(text, y):
            cv2.putText(info_panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

        put_info(f"Time: {elapsed_time:.2f} s", 40)
        put_info(f"Avg speed (2s): {average_speed:.2f} px/s", 80)
        put_info(f"Distance: {total_distance:.1f} px", 120)
        #put_info(f"Points: {len(trajectory)}", 160)

        # Объединение видео и панели
        combined = np.hstack((frame, info_panel))

        cv2.imshow("Object Tracking", combined)
        prev_gray = gray

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
