# debug_single_color_detect.py
import cv2
import numpy as np
import csv
import time
import sys
import traceback
import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=0, help="Camera index")
    p.add_argument("--w", type=int, default=1280, help="Frame width")
    p.add_argument("--h", type=int, default=720, help="Frame height")
    p.add_argument("--out", type=str, default="results.csv", help="CSV output")
    return p.parse_args()

def open_camera(idx, w, h):
    cam = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # WindowsでDirectShowを使う例
    if not cam.isOpened():
        return None, "VideoCapture.open failed"
    # try to set resolution
    ok_w = cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    ok_h = cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cam, (ok_w, ok_h)

def detect_color(frame, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    total_area = sum(areas)
    return mask, contours, total_area

def main():
    args = parse_args()
    try:
        cam, status = open_camera(args.cam, args.w, args.h)
        if cam is None:
            print("Camera open error:", status)
            sys.exit(1)
        else:
            print("Camera opened. set resolution status:", status)

        # 赤のHSV閾値（例）
        lower_red1 = np.array([0, 100, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 50])
        upper_red2 = np.array([179, 255, 255])

        with open(args.out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "total_area", "status"])

            for i in range(50):
                ret, frame = cam.read()
                if not ret or frame is None:
                    print("Frame read failed at iteration", i)
                    time.sleep(0.5)
                    continue

                mask1, cnts1, area1 = detect_color(frame, lower_red1, upper_red1)
                mask2, cnts2, area2 = detect_color(frame, lower_red2, upper_red2)
                total_area = area1 + area2

                status_str = "OK" if total_area > 5000 else "NG"
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, int(total_area), status_str])

                vis = frame.copy()
                cv2.drawContours(vis, cnts1+cnts2, -1, (0,255,0), 2)
                cv2.putText(vis, f"Area:{int(total_area)} {status_str}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                cv2.imshow("vis", vis)
                key = cv2.waitKey(200) & 0xFF
                if key == ord('q'):
                    break

    except Exception as e:
        print("Exception occurred:", e)
        traceback.print_exc()
    finally:
        try:
            cam.release()
        except:
            pass
        cv2.destroyAllWindows()
        print("Finished")

if __name__ == "__main__":
    main()
