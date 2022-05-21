# -*- coding: utf-8 -*-
import cv2
import time
import threading
import numpy as np
import mediapipe as mp
import tensorflow.keras
import show_grid_window as win

from collections import deque

# 비디오 화면 불러오기
cap = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lmk_list = []
num_of_frames = 600

i = 0
warmup_frame = 60

# 스택 또는 큐 처럼 사용할 수 있고 list 보다 월등한 옵션이다.
pts = deque(maxlen=64)

model = tensorflow.keras.models.load_model("./models/model_multi_mark.h5")


def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    for lmk in results.pose_landmarks.landmark:
        h, w, c = img.shape
        cx, cy = int(lmk.x * w), int(lmk.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return img


def make_landmark_timestep(results):
    c_lmk = []
    for lmk in results.pose_landmarks.landmark:
        c_lmk.append(lmk.x)
        c_lmk.append(lmk.y)
        c_lmk.append(lmk.z)
        c_lmk.append(lmk.visibility)
    return c_lmk


def draw_class_on_image(label, img, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (x, y)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img


def detect(model, lmk_list):
    global label
    lmk_list = np.array(lmk_list)
    lmk_list = np.expand_dims(lmk_list, axis=0)
    print(lmk_list.shape)
    results = model.predict(lmk_list)
    print(results.shape)
    lb_idx = np.argmax(results, axis=1)
    if lb_idx == 0:
        label = 'STOP'
    elif lb_idx == 1:
        label = 'l_hand_swing'
    elif lb_idx == 2:
        label = 'r_HAND'
    elif lb_idx == 3:
        label = '###_l_leg_swing'
    elif lb_idx == 4:
        label = '###R_LEG'
    return label


# python 파일 entry point
if __name__ == '__main__':
    win.make_window('image', 300, 120, 0, 0)

    label = "Warmup...."
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 1. background subtractor
    fgbg, fgbg2, fgbg3 = win.make_background_subtractor(cv2)

    ii = 0

    while len(lmk_list) <= num_of_frames:
        # 2. control trackbar
        ch1_min, ch2_min, ch3_min, ch1_max, ch2_max, ch3_max, select_channel = win.control_trackbar("image")

        ret, frame = cap.read()

        if ret:
            # 가우시안 블러
            blurred = cv2.GaussianBlur(frame, (0, 0), 1)

            # 3. color model
            rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            yuv = cv2.cvtColor(blurred, cv2.COLOR_BGR2YUV)
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

            # 4. 지정된 범위의 color 찾기
            if (select_channel == 0):
                channel = rgb
            elif (select_channel == 1):
                channel = hsv
            else:
                channel = yuv

            # channel = cv2.fastNlMeansDenoisingColoredMulti(channel,None,10,10,7,21)

            # channel = cv2.equalizeHist(channel)
            channel_planes = cv2.split(channel)

            # 밝기 성분에 대해서만 히스토그램 평활화 수행
            channel_planes0 = cv2.equalizeHist(channel_planes[0])
            channel_planes1 = cv2.equalizeHist(channel_planes[1])
            channel_planes2 = cv2.equalizeHist(channel_planes[2])

            channel = cv2.merge([channel_planes0, channel_planes1, channel_planes2])

            framefgbg, framefgbg2, framefgbg3 = win.make_remove_background_frame(frame, fgbg, fgbg2, fgbg3)

            # 5. color range mask
            mask = cv2.inRange(channel, (ch1_min, ch2_min, ch3_min), (ch1_max, ch2_max, ch3_max))
            # mask = cv2.inRange(hsv, (20, 100, 50), (60, 200, 150))
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0]

            center = None

            if len(cnts) > 0:

                # 원 계산하기
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # 너무 작은 반지름 제거
                if radius > 10:
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

            pts.appendleft(center)

            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue

                thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

            # 6. make all frame
            # framefgbg, framefgbg2, framefgbg3, frameGray, frameMask, frameRGB, frameHSV, frameYUV = win.make_all_frame(frame, fgbg, fgbg2, fgbg3, gray, mask)
            frameGray, frameMask = win.make_gray_mask_frame(frame, gray, mask)
            frameRGB, frameHSV, frameYUV = win.make_color_model_frame(frame)

            # 7. frame color model
            results, frameColorModel, frameTitle = win.make_color_model(pose, select_channel, frameRGB, frameHSV, frameYUV)

            i = i + 1

            if i > warmup_frame:
                print("Start Detect...")
                if results.pose_landmarks:

                    lmk = make_landmark_timestep(results)
                    lmk_list.append(lmk)
                    if len(lmk_list) == 10:
                        t1 = threading.Thread(target=detect, args=(model, lmk_list,))
                        t1.start()
                        lmk_list = []
                    frame = draw_landmark_on_image(mpDraw, results, frame)
            frame = draw_class_on_image(label, frame, 10, 30)

            # 8. add label
            frame = draw_class_on_image("Original", frame, 10, 60)

            # ret, thresh1 = cv2.threshold(framefgbg2,127,255, cv2.THRESH_BINARY)
            ret, thresh1 = cv2.threshold(framefgbg3,127,255, cv2.THRESH_BINARY)

            # frameOp1 = cv2.subtract(frame, framefgbg2)
            # frameOp2 = cv2.subtract(framefgbg2, frameRGB)
            frameOp3 = cv2.bitwise_and(frame, thresh1)
            frameOp3 = cv2.medianBlur(frameOp3,5)
            dst = frameOp3.copy()
            gray2 = cv2.cvtColor(frameOp3, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 15, maxRadius = 40)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0]:
                    cv2.circle(dst, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # frameOp4 = cv2.bitwise_and(frame, thresh1)
            # frameRGB = cv2.bitwise_and(frameOp3, thresh1)

            # cv2.imshow('origin1', frameOp1)
            # cv2.imshow('origin2', frameOp2)
            # cv2.imshow('origin3', frameOp3)
            cv2.imshow('origin4', dst)

            # 9. show grid window
            # win.show_window(frameOp3, frameColorModel, frameMask, frameGray, frameRGB, frameHSV, frameYUV, framefgbg, framefgbg2, framefgbg3, frameTitle)

            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
