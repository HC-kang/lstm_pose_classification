import cv2
import time
import numpy as np
import mediapipe as mp
import tensorflow.keras

from collections import deque

import utils
import config


def detect(model, lmk_list):
    lmk_list = np.array(lmk_list)
    lmk_list = np.expand_dims(lmk_list, axis=0)
    results = model.predict(lmk_list)
    lb_idx = np.argmax(results, axis=1)
    return lb_idx

def main():
    model = tensorflow.keras.models.load_model(config.model_path)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    
    idx = 0
    lmk_list = []
    pts = deque(maxlen=64)
    fps = 0
    

    # hough circle을 위한 함수, 변수 추가
    dist = lambda x1, y1, x2, y2: (x1 - x2)**2 + (y1 - y2)**2
    center = None
    prevCircle = None

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('./data/축구2-1.mp4')

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils
    
    label = "Warmup...."
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        start = time.time()
        if ret:
            fgmask = fgbg.apply(frame)
            fgmask = cv2.erode(fgmask, None, iterations=3)
            fgmask = cv2.dilate(fgmask, None, iterations=3)
            fgmask = cv2.threshold(fgmask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            frame2 = frame.copy()
            frame2[fgmask!=255] = (0,0,0)

            # fgmask2 = fgbg.apply(frame)
            # fgmask2 = cv2.erode(fgmask2, None, iterations=2)
            # fgmask2 = cv2.dilate(fgmask2, None, iterations=5)
            # fgmask2 = cv2.threshold(fgmask2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # frame3 = frame.copy()
            # frame3[fgmask2!=255] = (0,0,0)
            # # frame3 = cv2.Canny(frame3, 200, 220)
            # # frame3 = cv2.dilate(frame3, None, iterations=1)
            # # frame3 = cv2.erode(frame3, None, iterations=1)

            # grayFrame = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
            # blurFrame = cv2.GaussianBlur(grayFrame, (0, 0), 8)

            # hough circles를 활용한 원형 검출
            # circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1.2, 100, param1 = 25, param2 = 60, minRadius= 5, maxRadius=200)

            # if circles is not None:
            #     circles = np.uint16(np.around(circles))
            #     chosen = None
            #     for i in circles[0, :]:
            #         if chosen is None:
            #             chosen = i
            #         if prevCircle is not None:
            #             if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dist(i[0], i[1], prevCircle[0], prevCircle[1]):
            #                 chosen = i
            #                 center = chosen[0], chosen[1]
            #     cv2.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)
            #     cv2.circle(frame, (chosen[0], chosen[1]), chosen[2], (255, 0, 255), 3)
            #     prevCircle = chosen
            
            
            blurred = cv2.GaussianBlur(frame2, (0, 0), 1)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            mask = cv2.inRange(hsv, config.lowerColor, config.upperColor)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=5)
            
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[0]

            center = None
            if len(cnts) > 0:
                center, x, y, radius = utils.find_center(cnts)
                if radius > 10:
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                        (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
            pts.appendleft(center)
            
            frame = utils.draw_line(pts, frame)

            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frameRGB)
            idx = idx + 1
            if idx > config.warmup_frame:
                # print("Start Detect...")
                
                if results.pose_landmarks:
                    lmk = utils.make_landmark_timestep(results)
                    lmk, ck_x, ck_y = utils.convert_to_relative(lmk)
                    if center is None:
                        lmk = lmk + [0, 0, 0]
                    else:
                        lmk = lmk + [(center[0]/width)-ck_x, (center[1]/height)-ck_y, 100]
                    lmk_list.append(lmk)
                    frame = utils.draw_landmark_on_image(mpDraw, results, frame)

                    if len(lmk_list) == 22:
                        lmk_list = np.append(np.array(lmk_list), ([0]*102)).reshape(32, 32, 3)
                        print(lmk_list.shape)
                        label = config.labels[int(detect(model, lmk_list))]
                        lmk_list = []

            end = time.time()
            fps = str(int(1/(end-start)))
            frame = utils.write_on_image('FPS: '+fps, frame, (10, 75))
            frame = utils.write_on_image('['+str(idx)+']'+label, frame)
            # cv2.imshow('frame2', frame2)
            # cv2.imshow('frame3', frame3)
            cv2.imshow('fgmask', fgmask)
            cv2.imshow('mask', mask)
            # cv2.imshow('blurFrame', blurFrame)
            cv2.imshow('image', frame)
            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    

if __name__ == '__main__':
    main()