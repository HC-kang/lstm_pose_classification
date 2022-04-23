import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp

from collections import deque

import utils
import config


def produce_pose_data(label, num_of_frames = 300):
    lmk_list = []
    pts = deque(maxlen=64)
    
    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    fgbg = cv2.createBackgroundSubtractorMOG2()

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils
    
    while len(lmk_list) <= num_of_frames:
        ret, frame = cap.read()
        if frame is None:
            break
        if ret:
            fgmask = fgbg.apply(frame)
            fgmask = cv2.erode(fgmask, None, iterations=2)
            fgmask = cv2.dilate(fgmask, None, iterations=2)
            fgmask = cv2.threshold(fgmask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            frame2 = frame.copy()
            frame2[fgmask!=255] = (0,0,0)
            
            blurred = cv2.GaussianBlur(frame2, (0, 0), 1)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            mask = cv2.inRange(hsv, config.lowerColor, config.upperColor)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
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
            
            if results.pose_landmarks:
                lmk = utils.make_landmark_timestep(results)
                if center is None:
                    lmk = lmk + [0, 0, 0]
                else:
                    lmk = lmk + [center[0]/width, center[1]/height, 1]
                lmk_list.append(lmk)
                frame = utils.draw_landmark_on_image(mpDraw, results, frame)


            frame = utils.draw_class_on_image('['+str(len(lmk_list))+']'+label, frame)
            cv2.imshow('image', frame)
            cv2.imshow('frame2', frame2)
            cv2.imshow('fgmask', fgmask)
            if cv2.waitKey(1) == ord("q"):
                break
    else:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    
    data = pd.DataFrame(lmk_list)
    data.to_csv(f"./data/{label}.txt")
    
    return

# labels = ['stop', 'l_h_swing', 'r_h_swing', 'l_f_swing', 'r_f_swing']

if __name__ == '__main__':
    for label in config.labels:
        time.sleep(2)
        produce_pose_data(label, config.num_of_frames)
