import cv2
import numpy as np
import mediapipe as mp
import tensorflow.keras

from collections import deque

import utils
import config


def detect(model, lmk_list):
    lmk_list = np.array(lmk_list)
    lmk_list = np.expand_dims(lmk_list, axis=0)
    print(lmk_list.shape)
    results = model.predict(lmk_list)
    print(results.shape)
    lb_idx = np.argmax(results, axis=1)
    return lb_idx

def main():
    model = tensorflow.keras.models.load_model(config.model_path)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    
    idx = 0
    lmk_list = []
    pts = deque(maxlen=64)
    
    cap = cv2.VideoCapture('./data/축구2-1.mp4')

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils
    
    label = "Warmup...."
    while True:
        ret, frame = cap.read()
        print(frame.shape)
        if frame is None:
            break
        if ret:
            fgmask = fgbg.apply(frame)
            fgmask = cv2.erode(fgmask, None, iterations=3)
            fgmask = cv2.dilate(fgmask, None, iterations=3)
            fgmask = cv2.threshold(fgmask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            frame2 = frame.copy()
            frame2[fgmask!=255] = (0,0,0)
            
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
                print("Start Detect...")
                
                if results.pose_landmarks:
                    lmk = utils.make_landmark_timestep(results)
                    if center is None:
                        lmk = lmk + [0, 0, 0]
                    else:
                        lmk = lmk + [center[0]/width, center[1]/height, 1]
                    lmk_list.append(lmk)
                    frame = utils.draw_landmark_on_image(mpDraw, results, frame)

                    if len(lmk_list) == config.num_of_timestep:
                        label = config.labels[int(detect(model, lmk_list))]
                        lmk_list = []

            frame = utils.draw_class_on_image('['+str(idx)+']'+label, frame)
            cv2.imshow('image', frame)
            cv2.imshow('frame2', frame2)
            cv2.imshow('fgmask', fgmask)
            cv2.imshow('mask', mask)
            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    

if __name__ == '__main__':
    main()