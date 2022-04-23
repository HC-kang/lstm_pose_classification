import os
from typing import NamedTuple
import cv2
import mediapipe as mp
import numpy as np
from collections import namedtuple

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

def make_landmark_timestep(results):
    c_lmk = []
    for lmk in results.pose_landmarks.landmark:
        c_lmk.append(lmk.x)
        c_lmk.append(lmk.y)
        c_lmk.append(lmk.z)
        c_lmk.append(lmk.visibility)
    return c_lmk

def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    for lmk in results.pose_landmarks.landmark:
        h, w, c = img.shape
        cx, cy = int(lmk.x * w), int(lmk.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return img

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
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

def save_model(model):
    model_number = 1
    while True:
        if not os.path.isfile(f'./models/simple_lstm_model_{model_number:0>3d}.h5'):
            model.save(f'./models/simple_lstm_model_{model_number:0>3d}.h5')
            print('model generated at')
            print('--->', f'./models/simple_lstm_model_{model_number:0>3d}.h5')
            break
        else:
            model_number += 1
            
def find_center(cnts):
    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return center, x, y, radius


def draw_line(pts, frame):
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue

        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
        frame = cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255, 0.2), thickness)
    return frame


def RGB2HSV(rgb):
    V = max(rgb)    
    if V == 0:
        S = 0
    else:
        S = (V - min(rgb)) / V
    if V == rgb[0]:
        H = (60*(rgb[1] - rgb[2]))/(V-min(rgb))
    elif V == rgb[1]:
        H = (60*(rgb[2] - rgb[0]))/(V-min(rgb)) + 120
    elif V == rgb[2]:
        H = (60*(rgb[0] - rgb[1]))/(V-min(rgb)) + 240
    if H < 0:
        H += 360
    elif H > 360:
        H -= 360
    H //= 2
    return (int(H), int(S*255), int(V))
