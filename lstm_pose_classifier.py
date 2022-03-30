import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow.keras

cap = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lmk_list = []
num_of_frames = 600

i = 0
warmup_frame = 60

model = tensorflow.keras.models.load_model("./models/model_multi_mark.h5")

def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    for id, lmk in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lmk.x * w), int(lmk.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return img


def make_landmark_timestep(results):
    c_lmk = []
    for id, lmk in enumerate(results.pose_landmarks.landmark):
        c_lmk.append(lmk.x)
        c_lmk.append(lmk.y)
        c_lmk.append(lmk.z)
        c_lmk.append(lmk.visibility)
    return c_lmk

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

label = "Warmup...."
while len(lmk_list) <= num_of_frames:
    ret, frame = cap.read()
    if ret:

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
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
        frame = draw_class_on_image(label, frame)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)