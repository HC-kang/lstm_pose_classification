import cv2
import mediapipe as mp
import pandas as pd


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


def make_landmark_timestep(results):
    print(results.pose_landmarks.landmark)
    c_lmk = []
    for lmk in results.pose_landmarks.landmark:
        c_lmk.append(lmk.x)
        c_lmk.append(lmk.y)
        c_lmk.append(lmk.z)
        c_lmk.append(lmk.visibility)
    return c_lmk


def draw_landmark_on_image(mpDraw, results, img):
    
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    for id, lmk in enumerate(results.pose_landmarks.landmark):
        h, w, _ = img.shape
        print(id, lmk)
        cx, cy = int(lmk.x * w), int(lmk.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return img


def produce_pose_data(label, num_of_frames = 10):
    cap = cv2.VideoCapture('./data/chest_trapping.mp4')
    lmk_list = []
    while len(lmk_list) <= num_of_frames:
        ret, frame = cap.read()
        if frame is None:
            break
        if ret:

            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frameRGB)
            if results.pose_landmarks:

                lmk = make_landmark_timestep(results)
                lmk_list.append(lmk)

                frame = draw_landmark_on_image(mpDraw, results, frame)
            cv2.imshow('image', frame)
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
labels = ['chest_trapping']
import time
for label in labels:
    time.sleep(3)
    produce_pose_data(label, 1000)
