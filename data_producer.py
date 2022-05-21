import os
import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp

from collections import deque

import utils
import config


def produce_pose_data(label, num_of_frames=300):
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=64, detectShadows=False)

    idx = 0
    lmk_list = []
    pts = deque(maxlen=32)
    fps = 0

    radius = 0
    before_center = None
    expected_center = None

    cap = cv2.VideoCapture('./data/0516/yellow/data/' + label)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    while len(lmk_list) <= num_of_frames:
        ret, frame = cap.read()
        if frame is None:
            break
        if frame.shape[0] > frame.shape[1]:
            q = frame.shape[0]//4
            frame = frame[q:-q]
        frame = cv2.flip(frame, 1)
        start = time.time()
        if ret:
            lower = config.lowerColor
            higher = config.upperColor

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 배경 제거용 마스크 생성
            bg_mask = fgbg.apply(frame)
            bg_mask = cv2.erode(bg_mask, None, iterations=3)
            bg_mask = cv2.dilate(bg_mask, None, iterations=4)
            bg_mask = cv2.threshold(
                bg_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # 배경, 하늘 제거 프레임 생성
            frame_bg_rm = frame.copy()
            frame_bg_rm[bg_mask != 255] = (0, 0, 0)
            hsv_frame_find = cv2.cvtColor(frame_bg_rm, cv2.COLOR_BGR2HSV)

            color_mask_find = cv2.inRange(hsv_frame_find, lower, higher)
            color_mask_find = cv2.erode(color_mask_find, None, iterations=1)
            color_mask_find = cv2.dilate(color_mask_find, None, iterations=8)

            color_mask_see = cv2.inRange(hsv_frame, lower, higher)
            hsv_picker = cv2.bitwise_and(frame, frame, mask=color_mask_see)

            threshold1 = 200
            threshold2 = 220
            # area_min = 5000
            # threshold1 = cv2.getTrackbarPos('threshold1', 'edge_settings')
            # threshold2 = cv2.getTrackbarPos('threshold2', 'edge_settings')
            # area_min = cv2.getTrackbarPos('area_min', 'edge_settings')
            edge_frame = hsv_picker.copy()
            edge_frame[bg_mask != 255] = (0, 0, 0)
            edge_frame = cv2.Canny(edge_frame, threshold1, threshold2)
            # edge_frame = cv2.erode(edge_frame, None, iterations=1)
            # dilated_edge_frame = cv2.dilate(edge_frame, None, iterations=1)

            # getContour(dilated_edge_frame, frame, area_min)

            cnts = cv2.findContours(
                color_mask_find, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            cnts = list(cnts)
            cnts = [c for c in cnts if len(c) > 1]
            cnts = sorted(cnts, key=lambda x: len(x), reverse=True)

            center = None

            if len(cnts) > 0:  # 컨투어 잡힐때(=공이 있을 때)
                center, _, _, radius = utils.find_center(cnts)

                before_center = before_center if before_center is not None else center
                expected_center = expected_center if expected_center is not None else center

                if len(pts) > 0:
                    if pts:
                        if pts[-1] != None and pts[-1][:2] != center:
                            cv2.circle(frame, center, int(radius),
                                       (0, 0, 255), 2)

                distance = utils.get_distance(center, before_center)

                ratio = (distance // radius)+1
            else:  # 공이 없을 때
                if len(pts) > 0:
                    if pts:
                        if pts[-1] != None and pts[-1][:2] != before_center:
                            cv2.circle(frame, before_center, int(radius),
                                       (0, 0, 255), 2)

                    # TODO: 공이 없어졌을 때, 기존 위치 그대로 리턴하도록!

            # 공 궤적 표시
            # 공이 확실히 있는 것으로 판단 될 때 + 궤적이 이탈하지 않는다고 판단될 때.
            if center and ratio < 30:
                pts.appendleft((center[0], center[1], ratio))
                before_center = center
                expected_center = (
                    2 * center[0]-before_center[0], 2 * center[1]-before_center[1])
            elif pts and before_center and ratio < 30:  # 공을 못 찾았지만 before_center가 남아있어 이것을 기준으로 위치를 입력
                if pts[-1] != None and pts[-1][:2] != before_center:
                    pts.appendleft((before_center[0], before_center[1], ratio))
                    before_center = center
                    expected_center = center
            else:
                if pts:  # 공이 확실히 없어졌고, before_center가 이미 한 번 반영이 된 경우
                    if pts[-1] != None and pts[-1][:2] == before_center:
                        before_center = None
                        expected_center = None
                pts.appendleft(None)
                before_center = center
                expected_center = center

            frame = utils.draw_line(pts, frame)

            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frameRGB)

            idx = idx + 1
            if idx > config.warmup_frame:
                print('------ idx: ', idx)
                if results.pose_landmarks:
                    lmk = utils.make_landmark_timestep(results)
                    lmk, ck_x, ck_y = utils.convert_to_relative(lmk)
                    if center is None:
                        lmk = lmk + [0, 0, 0]
                    else:
                        lmk = lmk + [(center[0]/width)-ck_x,
                                     (center[1]/height)-ck_y, 100]
                    lmk_list.append(lmk)
                    frame = utils.draw_landmark_on_image(
                        mpDraw, results, frame)

            end = time.time()
            fps = str(int(1/(end-start)))
            frame = utils.write_on_image('FPS: '+fps, frame, (10, 75))
            frame = utils.write_on_image(
                '['+str(len(lmk_list))+']'+label, frame)

            winname1 = "frame"
            cv2.namedWindow(winname1)
            cv2.moveWindow(winname1, 0, 0)
            winname2 = "color_mask_find"
            cv2.namedWindow(winname2)
            cv2.moveWindow(winname2, 640, 0)
            winname3 = "frame_bg_rm"
            cv2.namedWindow(winname3)
            cv2.moveWindow(winname3, 0, 370)
            winname4 = "hsv_picker"
            cv2.namedWindow(winname4)
            cv2.moveWindow(winname4, 640, 370)

            cv2.imshow(winname1, frame)

            cv2.imshow(winname2, color_mask_find)

            cv2.imshow(winname3, frame_bg_rm)

            cv2.imshow(winname4, hsv_picker)

            if cv2.waitKey(1) == ord("q"):
                break
    else:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    data = pd.DataFrame(lmk_list)
    data.to_csv(f"./data/0516/yellow/label/{label}_r.txt")

    return


# labels = ['stop', 'l_h_swing', 'r_h_swing', 'l_f_swing', 'r_f_swing']
labels = os.listdir('./data/0516/yellow/data')

if __name__ == '__main__':
    for label in labels:
        # time.sleep(2)
        produce_pose_data(label, config.num_of_frames)
