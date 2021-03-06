import os
import cv2
import time
from pathlib import Path
import numpy as np
import mediapipe as mp
import tensorflow.keras

from collections import deque

import utils
import config

VIDEO_FILE = '테스트17.mp4'

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = os.path.join(BASE_DIR, 'data', VIDEO_FILE)

def onChange(x):
    pass


def yuv_setting_bar():
    setting = 'YUV_settings'
    cv2.namedWindow(setting)
    cv2.moveWindow(setting, 1280, 0)

    cv2.createTrackbar('Y_MAX', 'YUV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('Y_MAX', 'YUV_settings', config.upperColor_YUV[0])
    cv2.createTrackbar('Y_MIN', 'YUV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('Y_MIN', 'YUV_settings', config.lowerColor_YUV[0])
    cv2.createTrackbar('U_MAX', 'YUV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('U_MAX', 'YUV_settings', config.upperColor_YUV[1])
    cv2.createTrackbar('U_MIN', 'YUV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('U_MIN', 'YUV_settings', config.lowerColor_YUV[1])
    cv2.createTrackbar('V_MAX', 'YUV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('V_MAX', 'YUV_settings', config.upperColor_YUV[2])
    cv2.createTrackbar('V_MIN', 'YUV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('V_MIN', 'YUV_settings', config.lowerColor_YUV[2])
    
    
def edge_setting_bar():
    setting = 'edge_settings'
    cv2.namedWindow(setting)
    cv2.moveWindow(setting, 1280, 370)

    cv2.createTrackbar('threshold1', 'edge_settings', 0, 255, onChange)
    cv2.setTrackbarPos('threshold1', 'edge_settings', 200)
    cv2.createTrackbar('threshold2', 'edge_settings', 0, 255, onChange)
    cv2.setTrackbarPos('threshold2', 'edge_settings', 220)
    cv2.createTrackbar('area_min', 'edge_settings', 0, 30000, onChange)
    cv2.setTrackbarPos('area_min', 'edge_settings', 5000)


def detect(model, lmk_list):
    lmk_list = np.array(lmk_list)
    lmk_list = np.expand_dims(lmk_list, axis=0)
    results = model.predict(lmk_list)
    lb_idx = np.argmax(results, axis=1)
    return lb_idx


def getContour(img, imgContour, area_min):
    contours, heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_min:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x_, y_, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x_, y_), (x_+w, y_+h), (0, 255, 0), 5)


def main():
    model = tensorflow.keras.models.load_model(config.model_path)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=200,varThreshold=64,detectShadows=False)
    
    idx = 0
    lmk_list = []
    pts = deque(maxlen=16)
    fps = 0
    
    radius = 0
    before_center = None
    expected_center = None
    n_try = 0

    cap = cv2.VideoCapture(VIDEO_PATH)
    # cap = cv2.VideoCapture(0)
    # 불량
    # 1, 2: 슛, 3: 화각 좁음
    # 4(공 인식 끊김): 킥, 5: 패스 6: 트래핑 7: 헤딩, 8, 9, 10, 11: 드리블, 12: 트래핑, 13: 헤딩, 14, 15, 16: 슛, 17: 많이 튀는 영상

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
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            # frame1 = frame[:, :, 0]
            # frame2 = frame[:, :, 1]
            # frame3 = cv2.equalizeHist(frame[:, :, 2])
            # frame = cv2.merge([frame1, frame2, frame3])
            # frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            
            Y_MAX = cv2.getTrackbarPos('Y_MAX', 'YUV_settings')
            Y_MIN = cv2.getTrackbarPos('Y_MIN', 'YUV_settings')
            U_MAX = cv2.getTrackbarPos('U_MAX', 'YUV_settings')
            U_MIN = cv2.getTrackbarPos('U_MIN', 'YUV_settings')
            V_MAX = cv2.getTrackbarPos('V_MAX', 'YUV_settings')
            V_MIN = cv2.getTrackbarPos('V_MIN', 'YUV_settings')
            lower = np.array([Y_MIN, U_MIN, V_MIN])
            higher = np.array([Y_MAX, U_MAX, V_MAX])

            yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

            # 배경 제거용 마스크 생성
            bg_mask = fgbg.apply(frame)
            bg_mask = cv2.erode(bg_mask, None, iterations=3)
            bg_mask = cv2.dilate(bg_mask, None, iterations=4)
            bg_mask = cv2.threshold(bg_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # 하늘 제거용 마스크 생성
            sky_mask = cv2.inRange(yuv_frame, (105, 0, 195), (120, 40, 255))
            sky_mask = cv2.dilate(sky_mask, None, iterations=1)
            sky_mask = cv2.bitwise_not(sky_mask)
            
            # 땅 제거용 마스크 생성
            land_mask = cv2.inRange(yuv_frame, (10, 0, 150), (30, 80, 255))
            land_mask = cv2.dilate(land_mask, None, iterations=1)
            land_mask = cv2.bitwise_not(land_mask)
            
            # 배경, 하늘 제거 프레임 생성
            frame_bg_rm = frame.copy()
            frame_bg_rm[bg_mask!=255] = (0,0,0)
            yuv_frame_find = cv2.cvtColor(frame_bg_rm, cv2.COLOR_BGR2YUV)
            # hsv_frame_find = cv2.bitwise_and(hsv_frame_find, hsv_frame_find, mask=sky_mask)
            # hsv_frame_find = cv2.bitwise_and(hsv_frame_find, hsv_frame_find, mask=land_mask)
            
            color_mask_find = cv2.inRange(yuv_frame_find, lower, higher)
            color_mask_find = cv2.erode(color_mask_find, None, iterations=1)
            color_mask_find = cv2.dilate(color_mask_find, None, iterations=3)
            
            color_mask_see = cv2.inRange(yuv_frame, lower, higher)
            yuv_picker = cv2.bitwise_and(frame, yuv_frame, mask=color_mask_see)
            # hsv_picker = cv2.bitwise_and(hsv_picker, hsv_picker, mask=sky_mask)
            # hsv_picker = cv2.bitwise_and(hsv_picker, hsv_picker, mask=land_mask)

            threshold1 = cv2.getTrackbarPos('threshold1', 'edge_settings')
            threshold2 = cv2.getTrackbarPos('threshold2', 'edge_settings')
            area_min = cv2.getTrackbarPos('area_min', 'edge_settings')
            edge_frame = yuv_picker.copy()
            edge_frame[bg_mask!=255] = (0,0,0)
            edge_frame = cv2.Canny(edge_frame, threshold1, threshold2)
            # edge_frame = cv2.erode(edge_frame, None, iterations=1)
            dilated_edge_frame = cv2.dilate(edge_frame, None, iterations=1)
            
            getContour(dilated_edge_frame, frame, area_min)
            
            cnts = cv2.findContours(color_mask_find, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            
            cnts = list(cnts)
            cnts = [c for c in cnts if len(c) > 10]
            cnts = sorted(cnts, key=lambda x: len(x), reverse=True)
            cnts = np.array(cnts)

            center = None
            
            if len(cnts) > 0:  # 컨투어 잡힐때(=공이 있을 때)
                n_try = 0
                center, _, _, radius = utils.find_center(cnts)

                before_center = before_center if before_center is not None else center
                expected_center = expected_center if expected_center is not None else center

                if radius > 10 and len(pts) > 0:
                    if pts:
                        if pts[-1] != None and pts[-1][:2] != center:
                            cv2.circle(frame, center, int(radius),
                                   (0, 0, 255), 2)

                distance = utils.get_distance(center, before_center)

                ratio = (distance // radius)+1
            else:  # 공이 없을 때
                if radius > 10 and len(pts) > 0 :
                    if pts:
                        if pts[-1] != None and pts[-1][:2] != before_center:
                            cv2.circle(frame, before_center, int(radius),
                            (0, 0, 255), 2)
                            print('여기가 거기냐')
                    # TODO: 공이 없어졌을 때, 기존 위치 그대로 리턴하도록!

            # 공 궤적 표시
            if center and ratio < 30:  # 공이 확실히 있는 것으로 판단 될 때 + 궤적이 이탈하지 않는다고 판단될 때.
                pts.appendleft((center[0], center[1], ratio))
                before_center = center
                expected_center = (2 * center[0]-before_center[0], 2 * center[1]-before_center[1])
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
                        lmk = lmk + [(center[0]/width)-ck_x, (center[1]/height)-ck_y, 100]
                    lmk_list.append(lmk)
                    frame = utils.draw_landmark_on_image(mpDraw, results, frame)

                    if len(lmk_list) == config.num_of_timestep:
                        label = config.labels[int(detect(model, lmk_list))]
                        lmk_list = []

            end = time.time()
            fps = str(int(1/(end-start)))
            frame = utils.write_on_image('FPS: '+fps, frame, (10, 75))
            frame = utils.write_on_image('['+str(idx)+']'+label, frame)
            
            winname1 = "frame"
            cv2.namedWindow(winname1)
            cv2.moveWindow(winname1, 0, 0)
            winname2 = "color_mask_find"
            cv2.namedWindow(winname2)
            cv2.moveWindow(winname2, 640, 0)
            winname3 = "frame_bg_rm"
            cv2.namedWindow(winname3)
            cv2.moveWindow(winname3, 0, 370)
            winname4 = "yuv_picker"
            cv2.namedWindow(winname4)
            cv2.moveWindow(winname4, 640, 370)
            winname5 = "edge"
            cv2.namedWindow(winname5)
            cv2.moveWindow(winname5, 0, 740)
            winname6 = "dilated_edge"
            cv2.namedWindow(winname6)
            cv2.moveWindow(winname6, 640, 740)
            
            frame = cv2.resize(frame, dsize=(640, 360), interpolation=cv2.INTER_AREA)
            cv2.imshow(winname1, frame)
            color_mask_find = cv2.resize(color_mask_find, dsize=(640, 360), interpolation=cv2.INTER_AREA)
            cv2.imshow(winname2, color_mask_find)
            frame_bg_rm = cv2.resize(frame_bg_rm, dsize=(640, 360), interpolation=cv2.INTER_AREA)
            cv2.imshow(winname3, frame_bg_rm)
            yuv_picker = cv2.resize(yuv_picker, dsize=(640, 360), interpolation=cv2.INTER_AREA)
            cv2.imshow(winname4, yuv_picker)
            sky_mask = cv2.resize(sky_mask, dsize=(640, 360), interpolation=cv2.INTER_AREA)
            cv2.imshow(winname5, sky_mask)
            dilated_edge_frame = cv2.resize(dilated_edge_frame, dsize=(640, 360), interpolation=cv2.INTER_AREA)
            cv2.imshow(winname6, dilated_edge_frame)
            # edge_frame = cv2.resize(edge_frame, dsize=(640, 360), interpolation=cv2.INTER_AREA)
            # cv2.imshow(winname5, edge_frame)
            # dilated_edge_frame = cv2.resize(dilated_edge_frame, dsize=(640, 360), interpolation=cv2.INTER_AREA)
            # cv2.imshow(winname6, dilated_edge_frame)
            
            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    

if __name__ == '__main__':
    yuv_setting_bar()
    edge_setting_bar()
    main()
