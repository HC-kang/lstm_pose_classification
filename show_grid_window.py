import cv2
import numpy as np


def draw_class_on_image(label, img, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner_of_text = (x, y)
    font_scale = 1
    font_color = (0, 255, 0)
    thickness = 2
    line_type = 2
    cv2.putText(img, label,
                bottom_left_corner_of_text,
                font,
                font_scale,
                font_color,
                thickness,
                line_type)
    return img


def nothing(x):
    pass


def make_window(winname, width, height, position_x, position_y):
    cv2.namedWindow(winname)
    cv2.resizeWindow(winname=winname, width=width, height=height)
    cv2.moveWindow(winname=winname, x=position_x, y=position_y)

    cv2.createTrackbar('channel', winname, 0, 2, nothing)
    cv2.createTrackbar('RHY0', winname, 100, 255, nothing)
    cv2.createTrackbar('RHY1', winname, 200, 255, nothing)
    cv2.createTrackbar('GSU0', winname, 100, 255, nothing)
    cv2.createTrackbar('GSU1', winname, 200, 255, nothing)
    cv2.createTrackbar('BVV0', winname, 100, 255, nothing)
    cv2.createTrackbar('BVV1', winname, 200, 255, nothing)


def make_view(origin_frame, one_channel_frame, title, position_x, position_y):
    three_channel_frame = np.zeros_like(origin_frame)
    three_channel_frame[:, :, 0] = one_channel_frame
    three_channel_frame[:, :, 1] = one_channel_frame
    three_channel_frame[:, :, 2] = one_channel_frame
    draw_class_on_image(title, three_channel_frame, position_x, position_y)
    return three_channel_frame


def make_view_by_color_model(origin_frame, three_channel_frame, title, position_x, position_y):
    first_origin_channel_frame, second_origin_channel_frame, third_origin_channel_frame = cv2.split(three_channel_frame)
    first_channel_frame = make_view(origin_frame, first_origin_channel_frame, title + "_ch1", position_x, position_y)
    second_channel_frame = make_view(origin_frame, second_origin_channel_frame, title + "_ch2", position_x, position_y)
    third_channel_frame = make_view(origin_frame, third_origin_channel_frame, title + "_ch3", position_x, position_y)
    return first_channel_frame, second_channel_frame, third_channel_frame


def make_view_with_position(origin_frame, one_channel_frame, title, position_x, position_y):
    return make_view(origin_frame, one_channel_frame, title, position_x, position_y)


def make_view_with_only_label(origin_frame, one_channel_frame, title):
    return make_view(origin_frame, one_channel_frame, title, 10, 30)


def make_view_with_position_by_color_model(origin_frame, three_channel_frame, title, position_x, position_y):
    return make_view_by_color_model(origin_frame, three_channel_frame, title, position_x, position_y)


def make_view_with_only_label_by_color_model(origin_frame, three_channel_frame, title):
    return make_view_by_color_model(origin_frame, three_channel_frame, title, 10, 30)


def make_background_subtractor(cv2):
    mog = cv2.bgsegm.createBackgroundSubtractorMOG(history=1, nmixtures=3, backgroundRatio=0.7, noiseSigma=0)
    mog2 = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=10, detectShadows=False)
    gmg = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=20, decisionThreshold=0.7)
    return mog, mog2, gmg


def make_all_frame(frame, fgbg, fgbg2, fgbg3, gray, mask):
    remove_bg = fgbg.apply(frame)
    framefgbg = make_view_with_only_label(frame, remove_bg, "removeBG")

    remove_bg2 = fgbg2.apply(frame)
    framefgbg2 = make_view_with_only_label(frame, remove_bg2, "removeBG2")

    remove_bg3 = fgbg3.apply(frame)
    framefgbg3 = make_view_with_only_label(frame, remove_bg3, "removeBG3")
    
    frameGray = make_view_with_only_label(frame, gray, "grey")

    frameMask = make_view_with_only_label(frame, mask, "mask")
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = draw_class_on_image("RGB", frame_rgb, 10, 30)

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_hsv = draw_class_on_image("HSV", frame_hsv, 10, 30)

    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    frame_yuv = draw_class_on_image("YUV", frame_yuv, 10, 30)

    return framefgbg, framefgbg2, framefgbg3, frameGray, frameMask, frame_rgb, frame_hsv, frame_yuv
    # return make_remove_background_frame(frame, fgbg, fgbg2, fgbg3),\
    #        make_gray_mask_frame(frame, gray, mask),\
    #        make_color_model_frame(frame)


def make_remove_background_frame(frame, fgbg, fgbg2, fgbg3):
    remove_bg = fgbg.apply(frame)
    framefgbg = make_view_with_only_label(frame, remove_bg, "removeBG")

    remove_bg2 = fgbg2.apply(frame)
    framefgbg2 = make_view_with_only_label(frame, remove_bg2, "removeBG2")

    remove_bg3 = fgbg3.apply(frame)
    framefgbg3 = make_view_with_only_label(frame, remove_bg3, "removeBG3")
    return framefgbg, framefgbg2, framefgbg3


def make_gray_mask_frame(frame, gray, mask):
    frame_gray = make_view_with_only_label(frame, gray, "grey")

    frame_mask = make_view_with_only_label(frame, mask, "mask")
    return frame_gray, frame_mask


def make_color_model_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = draw_class_on_image("RGB", frame_rgb, 10, 30)

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_hsv = draw_class_on_image("HSV", frame_hsv, 10, 30)

    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    frame_yuv = draw_class_on_image("YUV", frame_yuv, 10, 30)
    return frame_rgb, frame_hsv, frame_yuv


def make_color_model(pose, select_channel, frame_rgb, frame_hsv, frame_yuv):
    if select_channel == 0:
        results = pose.process(frame_rgb)
        frame_color_model = frame_rgb
        frame_title = "RGB"
    elif select_channel == 1:
        results = pose.process(frame_hsv)
        frame_color_model = frame_hsv
        frame_title = "HSV"
    else:
        results = pose.process(frame_yuv)
        frame_color_model = frame_yuv
        frame_title = "YUV"
    return results, frame_color_model, frame_title


def control_trackbar(winname):
    channel1_min = cv2.getTrackbarPos('RHY0', winname)
    channel2_min = cv2.getTrackbarPos('GSU0', winname)
    channel3_min = cv2.getTrackbarPos('BVV0', winname)
    channel1_max = cv2.getTrackbarPos('RHY1', winname)
    channel2_max = cv2.getTrackbarPos('GSU1', winname)
    channel3_max = cv2.getTrackbarPos('BVV1', winname)

    channel_num = cv2.getTrackbarPos('channel', winname)
    return channel1_min, channel2_min, channel3_min, channel1_max, channel2_max, channel3_max, channel_num


def show_window(frame,
                frame_color_model,
                frame_mask,
                frame_gray,
                frame_rgb,
                frame_hsv,
                frame_yuv,
                framefgbg,
                framefgbg2,
                framefgbg3,
                frame_title):
    frame1, frame2, frame3 = make_view_with_only_label_by_color_model(frame, frame_color_model, frame_title)

    numpy_horizontal = np.hstack((frame, frame_mask, frame_gray, framefgbg))
    numpy_horizontal2 = np.hstack((frame_rgb, frame_hsv, frame_yuv, framefgbg2))
    numpy_horizontal3 = np.hstack((frame1, frame2, frame3, framefgbg3))

    numpy_vertical_concat = np.concatenate((numpy_horizontal, numpy_horizontal2, numpy_horizontal3), axis=0)
    cv2.imshow('image', numpy_vertical_concat)
