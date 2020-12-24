import cv2
import path
import numpy as np
import copy
import tensorflow as tf
from models.context_based.parking_context_recognizer.train import get_model
from models.context_based.parking_slot_detector.psd_detect import detect_slot
import config
from Camera.Undistortion import UndistortFisheye
from Camera.PerspectiveTransformation import BirdView


class LaneDetector:
    def __init__(self, config):

        self.config = config

    def gaussian_blur(self, img: np.array, kernel_size: int):  # 가우시안 필터
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def cannyEdge_img(self, img, min_thresh=70, max_thresh=210):
        return cv2.Canny(img, min_thresh, max_thresh)

    def weighted_img(self, img, initial_img, α=1, β=1.0, λ=0.0):  # 두 이미지 operlap 하기
        return cv2.addWeighted(initial_img, α, img, β, λ)

    def houghLines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(
            img,
            rho,
            theta,
            threshold,
            np.array([]),
            minLineLength=min_line_len,
            maxLineGap=max_line_gap,
        )
        return lines

    def region_of_interest(self, img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅

        mask = np.zeros_like(img)  # mask = img와 같은 크기의 빈 이미지

        if len(img.shape) > 2:  # Color 이미지(3채널)라면 :
            color = color3
        else:  # 흑백 이미지(1채널)라면 :
            color = color1

        # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
        cv2.fillPoly(mask, vertices, color)

        # 이미지와 color로 채워진 ROI를 합침
        ROI_image = cv2.bitwise_and(img, mask)

        return ROI_image

    def detect_position(self, img):
        return detect_slot(image=img)

    def detect_type(self, img):

        parking_context_recognizer = get_model()

        img = cv2.resize(img, (self.config["IMG_WIDTH"], self.config["IMG_HEIGHT"]))
        img = img[np.newaxis, :]
        result = parking_context_recognizer.predict(img)

        type_predict, angle_predict = result[0], result[1]
        tf.keras.backend.clear_session()

        return result

    def slot_detect(self, img):
        detect_slot()

    def detect_houghLines(self, img):

        height, width = img.shape[:2]  # 이미지 높이, 너비
        blur_img = self.gaussian_blur(img, 3)  # Blur 효과
        canny_img = self.cannyEdge_img(blur_img, 70, 210)

        vertices = np.array(
            [
                [
                    (50, height),
                    (width / 2 - 45, height / 2 + 60),
                    (width / 2 + 45, height / 2 + 60),
                    (width - 50, height),
                ]
            ],
            dtype=np.int32,
        )

        ROI_img = self.region_of_interest(canny_img, vertices)  # ROI 설정

        houghResult = self.houghLines(ROI_img, 1, 1 * np.pi / 180, 30, 10, 20)

        return houghResult


def test():
    print("lane detector test")

    #load images
    leftStream = cv2.VideoCapture(0)
    leftCamera = UndistortFisheye("left_Camera")
    leftBird = BirdView()
    print(leftStream.get(3), leftStream.get(4))

    # we can change size of frame to ex) 320X240
    # ret = cap.set(3,320)
    # ret = cap.set(4,240)

    # need to designate 4 points
    leftBird.setDimensions((186, 195), (484, 207), (588, 402), (97, 363)) #fisheyecalibration을 진행한 후인 Undistorted_Front_View.jpg의 4 개의 꼭짓점

    
    while(True):
        _, leftFrame = leftStream.read()
        leftView = leftCamera.undistort(leftFrame)
        topDown_left = leftBird.transfrom(leftView)
        gray = cv2.cvtColor(topDown_left, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Left Bird's Eye View", gray)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        if key == ord('p'): #pause
            cv2.waitKey(-1)

        # if key == ord("s"):
        #     cv2.imwrite("Capture.jpg", gray)

        for idx, img in enumerate(gray):
            type_result = laneDetector.detect_type(img)
            print("type_result", type_result)
            result, sess = laneDetector.detect_position(img)
            print("position result", result)
            sess.close()
            tf.reset_default_graph()

cap.release()
cv2.destroyAllWindows()
test()