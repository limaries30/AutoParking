#AVM 구성 코드

import cv2
import numpy as np
import imutils
from Camera.Undistortion import UndistortFisheye
from Camera.PerspectiveTransformation import EagleView
# from Camera.Stitcher import stitchTwoImages
import time

class avm:
    def __init__(self):
        self.__leftCamera = UndistortFisheye("left_Camera")
        self.__rightCamera = UndistortFisheye("right_Camera")

        self.__leftEagle = EagleView()
        self.__rightEagle = EagleView()
        # self.__frontEagle.setDimensions((149, 195), (439, 207), (528, 380), (37, 374))
        # self.__backEagle.setDimensions((164, 229), (469, 229), (588, 430), (45, 435))
        
        #reset left/right setDimensions
        self.__leftEagle.setDimensions((186, 195), (484, 207), (588, 402), (97, 363)) #fisheyecalibration을 진행한 후인 Undistorted_Front_View.jpg의 4 개의 꼭짓점
        self.__rightEagle.setDimensions((171, 240), (469, 240), (603, 452), (52, 441))
        # self.__leftEagle.setDimensions((186, 195), (484, 207), (588, 402), (97, 363)) #fisheyecalibration을 진행한 후인 Undistorted_Front_View.jpg의 4 개의 꼭짓점
        # self.__rightEagle.setDimensions((171, 240), (469, 240), (603, 452), (52, 441))

        self.__middleView = None
        self.__counter = 0

        # self.stitcher = stitchTwoImages("Bottom2Upper") #어떻게 활용해야할까???
        # self.upper = None
        # self.bottom = None
    
    def runAVM(self, leftFrame, rightFrame):
        leftView = self.__leftCamera.undistort(leftFrame)
        topDown_left = self.__leftEagle.transfrom(leftView)
        rightView = self.__rightCamera.undistort(rightFrame)
        topDown_right = self.__rightEagle.transfrom(rightView)
        # topDown_Back = cv2.flip(topDown_Back, 1) #flip left/right

        topDown_left , topDown_right = self.__reScale(topDown_left, topDown_right)
        # stitchingResult = self.__startStitching(topDown_Front)
        middleView = self.__getMiddleView(topDown_left)
        birdView = np.hstack((topDown_left, middleView, topDown_right))
        return birdView
    
    def __reScale(self, topDown_left, topDown_right):
        width_leftView = topDown_left.shape[1]
        width_rightView = topDown_right.shape[1]
        height_leftView = topDown_left.shape[0]
        height_rightView = topDown_right.shape[0]
        if height_leftView > height_rightView:
            newHeight = height_rightView
            ratio = height_rightView/height_leftView
            newWidth = int(ratio * width_leftView)
            topDown_left = cv2.resize(topDown_left, (newWidth, newHeight))
        else:
            newHeight = height_leftView
            ratio = height_leftView/height_rightView
            newWidth = int(ratio * width_rightView)
            topDown_right = cv2.resize(topDown_right, (newWidth, newHeight))
        
        return topDown_left, topDown_right
    
    def __getMiddleView(self, topDown_left):
        # the length of the image represents the distance in front or back of the car
        width_leftView = topDown_left.shape[1]
        if self.__middleView is None:
            realWidth_leftView = 13 # unit is cm
            realWidth_MiddleView = 29.5 # unit is cm
            ratio = int(width_leftView/realWidth_leftView)
            width_MiddleView = int(realWidth_MiddleView * ratio)
            height_MiddleView = int(topDown_left.shape[0])  
            self.__middleView = np.zeros((height_MiddleView, width_MiddleView//2, 3), np.uint8)
            # print(ratio)
        # else:
        #     #  self.__middleView[0:stitchingResult.shape[0], :]

        return self.__middleView

    # def __startStitching(self, accView): #어떻게 활용해야할까???
    #     if self.bottom is None:
    #         self.bottom = accView
    #         return None
    #     else:
    #         # time.sleep(0.5)
    #         self.upper = accView
    #         self.bottom = self.stitcher.stitch(self.bottom, self.upper)
    #         cv2.imshow("Result", self.bottom)
    #         height = accView.shape[0]
    #         return self.bottom[height:self.bottom.shape[0], :]

