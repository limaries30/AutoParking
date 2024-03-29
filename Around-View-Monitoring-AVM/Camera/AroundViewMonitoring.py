#AVM 구성 코드
#frontview, middleview, backview

import cv2
import numpy as np
import imutils
from Camera.Undistortion import UndistortFisheye
from Camera.PerspectiveTransformation import EagleView
from Camera.Stitcher import stitchTwoImages
import time

class avm:
    def __init__(self):
        self.__frontCamera = UndistortFisheye("Front_Camera")
        self.__backCamera = UndistortFisheye("Back_Camera")

        self.__frontEagle = EagleView()
        self.__backEagle = EagleView()
        # self.__frontEagle.setDimensions((149, 195), (439, 207), (528, 380), (37, 374))
        # self.__backEagle.setDimensions((164, 229), (469, 229), (588, 430), (45, 435))
        self.__frontEagle.setDimensions((186, 195), (484, 207), (588, 402), (97, 363)) #fisheyecalibration을 진행한 후인 Undistorted_Front_View.jpg의 4 개의 꼭짓점
        self.__backEagle.setDimensions((171, 240), (469, 240), (603, 452), (52, 441))

        self.__middleView = None
        self.__counter = 0

        self.stitcher = stitchTwoImages("Bottom2Upper") #어떻게 활용해야할까???
        self.upper = None
        self.bottom = None
    
    def runAVM(self, frontFrame, backFrame):
        frontView = self.__frontCamera.undistort(frontFrame)
        topDown_Front = self.__frontEagle.transfrom(frontView)
        backView = self.__backCamera.undistort(backFrame)
        topDown_Back = self.__backEagle.transfrom(backView)
        topDown_Back = cv2.flip(topDown_Back, 1)

        topDown_Front , topDown_Back = self.__reScale(topDown_Front, topDown_Back)
        # stitchingResult = self.__startStitching(topDown_Front)
        middleView = self.__getMiddleView(topDown_Front)
        birdView = np.vstack((topDown_Front, middleView, topDown_Back))
        return birdView
    
    def __reScale(self, topDown_Front, topDown_Back):
        width_FrontView = topDown_Front.shape[1]
        width_BackView = topDown_Back.shape[1]
        height_FrontView = topDown_Front.shape[0]
        height_BackView = topDown_Back.shape[0]

        if width_FrontView > width_BackView:
            newWidth = width_BackView
            ratio = width_BackView/width_FrontView
            newHeight = int(ratio * height_FrontView)
            topDown_Front = cv2.resize(topDown_Front, (newWidth, newHeight))
        else:
            newWidth = width_FrontView
            ratio = width_FrontView/width_BackView
            newHeight = int(ratio * height_BackView)
            topDown_Back = cv2.resize(topDown_Back, (newWidth, newHeight))
        
        return topDown_Front, topDown_Back
    
    def __getMiddleView(self, topDown_Front):
        # the length of the image represents the distance in front or back of the car
        height_FrontView = topDown_Front.shape[0]
        if self.__middleView is None:
            realHeight_FrontView = 13 # unit is cm
            realHeight_MiddleView = 29.5 # unit is cm
            ratio = int(height_FrontView/realHeight_FrontView)
            height_MiddleView = int(realHeight_MiddleView * ratio)
            width_MiddleView = int(topDown_Front.shape[1])  
            self.__middleView = np.zeros((height_MiddleView, width_MiddleView, 3), np.uint8)
            # print(ratio)
        # else:
        #     #  self.__middleView[0:stitchingResult.shape[0], :]

        return self.__middleView

    def __startStitching(self, accView): #어떻게 활용해야할까???
        if self.bottom is None:
            self.bottom = accView
            return None
        else:
            # time.sleep(0.5)
            self.upper = accView
            self.bottom = self.stitcher.stitch(self.bottom, self.upper)
            cv2.imshow("Result", self.bottom)
            height = accView.shape[0]
            return self.bottom[height:self.bottom.shape[0], :]