#fisheye image correction
#load parameters and save the undistorted image 
#input: image, parameters

import numpy as np
import cv2


def correct(img_in, k, d, dims):
    dim1 = img_in.shape[:2][::-1]
    assert dim1[0] / dim1[1] == dims[0] / dims[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, dims, cv2.CV_16SC2)
    img_out = cv2.remap(img_in, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return img_out


if __name__ == '__main__':
    Dims = tuple(np.load('./fisheye_camera_undistortion/parameters/Dims.npy'))
    K = np.load('./fisheye_camera_undistortion/parameters/K.npy')
    D = np.load('./fisheye_camera_undistortion/parameters/D.npy')

    img = cv2.imread('./fisheye_camera_undistortion/distort.jpg')
    img = correct(img, k=K, d=D, dims=Dims)
    img_resize = cv2.resize(img, dsize=(0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('', img_resize)
    cv2.waitKey(0)
    cv2.imwrite('./fisheye_camera_undistortion/undistorted.jpg', img_resize)
    cv2.destroyAllWindows()
