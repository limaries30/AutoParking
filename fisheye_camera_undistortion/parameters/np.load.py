import numpy as np

dataD=np.load('./fisheye_camera_undistortion/parameters/D.npy')
dataDims=np.load('./fisheye_camera_undistortion/parameters/Dims.npy') # Dims = dimension = display resolution
dataK=np.load('./fisheye_camera_undistortion/parameters/K.npy')

print(dataD, dataDims, dataK, sep='\n')