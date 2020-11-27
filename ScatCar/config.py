WHEEL= {
    'FRONT':{'LEFT':1,'RIGHT':2},
    'BACK':{'LEFT':1,'RIGHT':2}
}

VPS_NET = {
'MODEL_DEF': "models/vps_net/config/ps-4.cfg",
'WEIGHTS_PATH_YOLO': "weights/yolov3_4.pth",
'WEIGHTS_PATH_VPS': "weights/Customized.pth",
'CONF_THRES': 0.9,
'NMS_THRES': 0.5,
'IMG_SIZE': 416,
}

CONTEXT_RECOGNIZER_NET = {
'IMG_HEIGHT':192,
'IMG_WIDTH':64,
}