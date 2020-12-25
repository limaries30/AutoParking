# coding: utf-8
from __future__ import division, print_function
import logging

logging.getLogger("tensorflow").disabled = True

import tensorflow as tf


import time

from .config import *
from .utils.misc_utils import parse_anchors
import cv2
import numpy as np

from .utils.eval_utils import get_preds_gpu
from .utils.nms_utils import gpu_nms
from .model import yolov3


def detect_slot(
    angle,
    weight_path="models/context_based/weight_psd/fine_tuned_type_0",
    image=None,
):
    anchors = parse_anchors("models/context_based/parking_slot_detector/data/anchors.txt")
    class_num = 2
    if image is None:
        return None
    image = tf.to_float(image[np.newaxis, :])
    # setting placeholders
    is_training = tf.placeholder(dtype=tf.bool, name="phase_train")
    handle_flag = tf.placeholder(tf.string, [], name="iterator_handle_flag")
    # pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
    pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
    pred_quads_flag = tf.placeholder(tf.float32, [1, None, None])

    gpu_nms_op = gpu_nms(
        pred_quads_flag,
        pred_scores_flag,
        class_num,
        NMS_TOPK,
        THRESHOLD_OBJ,
        THRESHOLD_NMS,
        apply_rotate=True,
    )

    yolo_model = yolov3(class_num, anchors)
    with tf.variable_scope("yolov3"):
        pred_feature_maps = yolo_model.forward(image, is_training=is_training)

    gpu_nms_op = gpu_nms(
        pred_quads_flag,
        pred_scores_flag,
        class_num,
        NMS_TOPK,
        THRESHOLD_OBJ,
        THRESHOLD_NMS,
        apply_rotate=True,
    )

    image_angle = tf.dtypes.cast(angle, tf.float32)

    y_pred = yolo_model.predict(pred_feature_maps, image_angle)

    saver_to_restore = tf.train.Saver()
    weight_file = tf.train.latest_checkpoint(weight_path)  # ../weight_psd/fine_tuned_type_0

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer()])
        saver_to_restore.restore(sess, weight_file)

        # start_time = time.time()
        __y_pred = sess.run([y_pred], feed_dict={is_training: False})

        pred_content = get_preds_gpu(
            sess, gpu_nms_op, pred_quads_flag, pred_scores_flag, [0], __y_pred
        )

        for img_id, x_min, y_min, x_max, y_max, score, label, quad in pred_content:
            print(
                "lable: %d score: %.8f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f "
                % (
                    label,
                    score,
                    quad[0],
                    quad[1],
                    quad[2],
                    quad[3],
                    quad[4],
                    quad[5],
                    quad[6],
                    quad[7],
                )
            )

        return pred_content, sess


def test(
    weight_path="models/context_based/weight_psd/fine_tuned_type_0",
    image_dir="data/bird_1.jpg",
):

    image = cv2.imread(image_dir, cv2.COLOR_RGB2GRAY)
    detect_slot(image=image)
