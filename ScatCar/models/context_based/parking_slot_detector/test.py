# coding: utf-8
from __future__ import division, print_function
import logging

logging.getLogger("tensorflow").disabled = True

import tensorflow as tf
from tqdm import trange
import time

from config import *
from utils.data_utils import get_batch_data, parse_line
from utils.misc_utils import parse_anchors, read_class_names, AverageMeter
import cv2
import numpy as np

from utils.eval_utils import get_preds_gpu

# from utils.eval_utils import (
#     evaluate_on_gpu,
#     get_preds_gpu,
#     parse_gt_rec,
#     parse_gt_quadrangle,
#     calc_park_score,
#     park_eval,
#     eval_result_file,
# )
from utils.nms_utils import cpu_nms
from utils.plot_utils import plot_one_box, plot_one_quad
from model import yolov3

FLAG_CALC_ONLY_TIME = False


def test(weight_path):
    anchors = parse_anchors("./data/anchors.txt")
    # classes = read_class_names("./data/data.names")
    class_num = 2  # len(classes)

    image = cv2.imread(
        "../../../data/bird_1.jpg",
        cv2.COLOR_RGB2GRAY,
    )
    # print(image)

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

    image_angle = tf.dtypes.cast([0], tf.float32)

    y_pred = yolo_model.predict(pred_feature_maps, image_angle)

    saver_to_restore = tf.train.Saver()
    weight_file = tf.train.latest_checkpoint("../weight_psd/fine_tuned_type_0")

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer()])
        saver_to_restore.restore(sess, weight_file)

        # start_time = time.time()
        __y_pred = sess.run([y_pred], feed_dict={is_training: False})
        pred_content = get_preds_gpu(
            sess, gpu_nms_op, pred_quads_flag, pred_scores_flag, [0], __y_pred
        )
        print("__y_pred", __y_pred)


def evaluate(weight_path, eval_file, result_file):
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    print(weight_path, eval_file, result_file)

    # args params
    anchors = parse_anchors("data/anchors.txt")
    classes = read_class_names("data/data.names")
    class_num = len(classes)
    img_cnt = len(open(eval_file, "r").readlines())

    # read filename from eval file
    # print(os.path.basename(os.path.split(weight_path)[-2]))
    # print(os.path.basename(weight_path))

    filenames = {}
    eval_f = open(eval_file, "rt")
    for line in eval_f.readlines():
        img_idx, pic_path, boxes, labels, _, _, quads, angle = parse_line(line)
        # filename = os.path.basename(pic_path)
        filenames[img_idx] = pic_path

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

    ##################
    # tf.data pipeline
    ##################
    # print(eval_file)
    val_dataset = tf.data.TextLineDataset(eval_file)
    val_dataset = val_dataset.batch(1)
    val_dataset = val_dataset.map(
        lambda x: tf.py_func(
            get_batch_data,
            [x, class_num, [INPUT_WIDTH, INPUT_HEIGHT], anchors],
            [tf.int64, tf.float32, tf.float32, tf.float32, tf.float32, tf.int64],
        ),
        num_parallel_calls=NUM_THREADS,
    )
    val_dataset.prefetch(PREFETECH_BUFFER)
    iterator = val_dataset.make_one_shot_iterator()

    image_ids, image, y_true_13, y_true_26, y_true_52, image_angle = iterator.get_next()
    y_true = [y_true_13, y_true_26, y_true_52]

    image_ids.set_shape([None])
    image.set_shape([None, INPUT_HEIGHT, INPUT_WIDTH, 3])

    for y in y_true:
        y.set_shape([None, None, None, None, None])
    image_angle.set_shape([None])
    image_angle = tf.dtypes.cast(image_angle, tf.float32)

    ##################
    # Model definition
    ##################
    yolo_model = yolov3(class_num, anchors)
    with tf.variable_scope("yolov3"):
        pred_feature_maps = yolo_model.forward(image, is_training=is_training)
    loss = yolo_model.compute_loss(pred_feature_maps, y_true, image_angle)
    y_pred = yolo_model.predict(pred_feature_maps, image_angle)

    saver_to_restore = tf.train.Saver()
    weight_file = tf.train.latest_checkpoint(weight_path)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer()])
        saver_to_restore.restore(sess, weight_file)
        today = time.localtime()
        time_part = "{:02d}{:02d}{:02d}_{:02d}{:02d}".format(
            today.tm_year, today.tm_mon, today.tm_mday, today.tm_hour, today.tm_min
        )

        print("\n----------- start to eval -----------\n")

        val_loss_total, val_loss_conf, val_loss_class, val_loss_quad = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )
        f = open(result_file, "wt")
        val_preds = []
        # gt_dict = parse_gt_quadrangle(eval_file, [INPUT_WIDTH, INPUT_HEIGHT], letterbox_resize)

        # start_time = time.time()

        for j in trange(img_cnt):
            __image_ids, __y_pred, __loss = sess.run(
                [image_ids, y_pred, loss], feed_dict={is_training: False}
            )
            pred_content = get_preds_gpu(
                sess, gpu_nms_op, pred_quads_flag, pred_scores_flag, __image_ids, __y_pred
            )

            if not FLAG_CALC_ONLY_TIME:

                pic_path = filenames[j]
                f.write("%d %s " % (j, pic_path))

                if len(pred_content) > 0:
                    for img_id, x_min, y_min, x_max, y_max, score, label, quad in pred_content:
                        f.write(
                            "%d %.8f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f "
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
                        # gt_quads = gt_dict[img_id]
                        # park_score = calc_park_score(quad, gt_quads)

                f.write("\n")

                val_preds.extend(pred_content)
                val_loss_total.update(__loss[0])
                val_loss_conf.update(__loss[1])
                val_loss_class.update(__loss[2])
                val_loss_quad.update(__loss[3])

        # end_time = time.time()
        # print("Inference time {} sec".format(end_time - start_time))

        f.close()
    sess.close()
    tf.reset_default_graph()
    # if not FLAG_CALC_ONLY_TIME:
    #     eval_result_file(eval_file, result_file)


print("WEIGHT_PATH", WEIGHT_PATH)
test(WEIGHT_PATH)
