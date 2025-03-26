#!/usr/bin/python
# -*- coding:utf8 -*-

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
import argparse
import cv2
import numpy as np
import torch
import logging
from datasets.process import get_affine_transform
from datasets.transforms import build_transforms
from datasets.process import get_final_preds, get_final_preds_coor
from posetimation.zoo import build_model
from posetimation.config import get_cfg, update_config
from DSTA_main.utils.utils_bbox import box2cs
from DSTA_main.utils.common import INFERENCE_PHASE

# Please make sure that root dir is the root directory of the project
root_dir = os.path.abspath('../')

cfg = None
args = None


def init_pose_model(args:argparse.ArgumentParser):
    logger = logging.getLogger(__name__)
    cfg = get_cfg(args)
    update_config(cfg, args)
    logger.info("load :{}".format(args.weight))
    checkpoint_dict = torch.load(args.weight)
    model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['state_dict'].items()}
    new_model = build_model(cfg, INFERENCE_PHASE)
    new_model.load_state_dict(model_state_dict)
    model = new_model.cuda()
    return model


# model = get_inference_model()
# model = model.cuda()
image_transforms = build_transforms(None, INFERENCE_PHASE)
# image_size = np.array([288, 384])
image_size = np.array([192, 256])
aspect_ratio = image_size[0] / image_size[1]

def image_preprocess(image_data: np.ndarray, center, scale, frame_num):
    # output_folder = os.path.join("../Db/bbox")
    # os.makedirs(output_folder, exist_ok=True)

    trans_matrix = get_affine_transform(center, scale, 0, image_size)
    image_data = cv2.warpAffine(image_data, trans_matrix, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
    # img_path =  os.path.join(output_folder, f"{frame_num:08d}.jpg" )
    # cv2.imwrite(img_path, image_data)
    image_data = image_transforms(image_data)
    return image_data

def inference_topdown(model, image_list: np.ndarray, person_data, frame_num :int)-> np.ndarray:
    """
        image_list : [pprev_image, prev_image, cur_image]
        person_data = {'track_id' : [pprev_bbox, prev_bbox, cur_bbox]}
        inference pose estimation
    """

    batch_size = len(person_data)  # Total number of track_ids
    concat_input_list = []
    centers = []
    scales = []
    # For each track_id, process the bounding boxes and images
    # for track_id in range(batch_size):
    #     bbox = person_data[track_id]
    #     pprev_idx = min(0, len(image_list) -1)
    #     prev_idx = min(1, len(image_list) - 1)
    #     cur_idx = min(2, len(image_list) - 1)

    #     center, scale = box2cs(bbox, aspect_ratio)
    #     # scale = scale * 1.5
    #     centers.append(center)
    #     scales.append(scale)

    #     pprev_image_data  = image_preprocess(image_list[pprev_idx], center, scale,frame_num)
    #     prev_image_data   = image_preprocess(image_list[prev_idx], center, scale, frame_num)
    #     target_image_data = image_preprocess(image_list[cur_idx], center, scale, frame_num)

    #     # Add the preprocessed images to the list
    #     pprev_image_data = pprev_image_data.unsqueeze(0)
    #     prev_image_data = prev_image_data.unsqueeze(0)
    #     target_image_data = target_image_data.unsqueeze(0)
    #     concat_input = torch.cat((pprev_image_data, prev_image_data, target_image_data), 1).cuda()
    #     concat_input_list.append(concat_input)

    for track_id in range(batch_size):
        bbox = person_data[track_id]
        # prev_idx  = max(0, len(image_list) -1)
        # cur_idx = max(1, len(image_list) - 1)
        # next_idx  = max(2, len(image_list) - 1)
        prev_idx  = 1
        cur_idx = 2
        next_idx  = 3

        center, scale = box2cs(bbox, aspect_ratio)
        # scale = scale * 1.5
        centers.append(center)
        scales.append(scale)

        # pprev_image_data  = image_preprocess(image_list[pprev_idx], center, scale,frame_num)
        # prev_image_data   = image_preprocess(image_list[prev_idx], center, scale, frame_num)
        # target_image_data = image_preprocess(image_list[cur_idx], center, scale, frame_num)

        next_image_data  = image_preprocess(image_list[next_idx], center, scale,frame_num)
        prev_image_data   = image_preprocess(image_list[prev_idx], center, scale, frame_num)
        target_image_data = image_preprocess(image_list[cur_idx], center, scale, frame_num)

        # Add the preprocessed images to the list
        next_image_data = next_image_data.unsqueeze(0)
        prev_image_data = prev_image_data.unsqueeze(0)
        target_image_data = target_image_data.unsqueeze(0)
        concat_input = torch.cat((prev_image_data, target_image_data, next_image_data), 1).cuda()
        concat_input_list.append(concat_input)


    # Convert lists to tensors
    concat_input = torch.cat(concat_input_list, 0).cuda()  # Concatenate images along batch axis
    # Set model to evaluation mode
    model.eval()

    # Perform inference
    predictions = model(concat_input)
    # print(predictions.shape)
    pred_coor = predictions.pred_jts.detach().cpu().numpy()
    score_coor = predictions.maxvals.detach().cpu().numpy()

    pred_coor = pred_coor * [3, 4]
    pred_joint, pred_conf = get_final_preds_coor(pred_coor, score_coor, centers, scales, h=4,w=3)

    pred_keypoints_list = np.concatenate([pred_joint.astype(int), pred_conf], axis=2)
    pred_instances = []
    for track_id, _ in enumerate(person_data):
        pred_instances.append([person_data[track_id],pred_keypoints_list[track_id]])
    return np.array(pred_instances)
