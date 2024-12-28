import numpy as np
import pandas as pd
import os
import sys
from ..model.wrapper import Wrapper

from mmpose.apis import inference_topdown
from mmpose.evaluation.functional import nms
from mmpose.structures import (PoseDataSample, merge_data_samples,
                               split_instances)
from mmpose.apis import (convert_keypoint_definition, extract_pose_sequence,
                         inference_pose_lifter_model, inference_topdown,
                         init_model)
from .skeleton_processor import *
from ..lib import (FPSTimer, OneEuroFilter)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "tracker"))

try:
    from mmdet.apis import inference_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

class PoseEstimater(object):
    def __init__(self, wrapper: Wrapper =None):
        self.detector = wrapper.detector
        self.tracker = wrapper.tracker
        self.pose2d_estimator = wrapper.pose2d_estimator
        self.person_df = pd.DataFrame()
        self.person_id = None
        self.kpt_id = None
        self.pitch_hand_id = 10
        self.fps = None
        self.processed_frames = set()
        self.fps_timer = FPSTimer()
        self.is_detect = False
        self.kpt_buffer = []
  
    def detect_keypoints(self, image:np.ndarray, frame_num:int = None):
        if not self.is_detect:
            return 0
        fps = 0
        self.fps_timer.tic()
        if frame_num not in self.processed_frames:
            pred_instances, person_ids = self.process_image(image, select_id=self.person_id)
            new_person_df = merge_person_data(pred_instances, person_ids, frame_num)
            self.person_df = pd.concat([self.person_df, new_person_df], ignore_index=True)
            self.person_df = smooth_keypoints(self.person_df, person_ids, frame_num)
            self.processed_frames.add(frame_num)

        if self.kpt_id is not None:
            self.kpt_buffer = updateKptBuffer(self.person_df, self.person_id, self.kpt_id, frame_num)

        average_time = self.fps_timer.toc()
        fps = int(1/max(average_time, 0.00001))
        fps = fps if fps < 100 else 0
        return fps
           
    def process_image(self, img, select_id=None):
        """
        處理單張圖像，進行物件偵測、跟蹤和姿態估計。

        Args:
            model (dict): 包含檢測器、姿態估計器和跟蹤器的模型。
            img (np.ndarray): 輸入的圖像。
            select_id (int, optional): 選擇指定的追蹤ID，如果為None，則處理所有目標。

        Returns:
            Tuple: 預測的姿態實例和在線的追蹤ID。
        """
       
        # 進行物件偵測
        
        result = inference_detector(self.detector.detector, img, test_pipeline= self.detector.detector_test_pipeline)
        
        pred_instances = result.pred_instances
        det_result = pred_instances[pred_instances.scores >self.detector.detect_args.score_thr].cpu().numpy()
        
        # 篩選指定類別的邊界框
        bboxes = det_result.bboxes[det_result.labels == self.detector.detect_args.det_cat_id]
        scores = det_result.scores[det_result.labels == self.detector.detect_args.det_cat_id]
        bboxes = bboxes[nms(np.hstack((bboxes, scores[:, None])), self.detector.detect_args.nms_thr), :4]
        # 將新偵測的邊界框更新到跟蹤器
        online_targets = self.tracker.tracker.update(
            np.hstack((bboxes, np.full((bboxes.shape[0], 2), [0.9, 0]))), img.copy()
        )
    
        # 過濾出有效的邊界框和追蹤ID
        online_bbox, online_ids = filterValidTargets(online_targets, select_id)
        # 姿態估計
        pose_results = inference_topdown(self.pose2d_estimator.pose2d_estimator, img, np.array(online_bbox))
        data_samples = merge_data_samples(pose_results)

        return data_samples.get('pred_instances', None), online_ids

    def person_id(self):
        self.person_id = person_id
        print(f'person id: {self.person_id}')

    def setKptId(self, kpt_id):
        self.kpt_id = kpt_id
        print(f'person id: {self.kpt_id}')
    
    def setPitchHandId(self,kpt_id):
        self.pitch_hand_id = kpt_id

    def setDetect(self, status:bool):
        self.is_detect = status
   
    def get_person_df(self, frame_num=None, is_select=False, is_kpt=False):
        if self.person_df.empty:
            return pd.DataFrame()
        condition = pd.Series([True] * len(self.person_df))  # 初始條件設為全為 True
        if frame_num is not None:
            condition &= (self.person_df['frame_number'] == frame_num)
        
        if is_select and self.person_id is not None:
            condition &= (self.person_df['person_id'] == self.person_id)
 
        data = self.person_df.loc[condition].copy()
        
        if data.empty:
            return None
        
        if is_kpt:
            data = data['keypoints'].iloc[0]

        return data
    
    def set_processed_data(self, person_df:pd.DataFrame):
        if person_df.empty:
            return
        self.person_df = person_df
        self.processed_frames = {frame_num for frame_num in self.person_df['frame_number']}

    def update_person_df(self, x:float, y:float,frame_num:int, correct_kpt_idx:int):
        self.person_df.loc[(self.person_df['frame_number'] == frame_num) &
                            (self.person_df['person_id'] == self.person_id), 'keypoints'].iloc[0][correct_kpt_idx] = [x, y, 0.9, 1]

    def clear_keypoint_buffer(self):
        self.kpt_buffer = []

    def reset(self):
        self.person_df = pd.DataFrame()
        self.person_id = None
        self.kpt_id = None
        self.fps = None
        self.processed_frames = set()
        self.fps_timer = FPSTimer()
        self.smooth_filter = OneEuroFilter()
        self.is_detect = False
        self.kpt_buffer = []