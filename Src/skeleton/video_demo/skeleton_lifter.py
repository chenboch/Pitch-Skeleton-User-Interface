import numpy as np
import pandas as pd
from ..model.wrapper import Wrapper

from mmpose.evaluation.functional import nms
from mmpose.structures import (PoseDataSample, merge_data_samples,
                               split_instances)
from mmpose.apis import (convert_keypoint_definition, extract_pose_sequence,
                         inference_pose_lifter_model, inference_topdown)
from .skeleton_processor import *
from ..lib import (FPSTimer, OneEuroFilter)
import logging
try:
    from mmdet.apis import inference_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

class PoseLifter(object):
    def __init__(self, wrapper: Wrapper =None):
        self.logger = logging.getLogger(self.__class__.__name__)  # 獲取當前類的日誌對象
        self.logger.info("PoseLifter initialized with wrapper.")
        self.detector = wrapper.detector
        self.tracker = wrapper.tracker
        self.pose2d_estimator = wrapper.pose2d_estimator
        self.pose3d_estimator = wrapper.pose3d_estimator
        self.person_df = pd.DataFrame()
        self._track_id = None
        self._joint_id = None
        self._is_detect = False
        self._pitch_hand_id = 10
        self.fps = None
        self.processed_frames = set()
        self.fps_timer = FPSTimer()
       
        self.kpt_buffer = []
  
    def detect_keypoints(self, image:np.ndarray, frame_num:int = None):
        if not self._is_detect:
            return 0
        fps = 0
        self.fps_timer.tic()
        if frame_num not in self.processed_frames:
            pose_results, pred_instances, track_ids = self.process_image(image)
            new_person_df = merge_person_data(pred_instances, track_ids, frame_num)
            new_person_df = smooth_keypoints(self.person_df, new_person_df, track_ids)
            pose_results = update_pose_results(new_person_df, pose_results, track_ids)
            pred_3d_pred_instances = self.process_pose3d(pose_results, track_ids, image.shape)
            new_person_df = merge_3d_data(new_person_df, pred_3d_pred_instances, track_ids)
            print(new_person_df)
            self.person_df = pd.concat([self.person_df, new_person_df], ignore_index=True)
            
            self.processed_frames.add(frame_num)
        if self._joint_id is not None:
            self.kpt_buffer = updateKptBuffer(self.person_df, self._track_id, self._joint_id, frame_num)

        average_time = self.fps_timer.toc()
        fps = int(1/max(average_time, 0.00001))
        fps = fps if fps < 100 else 0
        return fps
           
    def process_image(self, img):
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
        online_bbox, online_ids = filterValidTargets(online_targets, self._track_id)
        # 姿態估計
        pose_results = inference_topdown(self.pose2d_estimator.pose2d_estimator, img, np.array(online_bbox))
        data_samples = merge_data_samples(pose_results)
    
        return pose_results, data_samples.get('pred_instances', None), online_ids

    def process_pose3d(self ,pose_results, track_ids, img_shape):
        """
        將 2D 骨架關鍵點轉換為 3D 骨架關鍵點。

        Args:
            img_shape: 輸入影像尺寸。
            data_sample: 包含關鍵點的 2D 數據樣本。

        Returns:
            pred_3d_data_samples: 3D 預測骨架數據樣本。
        """
        # 提取數據集名稱
        pose_det_dataset_name = self.pose2d_estimator.pose2d_estimator.dataset_meta['dataset_name']
        pose_lift_dataset_name = self.pose3d_estimator.pose3d_estimator.dataset_meta['dataset_name']
        pose_lift_dataset = self.pose3d_estimator.pose3d_estimator.cfg.test_dataloader.dataset

        # 初始化 2D 骨架轉換的結果容器
        pose_est_results_list = []
        pose_est_results_converted = []

        for i, data_sample in enumerate(pose_results):
            pred_instances = data_sample.pred_instances.cpu().numpy()
            keypoints = pred_instances.keypoints
            pose_results[i].set_field(track_ids[i], 'track_id')
            # 步驟 1: 轉換關鍵點格式
            pose_est_result_converted = convert_keypoints(
                pose_results[i], keypoints, pose_det_dataset_name, pose_lift_dataset_name
            )
            pose_est_results_converted.append([pose_est_result_converted])
        pose_est_results_list.append(pose_est_results_converted)

        # 步驟 2: 提取 2D 骨架序列
        pose_seq_2d = extract_pose_sequence(
                            pose_est_results_converted,
                            frame_idx=0,
                            causal=pose_lift_dataset.get('causal', False),
                            seq_len=pose_lift_dataset.get('seq_len', 1),
                            step=pose_lift_dataset.get('seq_step', 1)
                        )

        # 步驟 3: 進行 2D-to-3D 提升

        pose_lift_results = self._lift_to_3d(pose_seq_2d, img_shape[:2])

        # 步驟 4: 後處理 3D 骨架數據
        pose_lift_results = self._postprocess_pose_lift(pose_lift_results, pose_results)
        # 合併樣本並返回結果
        pred_3d_data_samples = merge_data_samples(pose_lift_results)
        return pred_3d_data_samples.get('pred_instances', None)

    def _lift_to_3d(self, pose_seq_2d, image_size):
        """使用模型進行 2D-to-3D 提升。"""
        norm_pose_2d = not self.pose3d_estimator.pose3d_args.disable_norm_pose_2d
        return inference_pose_lifter_model(
            self.pose3d_estimator.pose3d_estimator,
            pose_seq_2d,
            image_size=image_size,
            norm_pose_2d=norm_pose_2d
        )

    def _postprocess_pose_lift(self, pose_lift_results, pose_results):
        """後處理提升的 3D 骨架數據。"""
        for idx, pose_lift_result in enumerate(pose_lift_results):
            pose_lift_result.track_id = pose_results[idx].get('track_id', 1e4)

            pred_instances = pose_lift_result.pred_instances
            keypoints = pred_instances.keypoints
            keypoint_scores = pred_instances.keypoint_scores
            if keypoint_scores.ndim == 3:
                keypoint_scores = np.squeeze(keypoint_scores, axis=1)
                pose_lift_results[idx].pred_instances.keypoint_scores = keypoint_scores
            if keypoints.ndim == 4:
                keypoints = np.squeeze(keypoints, axis=1)

            keypoints = keypoints[..., [0, 2, 1]]
            keypoints[..., 0] = -keypoints[..., 0]
            keypoints[..., 2] = -keypoints[..., 2]

            # rebase height (z-axis)
            # if not args.disable_rebase_keypoint:
            keypoints[..., 2] -= np.min(
                keypoints[..., 2], axis=-1, keepdims=True)

            pose_lift_results[idx].pred_instances.keypoints = keypoints
        pose_lift_results = sorted(
                pose_lift_results, key=lambda x: x.get('track_id', 1e4))
        
        return pose_lift_results
    
    @property
    def track_id(self):
        """獲取當前追蹤的 track_id。"""
        return self._track_id

    @track_id.setter
    def track_id(self, value):
        """設置追蹤的 track_id，同時打印日誌。"""
        if value != self._track_id:
            self._track_id = value
            self.logger.info(f"Person ID set to: {self._track_id}")

    @property
    def joint_id(self):
        """獲取當前追蹤的 joint_id。"""
        return self._joint_id

    @joint_id.setter
    def joint_id(self, value):
        """設置追蹤的joint_id，同時打印日誌。"""
        if value != self._joint_id:
            self._joint_id = value
            self.logger.info(f"當前關節點: {self._joint_id}")
    
    @property
    def pitch_hand_id(self):
        """獲取當前追蹤的 joint_id。"""
        return self._pitch_hand_id

    @pitch_hand_id.setter
    def pitch_hand_id(self, value):
        """設置追蹤的joint_id，同時打印日誌。"""
        if value != self._pitch_hand_id:
            self._pitch_hand_id = value
            self.logger.info(f"當前投手關節點: {self._pitch_hand_id}")

    @property
    def is_detect(self):
        """獲取當前偵測的狀態。"""
        return self._is_detect

    @is_detect.setter
    def is_detect(self, status:bool):
        """設置當前偵測的狀態，同時打印日誌。"""
        if status != self._is_detect:
            self._is_detect = status
            self.logger.info(f"當前偵測的狀態: {self._is_detect}")
   
    def get_person_df(self, frame_num=None, is_select=False, is_kpt=False):
        if self.person_df.empty:
            return pd.DataFrame()
        condition = pd.Series([True] * len(self.person_df))  # 初始條件設為全為 True
        if frame_num is not None:
            condition &= (self.person_df['frame_number'] == frame_num)
        
        if is_select and self._track_id is not None:
            condition &= (self.person_df['track_id'] == self._track_id)
 
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
                            (self.person_df['track_id'] == self._track_id), 'keypoints'].iloc[0][correct_kpt_idx] = [x, y, 0.9, 1]

    def clear_keypoint_buffer(self):
        self.kpt_buffer = []

    def reset(self):
        self.person_df = pd.DataFrame()
        self._track_id = None
        self._joint_id = None
        self.fps = None
        self.processed_frames = set()
        self.fps_timer = FPSTimer()
        self.smooth_filter = OneEuroFilter()
        self._is_detect = False
        self.kpt_buffer = []