import numpy as np
import pandas as pd
from ..model.wrapper import Wrapper
from lib import *
from scipy.signal import savgol_filter
from mmpose.apis import inference_topdown
from mmpose.evaluation.functional import nms
from mmpose.structures import (PoseDataSample, merge_data_samples,
                               split_instances)
from mmpose.apis import (convert_keypoint_definition, extract_pose_sequence,
                         inference_pose_lifter_model, inference_topdown,)
from .skeleton_processor import *
import torch

try:
    from mmdet.apis import inference_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

class PoseEstimater:
    def __init__(self, model: Wrapper =None):
        self.model = model
        self.person_df = pd.DataFrame()
        self.pre_person_df = pd.DataFrame()
        self.person_id = None
        self.kpt_id = None
        self.pitch_hand_id = 10
        self.person_data = []
        self.processed_frames = set()
        self.fps_timer = FPSTimer()
        self.smooth_filter = OneEuroFilter()
        self.is_detect = False
        self.kpt_buffer = []

  
    def detectKpt(self, image:np.ndarray, frame_num:int = None, is_video:bool = False, is_3d:bool=False):
        if not self.is_detect:
            return image, pd.DataFrame(), 0

        fps = 0
        self.fps_timer.tic()
        if is_video: 
            #影片處理方式
            if frame_num not in self.processed_frames:
                pred_instances, person_ids = self.processImage(self.model, image, is_3d, select_id=self.person_id)
                self.person_df = self.mergePersonData(pred_instances, person_ids, frame_num)
                self.smoothKpt(person_ids, frame_num)
                self.processed_frames.add(frame_num)
            if self.kpt_id is not None:
                self.kpt_buffer = self.updateKptBuffer(frame_num)
        else:
            #real time 處理方式
            pred_instances, person_ids = self.processImage(self.model, image, is_3d, select_id=self.person_id)
            self.person_df = self.mergePersonData(pred_instances, person_ids)
            self.smoothKpt(person_ids, frame_num)
            self.pre_person_df = self.person_df.copy()
            if self.kpt_id:
                person_data = self.getPersonDf(is_select=True, is_kpt=True)
                if person_data is not None:
                    keypoint = person_data[self.kpt_id][:2]  # 確保取出的是有效的 [x, y] 數據
                    self.kpt_buffer.append(keypoint)

        average_time = self.fps_timer.toc()
        fps = int(1/max(average_time, 0.00001))
        fps = fps if fps < 100 else 0
        return fps


           
    def processImage(self, model, img, is_3d, select_id=None):
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
        
        result = inference_detector(model.detector, img, test_pipeline= model.detector_test_pipeline)
        
        pred_instances = result.pred_instances
        det_result = pred_instances[pred_instances.scores > model.detect_args.score_thr].cpu().numpy()
        
        # 篩選指定類別的邊界框
        bboxes = det_result.bboxes[det_result.labels == model.detect_args.det_cat_id]
        scores = det_result.scores[det_result.labels == model.detect_args.det_cat_id]
        bboxes = bboxes[nms(np.hstack((bboxes, scores[:, None])), model.detect_args.nms_thr), :4]
        # 將新偵測的邊界框更新到跟蹤器
        online_targets = model.tracker.update(
            np.hstack((bboxes, np.full((bboxes.shape[0], 2), [0.9, 0]))), img.copy()
        )
    
        # 過濾出有效的邊界框和追蹤ID
        online_bbox, online_ids = self.filterValidTargets(online_targets, select_id)
        # 姿態估計
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        pose_results = inference_topdown(model.pose2d_estimator, img, np.array(online_bbox))
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        data_samples = merge_data_samples(pose_results)
        if select_id and is_3d:
            
            self.process_pose3d(img, data_samples, model)

    
        return data_samples.get('pred_instances', None), online_ids

    def process_pose3d(self, img, data_sample, model):
        pose_est_results_list = []
        pose_det_dataset_name = model.pose2d_estimator.dataset_meta['dataset_name']
        pose_lift_dataset_name = model.pose3d_estimator.dataset_meta['dataset_name']
        pose_lift_dataset = model.pose3d_estimator.cfg.test_dataloader.dataset
        pose_est_results_converted = []
        pred_instances = data_sample.pred_instances.cpu().numpy()
        keypoints = pred_instances.keypoints
    
            # convert keypoints for pose-lifting
        pose_est_result_converted = PoseDataSample()
        pose_est_result_converted.set_field(
            pred_instances.clone(), 'pred_instances')
        pose_est_result_converted.set_field(
            data_sample.gt_instances.clone(), 'gt_instances')
        keypoints = convert_keypoint_definition(keypoints,
                                                pose_det_dataset_name,
                                                pose_lift_dataset_name)
        pose_est_result_converted.pred_instances.set_field(keypoints, 'keypoints')
        pose_est_result_converted.set_field(self.person_id,'track_id')
        pose_est_results_converted.append(pose_est_result_converted)
        pose_est_results_list.append(pose_est_results_converted.copy())
        pose_seq_2d = extract_pose_sequence(
            pose_est_results_list,
            frame_idx=0,
            causal=pose_lift_dataset.get('causal', False),
            seq_len=pose_lift_dataset.get('seq_len', 1),
            step=pose_lift_dataset.get('seq_step', 1))

        # conduct 2D-to-3D pose lifting
        norm_pose_2d = not model.pose3d_args.disable_norm_pose_2d
        pose_lift_results = inference_pose_lifter_model(
            model.pose3d_estimator,
            pose_seq_2d,
            image_size=img.shape[:2],
            norm_pose_2d=norm_pose_2d)
        for idx, pose_lift_result in enumerate(pose_lift_results):
            pose_lift_result.track_id = self.person_id

            pred_instances = pose_lift_result.pred_instances
            keypoints = pred_instances.keypoints
            keypoint_scores = pred_instances.keypoint_scores
            if keypoint_scores.ndim == 3:
                keypoint_scores = np.squeeze(keypoint_scores, axis=1)
                pose_lift_results[
                    idx].pred_instances.keypoint_scores = keypoint_scores
            if keypoints.ndim == 4:
                keypoints = np.squeeze(keypoints, axis=1)

            keypoints = keypoints[..., [0, 2, 1]]
            keypoints[..., 0] = -keypoints[..., 0]
            keypoints[..., 2] = -keypoints[..., 2]

            # rebase height (z-axis)
            if not model.pose3d_args.disable_rebase_keypoint:
                keypoints[..., 2] -= np.min(
                    keypoints[..., 2], axis=-1, keepdims=True)

            pose_lift_results[idx].pred_instances.keypoints = keypoints

        pose_lift_results = sorted(
            pose_lift_results, key=lambda x: x.get('track_id', 1e4))

        pred_3d_data_samples = merge_data_samples(pose_lift_results)
        pred_3d_instances = pred_3d_data_samples.get('pred_instances', None)

    def filterValidTargets(self, online_targets, select_id: int = None):
        """
        過濾出有效的追蹤目標。

        Args:
            online_targets (List): 所有在線追蹤的目標。
            select_id (int, optional): 選擇指定的追蹤ID。

        Returns:
            Tuple: 有效的邊界框和追蹤ID。
        """
        if not online_targets:
            return [], []

        # 將所有在線目標的邊界框和ID提取為兩個列表
        tlwhs = []
        track_ids = []

        for target in online_targets:
            tlwhs.append(target.tlwh)
            track_ids.append(target.track_id)

        # 將列表轉換為 NumPy array
        tlwhs = np.array(tlwhs)
        track_ids = np.array(track_ids)

        # 將數據轉為張量並放到 GPU 上
        tlwhs = torch.tensor(tlwhs, device='cuda')  # shape: (n, 4)
        track_ids = torch.tensor(track_ids, device='cuda')  # shape: (n,)

        # 計算面積 w * h
        areas = tlwhs[:, 2] * tlwhs[:, 3]  # w * h

        # 過濾面積大於 10 的邊界框
        valid_mask = areas > 10

        # 如果指定了 select_id，則進一步過濾
        if select_id is not None:
            valid_mask &= (track_ids == select_id)

        # 過濾有效的邊界框和追蹤ID
        valid_tlwhs = tlwhs[valid_mask]
        valid_track_ids = track_ids[valid_mask]

        # 將 (x1, y1, w, h) 轉為 (x1, y1, x2, y2)
        valid_bbox = torch.cat([valid_tlwhs[:, :2], valid_tlwhs[:, :2] + valid_tlwhs[:, 2:4]], dim=1)

        # 返回結果
        return valid_bbox.cpu().tolist(), valid_track_ids.cpu().tolist()

    def correct_person_id(self, before_correctId:int, after_correctId:int):
        if self.person_df.empty:
            return
    
        if (before_correctId not in self.person_df['person_id'].unique()) or (after_correctId not in self.person_df['person_id'].unique()):
            return

        if (before_correctId in self.person_df['person_id'].unique()) and (after_correctId in self.person_df['person_id'].unique()):
            for i in range(0, max(self.processed_frames)):
                condition_1 = (self.person_df['frame_number'] == i) & (self.person_df['person_id'] == before_correctId)
                self.person_df.loc[condition_1, 'person_id'] = after_correctId

    def setPersonId(self, person_id):
        self.person_id = person_id
        print(f'person id: {self.person_id}')

    def setKptId(self, kpt_id):
        self.kpt_id = kpt_id
        print(f'person id: {self.kpt_id}')
    
    def setPitchHandId(self,kpt_id):
        self.pitch_hand_id = kpt_id

    def setDetect(self, status:bool):
        self.is_detect = status

    def updateKptBuffer(self, frame_num:int, window_length=17, polyorder=2):
        filtered_df = self.person_df[
            (self.person_df['person_id'] == self.person_id) & 
            (self.person_df['frame_number'] < frame_num)
        ]
        if filtered_df.empty:
            return None
        filtered_df = filtered_df.sort_values(by='frame_number')
        kpt_buffer = []
        for kpts in filtered_df['keypoints']:
            kpt = kpts[self.kpt_id]
            if kpt is not None and len(kpt) >= 2:
                kpt_buffer.append((kpt[0], kpt[1]))
        
        # 如果緩衝區長度大於等於窗口長度，則應用Savgol濾波器進行平滑
        if len(kpt_buffer) >= window_length:
            # 確保窗口長度為奇數且不超過緩衝區長度
            if window_length > len(kpt_buffer):
                window_length = len(kpt_buffer) if len(kpt_buffer) % 2 == 1 else len(kpt_buffer) - 1
            # 確保多項式階數小於窗口長度
            current_polyorder = min(polyorder, window_length - 1)
            
            # 分別提取x和y座標
            x = np.array([point[0] for point in kpt_buffer])
            y = np.array([point[1] for point in kpt_buffer])
            
            # 應用Savgol濾波器
            x_smooth = savgol_filter(x, window_length=window_length, polyorder=current_polyorder)
            y_smooth = savgol_filter(y, window_length=window_length, polyorder=current_polyorder)
            
            # 將平滑後的座標重新打包
            smoothed_points = list(zip(x_smooth, y_smooth))
        else:
            # 緩衝區長度不足，直接使用原始座標
            smoothed_points = kpt_buffer

        return smoothed_points
    
    def getPersonDf(self, frame_num=None, is_select=False, is_kpt=False):
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
    
    def getPrePersonDf(self):
        if self.pre_person_df.empty:
            return pd.DataFrame()
        condition = pd.Series([True] * len(self.pre_person_df))  # 初始條件設為全為 True

        condition &= (self.pre_person_df['person_id'] == self.person_id)
 
        data = self.pre_person_df.loc[condition].copy()
        
        if data.empty:
            return None

        data = data['keypoints'].iloc[0][self.pitch_hand_id]
        return (data[0], data[1])
    
    def setProcessedData(self, person_df:pd.DataFrame):
        if person_df.empty:
            return
        self.person_df = person_df
        self.processed_frames = {frame_num for frame_num in self.person_df['frame_number']}

    def update_person_df(self, x:float, y:float,frame_num:int, correct_kpt_idx:int):
        self.person_df.loc[(self.person_df['frame_number'] == frame_num) &
                            (self.person_df['person_id'] == self.person_id), 'keypoints'].iloc[0][correct_kpt_idx] = [x, y, 0.9, 1]

    def clearKptBuffer(self):
        self.kpt_buffer = []

    def reset(self):
        self.person_df = pd.DataFrame()
        self.pre_person_df = pd.DataFrame()
        self.person_id = None
        self.kpt_id = None
        self.fps = None
        self.person_data = []
        self.processed_frames = set()
        self.fps_timer = FPSTimer()
        self.smooth_filter = OneEuroFilter()
        self.is_detect = False
        self.kpt_buffer = []