import numpy as np
import pandas as pd
from utils.one_euro_filter import OneEuroFilter
from utils.timer import FPS_Timer
import os
import sys
import numpy as np
from mmengine.logging import print_log
from scipy.signal import savgol_filter
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from utils.timer import FPS_Timer
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "tracker"))

try:
    from mmdet.apis import inference_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

class PoseEstimater:
    def __init__(self, model=None):
        self.model = model
        self.person_df = pd.DataFrame()
        self.pre_person_df = pd.DataFrame()
        self.person_id = None
        self.kpt_id = None
        self.fps = None
        self.person_data = []
        self.processed_frames = set()
        self.fps_timer = FPS_Timer()
        self.smooth_filter = OneEuroFilter()
        self.is_detect = False
        self.kpt_buffer = []
        self.joints = {
            "coco": {
                "keypoints": {
                    0: "鼻子",
                    1: "左眼",
                    2: "右眼",
                    3: "左耳",
                    4: "右耳",
                    5: "左肩",
                    6: "右肩",
                    7: "左手肘",
                    8: "右手肘",
                    9: "左手腕",
                    10: "右手腕",
                    11: "左髋",
                    12: "右髋",
                    13: "左膝",
                    14: "右膝",
                    15: "左腳踝",
                    16: "右腳踝"
                },
                "skeleton_links": [
                    [0, 1], [0, 2], [1, 3], [2, 4], # 頭
                    #軀幹
                    [5, 7], [7, 9],                 #左手
                    [6, 8], [8, 10],                #右手
                    [11, 13], [13, 15],   #左腿
                    [12, 14], [14, 16],   #右腿
                ],
                "left_points_indices": [[5, 7], [7, 9], [11, 13], [13, 15]],  # Indices of left hand, leg, and foot keypoints
                "right_points_indices": [[6, 8], [8, 10], [12, 14], [14, 16]]  # Indices of right hand, leg, and foot keypoints
            },
            "haple":{
                "keypoints": {
                    0: "鼻子",
                    1: "左眼",
                    2: "右眼",
                    3: "左耳",
                    4: "右耳",
                    5: "左肩",
                    6: "右肩",
                    7: "左肘",
                    8: "右肘",
                    9: "左腕",
                    10: "右腕",
                    11: "左髖",
                    12: "右髖",
                    13: "左膝",
                    14: "右膝",
                    15: "左踝",
                    16: "右踝",
                    17: "頭部",
                    18: "頸部",
                    19: "臀部",
                    20: "左大腳趾",
                    21: "右大腳趾",
                    22: "左小腳趾",
                    23: "右小腳趾",
                    24: "左腳跟",
                    25: "右腳跟"
                },
                "skeleton_links":[
                    [0, 1], [0, 2], [1, 3], [2, 4], # 頭
                    [5, 18], [6, 18], [17, 18],[18, 19],#軀幹
                    [5, 7], [7, 9],                 #左手
                    [6, 8], [8, 10],                #右手
                    [19, 11], [11, 13], [13, 15],   #左腿
                    [19, 12], [12, 14], [14, 16],   #右腿
                    [20, 24], [22, 24], [15, 24],   #左腳
                    [21, 25], [23, 25], [16, 25]    #右腳
                ],
                
                "left_points_indices": [[5, 18], [5, 7], [7, 9],[19, 11], [11, 13], [13, 15], [20, 24], [22, 24], [15, 24]],  # Indices of left hand, leg, and foot keypoints
                "right_points_indices": [[6, 18], [6, 8], [8, 10], [19, 12], [12, 14], [14, 16], [21, 25], [23, 25], [16, 25]],  # Indices of right hand, leg, and foot keypoints
                "angle_dict":{
                    # 'l_elbow_angle': [5, 7, 9],
                    '右手肘': [6, 8, 10]
                    # 'l_shoulder_angle': [18, 5, 7],
                    # 'r_shoulder_angle': [18, 6, 8],
                    # 'l_knee_angle': [11, 13, 15],
                    # 'r_knee_angle': [12, 14, 16]
                }
            },
        }

    def merge_person_data(self, pred_instances, person_ids: list, frame_num: int = None):
        person_bboxes = pred_instances['bboxes']
        if frame_num is None:
            self.person_data = []

        for person, pid, bbox in zip(pred_instances, person_ids, person_bboxes):
            keypoints_data = np.hstack((
                np.round(person['keypoints'][0], 2),
                np.round(person['keypoint_scores'][0], 2).reshape(-1, 1),
                np.full((len(person['keypoints'][0]), 1), False, dtype=bool)
            ))

            new_kpts = np.full((len(self.joints['haple']['keypoints']), keypoints_data.shape[1]), 0.9)
            new_kpts[:26] = keypoints_data
            person_info = {
                'person_id': pid,
                'bbox': bbox,
                'keypoints': new_kpts
            }

            if frame_num is not None:
                person_info['frame_number'] = frame_num

            self.person_data.append(person_info)

        return pd.DataFrame(self.person_data)
  
    def detect_kpt(self, image:np.ndarray, frame_num:int = None):
        if not self.is_detect:
            return image, pd.DataFrame(), 0
        fps = 0
        if frame_num not in self.processed_frames:
            self.fps_timer.tic()
            pred_instances, person_ids = self.process_one_image(self.model, image, select_id=self.person_id)
            average_time = self.fps_timer.toc()
            fps = int(1/max(average_time, 0.00001))
            self.person_df = self.merge_person_data(pred_instances, person_ids, frame_num)
            self.smooth_kpt(person_ids, frame_num)
        
        if frame_num is not None:
            self.processed_frames.add(frame_num)

        if self.kpt_id is not None:
            if frame_num is not None:
                self.kpt_buffer = self.update_kpt_buffer(frame_num)
            else:
                person_data = self.get_person_df_data(is_select=True, is_kpt=True)
                if person_data is not None:
                    keypoint = person_data[self.kpt_id][:2]  # 確保取出的是有效的 [x, y] 數據
                    self.kpt_buffer.append(keypoint)
        return image, self.person_df, fps

    def smooth_kpt(self, person_ids:list, frame_num=None):
        # 如果是即時處理，則初始化前一幀的數據，否則依賴 frame_slider 進行處理
        if frame_num is not None:
            curr_frame = frame_num
            if curr_frame == 0:
                return  # 初始幀，無需處理
            pre_frame_num = curr_frame - 1
        
        # 用於即時處理時的前一幀數據
        if frame_num is None and self.pre_person_df.empty:
            self.pre_person_df = self.person_df.copy()

        # 當前幀無數據時，跳過處理
        if self.person_df.empty:
            return
        
        for person_id in person_ids:
            # 如果使用 frame_slider，根據前後幀數據進行處理
            if frame_num is not None:
                pre_person_data = self.person_df.loc[(self.person_df['frame_number'] == pre_frame_num) &
                                                    (self.person_df['person_id'] == person_id)]
                curr_person_data = self.person_df.loc[(self.person_df['frame_number'] == curr_frame) &
                                                    (self.person_df['person_id'] == person_id)]
            # 即時處理時，使用 self.pre_person_df 作為前幀數據
            else:
                pre_person_data = self.pre_person_df.loc[self.pre_person_df['person_id'] == person_id]
                curr_person_data = self.person_df.loc[self.person_df['person_id'] == person_id]
            
            if curr_person_data.empty or pre_person_data.empty:
                continue  # 當前幀或前幀沒有該 person_id 的數據，跳過
            
            pre_kpts = pre_person_data.iloc[0]['keypoints']
            curr_kpts = curr_person_data.iloc[0]['keypoints']
            smoothed_kpts = []
            
            for pre_kpt, curr_kpt in zip(pre_kpts, curr_kpts):
                pre_kptx, pre_kpty = pre_kpt[0], pre_kpt[1]
                curr_kptx, curr_kpty, curr_conf, curr_label = curr_kpt[0], curr_kpt[1], curr_kpt[2], curr_kpt[3]
                
                if all([pre_kptx != 0, pre_kpty != 0, curr_kptx != 0, curr_kpty != 0]):
                    curr_kptx = self.smooth_filter(curr_kptx, pre_kptx)
                    curr_kpty = self.smooth_filter(curr_kpty, pre_kpty)
                
                smoothed_kpts.append([curr_kptx, curr_kpty, curr_conf, curr_label])
            
            # 更新當前幀的數據
            self.person_df.at[curr_person_data.index[0], 'keypoints'] = smoothed_kpts
        
        # 如果是即時處理，則更新前幀數據
        if frame_num is None:
            self.pre_person_df = self.person_df.copy()

    def process_one_image(self, model, img, select_id=None):
        """
        處理單張圖像，進行物件偵測、跟蹤和姿態估計。

        Args:
            model (dict): 包含檢測器、姿態估計器和跟蹤器的模型。
            img (np.ndarray): 輸入的圖像。
            select_id (int, optional): 選擇指定的追蹤ID，如果為None，則處理所有目標。

        Returns:
            Tuple: 預測的姿態實例和在線的追蹤ID。
        """

        # 從模型中提取組件和參數
        detect_args = model["Detector"]["args"]
        detector = model["Detector"]["detector"]
        test_pipeline = model["Detector"]["test_pipeline"]
        pose_estimator = model["Pose Estimator"]["pose estimator"]
        tracker = model["Tracker"]["tracker"]

        # 進行物件偵測
        result = inference_detector(detector, img, test_pipeline=test_pipeline)
        pred_instances = result.pred_instances
        det_result = pred_instances[pred_instances.scores > detect_args.score_thr].cpu().numpy()

        # 篩選指定類別的邊界框
        bboxes = det_result.bboxes[det_result.labels == detect_args.det_cat_id]
        scores = det_result.scores[det_result.labels == detect_args.det_cat_id]
        bboxes = bboxes[nms(np.hstack((bboxes, scores[:, None])), detect_args.nms_thr), :4]

        # 將新偵測的邊界框更新到跟蹤器
        online_targets = tracker.update(
            np.hstack((bboxes, np.full((bboxes.shape[0], 2), [0.9, 0]))), img.copy()
        )

        # 過濾出有效的邊界框和追蹤ID
        online_bbox, online_ids = self.filter_valid_targets(online_targets, select_id)

        # 姿態估計
        pose_results = inference_topdown(pose_estimator, img, np.array(online_bbox))
        data_samples = merge_data_samples(pose_results)

        return data_samples.get('pred_instances', None), online_ids

    def filter_valid_targets(self, online_targets, select_id=None):
        """
        過濾出有效的追蹤目標。

        Args:
            online_targets (List): 所有在線追蹤的目標。
            select_id (int, optional): 選擇指定的追蹤ID。

        Returns:
            Tuple: 有效的邊界框和追蹤ID。
        """
        valid_bbox = []
        valid_ids = []

        for target in online_targets:
            x1, y1, w, h = target.tlwh
            if (w * h > 10) and (select_id is None or target.track_id == select_id):
                valid_bbox.append([x1, y1, x1 + w, y1 + h])
                valid_ids.append(target.track_id)

        return valid_bbox, valid_ids

    def correct_person_id(self, before_correct_id:int, after_correct_id:int):
        if self.person_df.empty:
            return
    
        if (before_correct_id not in self.person_df['person_id'].unique()) or (after_correct_id not in self.person_df['person_id'].unique()):
            return

        if (before_correct_id in self.person_df['person_id'].unique()) and (after_correct_id in self.person_df['person_id'].unique()):
            for i in range(0, max(self.processed_frames)):
                condition_1 = (self.person_df['frame_number'] == i) & (self.person_df['person_id'] == before_correct_id)
                self.person_df.loc[condition_1, 'person_id'] = after_correct_id

    def set_person_id(self, person_id):
        self.person_id = person_id
        print(self.person_id)

    def set_kpt_id(self, kpt_id):
        self.kpt_id = kpt_id
        print(self.kpt_id)

    def set_detect(self, status:bool):
        self.is_detect = status

    def get_pre_person_df(self):
        return self.pre_person_df

    def update_kpt_buffer(self, frame_num:int, window_length=17, polyorder=2):
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
        
    def get_joint_dict(self):
        return self.joints

    def get_kpt_buffer(self):
        return self.kpt_buffer
    
    def get_person_df_data(self, frame_num=None, is_select=False, is_kpt=False):
        if self.person_df.empty:
            return pd.DataFrame()
        condition = pd.Series([True] * len(self.person_df))  # 初始條件設為全為 True
        if frame_num is not None:
            condition &= (self.person_df['frame_number'] == frame_num)
        
        if is_select and self.person_id is not None:
            condition &= (self.person_df['person_id'] == self.person_id)
 
        data = self.person_df.loc[condition]
        
        if data.empty:
            return None
        
        if is_kpt:
            data = data['keypoints'].iloc[0]

        return data
    
    def get_person_id(self):
        return self.person_id

    def set_processed_data(self, person_df:pd.DataFrame):
        if person_df.empty:
            return
        
        self.person_df = person_df
        self.processed_frames = {frame_num for frame_num in self.person_df['frame_number'] if frame_num != 0}

    def reset(self):
        self.person_df = pd.DataFrame()
        self.pre_person_df = pd.DataFrame()
        self.person_id = None
        self.kpt_id = None
        self.fps = None
        self.person_data = []
        self.processed_frames = set()
        self.fps_timer = FPS_Timer()
        self.smooth_filter = OneEuroFilter()
        self.is_detect = False
        self.kpt_buffer = []