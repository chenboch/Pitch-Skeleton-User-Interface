import numpy as np
import polars as pl
from ..model.wrapper import Wrapper
import queue
from .skeleton_processor import *
from ..utils import FPSTimer
from torch.profiler import profile, ProfilerActivity
import logging

class PoseLifter(object):
    def __init__(self, wrapper: Wrapper =None,  model_name: str = "vit-pose"):
        self.logger = logging.getLogger(self.__class__.__name__)  # 獲取當前類的日誌對象
        self.logger.info("PoseLifter initialized with wrapper.")
        self.detector = wrapper.detector
        self.tracker = wrapper.tracker
        self.pose2d_estimator = wrapper.pose2d_estimator
        self._model_name = self.pose2d_estimator.model_name
        self.pose3d_estimator = wrapper.pose3d_estimator
        self._person_df = pl.DataFrame()
        self._track_id = None
        self._joint_id = None
        self._is_detect = False
        self._pitch_hand_id = 10
        self.processed_frames = set()
        self.fps_timer = FPSTimer()
        self._bbox_buffer = []
        self.image_buffer = queue.Queue(3)
        self.kpt_buffer = []

    def detect_keypoints(self, image:np.ndarray, frame_num:int = None):

        for track_id in track_ids:
            # 從 new_person_df 中篩選該 track_id 的平滑數據
            person_data = new_person_df.filter(pl.col('track_id') == track_id)

            if person_data.height == 0:  # 如果該 track_id 無數據，跳過
                continue

            # 提取平滑後的關鍵點
            # smoothed_keypoints = np.array(person_data.select('keypoints')[0, 0])[:, :2]
            keypoints_list = person_data.select('keypoints').to_numpy()[0][0]
            # smoothed_keypoints = smoothed_keypoints.reshape(-1, 26)
            smoothed_keypoints = np.array([kp[:2] for kp in keypoints_list])
            # 更新到 pose_results 的 pred_instances 中
            smoothed_keypoints = smoothed_keypoints[:17] if len(smoothed_keypoints) == 20 else smoothed_keypoints
            smoothed_keypoints_tensor = torch.tensor(smoothed_keypoints, dtype=torch.float64)
        for pred_instance in pred_instances:
            # if pred_instance['track_id'] == track_id:  # 確保是正確的 track_id
            pred_instance['keypoints'][0] = smoothed_keypoints_tensor
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            pose_results = update_pose_results(new_person_df, pred_instances, track_ids)
            pose_results.track_id = torch.from_numpy(np.array(track_ids))
            pred_3d_pred_instances = self.pose3d_estimator.process_pose3d(pose_results, track_ids, image.shape)
            # self.fps_timer.toc()
            # print(f"pose3d time: {self.fps_timer.time_interval}, fps: {int(self.fps_timer.fps) if int(self.fps_timer.fps)  < 100 else 0}")
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            new_person_df = merge_3d_data(new_person_df, pred_3d_pred_instances, track_ids)
            self._person_df = new_person_df  if self._person_df.is_empty() else pl.concat([self._person_df, new_person_df])
            self.processed_frames.add(frame_num)


        if self._joint_id is not None:
            self.kpt_buffer = update_keypoint_buffer(self.person_df, self._track_id, self._joint_id, frame_num)

        self.fps_timer.toc()
        return  int(self.fps_timer.fps) if int(self.fps_timer.fps)  < 100 else 0

    @property
    def model_name(self):
        return self._model_name


    @model_name.setter
    def model_name(self, model_name):
        self._model_name = model_name


    @property
    def track_id(self):
        """獲取當前追蹤的 track_id。"""
        return self._track_id

    @track_id.setter
    def track_id(self, value):
        """設置追蹤的 track_id，同時打印日誌。"""
        if value != self._track_id:
            self._track_id = value
            self.logger.info("Person ID set to: %d", self._track_id)

    @property
    def joint_id(self):
        """獲取當前追蹤的 joint_id。"""
        return self._joint_id

    @joint_id.setter
    def joint_id(self, value):
        """設置追蹤的joint_id，同時打印日誌。"""
        if value != self._joint_id:
            self._joint_id = value
            self.logger.info("當前關節點: %d", self._joint_id)

    @property
    def pitch_hand_id(self):
        """獲取當前追蹤的 joint_id。"""
        return self._pitch_hand_id

    @pitch_hand_id.setter
    def pitch_hand_id(self, value):
        """設置追蹤的joint_id，同時打印日誌。"""
        if value != self._pitch_hand_id:
            self._pitch_hand_id = value
            self.logger.info("當前投手關節點: %d", self._pitch_hand_id)

    @property
    def is_detect(self):
        """獲取當前偵測的狀態。"""
        return self._is_detect

    @is_detect.setter
    def is_detect(self, status:bool):
        """設置當前偵測的狀態，同時打印日誌。"""
        if status != self._is_detect:
            self._is_detect = status
            self.logger.info("當前偵測的狀態: %d", self._is_detect)

    @property
    def person_df(self):
        return self._person_df

    @person_df.setter
    def person_df(self, load_df:pl.DataFrame):
        if load_df.is_empty():
            self.logger.info("讀取資料的狀態: %s", not load_df.is_empty())
            return
        self._person_df = load_df
        self.processed_frames = {frame_num for frame_num in self._person_df['frame_number']}
        self.logger.info("讀取資料的狀態: %s", not load_df.is_empty())
        print(self.person_df['track_id'].to_list()[0])
        exit()


    def get_person_df(self, frame_num=None, is_select=False, is_kpt=False) ->pl.DataFrame:
        if self._person_df.is_empty():
            return pl.DataFrame([])

        # 條件篩選
        condition = pl.Series([True] * len(self._person_df))
        if frame_num is not None:
            condition &= self._person_df["frame_number"] == frame_num

        if is_select and self._track_id is not None:
            condition &= self._person_df["track_id"] == self._track_id

        data = self._person_df.filter(condition)
        if data.is_empty():
            return None

        if is_kpt:
            data = data["keypoints"].to_list()[0]  # 獲取第一個值

        return data

    def update_person_df(self, x: float, y: float, frame_num: int, correct_kpt_idx: int):
        if self._person_df is None or self._person_df.is_empty():
            return  # 防止空 DataFrame 出錯
        update_keypoint = self._person_df.filter(
            (pl.col("frame_number") == frame_num) &
            (pl.col("track_id") == self._track_id)
        )["keypoints"][0].to_list()
        update_keypoint[correct_kpt_idx] = [x, y] + update_keypoint[correct_kpt_idx][2:]

        self._person_df = self._person_df.with_columns(
            pl.when(
                (pl.col("frame_number") == frame_num) &
                (pl.col("track_id") == self._track_id)
            )
            .then(
              pl.Series("keypoints", [update_keypoint])
            )
            .otherwise(pl.col("keypoints"))  # 如果條件不符合，保持原值
            .alias("keypoints")
        )

    def clear_keypoint_buffer(self):
        self.kpt_buffer = []

    def reset(self):
        self._person_df = pl.DataFrame()
        self._track_id = None
        self._joint_id = None
        self.processed_frames = set()
        self._is_detect = False
        self.kpt_buffer = []
