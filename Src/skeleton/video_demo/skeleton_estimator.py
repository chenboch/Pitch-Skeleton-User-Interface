import numpy as np
from skeleton.model.wrapper import Wrapper
from .skeleton_processor import *
from skeleton.utils import FPSTimer
import logging
import queue
from torch.profiler import profile, ProfilerActivity
import polars as pl

class PoseEstimater(object):
    def __init__(self, wrapper: Wrapper =None):
        self.logger = logging.getLogger(self.__class__.__name__)  # 獲取當前類的日誌對象
        self.logger.info("PoseEstimater initialized with wrapper.")
        self.detector = wrapper.detector
        self.tracker = wrapper.tracker
        self.pose2d_estimator = wrapper.pose2d_estimator
        self._model_name =  self.pose2d_estimator.model_name
        self.pose3d_estimator = wrapper.pose3d_estimator
        self._person_df = pl.DataFrame()
        self._bbox_buffer = []
        self._track_id = None
        self._joint_id = None
        self._is_detect = False
        self._pitch_hand_id = 10
        self.processed_frames = set()
        self.fps_timer = FPSTimer()
        self.image_buffer = queue.Queue(3)
        self.kpt_buffer = []

    def detect_keypoints(self, image:np.ndarray, frame_num:int = None):
        if self.image_buffer.full():  # 如果队列满了
            self.image_buffer.get()  # 弹出队列最前面的元素
        self.image_buffer.put(image)  # 将新元素加入队列

        if not self._is_detect:
            return 0
        self.fps_timer.tic()

        if frame_num not in self.processed_frames:
            if frame_num % 1 == 0:
                # self.fps_timer.tic()
                bboxes = self.detector.process_image(image)
                # self.fps_timer.toc()
                # print(f"tracking time: {self.fps_timer.time_interval}, fps: {int(self.fps_timer.fps) if int(self.fps_timer.fps)  < 100 else 0}")

                online_targets = self.tracker.process_bbox(image, bboxes)
                online_bbox, track_ids = filter_valid_targets(online_targets, self._track_id)
                self._bbox_buffer = [online_bbox, track_ids]
            else:
                online_bbox, track_ids = self._bbox_buffer

            if len(online_bbox) == 0 or len(track_ids) == 0:
                self.processed_frames.add(frame_num)
                self.fps_timer.toc()
                return  int(self.fps_timer.fps) if int(self.fps_timer.fps)  < 100 else 0
            # self.fps_timer.tic()
            pred_instances = self.pose2d_estimator.process_image(np.array(list(self.image_buffer.queue)), online_bbox, frame_num)
            # self.fps_timer.toc()
            # print(f"tracking time: {self.fps_timer.time_interval}, fps: {int(self.fps_timer.fps) if int(self.fps_timer.fps)  < 100 else 0}")

            new_person_df = merge_person_data(pred_instances, track_ids, self.pose2d_estimator.model_name,frame_num)
            new_person_df = smooth_keypoints(self._person_df, new_person_df, track_ids)
            self._person_df = pl.concat([self._person_df, new_person_df])
            self.processed_frames.add(frame_num)
        self.fps_timer.toc()
        if self._joint_id is not None and self._track_id is not None:
            self.kpt_buffer = update_keypoint_buffer(self._person_df, self._track_id, self._joint_id, frame_num)
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
        # if load_df != self._person_df:
        self._person_df = load_df
        self.processed_frames = {frame_num for frame_num in self._person_df['frame_number']}
        self.logger.info("讀取資料的狀態: %s", not load_df.is_empty())

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
            return pl.DataFrame([])

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