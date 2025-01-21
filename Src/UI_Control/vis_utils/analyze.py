import numpy as np
import polars as pl
from skeleton.datasets.halpe26 import halpe26_keypoint_info

class PoseAnalyzer:
    def __init__(self, pose_estimater):
        self.pose_estimater = pose_estimater
        self.angle_dict = halpe26_keypoint_info['angle_dict']
        self.analyze_df = pl.DataFrame()
        self.processed_frames = set()

    def addAnalyzeInfo(self, frame_num: int):
        """Analyze information for each frame up to the current frame."""
        if self.pose_estimater.track_id is None:
            return pl.DataFrame()
        
        person_kpt = self.pose_estimater.get_person_df(frame_num= frame_num, is_select= True,is_kpt=True)
        
        if person_kpt is None:
            return
        
        # new_analyze_data = []
       # 計算角度信息
        angle_info = self._update_analyze_information(person_kpt)

        # 展平字典，並添加 frame_number 信息
        new_analyze_data = {**{"frame_number": frame_num}, **angle_info}
        new_analyze_df = pl.DataFrame(new_analyze_data)
        if frame_num not in self.processed_frames:
            self.processed_frames.add(frame_num)
        if new_analyze_df.height > 0:
            if self.analyze_df is None:  # 檢查是否已有分析數據
                self.analyze_df = new_analyze_df
            else:
                self.analyze_df = pl.concat([self.analyze_df, new_analyze_df])
        self.analyze_df = self.analyze_df.sort("frame_number")

    def _calculate_angle(self, A, B, C):
        """Calculate the angle between three points A, B, and C."""
        BA = np.array(A) - np.array(B)
        BC = np.array(C) - np.array(B)
        dot_product = np.dot(BA, BC)
        magnitude_BA = np.linalg.norm(BA)
        magnitude_BC = np.linalg.norm(BC)
        cos_angle = dot_product / (magnitude_BA * magnitude_BC)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return int(np.degrees(angle_rad))

    def _update_analyze_information(self, person_kpt):
        """Update and return analyze information for the given keypoints."""
        info = {}
        for angle_name, kpt_list in self.angle_dict.items():
            A = person_kpt[kpt_list[0]][:2]
            B = person_kpt[kpt_list[1]][:2]
            C = person_kpt[kpt_list[2]][:2]
            info[angle_name] = self._calculate_angle(A, B, C)
        return info

    def get_frame_angle_data(self, frame_num: int = None, angle_name: str = None):   
        """
        獲取指定幀數和角度名稱的數據。
        :param frame_num: 指定幀數（可選）。
        :param angle_name: 指定角度名稱（可選）。
        :return: Tuple(pl.DataFrame, List[float] or float)
        """
        if self.analyze_df.is_empty():
            return pl.DataFrame(), []

        # 構建條件過濾
        condition = pl.Series([True] * len(self.analyze_df))
        if frame_num is not None:
            condition &= self.analyze_df["frame_number"] == frame_num

        # 過濾數據
        data = self.analyze_df.filter(condition)

        # 如果數據為空則返回
        if data.is_empty():
            return pl.DataFrame(), []

        # 如果指定了角度名稱
        if angle_name is not None:
            if frame_num is not None:
                # 提取單個幀數的角度數據
                angle_value = data.select(angle_name).to_series()[0]
                return data, angle_value
            else:
                # 提取所有幀數的指定角度數據
                frame_numbers = data["frame_number"].to_list()
                angles = data.select(angle_name).to_series().to_list()
                return frame_numbers, angles

        # 如果未指定角度名稱，返回整個過濾後的數據
        return data, []

    def reset(self):
        self.analyze_df = pl.DataFrame()
        self.processed_frames = set()

class JointAreaChecker:
    def __init__(self, image_size:tuple):
        self.image_width = image_size[0]
        self.image_height = image_size[1]

        # 設置1/5區域的邊界
        self.region_width = image_size[0]
        self.region_height = image_size[1] // 5

        # 區域左上角的坐標
        self.region_top_left = (0, 0)
        self.region_bottom_right = (self.region_width, self.region_height)

    def is_joint_in_area(self, joint_position:tuple)->bool:
        """檢查關節點是否在定義的區域內"""
        if joint_position is None:
            return False
        x, y = joint_position
        
        in_area = (self.region_top_left[0] <= x <= self.region_bottom_right[0] and
                   self.region_top_left[1] <= y <= self.region_bottom_right[1])
        return in_area