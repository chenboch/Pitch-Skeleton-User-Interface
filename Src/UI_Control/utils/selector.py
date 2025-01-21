import polars as pl
import numpy as np

class PersonSelector:
    def __init__(self):
        self._selected_id = None


    def select(self,search_person_df:pl.DataFrame, x: float = 0.0, y: float = 0.0):
        """選擇最大 bbox 或指定座標範圍內的 bbox，並記錄選擇的 track_id。"""
        if search_person_df.is_empty():
            return
        if not (x == 0 and y == 0):
            print(f"{x}, {y}")
            search_person_df = search_person_df.filter(
                    (pl.col('bbox').list.get(0) <= x) & 
                    (pl.col('bbox').list.get(1) <= y) &
                    (pl.col('bbox').list.get(0) + pl.col('bbox').list.get(2) >= x ) &
                    (pl.col('bbox').list.get(3) + pl.col('bbox').list.get(1) >= y )
                
            )
        search_person_df = search_person_df.sort('area',descending = True)
        self.selected_id = search_person_df["track_id"][0]

    @property
    def selected_id(self):
        return self._selected_id
    
    @selected_id.setter
    def selected_id(self, value):
        self._selected_id = value

    def reset(self):
        self.selected_id = None
        
class KptSelector:
    def __init__(self):
        self.selected_id = None

    def select(self, search_person_df: pl.DataFrame, x: float, y: float):
        """選擇離 (x, y) 最近的關鍵點 ID"""
        if search_person_df is None or search_person_df.is_empty():
            return

        def calculate_distance(kpt_x, kpt_y):
            return np.linalg.norm([kpt_x - x, kpt_y - y])

        selected_id = None
        min_distance = float("inf")

        # 使用 Polars 的 iter_rows() 來加速 DataFrame 遍歷
        for row in search_person_df.iter_rows(named=True):
            keypoints = row.get("keypoints", [])
            if not keypoints:
                continue  # 跳過空的 keypoints

            for kpt_id, kpt in enumerate(keypoints):
                if len(kpt) < 2:  # 確保關鍵點座標存在
                    continue
                
                kpt_x, kpt_y = map(int, kpt[:2])  # 只取前兩個座標
                distance = calculate_distance(kpt_x, kpt_y)

                if distance < min_distance:
                    min_distance = distance
                    selected_id = kpt_id

        self.selected_id = selected_id


    def reset(self):
        self.selected_id = None