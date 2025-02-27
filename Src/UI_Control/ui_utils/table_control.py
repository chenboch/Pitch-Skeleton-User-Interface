from PyQt5.QtWidgets import QTableWidgetItem, QTableWidget
from PyQt5.QtCore import Qt
import numpy as np
from skeleton.datasets import halpe26_keypoint_info,posetrack_keypoint_info


class KeypointTable:
    def __init__(self, table_widget: QTableWidget, pose_estimater):
        self.kpt_table = table_widget
        self.pose_estimater = pose_estimater
        # if pose_estimater.model_name == 'vit-pose':
        #     self.kpt_dict = halpe26_keypoint_info["keypoints"]
        # else:
        self.kpt_dict = posetrack_keypoint_info["keypoints"]
        self.correct_kpt_idx = None
        self.label_kpt = False
        self.clearTableView()

    def importDataToTable(self, frame_num: int):
        """將關鍵點數據導入表格視圖"""
        # 清空表格
        self.clearTableView()

        # 獲取人物 ID
        person_id = self.pose_estimater.track_id
        if person_id is None:
            return

        # 獲取指定幀數的數據
        person_data = self.pose_estimater.get_person_df(frame_num=frame_num, is_select=True)
        if person_data.is_empty():
            self.clearTableView()
            return

        # 獲取關鍵點信息
        num_keypoints = len(self.kpt_dict)

        # 設置表格行數
        if self.kpt_table.rowCount() < num_keypoints:
            self.kpt_table.setRowCount(num_keypoints)

        # 提取關鍵點數據並導入表格
        keypoints_list = person_data["keypoints"].to_list()[0]
        for kpt_idx, kpt in enumerate(keypoints_list):
            kptx, kpty, kpt_label = kpt[0], kpt[1], kpt[3]  # 獲取 x, y, label
            kpt_name = self.kpt_dict[kpt_idx]  # 對應的關鍵點名稱
            self.setTableItem(kpt_idx, kpt_name, kptx, kpty, kpt_label)


    def setTableItem(self, row: int, kpt_name: str, kptx: float, kpty: float, kpt_label: bool):
        """Set items in the table for the given row."""
        kpt_name_item = QTableWidgetItem(str(kpt_name))
        kptx_item = QTableWidgetItem(str(int(kptx)))
        kpty_item = QTableWidgetItem(str(int(kpty)))
        kpt_label_item = QTableWidgetItem("Y" if kpt_label else "N")

        for item in [kpt_name_item, kptx_item, kpty_item, kpt_label_item]:
            item.setTextAlignment(Qt.AlignRight)

        if kpt_name != "":
            self.kpt_table.setItem(row, 0, kpt_name_item)
        self.kpt_table.setItem(row, 1, kptx_item)
        self.kpt_table.setItem(row, 2, kpty_item)
        self.kpt_table.setItem(row, 3, kpt_label_item)

    def clearTableView(self):
        self.kpt_table.clear()
        self.kpt_table.setColumnCount(4)
        title = ["關節點", "X", "Y", "有無更改"]
        self.kpt_table.setHorizontalHeaderLabels(title)
        header = self.kpt_table.horizontalHeader()
        for i in range(4):
            header.setDefaultAlignment(Qt.AlignLeft)

    def onCellClicked(self, row:int, column:int):
        print("row:"+str(row))
        self.correct_kpt_idx = row
        self.label_kpt = True

    def sendToTable(self, kptx: float, kpty: float, kpt_label: int, frame_num: int):
        """Send corrected keypoint data back to the table and update the pose estimator."""
        self.setTableItem(self.correct_kpt_idx, "", kptx, kpty, kpt_label)
        self.pose_estimater.update_person_df(kptx, kpty, frame_num, self.correct_kpt_idx)
        self.label_kpt = False