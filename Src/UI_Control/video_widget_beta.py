from PyQt5.QtWidgets import *
# from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtGui import QPainter, QPen, QColor, QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QPointF
import numpy as np
import sys
import cv2
import os
from video_ui import Ui_video_widget
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
import cv2
import numpy as np
from utils.cv_thread import VideoToImagesThread
from utils.util import DataType
from utils.set_parser import set_detect_parser, set_tracker_parser
from utils.timer import Timer
from utils.vis_image import  draw_bbox
from utils.vis_pose import draw_points_and_skeleton, joints_dict, draw_tracking_skeleton
from utils.store import save_video
from topdown_demo_with_mmdet import process_one_image
from image_demo import detect_image
from mmcv.transforms import Compose
from mmengine.logging import print_log
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加相對於當前文件的父目錄的子目錄到系統路徑
sys.path.append(os.path.join(current_dir, "..", "tracker"))
from pathlib import Path
from tracker.mc_bot_sort import BoTSORT
from tracker.kalman_filter import KalmanFilter_Skeleton
from tracker.tracking_utils.timer import Timer
from mmpose.apis import init_model as init_pose_estimator
from mmpose.utils import adapt_mmdet_pipeline
from utils.one_euro_filter import OneEuroFilter
import pyqtgraph as pg
# 設置背景和前景顏色

# from pyqtgraph import LabelItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


class PoseVideoTabControl(QWidget):
    def __init__(self):
        super(PoseVideoTabControl, self).__init__()
        self.ui = Ui_video_widget()
        self.ui.setupUi(self)
        self.init_var()
        self.bind_ui()
        self.init_model()

    def bind_ui(self):
        self.ui.select_checkbox.setDisabled(True)
        self.ui.load_original_video_btn.clicked.connect(
            lambda: self.load_video(self.ui.video_name_label, self.db_path + "/videos/"))
        self.ui.load_processed_video_btn.clicked.connect(
            lambda: self.load_video(self.ui.video_name_label, self.db_path + "/videos/"))
        
        self.ui.play_btn.clicked.connect(self.play_btn_clicked)
        self.ui.back_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        )
        self.ui.forward_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        )
        self.ui.frame_slider.valueChanged.connect(self.analyze_frame)
        self.ui.video_keypoint_table.cellActivated.connect(self.on_cell_clicked)
        self.ui.video_frame_view.mousePressEvent = self.mousePressEvent
        self.ui.id_correct_btn.clicked.connect(self.correct_person_id)
        self.ui.start_code_btn.clicked.connect(self.toggle_detect)
        self.ui.select_checkbox.clicked.connect(self.toggle_select)
             
    def init_model(self):
        self.detector = init_detector(
            self.detect_args.det_config, self.detect_args.det_checkpoint, device=self.detect_args.device)
        self.detector.cfg.test_dataloader.dataset.pipeline[
            0].type = 'mmdet.LoadImageFromNDArray'
        self.detector_test_pipeline = Compose(self.detector.cfg.test_dataloader.dataset.pipeline)
        self.pose_estimator = init_pose_estimator(
            self.detect_args.pose_config,
            self.detect_args.pose_checkpoint,
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=self.detect_args.draw_heatmap)))
        )
        self.tracker = BoTSORT(self.tracker_args, frame_rate=30.0)
        self.smooth_filter = OneEuroFilter()
        self.kf_list = [KalmanFilter_Skeleton() for _ in range(26)]
        self.timer = Timer()


    def init_var(self):
        self.db_path = f"../../Db"
        self.is_play=False
        self.is_analyze = False
        self.is_detect = False
        
        self.select_person_id = -1
        self.processed_images=-1
        self.fps = 30
        self.video_images=[]
        self.video_path = ""
        self.is_threading=False
        self.video_scene = QGraphicsScene()
        self.video_scene.clear()
        self.correct_kpt_idx = 0
        self.video_name = ""
        self.start_frame_num = 0
        self.end_frame_num = 0
        self.processed_frames = set()
        self.person_df = pd.DataFrame()
        self.tracking_kpt = None
        self.person_data = []
        self.label_kpt = False
        self.select_frame = 0
        self.detect_args = set_detect_parser()
        self.tracker_args = set_tracker_parser()
        self.kpts_dict = joints_dict()['haple']['keypoints']

    def load_video(self, label_item, path: str, value_filter=None, mode=DataType.VIDEO):
        self.init_var()
        if self.ui.select_checkbox.isChecked():
            self.ui.select_checkbox.click()
        # 判斷是資料夾還是檔案
        if mode == DataType.FOLDER:
            data_path = QFileDialog.getExistingDirectory(self, mode.value['tips'], path)
        else:
            name_filter = value_filter or mode.value['filter']
            data_path, _ = QFileDialog.getOpenFileName(None, mode.value['tips'], path, name_filter)

        # 檢查是否有選取檔案或資料夾
        if not data_path:
            return

        # 更新 UI 顯示
        label_item.setText(os.path.basename(data_path))
        label_item.setToolTip(data_path)

        self.video_path = data_path
        label_item.setText("讀取影片中...")
        # 啟動影像處理執行緒
        self.v_t = VideoToImagesThread(self.video_path)
        self.v_t.emit_signal.connect(self.video_to_frame)
        self.v_t.start()

    def load_process_video(self, label_item, path: str, value_filter=None, mode=DataType.VIDEO):
        self.init_var()
        if self.ui.select_checkbox.isChecked():
            self.ui.select_checkbox.click()
        # 判斷是資料夾還是檔案
        if mode == DataType.FOLDER:
            data_path = QFileDialog.getExistingDirectory(self, mode.value['tips'], path)
        else:
            name_filter = value_filter or mode.value['filter']
            data_path, _ = QFileDialog.getOpenFileName(None, mode.value['tips'], path, name_filter)

        # 檢查是否有選取檔案或資料夾
        if not data_path:
            return

        # 更新 UI 顯示
        label_item.setText(os.path.basename(data_path))
        label_item.setToolTip(data_path)

        self.video_path = data_path
        label_item.setText("讀取影片中...")
        
        # 啟動影像處理執行緒
        self.v_t = VideoToImagesThread(self.video_path)
        self.v_t.emit_signal.connect(self.video_to_frame)
        self.v_t.start()

    def show_image(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView): 
        scene.clear()
        image = image.copy()
        image = cv2.circle(image, (0, 0), 10, (0, 0, 255), -1)
        w, h = image.shape[1], image.shape[0]
        bytesPerline = 3 * w
        qImg = QImage(image, w, h, bytesPerline, QImage.Format_RGB888).rgbSwapped()   
        pixmap = QPixmap.fromImage(qImg)
        scene.addPixmap(pixmap)
        GraphicsView.setScene(scene)
        GraphicsView.setAlignment(Qt.AlignLeft)
        GraphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def video_to_frame(self, video_images, fps, count):
        self.total_images = count
        self.ui.frame_slider.setMinimum(0)
        self.ui.frame_slider.setMaximum(count - 1)
        self.ui.frame_slider.setValue(0)
        self.ui.frame_num_label.setText(f'0/{count-1}')
        self.video_images = video_images
        self.show_image(self.video_images[0], self.video_scene, self.ui.video_frame_view)
        self.ui.video_resolution_label.setText( "(0,0) -" + f" {self.video_images[0].shape[1]} x {self.video_images[0].shape[0]}")
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.ui.video_name_label.setText(self.video_name)
        self.close_thread(self.v_t)
        self.fps = fps

    def close_thread(self, thread):
        thread.stop()
        thread = None
        self.is_threading=False
   
    def play_frame(self, start_num=0):
        for i in range(start_num, self.total_images):
            if not self.is_play:
                break
            if i > self.processed_images:
                self.processed_images = i
            self.ui.frame_slider.setValue(i)
            # to the last frame ,stop playing
            if i == self.total_images - 1 and self.is_play:
                self.play_btn_clicked()
            # time.sleep(0.1)
            cv2.waitKey(15)

    def merge_keypoint_datas(self, pred_instances):
        return [
            np.hstack(
                (
                    np.round(person['keypoints'][0], 2),
                    np.round(person['keypoint_scores'][0], 2).reshape(-1, 1),
                    np.full((len(person['keypoints'][0]), 1), False, dtype=bool)
                )
            )
            for person in pred_instances
        ]

    def merge_person_datas(self, frame_num, person_ids, person_bboxes, person_kpts):
        for pid, bbox, kpts in zip(person_ids, person_bboxes, person_kpts):
            new_kpts = np.zeros((len(self.kpts_dict),kpts.shape[1]))
            # haple
            new_kpts[:26] = kpts
            new_kpts[26:, 2] = 0.9
            self.person_data.append({
                'frame_number': frame_num,
                'person_id': pid,
                'bbox': bbox,
                'keypoints': new_kpts
            })
        self.person_df = pd.DataFrame(self.person_data)

    def play_btn_clicked(self):
        if self.video_path == "":
            QMessageBox.warning(self, "無法開始播放", "請先讀取影片!")
            return
        self.is_play = not self.is_play
        if self.is_play:
            self.ui.play_btn.setText("||")
            self.play_frame(self.ui.frame_slider.value())
        else:
            self.ui.play_btn.setText("▶︎")

    def update_person_df(self):
        person_id = self.select_person_id
        # 获取当前帧数
        frame_num = self.ui.frame_slider.value()
        # 获取表格中的数据并更新到 DataFrame 中
        for kpt_idx in range(self.ui.video_keypoint_table.rowCount()):
            kpt_name = self.ui.video_keypoint_table.item(kpt_idx, 0).text()
            kpt_x = float(self.ui.video_keypoint_table.item(kpt_idx, 1).text())
            kpt_y = float(self.ui.video_keypoint_table.item(kpt_idx, 2).text())
            # 更新 DataFrame 中对应的值
            self.person_df.loc[(self.person_df['frame_number'] == frame_num) &
                                (self.person_df['person_id'] == person_id), 'keypoints'].iloc[0][kpt_idx][:2] = [kpt_x, kpt_y]
        
        self.update_frame()

    def analyze_frame(self):
        frame_num = self.ui.frame_slider.value()

        self.ui.frame_num_label.setText(
            f'{frame_num}/{self.total_images - 1}')

        # no image to analyze
        if len(self.video_images) <= 0:
            return
        
        image = self.video_images[frame_num].copy()
    
        if self.ui.frame_slider.value() == (self.total_images-1):
            self.ui.play_btn.click()
            save_video(self.video_name, self.video_images, self.person_df, select_id=self.select_person_id)


        if frame_num not in self.processed_frames and self.is_detect:
            self.detect_kpt(image, frame_num)
            if self.select_person_id != -1:
                self.import_data_to_table(self.select_person_id, frame_num)

        self.update_frame()
                 
    def update_frame(self):
        curr_person_df, frame_num= self.obtain_curr_data()
        image = self.video_images[frame_num].copy()
        
        if not curr_person_df.empty and frame_num in self.processed_frames:
            #haple
            if self.ui.video_show_skeleton_checkBox.isChecked():
                image = draw_points_and_skeleton(image, curr_person_df, joints_dict()['haple']['skeleton_links'], 
                                                points_color_palette='gist_rainbow', skeleton_palette_samples='jet',
                                                points_palette_samples=10, confidence_threshold=0.3)
            if self.select_person_id != -1:
                image = draw_tracking_skeleton(image, self.tracking_kpt, joints_dict()['haple']['skeleton_links'], 
                                                points_color_palette='gist_rainbow', skeleton_palette_samples='jet',
                                                points_palette_samples=10, confidence_threshold=0.3)
            if self.ui.video_show_bbox_checkbox.isChecked():
                image = draw_bbox(curr_person_df, image)

        # 将原始图像直接显示在 QGraphicsView 中
        self.show_image(image, self.video_scene, self.ui.video_frame_view)

    def detect_kpt(self,image,frame_num:int):
        self.timer.tic()
        pred_instances, person_ids = process_one_image(self.detect_args,image,self.detector,
                                                       self.detector_test_pipeline,self.pose_estimator,
                                                       self.tracker, select_id= self.select_person_id)
        average_time = self.timer.toc()
        fps= int(1/max(average_time,0.00001))
        if fps <10:
            self.ui.video_fps_info_label.setText(f"0{fps}")
        else:
            self.ui.video_fps_info_label.setText(f"{fps}")
        
        person_kpts = self.merge_keypoint_datas(pred_instances)
        person_bboxes = pred_instances['bboxes']
        self.merge_person_datas(frame_num, person_ids, person_bboxes, person_kpts)
        self.smooth_kpt(person_ids)
        if self.select_person_id != -1:
            self.predict_kpt()
        self.processed_frames.add(frame_num)

    def obtain_curr_data(self):
        curr_person_df = pd.DataFrame()
        frame_num = self.ui.frame_slider.value()
        if not self.person_df.empty:
            curr_person_df = self.person_df.loc[(self.person_df['frame_number'] == frame_num)]
        return curr_person_df, frame_num

    def toggle_detect(self):
        self.is_detect = not self.is_detect
        self.ui.select_checkbox.setEnabled(True)
        self.ui.play_btn.click()

    def toggle_select(self):
        if not self.ui.select_checkbox.isChecked():
            self.select_person_id = -1
        else:
            self.ui.play_btn.click()
            
    def clear_table_view(self):
        # 清空表格視圖
        self.ui.video_keypoint_table.clear()
        # 設置列數
        self.ui.video_keypoint_table.setColumnCount(4)
        # 設置列標題
        title = ["關節點", "X", "Y", "有無更改"]
        self.ui.video_keypoint_table.setHorizontalHeaderLabels(title)
        # 將列的對齊方式設置為左對齊
        header = self.ui.video_keypoint_table.horizontalHeader()
        for i in range(4):
            header.setDefaultAlignment(Qt.AlignLeft)

    def import_data_to_table(self, person_id, frame_num):
        # 清空表格視圖
        self.clear_table_view()

        # 獲取特定人員在特定幀的數據
        person_data = self.person_df.loc[(self.person_df['frame_number'] == frame_num) & (self.person_df['person_id'] == person_id)]

        if person_data.empty:
            # print("未找到特定人員在特定幀的數據")
            self.clear_table_view()
            self.ui.select_checkbox.click()
            return

        # 確保表格視圖大小足夠
        num_keypoints = len(self.kpts_dict)
        if self.ui.video_keypoint_table.rowCount() < num_keypoints:
            self.ui.video_keypoint_table.setRowCount(num_keypoints)

        # 將關鍵點數據匯入到表格視圖中
        for kpt_idx, kpt in enumerate(person_data['keypoints'].iloc[0]): 
            kptx, kpty, kpt_label = kpt[0], kpt[1], kpt[3]
            kpt_name = self.kpts_dict[kpt_idx]
            kpt_name_item = QTableWidgetItem(str(kpt_name))
            kptx_item = QTableWidgetItem(str(np.round(kptx,1)))
            kpty_item = QTableWidgetItem(str(np.round(kpty,1)))
            if kpt_label :
                kpt_label_item = QTableWidgetItem("Y")
            else:
                kpt_label_item = QTableWidgetItem("N")
            kpt_name_item.setTextAlignment(Qt.AlignRight)
            kptx_item.setTextAlignment(Qt.AlignRight)
            kpty_item.setTextAlignment(Qt.AlignRight)
            kpt_label_item.setTextAlignment(Qt.AlignRight)
            self.ui.video_keypoint_table.setItem(kpt_idx, 0, kpt_name_item)
            self.ui.video_keypoint_table.setItem(kpt_idx, 1, kptx_item)
            self.ui.video_keypoint_table.setItem(kpt_idx, 2, kpty_item)
            self.ui.video_keypoint_table.setItem(kpt_idx, 3, kpt_label_item)

    def on_cell_clicked(self, row, column):
        self.correct_kpt_idx = row
        self.label_kpt = True
    
    def send_to_table(self, kptx, kpty, kpt_label):
        kptx_item = QTableWidgetItem(str(kptx))
        kpty_item = QTableWidgetItem(str(kpty))
        if kpt_label :
            kpt_label_item = QTableWidgetItem("Y")
        else:
            kpt_label_item = QTableWidgetItem("N")
        kptx_item.setTextAlignment(Qt.AlignRight)
        kpty_item.setTextAlignment(Qt.AlignRight)
        kpt_label_item.setTextAlignment(Qt.AlignRight)
        self.ui.video_keypoint_table.setItem(self.correct_kpt_idx, 1, kptx_item)
        self.ui.video_keypoint_table.setItem(self.correct_kpt_idx, 2, kpty_item)
        self.ui.video_keypoint_table.setItem(self.correct_kpt_idx, 3, kpt_label_item)
        self.update_person_df()

    def mousePressEvent(self, event):
        if self.label_kpt:
            pos = event.pos()
            scene_pos = self.ui.video_frame_view.mapToScene(pos)
            kptx, kpty = scene_pos.x(), scene_pos.y()
            kpt_label = 1
            if event.button() == Qt.LeftButton:
                self.send_to_table(kptx, kpty,kpt_label)
            elif event.button() == Qt.RightButton:
                kptx, kpty = 0, 0
                self.send_to_table(kptx, kpty, 0)
            self.label_kpt = False
            self.update_frame()

        if self.ui.select_checkbox.isChecked():
            pos = event.pos()
            scene_pos = self.ui.video_frame_view.mapToScene(pos)
            x, y = scene_pos.x(), scene_pos.y()
            if event.button() == Qt.LeftButton:
                self.person_id_selector(x, y)
                self.init_kalman_filter()
            if not self.is_play:
                self.ui.play_btn.click()
            
    def smooth_kpt(self,person_ids:list):
        for person_id in person_ids:
            pre_frame_num = 0
            person_kpt = self.person_df.loc[(self.person_df['person_id'] == person_id)]['keypoints']
            if len(person_kpt) > 0 and self.start_frame_num == 0 :
                self.start_frame_num = self.ui.frame_slider.value()
            curr_frame = self.ui.frame_slider.value()
            if curr_frame != 0:
                pre_frame_num = curr_frame - 1
            pre_person_data = self.person_df.loc[(self.person_df['frame_number'] == pre_frame_num) &
                                                (self.person_df['person_id'] == person_id)]
            curr_person_data = self.person_df.loc[(self.person_df['frame_number'] == curr_frame) &
                                                (self.person_df['person_id'] == person_id)]
            if not curr_person_data.empty and not pre_person_data.empty:
                pre_kpts = pre_person_data.iloc[0]['keypoints']
                curr_kpts = curr_person_data.iloc[0]['keypoints']
                smoothed_kpts = []
                for pre_kpt, curr_kpt in zip(pre_kpts, curr_kpts): 
                    pre_kptx, pre_kpty = pre_kpt[0], pre_kpt[1]
                    curr_kptx , curr_kpty, curr_conf, curr_label = curr_kpt[0], curr_kpt[1], curr_kpt[2], curr_kpt[3]
                    if pre_kptx != 0 and pre_kpty != 0 and curr_kptx != 0 and curr_kpty !=0:
                        curr_kptx = self.smooth_filter(curr_kptx, pre_kptx)
                        curr_kpty = self.smooth_filter(curr_kpty, pre_kpty)
                    smoothed_kpts.append([curr_kptx, curr_kpty, curr_conf, curr_label])
                self.person_df.at[curr_person_data.index[0], 'keypoints'] = smoothed_kpts

    def keyPressEvent(self, event):
        if event.key() == ord('D') or event.key() == ord('d'):
            self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        elif event.key() == ord('A') or event.key() == ord('a'):
            self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        else:
            super().keyPressEvent(event)
       
    def correct_person_id(self):
        if self.person_df.empty:
            return
        before_correct_id = self.ui.before_correct_id.value()
        after_correct_id = self.ui.after_correct_id.value()
        print(self.person_df['person_id'].unique())
        if (before_correct_id not in self.person_df['person_id'].unique()) or (after_correct_id not in self.person_df['person_id'].unique()):
            return

        frame_num = self.ui.frame_slider.value()
        if (before_correct_id in self.person_df['person_id'].unique()) and (after_correct_id in self.person_df['person_id'].unique()):
            for i in range(0, max(self.processed_frames)):
                condition_1 = (self.person_df['frame_number'] == i) & (self.person_df['person_id'] == before_correct_id)
                self.person_df.loc[condition_1, 'person_id'] = after_correct_id
        self.update_frame()

    def person_id_selector(self, x, y):
        curr_person_df, _= self.obtain_curr_data()

        if curr_person_df.empty:
            return
        
        selected_id = None
        max_area = -1

        for _, row in curr_person_df.iterrows():
            person_id = row['person_id']
            bbox = row['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            if x1 <= x <= x2 and y1 <= y <= y2:
                w = x2 - x1
                h = y2 - y1
                area = w * h

                if area > max_area:
                    max_area = area
                    selected_id = person_id

        self.select_person_id = selected_id
        print(self.select_person_id)

    def init_kalman_filter(self):
        curr_person_df, _= self.obtain_curr_data()
        print("init_kalmanfilter")
        if curr_person_df.empty:
            return
        
        kpts_list = curr_person_df.loc[(self.person_df['person_id'] == self.select_person_id)]['keypoints'].iloc[0]
        extracted_kpts = [kpt[:2] for kpt in kpts_list]
        print(kpts_list)
        self.mean_cov_list = [kf.initiate(np.array(kpt)) for kf, kpt in zip(self.kf_list, extracted_kpts)]
        print("kpts_list")
        print(extracted_kpts)

    def predict_kpt(self):
        self.mean_cov_list = [kf.predict(mean, cov) for kf, (mean, cov) in zip(self.kf_list, self.mean_cov_list)]

        curr_person_df, _= self.obtain_curr_data()
        if curr_person_df.empty:
            return
        new_kpts_list = curr_person_df.loc[(self.person_df['person_id'] == self.select_person_id)]['keypoints'].iloc[0]
        extracted_kpts = [kpt[:2] for kpt in new_kpts_list]
        self.mean_cov_list = [kf.update(mean, cov, np.array(kpt)) for kf, (mean, cov), kpt in zip(self.kf_list, self.mean_cov_list, extracted_kpts)]
        predicted_kpts = [mean[:2] for mean, _ in self.mean_cov_list]
        predicted_kpts_list = [[kpt[1], kpt[0], 0.9] for kpt in predicted_kpts]
        self.tracking_kpt = np.array(predicted_kpts_list)
        
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PoseVideoTabControl()
    window.show()
    sys.exit(app.exec_())