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
import cv2
import numpy as np
from utils.cv_thread import VideoToImagesThread
from utils.util import DataType
from utils.timer import FPS_Timer
from utils.vis_image import  draw_bbox, draw_video_traj, draw_angle_info
from utils.vis_graph import init_graph, update_graph
from utils.vis_pose import draw_points_and_skeleton, joints_dict
from utils.analyze import obtain_analyze_information
from utils.store import save_video
from utils.one_euro_filter import OneEuroFilter
from topdown_demo_with_mmdet import process_one_image
import pyqtgraph as pg

class PoseVideoTabControl(QWidget):
    def __init__(self, model, parent = None):
        super(PoseVideoTabControl, self).__init__(parent)
        self.ui = Ui_video_widget()
        self.ui.setupUi(self)
        self.init_var()
        self.bind_ui()
        self.model = model
        self.smooth_filter = OneEuroFilter()
        self.fps_timer = FPS_Timer()

    def bind_ui(self):
        self.ui.select_checkbox.setDisabled(True)
        self.ui.select_kpt_checkbox.setDisabled(True)
        self.ui.load_original_video_btn.clicked.connect(
            lambda: self.load_video(self.ui.video_name_label, self.db_path + "/videos/"))
        self.ui.load_processed_video_btn.clicked.connect(
            lambda: self.load_process_video(self.ui.video_name_label, self.db_path + "/videos/"))
        
        self.ui.play_btn.clicked.connect(self.play_btn_clicked)
        self.ui.back_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        )
        self.ui.forward_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        )
        self.ui.frame_slider.valueChanged.connect(self.analyze_frame)
        self.ui.keypoint_table.cellActivated.connect(self.on_cell_clicked)
        self.ui.frame_view.mousePressEvent = self.mousePressEvent
        self.ui.id_correct_btn.clicked.connect(self.correct_person_id)
        self.ui.start_code_btn.clicked.connect(self.toggle_detect)
        self.ui.select_checkbox.clicked.connect(self.toggle_select)
        self.ui.select_kpt_checkbox.clicked.connect(self.toggle_select_kpt)
             
    def init_var(self):
        self.graph =  pg.PlotWidget()
        self.angle_info_pos = None
        self.db_path = f"../../Db"
        self.is_play = False
        self.is_detect = False
        self.processed_images = -1
        self.select_person_id = None
        self.select_kpt_id = -1
        self.select_kpt_buffer = []
        self.video_images=[]
        self.video_path = ""
        self.json_path = ""
        self.is_threading = False
        self.angle_info = None
        self.video_scene = QGraphicsScene()
        self.curve_scene = QGraphicsScene()
        self.video_scene.clear()
        self.curve_scene.clear()
        self.correct_kpt_idx = 0
        self.video_name = ""
        self.processed_frames = set()
        self.person_df = pd.DataFrame()
        self.person_data = []
        self.label_kpt = False
        self.select_frame = 0
        self.kpts_dict = joints_dict()['haple']['keypoints']
        pg.setConfigOptions(foreground=QColor(113,148,116), antialias = True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

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

    def load_process_video(self, label_item, path: str, value_filter=None, mode=DataType.FOLDER):
        self.init_var()
        self.ui.select_checkbox.setEnabled(True)
        self.ui.select_kpt_checkbox.setEnabled(True)
        if self.ui.select_checkbox.isChecked():
            self.ui.select_checkbox.click()

        data_path = QFileDialog.getExistingDirectory(self, mode.value['tips'], path)

        if not data_path:
            return

        # 搜尋資料夾內的 .mp4 和 .json 檔案
        mp4_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.mp4') and not f.endswith('Sk26.mp4') ]
        json_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.json')]
        print(mp4_files)
        print(json_files)
        # 檢查是否有找到檔案
        if not mp4_files or not json_files:
            label_item.setText("未找到 .mp4 或 .json 檔案")
            return

        # 更新 UI 顯示
        label_item.setText(f"找到 {len(mp4_files)} 個影片和 {len(json_files)} 個 JSON 檔案")
        label_item.setToolTip(data_path)

        self.video_path = mp4_files[0]  # 儲存影片路徑列表
        self.json_path = json_files[0]  # 儲存 JSON 路徑列表
        label_item.setText("讀取資料夾中的檔案中...")

        try:       
            self.person_df = pd.DataFrame()
            self.person_df = pd.read_json(self.json_path)
            process_frame_nums = self.person_df['frame_number'].unique()
            self.processed_frames = sorted(set(frame_num for frame_num in process_frame_nums if frame_num!=0) )
        except Exception as e:
            print(f"加载 JSON 文件时出错：{e}")

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

    def show_graph(self, graph, scene, graphicview):
        graph.resize(graphicview.width(),graphicview.height())
        scene.addWidget(graph)
        graphicview.setScene(scene)
        graphicview.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def video_to_frame(self, video_images, fps, count):
        self.total_images = count
        self.ui.frame_slider.setMinimum(0)
        self.ui.frame_slider.setMaximum(count - 1)
        self.ui.frame_slider.setValue(0)
        self.ui.frame_num_label.setText(f'0/{count-1}')
        self.video_images = video_images
        image = self.video_images[0].copy()
        if self.ui.show_skeleton_checkbox.isChecked() and self.json_path == "":
            self.detect_kpt(image, frame_num = 0)
        self.update_frame()
        self.ui.video_resolution_label.setText( "(0,0) -" + f" {self.video_images[0].shape[1]} x {self.video_images[0].shape[0]}")
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.ui.video_name_label.setText(self.video_name)
        self.graph = init_graph(self.total_images)
        self.show_graph(self.graph, self.curve_scene, self.ui.curve_view)
        self.close_thread(self.v_t)

    def close_thread(self, thread):
        thread.stop()
        thread = None
        self.is_threading=False
   
    def play_frame(self, start_num=0):
        for i in range(start_num, self.total_images):
            self.ui.frame_slider.setValue(i)
            if not self.is_play:
                break
            if i > self.processed_images:
                self.processed_images = i
            if i == self.total_images - 1 and self.is_play:
                self.play_btn_clicked()
            cv2.waitKey(15)

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

            new_kpts = np.full((len(self.kpts_dict), keypoints_data.shape[1]), 0.9)
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

    def update_person_df(self, x, y, label):
        person_id = self.select_person_id
        # 获取当前帧数
        frame_num = self.ui.frame_slider.value()
        self.person_df.loc[(self.person_df['frame_number'] == frame_num) &
                            (self.person_df['person_id'] == person_id), 'keypoints'].iloc[0][self.correct_kpt_idx] = [x, y, 0.9, label]
        self.update_frame()

    def analyze_frame(self):
        frame_num = self.ui.frame_slider.value()

        self.ui.frame_num_label.setText(f'{frame_num}/{self.total_images - 1}')

        if not self.video_images:
            return

        image = self.video_images[frame_num].copy()

        if frame_num == self.total_images - 1:
            self.ui.play_btn.click()
            save_video(self.video_name, self.video_images, self.person_df, select_id=self.select_person_id)

        if frame_num not in self.processed_frames and self.is_detect:
            self.detect_kpt(image, frame_num = frame_num)

        if self.select_person_id:
            if self.angle_info_pos is None:
                person_kpt = self.obtain_data(frame_num,self.select_person_id, True)
                self.set_angle_info_pos(person_kpt)
            self.import_data_to_table(self.select_person_id, frame_num)
            person_data = self.obtain_data(person_id = self.select_person_id)
            self.angle_info = obtain_analyze_information(person_data, joints_dict()['haple']['angle_dict'], frame_num)
            self.graph = update_graph(self.graph, self.angle_info, frame_num)

        self.update_frame()
              
    def update_frame(self):
        frame_num = self.ui.frame_slider.value()
        curr_person_df = self.obtain_data(frame_num = frame_num, person_id = self.select_person_id)
        image = self.video_images[frame_num].copy()

        if not curr_person_df.empty and frame_num in self.processed_frames:
            if self.ui.select_checkbox.isChecked():
                curr_person_df = curr_person_df[curr_person_df['person_id'] == self.select_person_id]

            if self.ui.show_angle_checkbox.isChecked():
                image = draw_angle_info(image, self.angle_info, frame_num, self.angle_info_pos)

            if self.ui.show_skeleton_checkbox.isChecked():
                image = draw_points_and_skeleton(image, curr_person_df, 
                                                joints_dict()['haple']['skeleton_links'], 
                                                points_color_palette='gist_rainbow', 
                                                skeleton_palette_samples='jet',
                                                points_palette_samples=10, 
                                                confidence_threshold=0.3)
            
            if self.ui.show_bbox_checkbox.isChecked():
                image = draw_bbox(curr_person_df, image)

            if self.ui.select_kpt_checkbox.isChecked():
                image = draw_video_traj(image, self.person_df, self.select_person_id,
                                        self.select_kpt_id, frame_num)

        
                
        self.show_image(image, self.video_scene, self.ui.frame_view)

    def detect_kpt(self, image:np.ndarray, frame_num:int = None):
        self.fps_timer.tic()
        pred_instances, person_ids = process_one_image(self.model, image, select_id=self.select_person_id)
        average_time = self.fps_timer.toc()
        fps = int(1/max(average_time, 0.00001))
        self.ui.fps_info_label.setText(f"{fps:02}")
        self.person_df = self.merge_person_data(pred_instances, person_ids, frame_num)
        self.smooth_kpt(person_ids)
        
        if frame_num is not None:
            self.processed_frames.add(frame_num)

    def obtain_data(self, frame_num=None, person_id=None, is_kpt=False):
        condition = pd.Series([True] * len(self.person_df))  # 初始條件設為全為 True
        if frame_num is not None:
            condition &= (self.person_df['frame_number'] == frame_num)
        
        if person_id is not None:
            condition &= (self.person_df['person_id'] == person_id)
 
        data = self.person_df.loc[condition]
        
        if is_kpt:
            data = data['keypoints']
            
        return data
    
    def toggle_detect(self):
        self.is_detect = not self.is_detect
        self.ui.select_checkbox.setEnabled(True)
        self.ui.select_kpt_checkbox.setEnabled(True)
        self.ui.play_btn.click()

    def toggle_select(self):
        if not self.ui.select_checkbox.isChecked():
            self.select_person_id = None
        else:
            self.person_id_selector(0,0)

    def toggle_select_kpt(self):
        if not self.ui.select_kpt_checkbox.isChecked():
            self.select_kpt_id = -1
        else:
            self.select_kpt_id = 10

    def clear_table_view(self):
        self.ui.keypoint_table.clear()
        self.ui.keypoint_table.setColumnCount(4)
        title = ["關節點", "X", "Y", "有無更改"]
        self.ui.keypoint_table.setHorizontalHeaderLabels(title)
        header = self.ui.keypoint_table.horizontalHeader()
        for i in range(4):
            header.setDefaultAlignment(Qt.AlignLeft)

    def import_data_to_table(self, person_id:int, frame_num:int):
        self.clear_table_view()
        person_data = self.person_df.loc[(self.person_df['frame_number'] == frame_num) & (self.person_df['person_id'] == person_id)]

        if person_data.empty:
            self.clear_table_view()
            self.ui.select_checkbox.click()
            return
        
        num_keypoints = len(self.kpts_dict)
        if self.ui.keypoint_table.rowCount() < num_keypoints:
            self.ui.keypoint_table.setRowCount(num_keypoints)

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
            self.ui.keypoint_table.setItem(kpt_idx, 0, kpt_name_item)
            self.ui.keypoint_table.setItem(kpt_idx, 1, kptx_item)
            self.ui.keypoint_table.setItem(kpt_idx, 2, kpty_item)
            self.ui.keypoint_table.setItem(kpt_idx, 3, kpt_label_item)

    def on_cell_clicked(self, row, column):
        self.correct_kpt_idx = row
        self.label_kpt = True
     
    def send_to_table(self, kptx:float, kpty:float, kpt_label:int):
        kptx_item = QTableWidgetItem(str(kptx))
        kpty_item = QTableWidgetItem(str(kpty))
        if kpt_label :
            kpt_label_item = QTableWidgetItem("Y")
        else:
            kpt_label_item = QTableWidgetItem("N")
        kptx_item.setTextAlignment(Qt.AlignRight)
        kpty_item.setTextAlignment(Qt.AlignRight)
        kpt_label_item.setTextAlignment(Qt.AlignRight)
        self.ui.keypoint_table.setItem(self.correct_kpt_idx, 1, kptx_item)
        self.ui.keypoint_table.setItem(self.correct_kpt_idx, 2, kpty_item)
        self.ui.keypoint_table.setItem(self.correct_kpt_idx, 3, kpt_label_item)
        self.update_person_df(kptx, kpty, kpt_label)

    def mousePressEvent(self, event):
        pos = event.pos()
        scene_pos = self.ui.frame_view.mapToScene(pos)
        x, y = scene_pos.x(), scene_pos.y()

        if self.ui.select_checkbox.isChecked() and not self.label_kpt:
            if event.button() == Qt.LeftButton:
                self.person_id_selector(x, y)

        if self.label_kpt:
            if event.button() == Qt.LeftButton:
                self.send_to_table(x, y, 1)
            elif event.button() == Qt.RightButton:
                self.send_to_table(0, 0, 0)
            self.label_kpt = False

        self.update_frame()

    def smooth_kpt(self, person_ids:list):
        for person_id in person_ids:
            pre_frame_num = 0
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
        if (before_correct_id not in self.person_df['person_id'].unique()) or (after_correct_id not in self.person_df['person_id'].unique()):
            return

        frame_num = self.ui.frame_slider.value()
        if (before_correct_id in self.person_df['person_id'].unique()) and (after_correct_id in self.person_df['person_id'].unique()):
            for i in range(0, max(self.processed_frames)):
                condition_1 = (self.person_df['frame_number'] == i) & (self.person_df['person_id'] == before_correct_id)
                self.person_df.loc[condition_1, 'person_id'] = after_correct_id
        self.update_frame()

    def person_id_selector(self, x:float, y:float):
        curr_person_df = self.obtain_data(frame_num = self.ui.frame_slider.value())
        if curr_person_df.empty:
            return    
        selected_id = None
        max_area = -1
        for _, row in curr_person_df.iterrows():
            person_id = row['person_id']
            x1, y1, x2, y2 = map(int, row['bbox'])
            if x != 0 and y != 0:
                if not (x1 <= x <= x2 and y1 <= y <= y2):
                    continue
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                selected_id = person_id

        self.select_person_id = selected_id
        self.update_frame()
        print(self.select_person_id)

    def kpt_id_selector(self, x:float, y:float):
        def calculate_distance(point1, point2):
            return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        
        curr_person_df = self.obtain_data(self.ui.frame_slider.value())
        if curr_person_df.empty:
            return
        
        selected_id = None
        min_distance = float('inf')

        for person_kpts in curr_person_df['keypoints']:
            kpts_coords = person_kpts[:, :2].astype(int)
            distances = np.sqrt((kpts_coords[:, 0] - x)**2 + (kpts_coords[:, 1] - y)**2)
            
            min_idx = np.argmin(distances)
            if distances[min_idx] < min_distance:
                min_distance = distances[min_idx]
                selected_id = min_idx

        self.select_kpt_id = selected_id

    def set_angle_info_pos(self, person_kpt):
        person_kpt = person_kpt.iloc[0]
        x, y, _, _ = person_kpt[19]
        self.angle_info_pos = (int(x), int(y))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PoseVideoTabControl()
    window.show()
    sys.exit(app.exec_())