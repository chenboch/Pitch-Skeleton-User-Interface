from PyQt5.QtWidgets import *
from PyQt5.QtGui import QColor, QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QPointF, QTimer
import numpy as np
import sys
import cv2
import os
from video_ui import Ui_video_widget
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
from utils.vis_image import ImageDrawer
from utils.selector import PersonSelector, KptSelector
from cv_utils.cv_control import VideoLoader, JsonLoader
from skeleton.detect_skeleton import PoseEstimater
from utils.vis_graph import GraphPlotter
from utils.analyze import PoseAnalyzer
from utils.model import Model
import pyqtgraph as pg

class PoseVideoTabControl(QWidget):
    def __init__(self, model:Model, parent = None):
        super(PoseVideoTabControl, self).__init__(parent)
        self.ui = Ui_video_widget()
        self.ui.setupUi(self)
        self.model = model
        self.setupComponents()
        self.initVar()
        self.bindUI()
        
    def bindUI(self):
        self.ui.loadOriginalVideoBtn.clicked.connect(
            lambda: self.loadVideo(is_processed=False))
        self.ui.loadProcessedVideoBtn.clicked.connect(
            lambda: self.loadVideo(is_processed=True))
        
        self.ui.playBtn.clicked.connect(self.playBtnClicked)
        self.ui.backKeyBtn.clicked.connect(
            lambda: self.ui.frameSlider.setValue(self.ui.frameSlider.value() - 1)
        )
        self.ui.forward_key_btn.clicked.connect(
            lambda: self.ui.frameSlider.setValue(self.ui.frameSlider.value() + 1)
        )
        self.ui.frameSlider.valueChanged.connect(self.analyzeFrame)
        self.ui.KptTable.cellActivated.connect(self.onCellClicked)
        self.ui.FrameView.mousePressEvent = self.mousePressEvent
        self.ui.IdCorrectBtn.clicked.connect(self.correctId)
        self.ui.startCodeBtn.clicked.connect(self.toggle_detect)
        self.ui.selectCheckBox.stateChanged.connect(self.toggle_select)
        self.ui.showSkeletonCheckBox.stateChanged.connect(self.toggleShowSkeleton)
        self.ui.selectKptCheckBox.stateChanged.connect(self.toggleKptSelect)
        self.ui.showBboxCheckBox.stateChanged.connect(self.toggleShowBbox)
        self.ui.showAngleCheckBox.stateChanged.connect(self.toggleShowAngleInfo)

    def mousePressEvent(self, event):
        view_rect = self.ui.FrameView.rect()
        pos = event.pos()

        if not view_rect.contains(pos):
            return
        search_person_df = self.pose_estimater.getPersonDf(frame_num = self.ui.frameSlider.value())
        scene_pos = self.ui.FrameView.mapToScene(pos)
        x, y = scene_pos.x(), scene_pos.y()

        if self.ui.selectCheckBox.isChecked() and not self.label_kpt :
            if event.button() == Qt.LeftButton:
                self.person_selector.select(x, y, search_person_df)
                self.pose_estimater.setPersonId(self.person_selector.selected_id)
        
        if self.ui.selectKptCheckBox.isChecked() and not self.label_kpt :
            if event.button() == Qt.LeftButton:
                self.kpt_selector.select(x, y, search_person_df)
                self.pose_estimater.setKptId(self.kpt_selector.selected_id)

        if self.label_kpt:
            if event.button() == Qt.LeftButton:
                self.sendToTable(x, y, 1)
            elif event.button() == Qt.RightButton:
                self.sendToTable(0, 0, 0)
            self.label_kpt = False

            self.update_frame(self.ui.frameSlider.value())

    def keyPressEvent(self, event):
        if event.key() == ord('D') or event.key() == ord('d'):
            self.ui.frameSlider.setValue(self.ui.frameSlider.value() + 1)
        elif event.key() == ord('A') or event.key() == ord('a'):
            self.ui.frameSlider.setValue(self.ui.frameSlider.value() - 1)
        else:
            super().keyPressEvent(event)

    def setupComponents(self): 
        self.person_selector = PersonSelector()
        self.kpt_selector = KptSelector()
        self.pose_estimater = PoseEstimater(self.model)
        self.kpt_dict = self.pose_estimater.joints["haple"]["keypoints"]
        self.pose_analyzer = PoseAnalyzer(self.pose_estimater)
        self.graph_plotter = GraphPlotter(self.pose_analyzer)
        self.image_drawer = ImageDrawer(self.pose_estimater, self.pose_analyzer)
        self.video_loader = VideoLoader(self.image_drawer)

    def initVar(self):
        self.is_play = False
        self.view_scene = QGraphicsScene()
        self.curve_scene = QGraphicsScene()
        self.view_scene.clear()
        self.curve_scene.clear()
        self.correct_kpt_idx = 0
        self.label_kpt = False
        pg.setConfigOptions(foreground=QColor(113,148,116), antialias = True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
     
    def loadVideo(self,is_processed:bool = False):
        if self.is_play:
            self.ui.playBtn.click()
        self.reset()
        self.video_loader.loadVideo()
        self.check_video_load()

        if is_processed:
            json_loader = JsonLoader(self.video_loader.folder_path, self.video_loader.video_name)
            json_loader.load()
            self.pose_estimater.setProcessedData(json_loader.person_df)
            self.ui.showSkeletonCheckBox.setChecked(True)

    def reset(self):
        self.initVar()
        self.pose_estimater.reset()
        self.image_drawer.reset()
        self.person_selector.reset()
        self.kpt_selector.reset()
        self.pose_analyzer.reset()
        self.graph_plotter.reset()
        self.video_loader.reset()

    def check_video_load(self):
        """檢查影片是否讀取完成，並更新 UI 元素。"""
        # 檢查是否有影片名稱，若無則不執行後續操作
        if not self.video_loader.video_name:
            return
        # 若影片正在讀取中，定時檢查讀取狀況
        if self.video_loader.is_loading:
            self.ui.video_name_label.setText("讀取影片中")
            QTimer.singleShot(100, self.check_video_load)  # 每100ms 檢查一次
            return
        # 影片讀取完成後更新 UI 元素
        self._update_video_info()
        self._initialize_slider_and_labels()
        self._initialize_graph()
        self.update_frame(0)

    def _update_video_info(self):
        """更新與影片相關的資訊顯示在 UI 上。"""
        self.ui.videoNameLabel.setText(self.video_loader.video_name)
        video_size = self.video_loader.video_size
        self.ui.ResolutionLabel.setText(f"(0,0) - {video_size[0]} x {video_size[1]}")

    def _initialize_slider_and_labels(self):
        """初始化影片滑桿和相關的標籤。"""
        total_frames = self.video_loader.total_frames
        self.ui.frameSlider.setMinimum(0)
        self.ui.frameSlider.setMaximum(total_frames - 1)
        self.ui.frameSlider.setValue(0)
        self.ui.frameNumLabel.setText(f'0/{total_frames - 1}')

    def _initialize_graph(self):
        """初始化圖表和模型設定。"""
        total_frames = self.video_loader.total_frames
        self.graph_plotter._init_graph(total_frames)
        self.model.setImageSize(self.video_loader.video_size)
        self.showGraph(self.curve_scene, self.ui.CurveView)

    def showImage(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView): 
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

    def showGraph(self, scene, graphicview):
        graph = self.graph_plotter.getUpdateGraph()
        graph.resize(graphicview.width(),graphicview.height())
        scene.addWidget(graph)
        graphicview.setScene(scene)
        graphicview.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
   
    def playFrame(self, start_num=0):
        for i in range(start_num, self.video_loader.total_frames):
            if not self.is_play:
                break
            self.ui.frameSlider.setValue(i)
            if i == self.video_loader.total_frames - 1 and self.is_play:
                self.playBtnClicked()
            cv2.waitKey(15)

    def playBtnClicked(self):
        if self.video_loader.video_name == "":
            QMessageBox.warning(self, "無法播放影片", "請讀取影片!")
            return
        if self.video_loader.is_loading:
            QMessageBox.warning(self, "影片讀取中", "請稍等!")
            return
        self.is_play = not self.is_play
        self.ui.playBtn.setText("||" if self.is_play else "▶︎")
        if self.is_play:
            self.playFrame(self.ui.frameSlider.value())

    def analyzeFrame(self):
        frame_num = self.ui.frameSlider.value()
        self.ui.frameNumLabel.setText(f'{frame_num}/{len(self.video_loader.video_frames) - 1}')
        frame = self.video_loader.getVideoImage(frame_num)
        _, _, fps= self.pose_estimater.detectKpt(frame, frame_num)
        self.ui.FPSInfoLabel.setText(f"{fps:02d}")
        # if self.pose_estimater.person_id is not None:
            # self.pose_analyzer.addAnalyzeInfo(frame_num)
            # self.graph_plotter.updateGraph(frame_num)
        # self.importDataToTable(frame_num)
        self.update_frame(frame_num)
        if frame_num == self.video_loader.total_frames - 1:
            self.ui.playBtn.click()
            # self.video_loader.saveVideo()
                
    def update_frame(self, frame_num:int):
        image = self.video_loader.getVideoImage(frame_num)
        drawed_img = self.image_drawer.drawInfo(image, frame_num, self.pose_estimater.kpt_buffer)
        self.showImage(drawed_img, self.view_scene, self.ui.FrameView)
    
    def toggle_detect(self):
        self.ui.showSkeletonCheckBox.setChecked(True)
        image = self.video_loader.getVideoImage(0)
        _, _, _= self.pose_estimater.detectKpt(image, 0)
        # self.ui.selectCheckBox.setChecked(True)
        # self.ui.select_kptCheckBox.setChecked(True)
        # self.ui.show_angleCheckBox.setChecked(True)
        self.ui.playBtn.click()

    def toggle_select(self, state:int):
        if state == 2: 
            self.person_selector.select(search_person_df=self.pose_estimater.getPersonDf(frame_num=self.ui.frameSlider.value()))
            self.pose_estimater.setPersonId(self.person_selector.selected_id)
        else:
            self.pose_estimater.setPersonId(None)
        self.update_frame(self.ui.frameSlider.value())

    def toggleKptSelect(self, state:int):
        if state == 2:  
            self.pose_estimater.setKptId(10)
            self.image_drawer.setShowTraj(True)
        else:
            self.pose_estimater.setKptId(None)
            self.image_drawer.setShowTraj(False)

    def toggleShowSkeleton(self, state:int):
        if state == 2:  
            self.pose_estimater.setDetect(True)
            self.image_drawer.setShowSkeleton(True)
        else:
            self.pose_estimater.setDetect(False)
            self.image_drawer.setShowSkeleton(False)

    def toggleShowBbox(self, state:int):
        if state == 2:  
            self.image_drawer.setShowBbox(True)
        else:
            self.image_drawer.setShowBbox(False)

    def toggleShowAngleInfo(self, state:int):
        if state == 2:  
            self.image_drawer.setShowAngleInfo(True)
        else:
            self.image_drawer.setShowAngleInfo(False)

    def clearTableView(self):
        self.ui.KptTable.clear()
        self.ui.KptTable.setColumnCount(4)
        title = ["關節點", "X", "Y", "有無更改"]
        self.ui.KptTable.setHorizontalHeaderLabels(title)
        header = self.ui.KptTable.horizontalHeader()
        for i in range(4):
            header.setDefaultAlignment(Qt.AlignLeft)

    def importDataToTable(self, frame_num:int):
        self.clearTableView()
        person_id = self.pose_estimater.person_id
        if person_id is None:
            return
        person_data = self.pose_estimater.getPersonDf(frame_num=frame_num, is_select=True)
        if person_data.empty:
            self.clearTableView()
            self.ui.selectCheckBox.setChecked(False)
            return
        
        num_keypoints = len(self.kpt_dict)
        if self.ui.KptTable.rowCount() < num_keypoints:
            self.ui.KptTable.setRowCount(num_keypoints)

        for kpt_idx, kpt in enumerate(person_data['keypoints'].iloc[0]): 
            kptx, kpty, kpt_label = kpt[0], kpt[1], kpt[3]
            kpt_name = self.kpt_dict[kpt_idx]
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
            self.ui.KptTable.setItem(kpt_idx, 0, kpt_name_item)
            self.ui.KptTable.setItem(kpt_idx, 1, kptx_item)
            self.ui.KptTable.setItem(kpt_idx, 2, kpty_item)
            self.ui.KptTable.setItem(kpt_idx, 3, kpt_label_item)

    def onCellClicked(self, row, column):
        self.correct_kpt_idx = row
        self.label_kpt = True
     
    def sendToTable(self, kptx:float, kpty:float, kpt_label:int):
        kptx_item = QTableWidgetItem(str(kptx))
        kpty_item = QTableWidgetItem(str(kpty))
        if kpt_label :
            kpt_label_item = QTableWidgetItem("Y")
        else:
            kpt_label_item = QTableWidgetItem("N")
        kptx_item.setTextAlignment(Qt.AlignRight)
        kpty_item.setTextAlignment(Qt.AlignRight)
        kpt_label_item.setTextAlignment(Qt.AlignRight)
        self.ui.KptTable.setItem(self.correct_kpt_idx, 1, kptx_item)
        self.ui.KptTable.setItem(self.correct_kpt_idx, 2, kpty_item)
        self.ui.KptTable.setItem(self.correct_kpt_idx, 3, kpt_label_item)

        self.pose_estimater.update_person_df(kptx, kpty, self.ui.frameSlider.value(), self.correct_kpt_idx)
        self.update_frame(frame_num=self.ui.frameSlider.value())
 
    def correctId(self):
        before_correctId = self.ui.beforeCorrectId.value()
        after_correctId = self.ui.afterCorrectId.value()
        self.pose_estimater.correct_person_id(before_correctId, after_correctId)
        self.update_frame(self.ui.frameSlider.value())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PoseVideoTabControl()
    window.show()
    sys.exit(app.exec_())
