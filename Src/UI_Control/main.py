import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
from Pose2DTab.Camera_Widget.Widget import PoseCameraTabControl
from Pose2DTab.Video_Widget.Widget import PoseVideoTabControl as Pose2DVideoTabControl
from Pose2DTab.Pitch_Widget.Widget import PosePitchTabControl
from Pose3DTab.Video_Widget.Widget import PoseVideoTabControl as Pose3DVideoTabControl
from main_window import Ui_MainWindow
from utils.model import Model

import sys
import os


class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.model = Model()
        self.init_tabs()

    def init_tabs(self):
        self.camera_tab = PoseCameraTabControl(self.model)
        self.ui.Two_d_Tab.addTab(self.camera_tab, "2D 相機")
        self.video_tab = Pose2DVideoTabControl(self.model)
        self.ui.Two_d_Tab.addTab(self.video_tab, "2D 影片")
        self.pitch_tab = PosePitchTabControl(self.model)
        self.ui.Two_d_Tab.addTab(self.pitch_tab, "2D 投手")
        self.video3d_tab = Pose3DVideoTabControl(self.model)
        self.ui.Two_d_Tab.addTab(self.video3d_tab, "3D 影片")


    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec_())

