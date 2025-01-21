from ...BaseTab import BasePoseVideoTab
from .video_ui import Ui_video_widget
from skeleton import VidePose2DEstimater
from ...ui_utils import *

class PoseVideoTabControl(BasePoseVideoTab):
    def __init__(self, wrapper, parent = None):
        super().__init__(wrapper, parent)
        self.ui = Ui_video_widget()
        self.ui.setupUi(self)
        self.bindUI()
        # print(self.ui.load_original_video_btn)  # 確保按鈕已初始化
    
    def bindUI(self):
        self.ui.load_original_video_btn.clicked.connect(
            lambda: self.load_video(is_processed=False))
        self.ui.load_processed_video_btn.clicked.connect(
            lambda: self.load_video(is_processed=True))
        
        self.ui.play_btn.clicked.connect(self.play_btn_clicked)
        self.ui.back_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        )
        self.ui.forward_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        )
        self.ui.frame_slider.valueChanged.connect(self.analyze_frame)
        self.kpt_table = KeypointTable(self.ui.kpt_table,self.pose_estimater)
        self.ui.kpt_table.cellActivated.connect(self.kpt_table.onCellClicked)
        self.ui.frame_view.mousePressEvent = self.mouse_press_event
        self.ui.id_correct_btn.clicked.connect(self.correctId)
        self.ui.start_code_btn.clicked.connect(self.toggle_detect)
        self.ui.select_checkbox.stateChanged.connect(self.toggle_select)
        self.ui.show_skeleton_checkbox.stateChanged.connect(self.toggle_show_skeleton)
        self.ui.select_kpt_checkbox.stateChanged.connect(self.toggle_kpt_select)
        self.ui.show_bbox_checkbox.stateChanged.connect(self.toggle_show_bbox)
        self.ui.show_angle_checkbox.stateChanged.connect(self.toggle_show_angle_info)

    def setup_pose_estimater(self):
        """Setup 2D pose estimator."""
        self.pose_estimater = VidePose2DEstimater(self.wrapper)


    def keyPressEvent(self, event):
        if event.key() == ord('D') or event.key() == ord('d'):
            self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        elif event.key() == ord('A') or event.key() == ord('a'):
            self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        else:
            super().keyPressEvent(event)
