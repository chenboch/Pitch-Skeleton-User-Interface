from ...BaseTab import BasePoseVideoTab
from .video_ui import Ui_Video_UI
from skeleton import VidePose2DEstimater
from ...ui_utils import *

class PoseVideoTabControl(BasePoseVideoTab):
    def __init__(self, wrapper, parent = None):
        super().__init__(wrapper, parent)
        self.ui = Ui_Video_UI()
        self.ui.setupUi(self)
        self.bindUI()
        # print(self.ui.loadOriginalVideoBtn)  # 確保按鈕已初始化
    
    def bindUI(self):
        self.ui.loadOriginalVideoBtn.clicked.connect(
            lambda: self.load_video(is_processed=False))
        self.ui.loadProcessedVideoBtn.clicked.connect(
            lambda: self.load_video(is_processed=True))
        
        self.ui.playBtn.clicked.connect(self.play_btn_clicked)
        self.ui.backKeyBtn.clicked.connect(
            lambda: self.ui.frameSlider.setValue(self.ui.frameSlider.value() - 1)
        )
        self.ui.forwardKeyBtn.clicked.connect(
            lambda: self.ui.frameSlider.setValue(self.ui.frameSlider.value() + 1)
        )
        self.ui.frameSlider.valueChanged.connect(self.analyze_frame)
        self.kpt_table = KeypointTable(self.ui.KptTable,self.pose_estimater)
        self.ui.KptTable.cellActivated.connect(self.kpt_table.onCellClicked)
        self.ui.FrameView.mouse_press_event = self.mouse_press_event
        self.ui.IdCorrectBtn.clicked.connect(self.correctId)
        self.ui.startCodeBtn.clicked.connect(self.toggle_detect)
        self.ui.selectCheckBox.stateChanged.connect(self.toggle_select)
        self.ui.showSkeletonCheckBox.stateChanged.connect(self.toggleShowSkeleton)
        self.ui.selectKptCheckBox.stateChanged.connect(self.toggleKptSelect)
        self.ui.showBboxCheckBox.stateChanged.connect(self.toggleShowBbox)
        self.ui.showAngleCheckBox.stateChanged.connect(self.toggleShowAngleInfo)

    def setup_pose_estimater(self):
        """Setup 2D pose estimator."""
        self.pose_estimater = VidePose2DEstimater(self.wrapper)

