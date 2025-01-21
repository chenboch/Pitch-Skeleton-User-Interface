from PyQt5.QtWidgets import QMessageBox
import cv2
from cv_utils.cv_control import VideoLoader

class PlaybackController:
    def __init__(self, ui, analyze_frame_callback, video_loader:VideoLoader):
        self.ui = ui
        self.analyze_frame_callback = analyze_frame_callback
        self.video_loader = video_loader
        self.is_play = False
        
        # Connect UI elements to their corresponding methods
        self.setup_connections()

    def setup_connections(self):
        """Set up the connections for playback controls."""
        self.ui.play_btn.clicked.connect(self.play_btn_clicked)
        self.ui.back_key_btn.clicked.connect(self.backKeyPressed)
        self.ui.forward_key_btn.clicked.connect(self.forwardKeyPressed)
        self.ui.frame_slider.valueChanged.connect(self.analyze_frame_callback)

    def play_btn_clicked(self):
        """Handle play button click."""
        if self.video_loader.video_name == "":
            QMessageBox.warning(self.ui, "無法播放影片", "請讀取影片!")
            return
        if self.video_loader.is_loading:
            QMessageBox.warning(self.ui, "影片讀取中", "請稍等!")
            return
        
        self.is_play = not self.is_play
        self.ui.play_btn.setText("||" if self.is_play else "▶︎")
        if self.is_play:
            self.play_frame(self.ui.frame_slider.value())

    def play_frame(self, start_num: int = 0):
        """Play frames starting from the given frame number."""
        for i in range(start_num, self.video_loader.total_frames):
            if not self.is_play:
                break
            self.ui.frame_slider.setValue(i)
            self.analyze_frame_callback(i)  # Call the frame analysis method
            
            # If we reach the end of the video, stop playing
            if i == self.video_loader.total_frames - 1:
                self.is_play = False
                self.ui.play_btn.setText("▶︎")
                break
            
            cv2.waitKey(15)  # Adjust the delay as necessary

    def backKeyPressed(self):
        """Handle back key button press."""
        current_value = self.ui.frame_slider.value()
        self.ui.frame_slider.setValue(max(current_value - 1, self.ui.frame_slider.minimum()))

    def forwardKeyPressed(self):
        """Handle forward key button press."""
        current_value = self.ui.frame_slider.value()
        self.ui.frame_slider.setValue(min(current_value + 1, self.ui.frame_slider.maximum()))
