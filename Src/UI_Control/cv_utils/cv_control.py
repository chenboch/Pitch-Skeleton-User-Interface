import cv2
import numpy as np
import queue
import os
import polars as pl
from enum import Enum
from PyQt5.QtWidgets import QGraphicsScene
from ..vis_utils.vis_image import ImageDrawer
from .cv_thread import VideoCaptureThread, VideoWriterThread, VideoToImagesThread
from PyQt5.QtWidgets import QFileDialog
import logging

class Frame_Buffer(np.ndarray):
    def __new__(cls, input_array=None, max_size: int = 5):
        # Initialize an empty buffer
        obj = np.empty(0, dtype=object).view(cls)
        obj.frame_buffer = []
        obj.logger = logging.getLogger(cls.__name__)
        obj.max_size = max_size  # Store max_size in __new__
        return obj

    def __init__(self, input_array=None, max_size: int = 5):
        # Validate max_size
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("max_size must be a positive integer")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_size = max_size
        if input_array is not None:
            self.put(input_array)

    def put(self, frame):
        """Add a frame to the buffer, remove oldest if exceeding max_size"""
        if not isinstance(frame, np.ndarray):
            self.logger.warning("Input frame is not a NumPy array")
            frame = np.array(frame)

        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.max_size:
            self.frame_buffer.pop(0)  # Remove oldest frame
            self.logger.debug("Removed oldest frame, buffer size: %d", len(self.frame_buffer))
        else:
            self.logger.debug("Added frame, buffer size: %d", len(self.frame_buffer))

    def pop(self):
        """Remove and return the oldest frame, or None if buffer is empty"""
        if len(self.frame_buffer) == 0:
            self.logger.warning("Buffer is empty, nothing to pop")
            return None
        frame = self.frame_buffer.pop(0)
        self.logger.debug("Popped oldest frame, buffer size: %d", len(self.frame_buffer))
        return frame

    def __array_finalize__(self, obj):
        # Ensure frame_buffer, logger, and max_size are preserved
        if obj is None:
            return
        self.frame_buffer = getattr(obj, 'frame_buffer', [])
        self.logger = getattr(obj, 'logger', logging.getLogger(self.__class__.__name__))
        self.max_size = getattr(obj, 'max_size', 5)  # Default to 5 if not set

    def __len__(self):
        """Return the current number of frames in the buffer"""
        return len(self.frame_buffer)

    def __getitem__(self, index):
        """Access frames in the buffer by index"""
        return self.frame_buffer[index]

    def __str__(self):
        """String representation of the buffer"""
        return f"Frame_Buffer 包含 {len(self.frame_buffer)} 幀"

class Camera:
    def __init__(self, camera_idx: int = 0):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.camera_idx = camera_idx
        self.is_opened = False
        self.frame_count = 0
        self.fps_control = 1  # 每 fps_control 幀儲存一幀
        self.frame_size = None
        self.frame_buffer = Frame_Buffer(max_size=5)  # 使用 Frame_Buffer 類
        self.video_path = None
        self.video_writer = None
        self.video_thread = None

    def open_camera(self):
        """開啟攝影機，並設置回調來處理每一幀"""
        if self.is_opened:
            self.logger.warning("攝影機已開啟")
            return

        self.frame_count = 0
        try:
            self.video_thread = VideoCaptureThread(camera_index=self.camera_idx)
            self.video_thread.frame_ready.connect(self.buffer_frame)
            self.video_thread.start_capture()
            self.is_opened = True
            self.logger.info(f"攝影機 {self.camera_idx} 已開啟")
        except Exception as e:
            self.logger.error(f"開啟攝影機失敗: {str(e)}")
            self.is_opened = False
            raise

    def close_camera(self):
        """關閉攝影機，清理資源"""
        if not self.is_opened:
            self.logger.warning("攝影機未開啟")
            return

        if self.video_thread is not None:
            self.video_thread.stop_capture()
            self.video_thread = None
        if self.video_writer is not None:
            self.video_writer.stop_writing()
            self.video_writer.release()
            self.video_writer = None
        self.is_opened = False
        self.frame_buffer = Frame_Buffer(max_size=self.frame_buffer.max_size)  # 重置緩衝區，保留 max_size
        self.frame_size = None
        self.logger.info("攝影機已關閉")

    def toggle_camera(self, is_checked: bool):
        """根據 checkbox 狀態切換攝影機"""
        if is_checked:
            self.open_camera()
            if self.video_thread and self.video_thread.cap:
                frame_width = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(self.video_thread.cap.get(cv2.CAP_PROP_FPS) or 30)  # 預設 30 fps
                self.frame_size = (frame_width, frame_height)
                self.logger.info(f"攝影機參數: 寬 {frame_width}, 高 {frame_height}, FPS {fps}")
                return (frame_width, frame_height, fps)
            return (0, 0, 0)
        else:
            self.close_camera()
            return (0, 0, 0)

    def buffer_frame(self, frame: np.ndarray):
        """接收每一幀並進行處理"""
        self.frame_count += 1
        if self.is_opened and self.frame_count % self.fps_control == 0:
            self.frame_buffer.put(frame)
            self.logger.debug(f"緩衝區大小: {len(self.frame_buffer)}")

            if self.video_writer is not None and self.video_writer.is_writing:
                self.video_writer.write_frame(frame)

    def start_recording(self, filename: str):
        """開始錄製影片"""
        if not self.is_opened or self.video_thread is None:
            self.logger.error("無法錄製: 攝影機未開啟")
            return

        frame_width = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.video_thread.cap.get(cv2.CAP_PROP_FPS) or 30)
        self.video_path = filename

        try:
            self.video_writer = VideoWriterThread(filename, frame_width, frame_height, fps=fps)
            self.video_writer.start_writing()
            self.logger.info(f"開始錄製影片: {filename}")
        except Exception as e:
            self.logger.error(f"開始錄製失敗: {str(e)}")
            self.video_writer = None
            raise

    def stop_recording(self):
        """停止錄製影片"""
        if self.video_writer is not None:
            self.video_writer.stop_writing()
            self.video_writer.release()
            self.video_writer = None
            self.logger.info(f"停止錄製影片: {self.video_path}")
            self.video_path = None

    def set_camera_id(self, new_idx: int):
        """設置攝影機 ID"""
        if self.is_opened:
            self.logger.warning("請先關閉攝影機再更改 ID")
            return
        self.camera_idx = new_idx
        self.logger.info(f"當前攝影機 ID: {self.camera_idx}")

    def set_fps_control(self, fps: int):
        """設置幀率控制，每 fps 幀儲存一幀"""
        if fps <= 0:
            self.logger.error("fps_control 必須為正整數")
            raise ValueError("fps_control 必須為正整數")
        self.fps_control = fps
        self.logger.info(f"設置 fps_control 為: {fps}")

class DataType(Enum):
    DEFAULT = {"name": "default", "tips": "", "filter": ""}
    IMAGE = {"name": "image", "tips": "",
             "filter": "Image files (*.jpeg *.png *.tiff *.psd *.pdf *.eps *.gif)"}
    VIDEO = {"name": "video", "tips": "",
             "filter": "Video files ( *.WEBM *.MPG *.MP2 *.MPEG *.MPE *.MPV *.OGG *.MP4 *.M4P *.M4V *.AVI *.WMV *.MOV *.QT *.FLV *.SWF *.MKV)"}
    CSV = {"name": "csv", "tips": "",
           "filter": "Video files (*.csv)"}
    FOLDER = {"name": "folder", "tips": "", "filter": ""}

class VideoLoader:
    def __init__(self, image_drawer: ImageDrawer =None):
        self.image_drawer = image_drawer
        self.video_path = None
        self.folder_path = None
        self.video_size = None
        self.video_fps = None
        self.video_frames = None
        self.total_frames = None
        self.video_name = None
        self.is_loading = False

    def load_video(self, video_path:str = None):
        options = QFileDialog.Options()
        if video_path is None:
            video_path, _ = QFileDialog.getOpenFileName(None, "Select Video File", "", "Video Files (*.mp4 *.avi);;All Files (*)", options=options)
            if not video_path:
                return
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.folder_path = os.path.dirname(video_path)
        print(self.video_name)
        self.is_loading = True
        self.v_t = VideoToImagesThread(self.video_path)
        self.v_t.emit_signal.connect(self.video_to_frame)

        self.v_t.start()

    def video_to_frame(self, video_frames, fps, count):
        self.total_frames = count
        self.video_frames = video_frames
        self.video_fps = fps
        self.video_size = (self.video_frames[0].shape[1], self.video_frames[0].shape[0])

        self.close_thread(self.v_t)

    def close_thread(self, thread):
        thread.stop()
        self.is_loading = False
        thread = None

    def get_video_image(self, frame_num:int) -> np.ndarray:
        return self.video_frames[frame_num].copy()

    def save_video(self, model_name:str):
        ann_folder = os.path.join("../Db/Data/annotations/train")
        img_folder = os.path.join("../Db/Data/images", self.video_name)
        output_folder = os.path.join("../Db/output", self.video_name)
        ann_folder = os.path.join("../Db/Data/annotations/train")
        img_folder = os.path.join("../Db/Data/images", self.video_name)

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(ann_folder,exist_ok=True)

        json_path = os.path.join(output_folder, f"{self.video_name}_famipose.json")
        if model_name == "vit-pose":
            json_path = os.path.join(output_folder, f"{self.video_name}_vitpose.json")
        json_ann_path =  os.path.join(ann_folder, f"{self.video_name}.json")
        save_person_df = self.image_drawer.pose_estimater.person_df

        save_person_df.write_json(json_path)

        save_person_df.write_json(json_ann_path)

        save_location = os.path.join(output_folder, f"{self.video_name}.mp4")
        video_writer = cv2.VideoWriter(save_location, cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, self.video_size)

        if not video_writer.isOpened():
            print("Error while opening video writer!")
            return


        for frame_num, frame in enumerate(self.video_frames):
            # if frame_num % 2 == 0:
            video_writer.write(frame)

        video_writer.release()
        # save_location = os.path.join(output_folder, f"{formatted_path}_{self.video_name}_Sk26.mp4")
        save_location = os.path.join(output_folder, f"{self.video_name}_famipose.mp4")
        if model_name == "vit-pose":
            save_location = os.path.join(output_folder, f"{self.video_name}_vitpose.mp4")

        video_writer = cv2.VideoWriter(save_location, cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, self.video_size)
        for frame_num, frame in enumerate(self.video_frames):
            img_path =  os.path.join(img_folder, f"{frame_num:08d}.jpg" )
            cv2.imwrite(img_path, frame)
            if frame_num % 2 == 0:
                image = self.image_drawer.drawInfo(img = frame, frame_num = frame_num)
                video_writer.write(image)

        video_writer.release()
        print("Store video success")

    def reset(self):
        self.video_path = None
        self.folder_path = None
        self.video_size = None
        self.video_fps = None
        self.video_frames = None
        self.total_frames = None
        self.video_name = None
        self.is_loading = False

class JsonLoader:
    def __init__(self, folder_path:str = None, file_name:str = None, model_name:str = None):
        self.folder_path = folder_path
        self.file_name = file_name
        self.person_df = pl.DataFrame()
        self.model_name = model_name

    def load_json(self) -> pl.DataFrame:
        if self.folder_path == None or self.file_name == None:
            return
        json_path = os.path.join(self.folder_path, f"{self.file_name}_famipose.json")
        if self.model_name == "vit-pose":
             json_path = os.path.join(self.folder_path, f"{self.file_name}_vitpose.json")
        print(json_path)

        if not os.path.exists(json_path):
            return

        self.person_df = pl.read_json(json_path)
        print(self.person_df)
