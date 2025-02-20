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

class Camera:
    def __init__(self, camera_idx:int = 0):
        self.camera_idx = camera_idx
        self.is_opened = False
        self.frame_count = 0
        self.fps_control = 1
        self.frame_size = None
        self.frame_buffer = queue.Queue()
        self.video_path = None
        self.video_writer = None
        self.video_thread = None

    def open_camera(self):
        # 開啟相機，並設置回調來處理每一幀
        self.frame_count = 0
        self.video_thread = VideoCaptureThread(camera_index=self.camera_idx)
        self.video_thread.frame_ready.connect(self.buffer_frame)
        self.video_thread.start_capture()
        self.is_opened = True

    def close_camera(self):
        # 關閉相機，清理資源
        if self.video_thread is not None:
            self.video_thread.stop_capture()
            self.video_thread = None
        self.is_opened = False
        self.frame_buffer = queue.Queue()
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def toggleCamera(self, is_checked:bool):
        # 根據checkbox狀態切換相機
        if is_checked:
            self.open_camera()
            frame_width = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.video_thread.cap.get(cv2.CAP_PROP_FPS))
            self.frame_size = (frame_width, frame_height)
            return (frame_width, frame_height, fps)
        else:
            self.close_camera()
            self.frame_size = None
            return (0, 0, 0)

    def buffer_frame(self, frame:np.ndarray):
        # 接收每一幀並進行處理
        self.frame_count += 1
        if self.is_opened and self.frame_count % self.fps_control ==0:
            self.frame_buffer.put(frame)
            # print(self.frame_buffer.qsize())

        if self.video_writer is not None and self.video_writer.is_writing:
            self.video_writer.write_frame(frame)

    def startRecording(self, filename: str):
        # 開始錄製影片
        if self.video_thread is None:
            return
        frame_width = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.video_thread.cap.get(cv2.CAP_PROP_FPS))
        self.video_path = filename

        self.video_writer = VideoWriterThread(filename, frame_width, frame_height, fps=fps)
        self.video_writer.start_writing()

    def stop_recording(self):
        # 停止錄製影片
        if self.video_writer is not None:
            self.video_writer.stop_writing()  # 停止寫入
            self.video_writer.release()  # 釋放執行緒

    def setCameraId(self, new_idx: int):
        self.camera_idx = new_idx
        print(f'camera id: {self.camera_idx}')

    def setFPSControl(self, fps:int):
        self.fps_control = fps

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
        # relative_path = os.path.relpath(self.folder_path, start="C:/Users/user/Desktop/Pitch-Skeleton-User-Interface/Db/Db_KCGM_Baseball")
        # 将空格替换为下划线
        # formatted_path = relative_path.replace(" ", "_")

        # fm_path = formatted_path.replace("\\", "_")
        # formatted_path = formatted_path.replace("\\", "_")
        # print(formatted_path)
        formatted_path = ""
        fm_path = ""
        # output_folder = os.path.join("../Db/Record", formatted_path+"_"+self.video_name)
        ann_folder = os.path.join("../Db/Data/annotations/train")
        # img_folder = os.path.join("../Db/Data/images", formatted_path+"_"+self.video_name)
        output_folder = os.path.join("../Db/output", self.video_name)
        ann_folder = os.path.join("../Db/Data/annotations/train")
        img_folder = os.path.join("../Db/Data/images", self.video_name)

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(ann_folder,exist_ok=True)

        json_path = os.path.join(output_folder, f"{self.video_name}_Sk17.json")
        if model_name == "vit-pose":
            json_path = os.path.join(output_folder, f"{self.video_name}_Sk26.json")
        json_ann_path =  os.path.join(ann_folder, f"{self.video_name}.json")
        save_person_df = self.image_drawer.pose_estimater.person_df

        save_person_df.write_json(json_path)

        save_person_df.write_json(json_ann_path)

        # save_location = os.path.join(output_folder, f"{formatted_path}_{self.video_name}.mp4")
        save_location = os.path.join(output_folder, f"{self.video_name}.mp4")
        video_writer = cv2.VideoWriter(save_location, cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, self.video_size)
        # output_folder = os.path.join("../Db/Record", self.video_name)
        # ann_folder = os.path.join("../Db/Data/annotations/train")
        # img_folder = os.path.join("../Db/Data/images", self.video_name)

        # os.makedirs(output_folder, exist_ok=True)
        # os.makedirs(img_folder, exist_ok=True)
        # os.makedirs(ann_folder,exist_ok=True)
        # json_path = os.path.join(output_folder, f"{self.video_name}.json")

        # json_ann_path =  os.path.join(ann_folder, f"{self.video_name}.json")

        # save_person_df = self.image_drawer.pose_estimater.person_df

        # save_person_df.to_json(json_path, orient='records')

        # save_person_df.to_json(json_ann_path, orient='records')

        # save_location = os.path.join(output_folder, f"{self.video_name}_Sk26.mp4")

        # video_writer = cv2.VideoWriter(save_location, cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, self.video_size)

        if not video_writer.isOpened():
            print("Error while opening video writer!")
            return


        for frame_num, frame in enumerate(self.video_frames):
            # if frame_num % 2 == 0:
            video_writer.write(frame)

        video_writer.release()
        # save_location = os.path.join(output_folder, f"{formatted_path}_{self.video_name}_Sk26.mp4")
        save_location = os.path.join(output_folder, f"{self.video_name}_Sk17.mp4")
        if model_name == "vit-pose":
            save_location = os.path.join(output_folder, f"{self.video_name}_Sk26.mp4")

        video_writer = cv2.VideoWriter(save_location, cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, self.video_size)
        for frame_num, frame in enumerate(self.video_frames):
            img_path =  os.path.join(img_folder, f"{frame_num:08d}.jpg" )
            cv2.imwrite(img_path, frame)
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
        json_path = os.path.join(self.folder_path, f"{self.file_name}_Sk17.json")
        if self.model_name == "vit-pose":
             json_path = os.path.join(self.folder_path, f"{self.file_name}_Sk26.json")
        print(json_path)

        if not os.path.exists(json_path):
            return

        self.person_df = pl.read_json(json_path)
        print(self.person_df)
