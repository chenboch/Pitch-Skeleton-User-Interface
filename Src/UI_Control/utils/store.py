import os
import cv2
from utils.vis_pose import draw_points_and_skeleton, joints_dict
import pandas as pd

def obtain_frame_data(frame_num, person_df):
    try:
        return person_df.loc[(person_df['frame_number'] == frame_num)]
    except ValueError:
        print("ValueError")
        return pd.DataFrame()

def reset_keypoints(keypoints):
    modified_keypoints = keypoints.copy()
    for kpt in modified_keypoints:
        if kpt[0] == 0.0 and kpt[1] == 0.0:
            kpt[2] = 0
    return modified_keypoints

def reset_person_df(person_df):
    person_df['keypoints'] = person_df['keypoints'].apply(reset_keypoints)
    return person_df

def filter_person_df(person_df, select_id):
    filter_df = person_df[person_df['person_id'].isin([select_id])]
    return filter_df

def draw_image(frame, person_df):
    image = frame.copy()
    if not person_df.empty:
        image = draw_points_and_skeleton(image, person_df, joints_dict()['haple']['skeleton_links'],
                                         points_color_palette='gist_rainbow', skeleton_palette_samples='jet',
                                         points_palette_samples=10, confidence_threshold=0.3)
    return image

def save_video(video_name, video_images, person_df, select_id = None):
    output_folder = os.path.join("../../Db/Record", video_name)
    os.makedirs(output_folder, exist_ok=True)

    json_path = os.path.join(output_folder, f"{video_name}.json")
    save_person_df = reset_person_df(person_df)

    if select_id != None:
        save_person_df = filter_person_df(save_person_df,select_id)

    save_person_df.to_json(json_path, orient='records')

    video_size = (1920, 1080)
    fps = 30.0
    save_location = os.path.join(output_folder, f"{video_name}_Sk26.mp4")

    video_writer = cv2.VideoWriter(save_location, cv2.VideoWriter_fourcc(*'mp4v'), fps, video_size)

    if not video_writer.isOpened():
        print("Error while opening video writer!")
        return

    for frame_num, frame in enumerate(video_images):
        if frame is not None and frame.shape[:2] == (video_size[1], video_size[0]):
            curr_person_df = obtain_frame_data(frame_num, save_person_df)
            image = draw_image(frame, curr_person_df)
            video_writer.write(image)

    video_writer.release()
    print("Store video success")