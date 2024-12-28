import torch
import pandas as pd
import numpy as np
from ..lib import OneEuroFilterTorch
from ..datasets import *
from ..datasets import halpe26_keypoint_info
from scipy.signal import savgol_filter
from mmpose.structures import (PoseDataSample, merge_data_samples,
                               split_instances)
from mmpose.apis import (convert_keypoint_definition, extract_pose_sequence,
                         inference_pose_lifter_model, inference_topdown)

def filterValidTargets(online_targets, select_id: int = None):
    """
    過濾出有效的追蹤目標。

    Args:
        online_targets (List): 所有在線追蹤的目標。
        select_id (int, optional): 選擇指定的追蹤ID。

    Returns:
        Tuple: 有效的邊界框和追蹤ID。
    """
    if not online_targets:
        return [], []

    # 將所有在線目標的邊界框和ID提取為兩個列表
    tlwhs = []
    track_ids = []

    for target in online_targets:
        tlwhs.append(target.tlwh)
        track_ids.append(target.track_id)

    # 將列表轉換為 NumPy array
    tlwhs = np.array(tlwhs)
    track_ids = np.array(track_ids)

    # 將數據轉為張量並放到 GPU 上
    tlwhs = torch.tensor(tlwhs, device='cuda')  # shape: (n, 4)
    track_ids = torch.tensor(track_ids, device='cuda')  # shape: (n,)

    # 計算面積 w * h
    areas = tlwhs[:, 2] * tlwhs[:, 3]  # w * h

    # 過濾面積大於 10 的邊界框
    valid_mask = areas > 10

    # 如果指定了 select_id，則進一步過濾
    if select_id is not None:
        valid_mask &= (track_ids == select_id)

    # 過濾有效的邊界框和追蹤ID
    valid_tlwhs = tlwhs[valid_mask]
    valid_track_ids = track_ids[valid_mask]

    # 將 (x1, y1, w, h) 轉為 (x1, y1, x2, y2)
    valid_bbox = torch.cat([valid_tlwhs[:, :2], valid_tlwhs[:, :2] + valid_tlwhs[:, 2:4]], dim=1)

    # 返回結果
    return valid_bbox.cpu().tolist(), valid_track_ids.cpu().tolist()

def merge_person_data(pred_instances, track_ids: list, frame_num: int = None) ->pd.DataFrame:
    person_bboxes = pred_instances['bboxes']
    new_person_data = []  # 用於暫存新的數據
    for person, pid, bbox in zip(pred_instances, track_ids, person_bboxes):
        keypoints_data = np.hstack((
            np.round(person['keypoints'][0], 2),
            np.round(person['keypoint_scores'][0], 2).reshape(-1, 1),
            np.full((len(person['keypoints'][0]), 1), False, dtype=bool)
        ))

        new_kpts = np.full((len(halpe26_keypoint_info['keypoints']), keypoints_data.shape[1]), 0.9)
        new_kpts[:26] = keypoints_data
        
        person_info = {
            'track_id': pid,
            'bbox': bbox,
            'keypoints': new_kpts
        }
        if frame_num is not None:
            person_info['frame_number'] = frame_num

        new_person_data.append(person_info)


    # 將新的數據轉為 DataFrame 並合併到 self.person_df
    new_person_df = pd.DataFrame(new_person_data)

    return new_person_df

def smooth_keypoints(person_df: pd.DataFrame, new_person_df: pd.DataFrame, track_ids: list) -> pd.DataFrame:
    """
    平滑 2D 關鍵點數據。
    
    Args:
        person_df (pd.DataFrame): 包含上一幀數據的 DataFrame。
        new_person_df (pd.DataFrame): 包含當前幀數據的 DataFrame。
        track_ids (list): 要處理的 track_id 列表。
    
    Returns:
        pd.DataFrame: 平滑後的 DataFrame。
    """
    smooth_filter_dict = {}

    # 當前幀無數據時，返回原始 new_person_df
    if person_df.empty:
        return new_person_df

    for track_id in track_ids:
        # 選擇上一幀和當前幀的數據
        pre_person_data = person_df.loc[
            (person_df['frame_number'] == person_df['frame_number'].iloc[-1]) &
            (person_df['track_id'] == track_id)
        ]
        curr_person_data = new_person_df.loc[(new_person_df['track_id'] == track_id)]

        # 如果當前幀或前幀沒有該 track_id 的數據，跳過
        if curr_person_data.empty or pre_person_data.empty:
            continue
        
        # 初始化濾波器字典（如果不存在）
        if track_id not in smooth_filter_dict:
            smooth_filter_dict[track_id] = {joint: OneEuroFilterTorch() for joint in range(len(pre_person_data.iloc[0]['keypoints']))}

        # 獲取上一幀和當前幀的關鍵點數據
        pre_kpts = torch.tensor(pre_person_data.iloc[0]['keypoints'], device='cuda')
        curr_kpts = torch.tensor(curr_person_data.iloc[0]['keypoints'], device='cuda')
        smoothed_kpts = []

        # 使用濾波器平滑每個關節點
        for joint_idx, (pre_kpt, curr_kpt) in enumerate(zip(pre_kpts, curr_kpts)):
            pre_kptx, pre_kpty = pre_kpt[0], pre_kpt[1]
            curr_kptx, curr_kpty, curr_conf, curr_label = curr_kpt[0], curr_kpt[1], curr_kpt[2], curr_kpt[3]
            
            if all([pre_kptx.item() != 0, pre_kpty.item() != 0, curr_kptx.item() != 0, curr_kpty.item() != 0]):
                # 為每個關節應用單獨的濾波器
                curr_kptx = smooth_filter_dict[track_id][joint_idx](curr_kptx, pre_kptx)
                curr_kpty = smooth_filter_dict[track_id][joint_idx](curr_kpty, pre_kpty)
            smoothed_kpts.append([curr_kptx.cpu().item(), curr_kpty.cpu().item(), curr_conf.item(), curr_label.item()])
        
        # 更新當前幀的數據
        new_person_df.at[curr_person_data.index[0], 'keypoints'] = smoothed_kpts

    return new_person_df

def updateKptBuffer(person_df:pd.DataFrame, track_id:int, kpt_id: int,frame_num:int, window_length=17, polyorder=2)->list:
    filtered_df = person_df[
        (person_df['track_id'] == track_id) & 
        (person_df['frame_number'] < frame_num)
    ]
    if filtered_df.empty:
        return None
    filtered_df = filtered_df.sort_values(by='frame_number')
    kpt_buffer = []
    for kpts in filtered_df['keypoints']:
        kpt = kpts[kpt_id]
        if kpt is not None and len(kpt) >= 2:
            kpt_buffer.append((kpt[0], kpt[1]))
    
    # 如果緩衝區長度大於等於窗口長度，則應用Savgol濾波器進行平滑
    if len(kpt_buffer) >= window_length:
        # 確保窗口長度為奇數且不超過緩衝區長度
        if window_length > len(kpt_buffer):
            window_length = len(kpt_buffer) if len(kpt_buffer) % 2 == 1 else len(kpt_buffer) - 1
        # 確保多項式階數小於窗口長度
        current_polyorder = min(polyorder, window_length - 1)
        
        # 分別提取x和y座標
        x = np.array([point[0] for point in kpt_buffer])
        y = np.array([point[1] for point in kpt_buffer])
        
        # 應用Savgol濾波器
        x_smooth = savgol_filter(x, window_length=window_length, polyorder=current_polyorder)
        y_smooth = savgol_filter(y, window_length=window_length, polyorder=current_polyorder)
        
        # 將平滑後的座標重新打包
        smoothed_points = list(zip(x_smooth, y_smooth))
    else:
        # 緩衝區長度不足，直接使用原始座標
        smoothed_points = kpt_buffer

    return smoothed_points
    
def correct_track_id(person_df:pd.DataFrame, before_correctId:int, after_correctId:int, max_frame:int)->pd.DataFrame:
    if person_df.empty:
        return

    if (before_correctId not in person_df['track_id'].unique()) or (after_correctId not in person_df['track_id'].unique()):
        return

    if (before_correctId in person_df['track_id'].unique()) and (after_correctId in person_df['track_id'].unique()):
        for i in range(0, max(max_frame)):
            condition_1 = (person_df['frame_number'] == i) & (person_df['track_id'] == before_correctId)
            person_df.loc[condition_1, 'track_id'] = after_correctId
    return person_df

#process 3d joints
def update_pose_results(new_person_df, pose_results, track_ids):
    """
    將平滑後的關鍵點數據從 new_person_df 更新回 data_samples 的結構。

    Args:
        new_person_df (pd.DataFrame): 包含平滑後關鍵點數據的 DataFrame。
        data_samples: 原始的姿態數據樣本結構（包含 pred_instances）。
        track_ids (list): 處理的 track_id 列表。

    Returns:
        data_samples: 更新後的姿態數據樣本結構。
    """
    # 遍歷每個 track_id 並更新對應的數據
    for track_id in track_ids:
        # 從 new_person_df 中提取該 track_id 的平滑數據
        person_data = new_person_df.loc[new_person_df['track_id'] == track_id]
        if person_data.empty:
            continue  # 該 track_id 無數據，跳過
        # 提取平滑後的關鍵點
        smoothed_keypoints = np.array(person_data.iloc[0]['keypoints'])[:, :2]
        for pred_instance in pose_results[0].pred_instances:
            pred_instance['keypoints'][0] = smoothed_keypoints
    
    return pose_results

def convert_keypoints(pose_result, keypoints, det_name, lift_name):
    """轉換 2D 關鍵點的數據格式。"""
    converted_sample = PoseDataSample()
    converted_sample.set_field(pose_result.pred_instances.clone(), 'pred_instances')
    converted_sample.set_field(pose_result.gt_instances.clone(), 'gt_instances')
    
    # 轉換關鍵點定義
    keypoints = convert_keypoint_definition(keypoints, det_name, lift_name)
    converted_sample.pred_instances.set_field(keypoints, 'keypoints')
    converted_sample.set_field(pose_result.track_id, 'track_id')
    return converted_sample

def extract_3d_data(data_3d_samples, track_ids):
    """
    從 data_3d_samples 中提取 3D 骨架數據，並轉換為 DataFrame 格式。

    Args:
        data_3d_samples (list): 包含 3D 骨架數據的列表。
        frame_number (int): 當前幀號。

    Returns:
        pd.DataFrame: 包含提取數據的 DataFrame。
    """
    records = []
    for i, sample in enumerate(data_3d_samples):
        track_id = track_ids[i]
        keypoints = sample.keypoints
        records.append({
            'track_id': track_id,
            'keypoints_3d': keypoints
        })

    return pd.DataFrame(records)

def merge_3d_data(new_person_df, pred_3d_pred_instances, track_ids):

    """
    將 3D 骨架數據合併到 self.person_df 中。

    Args:
        data_3d_samples (list): 包含 3D 骨架數據的列表。
        frame_number (int): 當前幀號。
    """
    # 提取 3D 骨架數據
    data_df = extract_3d_data(pred_3d_pred_instances, track_ids)
    # 合併 3D 骨架數據到現有的 person_df
    new_person_df = pd.merge(
        new_person_df, data_df,
        on='track_id',
        how='outer',  # 確保新舊數據都保留
        suffixes=('', '_new')
    )
    return new_person_df
    # # 更新 keypoints_3d 列，優先使用新數據
    # new_person_df['keypoints_3d'] = new_person_df['keypoints_3d_new'].combine_first(new_person_df['keypoints_3d'])
    # self.person_df.drop(columns=['keypoints_3d_new'], inplace=True)
