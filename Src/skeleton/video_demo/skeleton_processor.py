import torch
import pandas as pd
import numpy as np
from ..lib import OneEuroFilterTorch
from ..datasets import *
from ..datasets import halpe26_keypoint_info
from scipy.signal import savgol_filter

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

def mergePersonData(pred_instances, person_ids: list, frame_num: int = None) ->pd.DataFrame:
    person_bboxes = pred_instances['bboxes']
    new_person_data = []  # 用於暫存新的數據
    for person, pid, bbox in zip(pred_instances, person_ids, person_bboxes):
        keypoints_data = np.hstack((
            np.round(person['keypoints'][0], 2),
            np.round(person['keypoint_scores'][0], 2).reshape(-1, 1),
            np.full((len(person['keypoints'][0]), 1), False, dtype=bool)
        ))

        new_kpts = np.full((len(halpe26_keypoint_info['keypoints']), keypoints_data.shape[1]), 0.9)
        new_kpts[:26] = keypoints_data
        
        person_info = {
            'person_id': pid,
            'bbox': bbox,
            'keypoints': new_kpts
        }
        if frame_num is not None:
            person_info['frame_number'] = frame_num

        new_person_data.append(person_info)


    # 將新的數據轉為 DataFrame 並合併到 self.person_df
    new_person_df = pd.DataFrame(new_person_data)

    return new_person_df

def smoothKpt(person_df:pd.DataFrame, person_ids: list, frame_num:int) -> pd.DataFrame :
    smooth_filter = OneEuroFilterTorch()
    curr_frame = frame_num
    if curr_frame == 0:
        return  # 初始幀，無需處理
    pre_frame_num = curr_frame - 1

    # 當前幀無數據時，跳過處理
    if person_df.empty:
        return
    
    for person_id in person_ids:
        # 如果使用 frame_slider，根據前後幀數據進行處理
        pre_person_data = person_df.loc[(person_df['frame_number'] == pre_frame_num) &
                                            (person_df['person_id'] == person_id)]
        curr_person_data = person_df.loc[(person_df['frame_number'] == curr_frame) &
                                            (person_df['person_id'] == person_id)]
    
        if curr_person_data.empty or pre_person_data.empty:
            continue  # 當前幀或前幀沒有該 person_id 的數據，跳過
        
        pre_kpts = torch.tensor(pre_person_data.iloc[0]['keypoints'], device='cuda')
        curr_kpts = torch.tensor(curr_person_data.iloc[0]['keypoints'], device='cuda')
        smoothed_kpts = []

        # 使用張量運算來進行平滑
        for pre_kpt, curr_kpt in zip(pre_kpts, curr_kpts):
            pre_kptx, pre_kpty = pre_kpt[0], pre_kpt[1]
            curr_kptx, curr_kpty, curr_conf, curr_label = curr_kpt[0], curr_kpt[1], curr_kpt[2], curr_kpt[3]
            
            if all([pre_kptx.item() != 0, pre_kpty.item() != 0, curr_kptx.item() != 0, curr_kpty.item() != 0]):
                curr_kptx = smooth_filter(curr_kptx, pre_kptx)
                curr_kpty = smooth_filter(curr_kpty, pre_kpty)
            smoothed_kpts.append([curr_kptx.cpu().item(), curr_kpty.cpu().item(), curr_conf.item(), curr_label.item()])
        
        # 更新當前幀的數據
        person_df.at[curr_person_data.index[0], 'keypoints'] = smoothed_kpts
    return person_df

def updateKptBuffer(person_df:pd.DataFrame, person_id:int, kpt_id: int,frame_num:int, window_length=17, polyorder=2)->list:
    filtered_df = person_df[
        (person_df['person_id'] == person_id) & 
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
    
def correct_person_id(person_df:pd.DataFrame, before_correctId:int, after_correctId:int, max_frame:int)->pd.DataFrame:
    if person_df.empty:
        return

    if (before_correctId not in person_df['person_id'].unique()) or (after_correctId not in person_df['person_id'].unique()):
        return

    if (before_correctId in person_df['person_id'].unique()) and (after_correctId in person_df['person_id'].unique()):
        for i in range(0, max(max_frame)):
            condition_1 = (person_df['frame_number'] == i) & (person_df['person_id'] == before_correctId)
            person_df.loc[condition_1, 'person_id'] = after_correctId
    return person_df