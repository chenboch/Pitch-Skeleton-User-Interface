from ..datasets import halpe26_keypoint_info
import torch
import numpy as np
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter

        
def mergePersonData(pred_instances, person_ids: list, frame_num: int = None):
    """
    Efficiently merge person data into a DataFrame.
    """
    person_bboxes = pred_instances['bboxes']
    data = []

    for person, pid, bbox in zip(pred_instances, person_ids, person_bboxes):
        keypoints_data = np.hstack((
            np.round(person['keypoints'][0], 2),
            np.round(person['keypoint_scores'][0], 2).reshape(-1, 1),
            np.full((len(person['keypoints'][0]), 1), False, dtype=bool)
        ))

        new_kpts = np.full((26, keypoints_data.shape[1]), 0.9)
        new_kpts[:keypoints_data.shape[0]] = keypoints_data

        person_info = {
            'person_id': pid,
            'bbox': bbox,
            'keypoints': new_kpts,
            'frame_number': frame_num
        }
        data.append(person_info)

    return pd.DataFrame(data)


def smoothKpt(self, person_ids: list, frame_num=None):

    self.pre_person_df = self.person_df.copy()

    # 當前幀無數據時，跳過處理
    if self.person_df.empty:
        return
    
    for person_id in person_ids:
        pre_person_data = self.pre_person_df.loc[self.pre_person_df['person_id'] == person_id]
        curr_person_data = self.person_df.loc[self.person_df['person_id'] == person_id]
        
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
                curr_kptx = self.smooth_filter(curr_kptx, pre_kptx)
                curr_kpty = self.smooth_filter(curr_kpty, pre_kpty)
            
            smoothed_kpts.append([curr_kptx.cpu().item(), curr_kpty.cpu().item(), curr_conf.item(), curr_label.item()])
        
        # 更新當前幀的數據
        self.person_df.at[curr_person_data.index[0], 'keypoints'] = smoothed_kpts