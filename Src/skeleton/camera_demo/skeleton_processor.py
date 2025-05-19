import torch
import polars as pl
import numpy as np
from scipy.signal import savgol_filter
from mmpose.structures import (PoseDataSample)
from mmpose.apis import (convert_keypoint_definition)
from ..utils import OneEuroFilterTorch
from ..datasets import halpe26_keypoint_info, posetrack_keypoint_info, halpe26_to_posetrack_keypoint_info


def filter_valid_targets(online_targets, select_id: int = None):
    """
    過濾出有效的追蹤目標。

    Args:
        online_targets (List): 所有在線追蹤的目標。
        select_id (int, optional): 選擇指定的追蹤ID。

    Returns:
        Tuple: 有效的邊界框 (Tensor) 和追蹤ID (Tensor)。
    """
    if not online_targets:
        return [], torch.empty((0,), dtype=torch.int32, device='cuda').tolist()

    tlwhs = []
    for target in online_targets:
        tlwhs.append(target.tlwh)
    # 直接生成張量以減少資料轉換
    tlwhs = torch.tensor(np.array(tlwhs), device='cuda')  # (n, 4)
    track_ids = torch.tensor([target.track_id for target in online_targets], dtype=torch.int32, device='cuda')  # (n,)

    # 計算面積 (w * h)
    areas = tlwhs[:, 2] * tlwhs[:, 3]

    # 過濾面積大於 10 的目標
    valid_mask = areas > 10

    # 如果指定了 select_id，則進一步過濾
    if select_id is not None:
        valid_mask &= (track_ids == select_id)

    # 根據過濾條件提取有效的邊界框和追蹤ID
    valid_tlwhs = tlwhs[valid_mask]
    valid_track_ids = track_ids[valid_mask]

    # 將 (x1, y1, w, h) 轉換為 (x1, y1, x2, y2)
    valid_bbox = torch.cat([valid_tlwhs[:, :2], valid_tlwhs[:, :2] + valid_tlwhs[:, 2:4]], dim=1)

    return valid_bbox.cpu().tolist(), valid_track_ids.cpu().tolist()

def coco2posetrack(keypoint):
    data = np.zeros((len(posetrack_keypoint_info['keypoints']), 4))
    data[:17] = keypoint
    # 計算中點並填充數據
    # x_chest = (keypoint[5][0] + keypoint[6][0]) / 2.0
    # y_chest = (keypoint[5][1] + keypoint[6][1]) / 2.0
    # s_chest = (keypoint[5][2] + keypoint[6][2]) / 2.0
    # data[1] = [x_chest, y_chest, s_chest, False]
    x_mhead = (keypoint[1][0] + keypoint[2][0]) / 2.0
    y_mhead = (keypoint[1][1] + keypoint[2][1]) / 2.0
    s_mhead = (keypoint[1][2] + keypoint[2][2]) / 2.0
    x_butt = (keypoint[11][0] + keypoint[12][0]) / 2.0
    y_butt = (keypoint[11][1] + keypoint[12][1]) / 2.0
    s_butt = (keypoint[11][2] + keypoint[12][2]) / 2.0
    data[17] = [x_mhead, y_mhead, s_mhead, False]
    data[18] = [x_butt, y_butt, s_butt, False]
    return data

def haple2posetrack(keypoint):
    # print(keypoint)
    # exit()
    data = np.zeros((len(posetrack_keypoint_info['keypoints']), 4))
    data[4:18] = keypoint[4:18]
    for src_i, dst_i in halpe26_to_posetrack_keypoint_info['keypoints'].items():
        data[dst_i] = keypoint[src_i]
        data[dst_i][3] = False
    x_mhead = (keypoint[17][0] + keypoint[18][0]) / 2.0
    y_mhead = (keypoint[17][1] + keypoint[18][1]) / 2.0
    s_mhead = (keypoint[17][2] + keypoint[18][2]) / 2.0
    data[17] = [ x_mhead, y_mhead, s_mhead, False]
    x_butt = (keypoint[11][0] + keypoint[12][0]) / 2.0
    y_butt = (keypoint[11][1] + keypoint[12][1]) / 2.0
    s_butt = (keypoint[11][2] + keypoint[12][2]) / 2.0
    data[18] = [x_butt, y_butt, s_butt, False]
    return data

def merge_person_data(pred_instances, track_ids: list, model_name:str, frame_num: int = None) -> pl.DataFrame:
    person_bboxes = pred_instances['bboxes']

    # 優化：提前創建列表，避免多次 append 操作
    new_person_data = []

    for person, pid, bbox in zip(pred_instances, track_ids, person_bboxes):
        keypoints_data = np.hstack((
            np.round(person['keypoints'][0], 2),
            np.round(person['keypoint_scores'][0], 2).reshape(-1, 1),
            np.full((len(person['keypoints'][0]), 1), False, dtype=bool)
        ))

        # 選擇模型類型進行處理
        if model_name == "vit-pose":
            new_kpts = haple2posetrack(keypoints_data)
            # new_kpts = np.full((halpe26_shape, keypoints_data.shape[1]), 0.9)
            # new_kpts[] = keypoints_data
        else:
            new_kpts = coco2posetrack(keypoints_data)
        new_kpts = new_kpts.tolist()
        # 轉換 bbox 為列表
        bbox = bbox.tolist()

        # 優化：將字典構建過程集中處理，減少冗余運算
        person_info = {
            'track_id': pid,
            'bbox': bbox,
            'area': np.round(bbox[2] * bbox[3], 2),
            'keypoints': new_kpts
        }
        if frame_num is not None:
            person_info['frame_number'] = frame_num

        new_person_data.append(person_info)

    # 使用 PyArrow 加速 DataFrame 構建
    new_person_df = pl.DataFrame(new_person_data)

    return new_person_df


import torch
import polars as pl
import numpy as np
import cv2  # 用於計算光流

def smooth_keypoints(person_df: pl.DataFrame, new_person_df: pl.DataFrame, track_ids: list,
                    images: np.ndarray) -> pl.DataFrame:
    """
    平滑 2D 關鍵點數據，結合光流和 One Euro Filter。

    Args:
        person_df (pl.DataFrame): 包含上一幀數據的 DataFrame。
        new_person_df (pl.DataFrame): 包含當前幀數據的 DataFrame。
        track_ids (list): 要處理的 track_id 列表。
        curr_image (np.ndarray): 當前幀圖像 (H, W, C)。
        prev_image (np.ndarray, optional): 前一幀圖像 (H, W, C)。

    Returns:
        pl.DataFrame: 平滑後的 DataFrame。
    """
    smooth_filter_dict = {}

    # 當前幀無數據時，返回原始 new_person_df
    if person_df.is_empty() or new_person_df.is_empty():
        return new_person_df

    # 如果有前一幀圖像，計算光流
    prev_image = images[1]
    curr_image = images[2]

    flow = compute_optical_flow(prev_image, curr_image)

    for track_id in track_ids:
        # 選擇上一幀和當前幀的數據
        pre_person_data = person_df.filter(
            (person_df['track_id'] == track_id)
        )
        curr_person_data = new_person_df.filter(new_person_df['track_id'] == track_id)

        # 如果當前幀或前幀沒有該 track_id 的數據，跳過
        if pre_person_data.is_empty() or curr_person_data.is_empty():
            continue

        # 初始化濾波器字典（如果不存在）
        if track_id not in smooth_filter_dict:
            keypoints_len = len(pre_person_data.select("keypoints").row(0)[0])
            smooth_filter_dict[track_id] = {joint: OneEuroFilterTorch() for joint in range(keypoints_len)}

        # 獲取上一幀和當前幀的關鍵點數據
        pre_kpts = torch.tensor(pre_person_data.select("keypoints").row(0)[0], device='cuda')
        curr_kpts = torch.tensor(curr_person_data.select("keypoints").row(0)[0], device='cuda')
        smoothed_kpts = []

        # 使用光流預測並融合
        for joint_idx, (pre_kpt, curr_kpt) in enumerate(zip(pre_kpts, curr_kpts)):
            pre_kptx, pre_kpty = pre_kpt[0], pre_kpt[1]
            curr_kptx, curr_kpty, curr_conf, curr_label = curr_kpt[0], curr_kpt[1], curr_kpt[2], curr_kpt[3]

            # 如果前一幀和當前幀坐標有效，且有光流數據
            if (flow is not None and
                all([pre_kptx.item() != 0, pre_kpty.item() != 0, curr_kptx.item() != 0, curr_kpty.item() != 0])):
                # 從光流場提取運動向量
                y, x = int(pre_kpty.item()), int(pre_kptx.item())
                if 0 <= y < flow.shape[0] and 0 <= x < flow.shape[1]:  # 檢查邊界
                    dx, dy = flow[y, x]
                    # 光流預測位置
                    pred_kptx = pre_kptx + dx
                    pred_kpty = pre_kpty + dy
                    # 融合光流預測和原始估計 (加權平均)
                    alpha = 0.3  # 光流權重，可調整
                    fused_kptx = alpha * pred_kptx + (1 - alpha) * curr_kptx
                    fused_kpty = alpha * pred_kpty + (1 - alpha) * curr_kpty
                else:
                    fused_kptx, fused_kpty = curr_kptx, curr_kpty  # 超出邊界時使用原始值
            else:
                fused_kptx, fused_kpty = curr_kptx, curr_kpty  # 無光流時使用原始值

            # 使用 One Euro Filter 平滑融合後的坐標
            if all([pre_kptx.item() != 0, pre_kpty.item() != 0, fused_kptx.item() != 0, fused_kpty.item() != 0]):
                smoothed_kptx = smooth_filter_dict[track_id][joint_idx](fused_kptx, pre_kptx)
                smoothed_kpty = smooth_filter_dict[track_id][joint_idx](fused_kpty, pre_kpty)
            else:
                smoothed_kptx, smoothed_kpty = fused_kptx, fused_kpty

            smoothed_kpts.append([smoothed_kptx.cpu().item(), smoothed_kpty.cpu().item(),
                                 curr_conf.item(), curr_label.item()])

        # 更新當前幀的數據
        new_person_df = new_person_df.with_columns(
            pl.when(new_person_df['track_id'] == track_id)
            .then(pl.Series("keypoints", [smoothed_kpts]))
            .otherwise(new_person_df["keypoints"])
            .alias("keypoints")
        )

    return new_person_df

def compute_optical_flow(prev_image, curr_image, scale_factor=0.25):
    # 轉為灰度並縮小
    prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    small_prev = cv2.resize(prev_gray, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_AREA)
    small_curr = cv2.resize(curr_gray, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_AREA)

    # 計算光流
    flow = cv2.calcOpticalFlowFarneback(small_prev, small_curr, None,
                                        pyr_scale=0.5, levels=2, winsize=10,
                                        iterations=2, poly_n=5, poly_sigma=1.1, flags=0)

    # 將光流縮放回原始尺寸
    flow = cv2.resize(flow, (w, h)) / scale_factor
    return flow

def update_keypoint_buffer(person_df:pl.DataFrame, track_id:int, kpt_id: int, kpt_buffer:list, window_length=1, polyorder=2)->list:

    filtered_df = person_df.filter(
        (person_df['track_id'] == track_id)
    )

    if filtered_df.is_empty():
        return None


    new_kpt_buffer = kpt_buffer
    kpt = filtered_df['keypoints'][0][kpt_id]
    if kpt is not None and len(kpt) >= 2:
        new_kpt_buffer.append((kpt[0], kpt[1]))

    # 如果緩衝區長度大於等於窗口長度，則應用Savgol濾波器進行平滑
    if len(new_kpt_buffer) >= window_length:
        # 確保窗口長度為奇數且不超過緩衝區長度
        if window_length > len(new_kpt_buffer):
            window_length = len(new_kpt_buffer) if len(new_kpt_buffer) % 2 == 1 else len(new_kpt_buffer) - 1
        # 確保多項式階數小於窗口長度
        current_polyorder = min(polyorder, window_length - 1)

        # 分別提取x和y座標
        x = np.array([point[0] for point in new_kpt_buffer])
        y = np.array([point[1] for point in new_kpt_buffer])

        # 應用Savgol濾波器
        x_smooth = savgol_filter(x, window_length=window_length, polyorder=current_polyorder)
        y_smooth = savgol_filter(y, window_length=window_length, polyorder=current_polyorder)

        # 將平滑後的座標重新打包
        smoothed_points = list(zip(x_smooth, y_smooth))
    else:
        # 緩衝區長度不足，直接使用原始座標
        smoothed_points = new_kpt_buffer

    return smoothed_points

def correct_track_id(person_df:pl.DataFrame, before_correctId:int, after_correctId:int, max_frame:int)->pl.DataFrame:
    if person_df.is_empty():
        return

    if (before_correctId not in person_df['track_id'].unique()) or (after_correctId not in person_df['track_id'].unique()):
        return

    if (before_correctId in person_df['track_id'].unique()) and (after_correctId in person_df['track_id'].unique()):
        for i in range(0, max(max_frame)):
            # condition_1 = (person_df['frame_number'] == i) & (person_df['track_id'] == before_correctId)
            person_df = person_df.with_columns(
                pl.when(
                    (person_df['frame_number'] == i) & (person_df['track_id'] == before_correctId)
                )
                .then(after_correctId)
                .otherwise(person_df['track_id'])
                .alias('track_id')
            )

    return person_df

#process 3d joints

def update_pose_results(new_person_df: pl.DataFrame, pred_instances, track_ids: list):
    """
    將平滑後的關鍵點數據從 new_person_df 更新回 data_samples 的結構。

    Args:
        new_person_df (pl.DataFrame): 包含平滑後關鍵點數據的 DataFrame。
        pose_results: 原始的姿態數據樣本結構（包含 pred_instances）。
        track_ids (list): 處理的 track_id 列表。

    Returns:
        pose_results: 更新後的姿態數據樣本結構。
    """

    # 遍歷每個 track_id 並更新對應的數據
    for track_id in track_ids:
        # 從 new_person_df 中篩選該 track_id 的平滑數據
        person_data = new_person_df.filter(pl.col('track_id') == track_id)

        if person_data.height == 0:  # 如果該 track_id 無數據，跳過
            continue

        # 提取平滑後的關鍵點
        keypoints_list = person_data.select('keypoints').to_numpy()[0][0]
        smoothed_keypoints = np.array([kp[:2] for kp in keypoints_list])
        # 更新到 pose_results 的 pred_instances 中
        smoothed_keypoints = smoothed_keypoints[:17]
        smoothed_keypoints_tensor = torch.tensor(smoothed_keypoints, dtype=torch.float64)

        for pred_instance in pred_instances:
            # pred_instance['keypoints'][0] = smoothed_keypoints_tensor.clone()
              pred_instance['keypoints'][0][:17] = smoothed_keypoints_tensor
    return pred_instances

def extract_3d_data(data_3d_samples, track_ids):
    """
    從 data_3d_samples 中提取 3D 骨架數據，並轉換為 DataFrame 格式。

    Args:
        data_3d_samples (list): 包含 3D 骨架數據的列表。
        frame_number (int): 當前幀號。

    Returns:
        pl.DataFrame: 包含提取數據的 DataFrame。
    """
    records = []
    for i, sample in enumerate(data_3d_samples):
        track_id = track_ids[i]

        keypoints = sample.keypoints[0].tolist()
        records.append({
            'track_id': track_id,
            'keypoints_3d': keypoints
        })

    return pl.DataFrame(records)

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
    new_person_df = new_person_df.join(
        data_df,
        on='track_id',
        how='left'
    )
    return new_person_df

def postprocess_pose_lift(pose_lift_results, pose_results):
    """後處理提升的 3D 骨架數據。"""
    for idx, pose_lift_result in enumerate(pose_lift_results):
        pose_lift_result.track_id = pose_results[idx].get('track_id', 1e4)

        pred_instances = pose_lift_result.pred_instances
        keypoints = pred_instances.keypoints
        keypoint_scores = pred_instances.keypoint_scores
        if keypoint_scores.ndim == 3:
            keypoint_scores = np.squeeze(keypoint_scores, axis=1)
            pose_lift_results[idx].pred_instances.keypoint_scores = keypoint_scores
        if keypoints.ndim == 4:
            keypoints = np.squeeze(keypoints, axis=1)

        keypoints = keypoints[..., [0, 2, 1]]
        keypoints[..., 0] = -keypoints[..., 0]
        keypoints[..., 2] = -keypoints[..., 2]

        # rebase height (z-axis)
        # if not args.disable_rebase_keypoint:
        keypoints[..., 2] -= np.min(
            keypoints[..., 2], axis=-1, keepdims=True)

        pose_lift_results[idx].pred_instances.keypoints = keypoints
    pose_lift_results = sorted(
            pose_lift_results, key=lambda x: x.get('track_id', 1e4))

    return pose_lift_results