import torch
from mmengine.structures import InstanceData
from mmpose.structures import PoseDataSample

# 模擬多個人物的輸入資料 (N = 2)
data = {
    'bboxes': [
        torch.tensor([471.0948, 198.7763, 762.6507, 1011.1557]),  # 第一個人物
        torch.tensor([300.0, 150.0, 600.0, 900.0])  # 第二個人物
    ],
    'keypoints': [[
        torch.tensor([[644., 281.], [615., 332.], [619., 222.], [619., 222.], [619., 222.],
                      [697., 360.], [530., 361.], [736., 474.], [516., 477.], [665., 510.],
                      [551., 512.], [672., 582.], [572., 577.], [667., 782.], [578., 767.],
                      [669., 930.], [547., 932.]])
    ], [
        torch.tensor([[500., 250.], [480., 300.], [470., 200.], [470., 200.], [470., 200.],
                      [530., 330.], [420., 340.], [550., 450.], [410., 460.], [500., 500.],
                      [450., 510.], [520., 570.], [470., 560.], [510., 750.], [460., 730.],
                      [530., 900.], [470., 890.]])
    ]],
    'keypoint_scores': [[
        torch.tensor([0.9349, 0.9285, 0.9039, 0.9039, 0.9039, 0.901, 0.8969, 0.9032, 0.8897,
                      0.901, 0.8785, 0.8725, 0.8742, 0.8848, 0.8945, 0.893, 0.8941])
    ], [
        torch.tensor([0.9102, 0.9055, 0.8901, 0.8901, 0.8901, 0.8888, 0.8700, 0.8899, 0.8655,
                      0.8750, 0.8600, 0.8500, 0.8602, 0.8703, 0.8801, 0.8800, 0.8810])
    ]]
}

# 轉換格式
pose_sample = PoseDataSample()

# 設定 metainfo，存儲所有 bbox
pose_sample.set_metainfo({'bboxes': torch.stack(data['bboxes'])})  # (N, 4)

# 使用 InstanceData 來封裝 keypoints 和 scores
pred_instances = InstanceData()
pred_instances.keypoints = torch.stack([k[0] for k in data['keypoints']])  # (N, K, 2)
pred_instances.keypoint_scores = torch.stack([s[0] for s in data['keypoint_scores']])  # (N, K)

# 設定 pred_instances
pose_sample.pred_instances = pred_instances

import numpy as np
for person in pred_instances:
    print(person['keypoints'])
    # exit()
    keypoints_data = np.hstack((
        np.round(person['keypoints'][0], 2),
        np.round(person['keypoint_scores'][0], 2).reshape(-1, 1),
        np.full((len(person['keypoints'][0]), 1), False, dtype=bool)
    ))

# 顯示結果
print(pose_sample)
