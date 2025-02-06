coco_keypoint_info = {
    "keypoints": {
        0: '骨盆 (root)',
        1: '右臀部',
        2: '右膝蓋',
        3: '右腳',
        4: '左臀部',
        5: '左膝蓋',
        6: '左腳',
        7: '脊椎',
        8: '胸腔',
        9: '頸部基底',
        10: '頭部',
        11: '左肩膀',
        12: '左手肘',
        13: '左手腕',
        14: '右肩膀',
        15: '右手肘',
        16: '右手腕'
    },
    "skeleton_links": [
        (0, 1),  # root (pelvis) -> right_hip
        (1, 2),  # right_hip -> right_knee
        (2, 3),  # right_knee -> right_foot
        (0, 4),  # root (pelvis) -> left_hip
        (4, 5),  # left_hip -> left_knee
        (5, 6),  # left_knee -> left_foot
        (0, 7),  # root (pelvis) -> spine
        (7, 8),  # spine -> thorax
        (8, 9),  # thorax -> neck_base
        (9, 10), # neck_base -> head
        (8, 11), # thorax -> left_shoulder
        (11, 12),# left_shoulder -> left_elbow
        (12, 13),# left_elbow -> left_wrist
        (8, 14), # thorax -> right_shoulder
        (14, 15),# right_shoulder -> right_elbow
        (15, 16) # right_elbow -> right_wrist
    ],
    # 右边关节连接
    "right_limb": [
        (1, 2),  # right_hip -> right_knee
        (2, 3),  # right_knee -> right_foot
        (8, 14), # thorax -> right_shoulder
        (14, 15),# right_shoulder -> right_elbow
        (15, 16) # right_elbow -> right_wrist
    ],
    # 左边关节连接
    "left_limb": [
        (4, 5),  # left_hip -> left_knee
        (5, 6),  # left_knee -> left_foot
        (8, 11), # thorax -> left_shoulder
        (11, 12),# left_shoulder -> left_elbow
        (12, 13) # left_elbow -> left_wrist
    ]
}