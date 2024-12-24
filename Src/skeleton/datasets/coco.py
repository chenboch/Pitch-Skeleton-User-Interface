coco_keypoint_info = {
    "keypoints": {
        0: "鼻子",
        1: "左眼",
        2: "右眼",
        3: "左耳",
        4: "右耳",
        5: "左肩",
        6: "右肩",
        7: "左手肘",
        8: "右手肘",
        9: "左手腕",
        10: "右手腕",
        11: "左髋",
        12: "右髋",
        13: "左膝",
        14: "右膝",
        15: "左腳踝",
        16: "右腳踝"
    },
    "skeleton_links": [
        [0, 1], [0, 2], [1, 3], [2, 4], # 頭
        #軀幹
        [5, 7], [7, 9],                 #左手
        [6, 8], [8, 10],                #右手
        [11, 13], [13, 15],   #左腿
        [12, 14], [14, 16],   #右腿
    ],
    "left_points_indices": [[5, 7], [7, 9], [11, 13], [13, 15]],  # Indices of left hand, leg, and foot keypoints
    "right_points_indices": [[6, 8], [8, 10], [12, 14], [14, 16]]  # Indices of right hand, leg, and foot keypoints
}