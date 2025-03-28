halpe26_keypoint_info = {
    "keypoints": {
        0: "鼻子",
        1: "左眼",
        2: "右眼",
        3: "左耳",
        4: "右耳",
        5: "左肩",
        6: "右肩",
        7: "左肘",
        8: "右肘",
        9: "左腕",
        10: "右腕",
        11: "左髖",
        12: "右髖",
        13: "左膝",
        14: "右膝",
        15: "左踝",
        16: "右踝",
        17: "頭部",
        18: "頸部",
        19: "臀部",
        20: "左大腳趾",
        21: "右大腳趾",
        22: "左小腳趾",
        23: "右小腳趾",
        24: "左腳跟",
        25: "右腳跟"
    },
    "skeleton_links":[
        [0, 1], [0, 2], [1, 3], [2, 4], # 頭
        [5, 18], [6, 18], [17, 18],[18, 19],#軀幹
        [5, 7], [7, 9],                 #左手
        [6, 8], [8, 10],                #右手
        [19, 11], [11, 13], [13, 15],   #左腿
        [19, 12], [12, 14], [14, 16],   #右腿
        [20, 24], [22, 24], [15, 24],   #左腳
        [21, 25], [23, 25], [16, 25]    #右腳
    ],
    "left_points_indices": [[5, 18], [5, 7], [7, 9],[19, 11], [11, 13], [13, 15], [20, 24], [22, 24], [15, 24]],  # Indices of left hand, leg, and foot keypoints
    "right_points_indices": [[6, 18], [6, 8], [8, 10], [19, 12], [12, 14], [14, 16], [21, 25], [23, 25], [16, 25]],  # Indices of right hand, leg, and foot keypoints
    "angle_dict":{
        '左手肘': [5, 7, 9],
        '右手肘': [6, 8, 10],
        '左肩': [11, 5, 7],
        '右肩': [12, 6, 8],
        '左膝': [11, 13, 15],
        '右膝': [12, 14, 16]
    }
}
halpe26_to_posetrack_keypoint_info = {
    "keypoints": {
        0: 0, # "鼻子"
        # 1: 0,#"左眼",
        # 2: 0,#"右眼",
        # 3: 3,#"左耳",
        # 4: 4,#"右耳",
        # 5: 5,#"左肩",
        # 6: 6,#"右肩",
        # 7: 7,#"左肘",
        # 8: 8,#"右肘",
        # 9: 9,#"左腕",
        # 10: 10,#"右腕",
        # 11: 11,#"左髖",
        # 12: 12,#"右髖",
        # 13: 13,#"左膝",
        # 14: 14,#"右膝",
        # 15: 15,#"左踝",
        # 16: 16,#"右踝",
        17: 2,#"頭部",
        18: 1,#"頸部",
        19: 18,#"臀部",
        # 20: -1,#"左大腳趾",
        # 21: -1,#"右大腳趾",
        # 22: -1,#"左小腳趾",
        # 23: -1,#"右小腳趾",
        # 24: -1,#"左腳跟",
        # 25: -1,#"右腳跟"
    },
}
