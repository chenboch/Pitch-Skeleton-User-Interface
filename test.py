import cv2
import numpy as np
import os

# 設定參數
input_image_path = './test.jpg'  # 請改成你的照片路徑
output_video_path = 'output_video.avi'  # 改為 AVI 格式
frame_rate = 30  # 每秒幀數
duration = 5     # 影片秒數

# 讀取圖片
img = cv2.imread(input_image_path)
if img is None:
    print("無法讀取圖片，請檢查路徑是否正確")
    exit()

height, width = img.shape[:2]
total_frames = frame_rate * duration

# 定義影片寫入器，使用 XVID 編碼
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 改為 XVID
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

# 生成每一幀並檢查寫入前一致性
for i in range(total_frames):
    new = img.copy()  # 創建副本
    if np.array_equal(img, new):
        print(f"Frame {i} before write: identical to original")
    else:
        print(f"Frame {i} before write: DIFFERENT from original")
    out.write(new)

# 釋放資源
out.release()
print(f"影片已生成：{output_video_path}")

# 從影片讀回幀並檢查一致性
cap = cv2.VideoCapture(output_video_path)
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 比較讀回的幀與原始圖像
    if np.array_equal(img, frame):
        print(f"Frame {frame_count} after read: identical to original")
    else:
        print(f"Frame {frame_count} after read: DIFFERENT from original")
        # 如果不同，計算最大像素差異
        diff = np.abs(img - frame)
        print(f"Max pixel difference: {diff.max()}")
    frame_count += 1

cap.release()
cv2.destroyAllWindows()