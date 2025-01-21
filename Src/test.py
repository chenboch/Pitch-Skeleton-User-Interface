import polars as pl

# 建立範例 DataFrame
df = pl.DataFrame({
    "track_id": [1, 1, 1, 1, 1],
    "bbox": [
        [410.754137, 204.889498, 1000.0],
        [409.68621, 204.371573, 1007.0],
        [408.319632, 203.939368, 1006.0],
        [406.528031, 203.582769, 1008.0],
        [404.998253, 203.296911, 1005.0]
    ],
    "area": [693307.13, 692634.65, 692057.1, 691308.36, 690879.34],
    "keypoints": [
        [[594.708618, 285.554138, 0.9], [111.0, 222.0, 0.8]],
        [[594.413269, 285.409882, 0.8],  [111.0, 222.0, 0.8]],
        [[591.90741, 285.17041, 0.8]],
        [[589.502747, 285.182312, 0.8]],
        None
    ],
    "frame_number": [47, 48, 49, 50, 51]
})

# 修改 frame_number = 50 的 keypoints
smoothed_kpts = [[400.90741, 285.17041, 0.8],[400.90741, 285.17041, 0.8]]

search_person_df = df.filter(
           (pl.col("frame_number") == 48) & 
                (pl.col("track_id") == 1)
    )
search_person_df =search_person_df["keypoints"].to_list()[0]
search_person_df[0] = [0,1] + search_person_df[0][2:]

print(search_person_df)
search_person_df[0] = 2 # 修改對應位置的值
search_person_df[1] = 3
print(search_person_df)
# search_person_df["keypoints"][0][0][0] = 1
# print(search_person_df["keypoints"][0][0])
new_person_df = df.with_columns(
        pl.when(
                (pl.col("frame_number") == 50) & 
                (pl.col("track_id") == 1)
            )
        .then(pl.Series("keypoints", [smoothed_kpts]))
        .otherwise(df["keypoints"])
        .alias("keypoints")
)


# print(new_person_df)
