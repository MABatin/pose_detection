## Introduction
Keypoint extraction is done using `mmpose` pose detection library.

For dataset, download from [drive_download](https://drive.google.com/drive/folders/1bjjf81XrGq5tNXQ_w1OcJQgQ8c1d3d6k?usp=share_link) and extract like this:
```text
pose_detection
|── data
    |── images
        |-- Image_1.jpg
        |-- Image_4.jpg
        |-- ...
    |── labels.csv
```

Run script `imabalance.py` for creating `unbalanced_keypoints.csv`.

`unbalanced_keypoints.csv` file format is:

image_file_name, label, <x_1, y_1>, <x_2, y_2>, ..., <x_n, y_n>

where each <x_n, y_n> corresponds to the coordinate of a keypoint. Each image has a total of 17 keypoints.
`body/2d_kpt_sview_rgb_img/topdown_heatmap`

Run script `balance.py` for creating `balanced_keypoints.csv`.

Balancing is done by using `RandomOverSampler` from `imblearn` to upsample minority labels.

Resultant `.csv` files are saved in
```text
pose_detection
|── results
    |-- balanced_keypoints.csv
    |-- unbalanced_keypoints.csv
```    

Run script `misc.py` for counting labels for each csv file





