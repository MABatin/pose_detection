import glob
import os.path as osp

import numpy as np
import pandas as pd
import mmcv

from mmpose.apis import (inference_top_down_pose_model, init_pose_model)


def extract_keypoints(model, img):
    # inference pose
    pose_results, returned_outputs = inference_top_down_pose_model(
        model,
        img,
        format='xywh',
        dataset=model.cfg.data.test.type)

    # return keypoints with (x, y) co-ordinates w/o confidence score
    keypoints = pose_results[0]['keypoints'][:, :-1].flatten()
    keypoints = [int(a) for a in keypoints]
    keypoints = [0 if i < 0 else i for i in keypoints]
    return keypoints


def create_csv(keypoints, lbl_csv, kp_csv):
    df = pd.read_csv(lbl_csv, sep=',', names=['img', 'label'])

    K = []
    for img in df['img']:
        K.append(keypoints[img])
    labels = df['label'].to_frame()

    # Create unbalanced dataframe
    df = df['img'].to_frame().join(pd.DataFrame(K))
    df = df.join(labels)

    df.to_csv(kp_csv, index=False, header=None)


def init_model(config, checkpoint):
    # initialize pose model
    model = init_pose_model(config, checkpoint)

    return model


def main():
    img_folder = 'data/images/'
    pose_config = 'configs/body/hrnet_w48_coco_256x192.py'
    pose_checkpoint = 'checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
    # pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'

    images = glob.glob(osp.join(img_folder, '*.jpg'), recursive=True)
    pose_model = init_model(pose_config, pose_checkpoint)
    keypoints_ = dict()

    # Inference pose on individual images and extract 17 keypoints
    for image in mmcv.track_iter_progress(images):
        image_name = image.split('\\')[-1]
        k = extract_keypoints(pose_model, image)
        keypoints = {image_name: k}
        keypoints_.update(keypoints)

    # Create unbalanced_keypoints.csv
    create_csv(keypoints_, lbl_csv='data/labels.csv', kp_csv='results/unbalanced_keypoints.csv')


if __name__ == '__main__':
    main()
