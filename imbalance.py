import os.path as osp
import pandas as pd
import mmcv

from mmpose.apis import (inference_top_down_pose_model, init_pose_model)


def extract_keypoints(model, root, images):
    keypoints = dict()
    # inference pose
    for img in mmcv.track_iter_progress(images):
        image = osp.join(root, img)
        pose_results, returned_outputs = inference_top_down_pose_model(
            model,
            image,
            format='xywh',
            dataset=model.cfg.data.test.type)
        kp = pose_results[0]['keypoints'][:, :-1].flatten()
        kp = [int(a) for a in kp]
        kp = [0 if i < 0 else i for i in kp]
        keypoints[img] = kp

    return keypoints


def create_csv(keypoints, dataset, out_csv):
    K = []
    for img in dataset['img']:
        K.append(keypoints[img])
    labels = dataset['label'].to_frame()

    # Create unbalanced dataframe
    df = dataset['img'].to_frame().join(pd.DataFrame(K))
    df = df.join(labels)

    df.to_csv(out_csv, index=False, header=None)


def init_model(config, checkpoint):
    # initialize pose model
    model = init_pose_model(config, checkpoint)

    return model


def main():
    img_folder = 'data/images/'
    in_csv = 'data/labels.csv'
    out_csv = 'results/unbalanced_keypoints.csv'
    pose_config = 'configs/body/hrnet_w48_coco_256x192.py'
    # pose_checkpoint = 'checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
    pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'

    dataset = pd.read_csv(in_csv, sep=',', names=['img', 'label'])
    images = dataset['img']

    pose_model = init_model(pose_config, pose_checkpoint)

    # Inference pose on individual images and extract 17 keypoints
    keypoints = extract_keypoints(pose_model, img_folder, images)

    # Create unbalanced_keypoints.csv
    create_csv(keypoints, dataset, out_csv)


if __name__ == '__main__':
    main()
