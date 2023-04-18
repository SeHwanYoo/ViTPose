# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

from xtcocotools.coco import COCO

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

from mmpose.datasets import DatasetInfo

import pandas as pd

# pose_config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py'
# pose_config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/4xmspn50_coco_256x192.py'
pose_config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
pose_checkpoint = 'outputs/20221007/latest.pth'

json_file = '../../datasets/GM/annotations/test_baby_keypoints.json'
img_root = '../../datasets/GM/images/test'
out_img_root = 'evaluation' 

def main():
    coco = COCO(json_file)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        pose_config, pose_checkpoint, device='cuda:1')

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    img_keys = list(coco.imgs.keys())

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    
    results_points = pd.DataFrame(columns=['file_name', 
                                           'id', 
                                           'head_x', 
                                           'head_y', 
                                           'eye_r_x', 
                                           'eye_r_y', 
                                           'eye_l_x', 
                                           'eye_l_y',
                                           'neck_x', 
                                           'neck_y', 
                                           'right_shoulder_x',
                                           'right_shoulder_y',
                                           'right_elbow_x',
                                           'right_elbow_y',
                                           'right_wrist_x',
                                           'right_wrist_y',
                                           'right_hand_x',
                                           'right_hand_y',
                                           'left_shoulder_x',
                                           'left_shoulder_y',
                                           'left_elbow_x',
                                           'left_elbow_y',
                                           'left_wrist_x',
                                           'left_wrist_y',
                                           'left_hand_x',
                                           'left_hand_y',
                                           'pelvis_x',
                                           'pelvis_y',
                                           'right_hip_x',
                                           'right_hip_y',
                                           'right_knee_x',
                                           'right_knee_y',
                                           'right_ankle_x',
                                           'right_ankle_y',
                                           'right_foot_x',
                                           'right_foot_y',
                                           'left_hip_x',
                                           'left_hip_y',
                                           'left_knee_x',
                                           'left_knee_y',
                                           'left_ankle_x',
                                           'left_ankle_y',
                                           'left_foot_x',
                                           'left_foot_y',
                                           ])

    # process each image
    # with open(os.path.join(out_img_root, 'result_keypoints.txt'), 'w') as f:
    for i in range(len(img_keys)):
        # get bounding box annotations
        image_id = img_keys[i]
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(img_root, image['file_name'])
        ann_ids = coco.getAnnIds(image_id)

        # make person bounding boxes
        person_results = []
        for ann_id in ann_ids:
            person = {}
            ann = coco.anns[ann_id]
            # bbox format is 'xywh'
            person['bbox'] = ann['bbox']
            person_results.append(person)

        # test a single image, with a list of bboxes
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image_name,
            person_results,
            bbox_thr=None,
            format='xywh',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        
        results_points.loc[i, 'file_name'] = image['file_name']
        results_points.loc[i, 'id'] = image['id']
        results_points.loc[i, 'head_x'] = pose_results[0]['keypoints'][0][0]
        results_points.loc[i, 'head_y'] = pose_results[0]['keypoints'][0][1]
        results_points.loc[i, 'eye_r_x']  = pose_results[0]['keypoints'][1][0]
        results_points.loc[i, 'eye_r_y']  = pose_results[0]['keypoints'][1][1]
        results_points.loc[i, 'eye_l_x']  = pose_results[0]['keypoints'][2][0]
        results_points.loc[i, 'eye_l_y'] = pose_results[0]['keypoints'][2][1]
        results_points.loc[i, 'neck_x']  = pose_results[0]['keypoints'][3][0]
        results_points.loc[i, 'neck_y']  = pose_results[0]['keypoints'][3][1]
        results_points.loc[i, 'right_shoulder_x'] = pose_results[0]['keypoints'][4][0]
        results_points.loc[i, 'right_shoulder_y'] = pose_results[0]['keypoints'][4][1]
        results_points.loc[i, 'right_elbow_x'] = pose_results[0]['keypoints'][5][0]
        results_points.loc[i, 'right_elbow_y'] = pose_results[0]['keypoints'][5][1]
        results_points.loc[i, 'right_wrist_x'] = pose_results[0]['keypoints'][6][0]
        results_points.loc[i, 'right_wrist_y'] = pose_results[0]['keypoints'][6][1]
        results_points.loc[i, 'right_hand_x'] = pose_results[0]['keypoints'][7][0]
        results_points.loc[i, 'right_hand_y'] = pose_results[0]['keypoints'][7][1]
        results_points.loc[i, 'left_shoulder_x'] = pose_results[0]['keypoints'][8][0]
        results_points.loc[i, 'left_shoulder_y'] = pose_results[0]['keypoints'][8][1]
        results_points.loc[i, 'left_elbow_x'] = pose_results[0]['keypoints'][9][0]
        results_points.loc[i, 'left_elbow_y'] = pose_results[0]['keypoints'][9][1]
        results_points.loc[i, 'left_wrist_x'] = pose_results[0]['keypoints'][10][0]
        results_points.loc[i, 'left_wrist_y'] = pose_results[0]['keypoints'][10][1]
        results_points.loc[i, 'left_hand_x'] = pose_results[0]['keypoints'][11][0]
        results_points.loc[i, 'left_hand_y'] = pose_results[0]['keypoints'][11][1]
        results_points.loc[i, 'pelvis_x'] = pose_results[0]['keypoints'][12][0]
        results_points.loc[i, 'pelvis_y'] = pose_results[0]['keypoints'][12][1]
        results_points.loc[i, 'right_hip_x'] = pose_results[0]['keypoints'][13][0]
        results_points.loc[i, 'right_hip_y'] = pose_results[0]['keypoints'][13][1]
        results_points.loc[i, 'right_knee_x'] = pose_results[0]['keypoints'][14][0]
        results_points.loc[i, 'right_knee_y'] = pose_results[0]['keypoints'][14][1]
        results_points.loc[i, 'right_ankle_x'] = pose_results[0]['keypoints'][15][0]
        results_points.loc[i, 'right_ankle_y'] = pose_results[0]['keypoints'][15][1]
        results_points.loc[i, 'right_foot_x'] = pose_results[0]['keypoints'][16][0]
        results_points.loc[i, 'right_foot_y'] = pose_results[0]['keypoints'][16][1]
        results_points.loc[i, 'left_hip_x'] = pose_results[0]['keypoints'][17][0]
        results_points.loc[i, 'left_hip_y'] = pose_results[0]['keypoints'][17][1]
        results_points.loc[i, 'left_knee_x'] = pose_results[0]['keypoints'][18][0]
        results_points.loc[i, 'left_knee_y'] = pose_results[0]['keypoints'][18][1]
        results_points.loc[i, 'left_ankle_x'] = pose_results[0]['keypoints'][19][0]
        results_points.loc[i, 'left_ankle_y'] = pose_results[0]['keypoints'][19][1]
        results_points.loc[i, 'left_foot_x'] = pose_results[0]['keypoints'][20][0]
        results_points.loc[i, 'left_foot_y'] = pose_results[0]['keypoints'][20][1]
        
        # result_name.append(image['file_name'])
        # result_id.append(str(image['id']))
        # result_head.append(str(pose_results[0]['keypoints'][0][0]) + ', ' + str(pose_results[0]['keypoints'][0][1]))
        # result_eye_r.append(str(pose_results[0]['keypoints'][1][0]) + ', ' + str(pose_results[0]['keypoints'][1][1]))
        # result_eye_l.append(str(pose_results[0]['keypoints'][2][0]) + ', ' + str(pose_results[0]['keypoints'][2][1]))
        # result_neck.append(str(pose_results[0]['keypoints'][3][0]) + ', ' + str(pose_results[0]['keypoints'][3][1]))
        
        # result_right_shoulder.append(str(pose_results[0]['keypoints'][4][0]) + ', ' + str(pose_results[0]['keypoints'][4][1]))
        # result_right_elbow.append(str(pose_results[0]['keypoints'][5][0]) + ', ' + str(pose_results[0]['keypoints'][5][1]))
        # result_right_wrist.append(str(pose_results[0]['keypoints'][6][0]) + ', ' + str(pose_results[0]['keypoints'][6][1]))
        # result_right_hand.append(str(pose_results[0]['keypoints'][7][0]) + ', ' + str(pose_results[0]['keypoints'][7][1]))
        # result_left_shoulder.append(str(pose_results[0]['keypoints'][8][0]) + ', ' + str(pose_results[0]['keypoints'][8][1]))
        # result_left_elbow.append(str(pose_results[0]['keypoints'][9][0]) + ', ' + str(pose_results[0]['keypoints'][9][1]))
        # result_left_wrist.append(str(pose_results[0]['keypoints'][10][0]) + ', ' + str(pose_results[0]['keypoints'][10][1]))
        # result_left_hand.append(str(pose_results[0]['keypoints'][11][0]) + ', ' + str(pose_results[0]['keypoints'][11][1]))
        # result_pelvis.append(str(pose_results[0]['keypoints'][12][0]) + ', ' + str(pose_results[0]['keypoints'][12][1]))
        # result_right_hip.append(str(pose_results[0]['keypoints'][13][0]) + ', ' + str(pose_results[0]['keypoints'][13][1]))
        # result_right_knee.append(str(pose_results[0]['keypoints'][14][0]) + ', ' + str(pose_results[0]['keypoints'][14][1]))
        # result_right_ankle.append(str(pose_results[0]['keypoints'][15][0]) + ', ' + str(pose_results[0]['keypoints'][15][1]))
        # result_right_foot.append(str(pose_results[0]['keypoints'][16][0]) + ', ' + str(pose_results[0]['keypoints'][16][1]))
        # result_left_hip.append(str(pose_results[0]['keypoints'][17][0]) + ', ' + str(pose_results[0]['keypoints'][17][1]))
        # result_left_knee.append(str(pose_results[0]['keypoints'][18][0]) + ', ' + str(pose_results[0]['keypoints'][18][1]))
        # result_left_ankle.append(str(pose_results[0]['keypoints'][19][0]) + ', ' + str(pose_results[0]['keypoints'][19][1]))
        # result_left_foot.append(str(pose_results[0]['keypoints'][20][0]) + ', ' + str(pose_results[0]['keypoints'][20][1]))

        if out_img_root == '':
            out_file = None
        else:
            os.makedirs(out_img_root, exist_ok=True)
            out_file = os.path.join(out_img_root, f'vis_{i}.jpg')

        vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=0,
            # radius=args.radius,
            # thickness=args.thickness,
            show=False, 
            out_file=out_file)
        
    results_points.to_csv(os.path.join(out_img_root, 'result_keypoints.csv'))
        
    # with open(os.path.join(out_img_root, 'result_keypoints.txt'), 'w') as f:
    #     for i in range(len(result_name)):
            
    #         ff_write = ''
    #         ff_write += result_name[i] + ' // \n'
    #         ff_write += result_id[i] + '// \n'
    #         ff_write += '[' + result_head[i] + '], \n'
    #         ff_write += '[' + result_eye_r[i] + '], \n'
    #         ff_write += '[' + result_eye_l[i] + '], \n'
    #         ff_write += '[' + result_neck[i] + '] \n'
    #         ff_write += '[' + result_right_shoulder[i] + '] \n'
    #         ff_write += '[' + result_right_elbow[i] + '] \n'
    #         ff_write += '[' + result_right_wrist[i] + '] \n'
    #         ff_write += '[' + result_right_hand[i] + '] \n'
    #         ff_write += '[' + result_left_shoulder[i] + '] \n'
    #         ff_write += '[' + result_left_elbow[i] + '] \n'
    #         ff_write += '[' + result_left_wrist[i] + '] \n'
    #         ff_write += '[' + result_left_hand[i] + '] \n'
    #         ff_write += '[' + result_pelvis[i] + '] \n'
    #         ff_write += '[' + result_right_hip[i] + '] \n'
    #         ff_write += '[' + result_right_knee[i] + '] \n'
    #         ff_write += '[' + result_right_ankle[i] + '] \n'
    #         ff_write += '[' + result_right_foot[i] + '] \n'
    #         ff_write += '[' + result_left_hip[i] + '] \n'
    #         ff_write += '[' + result_left_knee[i] + '] \n'
    #         ff_write += '[' + result_left_ankle[i] + '] \n'
    #         ff_write += '[' + result_left_foot[i] + '] \n'
            
    #         f.write(ff_write)


if __name__ == '__main__':
    main()
