# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

from xtcocotools.coco import COCO

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

from mmpose.datasets import DatasetInfo

pose_config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py'
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
    
    
    result_name = []
    result_id = []
    result_head = []
    result_eye_r = []
    result_eye_l = []
    result_neck = []
    
    result_right_shoulder = []
    result_right_elbow = []
    result_right_wrist = []
    result_right_hand = []
    result_left_shoulder = []
    result_left_elbow = []
    result_left_wrist = []
    result_left_hand = []
    result_pelvis = []
    result_right_hip = []
    result_right_knee = []
    result_right_ankle = []
    result_right_foot = []
    result_left_hip = []
    result_left_knee = []
    result_left_ankle = []
    result_left_foot = []

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
        
        result_name.append(image['file_name'])
        result_id.append(str(image['id']))
        result_head.append(str(pose_results[0]['keypoints'][0][0]) + ', ' + str(pose_results[0]['keypoints'][0][1]))
        result_eye_r.append(str(pose_results[0]['keypoints'][1][0]) + ', ' + str(pose_results[0]['keypoints'][1][1]))
        result_eye_l.append(str(pose_results[0]['keypoints'][2][0]) + ', ' + str(pose_results[0]['keypoints'][2][1]))
        result_neck.append(str(pose_results[0]['keypoints'][3][0]) + ', ' + str(pose_results[0]['keypoints'][3][1]))
        
        result_right_shoulder.append(str(pose_results[0]['keypoints'][4][0]) + ', ' + str(pose_results[0]['keypoints'][4][1]))
        result_right_elbow.append(str(pose_results[0]['keypoints'][5][0]) + ', ' + str(pose_results[0]['keypoints'][5][1]))
        result_right_wrist.append(str(pose_results[0]['keypoints'][6][0]) + ', ' + str(pose_results[0]['keypoints'][6][1]))
        result_right_hand.append(str(pose_results[0]['keypoints'][7][0]) + ', ' + str(pose_results[0]['keypoints'][7][1]))
        result_left_shoulder.append(str(pose_results[0]['keypoints'][8][0]) + ', ' + str(pose_results[0]['keypoints'][8][1]))
        result_left_elbow.append(str(pose_results[0]['keypoints'][9][0]) + ', ' + str(pose_results[0]['keypoints'][9][1]))
        result_left_wrist.append(str(pose_results[0]['keypoints'][10][0]) + ', ' + str(pose_results[0]['keypoints'][10][1]))
        result_left_hand.append(str(pose_results[0]['keypoints'][11][0]) + ', ' + str(pose_results[0]['keypoints'][11][1]))
        result_pelvis.append(str(pose_results[0]['keypoints'][12][0]) + ', ' + str(pose_results[0]['keypoints'][12][1]))
        result_right_hip.append(str(pose_results[0]['keypoints'][13][0]) + ', ' + str(pose_results[0]['keypoints'][13][1]))
        result_right_knee.append(str(pose_results[0]['keypoints'][14][0]) + ', ' + str(pose_results[0]['keypoints'][14][1]))
        result_right_ankle.append(str(pose_results[0]['keypoints'][15][0]) + ', ' + str(pose_results[0]['keypoints'][15][1]))
        result_right_foot.append(str(pose_results[0]['keypoints'][16][0]) + ', ' + str(pose_results[0]['keypoints'][16][1]))
        result_left_hip.append(str(pose_results[0]['keypoints'][17][0]) + ', ' + str(pose_results[0]['keypoints'][17][1]))
        result_left_knee.append(str(pose_results[0]['keypoints'][18][0]) + ', ' + str(pose_results[0]['keypoints'][18][1]))
        result_left_ankle.append(str(pose_results[0]['keypoints'][19][0]) + ', ' + str(pose_results[0]['keypoints'][19][1]))
        result_left_foot.append(str(pose_results[0]['keypoints'][20][0]) + ', ' + str(pose_results[0]['keypoints'][20][1]))

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
        
    with open(os.path.join(out_img_root, 'result_keypoints.txt'), 'w') as f:
        for i in range(len(result_name)):
            
            ff_write = ''
            ff_write += result_name[i] + ' // \n'
            ff_write += result_id[i] + '// \n'
            ff_write += '[' + result_head[i] + '], \n'
            ff_write += '[' + result_eye_r[i] + '], \n'
            ff_write += '[' + result_eye_l[i] + '], \n'
            ff_write += '[' + result_neck[i] + '] \n'
            ff_write += '[' + result_right_shoulder[i] + '] \n'
            ff_write += '[' + result_right_elbow[i] + '] \n'
            ff_write += '[' + result_right_wrist[i] + '] \n'
            ff_write += '[' + result_right_hand[i] + '] \n'
            ff_write += '[' + result_left_shoulder[i] + '] \n'
            ff_write += '[' + result_left_elbow[i] + '] \n'
            ff_write += '[' + result_left_wrist[i] + '] \n'
            ff_write += '[' + result_left_hand[i] + '] \n'
            ff_write += '[' + result_pelvis[i] + '] \n'
            ff_write += '[' + result_right_hip[i] + '] \n'
            ff_write += '[' + result_right_knee[i] + '] \n'
            ff_write += '[' + result_right_ankle[i] + '] \n'
            ff_write += '[' + result_right_foot[i] + '] \n'
            ff_write += '[' + result_left_hip[i] + '] \n'
            ff_write += '[' + result_left_knee[i] + '] \n'
            ff_write += '[' + result_left_ankle[i] + '] \n'
            ff_write += '[' + result_left_foot[i] + '] \n'
            
            f.write(ff_write)


if __name__ == '__main__':
    main()
