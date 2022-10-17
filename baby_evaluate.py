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
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    # parser = ArgumentParser()
    # parser.add_argument('pose_config', help='Config file for detection')
    # parser.add_argument('pose_checkpoint', help='Checkpoint file')
    # parser.add_argument('--img-root', type=str, default='', help='Image root')
    # parser.add_argument(
    #     '--json-file',
    #     type=str,
    #     default='',
    #     help='Json file containing image info.')
    # parser.add_argument(
    #     '--show',
    #     action='store_true',
    #     default=False,
    #     help='whether to show img')
    # parser.add_argument(
    #     '--out-img-root',
    #     type=str,
    #     default='',
    #     help='Root of the output img file. '
    #     'Default not saving the visualization images.')
    # parser.add_argument(
    #     '--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument(
    #     '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    # parser.add_argument(
    #     '--radius',
    #     type=int,
    #     default=4,
    #     help='Keypoint radius for visualization')
    # parser.add_argument(
    #     '--thickness',
    #     type=int,
    #     default=1,
    #     help='Link thickness for visualization')

    # args = parser.parse_args()
    # assert args.show or (args.out_img_root != '')

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
    
    results_output = {'file_name' : [], 
                      'keypoints' : []}

    # process each image
    with open(os.path.join(out_img_root, 'result_keypoints.txt'), 'w') as f:
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
            
            # image['file_name']
            # print(f'pose_results--------------------------->{type(pose_results)}')
            # raise TypeError('STOP')
            # results_output['file_name']
            # results_output['file_name'].append(image['file_name'])
            # results_output['keypoints'].append(pose_results)
            
            f.write(image['file_name'] + ' : [' + str(pose_results[0]['keypoints'][0][0]) + ', ' + str(pose_results[0]['keypoints'][0][1]) + ']')

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
        
    # with open(os.path.join(out_img_root, 'result_keypoints.txt'), 'w') as f:
    #     for key, val in results_output.items():
    #         f.write(val) 


if __name__ == '__main__':
    main()
