from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector
import cv2

import sys 

# sys.path.append('../')
# i/mport mmcv
from mmcv import Config

# local_runtime = False
# try:
#   from google.colab.patches import cv2_imshow  # for image visualization in colab
# except:
#   local_runtime = True

cfg = Config.fromfile(
    './configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py'
)

# pose_config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py'


# pose_checkpoint = 'work_dirs/hrnet_w32_coco_tiny_256x192/latest.pth'
pose_checkpoint = 'outputs/20221007/latest.pth'
det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# initialize pose model
pose_model = init_pose_model(cfg, pose_checkpoint)
# initialize detector
det_model = init_detector(det_config, det_checkpoint)

# img = 'tests/data/coco/000000196141.jpg'
img = '/home/sehwan/datasets/GM/images/test/004598.png'

# inference detection
mmdet_results = inference_detector(det_model, img)

# extract person (COCO_ID=1) bounding boxes from the detection results
person_results = process_mmdet_results(mmdet_results, cat_id=1)

# inference pose
pose_results, returned_outputs = inference_top_down_pose_model(pose_model,
                                                               img,
                                                               person_results,
                                                               bbox_thr=0.3,
                                                               format='xyxy',
                                                               dataset='TopDownCocoDataset')

# show pose estimation results
vis_result = vis_pose_result(pose_model,
                             img,
                             pose_results,
                             kpt_score_thr=0.,
                             dataset='TopDownCocoDataset',
                             show=False)

# reduce image size
vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)


cv2.imwrite('outputs/20221007/images/result.png', vis_result)


# if local_runtime:
#   from IPython.display import Image, display
#   import tempfile
#   import os.path as osp
#   import cv2
#   with tempfile.TemporaryDirectory() as tmpdir:
#     file_name = osp.join(tmpdir, 'pose_results.png')
#     cv2.imwrite(file_name, vis_result)
#     display(Image(file_name))
# else:
#   cv2_imshow(vis_result)