from mmcv import Config
# import baby_dataset 

import json
import os
import os.path as osp
from collections import OrderedDict
import tempfile

import numpy as np

from mmpose.core.evaluation.top_down_eval import (keypoint_nme,
												  keypoint_pck_accuracy)
from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.base import Kpt2dSviewRgbImgTopDownDataset
from mmcv import Config
@DATASETS.register_module()
class TopDownCOCOTinyDataset(Kpt2dSviewRgbImgTopDownDataset):
	
	def __init__(self,
				 ann_file,
				 img_prefix,
				 data_cfg,
				 pipeline,
				 dataset_info=None,
				 test_mode=False):
		super().__init__(
			ann_file, img_prefix, data_cfg, pipeline, dataset_info, coco_style=False, test_mode=test_mode)

		# flip_pairs, upper_body_ids and lower_body_ids will be used
		# in some data augmentations like random flip
		# self.ann_info['flip_pairs'] = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
									#    [11, 12], [13, 14], [15, 16]]
		self.ann_info['flip_pairs'] = [[1,2],[2,4],[4,3],[3,1], # Head
										[4,5],[4,9],[4,13],[13,14],[13,18], # Body
										[5,6],[6,7],[7,8], # Right arm
										[9,10],[10,11],[11,12], # Left arm
										[14,15],[15,16],[16,17], # Right leg
										[18,19],[19,20],[20,21] # Left leg
        ]
		self.ann_info['upper_body_ids'] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
		self.ann_info['lower_body_ids'] = (13, 14, 15, 16, 17, 18, 19, 20, 21)

		self.ann_info['joint_weights'] = None
		self.ann_info['use_different_joint_weights'] = False
  
		# self.ann_info['num_joints'] = 21
		# data_cfg['num_joints'] = 21

		self.dataset_name = 'coco_tiny'
		self.db = self._get_db()

	def _get_db(self):
		with open(self.ann_file) as f:
			anns = json.load(f)

		db = []
		for idx, ann in enumerate(anns):
      
			for a in ann:
				print(a)
      
			print(f'ann -> {ann.info}')
      
			image_file = osp.join(self.img_prefix, ann['file_name'])
			# get bbox
			bbox = ann['bbox']
			center, scale = self._xywh2cs(*bbox)
			# get keypoints
			keypoints = np.array(
				ann['keypoints'], dtype=np.float32).reshape(-1, 3)
			num_joints = keypoints.shape[0]
			joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
			joints_3d[:, :2] = keypoints[:, :2]
			joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
			joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

			sample = {
				'image_file': image_file,
				'center': center,
				'scale': scale,
				'bbox': bbox,
				'rotation': 0,
				'joints_3d': joints_3d,
				'joints_3d_visible': joints_3d_visible,
				'bbox_score': 1,
				'bbox_id': idx,
			}
			db.append(sample)

		return db

	def _xywh2cs(self, x, y, w, h):
		aspect_ratio = self.ann_info['image_size'][0] / self.ann_info[
			'image_size'][1]
		center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
		if w > aspect_ratio * h:
			h = w * 1.0 / aspect_ratio
		elif w < aspect_ratio * h:
			w = h * aspect_ratio

		# pixel std is 200.0
		scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
		# padding to include proper amount of context
		scale = scale * 1.25
		return center, scale

	def evaluate(self, results, res_folder=None, metric='PCK', **kwargs):
		metrics = metric if isinstance(metric, list) else [metric]
		allowed_metrics = ['PCK', 'NME']
		for metric in metrics:
			if metric not in allowed_metrics:
				raise KeyError(f'metric {metric} is not supported')

		if res_folder is not None:
			tmp_folder = None
			res_file = osp.join(res_folder, 'result_keypoints.json')
		else:
			tmp_folder = tempfile.TemporaryDirectory()
			res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

		kpts = []
		for result in results:
			preds = result['preds']
			boxes = result['boxes']
			image_paths = result['image_paths']
			bbox_ids = result['bbox_ids']

			batch_size = len(image_paths)
			for i in range(batch_size):
				kpts.append({
					'keypoints': preds[i].tolist(),
					'center': boxes[i][0:2].tolist(),
					'scale': boxes[i][2:4].tolist(),
					'area': float(boxes[i][4]),
					'score': float(boxes[i][5]),
					'bbox_id': bbox_ids[i]
				})
		kpts = self._sort_and_unique_bboxes(kpts)

		self._write_keypoint_results(kpts, res_file)
		info_str = self._report_metric(res_file, metrics)
		name_value = OrderedDict(info_str)

		if tmp_folder is not None:
			tmp_folder.cleanup()

		return name_value

	def _report_metric(self, res_file, metrics, pck_thr=0.3):
		"""Keypoint evaluation.

		Args:
		res_file (str): Json file stored prediction results.
		metrics (str | list[str]): Metric to be performed.
			Options: 'PCK', 'NME'.
		pck_thr (float): PCK threshold, default: 0.3.

		Returns:
		dict: Evaluation results for evaluation metric.
		"""
		info_str = []

		with open(res_file, 'r') as fin:
			preds = json.load(fin)
		assert len(preds) == len(self.db)

		outputs = []
		gts = []
		masks = []

		for pred, item in zip(preds, self.db):
			outputs.append(np.array(pred['keypoints'])[:, :-1])
			gts.append(np.array(item['joints_3d'])[:, :-1])
			masks.append((np.array(item['joints_3d_visible'])[:, 0]) > 0)

		outputs = np.array(outputs)
		gts = np.array(gts)
		masks = np.array(masks)

		normalize_factor = self._get_normalize_factor(gts)

		if 'PCK' in metrics:
			_, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr,
											  normalize_factor)
			info_str.append(('PCK', pck))

		if 'NME' in metrics:
			info_str.append(
				('NME', keypoint_nme(outputs, gts, masks, normalize_factor)))

		return info_str

	@staticmethod
	def _write_keypoint_results(keypoints, res_file):
		"""Write results into a json file."""

		with open(res_file, 'w') as f:
			json.dump(keypoints, f, sort_keys=True, indent=4)

	@staticmethod
	def _sort_and_unique_bboxes(kpts, key='bbox_id'):
		"""sort kpts and remove the repeated ones."""
		kpts = sorted(kpts, key=lambda x: x[key])
		num = len(kpts)
		for i in range(num - 1, 0, -1):
			if kpts[i][key] == kpts[i - 1][key]:
				del kpts[i]

		return kpts
	
	@staticmethod
	def _get_normalize_factor(gts):
		"""Get inter-ocular distance as the normalize factor, measured as the
		Euclidean distance between the outer corners of the eyes.

		Args:
			gts (np.ndarray[N, K, 2]): Groundtruth keypoint location.

		Return:
			np.ndarray[N, 2]: normalized factor
		"""

		interocular = np.linalg.norm(
			gts[:, 0, :] - gts[:, 1, :], axis=1, keepdims=True)
		return np.tile(interocular, [1, 2])


cfg = Config.fromfile(
    './configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
)

cfg.channel_cfg.num_output_channels = 21
cfg.channel_cfg.dataset_joints = 21
cfg.channel_cfg.dataset_channel = [ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
cfg.channel_cfg.inference_channel = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ]

# set basic configs
cfg.data_root = '/home/sehwan/datasets/GM'
# cfg.work_dir = 'work_dirs/hrnet_w32_coco_tiny_256x192'
cfg.work_dir = 'outputs'
cfg.gpu_ids = range(1)
cfg.seed = 0

# set log interval
cfg.log_config.interval = 1

# set evaluation configs
cfg.evaluation.interval = 10
cfg.evaluation.metric = 'PCK'
cfg.evaluation.save_best = 'PCK'

# set learning rate policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    step=[17, 35])
cfg.total_epochs = 40

# set batch size
cfg.data.samples_per_gpu = 16
cfg.data.val_dataloader = dict(samples_per_gpu=16)
cfg.data.test_dataloader = dict(samples_per_gpu=16)


# set dataset configs
cfg.data.train.type = 'TopDownCOCOTinyDataset'
cfg.data.train.ann_file = f'{cfg.data_root}/annotations/train_baby_keypoints.json'
cfg.data.train.img_prefix = f'{cfg.data_root}/images/train/'

cfg.data.val.type = 'TopDownCOCOTinyDataset'
cfg.data.val.ann_file = f'{cfg.data_root}/annotations/valid_baby_keypoints.json'
cfg.data.val.img_prefix = f'{cfg.data_root}/images/valid/'

cfg.data.test.type = 'TopDownCOCOTinyDataset'
cfg.data.test.ann_file = f'{cfg.data_root}/annotations/test_baby_keypoints.json'
cfg.data.test.img_prefix = f'{cfg.data_root}/images/test/'