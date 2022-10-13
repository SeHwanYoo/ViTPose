from mmpose.datasets import build_dataset
from mmpose.models import build_posenet
from mmpose.apis import train_model
import mmcv

import baby_config

# build dataset
datasets = [build_dataset(baby_config.cfg.data.train)]

# build model
model = build_posenet(baby_config.cfg.model)

# create work_dir
mmcv.mkdir_or_exist(baby_config.cfg.work_dir)

# train model
train_model(
    model, datasets, baby_config.cfg, distributed=False, validate=True, meta=dict())