import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

cfg_file = "ml3d/configs/pointpillars_kitti_40ch.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.PointPillars(**cfg.model)
# cfg.dataset['dataset_path'] = "/home/amlab/KITTI_point"
# dataset = ml3d.datasets.KITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
# pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)

# # use a cache for storing the results of the preprocessing (default path is './logs/cache')
# # dataset = ml3d.datasets.KITTI(dataset_path='/home/amlab/KITTI_point', use_cache=True)

# # prints training progress in the console.
# pipeline.run_train()

# use a cache for storing the results of the preprocessing (default path is './logs/cache')
dataset = ml3d.datasets.KITTI(dataset_path='/home/amlab/KITTI_point', use_cache=True)

# create the model with random initialization.
# model = ml3d.models.PointPillars()

pipeline = ml3d.pipelines.ObjectDetection(model=model, dataset=dataset, device='cuda:0', **cfg.pipeline)

# prints training progress in the console.
pipeline.run_train()