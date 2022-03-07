import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from open3d.ml.vis import Visualizer, BoundingBox3D, LabelLUT
from tqdm import tqdm
import time
import numpy as np

cfg_file = "ml3d/configs/pointpillars_kitti.yml"
# cfg_file = "ml3d/configs/pointpillars_kitti_40ch.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.PointPillars(**cfg.model)
model_pre = ml3d.models.PointPillars(**cfg.model)
cfg.dataset['dataset_path'] = "/home/amlab/KITTI_point"
# cfg.dataset['dataset_path'] = "/home/amlab/vtd_kitti_format"
# cfg.dataset['dataset_path'] = "/home/amlab/os_test"
# cfg.dataset['dataset_path'] = "/home/amlab/hesai_test"

dataset = ml3d.datasets.KITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)
pipeline_pre = ml3d.pipelines.ObjectDetection(model_pre, dataset=dataset, device="gpu", **cfg.pipeline)

# download the weights.
ckpt_folder = "./logs/"
os.makedirs(ckpt_folder, exist_ok=True)
ckpt_path = ckpt_folder + "pointpillars_vtd_20220217.pth"

ckpt_pretrain = ckpt_folder + "pointpillars_kitti_202012221652utc.pth"
# pointpillar_url = "https://storage.googleapis.com/open3d-releases/model-zoo/pointpillars_kitti_202012221652utc.pth"
# if not os.path.exists(ckpt_path):
#     cmd = "wget {} -O {}".format(pointpillar_url, ckpt_path)
#     os.system(cmd)

# load the parameters.
pipeline.load_ckpt(ckpt_path=ckpt_path)
pipeline_pre.load_ckpt(ckpt_path=ckpt_pretrain)

test_split = dataset.get_split("training")
# print(test_split.__dict__)


vis = Visualizer()
lut = LabelLUT()
for val in sorted(dataset.label_to_names.keys()):
    lut.add_label(val, val)

# run inference on a single example.
# returns dict with 'predict_labels' and 'predict_scores'.
boxes = []
data_list = []
for idx in tqdm(range(20)):
    data = test_split.get_data(idx)
    if np.max(np.unique(data['point'][:,3])) > 1:
        data['point'][:,3] = np.round(data['point'][:,3]/255.0 - 0.01,3)
    
    true_val = data['bounding_boxes']

    result = pipeline.run_inference(data)[0]
    print(result[0].__dict__)
    pred = {
    "name": 'KITTI' + '_' + str(idx),
    'points': data['point'],
    'bounding_boxes': result
    }
    data_list.append(pred)

    result_pre = pipeline_pre.run_inference(data)[0]

    pred_pre = {
    "name": 'KITTI_pre' + '_' + str(idx),
    'points': data['point'],
    'bounding_boxes': result_pre
    }
    data_list.append(pred_pre)

    true = {
        "name": 'KITTI_true' + '_' + str(idx),
        'points': data['point'],
        'bounding_boxes': true_val
    }
    data_list.append(true)
    

vis.visualize(data_list, lut, bounding_boxes=None)

