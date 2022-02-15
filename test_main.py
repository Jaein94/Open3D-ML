import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from open3d.ml.vis import Visualizer, BoundingBox3D, LabelLUT
from tqdm import tqdm

cfg_file = "ml3d/configs/pointpillars_kitti.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.PointPillars(**cfg.model)
# cfg.dataset['dataset_path'] = "/home/amlab/KITTI_point"
# cfg.dataset['dataset_path'] = "/home/amlab/vtd_kitti_format"
cfg.dataset['dataset_path'] = "/home/amlab/os_test"

dataset = ml3d.datasets.KITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)

# download the weights.
ckpt_folder = "./logs/"
os.makedirs(ckpt_folder, exist_ok=True)
ckpt_path = ckpt_folder + "pointpillars_kitti_202012221652utc.pth"
pointpillar_url = "https://storage.googleapis.com/open3d-releases/model-zoo/pointpillars_kitti_202012221652utc.pth"
if not os.path.exists(ckpt_path):
    cmd = "wget {} -O {}".format(pointpillar_url, ckpt_path)
    os.system(cmd)

# load the parameters.
pipeline.load_ckpt(ckpt_path=ckpt_path)

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
for idx in tqdm(range(5)):
    data = test_split.get_data(idx)

    result = pipeline.run_inference(data)[0]
    print(result[0].yaw)
    print(result[0].yaw)
    data_list.append({
    "name": 'KITTI' + '_' + str(idx),
    'points': data['point'],
    'bounding_boxes': result
    })

vis.visualize(data_list, lut, bounding_boxes=None)

