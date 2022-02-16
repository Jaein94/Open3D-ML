import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from open3d.ml.vis import Visualizer, BoundingBox3D, LabelLUT
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

cfg_file = "ml3d/configs/pointpillars_kitti.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.PointPillars(**cfg.model)
# cfg.dataset['dataset_path'] = "/home/amlab/KITTI_point"
# cfg.dataset['dataset_path'] = "/home/amlab/vtd_kitti_format"
# cfg.dataset['dataset_path'] = "/home/amlab/os_test"


def get_dict_data(raw_pcd):

    data = {
        'point': raw_pcd,
        'full_point': None,
        'feat': None,
        'calib': None,
    }

    return data


data = np.fromfile('/home/amlab/os_test/training/velodyne/0.bin',dtype=np.float32)
raw_pcd = data.reshape(-1,4)
plt.plot(raw_pcd[:,0],raw_pcd[:,1],'yo',markersize=0.2)

data = get_dict_data(raw_pcd)


# dataset = ml3d.datasets.KITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.ObjectDetection(model, dataset=None, device="gpu", **cfg.pipeline)

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

# test_split = dataset.get_split("training")
# print(test_split.__dict__)


# data = test_split.get_data(idx)
# true_val = data['bounding_boxes']
result = pipeline.run_inference(data)[0]
print(result)

for obj in result:
    x = obj.center[0]
    y = obj.center[1]
    z = obj.center[2]
    h = obj.yaw
    cls = obj.label_class
    size = obj.size
    plt.plot(x,y,'ro',markersize=1)

plt.show()



