import os
import tqdm
import json
from visual_nuscenes import NuScenes
use_gt = False
out_dir = './result_vis_bl/'
# result_json = "work_dirs/pp-nus/results_eval/pts_bbox/results_nusc"
# result_json="/opt/data/private/codeN3/ToC3D/work_dir_base_model/SEPatch3D-fast_320*800_61.08_52.37/Tue_Aug__5_17_36_41_2025/pts_bbox/results_nusc" # ours
result_json="/opt/data/private/codeN3/ToC3D/val/work_dirs/stream_petr_eva_vit_l/Sun_Dec_22_10_14_17_2024/pts_bbox/results_nusc" # baseline
# result_json="/opt/data/private/codeN3/ToC3D/val/work_dirs/stream_petr_light_eva_flexivit_l_1600/Wed_Jul_23_09_44_09_2025/pts_bbox/results_nusc" # toc3d
# result_json="/opt/data/private/codeN3/ToC3D/work_dirs_abla/SPS_ChangeTo_FixedPatch18_60.35_51.80/SPS_ChangeTo_FixedPatch18/Wed_Aug__6_09_56_22_2025/pts_bbox/results_nusc"
dataroot='/opt/data/private/codeN1/mmdetection3d/data/nuscenes/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if use_gt:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False, pred = False, annotations = "sample_annotation")
else:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False, pred = True, annotations = result_json, score_thr=0.2)

with open('{}.json'.format(result_json)) as f:
    table = json.load(f)
tokens = list(table['results'].keys())

for token in tqdm.tqdm(tokens):
    # if token == "66bd5bc1ef584d849b87f08b78e3beef": # 448
    # if token == "1808fbc3531c415fb77531c05b2547ae": # 1144
    # if token == "a9b8176bf0b546a4bc46afc631979805": # 1205
    if token == "4094ae4656fb4b8fb1906192b24a34cb": # 24 
        if use_gt:
            nusc.render_sample(token, out_path = "./result_vis_bl/"+token+"_gt.png", verbose=False)
        else:
            nusc.render_sample(token, out_path = "./result_vis_bl/"+token+"_pred.png", verbose=False)


# import os
# import json
# import tqdm
# import matplotlib.pyplot as plt
# from visual_nuscenes import NuScenes

# def render_sample(self,
#                   token,
#                   out_path=None,
#                   verbose=True,
#                   ax=None,
#                   box_color=None,
#                   hold_on=False):
#     """
#     Extended render_sample:
#     - supports external ax
#     - supports custom box color
#     - supports overlay rendering (hold_on=True)
#     """

#     import matplotlib.pyplot as plt
#     from nuscenes.utils.geometry_utils import view_points
#     import numpy as np
#     import cv2
#     import os

#     sample = self.get('sample', token)

#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=(16, 9))
#     else:
#         fig = ax.figure

#     cam_token = sample['data']['CAM_FRONT']
#     cam = self.get('sample_data', cam_token)

#     img_path = os.path.join(self.dataroot, cam['file_name'])
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     ax.imshow(img)
#     ax.axis('off')

#     cs = self.get('calibrated_sensor', cam['calibrated_sensor_token'])
#     cam_intrinsic = np.array(cs['camera_intrinsic'])

#     _, boxes, _ = self.get_sample_data(
#         cam_token,
#         box_vis_level=1,
#         use_flat_vehicle_coordinates=False
#     )

#     for box in boxes:
#         corners = box.corners()
#         corners_2d = view_points(corners, cam_intrinsic, normalize=True)[:2, :]

#         edges = [
#             (0, 1), (1, 2), (2, 3), (3, 0),
#             (4, 5), (5, 6), (6, 7), (7, 4),
#             (0, 4), (1, 5), (2, 6), (3, 7)
#         ]

#         color = box_color if box_color is not None else (1, 0, 0)

#         for i, j in edges:
#             ax.plot(
#                 corners_2d[0, [i, j]],
#                 corners_2d[1, [i, j]],
#                 color=color,
#                 linewidth=2
#             )

#     if out_path is not None:
#         plt.savefig(out_path, dpi=200, bbox_inches='tight')

#     if not hold_on:
#         plt.close(fig)


# out_dir = './result_vis_bl/'
# dataroot = '/opt/data/private/codeN1/mmdetection3d/data/nuscenes/'
# result_json = "/opt/data/private/codeN3/ToC3D/work_dir_base_model/SEPatch3D-fast_320*800_61.08_52.37/Tue_Aug__5_17_36_41_2025/pts_bbox/results_nusc"

# os.makedirs(out_dir, exist_ok=True)

# # ================== GT ==================
# nusc_gt = NuScenes(
#     version='v1.0-trainval',
#     dataroot=dataroot,
#     verbose=False,
#     pred=False,
#     annotations="sample_annotation"
# )

# # ================== Pred ==================
# nusc_pred = NuScenes(
#     version='v1.0-trainval',
#     dataroot=dataroot,
#     verbose=False,
#     pred=True,
#     annotations=result_json,
#     score_thr=0.25
# )

# with open(f"{result_json}.json") as f:
#     table = json.load(f)

# tokens = list(table['results'].keys())

# for token in tqdm.tqdm(tokens):
#     if token != "4094ae4656fb4b8fb1906192b24a34cb":
#         continue

#     fig, ax = plt.subplots(1, 1, figsize=(16, 9))

#     # ===== Pred：红色 =====
#     nusc_pred.render_sample(
#         token,
#         ax=ax,
#         box_color=(1, 0, 0),
#         hold_on=True,
#         verbose=False
#     )

#     # ===== GT：绿色 =====
#     nusc_gt.render_sample(
#         token,
#         ax=ax,
#         box_color=(0, 1, 0),
#         hold_on=True,
#         verbose=False
#     )

#     out_path = os.path.join(out_dir, f"{token}_gt_pred.png")
#     plt.savefig(out_path, dpi=200, bbox_inches='tight')
#     plt.close(fig)

#     print(f"Saved: {out_path}")



