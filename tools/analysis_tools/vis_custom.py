import json
import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R


# ===================== 读取相机参数 =====================
def load_camera_params(path):
    with open(path, "r") as f:
        data = json.load(f)

    cams = ["CAM_FRONT", "CAM_RIGHT", "CAM_BACK", "CAM_LEFT"]

    extrinsic_keys = [
        "top_center_lidar-to-front_camera-extrinsic",
        "top_center_lidar-to-right_camera-extrinsic",
        "top_center_lidar-to-back_camera-extrinsic",
        "top_center_lidar-to-left_camera-extrinsic",
    ]

    intrinsic_keys = [
        "front_camera-intrinsic",
        "right_camera-intrinsic",
        "back_camera-intrinsic",
        "left_camera-intrinsic",
    ]

    extrinsics, intrinsics = [], []

    for key in extrinsic_keys:
        extrinsics.append(np.array(data[key]["data"], dtype=np.float32))

    for key in intrinsic_keys:
        K = np.array(data[key]["data"], dtype=np.float32)
        if K.shape == (4,4):
            K = K[:3, :3]          # <<< 修复 4x4 intrinsic
        elif K.shape != (3,3):
            raise ValueError(f"Invalid intrinsic shape: {K.shape}")
        intrinsics.append(K)

    return cams, extrinsics, intrinsics


# ===================== 四元数 -> yaw =====================
def quat_to_yaw(q):
    r = R.from_quat([q[1], q[2], q[3], q[0]])  # [x,y,z,w]
    return r.as_euler('xyz', degrees=False)[2]


# ===================== 获取 8 个角点 =====================
def get_box_corners(translation, size, rotation_q):
    x, y, z = translation
    w, l, h = size
    yaw = quat_to_yaw(rotation_q)

    dx, dy, dz = l/2, w/2, h

    corners = np.array([
        [ dx,  dy, 0],
        [ dx, -dy, 0],
        [-dx, -dy, 0],
        [-dx,  dy, 0],
        [ dx,  dy, dz],
        [ dx, -dy, dz],
        [-dx, -dy, dz],
        [-dx,  dy, dz],
    ], dtype=np.float32)

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # 旋转 + 平移
    corners = (Rz @ corners.T).T + np.array([x, y, z], dtype=np.float32)
    return corners


# ===================== 投影到图像 =====================
def project_to_image(corners, K, T_l2c):
    # (8,3) → (8,4)
    corners_h = np.hstack([corners, np.ones((8, 1), dtype=np.float32)])

    # 直接雷达到相机，外参已给出
    corners_cam = (T_l2c @ corners_h.T).T

    # 丢弃相机后方的
    valid = corners_cam[:, 2] > 0.1
    corners_cam = corners_cam[valid]

    if len(corners_cam) == 0:
        return None

    pts = (K @ corners_cam[:, :3].T).T
    pts[:, 0] /= pts[:, 2]
    pts[:, 1] /= pts[:, 2]

    return pts[:, :2]


# ===================== 画 3D 框 =====================
def draw_box(img, pts, color=(0,0,255)):
    if pts is None or len(pts) != 8:
        return img

    pts = pts.astype(int)
    H, W = img.shape[:2]

    pts[:,0] = np.clip(pts[:,0], 0, W-1)
    pts[:,1] = np.clip(pts[:,1], 0, H-1)

    for i in range(4):
        cv2.line(img, pts[i], pts[(i+1)%4], color, 2)
        cv2.line(img, pts[i+4], pts[4+(i+1)%4], color, 2)
        cv2.line(img, pts[i], pts[i+4], color, 2)
    return img


# ===================== 主流程 =====================
def visualize(pred_json, cam_json, img_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    cams, extrinsics, intrinsics = load_camera_params(cam_json)

    with open(pred_json) as f:
        all_results = list(json.load(f)["results"].values())

    class_color = {
        "car": (0,0,255),
        "truck": (0,255,0),
        "bus": (255,0,0),
        "pedestrian": (0,255,255),
        "motorcycle": (255,0,255),
        "bicycle": (255,255,0),
    }

    for cam_idx, cam in enumerate(cams):
        cam_folder = os.path.join(img_dir, cam)
        imgs = sorted(os.listdir(cam_folder))

        T_l2c = extrinsics[cam_idx]
        K = intrinsics[cam_idx]

        for i, img_name in enumerate(imgs):
            if i >= len(all_results):
                break

            boxes = all_results[i]
            img = cv2.imread(os.path.join(cam_folder, img_name))

            for box in boxes:
                if box.get("detection_score", 0) < 0.25:
                    continue

                corners = get_box_corners(box["translation"], box["size"], box["rotation"])
                pts = project_to_image(corners, K, T_l2c)

                color = class_color.get(box["detection_name"], (255,255,255))
                img = draw_box(img, pts, color)

            cv2.imwrite(os.path.join(save_dir, f"{cam}_{img_name}"), img)
            print("Saved:", cam, img_name)


# ===================== 运行 =====================
if __name__ == "__main__":
    visualize(
        pred_json="/opt/data/private/codeN3/ToC3D/test/stream_petr_light_eva_flexivit_l/Thu_Dec__4_21_28_28_2025/pts_bbox/results_nusc.json",
        cam_json="/opt/data/private/codeN1/mmdetection3d/data/fourcam/intrinsic&lidar2camera_extrinsic.json",
        img_dir="/opt/data/private/codeN1/mmdetection3d/data/fourcam/",
        save_dir="/opt/data/private/codeN3/ToC3D/vis_out/"
    )


# import json
# import numpy as np
# import cv2
# import os
# from scipy.spatial.transform import Rotation as R
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon


# # ---------- 四元数转 yaw ----------
# def quat_to_yaw(q):
#     r = R.from_quat([q[1], q[2], q[3], q[0]])  # [x,y,z,w]
#     euler = r.as_euler('xyz', degrees=False)
#     return euler[2]


# # ---------- 获取 BEV box (只用 XY 平面) ----------
# def get_bev_corners(translation, size, rotation_q):
#     x, y, z = translation
#     w, l, h = size
#     yaw = quat_to_yaw(rotation_q)

#     # 车辆在 XY 平面上的四个角
#     dx = l / 2
#     dy = w / 2

#     corners = np.array([
#         [ dx,  dy],
#         [ dx, -dy],
#         [-dx, -dy],
#         [-dx,  dy],
#     ], dtype=np.float32)

#     # 旋转
#     c, s = np.cos(yaw), np.sin(yaw)
#     Rz = np.array([[c, -s], [s, c]])

#     corners = (Rz @ corners.T).T
#     corners[:, 0] += x
#     corners[:, 1] += y

#     return corners


# # ---------- 颜色表 ----------
# color_map = {
#     "car": "red",
#     "truck": "green",
#     "bus": "blue",
#     "pedestrian": "orange",
#     "motorcycle": "purple",
#     "bicycle": "cyan",
#     "default": "white"
# }


# # ---------- 可视化 BEV ----------
# def visualize_global_bev(pred_json, save_dir):
#     os.makedirs(save_dir, exist_ok=True)

#     with open(pred_json, "r") as f:
#         data = json.load(f)

#     results = list(data["results"].values())  # 顺序保证一致

#     for frame_idx, boxes in enumerate(results):
#         # ---- 创建 BEV 图 ----
#         fig, ax = plt.subplots(figsize=(10,10))
#         ax.set_title(f"Frame {frame_idx} - Global BEV")
#         ax.set_aspect("equal")

#         xs, ys = [], []

#         # 先收集所有 box 的中心用于定范围
#         for box in boxes:
#             t = box["translation"]
#             xs.append(t[0])
#             ys.append(t[1])

#         # 设置自适应视野
#         if len(xs) > 0:
#             ax.set_xlim(min(xs)-20, max(xs)+20)
#             ax.set_ylim(min(ys)-20, max(ys)+20)

#         # 画所有 box
#         for box in boxes:
#             translation = box["translation"]
#             size = box["size"]
#             rotation = box["rotation"]
#             category = box.get("detection_name", "default")
#             score = box.get("detection_score", 0)

#             if score < 0.2:
#                 continue

#             color = color_map.get(category, color_map["default"])

#             corners = get_bev_corners(translation, size, rotation)
#             poly = Polygon(corners, closed=True, edgecolor=color, fill=False, linewidth=2)
#             ax.add_patch(poly)

#             # 标注类别
#             ax.text(translation[0], translation[1], category, color=color)

#         save_path = os.path.join(save_dir, f"bev_{frame_idx:05d}.png")
#         plt.savefig(save_path, dpi=200)
#         plt.close()

#         print(f"Saved {save_path}")


# # ---------- 运行 ----------
# if __name__ == "__main__":
#     visualize_global_bev(
#         pred_json="/opt/data/private/codeN3/ToC3D/test/stream_petr_light_eva_flexivit_l/Thu_Dec__4_21_28_28_2025/pts_bbox/results_nusc.json",
#         save_dir="/opt/data/private/codeN3/ToC3D/vis_out/"
#     )

# import open3d as o3d
# import numpy as np


# def load_pcd(path):
#     pcd = o3d.io.read_point_cloud(path)
#     return pcd


# def create_3d_box(center, size, yaw, color=[1, 0, 0]):
#     """
#     center: [x, y, z]
#     size: [dx, dy, dz]
#     yaw: rotation around z-axis (in radians)
#     color: box color
#     """
#     dx, dy, dz = size
#     # 8 corner points
#     box = o3d.geometry.OrientedBoundingBox(
#         center=center,
#         R=o3d.geometry.get_rotation_matrix_from_xyz([0, 0, yaw]),
#         extent=[dx, dy, dz]
#     )

#     # convert to LineSet for visualization
#     lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
#     lineset.paint_uniform_color(color)
#     return lineset


# def visualize_scene(pcd_path, box_list):
#     """
#     box_list: list of dict
#         {
#             "translation": [x, y, z],
#             "size": [dx, dy, dz],
#             "yaw": angle_rad
#         }
#     """
#     pcd = load_pcd(pcd_path)

#     objs = [pcd]

#     for box in box_list:
#         translation = np.array(box["translation"])
#         size = np.array(box["size"])
#         yaw = box["yaw"]
#         color = box.get("color", [1, 0, 0])  # red default

#         box_obj = create_3d_box(translation, size, yaw, color=color)
#         objs.append(box_obj)

#     o3d.visualization.draw_geometries(objs)


# # 示例：你的 box（注意：yaw 使用弧度制）
# box_list = [
#     {
#         "translation": [583.4, 1453.45, -1.56],
#         "size": [4.0, 1.8, 1.6],   # dx, dy, dz
#         "yaw": 0.5,               # radians
#         "color": [0, 1, 0]        # green
#     }
# ]

# visualize_scene(
#    "/opt/data/private/codeN1/mmdetection3d/data/fourcam/LIDAR_TOP/LIDAR_TOP_1764138197_698720455.pcd",
#     box_list
# )

