from typing import List
import numpy as np
import os.path as osp
import torch
import mmcv
import cv2
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-trainval', dataroot='/opt/data/private/codeN1/mmdetection3d/data/nuscenes/', verbose=False)
filename_to_token = {sd['filename']: sd['sample_token'] for sd in nusc.sample_data}

def pad_image_to_match_mask(image, mask_shape, pad_color=(0, 0, 0), mode='bottom_right'):
    """
    Pad an image to match the given mask shape.
    
    Args:
        image (np.ndarray): Input image of shape (H, W, C)
        mask_shape (tuple): Target mask shape (H_mask, W_mask, _)
        pad_color (tuple): RGB color for padding (default: black)
        mode (str): 'center', 'bottom_right', or 'symmetric' padding mode.
    
    Returns:
        np.ndarray: Padded image with shape matching mask_shape
    """
    h_img, w_img, _ = image.shape
    h_mask, w_mask, _ = mask_shape

    pad_h = max(h_mask - h_img, 0)
    pad_w = max(w_mask - w_img, 0)

    if mode == 'bottom_right':
        # 在下方和右侧补齐
        top, bottom = 0, pad_h
        left, right = 0, pad_w
    elif mode == 'center':
        # 居中补齐
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
    elif mode == 'symmetric':
        # 上下左右均匀补齐
        top = bottom = pad_h // 2
        left = right = pad_w // 2
    else:
        raise ValueError("mode must be one of ['bottom_right', 'center', 'symmetric']")

    padded_img = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
    )
    return padded_img

def token_selection_vis(
    input_imgs: torch.Tensor,
    masks: List[torch.Tensor],
    keep_idxes: List[torch.Tensor],
    drop_idxes: List[torch.Tensor],
    img_norm_cfg: dict,
    output_path: str,
    img_metas: List[torch.Tensor],
    patch_size: int = 16,
    min_alpha: float = 0.3,
    max_alpha: float = 1.0
):
    '''visualization the token selection

    Args:
        input_imgs (torch.Tensor): with shape num_views x 3 x H x W
        masks (List[Tensor]): the token masks, 1 for important tokens, with shape num_views x H x W x 1
        img_norm_cfg (dict): the norm_cfg used to denorm the input_imgs
        output_path (str): the output_path of vis
    '''
    input_imgs = input_imgs.cpu().numpy()
    masks = [mask.cpu().numpy() for mask in masks]
    if keep_idxes is not None and drop_idxes is not None:
        keep_idxes = [idxes.cpu().numpy() for idxes in keep_idxes]
        drop_idxes = [idxes.cpu().numpy() for idxes in drop_idxes]
    for view_id in range(input_imgs.shape[0]):
        img = input_imgs[view_id]  
        img = np.transpose(img, [1, 2, 0])  # H x W x 3

        # denorm the img
        if img_norm_cfg is not None:
            mean = img_norm_cfg['mean']
            std = img_norm_cfg['std']

            # import ipdb; ipdb.set_trace()
            img = mmcv.imdenormalize(img, mean=mean, std=std, to_bgr=False)  # np.float32, 0~255.0

        alpha = np.ones([*img.shape[:2], 1], dtype=img.dtype) * 255
        for layer_id, layer_mask in enumerate(masks):
            layer_mask = layer_mask[view_id]  # Hp x Wp x 1

            # assert img.shape[0] // layer_mask.shape[0] == patch_size
            # assert img.shape[1] // layer_mask.shape[1] == patch_size

            #################################################################
            # # 800
            # if layer_mask.shape[0] + layer_mask.shape[1] == 63:
            #     patch_size = 18
            # elif layer_mask.shape[0] + layer_mask.shape[1] == 56:
            #     patch_size = 20

            if layer_mask.shape[0] + layer_mask.shape[1] == 112:
                patch_size = 20
            elif layer_mask.shape[0] + layer_mask.shape[1] == 103:
                patch_size = 22

            global filename_to_token
            filename = img_metas[0]['filename'][0][16:]
            token = filename_to_token[filename]

            #################################################################

            pixel_mask = np.repeat(layer_mask, patch_size, axis=1)
            pixel_mask = np.repeat(pixel_mask, patch_size, axis=0)  # H x W x 1
            
            pixel_mask = pixel_mask * (max_alpha - min_alpha) + min_alpha

            # print(patch_size, img.shape, alpha.shape, pixel_mask.shape, 111)
            # pad img
            img = pad_image_to_match_mask(img, pixel_mask.shape)
            alpha = np.ones([*img.shape[:2], 1], dtype=img.dtype) * 255

            out_img = np.concatenate([img, alpha * pixel_mask], axis=-1)
            out_filename = osp.join(output_path, f'view{view_id}_layer{layer_id}_{token}_ps{patch_size}.png')
            ori_filename = osp.join(output_path, f'view{view_id}_layer{layer_id}_{token}_ps{patch_size}_ori.png')
            mmcv.imwrite(out_img, out_filename)
            mmcv.imwrite(img, ori_filename)

        if keep_idxes is not None and drop_idxes is not None:
            num_patch_h = img.shape[0] // patch_size
            num_patch_w = img.shape[1] // patch_size
            alpha = np.ones([*img.shape[:2], 1], dtype=img.dtype) * 255
            for layer_id, layer_keep_idx in enumerate(keep_idxes):
                keep_idx = layer_keep_idx[view_id]
                assert keep_idx.max() < num_patch_h * num_patch_w

                keep_mask = np.zeros([num_patch_h * num_patch_w], dtype=img.dtype)
                keep_mask[keep_idx] = 1
                keep_mask = keep_mask.reshape(num_patch_h, num_patch_w, 1)

                keep_mask = np.repeat(keep_mask, patch_size, axis=1)
                keep_mask = np.repeat(keep_mask, patch_size, axis=0)

                keep_mask = keep_mask * (max_alpha - min_alpha) + min_alpha

                out_img = np.concatenate([img, alpha * keep_mask], axis=-1)
                out_filename = osp.join(output_path, f'view{view_id}_layer{layer_id}_keepidx.png')
                mmcv.imwrite(out_img, out_filename)