import cv2
import numpy as np


import numpy as np
import cv2


def classify_masks(masks, threshold=0.1, dilate_px=15):
    """
    masks: list of numpy binary masks (0/1 or True/False)
    threshold: fraction of image area for separating small vs large masks
    dilate_px: how much context to expand before grouping (higher = more grouping)

    Returns:
        small_hole_masks : list of merged masks for small regions
        large_hole_masks : list of merged masks for large regions
    """

    # --- Step 1: Dilate masks so nearby areas get grouped ---
    kernel = np.ones((dilate_px, dilate_px), np.uint8)
    grown_masks = [cv2.dilate(m.astype(np.uint8), kernel, iterations=1) for m in masks]

    # --- Step 2: Group masks that touch after dilation ---
    used = [False] * len(masks)
    groups = []

    for i in range(len(masks)):
        if used[i]:
            continue

        group = [i]
        used[i] = True

        for j in range(i + 1, len(masks)):
            if used[j]:
                continue
            if np.any(grown_masks[i] & grown_masks[j]):  # touching or overlapping
                group.append(j)
                used[j] = True

        groups.append(group)

    # --- Step 3: Merge masks inside each group & classify by area ---
    H, W = masks[0].shape
    img_area = H * W

    small_hole_masks = []
    large_hole_masks = []

    for g in groups:
        merged = np.zeros_like(masks[0], dtype=np.uint8)
        for idx in g:
            merged |= masks[idx].astype(np.uint8)

        area_fraction = merged.sum() / img_area

        if area_fraction < threshold:
            small_hole_masks.append(merged)
        else:
            large_hole_masks.append(merged)

    return small_hole_masks, large_hole_masks


def merge_masks(mask_list):
    if len(mask_list) == 0:
        return None
    merged = np.zeros_like(mask_list[0], dtype=np.uint8)
    for m in mask_list:
        merged |= m.astype(np.uint8)
    return merged
