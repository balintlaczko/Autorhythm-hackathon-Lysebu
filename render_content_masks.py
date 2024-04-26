from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import argparse
import numpy as np
import cv2
import pandas as pd
from sonification.utils.matrix import floodfill_from_point, matrix2binary
# from sonification.utils import matrix


def get_bg_mask(
        img: np.ndarray,
        patch_size: int = 30,
        stride: int = 5,
        threshold: int = 10,
        std_percentile: float = 35.0,
        blobs_min_area: int = 1,
        blobs_max_area: int = 2000,
        invert: bool = True) -> np.ndarray:
    # apply median filter to the image
    median_kernel_size = int(patch_size // 2 * 2 + 1)
    img = cv2.medianBlur(img, median_kernel_size)
    red = img[:, :, -1]
    red_norm = (red - red.min()) / (red.max() - red.min())
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # apply blur to remove noise
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # apply median filter to remove noise
    # median_kernel_size = int(patch_size // 2 * 2 + 1)
    # gray = cv2.medianBlur(gray, median_kernel_size)
    # threshold bg image noise
    # threshold = np.percentile(gray, threshold)
    # gray_threshold_mask = np.where(gray < threshold, 1, 0).astype(np.uint8)

    # loop through the image in kernels and check standard deviation of the pixel values in the patch
    # if the standard deviation is low, the patch is considered empty
    columns = ['i', 'j', 'std', 'red_max']
    patches2std = pd.DataFrame(columns=columns)
    kernel_size = patch_size
    # stride = patch_size // 4
    for i in range(0, gray.shape[0] - kernel_size + 1, stride):
        for j in range(0, gray.shape[1] - kernel_size + 1, stride):
            # patch = gray[i:i+kernel_size, j:j+kernel_size]
            patch = img[i:i+kernel_size, j:j+kernel_size]
            patch_red_norm = red_norm[i:i+kernel_size, j:j+kernel_size]
            # # apply gaussian blur to the patch
            # patch = cv2.GaussianBlur(
            #     patch, (5, 5), 0)
            # apply median filter to the patch
            # patch = cv2.medianBlur(patch, patch_size // 2 + 1)
            patch_row = {
                'i': i,
                'j': j,
                # standard deviation of the red channel
                'std': patch[:, :, -1].std(),
                'red_max': patch_red_norm.max(),
            }
            # add row to dataframe
            if patches2std.empty:
                patches2std = pd.DataFrame(patch_row, index=[0])
            else:
                patches2std = pd.concat(
                    [patches2std, pd.DataFrame(patch_row, index=[0])], ignore_index=True)

    # create a mask of patches with low standard deviation
    std_percentile = np.percentile(patches2std['std'], std_percentile)
    mask = np.zeros_like(gray)
    for _, row in patches2std.iterrows():
        if row['std'] < std_percentile and row['red_max'] < 0.05:
            i, j = int(row['i']), int(row['j'])
            mask[i:i+kernel_size, j:j+kernel_size] = 1

    # erode the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(
        mask.astype(np.uint8), kernel, iterations=4)
    # dilate the mask
    mask = cv2.dilate(
        mask.astype(np.uint8), kernel, iterations=4)

    # perform blob detection on the filtered images
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = blobs_min_area
    params.maxArea = blobs_max_area
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints_DoH = detector.detect(mask * 255)

    # make a floodfill mask from each point
    all_floodfills = np.zeros_like(mask)
    for point in keypoints_DoH:
        x, y = int(np.round(point.pt[0])), int(np.round(point.pt[1]))
        floodfill_mask = floodfill_from_point(mask, [x, y]) - mask
        if np.sum(floodfill_mask) < 0.3 * np.sum(mask):
            all_floodfills += floodfill_mask.astype(np.uint8)
    mask_filtered = mask + all_floodfills

    if invert:
        mask_filtered = 1 - mask_filtered

    # mask_filtered = np.where(
    #     mask_filtered + gray_threshold_mask > 0, 1, 0).astype(np.uint8)

    return matrix2binary(mask_filtered)


def process_image(img_file, img_folder, masks_folder, combined_folder, args):
    img_path = os.path.join(img_folder, img_file)
    img = cv2.imread(img_path)
    mask = get_bg_mask(
        img,
        patch_size=args.patch_size,
        stride=args.stride,
        threshold=args.threshold,
        std_percentile=args.std_percentile,
        blobs_min_area=args.blobs_min_area,
        blobs_max_area=args.blobs_max_area,
        invert=args.invert == 1)
    # write mask to file
    mask_path = os.path.join(masks_folder, img_file)
    cv2.imwrite(mask_path, mask)
    # write mask applied to image
    masked_img = img * mask[:, :, np.newaxis]
    masked_img_path = os.path.join(combined_folder, img_file)
    cv2.imwrite(masked_img_path, masked_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_folder", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--std_percentile", type=float, default=35.0)
    parser.add_argument("--blobs_min_area", type=int, default=1)
    parser.add_argument("--blobs_max_area", type=int, default=2000)
    parser.add_argument("--invert", type=int, default=1)
    args = parser.parse_args()
    img_folder = args.img_folder
    img_files = os.listdir(img_folder)
    img_files = [f for f in img_files if f.endswith('.jpg')]
    # exclude files that have "final" in their name
    img_files = [f for f in img_files if 'final' not in f]
    masks_folder = os.path.join(os.path.dirname(img_folder), 'masks')
    os.makedirs(masks_folder, exist_ok=True)
    combined_folder = os.path.join(os.path.dirname(img_folder), 'combined')
    os.makedirs(combined_folder, exist_ok=True)

    executor = ProcessPoolExecutor()
    jobs = [executor.submit(process_image, img_file, img_folder, masks_folder, combined_folder, args)
            for img_file in img_files]
    results = []
    for job in tqdm(as_completed(jobs), total=len(jobs)):
        results.append(job.result())

    print("Finished rendering masks.")
