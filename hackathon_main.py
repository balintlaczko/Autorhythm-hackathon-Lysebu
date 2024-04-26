# %%
import os
from sonification.utils.matrix import floodfill_from_point, matrix2binary
from sonification.utils import matrix
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

# %%
# visualize image
img_path = "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate06-2pos--s30071--ATG16L1--W0069--P001--T00001--Z001--C01.ome.jpg"
# img_path = "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate06-2pos--s22526--ATG14--W0017--P002--T00001--Z001--C01.ome.jpg"
# img_path = "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate06-2pos-final--s30071--ATG16L1--W0069--P002--T00001--Z001--C01.ome.jpg"
# img_path = "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate07-2pos--s14743--UVRAG--W0045--P001--T00001--Z001--C01.ome.jpg"
# img_path = "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate07-2pos-final--s22526--ATG14--W0017--P002--T00001--Z001--C01.ome.jpg"
# read image
img = cv2.imread(img_path)
# display image
# matrix.view(img)

# %%
# img = matrix.stretch_contrast(
#     img, in_min=40, in_percentile=99, out_min=0, out_max=255)
# # display contrast-stretched image
# matrix.view(img)

# %%
# convert to grayscale
# img = np.round(img).astype(np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# display grayscale image
# matrix.view(gray)
# minimum and maximum pixel values
gray.min(), gray.max()

# %%
# simple thresholding by percentile
percentile = 99.7
thresh_val = np.percentile(gray, percentile)
gray_thresh = np.where(gray > thresh_val, gray, 0)
if gray_thresh.max() == 0:
    thresh_val = 245
    gray_thresh = np.where(gray > thresh_val, gray, 0)
# display thresholded image
matrix.view(gray_thresh)

# %%
gray_thresh_bin = np.where(gray > thresh_val, 1, 0)
# erode the mask
kernel = np.ones((5, 5), np.uint8)
gray_thresh_bin = cv2.erode(
    gray_thresh_bin.astype(np.uint8), kernel, iterations=4)
# dilate the mask
kernel = np.ones((5, 5), np.uint8)
gray_thresh_bin = cv2.dilate(
    gray_thresh_bin.astype(np.uint8), kernel, iterations=4)
# mask image with the inverted binary mask
img_masked = img * (1 - gray_thresh_bin[:, :, np.newaxis])
# display masked image
matrix.view(img_masked)

# %%
# instead, replace the masked region with the median rgb value
median_rgb = np.median(img, axis=(0, 1))
img_masked_median = np.where(
    gray_thresh_bin[:, :, np.newaxis], median_rgb, img)
# display masked image
matrix.view(img_masked_median)

# %%
images = [
    "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate06-2pos--s30071--ATG16L1--W0069--P001--T00001--Z001--C01.ome.jpg",
    "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate06-2pos--s22526--ATG14--W0017--P002--T00001--Z001--C01.ome.jpg",
    "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate06-2pos-final--s30071--ATG16L1--W0069--P002--T00001--Z001--C01.ome.jpg",
    "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate07-2pos--s14743--UVRAG--W0045--P001--T00001--Z001--C01.ome.jpg",
    "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate07-2pos-final--s22526--ATG14--W0017--P002--T00001--Z001--C01.ome.jpg"
]

for img_path in images:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    percentile = 99.7
    thresh_val = np.percentile(gray, percentile)
    gray_thresh = np.where(gray > thresh_val, gray, 0)
    if gray_thresh.max() == 0:
        thresh_val = 250
        gray_thresh = np.where(gray > thresh_val, gray, 0)
    gray_thresh_bin = np.where(gray > thresh_val, 1, 0)
    # erode the mask
    kernel = np.ones((5, 5), np.uint8)
    gray_thresh_bin = cv2.erode(
        gray_thresh_bin.astype(np.uint8), kernel, iterations=4)
    # dilate the mask
    kernel = np.ones((5, 5), np.uint8)
    gray_thresh_bin = cv2.dilate(
        gray_thresh_bin.astype(np.uint8), kernel, iterations=4)
    # mask image with the inverted binary mask
    img_masked = img * (1 - gray_thresh_bin[:, :, np.newaxis])
    # display masked image
    matrix.view(img_masked)

# %%
# try to detect empty areas in the image


# %%
# visualize image
# img_path = "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate06-2pos--s30071--ATG16L1--W0069--P001--T00001--Z001--C01.ome.jpg"
# img_path = "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate06-2pos--s22526--ATG14--W0017--P002--T00001--Z001--C01.ome.jpg"
# img_path = "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate06-2pos-final--s30071--ATG16L1--W0069--P002--T00001--Z001--C01.ome.jpg"
img_path = "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate07-2pos--s14743--UVRAG--W0045--P001--T00001--Z001--C01.ome.jpg"
# img_path = "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate07-2pos-final--s22526--ATG14--W0017--P002--T00001--Z001--C01.ome.jpg"
# img_path = "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate05-2pos-final--s14439--TSG101--W0061--P001--T00001--Z001--C01.ome.jpg"
# read image
img = cv2.imread(img_path)
# display image
matrix.view(img)

# %%
# convert to grayscale
# img = np.round(img).astype(np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# display grayscale image
# matrix.view(gray)
# minimum and maximum pixel values
gray.min(), gray.max()

# %%
# loop through the image in kernels and check standard deviation of the pixel values in the patch
# if the standard deviation is low, the patch is considered empty
# the kernel size is 5x5
columns = ['i', 'j', 'std']
patches2std = pd.DataFrame(columns=columns)
kernel_size = 30
stride = 15
for i in range(0, gray.shape[0] - kernel_size, stride):
    for j in range(0, gray.shape[1] - kernel_size, stride):
        patch = gray[i:i+kernel_size, j:j+kernel_size]
        patch_row = {
            'i': i,
            'j': j,
            'std': patch.std()
        }
        # add row to dataframe
        if patches2std.empty:
            patches2std = pd.DataFrame(patch_row, index=[0])
        else:
            patches2std = pd.concat(
                [patches2std, pd.DataFrame(patch_row, index=[0])], ignore_index=True)

# %%
plt.plot(patches2std['std'])

# %%
# create a mask of patches with low standard deviation
std_percentile = np.percentile(patches2std['std'], 35)
empty_mask = np.zeros_like(gray)
for _, row in patches2std.iterrows():
    if row['std'] < std_percentile:
        i, j = int(row['i']), int(row['j'])
        empty_mask[i:i+kernel_size, j:j+kernel_size] = 1


# erode the mask
kernel = np.ones((5, 5), np.uint8)
empty_mask = cv2.erode(
    empty_mask.astype(np.uint8), kernel, iterations=4)
# dilate the mask
empty_mask = cv2.dilate(
    empty_mask.astype(np.uint8), kernel, iterations=4)
# display empty mask
# matrix.view(empty_mask * 255)

masked_img = img * (1 - empty_mask[:, :, np.newaxis])
matrix.view(masked_img)

# %%
# perform blob detection on the filtered images
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 1
params.maxArea = 2000
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
detector = cv2.SimpleBlobDetector_create(params)
keypoints_DoH = detector.detect(empty_mask * 255)

# draw the detected blobs on the original image
img_with_keypoints_DoH = cv2.drawKeypoints(masked_img, keypoints_DoH, np.array(
    []), (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# display the image with keypoints
matrix.view(img_with_keypoints_DoH)

# %%
# make a floodfill mask from each point
all_floodfills = np.zeros_like(empty_mask)
for point in keypoints_DoH:
    x, y = int(np.round(point.pt[0])), int(np.round(point.pt[1]))
    floodfill_mask = floodfill_from_point(empty_mask, [x, y]) - empty_mask
    if np.sum(floodfill_mask) < 0.3 * np.sum(empty_mask):
        all_floodfills += floodfill_mask.astype(np.uint8)
mask_filtered = empty_mask + all_floodfills

# %%
matrix.view(all_floodfills * 255)

# %%
masked_img = img * (1 - mask_filtered[:, :, np.newaxis])
matrix.view(masked_img)

# %%
# make it into function


def get_bg_mask(
        img: np.ndarray,
        patch_size: int = 30,
        std_percentile: float = 35.0,
        blobs_min_area: int = 1,
        blobs_max_area: int = 2000,
        invert: bool = True) -> np.ndarray:
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # loop through the image in kernels and check standard deviation of the pixel values in the patch
    # if the standard deviation is low, the patch is considered empty
    columns = ['i', 'j', 'std']
    patches2std = pd.DataFrame(columns=columns)
    kernel_size = patch_size
    stride = patch_size // 2
    for i in range(0, gray.shape[0] - kernel_size, stride):
        for j in range(0, gray.shape[1] - kernel_size, stride):
            patch = gray[i:i+kernel_size, j:j+kernel_size]
            patch_row = {
                'i': i,
                'j': j,
                'std': patch.std()
            }
            # add row to dataframe
            if patches2std.empty:
                patches2std = pd.DataFrame(patch_row, index=[0])
            else:
                patches2std = pd.concat(
                    [patches2std, pd.DataFrame(patch_row, index=[0])], ignore_index=True)

    # create a mask of patches with low standard deviation
    std_percentile = np.percentile(patches2std['std'], 35)
    mask = np.zeros_like(gray)
    for _, row in patches2std.iterrows():
        if row['std'] < std_percentile:
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

    return matrix2binary(mask_filtered)


# %%
# render all the masks for a folder of images
img_folder = "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg"
img_files = os.listdir(img_folder)
img_files = [f for f in img_files if f.endswith('.jpg')]
target_dir = os.path.join(os.path.dirname(img_folder), 'masks')
os.makedirs(target_dir, exist_ok=True)

for img_file in tqdm(img_files):
    img_path = os.path.join(img_folder, img_file)
    img = cv2.imread(img_path)
    mask = get_bg_mask(img)
    mask_path = os.path.join(target_dir, img_file)
    cv2.imwrite(mask_path, mask)
# %%
