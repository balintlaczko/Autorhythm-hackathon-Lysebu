# %%
from sonification.utils import matrix
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
# img_path = "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate07-2pos--s14743--UVRAG--W0045--P001--T00001--Z001--C01.ome.jpg"
# img_path = "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate07-2pos-final--s22526--ATG14--W0017--P002--T00001--Z001--C01.ome.jpg"
img_path = "/Users/balintl/Desktop/AUTORYTHM/HACKATHON_p62/2pos_plates_Autoph_p62_jpeg/composite_Simonsen-Autoph-plate01-batch1-plate05-2pos-final--s14439--TSG101--W0061--P001--T00001--Z001--C01.ome.jpg"
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
kernel_size = 20
stride = 10
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
std_50percentile = np.percentile(patches2std['std'], 40)
empty_mask = np.zeros_like(gray)
for _, row in patches2std.iterrows():
    if row['std'] < std_50percentile:
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
print(keypoints_DoH[0].pt, keypoints_DoH[0].size)
# %%
# mask out the detected blobs
for keypoint in keypoints_DoH:
    i, j = int(keypoint.pt[1]), int(keypoint.pt[0])
    size = int(keypoint.size)
    empty_mask[i:i+size, j:j+size] = 0
masked_img = img * (1 - empty_mask[:, :, np.newaxis])
matrix.view(masked_img)
# %%
