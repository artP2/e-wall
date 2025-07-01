#!/bin/env python3

import imageio.v3 as iio

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

class Imshower:
    i = 0
    def __init__(self, axis) -> None:
        self.axis = axis
        self.i = 0

    def __call__(self, img, title=None) -> None:
        if (img.dtype is bool):
            self.axis[self.i].imshow(img)
        else:
            self.axis[self.i].imshow(img, cmap='gray')
        if title:
            self.axis[self.i].set_title(title)
        self.i += 1


def main(img_path):
    # load image
    try:
        img = iio.imread(img_path)
    except FileNotFoundError:
        print(f"Error: file not found - {img_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # downscale image
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

    _, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,10),sharex=True, sharey=True)
    axs= axs.flatten()

    imshow = Imshower(axs)
    imshow(img, "Original")

    # preprocessing
    img_blur = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=0)
    img_lab = cv2.cvtColor(img_blur, cv2.COLOR_RGB2LAB)

    # superpixel segmentation using SLIC
    slic = cv2.ximgproc.createSuperpixelSLIC(img_lab, algorithm=cv2.ximgproc.SLIC)
    slic.iterate(10)
    slic.enforceLabelConnectivity(min_element_size=25)

    # superpixel boundaries
    slic_mask = slic.getLabelContourMask(thick_line=False)
    img_slic = img.copy()
    img_slic[slic_mask == 255] = [0,0,0]
    imshow(img_slic, "Superpixels")

    labels = slic.getLabels()

    img_median = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    img_median[:, :, 0] *= 0.1
    img_median[:, :, 1] *= 1.0
    img_median[:, :, 2] *= 1.0

    for i in range(slic.getNumberOfSuperpixels()):
        mask = labels == i
        img_median[mask] = img_median[mask].mean(axis=0)

    Z = img_median.reshape((-1,3))
    Z = Z.astype(np.float32)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    _, label,center= cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    center = center.astype(np.uint8)
    label = label.flatten()
    res = center[label]
    res2 = res.reshape((img.shape))
    imshow(cv2.cvtColor(res2, cv2.COLOR_LAB2RGB), "K-means")

    uniq, counts = np.unique(label, return_counts=True)
    sort_indices = np.argsort(-counts)
    bots = uniq[sort_indices[K//2:]]
    
    mask = np.zeros(center.shape[0])
    mask[bots] = 255
    mask = mask[label].reshape(img.shape[:2])
    mask = mask.astype(np.uint8)
    mask = cv2.GaussianBlur(mask, ksize=(3,3), sigmaX=0)
    imshow(mask, "Binary Mask")

    # apply morphological operations
    kernel = np.ones((5,5), np.uint8)
    
    # closing - dilation followed by erosion
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # opening - erosion followed by dilation
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    
    imshow(mask_opened, "Binary Mask After Morpho")

    # pixel connectivity
    connectivity = 4; 
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_opened, 
        connectivity, 
        ltype=cv2.CV_32S
    )

    labels_vis = cv2.applyColorMap(
        (labels * (255 / num_labels)).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    labels_vis[mask_opened == 0] = 0  # black background
    imshow(labels_vis, f"Connected Components ({connectivity}-conn)\nTotal: {num_labels-1}")

    # bounding boxes
    min_blob_size = 350
    img_with_boxes = img.copy()
    blob_count = 0

    for i in range(1, num_labels):  # skip background (label 0)
        x, y, w, h, area = stats[i]
        if area >= min_blob_size:
            blob_count += 1
            cv2.rectangle(img_with_boxes, (x,y), (x+w,y+h), (0,255,0), 2)

    imshow(img_with_boxes, f"Bounding Boxes (>{min_blob_size}px)\nBlobs: {blob_count}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if  len(sys.argv) != 2:
        print(f"Use: python {sys.argv[0]} <path/to/img>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    main(img_path)
