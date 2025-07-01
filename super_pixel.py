#!/bin/env python3

import imageio.v3 as iio

import cv2
import matplotlib.pyplot as plt
import numpy as np

class Imshower:
    i = 0
    def __init__(self, axis) -> None:
        self.axis = axis
        self.i = 0

    def __call__(self, img) -> None:
        if (img.dtype is bool):
            self.axis[self.i].imshow(img)
        else:
            self.axis[self.i].imshow(img, cmap='gray')
        self.i += 1


def main():
    img = iio.imread("./wall_low.jpg")
    # downscale image
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

    _, axs = plt.subplots(nrows=2, ncols=3, figsize=(15,5),sharex=True, sharey=True)
    axs= axs.flatten()

    imshow = Imshower(axs)
    imshow(img)

    img_blur = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=0)
    #imshow(img)

    img_lab = cv2.cvtColor(img_blur, cv2.COLOR_RGB2LAB)
    #imshow(img_lab)

    slic = cv2.ximgproc.createSuperpixelSLIC(img_lab, algorithm=cv2.ximgproc.SLIC)
    slic.iterate(10)
    slic.enforceLabelConnectivity(min_element_size=25)

    slic_mask = slic.getLabelContourMask(thick_line=False)
    img_slic = img.copy()
    img_slic[slic_mask == 255] = [0,0,0]
    imshow(img_slic)

    labels = slic.getLabels()

    #img_median = img.copy()
    img_median = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    img_median[:, :, 0] *= 0.1
    img_median[:, :, 1] *= 1.0
    img_median[:, :, 2] *= 1.0
    #img_median = img_median.astype(np.uint8)
    for i in range(slic.getNumberOfSuperpixels()):
        mask = labels == i
        img_median[mask] = img_median[mask].mean(axis=0)
    imshow(cv2.cvtColor(img_median.astype(np.uint8), cv2.COLOR_LAB2RGB))

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
    imshow(cv2.cvtColor(res2, cv2.COLOR_LAB2RGB))

    uniq, counts = np.unique(label, return_counts=True)
    sort_indices = np.argsort(-counts)
    bots = uniq[sort_indices[K//2:]]
    mask = np.zeros(center.shape[0])
    mask[bots] = 255
    mask = mask[label].reshape(img.shape[:2])
    mask = mask.astype(np.uint8)
    mask = cv2.GaussianBlur(mask, ksize=(3,3), sigmaX=0)
    imshow(mask)

    holds = cv2.bitwise_and(img, img, mask=mask)
    imshow(holds)


    #lsc = cv2.ximgproc.SuperpixelLSC()
    #lsc.iterate()
    #lsc = cv2.bitwise_not(lsc.getLabelContourMask())
    #img_lsc = cv2.bitwise_and(img, img, mask=slic_mask)
    #imshow(img_lsc)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
