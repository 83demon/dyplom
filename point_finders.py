import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from utils import Image_Helper


class MaximumColorFinder(Image_Helper):
    """Finds the point with the maximum color"""
    def find_max(self, channel):
        assert isinstance(channel,int)

        matrix = self.img[:,:,channel]
        coords = np.unravel_index(np.argmax(matrix), matrix.shape)  # return (y,x)
        coords = list(coords)
        return coords


class DensityFinder(Image_Helper):
    """Finds the point with the maximum color in an area of the biggest density(area). Due to limitations of
    recursion-depth it is advised to run on small images. It might be buggy."""
    def find_max(self, channel, verbose: bool = False):

        assert isinstance(channel,int)
        gray_img = self.img[:, :, channel]
        points = self._detect_shapes(gray_img, 255, verbose)
        coords = list(points)

        return coords

    def recur_find(self,arr, i, j, figs, fig, used):
        """Goes in particular direction. If there is no path, then it chooses next direction.
        Directions are prioritized."""
        m, n = arr.shape[0] - 1, arr.shape[1] - 1

        if arr[i, j] and (i, j) not in used:

            fig.append((i, j))
            used.append((i, j))

            if j + 1 <= n and (i, j + 1) not in fig and arr[i, j + 1]:
                self.recur_find(arr, i, j + 1, figs, fig, used)
            if i + 1 <= m and (i + 1, j) not in fig and arr[i + 1, j]:
                self.recur_find(arr, i + 1, j, figs, fig, used)
            if i - 1 >= 0 and (i - 1, j) not in fig and arr[i - 1, j]:
                self.recur_find(arr, i - 1, j, figs, fig, used)
            if j - 1 >= 0 and (i, j - 1) not in fig and arr[i, j - 1]:
                self.recur_find(arr, i, j - 1, figs, fig, used)

        return

    def find_max_area_fig(self,img: np.array, verbose: bool):
        """Finds a continious 2d figure with maximum area
        :::takes a 2d array of points with value 0 or 255
        :::returns list of indexes of points"""
        figures = []
        figure = []
        used = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                self.recur_find(img, i, j, figures, figure, used)
                if figure:
                    figures.append(tuple(figure))
                    figure.clear()
        if verbose:
            print(len(figures), [len(i) for i in figures], figures)
        return figures

    def apply_mask(self,figures, bin_img):
        """Selects only the biggest figure and denies others"""
        mask = [fig for fig in figures if (len(fig) == len(max(figures, key=len)) and len(fig) > 0)]
        mask, point = self.select_point_of_maxima(bin_img, mask)

        img = (np.zeros_like(bin_img)).astype(np.uint8)
        for index in mask:
            img[index] = 255
        return img, mask, point

    def select_point_of_maxima(self, bin_img,points):
        """Takes the list of figures of maximum area and finds a point with maximum color"""
        if len(points) == 0:
            point = np.unravel_index(np.argmax(bin_img), bin_img.shape)
            return (point), point
        elif len(points) == 1:  # we have one figure
            points = points[0]
            return points, points[np.argmax([bin_img[p] for p in points])]
        else:  # we have multiple figures of same area
            len_dict = {i: np.max([bin_img[p] for p in points[i]]) for i in range(len(points))}
            index = max(len_dict, key=len_dict.get)
            return points[index], points[index][np.argmax([bin_img[p] for p in points[index]])]
    def _detect_shapes(self, gray_img, threshold_max, verbose: bool):
        """Splits image into figures which matters "the most"(i.e. they have surpassed a decided limit). Resulting
        image is binary, i.e. pixel can black or white."""


        # threshold = cv.adaptiveThreshold(gray,threshold_max,cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        # cv2.THRESH_BINARY,9,13)        # it is decided not to use this one, because below is a simpler algorithm
        _, thresholded_image = cv.threshold(gray_img, 0, threshold_max,
                                    cv.THRESH_BINARY + cv.THRESH_TRIANGLE) # decides the threshold
        figures = self.find_max_area_fig(thresholded_image, verbose)
        img, mask, point = self.apply_mask(figures, thresholded_image)
        self.img[point] = (255, 0, 0)
        if verbose:
            fig, axes = plt.subplots(1, 4)
            axes[0].imshow(cv.cvtColor(thresholded_image, cv.COLOR_BGR2RGB))
            axes[1].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            axes[2].imshow(cv.cvtColor(gray_img, cv.COLOR_GRAY2RGB))
            axes[3].imshow(self.img)
            plt.show()
        return point

