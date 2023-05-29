from utils import Image_Helper
import numpy as np
import cv2
import time
import multiprocessing


class Blur(Image_Helper):

    def blur_with_gaussian(self,kernel_size=(7,7),kernel_sigma=(5,5)):
        return Image_Helper(img=cv2.GaussianBlur(self.img,kernel_size,*kernel_sigma))

    def blur_with_averaging(self, kernel_size=(7,7)):
        return Image_Helper(img=cv2.blur(self.img,kernel_size))

    def blur_with_median(self,kernel_size=5):
        return Image_Helper(img=cv2.medianBlur(self.img,kernel_size))

    def blur_with_bilateral(self,diameter=9,sigma_color=75,sigma_space=75):
        return Image_Helper(img=cv2.bilateralFilter(self.img,diameter,sigma_color,sigma_space))

class Focus(Image_Helper):

    @staticmethod
    def get_gaussian_filter(kernel_size, img, sigma=1, muu=0):
        x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                           np.linspace(-1, 1, kernel_size))
        dst = np.sqrt(x ** 2 + y ** 2)
        normal = 1 / (((2 * np.pi) ** 0.5) * sigma)
        gauss = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2))) * normal
        gauss = np.pad(gauss, [(0, img.shape[0] - gauss.shape[0]), (0, img.shape[1] - gauss.shape[1])], 'constant')
        return gauss

    @classmethod
    def fft_deblur_channel(cls, channel_img, kernel_size, kernel_sigma=5, factor='wiener'):
        gauss = cls.get_gaussian_filter(kernel_size, channel_img, kernel_sigma)
        img_fft = np.fft.fft2(channel_img)
        gauss_fft = np.fft.fft2(gauss)
        weiner_factor = 1 / (1 + (cls.signal_to_noise(channel_img) / np.abs(gauss_fft) ** 2))
        if factor != 'wiener':
            weiner_factor = factor
        recon = img_fft / gauss_fft
        recon *= weiner_factor
        recon = np.abs(np.fft.ifft2(recon))
        return recon

    def fft_deblur_rgb(self, kernel_size=7, kernel_sigma=5, factor='wiener'):
        """arguments = [(self.img[:,:,channel],kernel_size, kernel_sigma, factor) for channel in range(3)]
        pool = multiprocessing.Pool(processes=3)
        results = pool.starmap(self.fft_deblur_channel, arguments)
        pool.close()
        pool.join()"""
        results = []
        for channel in range(3):
            results.append(self.fft_deblur_channel(self.img[:,:,channel],kernel_size, kernel_sigma, factor))
        return Image_Helper(img=np.stack(results,axis=2).astype(np.uint8))

    @staticmethod
    def signal_to_noise(arr, axis=None, ddof=0):
        """
        ddof : int, optional
            Degrees of freedom correction for standard deviation. Default is 0.
        """
        mean = arr.mean(axis) if axis is not None else arr.mean()
        sd = arr.std(axis=axis, ddof=ddof)
        return mean / sd
