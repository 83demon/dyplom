import subprocess,os
import cv2 as cv
import time
import numpy as np
path_folder = r"C:\Users\vo1ku\Programming\PycharmProjects&iPython\Dyplom\photos\\"

def capture_an_image():
    cam = cv.VideoCapture(0)
    result, image = cam.read()
    return image if result else None

def capture_img_series(interval,number=2):
    imgs = []
    for i in range(number):
        time.sleep(interval)
        imgs.append(capture_an_image())
    return imgs

def show_image(imgs,img_processed,channels):
    cv.imshow('processed',img_processed)
    cv.imshow('imgs',imgs)
    cv.imshow('channels',channels)
    cv.waitKey(0)
    cv.destroyAllWindows()

def calculate_image_mean(imgs):
    threshold = 26
    data_img = {index:img for index,img in enumerate(imgs)}
    data_scores = {index:None for index in range(len(imgs))}
    imgs_mean = np.zeros(imgs[0].shape,dtype=np.uint8)

    for img in imgs:
        imgs_mean[:,:,:] += img[:,:,:]
    imgs_mean //= len(imgs)

    for i in data_img.keys():
        data_scores[i] = np.sum(imgs_mean-data_img[i])

    least_img_index = min(data_scores,key=data_scores.get)
    least_img = data_img[least_img_index]
    print(data_scores)
    print(least_img)


    blue = np.abs(imgs_mean[:,:,0]-least_img[:,:,0])
    green = np.abs(imgs_mean[:,:,1]-least_img[:,:,1])
    red = np.abs(imgs_mean[:,:,2]-least_img[:,:,2])
    print(((imgs_mean-least_img)<=threshold).sum().sum()/(imgs[0].shape[0]*imgs[0].shape[1]))
    print(np.sum(np.abs((imgs_mean-least_img))))
    return np.abs(imgs_mean-least_img), red, green, blue


def main():
    interval = 1 # in ms
    im_number = 2
    imgs = capture_img_series(interval,im_number)
    full_img,*channels = calculate_image_mean(imgs)
    imgs = np.concatenate(imgs, axis=1)
    channels_concat = np.concatenate(channels,axis=1)
    show_image(imgs,full_img,channels_concat)
    print(channels[1])



main()
