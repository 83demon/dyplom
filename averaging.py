import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from shift import shift


def crop_img(img,shape:tuple,bias:tuple=(0,0)):
    "Crops first part of image of shape 'shape'"
    return img[bias[0]:bias[0]+shape[0],bias[1]:bias[1]+shape[1],:]


def show_img(img,name=""):
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def read_img(filename):
    return cv.imread(filename)


def mean_vector(vec1,vec2):
    return np.sum((np.array(vec2)-np.array(vec1))/len(vec1),axis=0).tolist()


def find_max(img,channel,mode:str,verbose:bool):
    """finds position of a maximum of a specific channel, in every of shape[0]*matrices
        of shape (sqrt(shape[0]),sqrt(shape[0]))
        img has shape of (shape[0],shape[0])
        shape[0] must be a perfect square
        :::params: mode: {"simple","density"}"""
    points = []
    a = np.sqrt(img.shape[0]).astype(int)
    for i in range(a*a):
        matrix_rgb = img[(i//a)*a:((i//a)+1)*a,(i%a)*a:((i%a)+1)*a,:]
        matrix = matrix_rgb[:,:,channel]
        if mode=='density':
            point = detect_shapes(matrix,matrix_rgb,255,verbose)
            coords = list(point)
            coords[0] += (i//a)*a
            coords[1] += (i%a)*a
            points.append(coords)
        elif mode=='simple':
            coords = np.unravel_index(np.argmax(matrix),matrix.shape) #return (y,x)
            coords = list(coords)
            coords[0] += (i//a)*a
            coords[1] += (i%a)*a
            points.append(tuple(coords))

    return points


def recur_find(arr,i,j,figs,fig,used):
    m,n = arr.shape[0]-1,arr.shape[1]-1

    if arr[i,j] and (i,j) not in used:

        fig.append((i,j))
        used.append((i,j))

        if j+1<=n and (i,j+1) not in fig and arr[i,j+1]:
            recur_find(arr,i,j+1,figs,fig,used)
        if i+1<=m and (i+1,j) not in fig and arr[i+1,j]:
            recur_find(arr,i+1,j,figs,fig,used)
        if i-1>=0 and (i-1,j) not in fig and arr[i-1,j]:
            recur_find(arr,i-1,j,figs,fig,used)
        if j-1>=0 and (i,j-1) not in fig and arr[i,j-1]:
            recur_find(arr,i,j-1,figs,fig,used)


    return


def find_max_area_fig(array: np.array,verbose:bool):
    """Finds a continious 2d figure with maximum area
    :::takes an 2d array of points with value 0 or 255
    :::returns list of indexes of points"""
    figures = []
    figure = []
    used = []
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            recur_find(array,i,j,figures,figure,used)
            if figure:
                figures.append(tuple(figure))
                figure.clear()
    if verbose:
        print(len(figures),[len(i) for i in figures],figures)
    return figures


def apply_mask(figures,bin_img):
    """Selects only the biggest figure and denies others"""
    mask = [fig for fig in figures if (len(fig)==len(max(figures,key=len)) and len(fig)>0)]
    mask,point = select_point_of_maxima(bin_img,mask)

    img = (np.zeros_like(bin_img)).astype(np.uint8)
    for index in mask:
        img[index]=255
    return img,mask,point


def select_point_of_maxima(img,points):
    """Takes the list of figures of maximum area and finds a point with maximum color"""
    if len(points)==0:
        point = np.unravel_index(np.argmax(img),img.shape)
        return (point),point
    elif len(points)==1: # we have one figure
        points = points[0]
        return points, points[np.argmax([img[p] for p in points])]
    else: # we have multiple figures of same area
        len_dict = {i:np.max([img[p] for p in points[i]]) for i in range(len(points))}
        index = max(len_dict,key=len_dict.get)
        return points[index], points[index][np.argmax([img[p] for p in points[index]])]


def detect_shapes(img,rgb_img,threshold_max,verbose):

    gray = img

    #threshold = cv.adaptiveThreshold(gray,threshold_max,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,9,13)
    _,threshold = cv.threshold(gray,0,threshold_max,cv.THRESH_BINARY+cv2.THRESH_TRIANGLE)

    figures = find_max_area_fig(threshold,verbose)

    img,mask,point = apply_mask(figures,threshold)

    rgb_img[point] = (255,0,0)
    if verbose:
        fig, axes = plt.subplots(1,4)
        axes[0].imshow(cv.cvtColor(threshold, cv2.COLOR_BGR2RGB))
        axes[1].imshow(cv.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[2].imshow(cv.cvtColor(gray, cv2.COLOR_GRAY2RGB))
        axes[3].imshow(rgb_img)
        plt.show()

    return point



if __name__ == "__main__":

    path = r"photos\\mountains.jpg"

    img = read_img(path)
    iter_matrix_shape = 361
    bias_x,bias_y = 370, 350
    shift_x,shift_y = 3,4
    mode = 'density'
    verbpse = False
    fig,axes = plt.subplots(2,3,figsize=(16,19))

    for channel in [2,1,0]:

        cropped_img = crop_img(img, (iter_matrix_shape, iter_matrix_shape),(bias_y,bias_x))

        points = find_max(cropped_img,channel,mode,verbpse)

        cropped_img_shifted = shift(cropped_img,shift_y,shift_x)

        points_shifted = find_max(cropped_img_shifted,channel,mode,verbpse)


        #print(points)
        #print(points_shifted)
        result_vec = mean_vector(points,points_shifted)
        print(result_vec)

        image = cropped_img.copy()
        arrow_color = (0,0,255)
        for i in range(len(points)):
            image = cv2.arrowedLine(image,points[i],points_shifted[i],arrow_color,2)
        if channel==2:
            axes[0,0].imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            axes[0,1].imshow(cv2.cvtColor(cropped_img_shifted, cv2.COLOR_BGR2RGB))
        axes[1,channel].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


    plt.show()





    """"
    modes = [cv.THRESH_BINARY,cv.THRESH_BINARY_INV,cv.THRESH_TRUNC,cv.THRESH_TOZERO,cv.THRESH_TOZERO_INV]
    #modes1 = [mode+cv.THRESH_OTSU for mode in modes]
    modes2 = [mode+cv.THRESH_TRIANGLE for mode in modes]"""

