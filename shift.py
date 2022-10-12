import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
path = r"photos\\img4.jpg"
img = cv.imread(path)
shape = img.shape



def shift(img,vertical_shift=0,horizontal_shift=0):
    if vertical_shift>=0 and horizontal_shift>=0:
        return np.pad(img,((vertical_shift,0),(horizontal_shift,0),(0,0)))[:shape[0],:shape[1],:]
    elif vertical_shift<0 and horizontal_shift>=0:
        return np.pad(img,((0,-vertical_shift),(horizontal_shift,0),(0,0)))[-vertical_shift:,:shape[1],:]
    elif vertical_shift>=0 and horizontal_shift<0:
        return np.pad(img,((vertical_shift,0),(0,-horizontal_shift),(0,0)))[:shape[0],-horizontal_shift:,:]
    else:
        return np.pad(img,((0,-vertical_shift),(0,-horizontal_shift),(0,0)))[-vertical_shift:,-horizontal_shift:,:]

def print_propeties(imgs):
    for img in imgs:
        print(f"Shape is {img.shape}")


def visualize(img,name):
    cv.imwrite(name,img)
    #cv.imshow(name, img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

def make_hist(columns_measure,name,label,img_name):
    plt.hist(columns_measure)
    plt.title(name)
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.savefig(img_name)
    plt.clf()
    #plt.show()

def visualize_residuals(columns_measure,name,label,img_name):
    plt.plot([i for i in range(1,len(columns_measure)+1)],columns_measure)
    plt.title(name)
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.savefig(img_name)
    plt.clf()
    #plt.show()

def measure(img):
    return column_measure(img).max()

def column_measure(img):
    return img.sum(axis=-1).T.sum(axis=1)



for i in range(4):
    for j in range(4):
        if i==j and not i:
            continue

        folder_prefix = 'results\\'
        img_name = f'vertical {i}; horizontal {j}'
        histogram_prefix = 'hist '
        res_prefix = 'measure of '
        visualizatin_prefix = 'visualization of '

        img_sh = shift(img,i,j)
        res = abs(img_sh-img)
        measure_ij = measure(res)
        print(f"Measure for {i}-{j}: {measure_ij}")
        col_measure = column_measure(res)
        make_hist(col_measure,f'Histogram of vertical {i} and horizontal {j}.\nMaximum value is {measure_ij}',('value of a column measure','number of columns'),folder_prefix+histogram_prefix+img_name)
        visualize_residuals(col_measure,'',('number of  column','value of a column measure'),folder_prefix+res_prefix+img_name)
        visualize(res,folder_prefix+img_name+'.jpg')


