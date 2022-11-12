import cv2
import cv2 as cv
import numpy as np


def crop_img(img,shape:tuple,bias:tuple=(0,0)):
    "Crops first part of image of shape 'shape'"
    return img[bias[0]:bias[0]+shape[0],bias[1]:bias[1]+shape[1],:]

def find_max(img,channel):
    """finds position of a maximum of a specific channel, in every of shape[0]*matrices
        of shape (sqrt(shape[0]),sqrt(shape[0]))
        img has shape of (shape[0],shape[0])
        shape[0] must be a perfect square"""
    points = []
    a = np.sqrt(img.shape[0]).astype(int)
    for i in range(a*a):
        matrix = img[(i//a)*a:((i//a)+1)*a,(i%a)*a:((i%a)+1)*a,channel]
        detect_shapes(matrix,255)
        coords = np.unravel_index(np.argmax(matrix),matrix.shape) #return (y,x)
        coords = list(coords)
        coords[0] += (i//a)*a
        coords[1] += (i%a)*a
        #print(coords,matrix[coords]==matrix.max(),matrix.max(),matrix.min())
        #show_img(matrix)
        points.append(tuple(coords))
    return points




def show_img(img,name=""):
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def read_img(filename):
    return cv.imread(filename)

def mean_vector(vec1,vec2):
    return np.sum((np.array(vec2)-np.array(vec1))/len(vec1),axis=0).tolist()

def detect_shapes(img,threshold_max):
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt


    # converting image into grayscale image
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img

    #threshold = cv2.adaptiveThreshold(gray,threshold_max,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,9,13)
    _,threshold = cv2.threshold(gray,0,threshold_max,cv.THRESH_BINARY+cv2.THRESH_TRIANGLE)

    fig, axes = plt.subplots(1,2)
    axes[0].imshow(cv2.cvtColor(threshold, cv2.COLOR_BGR2RGB))

    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


    # displaying the image after drawing contours
    axes[1].imshow(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    plt.show()




channel = 2

path = r"photos\\rainbow.jpg"

img = read_img(path)
#show_img(img)
cropped_img = crop_img(img,(100,100),(0,0))
show_img(cropped_img[:,:,channel],"cropped")

points = find_max(cropped_img,channel)
print(points)

from shift import shift
cropped_img_shifted = shift(cropped_img,2,3)
show_img(cropped_img_shifted[:,:,channel])
points_shifted = find_max(cropped_img_shifted,channel)
print(points_shifted)

image = cropped_img
for i in range(len(points)):
    image = cv2.arrowedLine(image,points[i],points_shifted[i],(255,255,255),2)
show_img(image)

result_vec = mean_vector(points,points_shifted)
newly_shifted = shift(cropped_img_shifted,int(result_vec[0]),int(result_vec[1]))
show_img(newly_shifted)
print(result_vec)


"""img_name = r"photos\\img4_1.jpg"
modes = [cv.THRESH_BINARY,cv.THRESH_BINARY_INV,cv.THRESH_TRUNC,cv.THRESH_TOZERO,cv.THRESH_TOZERO_INV]
#modes1 = [mode+cv.THRESH_OTSU for mode in modes]
modes2 = [mode+cv.THRESH_TRIANGLE for mode in modes]"""

