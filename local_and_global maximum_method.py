import cv2
from image_helper import Image_Helper
from averaging import find_max
from shift import shift

imagepath = r"photos\\mountains.jpg"
split_vert, split_horiz = 3,3
Image = Image_Helper(filepath=imagepath)
Image_shifted = Image_Helper(img=shift(Image.img,3,5))
images = Image.split(split_vert,split_horiz)
images_shifted = Image_shifted.split(split_vert,split_horiz)
channel = 2 # BGR is by default

points_global_avg=[0,0]
points_shifted_global_avg=[0,0]
points_global=[]
points_shifted_global=[]
for i in range(split_vert):
    for j in range(split_horiz):
        img, shifted_img = images[j+i*split_vert].img, images_shifted[j+i*split_vert].img

        points =find_max(img,channel,"simple",False)
        points_shifted = find_max(shifted_img,channel,"simple",False)
        vector = [points_shifted[0]-points[0],points_shifted[1]-points[1]]

        points[0] += i*(Image.img.shape[0]//split_vert) # coordinates are (y,x)
        points[1] += j*(Image.img.shape[1]//split_horiz)

        points_shifted[0] += i*(Image_shifted.img.shape[0]//split_vert) # coordinates are (y,x)
        points_shifted[1] += j*(Image_shifted.img.shape[1]//split_horiz)

        points_global_avg[0] += points[0]//(split_vert*split_horiz)
        points_global_avg[1] += points[1]//(split_vert*split_horiz)
        points_shifted_global_avg[0] += points_shifted[0]//(split_vert*split_horiz)
        points_shifted_global_avg[1] += points_shifted[1]//(split_vert*split_horiz)

        points_global.append(tuple(points))
        points_shifted_global.append(tuple(points_shifted))

        print((i,j), points, points_shifted, vector)
        #images[j+i*split_vert].show()
        #images_shifted[i].show()

print()
print(points_global)
print(points_shifted_global)
arrow_color = (255,0,255)
for i in range(len(points_global)):
    image = cv2.arrowedLine(Image.img, (points_global[i][1],points_global[i][0]),
                                (points_shifted_global[i][1],points_shifted_global[i][0]), arrow_color, 4)  # translation (y,x) -> (x,y)

Image_Helper(img=image).show()