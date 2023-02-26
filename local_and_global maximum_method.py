import cv2
from utils import Image_Helper, SpecialList, Vector
from point_finders import MaximumColorFinder


class LocalAndGlobalMaximumShift(Image_Helper):

    def __init__(self, split_vert, split_horiz, vert_shift, horiz_shift, channel, img=None, filepath=None):
        super().__init__(img, filepath)
        self.Image_main = self
        self.Image_shifted = self.Image_main.shift(vertical_shift=vert_shift, horizontal_shift=horiz_shift)
        self.Images_main_split = self.Image_main.split(height_num=split_vert, width_num=split_horiz)
        self.Images_shifted_split = self.Image_shifted.split(height_num=split_vert, width_num=split_horiz)

        self.split_vert = split_vert
        self.split_horiz = split_horiz

        self.global_avg = SpecialList([0, 0])
        self.global_avg_shifted = SpecialList([0, 0])
        self.vectors = []

        self.channel = channel

    def find_global_vector(self):
        for i in range(self.split_vert):
            for j in range(self.split_horiz):
                img, shifted_img = self.Images_main_split[j + i * self.split_vert].img, \
                                   self.Images_shifted_split[j + i * self.split_vert].img

                points = SpecialList(MaximumColorFinder(img=img).find_max(self.channel))

                points_shifted = SpecialList(MaximumColorFinder(img=shifted_img).find_max(self.channel))

                # normalizing with the respect to number of subimage

                points[0] += i * (self.Image_main.shape[0] // self.split_vert)  # coordinates are (y,x)
                points[1] += j * (self.Image_main.shape[1] // self.split_horiz)

                points_shifted[0] += i * (self.Image_shifted.shape[0] // self.split_vert)  # coordinates are (y,x)
                points_shifted[1] += j * (self.Image_shifted.shape[1] // self.split_horiz)

                vector = Vector(points, points_shifted)
                self.vectors.append(vector)

                self.global_avg += points
                self.global_avg_shifted += points_shifted

        self.global_avg //= (self.split_vert * self.split_horiz)
        self.global_avg_shifted //= (self.split_vert * self.split_horiz)

        return Vector(self.global_avg,self.global_avg_shifted)


method = LocalAndGlobalMaximumShift(filepath=r"photos\\mountains.jpg",split_vert=3,split_horiz=3,
                                    vert_shift=3,horiz_shift=7,channel=2)
print(method.img.shape)
global_vect = method.find_global_vector()
print("Vectors: ")
for vector in method.vectors:
    print(vector)
    vector.draw(method.Image_main.img)
print("Global: ",global_vect)
global_vect.display(is_canvas=True,canvas=method.Image_main.img,name="Vectors",show_coordinates=True,size=3,color=(0,255,255))
