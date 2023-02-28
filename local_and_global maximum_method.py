import cv2
import numpy as np

from utils import Image_Helper, SpecialList, Vector
from point_finders import MaximumColorFinder


class LocalAndGlobalMaximumShift(Image_Helper):

    def __init__(self, split_vert, split_horiz, vert_shift, horiz_shift, channel, img=None, filepath=None):
        super().__init__(img, filepath)
        self.Image_main = self
        self.Image_shifted = self.Image_main.shift(vertical_shift=vert_shift, horizontal_shift=horiz_shift)
        self.Images_main_split = self.Image_main.split(height_num=split_vert, width_num=split_horiz)
        self.Images_shifted_split = self.Image_shifted.split(height_num=split_vert, width_num=split_horiz)

        self.vert_shift = vert_shift
        self.horiz_shift = horiz_shift
        self.split_vert = split_vert
        self.split_horiz = split_horiz

        self.global_avg = SpecialList([0, 0])
        self.global_avg_shifted = SpecialList([0, 0])
        self.vectors = []

        self.channel = channel

        self.O = None
        self.A = None
        self.B = None
        self.A1 = None
        self.B1 = None
        self.O_O1 = None # global shift, "s" vector
        self.O_A1 = None
        self.O_B1 = None

    def main(self):
        self.O_O1 = self._find_global_vector()
        self._assign_points(self.O_O1)
        OA = Vector(self.O,self.A)
        OB = Vector(self.O,self.B)
        self.O_A1 = Vector(self.A,self.A1) + self.O_O1 + OA
        self.O_B1 = Vector(self.B,self.B1) + self.O_O1 + OB
        a = (self.O_A1 - OA).get_direction()
        b = (self.O_B1 -OB).get_direction()
        transformation_matrix = np.matrix([a,b]).T
        #inv_matrix = np.linalg.pinv(transformation_matrix)
        #out = cv2.warpAffine(self.Image_main.img, inv_matrix, (self.Image_main.shape[1], self.Image_main.shape[0]))
        #Image_Helper(img=out).show()


    def _assign_points(self,global_vect):
        self.O = global_vect.end.copy()
        A_A1, B_B1 = self._select_A_and_B()
        self.A = A_A1.start.copy()
        self.A1 = A_A1.end.copy()
        self.B = B_B1.start.copy()
        self.B1 = B_B1.end.copy()

    def _select_A_and_B(self):
        assert len(self.vectors)>=2

        if len(selected:=[v for v in self.vectors if v.get_direction()==SpecialList([self.vert_shift,self.horiz_shift])]) >= 2:
            return selected[:2]
        elif len(selected)==1:
            item_first = selected.pop()
            for item_second in self.vectors:
                if item_second!=item_first:
                    return [item_first,item_second]
        else:
            return self.vectors[:2]

    def _find_global_vector(self):
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


method = LocalAndGlobalMaximumShift(filepath=r"photos\\moon.jpg",split_vert=2,split_horiz=2,
                                    vert_shift=3,horiz_shift=7,channel=2)
print(method.img.shape)
#method.main()
global_vect = method._find_global_vector()
print("Vectors: ")
for vector in method.vectors:
    print(vector)
    vector.draw(method.Image_main.img,size=4,color=(0,255,255))
print("Global: ",global_vect)
global_vect.display(is_canvas=True,canvas=method.Image_main.img,name="Vectors",show_coordinates=True,size=3,color=(255,2,255))
method.Image_main.save("211test.jpg")