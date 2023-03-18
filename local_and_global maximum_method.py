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

        self.O : SpecialList = None  # global shifted point
        self.A_tilda : SpecialList = None  # local starting point
        self.B_tilda : SpecialList = None  # local starting point
        self.A1 : SpecialList = None  # local resulting point
        self.B1 : SpecialList = None  # local resulting point
        self.O_tilda_O : Vector = None  # global shift, "s" vector
        self.O_A1 : Vector = None
        self.O_B1 : Vector = None
        self.O_A : Vector = None
        self.O_B : Vector = None

        self.L = None # transformation operator
        self.T = None # transformation matrix to new basys

    def main(self):
        self.O_tilda_O = self._find_global_vector()
        self._assign_points(self.O_tilda_O)
        self._calculate_basys_vectors()
        self._contsruct_transformation_operator()
        self._basys_change()

        if np.linalg.det(self.L)==0:
            print("Matrix is singular. Only shifting by -s")
            M = np.float32([[1,0, -self.O_tilda_O.get_direction()[1]],
                        [0,1, -self.O_tilda_O.get_direction()[0]]])
        else:
            L_inverted = np.linalg.inv(L)
            M = np.float32([[L_inverted[0,0], L_inverted[0,1], -self.O_tilda_O.get_direction()[1]],
                        [L_inverted[1,0], L_inverted[1,1], -self.O_tilda_O.get_direction()[0]]])

        img_restored = cv2.warpAffine(self.Image_shifted.img, M, (self.Image_main.shape[1],self.Image_main.shape[0]))
        Image_restored = Image_Helper(img=img_restored) # shifting by (-s) vector
        Image_restored.show("Restored")

    def _basys_change(self):
        e1 = (self.O_A.move_to_start())/self.O_A.get_length()
        e2 = (self.O_B.move_to_start())/self.O_B.get_length()
        E_1 = Vector(SpecialList([0,0]),SpecialList([0,1]))
        E_2 = Vector(SpecialList([0,0]),SpecialList([1,0]))
        """e_1x = np.cos(Vector.get_angle(e1,E_1))  # as E_1 and E_2 are unary vectors, we do not need to muptiply their length on np.cos()
        e_1y = np.cos(Vector.get_angle(e2,E_1))
        e_2x = np.cos(Vector.get_angle(e1,E_2))
        e_2y = np.cos(Vector.get_angle(e2,E_2))
        self.T = np.array([[e_1x,e_2x],[e_1y,e_2y]]) # from e->E basys
        self.L = np.linalg.inv(self.T)@self.L@self.T # matrix in basys E_1,E_2"""

        e_1x,e_1y = e1.get_direction()
        e_2x,e_2y = e2.get_direction()
        self.T = np.array([[e_1x,e_2x],[e_1y,e_2y]])  # from E->e basys
        self.L = self.T@self.L@np.linalg.inv(self.T)  # matrix in basys E_1,E_2"""



    def _contsruct_transformation_operator(self):
        O_A = self.O_A.move_to_start()
        O_B = self.O_B.move_to_start()
        O_A1_normalized = (self.O_A1.move_to_start())/O_A
        O_B1_normalized = (self.O_B1.move_to_start())/O_B
        L_1x = O_A1_normalized.get_length()*np.cos(Vector.get_angle(O_A,O_A1_normalized))
        L_1y = O_A1_normalized.get_length()*np.cos(Vector.get_angle(O_B,O_A1_normalized))
        L_2x = O_B1_normalized.get_length()*np.cos(Vector.get_angle(O_A,O_B1_normalized))
        L_2y = O_B1_normalized.get_length()*np.cos(Vector.get_angle(O_B,O_B1_normalized))
        self.L = np.array([[L_1x,L_2x],[L_1y,L_2y]])

    def _calculate_basys_vectors(self):
        O_A_tilda = Vector(self.O, self.A_tilda)
        O_B_tilda = Vector(self.O, self.B_tilda)
        self.O_A = O_A_tilda + self.O_tilda_O
        self.O_B = O_B_tilda + self.O_tilda_O
        self.O_A1 = Vector(self.A_tilda, self.A1) + self.O_tilda_O + self.O_A
        self.O_B1 = Vector(self.B_tilda, self.B1) + self.O_tilda_O + self.O_B

    def _assign_points(self,global_vect):
        self.O = global_vect.end.copy()
        A_tilda_A1, B_tilda_B1 = self._select_A_and_B()
        self.A_tilda = A_tilda_A1.start.copy()
        self.A1 = A_tilda_A1.end.copy()
        self.B_tilda = B_tilda_B1.start.copy()
        self.B1 = B_tilda_B1.end.copy()

    def _select_A_and_B(self):
        assert len(self.vectors)>=2
        angles_map = {}
        for i in range(len(self.vectors)):
            for j in range(len(self.vectors)):
                if i!=j and (i,j) not in angles_map.keys() and (j,i) not in angles_map.keys():
                    angles_map[(i,j)] = Vector.get_angle(self.vectors[i],self.vectors[j])

        angles_map = dict(sorted(angles_map.items(), key=lambda item: item[1],reverse=True))
        i,j = list(angles_map.keys())[0]
        return self.vectors[i],self.vectors[j]
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
method.main()
print(method.O_O1.get_direction())
#method.Image_main.show("main")
#method.Image_shifted.show("shited")
"""global_vect = method._find_global_vector()
print("Vectors: ")
for vector in method.vectors:
    print(vector)
    vector.draw(method.Image_main.img,size=4,color=(0,255,255))
print("Global: ",global_vect)
global_vect.display(is_canvas=True,canvas=method.Image_main.img,name="Vectors",show_coordinates=True,size=3,color=(255,2,255))
method.Image_main.save("211test.jpg")"""