import cv2
import numpy as np

from utils import Image_Helper, SpecialList, Vector, VectorHelper
from point_finders import CenterMassFinder


class LocalAndGlobalMaximumShift(Image_Helper):

    def __init__(self, split_vert, split_horiz, vert_shift, horiz_shift, angle_rotate, channel, img=None, filepath=None):
        super().__init__(img, filepath)
        self.Image_main = self
        self.Image_shifted = self.Image_main.shift(vertical_shift=vert_shift, horizontal_shift=horiz_shift).rotate(angle_rotate)
        self.Images_main_split = self.Image_main.split(height_num=split_vert, width_num=split_horiz)
        self.Images_shifted_split = self.Image_shifted.split(height_num=split_vert, width_num=split_horiz)
        self.Image_restored = None

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

        self.L = np.zeros((2,2)) # transformation operator
        self.T = None # transformation matrix to new basys

    def main(self):
        self.O_tilda_O = self._find_global_vector()
        self._assign_points()
        self._calculate_basys_vectors()
        if self._able_to_create_operator():
            self._contsruct_transformation_operator()
            self._basys_change()

        if np.linalg.det(self.L)==0:
            print("Matrix is singular. Only shifting by -s")
            M = np.float32([[1,0, -self.O_tilda_O.get_direction()[1]],
                        [0,1, -self.O_tilda_O.get_direction()[0]]])
        else:
            L_inverted = np.linalg.inv(self.L)
            print("L\n",self.L)
            print("L inverted\n",L_inverted)

            M = np.float32([[L_inverted[0,0], L_inverted[0,1], -self.O_tilda_O.get_direction()[1]],
                        [L_inverted[1,0], L_inverted[1,1], -self.O_tilda_O.get_direction()[0]]])

        img_restored = cv2.warpAffine(self.Image_shifted.img, M, (self.Image_main.shape[1],self.Image_main.shape[0]))
        self.Image_restored = Image_Helper(img=img_restored)
        return self.Image_restored

    def _basys_change(self):
        e1 = (self.O_A.move_to_start())/self.O_A.get_length()
        e2 = (self.O_B.move_to_start())/self.O_B.get_length()

        """E_1 = Vector(SpecialList([0,0]),SpecialList([0,1]))
        E_2 = Vector(SpecialList([0,0]),SpecialList([1,0]))

        e_1x = np.cos(VectorHelper.get_angle(e1,E_1))
        e_1y = np.cos(VectorHelper.get_angle(e2,E_1))

        e_2x = np.cos(VectorHelper.get_angle(e1,E_2))
        e_2y = np.cos(VectorHelper.get_angle(e2,E_2))

        self.T = np.array([[e_1x,e_2x],[e_1y,e_2y]]) # from e->E basys
        self.L = np.linalg.inv(self.T)@self.L@self.T # matrix in basys E_1,E_2"""

        e_1x,e_1y = e1.get_direction()
        e_2x,e_2y = e2.get_direction()
        self.T = np.array([[e_1x,e_2x],[e_1y,e_2y]])  # from E->e basys
        self.L = self.T@self.L@np.linalg.inv(self.T)  # matrix in basys E_1,E_2

    def _able_to_create_operator(self):
        return self.O_A!=self.O_A1 and self.O_B!=self.O_B1

    def _contsruct_transformation_operator(self):
        e_1 = self.O_A.move_to_start()/self.O_A.get_length()
        e_2 = self.O_B.move_to_start()/self.O_B.get_length()
        O_A1_normalized = (self.O_A1.move_to_start())/self.O_A.get_length()
        O_B1_normalized = (self.O_B1.move_to_start())/self.O_B.get_length()
        line_e1 = VectorHelper.get_line_eq_coeffs(e_1)
        line_e2 = VectorHelper.get_line_eq_coeffs(e_2)

        parallel_e2_line_OA1 = VectorHelper.move_line_to_point(line_e2,O_A1_normalized)
        parallel_e1_line_OA1 = VectorHelper.move_line_to_point(line_e1,O_A1_normalized)
        L_1x_abs = VectorHelper.find_intersection(parallel_e2_line_OA1,line_e1)
        L_1y_abs = VectorHelper.find_intersection(parallel_e1_line_OA1,line_e2)
        projection_vector_L_1x = Vector(e_1.start,SpecialList(L_1x_abs))
        projection_vector_L_1y = Vector(e_2.start,SpecialList(L_1y_abs))
        L_1x = projection_vector_L_1x.get_length()/e_1.get_length()*VectorHelper.co_direction(e_1,projection_vector_L_1x)
        L_1y = projection_vector_L_1y.get_length()/e_2.get_length()*VectorHelper.co_direction(e_2,projection_vector_L_1y)

        parallel_e2_line_OB1 = VectorHelper.move_line_to_point(line_e2,O_B1_normalized)
        parallel_e1_line_OB1 = VectorHelper.move_line_to_point(line_e1,O_B1_normalized)
        L_2x_abs = VectorHelper.find_intersection(parallel_e2_line_OB1,line_e1)
        L_2y_abs = VectorHelper.find_intersection(parallel_e1_line_OB1,line_e2)
        projection_vector_L_2x = Vector(e_1.start,SpecialList(L_2x_abs))
        projection_vector_L_2y = Vector(e_2.start,SpecialList(L_2y_abs))
        L_2x = projection_vector_L_2x.get_length()/e_1.get_length()*VectorHelper.co_direction(e_1,projection_vector_L_2x)
        L_2y = projection_vector_L_2y.get_length()/e_2.get_length()*VectorHelper.co_direction(e_2,projection_vector_L_2y)

        self.L = np.array([[L_1x,L_2x],[L_1y,L_2y]])

    def _calculate_basys_vectors(self):
        O_A_tilda = Vector(self.O, self.A_tilda)
        O_B_tilda = Vector(self.O, self.B_tilda)
        self.O_A = O_A_tilda + self.O_tilda_O
        self.O_B = O_B_tilda + self.O_tilda_O
        self.O_A1 = Vector(self.O, self.A1)
        self.O_B1 = Vector(self.O, self.B1)

    def _assign_points(self):
        self.O = self.O_tilda_O.end.copy()
        A_tilda_A1, B_tilda_B1 = self._select_A_and_B()
        self.A_tilda = A_tilda_A1.start.copy()
        self.A1 = A_tilda_A1.end.copy()
        self.B_tilda = B_tilda_B1.start.copy()
        self.B1 = B_tilda_B1.end.copy()

    def _select_A_and_B(self):
        """Selects two vectors, which will form an angle (BOA) closest to the 90 degrees.
            B = B_tilda + O_O_tilda
            A = A_tilda + O_O_tilda"""
        assert len(self.vectors)>=2
        angles_map = {} #  key is pair of indexes of vectors, value: pair: 1st is how far the angle from 90 degrees is,2nd is real value
        for i in range(len(self.vectors)):
            for j in range(len(self.vectors)):
                if i!=j and (i,j) not in angles_map.keys() and (j,i) not in angles_map.keys():
                    A_tilda = self.vectors[i].start
                    B_tilda = self.vectors[j].start
                    OA = Vector(self.O, A_tilda) + self.O_tilda_O
                    OB = Vector(self.O, B_tilda) + self.O_tilda_O
                    angles_map[(i,j)] = (abs(VectorHelper.get_angle(OA,OB)-90),VectorHelper.get_angle(OA,OB))

        angles_map = dict(sorted(angles_map.items(), key=lambda item: item[1][0]))
        i,j = list(angles_map.keys())[0]
        return self.vectors[i],self.vectors[j]
    def _find_global_vector(self):
        for i in range(self.split_vert):
            for j in range(self.split_horiz):
                img, shifted_img = self.Images_main_split[j + i * self.split_vert].img, \
                                   self.Images_shifted_split[j + i * self.split_vert].img

                points = SpecialList(CenterMassFinder(img=img).find_max(self.channel))

                points_shifted = SpecialList(CenterMassFinder(img=shifted_img).find_max(self.channel))

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


method = LocalAndGlobalMaximumShift(filepath=r"photos\\tree.jpg",split_vert=2,split_horiz=2,
                                    vert_shift=0,horiz_shift=0,angle_rotate=3,channel=2)
print(method.img.shape)
method.main()
global_vect = method.O_tilda_O
method.Image_main.show("Original").save("Tree_Original.jpg")
method.Image_shifted.show("Shifted").save("Tree_Shifted.jpg")
method.Image_restored.show("Restored").save("Tree_Restored.jpg")
print("Vectors: ")
for vector in method.vectors:
    print(vector)
    vector.draw(method.Image_main.img,size=4,color=(0,255,255))
print("Global: ",global_vect)
global_vect.display(is_canvas=True,canvas=method.Image_main.img,name="Vectors",show_coordinates=True,size=3,color=(255,2,255))
method.Image_main.save("Music_vectors.jpg")


Image_Helper(img=np.abs(method.Image_main.img-method.Image_restored.img)).show("Difference").save("Tree_Difference.jpg")
