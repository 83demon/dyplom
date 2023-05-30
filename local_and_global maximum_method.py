import cv2
import numpy as np
import time
from utils import Image_Helper, SpecialList, Vector, VectorHelper
from point_finders import CenterMassFinder
from focus import Focus

class SeriesPreprocessor:
    def __init__(self, img_arrays):
        if len(img_arrays)<2:
            raise ValueError("Number of images provided is less than two.")
        img_shapes = set()
        for img in img_arrays:
            img_shapes.add(img.shape)
        if len(img_shapes)!=1:
            raise ValueError("Images are of different shapes.")
        self.results = []
        self.length = len(img_arrays)
        self.img_arrays = img_arrays

    def main(self,show=False,focus=False,brigth=True):
        sample = [Image_Helper(img=self.img_arrays[0]).find_countors_area(self.img_arrays[0])]
        for i in range(self.length-1):
            image_restored_object = LocalAndGlobalMaximumShift(img=self.img_arrays[i],img_shifted=self.img_arrays[i+1]).main(sample, show=show, focus=focus)
            self.results.append(image_restored_object)
        median = np.median(sample)
        index_of_candidate = np.abs(sample - median).argmin()
        res = None
        if not focus:
            res = Focus(img=self.img_arrays[index_of_candidate]).fft_deblur_rgb()
        else:
            res = self.results[index_of_candidate]
        if brigth:
            res = res.change_brightness()
        return res

class LocalAndGlobalMaximumShift(Image_Helper):

    def __init__(self, split_vert=2, split_horiz=2, channel=2, img=None,filepath=None, img_shifted=None,filepath_shifted=None):
        super().__init__(img, filepath)
        self.Image_main = self
        self.Image_shifted = Image_Helper(img=img_shifted,filepath=filepath_shifted)
        self.Images_main_split = self.Image_main.split(height_num=split_vert, width_num=split_horiz)
        self.Images_shifted_split = self.Image_shifted.split(height_num=split_vert, width_num=split_horiz)
        self.Image_restored = None

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

    def main(self,sample:list,show=False,focus=False):
        self.O_tilda_O = self._find_global_vector()
        self._assign_points()
        self._calculate_basys_vectors()
        if self._able_to_create_operator():
            self._contsruct_transformation_operator()
            self._basys_change()
        res = None
        if np.linalg.det(self.L)==0:
            print("Matrix is singular. No transformations will be done.")
            res = self.Image_main
        else:
            shift_vec = np.array([self.O_tilda_O.get_direction()]).reshape(2,1)
            rotation_matrix = np.hstack((self.L,shift_vec))
            stacked_temp_matr = np.vstack((rotation_matrix,np.array([0,0,1])))
            temp_inv_mat = np.linalg.inv(stacked_temp_matr)
            rotation_matrix_inversed = cv2.invertAffineTransform(rotation_matrix)
            img_restored = cv2.warpAffine(self.Image_shifted.img, rotation_matrix_inversed, (self.Image_main.shape[1],self.Image_main.shape[0]))
            self.Image_restored = Image_Helper(img=img_restored)
            res = self.Image_restored
        sample.append(self.find_countors_area(res.img))
        if focus:
            res = Focus(img=self.Image_restored.img).fft_deblur_rgb()
        if show:
            self.show_difference()
        return res

    def _basys_change(self):
        e1 = (self.O_A.move_to_start())/self.O_A.get_length()
        e2 = (self.O_B.move_to_start())/self.O_B.get_length()
        e_1x,e_1y = e1.get_direction()
        e_2x,e_2y = e2.get_direction()
        self.T = np.array([[e_1x,e_2x],[e_1y,e_2y]])  # from E->e basys
        self.L = self.T@self.L@np.linalg.inv(self.T)  # matrix in basys E_1,E_2

        """ Alternative version:
        E_1 = Vector(SpecialList([0,0]),SpecialList([0,1]))
        E_2 = Vector(SpecialList([0,0]),SpecialList([1,0]))

        e_1x = np.cos(VectorHelper.get_angle(e1,E_1))
        e_1y = np.cos(VectorHelper.get_angle(e2,E_1))

        e_2x = np.cos(VectorHelper.get_angle(e1,E_2))
        e_2y = np.cos(VectorHelper.get_angle(e2,E_2))

        self.T = np.array([[e_1x,e_2x],[e_1y,e_2y]]) # from e->E basys
        self.L = np.linalg.inv(self.T)@self.L@self.T # matrix in basys E_1,E_2
        """

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

                # normalizing with the respect to a number of subimages

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

    def difference(self):
        return Image_Helper(img=np.abs(self.Image_main.img-self.Image_restored.img))

    def show_difference(self):
        self.difference().show("Difference")

