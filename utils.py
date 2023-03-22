import sys

import cv2
import numpy as np
import math
from copy import copy

class Image_Helper:
    color_to_channel_map = {2: "Red", 1: "Green", 0: "Blue"}

    def __init__(self, img=None, filepath=None, ):
        if filepath:
            self.img = Image_Helper._read(filepath)
        elif img.any():
            self.img = img
        else:
            raise ValueError("No arguments were provided")
        self.active_channel = None  # active channel to display

    def display_coordinates(self, event, x, y, flags, params):
        if event == 2:
            if not isinstance(self.active_channel, int):
                print(f"Coordinates (y,x): {y,x}; RGB is : {(self.img[y, x, 2], self.img[y, x, 1], self.img[y, x, 0])}")
            else:
                print(
                    f"Coordinates (y,x): {y,x}; Color is {Image_Helper.color_to_channel_map[self.active_channel]} : {self.img[y, x, self.active_channel]}")

    @property
    def shape(self):
        return self.img.shape

    @staticmethod
    def _read(filename):
        return cv2.imread(filename)

    def crop(self, shape: tuple, bias: tuple = (0, 0)):
        "Crops first part of image of shape 'shape'"
        return Image_Helper(img=self.img[bias[0]:bias[0] + shape[0], bias[1]:bias[1] + shape[1], :])

    def save(self,name,prefix=r".\\saved_photos\\"):
        return cv2.imwrite(prefix+name,self.img)

    def show(self, name="", show_coordinates=False, channel=None):
        self.active_channel = channel
        if show_coordinates:
            cv2.namedWindow(name)
            cv2.setMouseCallback(name, self.display_coordinates)
        cv2.imshow(name, self.img[:, :, channel] if isinstance(channel, int) else self.img)
        cv2.moveWindow(name,0,0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def show_multiple_images(images: list, captions: list = None, step_by_step: bool = False,
                             show_coordinates: bool = False, channel=None):
        """images is list of type Image_Helper, captions is list of str"""
        if captions:
            assert len(images) == len(captions)
        if step_by_step:
            for i in range(len(images)):
                images[i].show(name=str(i) if not captions else captions[i], show_coordinates=show_coordinates,
                               channel=channel)
        else:
            for i in range(len(images)):
                cv2.imshow(str(i) if not captions else captions[i],
                           images[i].img[:, :, channel] if isinstance(channel, int) else images[i].img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def split(self, height_num, width_num):
        """Splits given image into height_num * width_num smaller images"""
        height_length = self.img.shape[0] // height_num
        width_length = self.img.shape[1] // width_num
        return [Image_Helper(
            img=self.img[i * height_length:(i + 1) * height_length, j * width_length:(j + 1) * width_length])
                for i in range(height_num) for j in range(width_num)]



    def shift(self, vertical_shift=0, horizontal_shift=0):
        if vertical_shift >= 0 and horizontal_shift >= 0:
            return Image_Helper(
                img=np.pad(self.img, ((vertical_shift, 0), (horizontal_shift, 0), (0, 0)))[:self.shape[0],
                    :self.shape[1], :])
        elif vertical_shift < 0 and horizontal_shift >= 0:
            return Image_Helper(
                img=np.pad(self.img, ((0, -vertical_shift), (horizontal_shift, 0), (0, 0)))[-vertical_shift:,
                    :self.shape[1],
                    :])
        elif vertical_shift >= 0 and horizontal_shift < 0:
            return Image_Helper(
                img=np.pad(self.img, ((vertical_shift, 0), (0, -horizontal_shift), (0, 0)))[:self.shape[0],
                    -horizontal_shift:,
                    :])
        else:
            return Image_Helper(
                img=np.pad(self.img, ((0, -vertical_shift), (0, -horizontal_shift), (0, 0)))[-vertical_shift:,
                    -horizontal_shift:, :])

class SpecialList(list):
    def __init__(self, data: list = []):
        super().__init__(data)

    def __add__(self, other):
        if isinstance(other,(int,float)):
            return SpecialList([other + i for i in self])
        else:
            assert len(self) == len(other)

            new = SpecialList()
            for i in range(len(self)):
                new.append(self[i] + other[i])
            return new

    def __iadd__(self, other):
        self = self + other
        return self

    def __radd__(self, other):
        return self+other
    def __neg__(self):
        new = SpecialList(self)
        for i in range(len(self)):
            new[i] = -new[i]
        return new

    def __invert__(self):
        return SpecialList([1/self[i] for i in range(len(self))])

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other,(int,float)):
            return SpecialList([other*i for i in self])
        elif isinstance(other,list):
            assert len(self)==len(other)
            return SpecialList([other[i]*self[i] for i in range(len(self))])

    def __rmul__(self, other):
        return self*other

    def __truediv__(self, other):
        if isinstance(other,(int,float)):
            return SpecialList([i/other for i in self])
        elif isinstance(other,SpecialList):
            assert len(self)==len(other)
            return self * ~other

    def __floordiv__(self, other):
        if isinstance(other,(int,float)):
            return SpecialList([i//other for i in self])
        elif isinstance(other,SpecialList):
            assert len(self)==len(other)
            return SpecialList(list(map(math.floor,self/other)))

    def copy(self):
        return copy(self)


class Vector:

    def __init__(self, start: SpecialList, end: SpecialList):
        assert isinstance(start,SpecialList) and isinstance(end,SpecialList) \
               and len(start)==len(end)==2 and (end-start)!=[0,0]
        self.start = start
        self.end = end

    def __repr__(self):
        return f"(Start: {self.start}; End: {self.end}; Direction:{self.get_direction()})"

    def get_direction(self):
        return self.end - self.start

    def get_length(self):
        return np.sqrt(np.sum([i**2 for i in self.get_direction()]))

    def draw(self,canvas,size=3,color=(255,0,255)):
        return cv2.arrowedLine(canvas, (self.start[1],self.start[0]), (self.end[1],self.end[0]), color,size)  # however, the coordinate system is (y,x), though input requires a form of (x,y)
    def display(self,is_canvas=False,canvas=None,size=3,color=(255,0,255),name="",show_coordinates=False):
        if not is_canvas:
            canvas = np.zeros((max(self.start[1],self.end[1],1),max(self.start[0],self.end[0],1),3), np.uint8)
        canvas = self.draw(canvas,size,color)
        Image_Helper(img=canvas).show(name=name,show_coordinates=show_coordinates)

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.start,self.end+other.get_direction())
        else:
            raise TypeError(f"Bad type argument given: {type(other)}. Expected type is {type(self)}")

    def __sub__(self,other):
        if isinstance(other, Vector):
            return Vector(self.start+other.get_direction(),self.end)
        else:
            raise TypeError(f"Bad type argument given: {type(other)}. Expected type is {type(self)}")

    def __mul__(self, other):
        if isinstance(other,(int,float)):
            #  return Vector(self.start,self.end+SpecialList(list(map(int,self.get_direction()*(other-1))))) ## approximate coordinates for visualising
            if isinstance(other,float):
                print("Perfoming the Vector multiplication with a Float number. Pixels coordinates on an image "
                      "are to be an Int type.",file=sys.stderr)
            return Vector(self.start, self.end + self.get_direction() * (other - 1))
        elif isinstance(other, Vector):
            self_x,self_y = self.get_direction()
            other_x, other_y = other.get_direction()
            return self_x*other_x + self_y*other_y
        else:
            raise TypeError(f"Bad type argument given: {type(other)}. Expected type is int or float.")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            assert other != 0 and other > 0
            return self * (1/other)
        else:
            raise TypeError(f"Bad type argument given: {type(other)}. Expected type is int or float.")

    def __rmul__(self, other):
        return self*other

    def __neg__(self):
        return Vector(self.start,self.start-self.get_direction())

    def __eq__(self, other):
        if isinstance(other,Vector):
            return self.end==other.end and self.start == other.start
        else:
            raise TypeError("Can not compare Vector with non-Vector")

    def move_to_start(self):
        """Moves the vector to the beggining of the coordinate system, i.e. starting position is at (0,0)"""
        return Vector(self.start-self.start,self.end-self.start)


class VectorHelper:

    @staticmethod
    def get_angle(self:Vector,other:Vector,mode='degrees'):
        if isinstance(other,Vector) and isinstance(self,Vector):
            if mode=='degrees':
                return np.arccos(self*other/self.get_length()/other.get_length())*180/np.pi
            elif mode=='radians':
                return np.arccos(self * other / self.get_length() / other.get_length())
        else:
            raise TypeError("Arguments are expected to be a Vector type.")

    @staticmethod
    def get_line_eq_coeffs(vector: Vector):
        """Returns triplet (True,k,b) of equation y=kx+b, if the line is vertical, it returns (False,c), where x=c."""
        if isinstance(vector,Vector):
            if round(vector.start[1]-vector.end[1],ndigits=8)!=0:  # check if (x_1 - x_2) != 0
                k = (vector.start[0]-vector.end[0])/(vector.start[1]-vector.end[1])
                b = vector.start[0]-k*vector.start[1]
                return (True,k,b)
            else:
                return (False,vector.start[1])
        else:
            raise TypeError("Vector type is expected.")

    @staticmethod
    def move_line_to_point(triplet: tuple, vector:Vector):
        """Moves the given line to the end of a vector, so that vector touches the line with its end."""
        if isinstance(triplet,tuple) and isinstance(vector,Vector) and len(triplet) in [2,3]:
            if triplet[0]:
                k = triplet[1]
                b = vector.end[0]-vector.end[1]*k
                return (True,k,b)
            else:
                return (False,vector.end[1])

        else:
            raise TypeError("Arguments are expected to be tuple and Vector types respectively. Length of triplet must be"
                            "either 2 or 3.")

    @staticmethod
    def find_intersection(triplet1: tuple, triplet2: tuple):
        """Finds the (y,x) of intersection of two lines."""
        if triplet1[0] and triplet2[0]:
            _,k1,b1 = triplet1
            _, k2,b2 = triplet2
            x = (b2-b1)/(k1-k2)
            y = k1*x+b1
        elif triplet1[0] and not triplet2[0]:
            _,x = triplet2
            _, k1,b1 = triplet1
            y = k1*x+b1
        elif not triplet1[0] and triplet2[0]:
            _,x = triplet1
            _,k2,b2 = triplet2
            y = k2*x+b2
        else:
            raise ValueError("Two vertical lines do not have single point of intersection.")
        return (y,x)

    @classmethod
    def co_linearance(cls,vector1:Vector,vector2:Vector):
        """Returns True if vectors are colinear, i.e. vectors are parallel, otherwise returns False."""
        line1 = cls.get_line_eq_coeffs(vector1)
        line2 = cls.get_line_eq_coeffs(vector2)
        if (line1[0] and line2[0]) or (not line1[0] and not line2[0]):
            return np.round(line1[1]-line2[1],decimals=8)==0
        else:
            return False

    @classmethod
    def co_direction(cls,vector1:Vector,vector2:Vector):
        """Returns 1 if colinear vectors are co-directed. Returns -1 if colinear vectors are oppositely directed.
        Throws an error if vectors are not colinear."""
        if cls.co_linearance(vector1,vector2):
            vector1_direction = vector1.get_direction()
            vector2_direction = vector2.get_direction()
            res = np.sign(vector1_direction[0])==np.sign(vector2_direction[0]) \
                   and np.sign(vector1_direction[1])==np.sign(vector2_direction[1])
            return 1 if res else -1
        else:
            raise AssertionError("Vectors are not colinear.")
