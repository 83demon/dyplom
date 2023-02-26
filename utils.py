import cv2
import numpy as np
import math

class Image_Helper:
    color_to_channel_map = {2: "Red", 1: "Green", 0: "Blue"}

    def __init__(self, img=None, filepath=None, ):
        if filepath:
            self.img = Image_Helper.read(filepath)
        elif img.any():
            self.img = img
        else:
            raise ValueError("No arguments were provided")
        self.active_channel = None  # active channel to display

    def display_coordinates(self, event, x, y, flags, params):
        if event == 2:
            if not isinstance(self.active_channel, int):
                print(f"Coordinates: {x, y}; RGB is : {(self.img[y, x, 2], self.img[y, x, 1], self.img[y, x, 0])}")
            else:
                print(
                    f"Coordinates: {x, y}; Color is {Image_Helper.color_to_channel_map[self.active_channel]} : {self.img[y, x, self.active_channel]}")

    @property
    def shape(self):
        return self.img.shape

    @staticmethod
    def read(filename):
        return cv2.imread(filename)

    def crop(self, shape: tuple, bias: tuple = (0, 0)):
        "Crops first part of image of shape 'shape'"
        return Image_Helper(img=self.img[bias[0]:bias[0] + shape[0], bias[1]:bias[1] + shape[1], :])

    def show(self, name="", show_coordinates=False, channel=None):
        self.active_channel = channel
        if show_coordinates:
            cv2.namedWindow(name)
            cv2.setMouseCallback(name, self.display_coordinates)
        cv2.imshow(name, self.img[:, :, channel] if isinstance(channel, int) else self.img)
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


class Vector:

    def __init__(self, start: SpecialList, end: SpecialList):
        assert isinstance(start,SpecialList) and isinstance(end,SpecialList) \
               and len(start)==len(end)==2
        self.start = start
        self.end = end

    def __repr__(self):
        return f"Vector's start is {self.start}, end is {self.end}"

    def get_direction(self):
        return self.end - self.start

    def get_length(self):
        return np.sqrt(np.sum(i**2 for i in self.get_direction()))

    def draw(self,canvas,size=3,color=(255,0,255)):
        cv2.arrowedLine(canvas, self.start, self.end, color,size)  # translation (y,x) -> (x,y)
    def display(self,is_canvas=False,canvas=None,size=3,color=(255,0,255)):
        if not is_canvas:
            canvas = np.zeros((max(self.start[1],self.end[1],1),max(self.start[0],self.end[0],1),3), np.uint8)
        self.draw(canvas,color,size)
        Image_Helper(img=canvas).show("Vector")

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
            return Vector(self.start,self.end+self.get_direction()*(other-1))
        else:
            raise TypeError(f"Bad type argument given: {type(other)}. Expected type is int or float.")

    def __rmul__(self, other):
        return self*other

    def __neg__(self):
        return Vector(self.start,self.start-self.get_direction())
