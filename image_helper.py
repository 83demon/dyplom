import cv2 as cv

class Image_Helper:

    def __init__(self,img=None,filepath=None,):
        if filepath:
            self.img = Image_Helper.read(filepath)
        elif img.any():
            self.img = img
        else:
            raise ValueError("No arguments were provided")

    @property
    def shape(self):
        return self.img.shape

    @staticmethod
    def read(filename):
        return cv.imread(filename)

    @staticmethod
    def crop(img, shape: tuple, bias: tuple = (0, 0)):
        "Crops first part of image of shape 'shape'"
        return img[bias[0]:bias[0] + shape[0], bias[1]:bias[1] + shape[1], :]

    def show(self, name=""):
        cv.imshow(name, self.img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def split(self,height_num,width_num):
        """Splits given image into height_num * width_num smaller images"""
        height_length = self.img.shape[0]//height_num
        width_length = self.img.shape[1]//width_num
        return [Image_Helper(img=self.img[i*height_length:(i+1)*height_length,j*width_length:(j+1)*width_length])
                for i in range(height_num) for j in range(width_num)]

    def find_center_mass(self):
        """Finds coordinates of the center of mass of a point in terms of color"""








