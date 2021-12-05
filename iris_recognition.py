import numpy as np
import cv2 as cv

global pupil_1
global pupil_2


class IrisDetection:
    def __init__(self, img_path1, img_path2):
        self.path_1 = img_path1
        self.img_loaded_1 = None
        self.img_gray_1 = None
        self.gaussian_1 = None
        self.img_edge_1 = None
        self.binary_1 = None
        self.path_2 = img_path2
        self.img_loaded_2 = None
        self.img_gray_2 = None
        self.gaussian_2 = None
        self.img_edge_2 = None
        self.binary_2 = None
        self.pupil_1 = None
        self.pupil_2 = None
        self.mask_1 = None
        self.mask_2 = None
        self.mask_3 = None
        self.mask_4 = None
        self.iris_1 = None
        self.iris_2 = None

    def start(self):
        self.read_img()
        self.img_2_gray()
        self.gaussian_filter()
        self.canny_img()
        self.hough_transform_pupil()
        self.detect_iris()
        self.sub_mask()
        # self.write_txt()
        return self.iris_1, self.iris_2

    def read_img(self):
        self.img_loaded_1 = cv.imread(self.path_1)
        self.img_loaded_2 = cv.imread(self.path_2)
        numpy_horizontal_concat = np.concatenate(
            (self.img_loaded_1, self.img_loaded_2), axis=1)
        """
        cv.imshow('Images originales', numpy_horizontal_concat)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)
        """

    def img_2_gray(self):
        self.img_gray_1 = cv.cvtColor(self.img_loaded_1, cv.COLOR_BGR2GRAY)
        self.img_gray_2 = cv.cvtColor(self.img_loaded_2, cv.COLOR_BGR2GRAY)
        numpy_horizontal_concat = np.concatenate(
            (self.img_gray_1, self.img_gray_2), axis=1)
        """
        cv.imshow('Images Gris', numpy_horizontal_concat)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)
        """

    def gaussian_filter(self):
        self.gaussian_1 = cv.GaussianBlur(self.img_gray_1, (9, 9), 0)
        self.gaussian_2 = cv.GaussianBlur(self.img_gray_2, (9, 9), 0)
        numpy_horizontal_concat = np.concatenate(
            (self.gaussian_1, self.gaussian_2), axis=1)
        """
        cv.imshow('Images Gauss', numpy_horizontal_concat)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)
        """

    def canny_img(self):
        self.img_edge_1 = cv.Canny(
            image=self.gaussian_1, threshold1=50, threshold2=100)
        self.img_edge_2 = cv.Canny(
            image=self.gaussian_2, threshold1=50, threshold2=100)
        numpy_horizontal_concat = np.concatenate(
            (self.img_edge_1, self.img_edge_2), axis=1)
        """
        cv.imshow('Images Canny', numpy_horizontal_concat)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)
        """

    def hough_transform_pupil(self):
        # Les pixels au dessus de 150 deviennent blancs
        ret, self.binary_1 = cv.threshold(
            self.img_edge_1, 150, 255, cv.THRESH_BINARY)
        # On utilise le contour TREE afin de lister tous les contours avec un ordre hierarchique
        # contours, hierarchy = cv.findContours(self.binary_1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Possible de dessiner les contours avec cv.drawContours(img, contours, -1, (0,255,0), 3)
        # Image en 8 bits, methode, dp, minDist entre les centres
        circles = cv.HoughCircles(self.binary_1, cv.HOUGH_GRADIENT, 1, self.gaussian_1.shape[0],
                                  param1=100, param2=30,
                                  minRadius=1, maxRadius=65)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle outline
                radius = i[2]
                self.mask_1 = np.zeros_like(self.img_loaded_1)
                self.mask_1 = cv.circle(
                    self.mask_1, center, radius, (255, 255, 255), -1)
                #cv.circle(self.img_loaded_1, center, radius, (255, 0, 255), 3)
                self.pupil_1 = (center[0], center[1], radius)

        # Image 2

        ret, self.binary_2 = cv.threshold(
            self.img_edge_2, 150, 255, cv.THRESH_BINARY)

        circles = cv.HoughCircles(self.binary_2, cv.HOUGH_GRADIENT, 1, self.gaussian_2.shape[0],
                                  param1=100, param2=30,
                                  minRadius=1, maxRadius=65)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle outline
                radius = i[2]
                self.mask_3 = np.zeros_like(self.img_loaded_1)
                self.mask_3 = cv.circle(
                    self.mask_3, center, radius, (255, 255, 255), -1)
                #cv.circle(self.img_loaded_2, center, radius, (255, 0, 255), 3)
                self.pupil_2 = (center[0], center[1], radius)

        numpy_horizontal_concat = np.concatenate(
            (self.img_loaded_1, self.img_loaded_2), axis=1)
        """
        cv.imshow('Images Pupille', numpy_horizontal_concat)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)
        """

    def detect_iris(self):
        # La distance minimum est donnée par le double du rayon de la pupille ( donc son diamètre ) afin d'éviter d'autres cercles
        circles = cv.HoughCircles(self.binary_1, cv.HOUGH_GRADIENT, 1, self.pupil_1[2] * 2, param1=100, param2=30,
                                  minRadius=50, maxRadius=120)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # On indique que le centre du nouveau cercle doit être celui de la pupille
                center = (self.pupil_1[0], self.pupil_1[1])
                radius = i[2]
                # Retourne un nouvel array rempli de zero
                self.mask_2 = np.zeros_like(self.img_loaded_1)
                self.mask_2 = cv.circle(
                    self.mask_2, center, radius, (255, 255, 255), -1)
                #cv.circle(self.img_loaded_1, center, radius, (255, 0, 255), 3)
                self.iris_1 = (center[0], center[1], radius)

        # Image 2

        circles = cv.HoughCircles(self.binary_2, cv.HOUGH_GRADIENT, 1, self.pupil_2[2] * 2, param1=100, param2=30,
                                  minRadius=50, maxRadius=120)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # On indique que le centre du nouveau cercle doit être celui de la pupille
                center = (self.pupil_2[0], self.pupil_2[1])
                radius = i[2]
                # Retourne un nouvel array rempli de zero
                self.mask_4 = np.zeros_like(self.img_loaded_2)
                self.mask_4 = cv.circle(
                    self.mask_4, center, radius, (255, 255, 255), -1)
                #cv.circle(self.img_loaded_2, center, radius, (255, 0, 255), 3)
                self.iris_2 = (center[0], center[1], radius)

        numpy_horizontal_concat = np.concatenate(
            (self.img_loaded_1, self.img_loaded_2), axis=1)
        """
        cv.imshow('Images Iris', numpy_horizontal_concat)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)
        """

    def sub_mask(self):
        mask_final_1 = cv.subtract(self.mask_2, self.mask_1)
        result1 = cv.cvtColor(self.img_loaded_1, cv.COLOR_BGR2BGRA)
        result1[:, :, 3] = mask_final_1[:, :, 0]
        cv.imwrite("mask_1.png", result1)

        # Image 2
        mask_final_2 = cv.subtract(self.mask_4, self.mask_3)
        result2 = cv.cvtColor(self.img_loaded_2, cv.COLOR_BGR2BGRA)
        result2[:, :, 3] = mask_final_2[:, :, 0]
        cv.imwrite("mask_2.png", result2)

    def write_txt(self):
        print(self.iris_1, self.iris_2)
        f = open("iris.txt", "w")
        f.write(str(self.iris_1))
        f.close()
        f = open("iris.txt", "a")
        f.write(str(self.iris_2))
        f.close()


# Pour la création d'un masque des paupières voir https://books.google.be/books?id=SI29AgAAQBAJ&pg=PA149&lpg=PA149&dq=iris+recon+matlab&source=bl&ots=Ae633czDjg&sig=ACfU3U3wu6AIhpf8zz56CKe0WL6yP-YAug&hl=fr&sa=X&ved=2ahUKEwjQucmpyrT0AhXP3KQKHS_OBOEQ6AF6BAgUEAM#v=onepage&q=iris%20recon%20matlab&f=false
class IrisRecognition():
    def __init__(self, mask1, mask2, iris_1, iris_2):
        self.mask_1 = cv.imread(mask1, cv.IMREAD_UNCHANGED)
        self.mask_2 = cv.imread(mask2, cv.IMREAD_UNCHANGED)
        self.iris_1 = iris_1
        self.iris_2 = iris_2

    def start(self):
        self.transparency()
        self.crop()
        self.normalisation()

    def crop(self):
        # (1) Convert to gray, and threshold
        gray = cv.cvtColor(self.mask_1, cv.COLOR_BGR2GRAY)
        th, threshed = cv.threshold(gray, 240, 255, cv.THRESH_BINARY_INV)

        # (2) Morph-op to remove noise
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
        morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

        # (3) Find the max-area contour
        cnts = cv.findContours(morphed, cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)[-2]
        cnt = sorted(cnts, key=cv.contourArea)[-1]

        # (4) Crop and save it
        x, y, w, h = cv.boundingRect(cnt)
        dst = self.mask_1[y:y+h, x:x+w]
        cv.imwrite("crop_1.png", dst)

        # (1) Convert to gray, and threshold
        gray = cv.cvtColor(self.mask_2, cv.COLOR_BGR2GRAY)
        th, threshed = cv.threshold(gray, 240, 255, cv.THRESH_BINARY_INV)

        # (2) Morph-op to remove noise
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
        morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

        # (3) Find the max-area contour
        cnts = cv.findContours(morphed, cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)[-2]
        cnt = sorted(cnts, key=cv.contourArea)[-1]

        # (4) Crop and save it
        x, y, w, h = cv.boundingRect(cnt)
        dst = self.mask_2[y:y+h, x:x+w]
        cv.imwrite("crop_2.png", dst)

    def transparency(self):
        trans_mask = self.mask_1[:, :, 3] == 0
        self.mask_1[trans_mask] = [255, 255, 255, 0]
        #cv.imwrite("test alpha.png", self.mask_1)

        trans_mask = self.mask_2[:, :, 3] == 0
        self.mask_2[trans_mask] = [255, 255, 255, 0]
        #cv.imwrite("test alpha2.png", self.mask_2)

    def normalisation(self):
        # Do the polar rotation along 1024 angular steps with a radius of 256 pixels.
        img_crop1 = cv.imread('crop_1.png')
        polar_img = cv.warpPolar(
            img_crop1, (256, 1024), (self.iris_1[0], self.iris_1[1]), self.iris_1[2] * 2, cv.WARP_POLAR_LINEAR)
        # Rotate it sideways to be more visually pleasing
        polar_img = cv.rotate(polar_img, cv.ROTATE_90_COUNTERCLOCKWISE)

        # crop image
        polar_img = polar_img[int(polar_img.shape[0] / 2)
                                  : polar_img.shape[0], 0: polar_img.shape[1]]
        polar_img = cv.cvtColor(polar_img, cv.COLOR_BGR2GRAY)

        _, threshold = cv.threshold(polar_img, 100, 255, cv.THRESH_BINARY)
        cv.imwrite("foreground.png", threshold)
        cv.imshow('threshold', threshold)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)

        # Do the polar rotation along 1024 angular steps with a radius of 256 pixels.
        img_crop2 = cv.imread('crop_2.png')
        polar_img2 = cv.warpPolar(
            img_crop2, (256, 1024), (self.iris_2[0], self.iris_2[1]), self.iris_2[2] * 2, cv.WARP_POLAR_LINEAR)
        # Rotate it sideways to be more visually pleasing
        polar_img2 = cv.rotate(polar_img2, cv.ROTATE_90_COUNTERCLOCKWISE)

        # crop image
        polar_img2 = polar_img2[int(
            polar_img2.shape[0] / 2): polar_img2.shape[0], 0: polar_img2.shape[1]]
        polar_img2 = cv.cvtColor(polar_img2, cv.COLOR_BGR2GRAY)

        _, threshold2 = cv.threshold(polar_img2, 100, 255, cv.THRESH_BINARY)
        cv.imwrite("foreground2.png", threshold2)
        cv.imshow('threshold 2', threshold2)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)

        """
        bbox = cv.boundingRect(threshold)
        x, y, w, h = bbox
        foreground = polar_img
        cv.imwrite("foreground.png", foreground)
        """


if __name__ == "__main__":
    test = IrisDetection('database/002/01.bmp', 'database/001/01.bmp').start()
    IrisRecognition('mask_1.png', 'mask_2.png', test[0], test[1]).start()
