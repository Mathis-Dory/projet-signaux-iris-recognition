import numpy
import numpy as np
import cv2 as cv


class IrisRecognition:
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
        self.decoupe_1 = None
        self.decoupe_2 = None

    def start(self):
        self.read_img()
        self.img_2_gray()
        self.gaussian_filter()
        self.canny_img()
        self.hough_transform_pupil()
        self.detect_iris()
        self.sub_mask()
        self.eyelid_detection()

    def read_img(self):
        self.img_loaded_1 = cv.imread(self.path_1)
        self.img_loaded_2 = cv.imread(self.path_2)
        numpy_horizontal_concat = np.concatenate((self.img_loaded_1, self.img_loaded_2), axis=1)
        cv.imshow('Images originales', numpy_horizontal_concat)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)

    def img_2_gray(self):
        self.img_gray_1 = cv.cvtColor(self.img_loaded_1, cv.COLOR_BGR2GRAY)
        self.img_gray_2 = cv.cvtColor(self.img_loaded_2, cv.COLOR_BGR2GRAY)
        numpy_horizontal_concat = np.concatenate((self.img_gray_1, self.img_gray_2), axis=1)
        cv.imshow('Images Gris', numpy_horizontal_concat)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)

    def gaussian_filter(self):
        self.gaussian_1 = cv.GaussianBlur(self.img_gray_1, (9, 9), 0)
        self.gaussian_2 = cv.GaussianBlur(self.img_gray_2, (9, 9), 0)
        numpy_horizontal_concat = np.concatenate((self.gaussian_1, self.gaussian_2), axis=1)
        cv.imshow('Images Gauss', numpy_horizontal_concat)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)


    def canny_img(self):
        self.img_edge_1 = cv.Canny(image=self.gaussian_1, threshold1=50, threshold2=100)
        self.img_edge_2 = cv.Canny(image=self.gaussian_2, threshold1=50, threshold2=100)
        numpy_horizontal_concat = np.concatenate((self.img_edge_1, self.img_edge_2), axis=1)
        cv.imshow('Images Canny', numpy_horizontal_concat)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)


    def hough_transform_pupil(self):
        # Les pixels au dessus de 150 deviennent blancs
        ret, self.binary_1 = cv.threshold(self.img_edge_1, 150, 255, cv.THRESH_BINARY)
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
                self.mask_1 = numpy.zeros_like(self.img_loaded_1)
                self.mask_1 = cv.circle(self.mask_1, center, radius, (255, 255, 255), -1)
                cv.circle(self.img_loaded_1, center, radius, (255, 0, 255), 3)
                self.pupil_1 = (center[0], center[1], radius)


        # Image 2

        ret, self.binary_2 = cv.threshold(self.img_edge_2, 150, 255, cv.THRESH_BINARY)

        circles = cv.HoughCircles(self.binary_2, cv.HOUGH_GRADIENT, 1, self.gaussian_2.shape[0],
                                  param1=100, param2=30,
                                  minRadius=1, maxRadius=65)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle outline
                radius = i[2]
                self.mask_3 = numpy.zeros_like(self.img_loaded_1)
                self.mask_3 = cv.circle(self.mask_3, center, radius, (255, 255, 255), -1)
                cv.circle(self.img_loaded_2, center, radius, (255, 0, 255), 3)
                self.pupil_2 = (center[0], center[1], radius)

        numpy_horizontal_concat = np.concatenate((self.img_loaded_1, self.img_loaded_2), axis=1)
        cv.imshow('Images Pupille', numpy_horizontal_concat)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)

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
                self.mask_2 = numpy.zeros_like(self.img_loaded_1)
                self.mask_2 = cv.circle(self.mask_2, center, radius, (255, 255, 255), -1)
                cv.circle(self.img_loaded_1, center, radius, (255, 0, 255), 3)

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
                self.mask_4 = numpy.zeros_like(self.img_loaded_2)
                self.mask_4 = cv.circle(self.mask_4, center, radius, (255, 255, 255), -1)
                cv.circle(self.img_loaded_2, center, radius, (255, 0, 255), 3)

        numpy_horizontal_concat = np.concatenate((self.img_loaded_1, self.img_loaded_2), axis=1)
        cv.imshow('Images Iris', numpy_horizontal_concat)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)

    def sub_mask(self):
        mask_final_1 = cv.subtract(self.mask_2, self.mask_1)
        result1 = cv.cvtColor(self.img_loaded_1, cv.COLOR_BGR2BGRA)
        result1[:, :, 3] = mask_final_1[:, :, 0]
        cv.imwrite("mask_1.png", result1)
        self.decoupe_1 = cv.imread("mask_1.png")

        # Image 2

        mask_final_2 = cv.subtract(self.mask_4, self.mask_3)
        result2 = cv.cvtColor(self.img_loaded_2, cv.COLOR_BGR2BGRA)
        result2[:, :, 3] = mask_final_2[:, :, 0]
        cv.imwrite("mask_2.png", result2)
        self.decoupe_2 = cv.imread("mask_2.png")

    # TODO
    def eyelid_detection(self):
        self.img_loaded_1 = cv.imread('mask_1.png')
        self.img_loaded_2 = cv.imread('mask_2.png')
        self.img_gray_1 = cv.cvtColor(self.img_loaded_1, cv.COLOR_BGR2GRAY)
        self.img_gray_2 = cv.cvtColor(self.img_loaded_2, cv.COLOR_BGR2GRAY)
        self.img_edge_1 = cv.Canny(image=self.img_gray_1, threshold1=50, threshold2=100)
        self.img_edge_2 = cv.Canny(image=self.img_gray_2, threshold1=50, threshold2=100)

        #lines = cv.HoughLines(self., 1, np.pi / 180, 150, None, 0, 0)




IrisRecognition('database/002/01.bmp', 'database/001/01.bmp').start()
