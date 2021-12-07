import numpy as np
import cv2 as cv
from tkinter import Tk
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox


class IrisDetection:
    def __init__(self, img_path1, img_path2):
        self.path_1 = img_path1
        self.original_1 = None
        self.img_loaded_1 = None
        self.img_gray_1 = None
        self.gaussian_1 = None
        self.img_edge_1 = None
        self.binary_1 = None
        self.path_2 = img_path2
        self.original_1 = None
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
        # Méthode appelée lors de la création d'un objet, renvoyant le tuple des valeurs
        # pour les iris et pupilles de chaque oeil
        self.read_img()
        self.img_2_gray()
        self.gaussian_filter()
        self.canny_img()
        self.hough_transform_pupil()
        self.detect_iris()
        self.sub_mask()
        self.display()
        return self.iris_1, self.iris_2, self.pupil_1, self.pupil_2

    def read_img(self):
        # Méthode pour lire une image et utiliser ses données par la suite
        self.img_loaded_1 = cv.imread(self.path_1)
        self.img_loaded_2 = cv.imread(self.path_2)
        self.original_1 = cv.imread(self.path_1)
        self.original_2 = cv.imread(self.path_2)

    def img_2_gray(self):
        # On convertit les images lues en gris
        self.img_gray_1 = cv.cvtColor(self.img_loaded_1, cv.COLOR_BGR2GRAY)
        self.img_gray_2 = cv.cvtColor(self.img_loaded_2, cv.COLOR_BGR2GRAY)

    def gaussian_filter(self):
        # On applique un filtre de Gauss
        self.gaussian_1 = cv.GaussianBlur(self.img_gray_1, (5, 5), 0)
        self.gaussian_2 = cv.GaussianBlur(self.img_gray_2, (5, 5), 0)

    def canny_img(self):
        # Filtre de Canny pour détecter les contours
        self.img_edge_1 = cv.Canny(
            image=self.gaussian_1, threshold1=60, threshold2=70)
        self.img_edge_2 = cv.Canny(
            image=self.gaussian_2, threshold1=60, threshold2=70)

    def hough_transform_pupil(self):
        # Les pixels au dessus de 150 deviennent blancs
        ret, self.binary_1 = cv.threshold(
            self.img_edge_1, 100, 255, cv.THRESH_BINARY)
        # On utilise le contour TREE afin de lister tous les contours avec un ordre hierarchique
        # contours, hierarchy = cv.findContours(self.binary_1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Possible de dessiner les contours avec cv.drawContours(img, contours, -1, (0,255,0), 3)

        # Image en 8 bits, methode, dp, minDist entre les centres
        circles = cv.HoughCircles(self.binary_1, cv.HOUGH_GRADIENT, 1, self.gaussian_1.shape[0],
                                  param1=200, param2=1,
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
                self.pupil_1 = (center[0], center[1], radius)
                cv.circle(self.img_loaded_1, center, radius, (255, 0, 255), 3)
        cv.imwrite("pupil1.png", self.img_loaded_1)

        # Image 2

        ret, self.binary_2 = cv.threshold(
            self.img_edge_2, 150, 255, cv.THRESH_BINARY)

        circles = cv.HoughCircles(self.binary_2, cv.HOUGH_GRADIENT, 1, self.gaussian_2.shape[0],
                                  param1=200, param2=1,
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
                self.pupil_2 = (center[0], center[1], radius)
                cv.circle(self.img_loaded_2, center, radius, (255, 0, 255), 3)
        cv.imwrite("pupil2.png", self.img_loaded_2)

    def detect_iris(self):
        # La distance minimum entre deux cercles est donnée par le rayon de la pupille * 8 afin d'éviter
        # d'autres cercles
        circles = cv.HoughCircles(self.binary_1, cv.HOUGH_GRADIENT, 1, self.pupil_1[2] * 8, param1=200, param2=1,
                                  minRadius=80, maxRadius=110)
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
                cv.circle(self.img_loaded_1, center, radius, (255, 0, 255), 3)
                self.iris_1 = (center[0], center[1], radius)

        # Image 2

        circles = cv.HoughCircles(self.binary_2, cv.HOUGH_GRADIENT, 1, self.pupil_2[2] * 8, param1=200, param2=1,
                                  minRadius=80, maxRadius=110)
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
                cv.circle(self.img_loaded_2, center, radius, (255, 0, 255), 3)
                self.iris_2 = (center[0], center[1], radius)

    def sub_mask(self):
        # On va supprimer la pupille et l'extérieur de l'iris,
        # on créé un fichier contenant le résultat du détourage pour chaque image

        # Image 1
        mask_final_1 = cv.subtract(self.mask_2, self.mask_1)
        result1 = cv.cvtColor(self.img_loaded_1, cv.COLOR_BGR2BGRA)
        result1[:, :, 3] = mask_final_1[:, :, 0]
        cv.imwrite("mask_1.png", result1)

        # Image 2
        mask_final_2 = cv.subtract(self.mask_4, self.mask_3)
        result2 = cv.cvtColor(self.img_loaded_2, cv.COLOR_BGR2BGRA)
        result2[:, :, 3] = mask_final_2[:, :, 0]
        cv.imwrite("mask_2.png", result2)

    def display(self):
        numpy_concat_grey = np.concatenate(
            (self.img_gray_1, self.img_gray_2), axis=1)
        numpy_concat_gauss = np.concatenate(
            (self.gaussian_1, self.gaussian_2), axis=1)
        numpy_concat_canny = np.concatenate(
            (self.img_edge_1, self.img_edge_2), axis=1)
        numpy_concat_binary = np.concatenate(
            (self.binary_1, self.binary_2), axis=1)

        pupil1 = cv.imread("pupil1.png")
        pupil2 = cv.imread("pupil2.png")
        numpy_concat_pupils = np.concatenate(
            (pupil1, pupil2), axis=1)
        numpy_concat_iris = np.concatenate(
            (self.img_loaded_1, self.img_loaded_2), axis=1)


        ver1 = np.concatenate(
            (numpy_concat_grey, numpy_concat_gauss, numpy_concat_canny, numpy_concat_binary), axis=0)
        ver2 = np.concatenate((numpy_concat_pupils, numpy_concat_iris), axis=0)
        # vertical = np.column_stack((ver1, ver2))
        cv.imshow("Images grises -> Gauss -> Canny -> binaires", ver1)
        cv.imshow("Images pupilles -> iris", ver2)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)


# Pour la création d'un masque des paupières voir https://books.google.be/books?id=SI29AgAAQBAJ&pg=PA149&lpg=PA149&dq=iris+recon+matlab&source=bl&ots=Ae633czDjg&sig=ACfU3U3wu6AIhpf8zz56CKe0WL6yP-YAug&hl=fr&sa=X&ved=2ahUKEwjQucmpyrT0AhXP3KQKHS_OBOEQ6AF6BAgUEAM#v=onepage&q=iris%20recon%20matlab&f=false


class IrisRecognition():
    def __init__(self, mask1, mask2, iris_1, iris_2, pupil_1, pupil_2):
        self.mask_1 = cv.imread(mask1, cv.IMREAD_UNCHANGED)
        self.mask_2 = cv.imread(mask2, cv.IMREAD_UNCHANGED)
        self.iris_1 = iris_1
        self.iris_2 = iris_2
        self.pupil_1 = pupil_1
        self.pupil_2 = pupil_2
        self.polar_1 = None
        self.polar_2 = None

    def start(self):
        self.transparency()
        self.crop()
        self.normalisation()
        self.display()
        self.get_keypoints()

    def crop(self):
        # Méthode pour détourer le vide inutile à l'extérieur de l'iris,
        # il enregistre une image détourée pour chaque oeil
        # (1) Convertit en gris et création d'une limite
        gray = cv.cvtColor(self.mask_1, cv.COLOR_BGR2GRAY)
        th, threshed = cv.threshold(gray, 240, 255, cv.THRESH_BINARY_INV)

        # (2) Morph-op pour supprimer le bruit
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
        morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

        # (3) Cherche la zone du contour MAX
        cnts = cv.findContours(morphed, cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)[-2]
        cnt = sorted(cnts, key=cv.contourArea)[-1]

        # (4) Détoure et enregistre
        x, y, w, h = cv.boundingRect(cnt)
        dst = self.mask_1[y:y + h, x:x + w]
        cv.imwrite("crop_1.png", dst)

        # (1) Convertit en gris et création d'une limite
        gray = cv.cvtColor(self.mask_2, cv.COLOR_BGR2GRAY)
        th, threshed = cv.threshold(gray, 240, 255, cv.THRESH_BINARY_INV)

        # (2) Morph-op pour supprimer le bruit
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
        morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

        # (3) Cherche la zone du contour MAX
        cnts = cv.findContours(morphed, cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)[-2]
        cnt = sorted(cnts, key=cv.contourArea)[-1]

        # (4) Détoure et enregistre
        x, y, w, h = cv.boundingRect(cnt)
        dst = self.mask_2[y:y + h, x:x + w]
        cv.imwrite("crop_2.png", dst)

    def transparency(self):
        trans_mask = self.mask_1[:, :, 3] == 0
        self.mask_1[trans_mask] = [255, 255, 255, 0]

        trans_mask = self.mask_2[:, :, 3] == 0
        self.mask_2[trans_mask] = [255, 255, 255, 0]

    def normalisation(self):
        img_crop1 = cv.imread('crop_1.png')

        polar_img = cv.warpPolar(
            img_crop1, (256, 1024), (img_crop1.shape[0] / 2,
                                     img_crop1.shape[1] / 2), self.iris_1[2] * 2,
            cv.WARP_POLAR_LINEAR)
        polar_img = cv.rotate(polar_img, cv.ROTATE_90_COUNTERCLOCKWISE)

        polar_img = polar_img[int(polar_img.shape[0] / 2): polar_img.shape[0], 0: polar_img.shape[1]]
        polar_img = cv.cvtColor(polar_img, cv.COLOR_BGR2GRAY)
        self.polar_1 = polar_img

        cv.imwrite("foreground.png", polar_img)

        img_crop2 = cv.imread('crop_2.png')

        polar_img2 = cv.warpPolar(
            img_crop2, (256, 1024), (img_crop2.shape[0] / 2,
                                     img_crop2.shape[1] / 2), self.iris_2[2] * 2,
            cv.WARP_POLAR_LINEAR)
        polar_img2 = cv.rotate(polar_img2, cv.ROTATE_90_COUNTERCLOCKWISE)

        # crop image
        polar_img2 = polar_img2[int(
            polar_img2.shape[0] / 2): polar_img2.shape[0], 0: polar_img2.shape[1]]
        polar_img2 = cv.cvtColor(polar_img2, cv.COLOR_BGR2GRAY)
        self.polar_2 = polar_img2

        cv.imwrite("foreground2.png", polar_img2)

    def get_keypoints(self):
        img = cv.imread('foreground.png')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        alg = cv.AKAZE_create()
        (kp, desc) = alg.detectAndCompute(gray, None)

        img2 = cv.imread('foreground2.png')
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        (kp2, desc2) = alg.detectAndCompute(gray2, None)

        bf = cv.BFMatcher(cv.NORM_HAMMING)
        matches = bf.knnMatch(desc, desc2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        if len(good) >= 40:
            print('\033[96m' + "Les iris sont pareilles, avec un score de " + str(len(good)))
            return True
        else:
            print("\033[91m" + "Les iris sont différentes, avec un score de " + str(len(good)))
            return False

    def display(self):
        numpy_concat_vertical = np.concatenate(
            (self.polar_1, self.polar_2), axis=0)

        cv.imshow("Images polarisees", numpy_concat_vertical)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)


if __name__ == "__main__":
    Tk().withdraw()
    tk.messagebox.showinfo(title="Iris Recognition", message="Choisissez un premier iris")
    filename_1 = askopenfilename(title="Choisir un iris 1")

    tk.messagebox.showinfo(title="Iris Recognition", message="Choisissez un second iris")
    filename_2 = askopenfilename(title="Choisir un iris 2")
    data = IrisDetection(filename_1, filename_2).start()
    IrisRecognition('mask_1.png', 'mask_2.png',
                    data[0], data[1], data[2], data[3]).start()
