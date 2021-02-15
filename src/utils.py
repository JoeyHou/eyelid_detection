import cv2
from imutils import face_utils
import imutils
import dlib

from os import listdir

import math
import numpy as np
import pandas as pd
from scipy import optimize


# Define text info
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

def predict_landmarks(image, predictor):
    """
    Predicting the landmarks given a image filepath and a predictor object
    nput:
        - image          : image object cv2.imread
        - predictor      : predictor object
    return:
        - image          : output image
        - shape          : list of landmark locations
    """

    face_detector = dlib.get_frontal_face_detector()
    output = image.copy()

    # image = imutils.resize(image, width = 250)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    rects = face_detector(gray, 1)
    if len(rects) == 0:
        rects = dlib.rectangles()
        h, w = image.shape[0], image.shape[1]
        rec = dlib.rectangle(0, 0, w, h)
        rects.append(rec)

    shape = 0
    # loop over the face detections
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # shape = shape[36:48, :]

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(output, (x, y), 3, (0, 0, 255), -1)
    # return cv2.cvtColor(output, cv2.COLOR_BGR2RGB), shape
    return output, shape
    # return image, shape

def detect_eye(image, shape):
    # print(shape)
    left_x = [i[0] for i in shape[:6]]
    left_y = [i[1] for i in shape[:6]]
    left_size = int((max(left_x) - min(left_x)) * 2.5)
    left_top = int(np.mean(left_y) - left_size / 2)
    left_left = int((max(left_x) + min(left_x) - left_size) / 2)
    left_eye = image[left_top: left_top + left_size, \
                     left_left: left_left + left_size]

    right_x = [i[0] for i in shape[6:]]
    right_y = [i[1] for i in shape[6:]]
    right_size = int((max(right_x) - min(right_x)) * 2.5)
    right_top = int(np.mean(right_y) - right_size / 2)
    right_left = int((max(right_x) + min(right_x) - right_size) / 2)
    right_eye = image[right_top: right_top + right_size, \
                     right_left: right_left + right_size]

    return left_eye, right_eye


# def detect_eye(image, filename, haarcascade_path):
#     '''
#     input:
#         image: input image (in RGB)
#         filename: name of the file
#     output:
#         img1: image segmentation of one of an eye (type: np.ndarray)
#         img2: image segmentation of another eye (type: np.ndarray)
#         outpue: original image with the rectangle plotted on the segmented areas
#     '''
#     # 1. Readin the image and return if there is an error
#     # print('=================================')
#     # print('Now Processing:', filename + '...')
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     output = gray.copy()
#
#     # 2. Processing image - detect eyes using CascadeClassifier
#     gray = cv2.medianBlur(gray, 7)
#
#     eye_cascade = cv2.CascadeClassifier(haarcascade_path + 'haarcascade_eye.xml')
#     eyes = eye_cascade.detectMultiScale(gray)
#     if len(eyes) < 2:
#         # print('[Error!] Stupit classifier!!!')
#         return -1, -1, -1
#     # Pick the two "eyes" with largest sizes
#     eyes = sorted(eyes, key = lambda x: x[2] + x[3], reverse = True)[:2]
#     # Sort the "eyes" by their x position
#     eyes = sorted(eyes, key = lambda x: x[0], reverse = False)
#     # 3. Crop the original image and return the two eye segments
#     result = []
#     for (ex,ey,ew,eh) in eyes:
#         eye = output[ey-25:ey + eh + 25, ex - 25:ex + ew + 25]
#         result.append(eye)
#
#     # print('Done processing:', filename + '!')
#     return result[0], result[1], output


def find_best_circle(x, y):
    """
    find the best circle given the list of points
    """
    method_2 = "leastsq"

    x_m, y_m = np.mean(x), np.mean(y)
    def calc_R(xc, yc):
        """
        calculate the distance of each 2D points from the center (xc, yc)
        """
        return ((x-xc)**2 + (y-yc)**2) ** (1/2)

    def f_2(c):
        """
        calculate the algebraic distance between the data points and
        the mean circle centered at c=(xc, yc)
        """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2       = calc_R(*center_2)
    R_2        = Ri_2.mean()
    residu_2   = sum((Ri_2 - R_2)**2)

    return xc_2, yc_2, R_2

def measure_single_image(image, predictor):
    """
    input:
        - image          : the image
        - predictor      : predictor object
    return:
        - output         : Output image
        - (xc1, y_lower) : Lower intercection
        - (xc1, yc1)     : Circle center
        - (xc1, y_upper) : Upper intercection
    """

    # Check if image file exists
    # if filepath not in listdir(tesing_img_path):
    #     print('[ERROR] File not found! Current file:', tesing_img_path + filepath)
    #     return None
    # print(tesing_img_path + filepath)

    # Call landmark predictor
    output, shape = predict_landmarks(image, predictor)

    # Find the best fit circle of the iris
    circle_point_x = shape[(1, 3, 5, 7), 0]
    circle_point_y = shape[(1, 3, 5, 7), 1]
    xc1, yc1, r1 = find_best_circle(circle_point_x, circle_point_y)
    xc1, yc1, r1 = int(xc1), int(yc1), int(r1)
    cv2.circle(output, (xc1, yc1), r1, (0, 255, 255), 1)

    # Find the best fir curve of the upper eyelid
    upper_point_x = shape[(1, 2, 3), 0]
    upper_point_y = shape[(1, 2, 3), 1]
    xc2, yc2, r2 = find_best_circle(upper_point_x, upper_point_y)
    xc2, yc2, r2 = int(xc2), int(yc2), int(r2)
    y_upper = yc2 - (r2 ** 2 - (xc1 - xc2) ** 2) ** (1 / 2)
    if y_upper < 0:
        y_upper = 0
    # elif y_upper > yc1:
    #     y_upper = yc1
    else:
        y_upper = int(y_upper)
    upper_in_mm = round(11.65 * (yc1 - y_upper) / (r1 * 2), 2)
    cv2.putText(output, str(upper_in_mm), (xc1 + 20, y_upper - 20), font, fontScale, fontColor, lineType)
    cv2.arrowedLine(output, (xc1, yc1), (xc1, y_upper), (0, 255, 255), 2)

    # Find the best fir curve of the lower eyelid
    lower_point_x = shape[(4, 6, 0), 0]
    lower_point_y = shape[(4, 6, 0), 1]
    xc3, yc3, r3 = find_best_circle(lower_point_x, lower_point_y)
    xc3, yc3, r3 = int(xc3), int(yc3), int(r3)
    y_lower = yc3 + (r3 ** 2 - (xc1 - xc3) ** 2) ** (1 / 2)
    if y_lower <= yc1:
        y_lower = yc1
    elif y_lower >= output.shape[0]:
        y_lower = yc1 + r1
    else:
        y_lower = int(y_lower)
    lower_in_mm = round(11.65 * (y_lower - yc1) / (r1 * 2), 2)
    cv2.putText(output, str(lower_in_mm), (xc1 + 20, y_lower + 20), font, fontScale, fontColor, lineType)
    cv2.arrowedLine(output, (xc1, yc1), (xc1, y_lower), (0, 255, 255), 2)

    # Adding a point at the center
    cv2.circle(output, (xc1, yc1), 2, (0, 0, 255), 2)
    # plt.imshow(output)
    # plt.title(filepath)
    # plt.show()
    return output, (xc1, y_lower), (xc1, yc1), (xc1, y_upper), r1
