# import the necessary packages
import argparse
import cv2
from os import listdir

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
all_points = []
refPt = (0, 0)

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        all_points.append(refPt)
        # print(refPt)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread("../../data/processed/left/left_" + args["image"] + '.jpg')
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

h, w = image.shape[0], image.shape[1]

file_path = 'left_' + args["image"] + '.jpg'
out_put_string = ("  <image file='%s'> \n\
        <box top='0' left='0' width='%d' height='%d'> \n"% (file_path, w, h))

# keep looping until the 'q' key is pressed
while True:

    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # plot points as they are clicked
    for point_circle in all_points:
        cv2.circle(image, point_circle, radius=3, color=(0, 255, 0), thickness=-1)

    point_idx = 1

    if len(all_points) == 8:
        for point in all_points:
          out_put_string += "      <part name='%d' x='%d' y='%d'/>\n"%(point_idx, point[0], point[1])
          point_idx += 1
        out_put_string += "    </box>\n  </image>\n"
        # close all open windows
        cv2.destroyAllWindows()
        print(out_put_string)
        break