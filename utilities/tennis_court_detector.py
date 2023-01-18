# import the necessary packages
import numpy as np
import cv2
#import opencv_wrapper as cvw

# import  poly_point_isect as bot

# construct the argument parse and parse the arguments
# load the image
# image = cv2.imread("tennis.jpg")
frame_count = 1
image_path = "C:\\Users\\miche\\Documents\\GitHub\\sports_object_detection\\output" + f"/frame{frame_count}.jpg"
image = cv2.imread(image_path)
# define the list of boundaries
boundaries = [([180, 180, 100], [255, 255, 255])]
thresh=180

# # loop over the boundaries
# for (lower, upper) in boundaries:
#     # create NumPy arrays from the boundaries
#     lower = np.array(lower, dtype="uint8")
#     upper = np.array(upper, dtype="uint8")

#     # find the colors within the specified boundaries and apply
#     # the mask
#     mask = cv2.inRange(image, lower, upper)
#     output = cv2.bitwise_and(image, image, mask=mask)

# Start my code
#gray = cvw.bgr2gray(output)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_ret, binary_gray = cv2.threshold(gray, 200, 200, cv2.THRESH_BINARY)
cv2.imwrite("test.jpg", binary_gray)
binary_gray = cv2.imread("test.jpg")
cv2.imshow("detected thresh", binary_gray)
corners = cv2.cornerHarris(gray, 9, 3, 0.01)

corners_norm = np.empty(corners.shape, dtype=np.float32)
cv2.normalize(corners, corners_norm, alpha=0, beta=255)
corners_norm_scaled = cv2.convertScaleAbs(corners_norm)

# Drawing a circle around corners
for i in range(corners_norm.shape[0]):
    for j in range(corners_norm.shape[1]):
        if int(corners_norm[i,j]) > thresh:
            cv2.circle(corners_norm_scaled, (j,i), 5, (0), 2)

# thresh = cvw.threshold_otsu(corners)
# dilated = cvw.dilate(thresh, 3)

# contours = cvw.find_external_contours(dilated)

# for contour in contours:
#     cv2.circle(image, contour.center, 3, cvw.Color.RED, -1)

cv2.imshow("Image", image)
cv2.waitKey(0)