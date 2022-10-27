# Import libraries
import os
import sys
#import glob # do sciezek
import cv2
import numpy as np

def findClosestRange(pixel):
    """Function returning one of specified Hue areas
    """
    HUE_AREA_INDEX = 0
    # Generate HSV Hue groups(ranges) in list
    HUE_AREAS = []
    for i in range(10):
        HUE_AREAS.append(np.linspace(i*20, i*20+19, 20, True, False, int, axis=0))
    
    # Get pixel Hue value
    pixel_hue = pixel[0]

    # Find in which Hue area is pixel
    for area in HUE_AREAS:
        if pixel_hue in area:
            #print(f"pixel {pixel} jest w {area}")
            return HUE_AREA_INDEX
        HUE_AREA_INDEX = HUE_AREA_INDEX + 1

def replacePixel(area, x, y):
    """Function replacing pixel with image in specified Hue range
    """


# Paths to images
SAMPLE_IMAGE_PATH = "/Users/machofv/projects/MosaicTransform/dog_0.png"

# Constants
REPLACING_IMAGE_SIZE = 30

# Load sample image, find its dimensions
sample_img = cv2.imread(SAMPLE_IMAGE_PATH)
sample_height, sample_width, _ = sample_img.shape

cv2.imshow("Pies", sample_img)

# Prepare empty array in which result image(mosaiced) will be stored
result_img = np.array(sample_height*REPLACING_IMAGE_SIZE, sample_width*REPLACING_IMAGE_SIZE)

# Convert sample image to HSV representation
sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2HSV)
cv2.imshow("Pies HSV", sample_img)

#print(findClosestRange(sample_img[0][0]))

# Iterate over whole image
for i in range(sample_height):
    for j in range(sample_width):
        sample_pixel_area = findClosestRange(sample_img[i][j])
        image_replacing = replacePixel(sample_pixel_area)
        result_img[i][j] = image_replacing
        

        

cv2.waitKey()
cv2.destroyAllWindows()

"""HSV Representation
    OpenCV format   = H (0-180) ; S (0-255) ; V (0-255)
"""

"""TO DO:
replacePixel():
    - przyjmuje numer area
    - z listy PATHow z danego area loso wybiera jeden PATH
    - laduje PATH do obrazka
    zwraca: 
    - obrazek z Hue area


"""