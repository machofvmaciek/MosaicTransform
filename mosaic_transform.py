# Import libraries
import os
import sys
#import glob # do sciezek
import cv2
import random
import numpy as np

import config_images

# Paths to images
SAMPLE_IMAGE_PATH = "/Users/machofv/projects/MosaicTransform/dog_0.png"
#SAMPLE_IMAGE_PATH = "/Users/machofv/projects/MosaicTransform/B-35_template.png"

# Constants
REPLACING_IMAGE_SIZE = 20

def generateRanges(N=20):
    """Function generating Hue Ranges - dividing 0-179 Hue values equally
    """
    # Generate HSV Hue groups(ranges) in list
    HUE_AREAS = []
    for i in range(9):
        HUE_AREAS.append(np.linspace(i*20, i*20+19, 20, True, False, int, axis=0))

    return HUE_AREAS
def findClosestRange(pixel, HUE_AREAS):
    """Function returning one of specified Hue areas
    """
    HUE_AREA_INDEX = 0
    
    # Get pixel Hue value
    pixel_hue = pixel[0]

    # Find in which Hue area is pixel
    for area in HUE_AREAS:
        if pixel_hue in area:
            #print(f"pixel {pixel} jest w {area}")
            return HUE_AREA_INDEX
        HUE_AREA_INDEX = HUE_AREA_INDEX + 1

def replacePixel(area):
    """Function replacing pixel with image in specified Hue range
    """
    # Get list with paths to all images in specified Hue range
    templates_in_area = config_images.TEMPLATES[area]

    # Randomize image in range, load it
    PATH_IMG_CHOSEN = random.choice(templates_in_area)
    img = cv2.imread(str(PATH_IMG_CHOSEN))

    return img
    
# Load sample image, find its dimensions
sample_img = cv2.imread(SAMPLE_IMAGE_PATH)
sample_img = cv2.resize(sample_img, [60, 60])
sample_height, sample_width, _ = sample_img.shape

cv2.imshow("Pies", sample_img)

# Prepare empty array in which result image(mosaiced) will be stored
result_img = sample_img.copy()
result_img = cv2.resize(result_img, [sample_height*REPLACING_IMAGE_SIZE, sample_width*REPLACING_IMAGE_SIZE])

# Convert sample image to HSV representation
sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2HSV)
cv2.imshow("Pies HSV", sample_img)

#print(findClosestRange(sample_img[0][0]))

# Prepare Hue Ranges
HUE_AREAS = generateRanges()

# Iterate over whole image
for i in range(sample_height):
    for j in range(sample_width):
        sample_pixel_area = findClosestRange(sample_img[i][j], HUE_AREAS)
        image_substitute = replacePixel(sample_pixel_area)
        for ii in range(REPLACING_IMAGE_SIZE):
            for jj in range(REPLACING_IMAGE_SIZE):
                result_img[i*REPLACING_IMAGE_SIZE+ii][j*REPLACING_IMAGE_SIZE+jj] = image_substitute[ii][jj]
        

# sample_pixel_area = findClosestRange(sample_img[0][0], HUE_AREAS)
# image_substitute = replacePixel(sample_pixel_area)        
# for ii in range(REPLACING_IMAGE_SIZE):
#     for jj in range(REPLACING_IMAGE_SIZE):
#         result_img[0*REPLACING_IMAGE_SIZE+ii][0*REPLACING_IMAGE_SIZE+jj] = image_substitute[ii][jj]

cv2.imshow("result hsv", result_img)
cv2.imwrite("pies_result.png", result_img)
cv2.imshow("result bgr", cv2.cvtColor(result_img, cv2.COLOR_HSV2BGR))
cv2.imwrite("pies_resultbgr.png", result_img)

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