import numpy as np
import sys
import cv2
from pathlib import Path

DIR_BASE = Path().absolute()
DIR_CONFIG_bgr = DIR_BASE / "config_images_bgr.txt"

DIR_TREES = DIR_BASE / "TREES"
DIR_DATASET = DIR_TREES


def getImages():
    """Function used to load images from given Directory
    """
    # Get all files from images directory
    PATHS = Path(DIR_DATASET).glob('**/*')
    
    FILES = [file for file in PATHS if file.is_file()]
    
    # Load all images from files and store them in a list    
    images = []
    for file in FILES:
        try:
            images.append(cv2.imread(str(file)))
        except:
            print(f"Failed to open {file}")
            sys.exit()
    
    print(f"Loaded {len(images)} images")
    return images

def createFile():
    try:
        file = open(DIR_CONFIG_bgr, 'w')
        return file
    except:
        print("Failed to create config file!")
        sys.exit()

def getDominantColor(img):
    """Function used to calculate dominant color from given image. Image must be <numpy.ndarray> type
    """
    colors, count = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

def getColorsOfImages(imgs):
    """Function returning dominant color values for each image in imgs
    """
    colors = []
    for img in imgs:
        colors.append(getDominantColor(img))
    
    return colors

images = getImages()

# images_bgrs = [[100, 100, 100], [100, 100, 200]]
images_bgrs = getColorsOfImages(images)


config_file = createFile()
# Write average bgr of each image into file
#config_file.write("\n".join(str(bgr) for bgr in images_bgrs))
# config_file.write("\n".join(images_bgrs))

np.savetxt(config_file, images_bgrs)

config_file.close()

# print(type(images_bgrs[0]))
# print(images_bgrs)
# print("-------------------")

# config_file = open(DIR_CONFIG_bgr, 'r')
# read_bgrs = list()
# for line in config_file:
#     #read_bgrs.append(np.array(line[:-1], dtype=np.int8))
#     read_bgrs.append(np.array(line[:-1]))

# print(read_bgrs)
# print(type(read_bgrs[0]))
# config_file.close()
