import os
import sys
import cv2
import random
import numpy as np

from pathlib import Path

DIR_BASE = Path().absolute()
DIR_IMAGES = DIR_BASE / "images"

def getSample():
    """Function used to load sample image
    """
    PATH_IMG = DIR_BASE / "dog_1.png"
    
    try:
        img = cv2.imread(str(PATH_IMG))
        return img
    except:
        print(f"Failed to open {PATH_IMG}")
        sys.exit()
    

def getImages():
    """Function used to load images used to replace original image
    """
    # Get all files from images directory
    PATHS = Path(DIR_IMAGES).glob('**/*')
    
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

def divideImage(img, N=20):
    """Function dividing input image into N^2 cells
    """
    # Get image size
    img_height, img_width, _ = img.shape
    
    # Make sure image is divisible into N^2 cells
    if (img_height % N != 0) & (img_width % N != 0):
        print(f"Image with dimensions [{img_height}, {img_width}] cannot be divided equally into {N**2} cells")
        sys.exit()
    
    # Calculate cell dimensions
    cell_height, cell_width = int(img_height / N), int(img_width / N)

    # Divide sample into N^2 cells
    cells = []
    for i in range(N):
        for j in range(N):
            cells.append(img[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width])

    print(f"Sample image was divided into {len(cells)} cells")

    return cells

def getClosestColor(colors_list, color):
    colors_list = np.array(colors_list)
    color = np.array(color)
    distances = np.sqrt(np.sum((colors_list - color)**2, axis=1))

    index_of_smallest_dist = np.where(distances==np.amin(distances))
    smallest_dist = colors_list[index_of_smallest_dist]
    smallest_dist = smallest_dist[0]
    # print(smallest_dist[::-1])
    
    return int(index_of_smallest_dist[0]), smallest_dist[::-1]

sample_img = getSample()

cv2.imshow("Original image", sample_img)
# sample_img = cv2.resize(sample_img, [600, 480])

images = getImages()
sample_cells = divideImage(sample_img)

images_colors = getColorsOfImages(images)

print(f"images color outside func{images_colors}")


for i,cell in enumerate(sample_cells):
    cell_color = getDominantColor(cell)
    print(f"cell color {cell_color}")
    cv2.imshow("cell", cell)

    # closest_color_index, closest_color = list(getClosestColor(images_colors, cell_color))
    closest_color_index, closest_color = getClosestColor(images_colors, cell_color)
    print(f"closest color to cell {closest_color}")
    print(f"index of closest color to cell {closest_color_index}")
    print("----------")
    cv2.imshow("dopasowanie", images[closest_color_index])
    print(images_colors[closest_color_index])


    # closest_color_index = images_colors.index(closest_color)
    # print(closest_color_index)



    if i==0:
        break



# # print(images_colors)
# for color in images_colors:
#     print(color)
#     closest_color = getClosestColor(images_colors, color)
#     print("--------------")

# print(images[index_closest_color])

cv2.waitKey()
cv2.destroyAllWindows()

"""
cropped = img[start_row:end_row, start_col:end_col]
"""