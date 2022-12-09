import os
import sys
import cv2
import random
import numpy as np

from pathlib import Path
from scipy import spatial

# Constants
DIR_BASE = Path().absolute()
DIR_IMAGES = DIR_BASE / "images"
DIR_POKEMONS = DIR_BASE / "pokemon_dataset"
DIR_FLOWERS = DIR_BASE / "flowers"
DIR_TREES = DIR_BASE / "Trees"
DIR_BUTTERFLIES = DIR_BASE / "butterflies"
DIR_PEOPLES = DIR_BASE / "Peoples"
DIR_DOGS = DIR_BASE / "dogs"
DIR_COMBO = DIR_BASE / "Database"

DIR_DATASET = DIR_COMBO
GRID_SIZE = 60
RESIZE_RATIO = 20

def getSample():
    """Function used to load sample image
    """
    # PATH_IMG = DIR_BASE / "dog_1.png"
    # PATH_IMG = DIR_BASE / "combo1.png"
    # PATH_IMG = DIR_BASE / "kfc.png"
    PATH_IMG = DIR_BASE / "macos_wallpaper.png"
    # PATH_IMG = DIR_BASE / "butterfly.jpg"
    # PATH_IMG = DIR_BASE / "tree_synt.png"
    # PATH_IMG = DIR_BASE / "tony1.png"
    # PATH_IMG = DIR_BASE / "minecraft_steve.png"
    
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

def resizeImage(img, RATIO):
    dimensions = (int(img.shape[1] * RATIO/100), int(img.shape[0] * RATIO/100))
    return cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)

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
    """Function returning index and dominant color of image with closest dominant color
    """
    colors_list = np.array(colors_list)
    
    color = np.array(color)
    distances = np.sqrt(np.sum((colors_list - color)**2, axis=1))

    index_of_smallest_dist = np.where(distances==np.amin(distances))
    # print(index_of_smallest_dist[0][0])
    # print(f"index_of_smallest_dist {int(index_of_smallest_dist[0])}")
    
    smallest_dist = colors_list[index_of_smallest_dist]
    smallest_dist = smallest_dist[0]
    # print(f"smallest dist {smallest_dist}")
    
    # return int(index_of_smallest_dist[0][0])
    return index_of_smallest_dist[0][0]
def getClosestColorMine(colors_list, color):
    """Function returning index and dominant color of image with closest dominant color
    """
    errors = []
    for color_in_list in colors_list:
        errors.append((int((color_in_list[0] - color[0]))**2 + int((color_in_list[1] - color[1]))**2 + int((color_in_list[2] - color[2]))**2)**(0.5))
        # errors.append(((color_in_list[0] - color [0])/2)**2 + ((color_in_list[1] - color [1])/2)**2 + ((color_in_list[2] - color [2])/2)**2)
    
    min_err = min(errors)
    min_err_index = errors.index(min_err)

    return min_err_index

def getClosestImage(colors_list, color, TREE) :
    closest = TREE.query(color)

    return closest[1]

def pickSubImage(imgs, img_color, colors_list, TREE):
    """Function choosing image to replace original image based on dominant BGR color
    """
    # Get index of image with closes dominant BGR color
    # closest_img_index = getClosestColorMine(colors_list, img_color)
    
    closest_img_index = getClosestImage(colors_list, img_color, TREE)

    # Check if 
    # Resize chosen image to match size of cell
    img_chosen = cv2.resize(imgs[closest_img_index], [40, 40])
    
    return img_chosen

def createImgFromCells(cells, org_img, org_img_dims):
    """Function converting list of cells into image with corresponding shape as org_img
    """
    # Create yet empty Array to store reproduced Image
    # img = np.empty_like(org_img)
    img = np.zeros((org_img_dims[0]*40, org_img_dims[1]*40))

    # List containing rows of new Image
    img_rows = []

    # Create rows of new Image and write them into a list
    for i in range(org_img_dims[1]):
        # img_rows.append(np.concatenate(cells[(i*N):(i*N+N)], axis=1))
        img_rows.append(np.concatenate(cells[(i*org_img_dims[0]):(i*org_img_dims[0]+org_img_dims[0])], axis=1))
        
    print(len(img_rows))
    print(len(img_rows[0]))
    # Transform rows into Image
    img = np.concatenate(img_rows)

    cv2.imshow("Mosaic image", img)
    return img


sample_img = getSample()

cv2.imshow("Original image", sample_img)
# sample_img = cv2.resize(sample_img, [600, 480])

sample_img = resizeImage(sample_img, RESIZE_RATIO)
cv2.imshow("resized", sample_img)

# Load images and get its colors
images = getImages()
images_colors = getColorsOfImages(images)

# Define KDTree object
TREE = spatial.KDTree(images_colors)

# Epty list to store cells of new image
img_new_cells = list()
images_used = list()

image_dims = (sample_img.shape[1], sample_img.shape[0])

imax = image_dims[0]-1
jmax = image_dims[1]-1

for i in range(0,image_dims[1]):
    for j in range(0,image_dims[0]):
        [b, g, r] = sample_img[i][j]
        color = [b, g, r]
        # zmienic zeby zwracal tez liste images_used
        img_new_cells.append(pickSubImage(images, color, images_colors, images_used, TREE))
# for j in range(jmax):
#     for i in range(imax):
#         print(i, j)
#         [b, g, r] = sample_img[i][j]
#         color = [b, g, r]
#         img_new_cells.append(pickSubImage(images, color, images_colors, None))
        

# cv2.imshow("chosen img", img_new_cells[0])
# print(len(img_new_cells))
img_new = createImgFromCells(img_new_cells, sample_img, image_dims)
print(img_new.shape)
cv2.imwrite("treemosaic.png", img_new)

# color = getDominantColor(sample_img[100])
# print(color)
# 
# sample_cells = divideImage(sample_img, N=GRID_SIZE)

# images_colors = getColorsOfImages(images)





# for i,cell in enumerate(sample_cells):

#     cell_color = getDominantColor(cell)
#     # print(f"cell color {cell_color}")

#     # Replace current cell with new Image
#     img_new_cells.append(pickSubImage(images, cell, images_colors, TREE))


cv2.waitKey()
cv2.destroyAllWindows()
"""
dims =  (width, height)
        (szerokosc, wysokosc)
"""