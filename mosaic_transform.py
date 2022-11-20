import os
import sys
import cv2
import random
import numpy as np

from pathlib import Path

# Constants
DIR_BASE = Path().absolute()
DIR_IMAGES = DIR_BASE / "images"
DIR_POKEMONS = DIR_BASE / "pokemon_dataset"
DIR_FLOWERS = DIR_BASE / "flowers"
DIR_TREES = DIR_BASE / "TREES"

DIR_DATASET = DIR_TREES
GRID_SIZE = 100

def getSample():
    """Function used to load sample image
    """
    # PATH_IMG = DIR_BASE / "dog_1.png"
    # PATH_IMG = DIR_BASE / "combo.png"
    PATH_IMG = DIR_BASE / "kfc.png"
    PATH_IMG = DIR_BASE / "Tree.jpg"
    
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

def pickSubImage(imgs, img, colors_list):
    """Function choosing image to replace original image based on dominant BGR color
    """
    # Calculate domiant RGB color of given image
    img_color = getDominantColor(img)

    # Get index of image with closes dominant BGR color
    closest_img_index = getClosestColor(colors_list, img_color)

    # Resize chosen image to match size of cell
    img = cv2.resize(imgs[closest_img_index], [img.shape[1], img.shape[0]])
    
    return img

def createImgFromCells(cells, org_img, N):
    """Function converting list of cells into image with corresponding shape as org_img
    """
    # Create yet empty Array to store reproduced Image
    img = np.empty_like(org_img)

    # List containing rows of new Image
    img_rows = []

    # Create rows of new Image and write them into a list
    for i in range(N):
        img_rows.append(np.concatenate(cells[(i*N):(i*N+N)], axis=1))

    # Transform rows into Image
    img = np.concatenate(img_rows)

    cv2.imshow("img new", img)
    return img

sample_img = getSample()

cv2.imshow("Original image", sample_img)
# sample_img = cv2.resize(sample_img, [600, 480])

images = getImages()
sample_cells = divideImage(sample_img, N=GRID_SIZE)

images_colors = getColorsOfImages(images)

# Epty list to store cells of new image
img_new_cells = []

for i,cell in enumerate(sample_cells):

    cell_color = getDominantColor(cell)
    # print(f"cell color {cell_color}")

    # Replace current cell with new Image
    img_new_cells.append(pickSubImage(images, cell, images_colors))

img_new = createImgFromCells(img_new_cells, sample_img, GRID_SIZE)

cv2.waitKey()
cv2.destroyAllWindows()