import os
import sys
import cv2
import re
import numpy as np
import time

from pathlib import Path
from scipy import spatial

# Constants
DIR_BASE = Path().absolute()
DIR_DATASET = DIR_BASE / "Database"

TILE_SIZE = 40

def getSample(img_name: str) -> "np.ndarray":
    """Function used to load sample image.

    Args:
        img_name: name of the image to be mosaiced. Image must be present in the same directory as script
    
    Returns:
        img: BGR image to be mosaiced
    """

    PATH_IMG = DIR_BASE / img_name
    
    try:
        img = cv2.imread(str(PATH_IMG))
        return img
    except:
        print(f"Failed to open {PATH_IMG}")
        sys.exit()

def getImages(DIR_DATASET: str) -> list[np.ndarray]:
    """Function used to load images used to replace original image.

    Args:
        DIR_DATASET: path to directory with images

    Returns:
        images: list containing all of the images inside DIR_DATASET
    """

    # Get all files from images directory
    PATHS = Path(DIR_DATASET).glob('**/*')
    
    FILES = [file for file in PATHS if file.is_file()]
    
    # Load all images from files and store them in a list    
    images = []
    for file in FILES:
        try:
            images.append(cv2.imread(str(file)))
        except ValueError:
            print(f"Failed to open {file}")
            sys.exit()
    
    print(f"Loaded {len(images)} images")
    return images

def resizeImage(img: np.ndarray, RATIO: int) -> np.ndarray:
    """Function resizing image by specified ratio.

    Args:
        img: image to be resized
        RATIO: size of orignal image in percents

    Returns:
        Resized image
    """

    dimensions = (int(img.shape[1] * RATIO/100), int(img.shape[0] * RATIO/100))
    return cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)

def getClosestImage(colors_list: list[np.ndarray], color: list[int], TREE: spatial.KDTree) -> int:
    """Function picking image with the closest dominant color to given pixel

    Args:
        colors_list: list containing dominant color of every image in dataset
        color: list with BGR values of the pixel
        TREE: scipy.spatial.KDTree

    Returns:
        Index of picked image
    """
    _, closest  = TREE.query(color)

    return closest

def pickSubImage(   imgs: list[np.ndarray],
                    pixel_color: list[int],
                    colors_list: list[np.ndarray],
                    TREE: spatial.KDTree,
                    TILE_SIZE: int) -> np.ndarray:
    """Function picking image to replace pixel based on dominant BGR color. Uses KDTree to pick.

    Args:
        imgs: list of images from dataset
        pixel_color: list containing BGR values of the pixel
        colors_list: list containing dominant color of every image in dataset
        TREE: scipy.spatial.KDTree
        TILE_SIZE: size of replacing image(tile)

    Returns:
        img_chosen: approprietly resized image replacing a pixel
    """

    closest_img_index = getClosestImage(colors_list, pixel_color, TREE)

    # Resize chosen image to match size of cell
    img_chosen = cv2.resize(imgs[closest_img_index], [TILE_SIZE, TILE_SIZE])
    
    return img_chosen

def pickSubImageMSE(imgs: list[np.ndarray],
                    pixel_color: list[int],
                    colors_list: list[np.ndarray],
                    TILE_SIZE: int) -> np.ndarray:
    """Function picking image to replace pixel based on dominant BGR color. Uses MSE to pick.

    Args:
        imgs: list of images from dataset
        pixel_color: list containing BGR values of the pixel
        colors_list: list containing dominant color of every image in dataset
        TILE_SIZE: size of replacing image(tile)

    Returns:
        img_chosen: approprietly resized image replacing a pixel
    """
    errors = []
    for color_in_list in colors_list:
        errors.append((int((color_in_list[0] - pixel_color[0]))**2 + int((color_in_list[1] - pixel_color[1]))**2 + int((color_in_list[2] - pixel_color[2]))**2)**(0.5))
    
    min_err = min(errors)
    min_err_index = errors.index(min_err)

    # Resize chosen image to match size of cell
    img_chosen = cv2.resize(imgs[min_err_index], [TILE_SIZE, TILE_SIZE])

    return img_chosen

def createImgFromCells(cells: list[np.ndarray], org_img_dims: tuple[int], TILE_SIZE: int) -> np.ndarray:
    """Function converting list of cells into image with corresponding shape as org_img

    Args:
        cells: list contating images replacing pixels
        org_img_dims: dimensions of original image as (width, height)
        TILE_SIZE: size of replacing image(tile)

    Returns:
        img: mosaiced image
    """
    # Create yet empty Array to store reproduced Image
    # img = np.empty_like(org_img)
    img = np.zeros((org_img_dims[0]*TILE_SIZE, org_img_dims[1]*TILE_SIZE))

    # List containing rows of new Image
    img_rows = []

    # Create rows of new Image and write them into a list
    for i in range(org_img_dims[1]):
        # img_rows.append(np.concatenate(cells[(i*N):(i*N+N)], axis=1))
        img_rows.append(np.concatenate(cells[(i*org_img_dims[0]):(i*org_img_dims[0]+org_img_dims[0])], axis=1))
        
    # Transform rows into Image
    img = np.concatenate(img_rows)

    cv2.imshow("Mosaic image", img)
    return img

def getDominantColor(imgs: list[np.ndarray]) -> list[np.ndarray]:
    """Function calculating dominant color of provided images.

    Args:
        imgs: list of images from the dataset

    Returns:
        colors: list of np.ndarray containing dominant color of imgs
    """
    colors = list()
    
    for img in imgs:
        colors.append(np.array(img).mean(axis=0).mean(axis=0))

    return colors

def main(IMG_NAME, RESIZE_RATIO, FLAG_SAVE_IMAGE, FLAG_USE_MSE):
    # Save start time
    time_start = time.perf_counter()

    # Load and display sample to be mosaiced
    sample_img = getSample(IMG_NAME)
    cv2.imshow("Original image", sample_img)

    # Load images and calculate their dominant colors
    images = getImages(DIR_DATASET)
    images_colors = getDominantColor(images)

    # Resize the sample accrodingly
    sample_img = resizeImage(sample_img, RESIZE_RATIO)
    image_dims = (sample_img.shape[1], sample_img.shape[0])

    if not FLAG_USE_MSE:
        # Define KDTree object
        TREE = spatial.KDTree(images_colors)

    # Empty list to store photos replacing pixels
    img_new_cells = list()

    # Iterate through image
    for i in range(0,image_dims[1]):
        for j in range(0,image_dims[0]):
            # Get values of cuurent pixel
            [b, g, r] = sample_img[i][j]
            color = [b, g, r]
            
            # Pick image replacing pixel
            if FLAG_USE_MSE:
                img_new_cells.append(pickSubImageMSE(images, color, images_colors, TILE_SIZE))
            else:
                img_new_cells.append(pickSubImage(images, color, images_colors, TREE, TILE_SIZE))

    # Produce a new image
    img_new = createImgFromCells(img_new_cells, image_dims, TILE_SIZE)

    print(f"Mosaicing an image took {time.perf_counter() - time_start} seconds")

    if FLAG_SAVE_IMAGE:
        img_mosaic_name = re.sub(r'\.', '_mosaic.', IMG_NAME)
        cv2.imwrite(img_mosaic_name, img_new)

    cv2.waitKey()
    cv2.destroyAllWindows()