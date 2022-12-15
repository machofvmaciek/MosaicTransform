# This is a wrapper script for mosaic_transform.py. It validates and parses arguments.
import sys
import argparse
import mosaic_transform

FLAG_SAVE_IMAGE = False
FLAG_USE_MSE = False
RESIZE_RATIO_DEFAULT = 15

parser = argparse.ArgumentParser(description="Create a mosaic from given image")

parser.add_argument("img_name", type=str, metavar="image name", help="name of the image to be mosaiced. Example: 'dog.png'")
parser.add_argument("-r" ,"--resize-ratio", type=int, metavar="", help="size of orignal image in percents. Default value is 20")

parser.add_argument("-mse", "--use-mse", action="store_true", help="use MSE to pick image")
parser.add_argument("-s", "--save-image", action="store_true", help="save mosaiced image into a file")

args = parser.parse_args()

def run_mosaic_transform(img_name, RESIZE_RATIO, FLAG_SAVE_IMAGE, FLAG_USE_MSE):
    
    # Run mosaic_transform.py with provided arguments
    mosaic_transform.main(img_name, RESIZE_RATIO, FLAG_SAVE_IMAGE, FLAG_USE_MSE)
    sys.exit()

if __name__ == '__main__':
    # Process cli arguments
    if not args.img_name:
        sys.print_help()
        print("Specify name of the image to be mosaiced")
    else:
        img_name = args.img_name

    if args.resize_ratio:
        RESIZE_RATIO = args.resize_ratio
    else:
        RESIZE_RATIO = RESIZE_RATIO_DEFAULT

    if args.save_image:
        FLAG_SAVE_IMAGE = True

    if args.use_mse:
        FLAG_USE_MSE = True
    
    run_mosaic_transform(img_name, RESIZE_RATIO, FLAG_SAVE_IMAGE, FLAG_USE_MSE)
