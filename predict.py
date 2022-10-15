
from PIL import Image
import PIL
import cv2
import numpy as np

from yolo import YOLO
import os
import argparse 
from utils.tif import save_tile
from utils.mid import mid
parser = argparse.ArgumentParser(description='inputpath and outputpath')
parser.add_argument('-pt', '--project_type')
parser.add_argument('-mp', '--modelpth')
parser.add_argument('--input')
parser.add_argument('--output')
args = parser.parse_args()
impath=args.input
outpath = args.output
if  not os.path.exists(outpath):
    os.mkdir(outpath)
if  not os.path.exists(os.path.join(outpath,'vis')):
    os.mkdir(os.path.join(outpath,'vis'))

yolo = YOLO(args.modelpth, outpath)

if args.project_type == 'small':
    for file in os.listdir(os.path.join(impath)):
        img=Image.open(os.path.join(impath,file))
        imaged=yolo.detect_image(img,file[:-4])
        imaged.save(os.path.join(outpath,'vis',file))

elif args.project_type == 'big':
    save_tile(impath, outpath, yolo)

elif args.project_type == 'mid':
    for file in os.listdir(os.path.join(impath)):
        mid(os.path.join(impath,file), outpath, yolo)
else:
    print("erro")

    

    
    
   