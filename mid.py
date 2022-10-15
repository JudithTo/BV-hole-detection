import os
import numpy
from utils.tif import nms
from PIL import Image
import numpy as np
import glob
import torch
import cv2
#中图模式下的检测
def mid(file_path, outpath, yolo):
    name = os.path.basename(file_path).split('.')[0]
    img=Image.open(file_path)
    img = np.array(img)
    height,width,d = img.shape
    x_mins = np.arange(0, width, 800) 
    y_mins = np.arange(0, height, 800)
    for  i,x_min in enumerate(x_mins) :
        for j, y_min in enumerate(y_mins) :
            x_max = x_min+1000 if x_min+1000<width else width
            y_max = y_min+1000 if y_min+1000<height else height
            tile_data = img[y_min:y_max, x_min:x_max]
            w,h,_ = tile_data.shape
            if h!=1000 or w!=1000:
                im_pad = np.ones((1000,1000,3),dtype ='int8')
                im_pad = im_pad*255
                im_pad[0:w,0:h,0:3]=tile_data
                tile_data = im_pad    
            name_tile = name+'_'+str(i)+'_'+str(j)
            tile_data = Image.fromarray(np.uint8(tile_data))
            imaged=yolo.detect_image(tile_data,name_tile)
            imaged.save(os.path.join(outpath+'/vis/',name_tile+'.png'))
    txtpath = glob.glob(os.path.join(outpath,'detection-results_data',name)+'*.txt')
    conflist =[]
    xminlist =[]
    yminlist =[]
    xmaxlist =[]
    ymaxlist =[]
    for txt in txtpath:
        with open(txt) as f:
            for line in f.readlines():
                line =line.split()
                confi=float(line[1])
                if confi<0.001:
                    continue
                file = os.path.basename(txt)
            
                xx = int(file[:-4].split('_')[-2])
                yy = int(file[:-4].split('_')[-1])
                xmin = int(line[2]) if int(line[2])>=0 else 0
                ymin = int(line[3]) if int(line[3])>=0 else 0
                xmax = int(line[4]) if int(line[4])<1000 else 999
                ymax = int(line[5]) if int(line[5])<1000 else 999
                xmin = xx*800+xmin
                ymin = yy *800+ymin
                xmax = xx*800+xmax
                ymax = yy*800+ymax
                conflist.append(confi)
                xminlist.append(xmin)
                yminlist.append(ymin)
                xmaxlist.append(xmax)
                ymaxlist.append(ymax)
    xmin= torch.from_numpy(np.array(xminlist))
    ymin = torch.from_numpy(np.array(yminlist))
    xmax = torch.from_numpy(np.array(xmaxlist))
    ymax = torch.from_numpy(np.array(ymaxlist))
    score = torch.from_numpy(np.array(conflist))
    keep=nms(xmin,ymin,xmax,ymax,score)
    xmin = np.array(xmin[keep])
    ymin = np.array(ymin[keep])
    xmax = np.array(xmax[keep])
    ymax = np.array(ymax[keep])
    score = np.array(score[keep])
    with open(os.path.join(outpath,'detection.txt'),'w') as f:
        for i in range(len(score)):
            print(xmin[i])
            label = '{} {:.2f}'.format('hole', score[i])
            cv2.rectangle(img,(xmin[i],ymin[i]),(xmax[i],ymax[i]),(0,0,255),3)
            f.write("hole"+" "+str(score[i])+" "+str(xmin[i])+" "+str(ymin[i])+" "+str(xmax[i])+" "+str(ymax[i])+'\n')
    cv2.imwrite(os.path.join(outpath,'detection.png'),img)