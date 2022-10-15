import rasterio as rio
from rasterio import windows
from osgeo import ogr,osr
import numpy as np
from rasterio.enums import Resampling
from PIL import Image
import os
import torch
from shapely import geometry
import geopandas as gpd
def imagexy2geo(trans, col, row):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''
    px = trans.a*col + row * trans.b + trans.c
    py = trans.d* col + trans.e*row + trans.f

    return px, py

def nms(x1,y1,x2,y2, scores, threshold=0.1):
        # x1 = bboxes[:,0]
        # y1 = bboxes[:,1]
        # x2 = bboxes[:,2]
        # y2 = bboxes[:,3]

        x1= torch.tensor(x1,dtype=torch.float32)
        y1= torch.tensor(y1,dtype=torch.float32)
        x2= torch.tensor(x2,dtype=torch.float32)
        y2= torch.tensor(y2,dtype=torch.float32)
        areas = (x2-x1)*(y2-y1) 
         # [N,] 每个bbox的面积
        _, order = scores.sort(0, descending=True)    # 降序排列


        keep = []
        while order.numel() > 0:       # torch.numel()返回张量元素个数
            if order.numel() == 1:     # 保留框只剩一个
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()    # 保留scores最大的那个框box[i]
                keep.append(i)

            # 计算box[i]与其余各框的IOU(思路很好)
            xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]
            iou = inter / torch.tensor((areas[i]+areas[order[1:]]-inter),dtype=torch.float32)  # [N-1,]
            idx = (iou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
            if idx.numel() == 0:
                break
            order = order[idx+1]  # 修补索引之间的差值
        return keep
def tiletxt2txt(trans,touying, outpath):
    conflist =[]
    xminlist =[]
    yminlist =[]
    xmaxlist =[]
    ymaxlist =[]
    for file in os.listdir(os.path.join(outpath,'detection-results_data/')):
        with open(os.path.join(outpath,'detection-results_data/',file)) as f:
            for line in f.readlines():
                line =line.split()
                confi=float(line[1])
                if confi<0.001:
                    continue
                print(confi)
                xx = int(file[:-4].split('_')[0])
                yy = int(file[:-4].split('_')[1])
                print(xx)
                print(yy)
                xmin = int(line[2]) if int(line[2])>=0 else 0
                ymin = int(line[3]) if int(line[3])>=0 else 0
                xmax = int(line[4]) if int(line[4])<1000 else 999
                ymax = int(line[5]) if int(line[5])<1000 else 999
                xmin1 = xx*800+xmin
                ymin1 = yy *800+ymin
                xmax1 = xx*800+xmax
                ymax1 = yy*800+ymax
                conflist.append(confi)
                xminlist.append(xmin1)
                yminlist.append(ymin1)
                xmaxlist.append(xmax1)
                ymaxlist.append(ymax)
                # con.append(confi)
            
    xmin= torch.from_numpy(np.array(xminlist))

    ymin = torch.from_numpy(np.array(yminlist))
    xmax = torch.from_numpy(np.array(xmaxlist))
    ymax = torch.from_numpy(np.array(ymaxlist))
    score = torch.from_numpy(np.array(conflist))
    print(xmin)
    keep=nms(xmin,ymin,xmax,ymax,score)
    xmin = np.array(xmin[keep])
    ymin = np.array(ymin[keep])
    xmax = np.array(xmax[keep])
    ymax = np.array(ymax[keep])
    score = np.array(score[keep])
    print(xmin)
    # driver = ogr.GetDriverByName("ESRI Shapefile")
    # data_source = driver.CreateDataSource(os.path.join(outpath,'detection_poly.shp')) ## shp文件名称
    # srs = osr.SpatialReference()
    # srs.ImportFromEPSG(4326)
    # layer = data_source.CreateLayer("Polygon", srs, ogr.wkbPolygon) ## 图层名称要与shp名称一致
    # field_name = ogr.FieldDefn("Name", ogr.OFTString) ## 设置属性
    # field_name.SetWidth(20)  ## 设置长度
    # layer.CreateField(field_name)  ## 创建字段
    # feature = ogr.Feature(layer.GetLayerDefn())
    # feature.SetField("Name", "polygon") 
    poly  =[]
    with open(os.path.join(outpath,'detection.txt'),'w') as f:
        for i in range(len(score)):
            minlon,minlat = imagexy2geo(trans,xmin[i],ymin[i])
            maxlon,maxlat = imagexy2geo(trans,xmax[i],ymax[i])
            poly.append(geometry.Polygon([(minlon,minlat),(minlon,maxlat),(maxlon,maxlat),(maxlon,minlat),(minlon,minlat)]))
            f.write("hole"+" "+str(score[i])+" "+str(xmin[i])+" "+str(ymin[i])+" "+str(xmax[i])+" "+str(ymax[i])+'\n')
    po = gpd.GeoSeries(poly)
    cq = gpd.GeoDataFrame(geometry = po, crs = touying)
    if not os.path.exists(os.path.join(outpath,'detection_shp')):
        os.mkdir(os.path.join(outpath,'detection_shp'))
    cq.to_file(os.path.join(outpath,'detection_shp','detection_poly.shp'),driver='ESRI Shapefile', encoding = 'utf-8')
    # feature = None ## 关闭属性
    # data_source = None ## 关闭数据

def save_tile(raster_path , outpath, yolo):
    src= rio.open(raster_path)
    touying = src.crs
    trans = src.transform
    height = src.height
    width = src.width
    x_mins = np.arange(0, width, 800) 
    y_mins = np.arange(0, height, 800)
    # for  i,x_min in enumerate(x_mins) :
    #     for j, y_min in enumerate(y_mins) :
    #         x_max = x_min+1000 if x_min+1000<width else width
    #         y_max = y_min+1000 if y_min+1000<height else height
    #         x_geo_min ,y_geo_min = imagexy2geo(trans, x_min, y_min)
    #         x_geo_max ,y_geo_max = imagexy2geo(trans, x_max, y_max)
    #         tb = (x_geo_min,y_geo_max, x_geo_max, y_geo_min)
    #         window = windows.from_bounds(*tb, transform=trans)
    #         window = windows.Window(*[round(x) for x in window.flatten()])
    #         wi_tile = x_max-x_min
    #         he_tile = y_max - y_min
    #         tile_data = src.read(
    #             indexes=[1,2,3],
    #             window=window,
    #             out_shape=(he_tile, wi_tile),
    #             resampling=Resampling.bilinear,
    #         )
    #         # print(tile_data[0:3,]==0)
    #         tile_data[tile_data[0:3,]==0]=255
            
    #         tile_data = tile_data.transpose(1,2,0)

    #         print(tile_data.shape)
    #         w,h,_ = tile_data.shape
    #         if h!=1000 or w!=1000:
    #             im_pad = np.ones((1000,1000,3),dtype ='int8')
    #             im_pad = im_pad*255
    #             im_pad[0:w,0:h,0:3]=tile_data
    #             tile_data = im_pad
    #             print(tile_data.dtype)
    #         im = Image.fromarray(np.uint8(tile_data))
    #         name = str(i)+'_'+str(j)+'_'
    #         imaged=yolo.detect_image(im,name)
    #         imaged.save(os.path.join(outpath+'/vis/',name+'.png'))
    tiletxt2txt(trans, touying,outpath)