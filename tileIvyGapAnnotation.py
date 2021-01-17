## tileIvyGapAnnotation.py
#
# tile Ivy Gap images based on annotation masks
# python 3
#
# usage:
# python3 tileIvyGapAnnotation.py image_file_name.jpg
#
# [[30,132,155], Leading Edge (LE)
#  [203,7,202], Infiltrating Tumor (IT)
#  [255,91,1], Hyperplastic blood vessels in infiltrating tumor (IThbv); Hyperplastic blood vessels in cellular tumor (CThbv)
#  [10,201,6], Cellular Tumor (CT)
#  [58,203,247], Perinecrotic zone (CTpnz)
#  [2,10,202], Pseudopalisading cells but no visible necrosis (CTpnn)
#  [7,202,160], Pseudopalisading cells around necrosis (CTpan)
#  [255,44,1], Microvascular proliferation (CTmvp)
#  [7,6,6], Necrosis (CTne)
#  [255,255,255], background]
#
#
# Kun-Hsing Yu
# 2019.7.19
# modified paths by Chen Lu 2020.3.21

from PIL import Image
import numpy as np
import scipy
import scipy.misc
import sys
import os

imageFileName = sys.argv[1]
print(imageFileName)
imageFullPath = '/n/data2/hms/dbmi/kyu/lab/datasets/IvyGap/HE/'+imageFileName
print(imageFullPath)

#print(imageFileName)
Image.MAX_IMAGE_PIXELS = 10000000000
#Image.warnings.simplefilter('error', Image.DecompressionBombWarning)

list_of_colors = [[30,132,155],[203,7,202],[255,91,1],[10,201,6],[58,203,247],[2,10,202],[7,202,160],[255,44,1],[7,6,6],[255,255,255]]
color = [155,155,155]

xPatch=1000
yPatch=1000

xStride=500
yStride=500

def closest(colorlist,color):
    colorlist = np.array(colorlist)
    color = np.array(color)
    distances = np.sum((colorlist-color)**2,axis=1)
    #distances = np.sqrt(np.sum((colorlist-color)**2,axis=1))
    index_of_smallest = np.where(distances==np.amin(distances))
    smallest_distance = colorlist[index_of_smallest]
    if (np.amin(distances)>20000):
        print(str(color)+":"+str(np.amin(distances)))
    return int(index_of_smallest[0][0])


def main():
    heImage = Image.open(imageFullPath + '.jpg')
    #heImage = Image.open("IvyGapImages/"+imageFileName[0:-4]+"_H.jpg")
    maskImage = Image.open(imageFullPath + 'A.jpg')
    #maskImage = Image.open("IvyGapImages/"+imageFileName)
    heImageArray = np.asarray(heImage)
    maskImageArray = np.asarray(maskImage)
    xDim=maskImageArray.shape[0]
    yDim=maskImageArray.shape[1]
    result_path = "/n/data2/hms/dbmi/kyu/lab/cl427/segmentation/"+imageFileName
    os.makedirs(result_path)
    f = open(result_path+"/annotation_R.txt", "w")

    for i in range(xDim//xStride-1):
        for j in range(yDim//xStride-1):
            maskPatch = maskImageArray[(xStride*i):(xStride*i+xPatch),(yStride*j):(yStride*j+yPatch)]
            rawUnique, rawCounts = np.unique(maskPatch.reshape(-1, maskPatch.shape[2]), axis=0, return_counts=True)
            closestColor = np.zeros(rawUnique.shape[0]) # to handle color drift in JPG compression
            for k in range(rawUnique.shape[0]):
                closestColor[k] = closest(list_of_colors,rawUnique[k])
            finalCounts = np.zeros(len(list_of_colors))
            for k in range(len(list_of_colors)):
                finalCounts[k]=np.sum(rawCounts[np.where(closestColor==k)])
            if (np.max(finalCounts)>(xPatch*yPatch*0.5)):
                _=f.write(str(xStride*i)+'\t'+str(yStride*j)+'\t'+str(np.argmax(finalCounts))+'\n')
                imagePatch = Image.fromarray(heImageArray[(xStride*i):(xStride*i+xPatch),(yStride*j):(yStride*j+yPatch)])
                outputFileName = "/n/data2/hms/dbmi/kyu/lab/cl427/segmentation/"+imageFileName+"/seg_"+str(xStride*i)+"_"+str(yStride*j)+".jpg"
                imagePatch.save(outputFileName)
    f.close()

if __name__ == '__main__':
    main()
