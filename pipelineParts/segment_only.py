from pathlib import Path
import pyvips
from torchvision import transforms

from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
import matplotlib.pyplot as plt
import cv2


import numpy as np
import sys
from time import time
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 2000000000

###################################################
# Color Normalizing Functions
###################################################

def cum_hist_3D(pic_array):
    cumHist = np.zeros((256,3))
    for i in range(3):
        _hist, _ = np.histogram(pic_array[:,:,i],bins=256,range=(-0.5,255.5))
        _cum_hist = _hist.cumsum()
        cumHist[:,i] = _cum_hist/_cum_hist[-1]

    return cumHist

def createHistMatchMap(cumHistRef,cumHistSrc):

    histMatchMap = np.zeros((256,3),dtype=np.uint8)
    for i in range(3):
        for a in range(256):
            histMatchMap[a,i] = np.argmin(np.abs(cumHistRef[:,i]-cumHistSrc[a,i]))
    return histMatchMap

def histMatch(histMatchMap,src_array):
    out = np.zeros(src_array.shape,dtype=np.uint8)
    for i in range(3):
        out[:,:,i] = histMatchMap[src_array[:,:,i],i]
    return out

###################################################
# Segmentation post-processing functions
###################################################

def find_unlabeled_pixel(mask, value, labels, lab, to_search):
    # If we find an unlabeled pixel, we increase the
    # labe number, label it, and add it to "to_search"

    inds = np.nonzero((mask == value) & (labels == 0))
    if len(inds[0]) > 0:
        lab += 1
        i = inds[0][0]
        j = inds[1][0]
        labels[i,j] = lab

    return labels, lab, to_search

###################################################

def find_adjacent_pixels(i,mask,value,labels,lab,to_search):
    # First, remove from "to_search", if there
    to_search.remove[i]

    #Now, search adjacent blocks
    for d in [0,1]:
        for f in [-1,1]:
            I = i.copy() #i is a list of two indices
            I[d] += f
            if I[d] >= 0 and I[d] < mask.shape[d]: # Make sure it is in range
                if mask[tuple(I)] == value and labels[tuple(I)] == 0:
                    labels[tuple(I)] = lab
                    to_search += [I]

    return labels, to_search

####################################################

def create_cluster_masks(mask,value):
    labels = np.zeros(mask.shape,dtype=np.int64)

    # Run the algorithm
    to_search = []
    labels, lab, to_search = find_unlabeled_pixel(mask,value,labels,0,to_search)
    while(len(to_search) > 0):
            while(len(to_search) > 0):
                labels, to_search = find_adjacent_pixels(to_search[0],mask,value,labels,lab,to_search)
            labels,lab,to_search = find_unlabeled_pixel(mask,value,labels,lab,to_search)

    # Find the cluster index with the most pixels
    cluster_counts = np.bincount(labels.flatten())[1:]
    try:
        max_i = np.argmax(cluster_counts)+1
    except:
        max_i = None

    return labels, max_i

####################################################

def fillHoles(mask):

    D = 50
    r = mask.shape[0]
    c = mask.shape[1]
    for i in range(r):
        for j in range(c):
            value = mask[i][j]
            if(value == 0):
                iPlus = i+D
                iMinus = i-D
                jPlus = j+D
                jMinus = j-D
                if (iPlus < r and mask[iPlus][j] == 0):
                    for w in range(i,iPlus+1):
                        mask[w][j] = 0
                if (iMinus >= 1 and mask[iMinus][j] == 0):
                    for w in range(iMinus,i+1):
                        mask[w][j] = 0
                if (jPlus < c and mask[i][jPlus] == 0):
                    for x in range(j,jPlus+1):
                        mask[i][x] = 0
                if (jMinus >= 1 and mask[i][jMinus] == 0):
                    for x in range(jMinus, j+1):
                        mask[i][x] = 0
    return mask

####################################################
        
# accuracy funciton for segmentation
def acc(input, target):
    target = target.squeeze(1)
    return (input.argmax(dim=1)==target).float().mean()

size=500
learn = load_learner('../models') # segmentation model

print('Segmenting images...')
for path in Path().glob('input/*.tif'):
    fullname = str(path).split('/')[-1].split('.')[0]
    print(fullname)
    
#    img0 = pyvips.Image.new_from_file(str(path))
    img0 = cv2.imread(str(path))   
    #out2 = cv2.imread(str(path))

    # Recolor the images
    tmp0 = img0
#    src2 = np.ndarray(buffer=tmp0.write_to_memory(),
#            dtype=format_to_dtype[tmp0.format],
#            shape=[tmp0.height, tmp0.width, tmp0.bands])[:,:,::-1]
    cumHistSrc = cum_hist_3D(tmp0)
    cumHistRef = np.load('input/cumHistMultiRef.npy')
    histMatchMap = createHistMatchMap(cumHistRef,cumHistSrc)
    out2 = histMatch(histMatchMap,tmp0)

            
            
    ###INSERT ABOUT NO_PHENOTYPE HERE###
    #img = out2.copy()
    img = Image(transforms.ToTensor()(out2[:,:,::-1].copy()))
    img.resize(size)
    pred = learn.predict(img)[0]
    imgIn = out2.copy()
    maskIn = pred.data.detach().numpy()[0].astype('float32')
    labels,max_i = create_cluster_masks(maskIn,0)

    maskIn[labels!=max_i] = 1
    maskIn = fillHoles(maskIn)
    maskOut = cv2.resize(maskIn,(imgIn.shape[1],imgIn.shape[0]))
    cv2.imwrite(f'output/mask_{fullname}.jpg',maskOut)
    masked = imgIn.copy()
    masked[maskOut==1] = 255
    cv2.imwrite(f'glomOnly/glomTuft_{fullname}.jpg',masked)
