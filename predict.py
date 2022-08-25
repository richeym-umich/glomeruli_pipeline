import pyvips
import torch
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
import matplotlib.pyplot as plt
import cv2

import numpy as np
import sys
from time import time
#import pyvips
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 2000000000

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

####################################################
# Color normalization functions (histogram matching)
####################################################

# Reference: http://paulbourke.net/miscellaneous/equalisation/
# also https://github.com/scikit-image/scikit-image/blob/main/skimage/exposure/histogram_matching.py
# and https://github.com/aiethn/histogram-matching/blob/master/histogram_matching.py
# Note: the "reference" image is the source for the color histogram,
# and the "source" image is the source for the image whose colors will be updated.

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
      # Find component of the reference cumulative histogram that is closest
      # to the current component of the source cumulative histogram
      histMatchMap[a,i] = np.argmin(np.abs(cumHistRef[:,i]-cumHistSrc[a,i]))

  return histMatchMap

def histMatch(histMatchMap,src_array):
  out = np.zeros(src_array.shape,dtype=np.uint8)
  for i in range(3):
    out[:,:,i] = histMatchMap[src_array[:,:,i],i]
    
  return out

####################################################
# Object detection functions
####################################################

# Get maximum intersection-over-area (inspired by implementation for IOU at https://pytorch.org/docs/stable/_modules/torchvision/ops/boxes.html)
# This is to check if one box is a subset of another (i.e. it caught a partial object on the edge of the cropped image)
def maxIOA(boxes1, boxes2):

  area1 = torchvision.ops.box_area(boxes1) # [N]
  area2 = torchvision.ops.box_area(boxes2) # [M]

  lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
  rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

  wh = (rb - lt).clamp(min=0)  # [N,M,2]
  inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

  max_ioa = torch.max(inter/area1[:, None],inter/area2) # [N,M]

  return max_ioa

# partial object suppression, i.e. remove boxes that identify a partial object,
# likely picked up at the edge of the cropped image
def pos(boxes,all_scores,pds_threshold):

  n = len(boxes)
  keep = torch.arange(n,dtype=int).tolist()
  max_ioa = maxIOA(boxes,boxes)

  for i in range(1,n):
    for j in range(i):
      if max_ioa[i,j] > pds_threshold:
        areas = torchvision.ops.box_area(boxes[[i,j]])
        min_i = torch.argmin(areas)
        max_i = torch.argmax(areas)
        remove = (i,j)[min_i]
        if remove in keep:
          keep.remove(remove)

  return torch.as_tensor(keep,dtype=torch.int64)


def predict(model2,val_pic,fout,min_score=0.8):

  model2.eval()

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  img0 = PIL.Image.open(f'output/{val_pic}_small.jpg').convert("RGB");

  # Get prediction
  n_stride = 1000
  n_wide = 1500

  all_boxes = torch.zeros((0,4),device=device)
  all_scores = torch.zeros((0),device=device)
  all_labels = torch.zeros((0),device=device,dtype=torch.int)
  print('Begin predicting by tile...')

  dims_tmp = img0.size
  dims = (dims_tmp[1],dims_tmp[0])

  for i in range((dims[0]+n_wide-n_stride) // n_stride):
    for j in range((dims[1]+n_wide-n_stride) // n_stride):
      tmp0 = img0.crop((n_stride*j,n_stride*i,min(dims[1],n_wide + n_stride*j),min(dims[0],n_wide + n_stride*i)))
      tmp = transforms.ToTensor()(np.asarray(tmp0))
      with torch.no_grad():
        prediction = model2([tmp.to(device)]);

      pboxes = prediction[0]['boxes']
      pboxes[:,[0,2]] += n_stride*j
      pboxes[:,[1,3]] += n_stride*i

      keep = torchvision.ops.remove_small_boxes(pboxes,50) # was using 100
      all_boxes = torch.cat((all_boxes,pboxes[keep]),0)
      all_scores = torch.cat((all_scores,prediction[0]['scores'][keep]),0)
      all_labels = torch.cat((all_labels,prediction[0]['labels'][keep]),0)


  img0.close()
  model2 = None
  del model2
  print(f'Done predicting by tile')

  # Keep only scores above min_score (defaults to 0.8)
  all_boxes = all_boxes[all_scores>min_score]
  all_labels = all_labels[all_scores>min_score]
  all_scores = all_scores[all_scores>min_score]

  # Remove partial objects
  keep = pos(all_boxes,all_scores,0.7)
  all_boxes = all_boxes[keep]
  all_scores = all_scores[keep]
  all_labels = all_labels[keep]

  # Remove high aspect ratio boxes
  aspect = (all_boxes[:,2] - all_boxes[:,0])/(all_boxes[:,3] - all_boxes[:,1])
  keep = (aspect < 2) & (aspect > 0.5)
  all_boxes = all_boxes[keep]
  all_labels = all_labels[keep]
  all_scores = all_scores[keep]
  
  cat_names = ['No_phenotype', 'Global_sclerosis', 'Other']

  # Write out predictions, scaled up by 0.35 in each dimension
  for i,box in enumerate(all_boxes):
    fout.write(f'{val_pic}\t{i}\t{int(float(box[0])/0.35)}\t{int(float(box[1])/0.35)}\t{int(float(box[2])/0.35)}\t{int(float(box[3])/0.35)}\t{cat_names[all_labels[i]-1]}\t{all_scores[i]}\n')
  fout.flush()

####################################################
# Segmentation post-processing functions
####################################################

def find_unlabeled_pixel(mask,value,labels,lab,to_search):
  # If we find an unlabeled pixel, we increase the
  # label number, label it, and add it to "to_search"

  inds = np.nonzero((mask == value) & (labels == 0))
  if len(inds[0]) > 0:
    lab += 1
    i = inds[0][0]
    j = inds[1][0]
    labels[i,j] = lab
    to_search += [[i,j]]

  return labels, lab, to_search

######################################################3

def find_adjacent_pixels(i,mask,value,labels,lab,to_search):
    # First, remove from "to_search", if there
    to_search.remove(i)

    # Now, search adjacent blocks 
    for d in [0,1]:
      for f in [-1,1]:
        I = i.copy() #i is a list of two indices
        I[d] += f
        if I[d] >= 0 and I[d] < mask.shape[d]: # Make sure it is in range
          if mask[tuple(I)] == value and labels[tuple(I)] == 0:
            labels[tuple(I)] = lab
            to_search += [I]

    return labels, to_search

###########################################################

def create_cluster_masks(mask,value):
    labels = np.zeros(mask.shape,dtype=np.int64)

    # Run the algorithm
    to_search = []
    labels, lab, to_search = find_unlabeled_pixel(mask,value,labels,0,to_search)
    while(len(to_search) > 0):
      while(len(to_search) > 0):
        labels, to_search = find_adjacent_pixels(to_search[0],mask,value,labels,lab,to_search)
      labels, lab, to_search = find_unlabeled_pixel(mask,value,labels,lab,to_search)

    # Find the cluster index with the most pixels
    cluster_counts = np.bincount(labels.flatten())[1:]
    try:
      max_i = np.argmax(cluster_counts)+1
    except:
      max_i = None
        
    return labels, max_i

###########################################################

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
        if(jMinus >= 1 and mask[i][jMinus] == 0):
          for x in range(jMinus,j+1):
            mask[i][x] = 0

  return mask

###########################################################


##########################################################################################
##########################################################################################
# THE FULL PIPELINE
##########################################################################################
##########################################################################################
  
#######################################
# First, coarsen the full res image
#######################################

if not os.path.exists('output'):
  os.makedirs('output')

val_pic = sys.argv[1]

if len(sys.argv) > 2:
  input_dir = sys.argv[2]
else:
  input_dir = 'input'

begin = time()
print('Coarsening image...')
im = pyvips.Image.new_from_file(f'{input_dir}/{val_pic}.tif',
                                access='sequential')
resized = im.resize(0.35)
im = None
del im
print(f'Done coarsening image, time: {time()-begin}')

#######################################
# Use histogram matching to recolor
# the coarsened image
#######################################

begin = time()
print('Normalizing color...')
src = np.ndarray(buffer=resized.write_to_memory(),
                 dtype=format_to_dtype[resized.format],
                 shape=[resized.height, resized.width, resized.bands])
src = src[:,:,::-1]
cumHistSrc = cum_hist_3D(src)
cumHistRef = np.load('input/cumHistMultiRef.npy')
histMatchMap = createHistMatchMap(cumHistRef,cumHistSrc)
out = histMatch(histMatchMap,src)
cv2.imwrite(f'output/{val_pic}_small.jpg',out)
print(f'Done normalizing color, time: {time()-begin}')

resized = None
src = None
out = None
del resized
del src
del out

#######################################
# Next, run the object detection model,
# write the (scaled up) boxes out to a text file
#######################################

min_score = 0.8
model_num = 21
epoch = 29
#"""
print('Running object detection...')
begin = time()
if torch.cuda.is_available():
  model = torch.load(f'models/model{model_num}_epoch{epoch}')
else:
  model = torch.load(f'models/model{model_num}_epoch{epoch}',map_location=torch.device('cpu'))

with open(f'output/detections_{val_pic}.txt','w') as fout:
  fout.write('#image\tindex\txmin\tymin\txmax\tymax\tcategory\tscore\n')
  fout.flush()
  
  predict(model,val_pic,fout,min_score=min_score)
model = None
del model
print(f'Model: {model_num}, Epoch: {epoch}, picture: {val_pic}')
print(f'Done with objection detection, time: {time()-begin}')
#"""
#######################################
# Crop and segment the nonpathologic glomeruli
# from the full res images
#######################################

# accuracy function for segmentation
def acc(input, target):
    target = target.squeeze(1)
    return (input.argmax(dim=1)==target).float().mean()

size = 500
learn = load_learner('models') # segmentation model

# Create folder for segmented images
if not os.path.exists(f'output/segmented_{val_pic}'):
  os.makedirs(f'output/segmented_{val_pic}')

print('Cropping and segmenting full resolution glomeruli...')
begin = time()

# Open full image
img0 = pyvips.Image.new_from_file(f'{input_dir}/{val_pic}.tif')

# Create folder for cropped images
if not os.path.exists(f'output/cropped_{val_pic}'):
  os.makedirs(f'output/cropped_{val_pic}')

def check_edge(cluster,shape):

  """
  print('Left:',np.sum(cluster[:,1]==0))
  print('Right:',np.sum(cluster[:,1]==(shape[1]-1)))
  print('Top:',np.sum(cluster[:,0]==0))
  print('Bottom:',np.sum(cluster[:,0]==(shape[0]-1)))
  """

  return np.array([np.sum(cluster[:,1]==0),
                   np.sum(cluster[:,1]==(shape[1]-1)),
                   np.sum(cluster[:,0]==0),
                   np.sum(cluster[:,0]==(shape[0]-1))],dtype=np.int)
  
# Open file with box coordinates
i = 0
adjusted = np.zeros((0,5),dtype=np.int32)
duplicates = []
with open(f'output/detections_{val_pic}.txt') as fin:
  with open(f'output/adj_detections_{val_pic}.txt','w') as fout:
    fout.write('#image\tindex\txmin\tymin\txmax\tymax\tcategory\tscore\n')
    fin.readline()
    for line in fin:
      data = line.split()
      dw = int(0.15*(int(data[4]) - int(data[2])))
      dh = int(0.15*(int(data[5]) - int(data[3])))

      xmin0 = max(int(data[2])-dw,0)
      ymin0 = max(int(data[3])-dh,0)
      xmax0 = min(int(data[4])+dw,img0.width-1)
      ymax0 = min(int(data[5])+dh,img0.height-1)
      dx0 = xmax0-xmin0
      dy0 = ymax0-ymin0
    
      tmp0 = img0.extract_area(xmin0,ymin0,dx0,dy0)

      # Recolor the cropped images
      src2 = np.ndarray(buffer=tmp0.write_to_memory(),
                        dtype=format_to_dtype[tmp0.format],
                        shape=[tmp0.height, tmp0.width, tmp0.bands])[:,:,::-1]
      out2 = histMatch(histMatchMap,src2)
      cv2.imwrite(f'output/cropped_{val_pic}/{data[6]}_{val_pic}_{i}.tif',out2)

      # Segment if non-pathologic
      if (data[6] == 'No_phenotype'):

        img = Image(transforms.ToTensor()(out2[:,:,::-1].copy()))
        img.resize(size)
        pred = learn.predict(img)[0]
        imgIn = out2
        maskIn = pred.data.detach().numpy()[0].astype('float32')
        labels,max_i = create_cluster_masks(maskIn,0)
      
        if max_i is not None:
          cluster = np.argwhere(labels==max_i)
          expand = check_edge(cluster,maskIn.shape)
        
          jj = 0
          N = 1
          if (np.sum(expand) > 0):
            N += 1
          
          while (jj < N):
            ratio = [imgIn.shape[0]/maskIn.shape[0],imgIn.shape[1]/maskIn.shape[1]]

            xmax0 = min(xmin0+(int(np.max(cluster[:,1])*ratio[1])+100) + int(expand[1]*ratio[1]),img0.width-1)
            xmin0 = max(0,xmin0+(int(np.min(cluster[:,1])*ratio[1])-100) - int(expand[0]*ratio[1])) # Do this AFTER xmax0
            ymax0 = min(ymin0+(int(np.max(cluster[:,0])*ratio[0])+100) + int(expand[3]*ratio[0]),img0.height-1)
            ymin0 = max(0,ymin0+(int(np.min(cluster[:,0])*ratio[0])-100) - int(expand[2]*ratio[0])) # Do this AFTER ymax0
        
            tmp0 = img0.extract_area(xmin0,ymin0,xmax0-xmin0,ymax0-ymin0)
        
            # Recolor the cropped images
            src2 = np.ndarray(buffer=tmp0.write_to_memory(),
                              dtype=format_to_dtype[tmp0.format],
                              shape=[tmp0.height, tmp0.width, tmp0.bands])[:,:,::-1]
            out2 = histMatch(histMatchMap,src2)
            img = Image(transforms.ToTensor()(out2[:,:,::-1].copy()))

            img.resize(size)
            pred = learn.predict(img)[0]
            imgIn = out2
            maskIn = pred.data.detach().numpy()[0].astype('float32')
            labels,max_i = create_cluster_masks(maskIn,0)
            cluster = np.argwhere(labels==max_i)
            expand = check_edge(cluster,maskIn.shape)
            if (np.sum(expand) > 0) and N < 5:
              N += 2

            jj += 1

          cv2.imwrite(f'output/cropped_{val_pic}/{data[6]}_{val_pic}_{i}B.tif',out2)

          # Check for duplicates
          if len(adjusted)>0:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            max_ioa = maxIOA(torch.tensor(adjusted[:,1:],device=device),
                             torch.tensor([[xmin0,ymin0,xmax0,ymax0]],device=device))
            if torch.max(max_ioa).item() > 0.8:
              duplicates.append(i)
              i += 1
              continue # don't save this duplicate glomerulus as a segmented image or in the adjusted detections
                           
          adjusted = np.append(adjusted,[[i,xmin0,ymin0,xmax0,ymax0]],axis=0)
          fout.write(f'{val_pic}\t{i}\t{xmin0}\t{ymin0}\t{xmax0}\t{ymax0}\t{data[6]}\t{data[7]}\n')
          maskIn[labels!=max_i] = 1
          maskIn = fillHoles(maskIn)
          maskOut = cv2.resize(maskIn,(imgIn.shape[1],imgIn.shape[0]))    
          masked = imgIn.copy()
          masked[maskOut==1] = 255
          cv2.imwrite(f'output/segmented_{val_pic}/glomTuft_{val_pic}_{i}.jpg',masked)

      else:
        fout.write(line)
      i += 1

np.savetxt(f'output/duplicates_{val_pic}.txt',np.asarray(duplicates),fmt='%i')
    
img0 = None
del img0
print(f'Done cropping and segmenting glomeruli, time: {time()-begin}')
