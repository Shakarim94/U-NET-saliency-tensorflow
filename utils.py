from __future__ import division
import numpy as np
from PIL import Image

#import gc
#import os
#import sys


#loads images as grayscale, resizes to (256,256), normalizes to [0,1] interval, converts to [1,h,w,1] array
def load_images_float32(path):
    im = Image.open(path).convert('L')
    im_256 = im.resize((256,256), resample=Image.BICUBIC)
    
    im_256_np = np.expand_dims(np.expand_dims(np.array(im_256), axis=0), axis=3).astype(np.float32) / 255.0
    im_np = np.expand_dims(np.expand_dims(np.array(im), axis=0), axis=3).astype(np.float32) / 255.0
    return im_256_np, im_np

#loads images as RGB,  resizes to (256,256), normalizes to [0,1] interval, converts to [1,h,w,3] array
def load_images_RGB_float32(path):
    im = Image.open(path).convert('RGB')
    im_size = im.size
    im_256 = im.resize((256,256), resample=Image.BICUBIC)
    
    im_256_np = np.expand_dims(np.array(im_256), axis=0).astype(np.float32) / 255.0
    im_np = np.expand_dims(np.array(im), axis=0).astype(np.float32) / 255.0
    return im_256_np, im_np, im_size

#used for training dataset
#loads as images and saliency maps, resizes to (256,256)
#normalizes to [0,1] interval, converts to [?,h,w,c] arrays
#optionally applies data augmentation according to scale
def load_img_map_float32_aug(imglist, maplist, augment_scale):
    assert len(imglist)== len(maplist)
    num = len(imglist)
    img_array = np.zeros((num, 256,256,3), dtype='float32')
    map_array = np.zeros((num, 256,256,1), dtype='float32')

    for idx in xrange(num):
        im = Image.open(imglist[idx]).convert('RGB')
        mp = Image.open(maplist[idx]).convert('L')
        
        #random data augmentation
        random_mode = np.random.randint(0, augment_scale)
        im = data_augmentation(np.array(im), random_mode)
        mp = data_augmentation(np.array(mp), random_mode)
        
        #making shape (1,256,256,c)
        img_array[idx]= np.expand_dims(im, axis=0).astype(np.float32) / 255.0
        map_array[idx]= np.expand_dims(np.expand_dims(mp, axis=0), axis=3).astype(np.float32) / 255.0
    return img_array, map_array

	
#saves concatenated image of RGB image, ground truth saliency map and predicted saliency map
def save_images(filepath, image, true_map=None, pred_map=None):
    # assert the pixel value range is 0-255
    if true_map is None:
        cat_image = np.squeeze(image)
        im = Image.fromarray(cat_image.astype('uint8')).convert('L')
    else:
        true_map=np.concatenate([true_map,true_map,true_map], 3)
        pred_map=np.concatenate([pred_map,pred_map,pred_map], 3)
        #shape becomes (1,h,w,3)
        
        image = np.squeeze(image)
        true_map = np.squeeze(true_map)
        pred_map = np.squeeze(pred_map)
        #shape becomes (h,w,3)
        
        cat_image = np.concatenate([image, true_map, pred_map], axis=1)
        im = Image.fromarray(cat_image.astype('uint8')).convert('RGB')

    im.save(filepath, 'png')


#Peak signal-to-noise ratio calculation in dB
def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr

#calculates mean absolute error
def cal_mae(im1, im2):
	#assert pixel value in range 0-1
    mae = (np.abs(im1.astype(np.float) - im2.astype(np.float))).mean()
    return mae

#calculates precision and recall of a single image
def cal_precision_and_recall(gt_map, pred_map):
    # assert that the pixel values are in range [0,1] and float32 type
    pred_map = np.squeeze(pred_map)
    gt_map = np.squeeze(gt_map)
    
    #thresholding the pred_map to get the binary version
    t = np.mean(pred_map) * 2
    pred_map_thr = (pred_map>t) * 1
    
    true_neg = np.sum((pred_map_thr==0) * (gt_map==0) * 1)
    false_neg = np.sum((pred_map_thr==0) * (gt_map==1) * 1)
    false_pos = np.sum((pred_map_thr==1) * (gt_map==0) * 1)
    true_pos = np.sum((pred_map_thr==1) * (gt_map==1) * 1)
    
    if true_pos==0 and false_pos==0 and false_neg==0:
        return 1,1
    elif true_pos==0 and (false_pos>0 or false_neg>0):
        return 0,0
    else:
        precision = true_pos / (true_pos + false_pos)
        
        recall = true_pos / (true_pos + false_neg)
        
        return precision, recall

#calculates the F-measure of single image
#DON'T use this to calculate avg F-measure of a dataset
#use cal_precision_and_recall() for all images and then calculate avg. F-measure
def cal_fmeasure_img(gt_map, pred_map, beta=0.3):
    # assert that the pixel values are in range [0,1] and float32 type
    pred_map = np.squeeze(pred_map)
    gt_map = np.squeeze(gt_map)
    
    #thresholding the pred_map to get the binary version
    t = np.mean(pred_map) * 2
    pred_map_thr = (pred_map>t) * 1
    
    true_neg = np.sum((pred_map_thr==0) * (gt_map==0) * 1)
    false_neg = np.sum((pred_map_thr==0) * (gt_map==1) * 1)
    false_pos = np.sum((pred_map_thr==1) * (gt_map==0) * 1)
    true_pos = np.sum((pred_map_thr==1) * (gt_map==1) * 1)
    
    if true_pos==0 and false_pos==0 and false_neg==0:
        return 1
    elif true_pos==0 and (false_pos>0 or false_neg>0):
        return 0
    else:
        precision = true_pos / (true_pos + false_pos)
        
        recall = true_pos / (true_pos + false_neg)
        
        beta_2 = beta**2
        return (1+beta_2)*precision*recall/(beta_2*precision + recall)


#applies data augmentation according to the mode
def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)
