#!/usr/bin/python

import math
import sys
import os
import re
import subprocess
import numpy as np
import pandas as pd
import argparse
import operator
import mxnet as mx
import struct
import cv2
from collections import namedtuple
import scipy.io
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import textwrap

def parse_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\

Dump featuremaps at a given layer, side-by-side with input image

Example usage:
   ~/investigation_tools/colorspace/dump_featuremaps.py 
   --image_dir_root /home/data/images/Colorspace/imagenet/png/val 
   --image_list_file /home/data/images/Colorspace/imagenet/db/ColorSpace_val_short.lst 
   --checkpoint_path ~/exps/resnet34/checkpoint 
   --epoch 22 --layer_name _plus2_output

'''))

    parser.add_argument('--image_dir_root',
                        metavar = '<String>',
                        help = "A directory with subdirs '601' and '709'",
                        required = True,
                        type = str)

    parser.add_argument('--image_list_file',
                        metavar = '<String>',
                        help = "Image list file, like those used to build RecordIO databases",
                        required = True,
                        type = str)

    parser.add_argument('--checkpoint_path',
                        metavar = '<String>',
                        help = "Path of scheckpoint folder",
                        required = True,
                        type = str)
    
    parser.add_argument('--epoch',
                        metavar = '<Number>',
                        help = "epoch number to be listed",
                        required = True,
                        type = int)

    parser.add_argument('--layer_name',
                        metavar = '<String>',
                        help = "Name of the layer to dump, e.g. _plus12_output",
                        required = True,
                        type = str)

    args = parser.parse_args()

    return args

def load_image(input_filename, color=3):
    if color == 3:
        img = mx.image.imread(input_filename)
    else:
        img = mx.image.imread(input_filename, flag=0)

    img = img.transpose((2,0,1))
    img = img.expand_dims(axis=0)
    img = img.asnumpy()
    img = mx.nd.array(img)

    #coolimg = coolimg.asnumpy()
    #newcoolimg = coolimg
    #newcoolimg[:,:,0] = coolimg[:,:,0]
    #newcoolimg[:,:,1] = coolimg[:,:,2]
    #newcoolimg[:,:,2] = coolimg[:,:,0]
    #coolimgconv = cv2.cvtColor(newcoolimg, cv2.COLOR_YCR_CB2RGB) 

    #cv2.imwrite("heresyuvimage.png", newcoolimg)
    #cv2.imwrite("givemeimage.png", coolimgconv)
    
    return img

args = parse_args()

# Dump the activations at the output of a given layer
# Could use this to dump a fully connected layer (one-dimensional vector)
# and use for exploring clustering.

checkpoint_dir  = args.checkpoint_path + "/checkpoint"
checkpoint_num  = args.epoch
image_dir_root  = args.image_dir_root
image_list_file = args.image_list_file
layer_name      = args.layer_name

fd = open( image_list_file, "r" )

Batch = namedtuple('Batch', ['data'])

sym, arg_params, aux_params = mx.model.load_checkpoint(checkpoint_dir, checkpoint_num)

#for key in arg_params:
 #   print(key)

wgts = arg_params['fc1_weight']
print("WEIGHTS SHAPE: {}".format(wgts.shape))
wgts_shape = wgts.shape
#print("WEIGHTS: {}".format(wgts))

sd = open("fcweights.txt","w")
best601 = np.zeros(10)
position601 = np.zeros(10)
best709 = np.zeros(10)
position709 = np.zeros(10)
wgts = wgts.asnumpy()
counts = 0

fd = open( image_list_file, "r" )

all_layers = sym.get_internals()
outs = all_layers.list_outputs()
outs.sort()

#for elt in outs:
    #print( elt )

# Symbol e.g. _plus8
sym = all_layers[layer_name]

# Create module by given symbol
mod = mx.mod.Module(symbol = sym, label_names = None, context=mx.gpu())

mod.bind(for_training=False, data_shapes=[('data', (1,3,256,256))])

mod.set_params(arg_params, aux_params)

# pattern must be changed to fit image names

#0	0.000000	601/ILSVRC2012_val_00000002-601.png
#26	0.000000	601/suzie00027-601-scaled.png
pattern = "([0-9]+)\s+([0-9\.]+)\s+([67]0[19])(/.*\-)([67]0[19])(.*)(\.png|\.bmp)"
#0	0.000000	601/n01440764_10183-601.png

# Bike_Ride00007-601.bmp
#pattern = "(.*\.)(png|bmp)"

#fd_out = open( "_plus12_output.csv", "w" )

gain = 1

count = 0

# Create dictionaries for each class, m based on pattern arrangement

# In our case, our classes are for Rec. 601 and Rec. 709
dict601 = {}
dict709 = {}

while 1:
    line = fd.readline()
    
    if line == "":
        break

    m = re.search( pattern, line )
    if (m):
    
        item_num  = int(m.group(1))
        classval  = float(m.group(2))
        coldir    = m.group(3)
        guts      = m.group(4)
        colspace  = m.group(5)
        jazz      = m.group(6)
        ext       = m.group(7)
    
        #guts      = m.group(1)
        #ext       = m.group(2)
    
        #print( item_num, classval, colspace, ext )

        img_name = image_dir_root+"/"+coldir+guts+colspace+jazz+ext
        #print(colspace)
        #print(guts)
        #print(bgr_img.shape)
        #img_name = image_dir_root+"/"+guts+ext
        #print(img_name)

        if (classval < 1):
            dict601[guts] = img_name
        else:
            dict709[guts] = img_name

#activations = np.array([])
#max_act = np.array([])

for the_key in sorted(dict601):

    for colspace in ( "601", "709" ):
    
        if colspace == "601":
            print(colspace)
            print(the_key)
            if the_key in dict601:
                img_name = dict601[the_key]
        elif colspace == "709":
            print(colspace)
            print(the_key)
            if the_key in dict709:
                img_name = dict709[the_key]
        
        img = load_image(img_name)

        #print ("Loaded " + img_name)
        
        yuv_img = cv2.imread(img_name)
        rgb_img = cv2.cvtColor(yuv_img, cv2.COLOR_YCR_CB2RGB)

        img_shape = img.shape
        img_hgt = img_shape[2]
        img_wid = img_shape[3]
        
        mod.forward(Batch([mx.nd.array(img)]))
        out = mod.get_outputs()[0]
        out = out.asnumpy()

        out_shape = out.shape
        #sys.stderr.write("out.shape=")
        #sys.stderr.write(str(out_shape) + "\n")
        
        ##print(out_shape)

        # out.shape is (1, 256, 16, 16)
        num_maps = out_shape[1]
        hgt      = out_shape[2]
        wid      = out_shape[3]

        tile_cols = int(math.ceil(math.sqrt(num_maps)))
        tile_rows = int(math.ceil(float(num_maps)/tile_cols))

        padded_hgt = hgt + 2
        padded_wid = wid + 2
        
        border_color = 192
        
        #print( tile_rows, padded_hgt, tile_cols, padded_wid )
        
        alloc_hgt = max(img_hgt, tile_rows * padded_hgt)
        alloc_wid = img_hgt    + tile_cols * padded_wid
        
        montage = np.zeros((alloc_hgt, alloc_wid))
        montage = montage.astype(np.uint8)

        img_lum = img[0,2,:,:]
        img_lum = img_lum.asnumpy()
        
        montage[0:img_hgt, (tile_cols * padded_wid):alloc_wid] = img_lum;
        
        feature_count = 0

        res = cv2.resize(rgb_img,(64,64))
        
        print('out: {}'.format(count))
        
        # Create and print montage of feature maps for each image at selected layer

        if (count == 0):
            ftrstore = np.zeros((tile_rows,tile_cols,9,hgt,wid))
            imgstore = np.zeros((tile_rows,tile_cols,9,64,64,3))
            cv2.imwrite("firstimg.png", res)
            #print(res.shape)
            #print(hgt)
            #print(wid)

        for tile_row in range(tile_rows):
            for tile_col in range(tile_cols):
                # Write borders
                for col_off in range (padded_wid):
                    montage[tile_row     * padded_hgt,     tile_col * padded_wid + col_off] = border_color
                    montage[(tile_row+1) * padded_hgt - 1, tile_col * padded_wid + col_off] = border_color
                for row_off in range (padded_hgt):
                    montage[tile_row * padded_hgt + row_off, tile_col     * padded_wid]     = border_color
                    montage[tile_row * padded_hgt + row_off, (tile_col+1) * padded_wid - 1] = border_color
        
                tile_idx = tile_row * tile_cols + tile_col

                if tile_idx < num_maps:
                    tile = np.around(gain * np.squeeze(out[0,tile_idx,:,:]) )
                    tile = np.clip( tile, 0, 255 )
                    tile = tile.astype(np.uint8)

                    ##activations = np.append(activations,np.amax(np.squeeze(out[0,tile_idx,:,:])))
                    if (tile_row == 0) and (tile_col == 2):
                        print(np.amax(np.squeeze(out[0,tile_idx,:,:])))

                    #print('feature: {}\tactivations: {}'.formatftrstore(feature_count,np.sum(tile)))

                    feature_count += 1

                    #print("out: {}\tfeature map: {}\tactivations: {}".format(pic_count,feature_count,np.sum(tile)))
                    
                    #print( tile.shape )

                    montage[tile_row * padded_hgt + 1:tile_row * padded_hgt + 1 + hgt, tile_col * padded_wid + 1:tile_col * padded_wid + 1 + wid] = tile

                    tile_shape = tile.shape
                    pixelcount = 0

                    #for row in range(tile_shape[0]):
                     #   for col in range(tile_shape[1]):
                      #      if tile[row][col] > 0:
                       #         pixelcount += 1

                    img_vals = np.array([])
                    for i in range(0,9):
                        #print('ROWS: {}'.format(tile_row))
                        #print('COLUMNS: {}'.format(tile_col))
                        #print(ftrstore.shape)
                        img_vals = np.append(img_vals,np.amax(ftrstore[tile_row][tile_col][i]))
                    if (tile_row == 0) and (tile_col == 2):
                        print('TOP NINE:')
                        print(img_vals)
                        print('MINIMUM: {} VS TILE: {}'.format(np.amax(ftrstore[tile_row][tile_col][np.argmin(img_vals)]),np.amax(np.squeeze(out[0,tile_idx,:,:]))))
                        
                    if (np.amin(img_vals) < np.amax(np.squeeze(out[0,tile_idx,:,:]))):
                        ftrstore[tile_row][tile_col][np.argmin(img_vals)] = np.squeeze(out[0,tile_idx,:,:])
                        imgstore[tile_row][tile_col][np.argmin(img_vals)] = res

                    # Sort new vals
                    placements = np.argsort(-img_vals)
                    placeholder_ftr = np.zeros((9,hgt,wid))
                    placeholder_img = np.zeros((9,64,64,3))
                    placecount = 0
                    for key in placements:
                        placeholder_ftr[placecount] = ftrstore[tile_row][tile_col][key]
                        placeholder_img[placecount] = imgstore[tile_row][tile_col][key]
                        placecount += 1
                    ftrstore[tile_row][tile_col] = placeholder_ftr
                    imgstore[tile_row][tile_col] = placeholder_img
                    if (tile_row == 0) and (tile_col == 2):
                        print(placements)
                        print(img_vals)
                    
        #plt.imshow(montage, cmap='gray')
        #plt.show()

        cv2.imwrite("out_%04d.png" % count, montage)
               
        #for i in range(out.shape[1]):
        #    fd_out.write( "%12.9f," % out[0,i] )    

        #fd_out.write("\n")
        
        count = count + 1
        sys.stderr.write(".")
        
##activations.shape = (count,feature_count)

#for feature in range(feature_count):
 #   print('Feature Map: {}'.format(feature))
  #  perf = np.array([])
   # for out in range(count):
        #print('count: {} feature: {}'.format(out,feature))
        #perf = np.append(perf,activations[out][feature])
    #for img in range(0,9):
     #   top = np.argmax(perf)
      #  max_act = np.append(max_act,top)
       # print('Best Out: {}\tActivations: {}'.format(top,np.amax(perf)))
        #perf[top] = 0

#print(activations[5])

#max_act.shape = (feature_count,9)
#print(max_act[9])
#print(max_act[10])
#print(max_act[11])

# Create and print montage of 16 best images and activations for each feature map

feature_shape = ftrstore.shape
pic_shape     = imgstore.shape

num_tiles_rows = feature_shape[0]
num_tiles_cols = feature_shape[1]
 
num_maps = feature_shape[2]
ftr_hgt  = feature_shape[3]
ftr_wid  = feature_shape[4]
num_pics = pic_shape[2]
pic_hgt  = pic_shape[3]
pic_wid  = pic_shape[4]

padded_ftr_hgt = int(ftr_hgt + (ftr_hgt / 4))
padded_ftr_wid = int(ftr_wid + (ftr_wid / 4))
padded_pic_hgt = int(pic_hgt + (pic_hgt / 4))
padded_pic_wid = int(pic_wid + (pic_wid / 4))

tile_cols = int(math.ceil(math.sqrt(num_maps)))
tile_rows = int(math.ceil(float(num_maps)/tile_cols))
        
border_color = 192

alloc_ftr_hgt = ftr_hgt * 4
alloc_ftr_wid = ftr_wid * 4

alloc_pic_hgt = pic_hgt * 4
alloc_pic_wid = pic_wid * 4

newpad_ftr_hgt = alloc_ftr_hgt + 2
newpad_ftr_wid = alloc_ftr_wid + 2
newpad_pic_hgt = alloc_pic_hgt + 2
newpad_pic_wid = alloc_pic_wid + 2

new_hgt = num_tiles_rows * newpad_ftr_hgt
new_wid = num_tiles_cols * newpad_ftr_wid
new_pic_hgt = num_tiles_rows * newpad_pic_hgt
new_pic_wid = num_tiles_cols * newpad_pic_wid
#print(alloc_ftr_hgt)
#print(alloc_ftr_wid)
#print(montage.shape)

#img_lum = img[0,2,:,:]
#img_lum = img_lum.asnumpy()
        
#acttage[0:img_hgt, (tile_cols * padded_ftr_wid):alloc_ftr_wid] = img_lum;

ftrtage = np.zeros((new_hgt, new_wid))
ftrtage = ftrtage.astype(np.uint8)
imgtage = np.zeros((new_pic_hgt, new_pic_wid, 3))
imgtage = imgtage.astype(np.uint8)

for num_tile_row in range(num_tiles_rows):
    for num_tile_col in range(num_tiles_cols):
        acttage = np.zeros((alloc_ftr_hgt, alloc_ftr_wid))
        acttage = acttage.astype(np.uint8)
        pictage = np.zeros((alloc_pic_hgt, alloc_pic_wid, 3))
        pictage = pictage.astype(np.uint8)
        for tile_row in range(tile_rows):
            for tile_col in range(tile_cols):
                # Write borders
                for col_off in range (padded_ftr_wid):
                    acttage[tile_row     * padded_ftr_hgt,     tile_col * padded_ftr_wid + col_off] = border_color
                    acttage[(tile_row+1) * padded_ftr_hgt - 1, tile_col * padded_ftr_wid + col_off] = border_color
                for row_off in range (padded_ftr_hgt):
                    acttage[tile_row * padded_ftr_hgt + row_off, tile_col     * padded_ftr_wid]     = border_color
                    acttage[tile_row * padded_ftr_hgt + row_off, (tile_col+1) * padded_ftr_wid - 1] = border_color

                for col_off in range (padded_pic_wid):
                    pictage[tile_row     * padded_pic_hgt,     tile_col * padded_pic_wid + col_off] = border_color
                    pictage[(tile_row+1) * padded_pic_hgt - 1, tile_col * padded_pic_wid + col_off] = border_color
                for row_off in range (padded_pic_hgt):
                    pictage[tile_row * padded_pic_hgt + row_off, tile_col     * padded_pic_wid]     = border_color
                    pictage[tile_row * padded_pic_hgt + row_off, (tile_col+1) * padded_pic_wid - 1] = border_color

                tile_idx = tile_row * tile_cols + tile_col

                if tile_idx < num_maps:
                    tile = np.around(gain * np.squeeze(ftrstore[num_tile_row,num_tile_col,tile_idx,:,:]))
                    pic  = np.around(gain * np.squeeze(imgstore[num_tile_row,num_tile_col,tile_idx,:,:]))

                    #print(pic.shape)
                    
                    # Add in images and featuremaps to empty montage

                    acttage[tile_row * padded_ftr_hgt + int(ftr_hgt / 8):tile_row * padded_ftr_hgt + int(ftr_hgt / 8) + ftr_hgt, tile_col * padded_ftr_wid + int(ftr_wid / 8):tile_col * padded_ftr_wid + int(ftr_wid / 8) + ftr_wid] = tile

                    pictage[tile_row * padded_pic_hgt + int(pic_hgt / 8):tile_row * padded_pic_hgt + int(pic_hgt / 8) + pic_hgt, tile_col * padded_pic_wid + int(pic_wid / 8):tile_col * padded_pic_wid + int(pic_wid / 8) + pic_wid] = pic

        #sys.stdout.write("ftr_hgt=%d, ftr_wid=%d\n"%(ftr_hgt, ftr_wid))
        #sys.stdout.write("acttage.shape=" + str(acttage.shape) + "\n")
        #print(num_tile_row * newpad_ftr_hgt + 1, num_tile_row * newpad_ftr_hgt + 1 + (ftr_hgt * 4), num_tile_col * newpad_ftr_wid + 1, num_tile_col * newpad_ftr_wid + 1 + (ftr_wid * 4))
        #print( ftrtage.shape )
        
        ftrtage[num_tile_row * newpad_ftr_hgt + 1:num_tile_row * newpad_ftr_hgt + 1 + (ftr_hgt * 4), num_tile_col * newpad_ftr_wid + 1:num_tile_col * newpad_ftr_wid + 1 + (ftr_wid * 4)] = acttage

        imgtage[num_tile_row * newpad_pic_hgt + 1:num_tile_row * newpad_pic_hgt + 1 + (pic_hgt * 4), num_tile_col * newpad_pic_wid + 1:num_tile_col * newpad_pic_wid + 1 + (pic_wid * 4)] = pictage

        if num_tile_row == 7 and num_tile_col == 12:
            cv2.imwrite("best256_103.png", acttage)

#print(acttage.shape)
#print(ftrtage.shape)

cv2.imwrite("bestfeatures%s.png" % layer_name, ftrtage)
cv2.imwrite("bestimages%s.png" % layer_name, imgtage)

#fd_out.close()
print("")
        

