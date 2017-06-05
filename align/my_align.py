"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random
from time import sleep



img = misc.imread(image_path)
                        img = img[:,:,0:3]

                        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        if nrof_faces>0:
                            det = bounding_boxes[:,0:4]
                            img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces>1:
                                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                img_center = img_size / 2
                                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                                det = det[index,:]
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0]-args.margin/2, 0)
                            bb[1] = np.maximum(det[1]-args.margin/2, 0)
                            bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                            bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                            scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                            nrof_successfully_aligned += 1
                            misc.imsave(output_filename, scaled)
                            text_file.write('%s %d %d %d %d\n' % (output_filename, bb[0], bb[1], bb[2], bb[3]))
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))

    print('Total number of images: %d' % nrof_images_total)
