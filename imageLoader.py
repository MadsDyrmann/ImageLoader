# -*- coding: utf-8 -*-

'''
Author: Mads Dyrmann
'''

import numpy as np
from skimage import io,transform
import os

class imageLoader:
    def __init__(self):
        self.inputs = []
        self.targets = []
        self.imagesize = (256,256,3)

        
    def iterate_minibatchesList(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]



    def iterate_minibatchesImage(self, batchsize, shuffle=False):
        assert len(self.inputs) == len(self.targets)
        if shuffle:
            indices = np.arange(len(self.inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(self.inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            
            x = np.empty((batchsize,)+self.imagesize)
            for ix, filename in enumerate(self.inputs[excerpt]):
                im = io.imread(filename)
                x[ix,:]=transform.resize(im,self.imagesize)
            
            yield x, self.targets[excerpt]


    #Update image-list from csv-file
    def inputsFromCSV(self, csvpath):       
        import csv
        with open(csvpath, 'rb') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in csvreader:     
                self.inputs.append(row[0])
                self.targets.append(row[1])


    def inputsFromFilePath(self, filepath):
        #Find all images in folder and subfolder
        for root, dirnames, filenames in os.walk(filepath):
            for filename in filenames:
                if filename.lower().endswith(('.jpg','.jpg','.png','.tif','.tiff')):
                    self.inputs.append(os.path.join(root, filename))
        
        self.targets=[x.split('/')[-2] for x in self.inputs]
        
