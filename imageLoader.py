# -*- coding: utf-8 -*-
'''
Copyright 2017 Mads Dyrmann

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
'''

'''
Verision History: 

Ver 1.1.5: 2017-09-11: Specified datatype for onehot labels and images as int32 and float32, respectively.
'''

import numpy as np
from skimage import io, transform
import os


class imageLoader:
    def __init__(self):
        self.inputs = []
        self.targets = []
        self.nSamples = 0
        self.imagesize = (256, 256, 3)
        self.targetsNumerical = []
        self.targetsOneHot = []
        self.labelsDict = {}
        self.numericalDictionary = None
        self.nClasses = []
        self.inputpath = ''
        
        self.__version__ = '1.1.5'



    #Generator, which loops over a list of paths to files
    def iterate_minibatchesList(self, batchsize, shuffle=False, returnstyle='numerical'):
        assert len(self.inputs) == len(self.targets)
        if shuffle:
            indices = np.arange(len(self.inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(self.inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            if returnstyle == 'numerical':
                yield self.inputs[excerpt], self.targetsNumerical[excerpt]
            if returnstyle == 'onehot':
                yield self.inputs[excerpt], self.targetsOneHot[excerpt]
            if returnstyle == 'label':
                yield self.inputs[excerpt], self.targets[excerpt]


    #Generator, which loops over a images of paths to files
    def iterate_minibatchesImage(self, batchsize, shuffle=False, returnstyle='numerical'):
        assert len(self.inputs) == len(self.targets)
        if shuffle:
            indices = np.arange(len(self.inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(self.inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
                inputs = [self.inputs[x] for x in excerpt]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
                inputs = self.inputs[excerpt]


            x = np.empty((batchsize,)+self.imagesize,dtype=np.float32)
            for ix, filename in enumerate(inputs):
                im = io.imread(filename)
                x[ix, :] = transform.resize(im, self.imagesize).astype(np.float32)
            if returnstyle == 'numerical':
                yield x, self.targetsNumerical[excerpt]
            if returnstyle == 'onehot':
                yield x, self.targetsOneHot[excerpt]
            if returnstyle == 'label':
                yield x, self.targets[excerpt]

    def getImagesAndLabels(self, indices, returnstyle='numerical'):
        assert len(self.inputs) == len(self.targets)

        inputs = [self.inputs[x] for x in indices]

        x = np.empty((len(indices),)+self.imagesize,dtype=np.float32)
        for ix, filename in enumerate(inputs):
            im = io.imread(filename)
            x[ix, :] = transform.resize(im, self.imagesize).astype(np.float32)
        if returnstyle == 'numerical':
            yield x, self.targetsNumerical[indices]
        if returnstyle == 'onehot':
            yield x, self.targetsOneHot[indices]
        if returnstyle == 'label':
            yield x, self.targets[excerpt]

    #Update one-hot targets
    def oneHotTargets(self, numClasses=None):
        #Use numClasses for overwriting the number of targets in current dataset.
        #Useful if, e.g. the test-set only contains 3 classes and the training contains 5 classes
        if not numClasses:
            numClasses = self.nClasses
        self.targetsOneHot = np.eye(numClasses)[self.targetsNumerical].astype(np.int32)

    # Update image-list from csv-file
    def inputsFromCSV(self, csvpath, numericalTargets=False):
        self.inputpath = csvpath
        
        import csv
        with open(csvpath, 'rb') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';', quotechar='"')
            next(csvreader, None)  # Skip header
            for row in csvreader:
                self.inputs.append(row[0])
                self.targets.append(row[2])
                self.targetsNumerical.append(int(row[1]))
        self.numericalDictionary = {key:ix for ix,key in enumerate(list(set(self.targets)))}
        self.updateDataStats()

    #Create list of images from input path, where folder-names are used as labels
    def inputsFromFilePath(self, filepath):
        self.inputpath = filepath
        # Find all images in folder and subfolder
        for root, dirnames, filenames in os.walk(filepath):
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.jpg', '.png', '.tif', '.tiff')):
                    self.inputs.append(os.path.join(root, filename))

        self.targets = [x.split('/')[-2] for x in self.inputs]
        
        # Create  numerical labels if they do not exist
        if not self.numericalDictionary:
            self.numericalDictionary = {key:ix for ix,key in enumerate(list(set(self.targets)))}
            
        self.targetsNumerical = [self.numericalDictionary[x] for x in self.targets]
        self.updateDataStats()
        
    def updateDataStats(self):
        self.nSamples = len(self.inputs)
        self.nClasses = len(self.numericalDictionary)
        self.oneHotTargets()
              

## Usage:
##   trainpath = '/media/mads/Eksternt drev/Images used in phd thesis/Cropped for classification/GeneratedDatasetImages128x128_2016-12-16/Train'
##   testpath = '/media/mads/Eksternt drev/Images used in phd thesis/Cropped for classification/GeneratedDatasetImages128x128_2016-12-16/Test'
##   il_train2, il_test2 = imageLoader.setupTrainValAndTest(trainpath=trainpath,testpath=testpath,valpath=None,imagesize=(28,28,3))
##


def setupTrainValAndTest(trainpath=None,testpath=None,valpath=None,imagesize=(None,None,None)):
    assert trainpath is not None
    # Setup classes, and make correct label setup between the three classes
    returnClasses=[]
    if trainpath:
        il_train = imageLoader()
        il_train.imagesize = (imagesize[0],imagesize[1],imagesize[2])
        if os.path.isfile(trainpath):
            il_train.inputsFromCSV(trainpath)
        else:
            il_train.inputsFromFilePath(trainpath)
        il_train.oneHotTargets()
        returnClasses.append(il_train)
    if testpath:
        il_test = imageLoader()
        il_test.numericalDictionary = il_train.numericalDictionary #Use same dictionary as for training
        il_test.imagesize = (imagesize[0],imagesize[1],imagesize[2])
        if os.path.isfile(testpath):
            il_test.inputsFromCSV(testpath)
        else:
            il_test.inputsFromFilePath(testpath)        
        il_test.oneHotTargets(numClasses=il_train.nClasses)
        returnClasses.append(il_test)
    if valpath:
        il_val = imageLoader()
        il_val.numericalDictionary = il_train.numericalDictionary #Use same dictionary as for training
        il_val.imagesize = (imagesize[0],imagesize[1],imagesize[2])
        if os.path.isfile(testpath):
            il_val.inputsFromCSV(valpath)
        else:
            il_val.inputsFromFilePath(valpath)   
        il_val.oneHotTargets(numClasses=il_train.nClasses)
        returnClasses.append(il_val)
    # Return an instance for train, val and test if paths are provided
    return returnClasses
    
    
