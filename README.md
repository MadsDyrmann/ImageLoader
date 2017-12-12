# ImageLoader

This python class is used for loading data for machine-learning projects.
Main points:
* Contains generators, which are useful for batch-processing
* Able to shufftle data for each epoch
* Can learn labels of images from foldernames, csv-files, or from a image/label file pair like in VOC-dataset
* Can return labels as numerical labels (0,1,2,...), as text-labels (dog, cat, house,...), as one-hot decoded (0,0,1,0,0,0,...), as file paths (data/train/1.txt), and as decoded images.

## Use cases
### Load images and use their foldernames as labels
```
#Create object for holding test set 
il_test = imageLoader()
#Set the input
il_test.inputsFromFilePath(filepath='data/Test')

# Iterate over data and labels with a batch size of 13, return images an array of decoded images, return labels as 
for data, labels in il_test.iterate_minibatches(batchsize=13, datastyle='image', shuffle=True, labelstyle = 'label'):
    pass
```

### Load images, but return only the paths to the images and label-files
```
#Create object for holding test set 
il_test = imageLoader()
#Set the input
il_test.inputsFromFilePath(filepath='./data/images/test', targetpath='./data/targets/test')

for data, labels in il_test.iterate_minibatches(batchsize=3, datastyle='path', shuffle=True, labelstyle = 'path'):
    pass
```

### Load images for semantic segmentation, and decode labels and samples
```
#Create object for holding test set 
il_test = imageLoader()
#Set the input
il_test.inputsFromFilePath(filepath='./data/images/test', targetpath='./data/targets/test')

for data, labels in il_test.iterate_minibatches(batchsize=3, datastyle='image', shuffle=True, labelstyle = 'image'):
    pass
```


### Setup train (and/or/not) test (and/or/not) validation data
```
trainpath = './data/images/train'
testpath = './data/images/test'

imagesize=256
il_train, il_test = imageLoader.setupTrainValAndTest(trainpath=trainpath, testpath=testpath, valpath=None, imagesize=(imagesize, imagesize, 3))
```

## Other useful functions
### Export to CSV
```
il_test.exportCSV('./test.csv')
```
### Get list of samples and targets and targets as one-hot decoded
```
il_test.inputs
il_test.targets
il_test.targetsOneHot
```
### Get dictionary translating numerical labels to string-labels (e.g. 0:cat, 1:dog, 2:house)
```
il_test.labelsDict
```
