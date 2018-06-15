
from ImageLoader.imageLoader import imageLoader
#from imageLoader import imageLoader

import scipy






#Test generator that starts over
il_test = imageLoader()
il_test.inputsFromFilePath(filepath='/media/mads/Data/TrainTestValDataset/GeneratedDatasetImages299x299_2018-01-11_SIMPLIFIED/Test')

ix=0
batchsize=10
for data, labels in il_test.iterate_minibatches(batchsize=batchsize, datastyle='image', shuffle=True, resize=True, labelstyle = 'label', startOverAfterFinished=True):
    ix+=1
    print(batchsize*ix,il_test.nSamples)







#Export list of labels and load them from other instance
il_train = imageLoader()
il_train.inputsFromFilePath(filepath='/media/mads/Data/TrainTestValDataset/GeneratedDatasetImages299x299_2018-01-11_SIMPLIFIED/Train')
il_train.exportDict(labelpath='labels.txt')

il_test = imageLoader()
il_test.inputsFromFilePath(filepath='/media/mads/Data/TrainTestValDataset/GeneratedDatasetImages299x299_2018-01-11_SIMPLIFIED/Test',labelspath='labels.txt')
img,lbl = il_test.getImagesAndLabels(returnstyle='numerical', shuffle=True)
for i in range(10):
    scipy.misc.imshow(img[i])
    print(lbl[i])
    pass



#########################

#Load images for semantic segmentation
il_test = imageLoader()
il_test.inputsFromFilePath(filepath='./SamplessemanticSegmentaiontrain/images', targetpath='./SamplessemanticSegmentaiontrain/targets')

for data, labels in il_test.iterate_minibatches(batchsize=3, datastyle='image', shuffle=True, labelstyle = 'path', resize=False):
    pass







#########################

#Load images for semantic segmentation, but return only the paths
il_test = imageLoader()
il_test.inputsFromFilePath(filepath='./SamplessemanticSegmentaiontrain/images', targetpath='./SamplessemanticSegmentaiontrain/targets')

for data, labels in il_test.iterate_minibatches(batchsize=3, datastyle='path', shuffle=True, labelstyle = 'path'):
    pass


##########################


#Load images and use their foldernames as labels
il_test = imageLoader()
il_test.inputsFromFilePath(filepath='/mnt/AU_BrugerDrev/Database/TrainTestValDataset/GeneratedDatasetImages256x256_2016-12-16_SIMPLIFIED/Test')

for data, labels in il_test.iterate_minibatches(batchsize=13, datastyle='image', shuffle=True, labelstyle = 'label'):
    pass

##########################

il_test = imageLoader()
il_test.inputsFromFilePath(filepath='/mnt/AU_BrugerDrev/Database/TrainTestValDataset/GeneratedDatasetImages256x256_2016-12-16_SIMPLIFIED/Test')

# Get all images
data, labels = il_test.getImagesAndLabels(returnstyle='numerical', zeromean=False, normalize=False, resize=True, preprocessor=None)







#########################

trainpath = '/home/mads/AU_BrugerDrev/Database/TrainTestValDataset/GeneratedDatasetImages256x256_2016-12-16_SIMPLIFIED/Train'
testpath = '/home/mads/AU_BrugerDrev/Database/TrainTestValDataset/GeneratedDatasetImages256x256_2016-12-16_SIMPLIFIED/Test'

# Read in data
#imagesize=256
#il_train, il_test = imageLoader.setupTrainValAndTest(trainpath=trainpath, testpath=testpath, valpath=None, imagesize=(imagesize, imagesize, 3))







#Export csv
#il_test.exportCSV('./test.csv')