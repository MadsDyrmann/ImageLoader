
from ImageLoader.imageLoader import imageLoader

il_test = imageLoader()
il_test.inputsFromFilePath(filepath='/home/mads/AU_BrugerDrev/Database/TrainTestValDataset/GeneratedDatasetImages256x256_2016-12-16_SIMPLIFIED/Test')


#data, labels = next(il_test.iterate_minibatchesImage(batchsize=10))

for data, labels in il_test.iterate_minibatchesImage(batchsize=13, shuffle=True, returnstyle = 'label'):
    pass
