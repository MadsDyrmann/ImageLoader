
from imageLoader import imageloader

il_test = imageloader()
il_test.inputsFromFilePath(filepath='/media/mads/79131cf3-d458-45c7-bb29-830bc120a265/Phd/Software/DeepNetworks/SingleImageClassificationUsingCaffe/GeneratedDatasetImages_pad_256x256_2016-10-27/Test')


for data, labels in il_test.iterate_minibatchesImage(batchsize=10):
    pass
