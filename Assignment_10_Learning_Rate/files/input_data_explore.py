import matplotlib.pyplot as plt
import numpy as np
import torchvision



def imshow(image,class_name ):
    #  Restoring from a normalized form
    image = image / 2 + 0.5
    image = image.numpy()
    fig = plt.figure(figsize=(9,9))
    plt.imshow(np.transpose(image, (1, 2, 0)),interpolation='none')
    plt.title(class_name)



def show_train_data(train_data_loader, list_of_classes):
    # Getting a batch from train dataloader
    train_data_iter = iter(train_data_loader)
    images, labels = train_data_iter.next()
    for class_name in range(len(list_of_classes)):
        index = [i for i in range(len(labels)) if labels[i] == class_name]
        imshow(torchvision.utils.make_grid(images[index[0:10]],nrow=10,padding=3,scale_each=True),list_of_classes[class_name])