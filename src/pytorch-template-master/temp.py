from data_loader.data_loaders import MITIndoorDataset, MITIndoorDataLoader,MnistDataLoader
import os
import cv2

if __name__ == '__main__':
    test_dataset = MnistDataLoader(data_dir='./data/MITIndoor', batch_size=4)
    for i, (img, label) in enumerate(test_dataset):
        print(i, img.shape, label)
        print(label)

        if i > 10:
            break

    # dataloader = MITIndoorDataLoader(data_dir='./data/MITIndoor', batch_size=4)
    # for i, (img, label) in enumerate(dataloader):
    #     print(i, img.shape, label)

    #     if i > 10:
    #         break

    # Use cv2 to read all img in the folder, if any error occurs, remove the img

    # for img in os.listdir('./data/MITIndoor'):
    #     try:
    #         cv2.imread('./data/MITIndoor/' + img)
    #     except:
    #         os.remove('./data/MITIndoor/' + img)