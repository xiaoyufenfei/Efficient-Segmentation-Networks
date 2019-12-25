# converting trainIDs to labelIDs for evaluating the test set segmenatation results of the cityscapes dataset

import numpy as np
import os
from PIL import Image



# index: trainId from 0 to 18, 19 semantic class   val: labelIDs
cityscapes_trainIds2labelIds = np.array([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
                                        dtype=np.uint8)


def trainIDs2LabelID(trainID_png_dir, save_dir):
    print('save_dir:  ', save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    png_list = os.listdir(trainID_png_dir)
    for index, png_filename in enumerate(png_list):
        #
        png_path = os.path.join(trainID_png_dir, png_filename)
        # print(png_path)
        print('processing(', index, '/', len(png_list), ') ....')
        image = Image.open(png_path)  # image is a PIL #image
        pngdata = np.array(image)
        trainID = pngdata  # model prediction
        row, col = pngdata.shape
        labelID = np.zeros((row, col), dtype=np.uint8)
        for i in range(row):
            for j in range(col):
                labelID[i][j] = cityscapes_trainIds2labelIds[trainID[i][j]]

        res_path = os.path.join(save_dir, png_filename)
        new_im = Image.fromarray(labelID)
        new_im.save(res_path)


if __name__ == '__main__':
    trainID_png_dir = '../server/cityscapes/predict/ENet'
    save_dir = '../server/cityscapes/predict/cityscapes_submit/'
    trainIDs2LabelID(trainID_png_dir, save_dir)
