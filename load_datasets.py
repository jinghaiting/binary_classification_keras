import os
import numpy as np
from tqdm import tqdm
from skimage import io
from skimage import transform
from paths import root_dir, mkdir_if_not_exist
from sklearn.utils import shuffle

import matplotlib.pyplot as plt  # 画图

datasets_dir = os.path.join(root_dir, 'datasets')
cached_dir = os.path.join(root_dir, 'cache')
mkdir_if_not_exist(dir_list=[cached_dir])  # paths.py文件处理


def process_data():
    images = []
    labels = []

    for AK_or_SK_dir in tqdm(os.listdir(datasets_dir)):
        # AK ==> [1,0]  Sk ==> [0,1]
        if 'AK' in AK_or_SK_dir:
            label = [1, 0]
        elif 'SK' in AK_or_SK_dir:
            label = [0, 1]
        else:
            print('AK_or_SK_dir is error!')
        for person_name_dir in tqdm(os.listdir(os.path.join(datasets_dir, AK_or_SK_dir))):  # 给路径，而不是文件名
            for image_name in os.listdir(os.path.join(datasets_dir, AK_or_SK_dir, person_name_dir)):
                img_path = os.path.join(datasets_dir, AK_or_SK_dir, person_name_dir, image_name)
                image = io.imread(img_path)
                image = transform.resize(image, (224, 224),
                                         order=1, mode='constant',
                                         cval=0, clip=True,
                                         preserve_range=True,
                                         anti_aliasing=True)
                image = image.astype(np.uint8)
                images.append(image)
                labels.append(label)
    return images, labels


def load_datasets():
    images_npy_filename = os.path.join(cached_dir, 'images_data.npy')
    labels_npy_filename = os.path.join(cached_dir, 'labels.npy')

    if os.path.exists(images_npy_filename) and os.path.exists(labels_npy_filename):
        images = np.load(images_npy_filename)
        labels = np.load(labels_npy_filename)
    else:
        images, labels = process_data()
        # 打乱后保存
        images, labels = shuffle(images, labels)
        np.save(images_npy_filename, images)
        np.save(labels_npy_filename, labels)

    return images, labels



if __name__ == '__main__':

    X, y = load_datasets()
    plt.imshow(X[7])  #画在画布上
    plt.show()        #显示
    print(X.shape)
    print(y.shape)
    print(len(X))

