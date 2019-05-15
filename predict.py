import os
from skimage import io
from paths import root_dir
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# 指定使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

images_dir = os.path.join(root_dir, 'images')
model_path = os.path.join(root_dir, 'model_data', 'model.h5')
class_name = {0: 'AK', 1: 'SK'}

if __name__ == '__main__':
    # 导入模型
    model = load_model(model_path)

    for AK_or_SK in os.listdir(images_dir):
        for picture_name in os.listdir(os.path.join(images_dir, AK_or_SK)):
            # 读取图片
            img_path = os.path.join(images_dir, AK_or_SK, picture_name)
            img = image.load_img(img_path, target_size=(224, 224))  # 通道3默认
            img = image.img_to_array(img)  # 变为numpy数组
            img = np.expand_dims(img, axis=0)  # 扩充维度

            # 预测
            preds = model.predict(img)

            # 打印图片类别
            # print(preds)
            y_pred = np.argmax(preds, axis=1)

            label = class_name[y_pred[0]]   #y_pred[0]解释：打印出来类似于[0]  [1] ,所以取列表的第一个元素，即索引[0]

            print(picture_name, '的预测概率是：')
            print(preds, ' --> ', label)
