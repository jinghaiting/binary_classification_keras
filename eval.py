import os
from load_train_test_data import load_test_data
from paths import root_dir
from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 指定使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

model_path = os.path.join(root_dir, 'model_data', 'model.h5')

if __name__ == '__main__':
    # 加载测试数据
    X_test, y_test = load_test_data()

    # 导入模型
    model = load_model(model_path)

    # 预测
    y_pred = model.predict(X_test)

    # one-hot ==> 标签
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    # 计算准确率、精确率、召回率、F1
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("accuracy_score = %.2f" % accuracy)
    print("precision_score = %.2f" % precision)
    print("recall_score = %.2f" % recall)
    print("f1_score = %.2f" % f1)
