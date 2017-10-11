#coding:utf-8
# 利用SVM进行人脸识别实例：
#下载实例，总共有1288个实例，特征值特征向量有1850个，所以要降维，有7个人需要辨别
from __future__ import print_function

from time import time  #每个步骤要计时
import logging         #打印出程序进展信息
import matplotlib.pyplot as plt   #绘图工具，识别出来的人脸进行画图

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC  #支持向量机


print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
# print the processing of program

###############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
#下载数据，作为一个类似于字典形式的值
# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
#返回有多少个图，和提取image和shape的值给h,w
# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
#每一行是一个实例，每一列是一个特征值
n_features = X.shape[1]
#返回矩阵的列数，即特征向量个数

# the label to predict is the id of the person
y = lfw_people.target
#返回数据集实例对应的（标记）不同的人
target_names = lfw_people.target_names
#返回类别中有哪几个名字，即有多少个人
n_classes = target_names.shape[0]
#即有多少类，多少人

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)
#将数据集拆分为训练集和测试集，X,Y


###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
#应用PCA来进行分类，降维提取特征值
n_components = 150
#组成元素的数量
print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
#初始时间
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
#随机PCA降维方法，对训练集中的向量进行建模
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))
#SVM算法，对人脸照片提取特征值
print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
#特征量中所有的特征向量，通过PCA转化为低维向量
X_test_pca = pca.transform(X_test)
#降维工作
print("done in %0.3fs" % (time() - t0))


###############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
#C：类似于权重吧，，，，gamma：代表多少特征点将会被使用，即选取特征值得比例，与前面的权重，看那一对能产生比较好的准确率
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
#将上一步所有的参数组合都放入SVC中，看那个准确率比较好，两两组合
clf = clf.fit(X_train_pca, y_train)
#！！！！关键：特征向量的数据，每一个特征向量的实际值进行建模
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
#SVM已调用，并且已建立模型

###############################################################################
# Quantitative evaluation of the model quality on the test set
#评估模型
print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
#对于新来的测试集的数据进行预测
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
#测试集的标签以及测试的标签对比
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
#建立nxn的矩阵，看预测和实际的标签


###############################################################################
# Qualitative evaluation of the predictions using matplotlib
#打印出
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]
#预测的人名
plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
#
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()

