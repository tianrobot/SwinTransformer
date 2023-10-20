import os
import torch

import numpy as np
import matplotlib.pyplot as plt
import shutil
import random

from PIL import Image
from torchvision import transforms
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans




# image = Image.open('dataset/Testing/pituitary_tumor/image(15).jpg')
# plt.imshow(image)
# plt.show()
# image.show()
# total_pixels = w * h
# w, h = image.size

# resolution
# resolution = (w, h)
# print(resolution)

# pixels = list(image.getdata())

# print(image.getdata())

# pixels_array = [pixels[i * w:(i+1) * w] for i in range(h)]

# print(pixels_array[0][0])

# pixels = image.load()
# total_pixels = image.width * image.height

# image_array = np.array(image)
# print(image_array)

# pixels_sum = np.sum(image_array)

# print(pixels_sum)

# if __name__ == '__main__':

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # img_size = 384
    # data_transform = transforms.Compose([transforms.Resize(int(img_size * 1.14)),
                                         # transforms.CenterCrop(img_size),
                                         # transforms.ToTensor(),
                                         # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # img_path = "dataset/Testing/pituitary_tumor/image(15).jpg"
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path)
    # print('1: ',img)
    # Convert MRIs into RGB image
    # img = img.convert('RGB')
    # plt.imshow(img)
    # [N, C, H, W]
    # img = data_transform(img)
    # print('2: ',img)
    # expand batch dimension
    # img = torch.unsqueeze(img, dim=0)
    # print('3: ',img)


# Generate some random data
# np.random.seed(0)
# X = np.random.rand(100, 1)
# y = 2 + 5*X + 10*X**2 - 20*X**3 + 10*X**4 + np.random.randn(100, 1)

# Fit a polynomial regression model with degree 10(10 features)
# poly_features = PolynomialFeatures(degree=10, include_bias=False)
# X_poly = poly_features.fit_transform(X)
# lin_reg = LinearRegression()
# lin_reg.fit(X_poly, y)

# Plot the results
# plt.scatter(X, y)
# plt.plot(X, lin_reg.predict(X_poly), color='r')
# plt.show()


# Fit a polynomial regression model with degree 10 and L2 regularization
# ridge_reg = Ridge(alpha=0.5, solver='cholesky')
# ridge_reg.fit(X_poly, y)

# Plot the results
# plt.scatter(X, y)
# plt.plot(X, ridge_reg.predict(X_poly), color='r')
# plt.show()


# Generate some random data
# np.random.seed(0)
# X1 = np.random.randn(50, 2) + np.array([2, 2])
# X2 = np.random.randn(50, 2) + np.array([-2, -2])
# X = np.vstack((X1, X2))
# y = np.hstack((np.zeros(50), np.ones(50)))

# Plot the data
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

# Define the pipeline
# svm_clf = Pipeline([
    # ("scaler", StandardScaler()),
    # ("svm", SVC(kernel="rbf", gamma=0.1, C=100))
# ])

# Fit the model
# svm_clf.fit(X, y)

# Plot the decision boundary
# x1s = np.linspace(-5, 5, 100)
# x2s = np.linspace(-5, 5, 100)
# x1, x2 = np.meshgrid(x1s, x2s)
# X_new = np.c_[x1.ravel(), x2.ravel()]
# y_pred = svm_clf.predict(X_new).reshape(x1.shape)
# plt.contourf(x1, x2, y_pred, alpha=0.3)
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()


# Generate a 2D dataset with 2 clusters
# X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)

# Plot the dataset
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()


# 源图片数据集目录
source_dataset_dir = '/Users/tianjiexin/Downloads/CT/non-COVID'
# 目标图片数据集目录
target_dataset_dir = '/Users/tianjiexin/Downloads/CT/test/non-COVID'
# 随机抽取的图片数量
num_images_to_extract = 246

# 获取源数据集中的所有图片文件名
image_files = os.listdir(source_dataset_dir)

# 随机选择一些图片
selected_images = random.sample(image_files, num_images_to_extract)

# 将选中的图片复制到目标数据集目录并从源数据集中删除
for image_file in selected_images:
    source_path = os.path.join(source_dataset_dir, image_file)
    target_path = os.path.join(target_dataset_dir, image_file)
    shutil.copyfile(source_path, target_path)
    os.remove(source_path)

print("图片抽取完成！")



# https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification