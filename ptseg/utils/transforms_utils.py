import numpy as np
import cv2
import random
import os

# calculate means and std
# root = 'D:\\'
# train_txt_path = './train_val_list.txt'
root = 'D:\\PycharmWorkspace\\Torch-Texture-Classification\\dataset\\train'
train_txt_path = 'train.csv'
# train_txt_path = 'D:\\PycharmWorkspace\\Torch-Texture-Classification\\dataset\\train\\train.csv'


img_h, img_w = 224, 224
# imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []
img_list = []
with open(os.path.join(root, train_txt_path), 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)  # shuffle , 随机挑选图片
    CNum = len(lines) // 2

    for i in range(CNum):

        img_path = os.path.join(root, lines[i].rstrip().split(',')[0])
        print(i, img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_h, img_w))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
imgs = np.concatenate(img_list, axis=3)

imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
means.reverse()  # BGR --> RGB
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
