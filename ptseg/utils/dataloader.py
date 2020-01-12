from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import csv
from PIL import Image


class TextureDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.image_label_list = self.read_file()
        self.len = len(self.image_label_list)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        print('image label list length:', self.len)

    def __getitem__(self, item):
        index = item % self.len
        image_path, label = self.image_label_list[index]

        path = os.path.join(self.root, image_path)
        image = Image.open(path)
        img = self.transform(image)
        # label = np.array(label)
        label = torch.LongTensor(label)
        return img, label

    def _get_label_by_item(self, item):
        index = item % self.len
        _, label = self.image_label_list[index]
        label = label[0]
        return label

    def __len__(self):
        return self.len

    def read_file(self):
        image_label_list = []
        if self.train:
            csv_name = 'train.csv'
        else:
            csv_name = 'test.csv'
        csv_file = open(os.path.join(self.root, csv_name))
        csv_reader = csv.reader(csv_file)
        _ = next(csv_reader)  # read headers and move the pointer to the next line
        for content in csv_reader:
            name = content[0]
            labels = []
            for value in content[1:]:
                labels.append(int(value))
            image_label_list.append((name, labels))
        return image_label_list


def get_sampler(train_dataset, num_classes):
    imgcount_per_class = [0] * num_classes
    num_images = train_dataset.len
    for index in range(num_images):
        image_label = train_dataset._get_label_by_item(index)
        # print(image_label[1].item())
        imgcount_per_class[image_label] = imgcount_per_class[image_label] + 1
    weight_per_class = [0.] * num_classes
    for index in range(num_classes):
        weight_per_class[index] = num_images / float(imgcount_per_class[index])
    weight = [0] * num_images
    for index in range(num_images):
        image_label = train_dataset._get_label_by_item(index)
        weight[index] = weight_per_class[image_label]
    weight = torch.FloatTensor(weight)
    s = torch.utils.data.sampler.WeightedRandomSampler(weight, num_images)
    return s

    # train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
    #                                        sampler=sampler, num_workers=args.workers, pin_memory=True)
