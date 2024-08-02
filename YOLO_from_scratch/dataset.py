"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""
import os
import torch
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        # img    label
        # xxx1.jpg  xxx1.txt
        # xxx2.jpg  xxx2.txt
        # ......
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform


    def __len__(self):
        return len(self.annotations)    # len(df) return the data rows of the df == number of images/txts


    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])   # get xxxx.txt path  .ioc[index, 1] only for dataframe
        # xxxx.txt:
        # "class_label x y width height\n"  type=str
        # "class_label x y width height\n"  type=str
        # ......
        boxes = []   # all boxes for xxxx.txt xxxx.jpg  [[],[],[],...,[]] each box is [class_pred, x, y, width, height]
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [float(x) if float(x) != int(float(x)) else int(x) for x in label.replace('\n', '').split()]
                boxes.append([class_label, x, y, width, height])

        image_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(image_path)
        boxes = torch.tensor(boxes)  # list -> Tensor
        if self.transform:
            # image = self.transform(image)   # Image -> Tensor  # only run this file -> uncomment this line
            image, boxes = self.transform(image, boxes)

        # Convert boxes list to cell matrix (relative to image -> relative to cell)
        label_matrix = torch.zeros((self.S, self.S, self.C + self.B * 5))   # (S, S, 30)
        for box in boxes:
            box = box.tolist()   # Tensor -> list [class_label, x, y, width, height]  relative to image
            class_label, x, y, width, height = box
            class_label = int(class_label)
            x_cell = x * self.S
            y_cell = y * self.S
            # i,j represent the cell column (x) and cell rows (y)
            i = int(x_cell)
            j = int(y_cell)
            x_cell = x_cell - i
            y_cell = y_cell - j
            # Calculating the width and height of boungding box relative to the cell
            # is done by the following, with width as the example:
            # width_pixels = (width * self.image_width)
            # cell_pixels = (self.image_width)
            # Then to find the width relative to the cell is simply: width_pixels/cell_pixels,
            # simplification leads to the formulas below
            width_cell = width * self.S
            height_cell = height * self.S

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object per cell
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                label_matrix[i, j, 21:22] = x_cell
                label_matrix[i, j, 22:23] = y_cell
                label_matrix[i, j, 23:24] = width_cell
                label_matrix[i, j ,24:25] = height_cell
                # label_matrix[i, j, 21:25] = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1
                # label_matrix:(S, S, 30)
                # [0-19:class_label, 20:obj_conf, 21-24:[x_cell, y_cell, width_cell, height_cell], 25-29:[0,0,0,0,0]]

            return image, label_matrix


if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    train_set = VOCDataset(r'F:\Pycharm\pycharm_project_2\my_yolo\VOC\8examples.csv',
                           r'F:\Pycharm\pycharm_project_2\my_yolo\VOC\images',r'F:\Pycharm\pycharm_project_2\my_yolo\VOC\labels',transform=transform)
    image, label = train_set[1]  # image:(3, 448, 448)
    image = image.permute(1, 2, 0).numpy()  # (3, 448, 448) -> (448, 488, 3), Tensor->Numpy
    plt.imshow(image)  # plt可以用来直接显示张量图片(可能还需要.clamp(0, 1)将张量值转换到0-1之间，但此处使用了ToTensor()已转换张量值到0-1之间)，不过最好先转换为numpy形式
    plt.show()







