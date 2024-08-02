"""
Main file for training Yolo model on Pascal VOC dataset
"""
import torch
from torchvision.transforms import transforms
from tqdm import tqdm

from torch.utils.data import DataLoader
from dataset import VOCDataset
from model import Yolov1
from loss import  YoloLoss
from utils import get_bboxes
from map import mean_average_precision
from utils import cellboxes_to_boxes
from utils import nms
from utils import plot_image


# 设置随机种子，使得torch随机产生的一串随机数是相同的，以便实验复现
seed = 123
torch.manual_seed(seed)


# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 16  # 64 in original paper
BATCH_SIZE = 16
WEIGHT_DECAY = 0
# EPOCHS = 1000
EPOCHS = 10
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = r"F:\Pycharm\pycharm_project_2\my_yolo\pth\overfit_100examples_230epochs.pth"
IMG_DIR = r'F:\Pycharm\pycharm_project_2\my_yolo\VOC\images'
LABEL_DIR = r'F:\Pycharm\pycharm_project_2\my_yolo\VOC\labels'


def train_fn(train_loader, model, optimizer, loss_fn):
    mean_loss = []
    train_loader_tqdm = tqdm(train_loader, leave=True)
    for batch_idx, (imgs, labels) in enumerate(train_loader_tqdm):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        # forward propagation
        preds = model(imgs)
        loss = loss_fn(preds, labels)
        mean_loss.append(loss.item())

        # backward propagation
        optimizer.zero_grad()
        loss.backward()    # retain_graph=True
        optimizer.step()

        train_loader_tqdm.set_postfix(loss=loss.item())   # set postfix for train_loader_tqdm
        print(f"Mean Loss: {sum(mean_loss)/len(mean_loss)}")


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms   # list of transforms

    def __call__(self, img, boxes):
        for t in self.transforms:
            img, boxes = t(img), boxes

        return img, boxes


transforms = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


train_set = VOCDataset(csv_file=r'F:\Pycharm\pycharm_project_2\my_yolo\VOC\100examples.csv', img_dir=IMG_DIR, label_dir=LABEL_DIR, transform=transforms)
val_set = VOCDataset(csv_file=r'F:\Pycharm\pycharm_project_2\my_yolo\VOC\100examples.csv', img_dir=IMG_DIR, label_dir=LABEL_DIR, transform=transforms)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)



loss_fn = YoloLoss()   # S=7, B=2, C=20




def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)  # in_channels=3, architecture_config=architecture_config, **kwargs -> split_size=7, num_boxes=2, num_classes=20
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    for epoch in range(EPOCHS):

        train_fn(train_loader, model, optimizer, loss_fn)
        all_pred_boxes, all_true_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4, device="cpu")
        map = mean_average_precision(all_pred_boxes, all_true_boxes, iou_threshold=0.5, box_format="midpoint")

        if map > 0.9:
            checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
            torch.save(checkpoint, LOAD_MODEL_FILE)

        print(f"epoch: {epoch}, map: {map}")



def test():
    if LOAD_MODEL == True:
        model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        checkpoint = torch.load(LOAD_MODEL_FILE, map_location=torch.device("cpu"))
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for epoch in range(EPOCHS):
            for x, y in train_loader:
               x = x.to(DEVICE)
               for idx in range(8):
                   bboxes = cellboxes_to_boxes(model(x))
                   bboxes = nms(bboxes[idx], iou_threshold=0.9, prob_threshold=0.05, box_format="midpoint")
                   plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

               import sys
               sys.exit()

            # pred_boxes, target_boxes = get_bboxes(
            #     train_loader, model, iou_threshold=0.5, threshold=0.4
            # )
            #
            # mean_avg_prec = mean_average_precision(
            #     pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            # )
            # print(f"Train mAP: {mean_avg_prec}")


if __name__ == '__main__':
    # main()
    test()
