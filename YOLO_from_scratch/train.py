"""
Main file for training Yolo model on Pascal VOC dataset
"""
import torch
from torchvision.transforms import transforms
from tqdm import tqdm

from torch.utils.data import DataLoader
# from dataset import VOCDataset
# from model import Yolov1
# from loss import  YoloLoss
# from utils import get_bboxes
# from map import mean_average_precision


seed = 123
torch.manual_seed(seed)


# Hyperparameters
LEARNING_RATE = 2e-5
LEARNING_RATE = 2e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 16  # 64 in original paper
BATCH_SIZE = 16
WEIGHT_DECAY = 0
# EPOCHS = 1000
EPOCHS = 500
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
# LOAD_MODEL_FILE = r"/kaggle/working/overfit.pth.tar"
LOAD_MODEL_FILE = r"/kaggle/working/overfit.pth"
IMG_DIR = r'/kaggle/input/voc-dataset/images'
LABEL_DIR = r'/kaggle/input/voc-dataset/labels'


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


train_set = VOCDataset(csv_file=r'/kaggle/input/voc-dataset/100examples.csv', img_dir=IMG_DIR, label_dir=LABEL_DIR, transform=transforms)
val_set = VOCDataset(csv_file=r'/kaggle/input/voc-dataset/100examples.csv', img_dir=IMG_DIR, label_dir=LABEL_DIR, transform=transforms)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)


model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)     # in_channels=3, architecture_config=architecture_config, **kwargs -> split_size=7, num_boxes=2, num_classes=20
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = YoloLoss()   # S=7, B=2, C=20




def main():
    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)
        all_pred_boxes, all_true_boxes = get_bboxes(val_loader, model, iou_threshold=0.3, threshold=0.8, device="cuda")
        mean_ap = mean_average_precision(all_pred_boxes, all_true_boxes, iou_threshold=0.5, box_format="midpoint")
        print(f"epoch: {epoch}, map: {mean_ap}")

        if mean_ap > 0.99 and epoch > 120:
            checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
            torch.save(checkpoint, LOAD_MODEL_FILE)
            break






if __name__ == '__main__':
    main()
