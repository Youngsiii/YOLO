import torch
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches







def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """
    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would be have (N, S, S, 4) in shape
    # boxes_preds: (N, S, S, 4)
    # boxes_labels: (N, S, S, 4)

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)  # use torch.max for tensor, torch.max support compare two tensor
    y1 = torch.max(box1_y1, box2_y1)  # don't use max, max just support one iterable to find the max number in it
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = ((x2 - x1).clamp(0)) * ((y2 - y1).clamp(0))
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)







# bboxes:[class_pred, prob_score, x1, y1, x2, y2] for each box -> [[class_pred, prob_score, x1, y1, x2, y2],[],[],[],[]]
def nms(bboxes, iou_threshold, prob_threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each box specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        prob_threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes


    Returns:
        list: bboxes after performing NMS given a specific IoU threshold


    """
    assert type(bboxes) == list
    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)   # from big prob_threshold to small
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [box
                  for box in bboxes
                  if box[0] != chosen_box[0]
                  or intersection_over_union(
                        torch.tensor(box[2:]),
                        torch.tensor(chosen_box[2:]),
                        box_format=box_format) < iou_threshold
                  ]

        bboxes_after_nms.append(chosen_box)


    return bboxes_after_nms











# pred_boxes:[[train_idx, class_pred, prob_score, x1, y1, x2, y2],[],[],[],[]]

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=20):
    """
    Calculates mean average precision

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each box specified as [train_idx, class_pred, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold

    """

    # list storing all AP for respective classes
    average_precisions = []
    # used for numerical stability later on
    epsilon = 1e-6
    for c in range(num_classes):   # get AP for each class
        # Go through all predictions and targets, and only add the ones belong to the current class c
        detections = [detection for detection in pred_boxes if detection[1] == c]
        ground_truths = [true_box for true_box in true_boxes if true_box[1] == c]
        total_true_bboxes = len(ground_truths)
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        # sort by box probabilities which is index 2
        detections = sorted(detections, key=lambda x: x[2], reverse=True)   # from big to small
        TP = torch.zeros((len(detections)))   # [0,0,0,0,...,0]   torch.zeros((3))在创建一维张量时候与torch.zeros(3)是一样的，但是在创建多维张量时必须用元组tuple,
        FP = torch.zeros((len(detections)))   # [0,0,0,0,...,0]   因此为了养成习惯，即使是一维也用tuple

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3, img 1 has 5
        # then we will obtain a dictionary with amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])   # {0:3, 1:5}

        # We then go through each key, val in the dictionary
        # and convert to the following (w.r.t same examplt)
        # amount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)            # {0:[0,0,0], 1:[0,0,0,0,0]}

        for detecion_idx, detection in enumerate(detections):  # to validate detection TP or FP
            train_idx = detection[0]  # img
            # Only take out the ground_truths that have the same training idx (image) as detection
            truth_boxes = [gt for gt in ground_truths if gt[0] == train_idx]   # get GT in train_idx image

            num_gts = len(truth_boxes)
            best_iou = 0
            for gt_idx, gt in enumerate(truth_boxes):
                iou = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)
                if iou > best_iou:
                    best_iou = iou
                    best_iou_idx = gt_idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[train_idx][best_iou_idx] == 0:
                    TP[detecion_idx] = 1
                    amount_bboxes[train_idx][best_iou_idx] = 1
                else:
                    FP[detecion_idx] = 1
            # if IOU is lower then the detection is a false positive
            else:
                FP[detecion_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)   # [1, 0, 1, 1, 0, 0] -> [1, 1, 2, 3, 3, 3]
        FP_cumsum = torch.cumsum(FP, dim=0)   # [0, 1, 0, 0, 1, 1] -> [0, 1, 1, 1, 2, 3]
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        AP = torch.trapz(precisions, recalls)
        average_precisions.append(AP)

    return sum(average_precisions) / len(average_precisions)

#
# def plot_image(image, boxes):
#     """Plots predicted bounding boxes on the image"""
#     im = np.array(image)
#     height, width, _ = im.shape
#
#     # Create figure and axes
#     fig, ax = plt.subplots(1)
#     # Display the image
#     ax.imshow(im)
#
#     # box[0] is x midpoint, box[2] is width
#     # box[1] is y midpoint, box[3] is height
#
#     # Create a Rectangle potch
#     for box in boxes:
#         box = box[2:]
#         assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
#         upper_left_x = box[0] - box[2] / 2
#         upper_left_y = box[1] - box[3] / 2
#         rect = patches.Rectangle(
#             (upper_left_x * width, upper_left_y * height),
#             box[2] * width,
#             box[3] * height,
#             linewidth=1,
#             edgecolor="r",
#             facecolor="none",
#         )
#         # Add the patch to the Axes
#         ax.add_patch(rect)
#
#
#     plt.show()

#
# def get_bboxes(loader, model, iou_threshold, threshold, pred_format="cells", box_format="midpoint", device="cpu"):
#     all_pred_boxes = []    # get all pred boxes
#     all_true_boxes = []    # get all true boxes
#     # make sure model is in eval before get bboxes
#     model.eval()
#     train_idx = 0   # initialize train_idx = 0, train_idx:0,1,2,3,4,...
#
#     for batch_idx, (x, labels) in enumerate(loader):    # x:images, labels:label_matrixes  batch_size:0,1,2,...,(num_batch-1)
#         x = x.to(device)
#         labels = labels.to(device)
#
#         with torch.no_grad():
#             predictions = model(x)  # (BS, S*S*30)
#
#         batch_size = x.shape[0]     # BS
#         true_bboxes = cellboxes_to_boxes(labels)    # labels:(BS, S, S, 30) -> list: [BS, S*S, 6]
#         bboxes = cellboxes_to_boxes(predictions)    # predictions:(BS, S*S*30) -> list: [BS, S*S, 6]
#
#         for idx in range(batch_size):     # idx:0,1,2,3,...,(BS-1)
#             nms_boxes = nms(bboxes[idx], iou_threshold, threshold, box_format)   # list: [<S*S , 6] for idx_sample
#
#             for nms_box in nms_boxes:
#                 all_pred_boxes.append([train_idx]+nms_box)   # list: [<S*S, 7]  [class_pred, conf, x, y, w, h] -> [train_idx, class_pred, conf, x, y, w, h]
#
#             for box in true_bboxes[idx]:
#                 if box[1] > threshold:
#                     all_true_boxes.append([train_idx]+box)   # list: [<S*S, 7]   for the same image, the train_idx must be same for all_pred_boxes and all_true_boxes
#
#             train_idx += 1   # train_idx = 0,1,2,..., (num_batch*BS-1)
#     model.train()
#
#     # the number of all_pred_boxes is usually different from the number of all_true_boxes
#     return all_pred_boxes, all_true_boxes     # list: [num_batch, BS, <S*S(*), 7]    list: [num_batch, BS, <S*S(#), 7]
#



# def convert_cellboxes(predictions, S=7):
#     # predictions:(BS, S*S*30)
#     predictions = predictions.to("cpu")  # (BS, S*S*30)
#     batch_size = predictions.shape[0]    # BS
#     predictions = predictions.reshape(batch_size, S, S, 30)  # (BS, S*S*30) -> (BS, S, S, 30)
#     bboxes1 = predictions[..., 21:25]    # (BS, S, S, 4)
#     bboxes2 = predictions[..., 26:30]    # (BS, S, S, 4)
#     scores = torch.cat((predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0)   # (BS, S, S)->(1, BS, S, S)->(2, BS, S, S)
#     best_confidence, best_box = torch.max(scores, dim=0)  # (BS, S, S), (BS, S, S)
#     best_box = best_box.unsqueeze(-1)    # (BS, S, S, 1)
#     best_confidence = best_confidence.unsqueeze(-1)  # (BS, S, S, 1)
#     best_boxes = best_box * bboxes2 + (1-best_box) * bboxes1    # (BS, S, S, 4)
#     x_cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(3)   # (BS, 7, 7) -> (BS, 7, 7, 1)
#     y_cell_indices = x_cell_indices.permute(0, 2, 1, 3)         # (BS, 7*, 7#, 1) -> (BS, 7#, 7*, 1)
#     x_image = 1 / S * (best_boxes[..., 0:1] + x_cell_indices)   # x_cell -> x_image   (BS, S, S, 1)
#     y_image = 1 / S * (best_boxes[..., 1:2] + y_cell_indices)   # y_cell -> y_image   (BS, S, S, 1)
#     w_h_image = 1 / S * best_boxes[..., 2:4]                    # w_h_cell -> w_h_image   (BS, S, S, 2)
#     converted_bboxes = torch.cat((x_image, y_image, w_h_image), dim=-1)    # (BS, S, S, 4)
#     predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)       # (BS, S, S, 20)->(BS, S, S)->(BS, S, S, 1)
#     converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)   # (BS, S, S, 1+1+4) -> (BS, S, S, 6)
#     return converted_preds   # (BS, S, S, 6)

#
# def cellboxes_to_boxes(out, S=7):   # out is Yolo out (BS, S*S*30)
#     converted_pred = convert_cellboxes(out).reshape(out.shape[0], S*S, -1)   # (BS, S*S*30)->(BS, S, S, 6)->(BS, S*S, 6)
#     converted_pred[..., 0] = converted_pred[..., 0].long()   # class_label = class_label.long()
#     all_boxes = []    # convert (BS, S*S, 6) -> list:[[],[],...,[]]
#
#     for ex_idx in range(out.shape[0]):    # ex_idx: 0,1,2,3,...,(BS-1)
#         bboxes = []                       # the sample of No. ex_idx -> bboxes=[] -> bboxes=[[class_pred, conf, x, y, w, h] * (S*S)]
#         for bbox_idx in range(S * S):     # the cell box of No. bbox_idx
#             bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])  # [class_pred, conf, x, y, w, h]
#
#         all_boxes.append(bboxes)
#
#     return all_boxes   # all_boxes = [bboxes * BS] = [[class_pred, conf, x, y, w, h] * (S*S) * BS]    list: [BS, S*S, 6]
#

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])





"""
训练结束后，保存模型的状态字典和优化器的状态字典
state_dict = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "loss": loss}
torch.save(state_dict, "path/to/your/model.pth")    # 可以选择.pt或.pth扩展名，但实际扩展名并不影响保存和加载过程

加载模型状态字典
checkpoint = torch.load("path/to/your/model_pth")  # checkpoint == state_dict
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])




也可以保存整个模型，而不仅仅保存状态字典，但是不是很推荐，因为状态字典更灵活并且占用空间更小
torch.save(model, "path/to/your/model.pth")

加载整个模型的时候不需要重构模型
model = torch.load("path/to/your/model.pth")
"""


def convert_cellboxes(predictions, S=7):
    """
    将YOLO预测输出(BS, S*S*30) 提取出预测类别pred_class，每个cell最好的box置信度best_confidence和每个cell最好的box坐标
    box的坐标需要从相对于cell归一化转变为相对于image归一化
    输出张量形状为(BS, S, S, 6)

    """
    # preditions:(BS, S*S*30)  Yolo outout want to convert predicitions to boxes list:(BS, S, S, 6)  6:[pred_class, score, x_image, y_image, w_image, h_image]
    predictions = predictions.to('cpu')
    BS = predictions.shape[0]
    predictions = predictions.reshape(BS, S, S, 30)  # (BS, S, S, 30)
    bboxes1 = predictions[..., 21:25]   # (BS, S, S, 4)
    bboxes2 = predictions[..., 26:30]   # (BS, S, S, 4)
    scores = torch.cat((predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0)   # (BS, S, S),(BS, S, S)->(1, BS, S, S),(1, BS, S, S)->(2, BS, S, S)
    best_box = scores.argmax(0).unsqueeze(-1)  # (BS, S, S)->(BS, S, S, 1)   dim为1的值:0 or 1
    best_boxes = best_box * bboxes2 + (1-best_box) * bboxes1   # (BS, S, S, 4)  4: [x_cell, y_cell, w_cell, h_cell]
    x_cell_indices = torch.arange(7).repeat(BS, 7, 1).unsqueeze(-1)    # (BS, 7, 7)->(BS, 7, 7, 1)
    y_cell_indices = x_cell_indices.permute(0, 2, 1, 3)   # (BS, 7*, 7#, 1)->(BS, 7#, 7*, 1)
    x_image = (best_boxes[..., 0:1] + x_cell_indices)/S     # (BS, 7, 7, 1)
    y_image = (best_boxes[..., 1:2] + y_cell_indices)/S     # (BS, 7, 7, 1)
    w_h_image = best_boxes[..., 2:4]/S  # (BS, 7, 7, 2)
    converted_bboxes = torch.cat((x_image, y_image, w_h_image), dim=-1)   # (BS, 7, 7, 4)
    pred_class = predictions[..., 0:20].argmax(-1).unsqueeze(-1)     # (BS, 7, 7)->(BS, 7, 7, 1)   dim为1的值:0,1,2,3,4,...,19
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)   # (BS, 7, 7)->(BS, 7, 7, 1)
    converted_preds = torch.cat((pred_class, best_confidence, converted_bboxes), dim=-1)  # (BS, 7, 7, 1),(BS, 7, 7, 1),(BS, 7, 7, 4)->(BS, 7, 7, 6)
    return converted_preds   # (BS, 7, 7, 6)



def cellboxes_to_boxes(out, S=7):
    """
    YOLO预测输出转换为列表
    YOLO预测输出张量(BS, S*S*30)
    列表形状[BS, S*S, 6] dim=6的值为列表[pred_class, best_confidence, x_image, y_image, w_image, h_image]

    """
    # out: (BS, S*S*30)  Yolo output
    # boxes tensor: (BS, S, S, 6) -> boxes list: [BS, S*S, 6]  6: [class_pred, confidence, x_image, y_image, w_image, h_image]
    BS = out.shape[0]
    converted_pred = convert_cellboxes(out).reshape(BS, S*S, -1)   # (BS, S*S, 6)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(BS):
        bboxes = []
        for bbox_idx in range(S*S):
            box = converted_pred[ex_idx, bbox_idx, :]   # tensor [class_pred, confidence, x_image,y_image,w_image,h_image]
            bboxes.append([x.item() for x in box])     # bboxes = [[class_pred, confidence, x_image,y_image,w_image,h_image],[],[],[],[]] for batch ex_idx  [1, S*S, 6]

        all_bboxes.append(bboxes)   # list shape:[BS, S*S, 6]

    return all_bboxes


def get_bboxes(loader, model, iou_threshold=0.5, prob_threshold=0.3, box_format="midpoint", device="cuda", pred_format="cells"):
    all_pred_boxes = []    # 存储所有的预测边界框列表[[train_idx, pred_class, confidence, x_img, y_img, w_img, h_img],[],[],...,[]]
    all_true_boxes = []    # 存储所有的目标边界框列表[[train_idx, pred_class, confidence, x_img, y_img, w_img, h_img],[],[],...,[]]
    model.eval()  # 将模型设置为验证模式
    train_idx = 0   # 表示这个框在第几张图片上

    for batch_idx, (imgs, labels) in enumerate(loader):    # batch_idx:0,1,2,3,4,...
        imgs = imgs.to(device)    # (BS, 3, H, W)
        labels = labels.to(device)    # (BS, S, S, 30)
        BS = imgs.shape[0]


        with torch.no_grad():    # 上下文管理器停用梯度计算
            predictions = model(imgs)   # (BS, S*S*30)

        pred_bboxes = cellboxes_to_boxes(predictions)   # pred tensor (BS, S*S*30) -> lists of boxes [BS, S*S, 6]
        true_bboxes = cellboxes_to_boxes(labels)        # true tensor (BS, S, S, 30) -> lists of boxes [BS, S*S, 6]

        for idx in range(BS):   # 遍历每张图片,对图片上的预测边界框和目标边界框进行筛选
            idx_pred_boxes = pred_bboxes[idx]    # lists [S*S, 6]
            nms_idx_boxes = nms(idx_pred_boxes, iou_threshold, prob_threshold, box_format)   # lists [<S*S, 6]
            for box in nms_idx_boxes:
                all_pred_boxes.append([train_idx] + box)    # lists [..., 7]

            idx_true_boxes = true_bboxes[idx]    # lists [S*S, 6]
            for box in idx_true_boxes:
                if box[1] > prob_threshold:
                    all_true_boxes.append([train_idx] + box)   # lists [<S*S, 6]

            train_idx += 1
    model.train()   # 将模型恢复为训练模式
    return all_pred_boxes, all_true_boxes


def plot_image(image, boxes):   # image只有一张，boxes为box的列表，每个box为[pred_class, confidence, x_img, y_img, w_img, h_img]
    image = np.array(image)      # 图像对象转换为Numpy数组 [H, W, C]
    height, width, _ = image.shape
    class_list = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                  "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    fig, ax = plt.subplots(1)  # 创建图像窗口和坐标轴对象，图像窗口中只有一个子图
    ax.imshow(image)   # 在坐标轴ax中显示图像

    for box in boxes:
        category_name = int(box[0])
        box = box[2:]   # box:[x_img, y_img, w_img, h_img]
        upper_left_x = box[0] - (box[2]/2)
        upper_left_y = box[1] - (box[3]/2)
        rect = patches.Rectangle(xy=(upper_left_x * width, upper_left_y * height), width=box[2] * width, height=box[3] * height, linewidth=1, edgecolor='r', facecolor="none")
        ax.add_patch(rect)
        ax.text(upper_left_x * width, upper_left_y * height, class_list[category_name], color='red', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5))
    plt.show()






