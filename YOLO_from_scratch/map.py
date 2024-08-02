import torch
from iou import intersection_over_union
from collections import Counter


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














