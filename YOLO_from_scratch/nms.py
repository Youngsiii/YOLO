
import torch
from .iou import intersection_over_union

# bboxes:[class_pred, prob_score, x1, y1, x2, y2] for each box -> [[class_pred, prob_score, x1, y1, x2, y2],[],[],[],[]]
def nms(bboxes, iou_threshold, prob_threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each box specified as [class_pred, prob_score, x1, y1, x2, y2], num = S*S
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

















