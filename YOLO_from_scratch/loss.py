"""
Implementation of Yolo loss Function from the original yolo paper
"""
import torch
from torch import nn
from iou import intersection_over_union

class YoloLoss(nn.Module):
    # Calculate the loss for yolo (v1) model
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S  # split size of image (in paper 7)
        self.B = B  # number of boxes (in paper 2)
        self.C = C  # number of classes (in paper and VOC dataset is 20)
        self.lamda_noobj = 0.5  # These are from Yolo paper, signifying how much we should pay
        self.lamda_coord = 5    # loss for no object (noobj) and the box coordinates (coord)

    def forward(self, predictions, targets):
        # predictions:(BS, S*S*30) -> (BS, S, S, 30)
        # targets:(BS, S, S, 25)
        # predictions are shaped (BS, S*S*(C+B*5)) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C+self.B * 5)
        iou_box1 = intersection_over_union(predictions[..., 21:25], targets[..., 21:25], box_format="midpoint")  # (BS, S, S, 4),(BS, S, S, 4)->(BS, S, S, 1)
        iou_box2 = intersection_over_union(predictions[..., 26:30], targets[..., 21:25], box_format="midpoint")  # (BS, S, S, 4),(BS, S, S, 4)->(BS, S, S, 1)
        # iou_box1 = iou_box1.unsqueeze(0)  # (1, BS, S, S, 1)
        # iou_box2 = iou_box2.unsqueeze(0)  # (1, BS, S, S, 1)
        ious = torch.cat([iou_box1.unsqueeze(0), iou_box2.unsqueeze(0)], dim=0)  # (2, BS, S, S, 1)

        # Take the box with highest IoU out of the two predictions
        # Note that best_box will be indices of 0,1 for which bbox was best
        max_iou, best_box = torch.max(ious, dim=0)     # (BS, S, S, 1),(BS, S, S, 1)
        exist_box = targets[..., 20:21]    # (BS, S, S, 1)



        # BOX COORDINATES
        # Set boxes with no object in them to 0. We only take out one of the two predictions,
        # which is the one with highest IoU calculated previously.
        predictions_box = exist_box * ((best_box * predictions[..., 26:30] + (1 - best_box) * predictions[..., 21:25])) # (BS, S, S ,4)
        # predictions_box = exist_box * predictions_box  # (BS, S, S, 4)
        targets_box = exist_box * targets[..., 21:25]  # (BS, S, S, 4)
        # targets_box = exist_box * targets_box  # (BS, S, S, 4)

        # Take sqrt of width, height of boxes
        predictions_box[..., 2:4] = torch.sign(predictions_box[..., 2:4]) * torch.sqrt(torch.abs(predictions_box[..., 2:4] + 1e-6))  # (BS, S, S, 2)   1e-6一定要放在torch.abs中
        targets_box[..., 2:4] = torch.sqrt(torch.abs(targets_box[..., 2:4] + 1e-6))
        coord_loss = self.mse(torch.flatten(predictions_box, end_dim=-2), torch.flatten(targets_box, end_dim=-2))  # (BS*S*S, 4), (BS*S*S, 4)

        # OBJECT LOSS
        predictions_obj_conf = exist_box * ((best_box * predictions[..., 25:26] + (1-best_box) * predictions[..., 20:21]))  # (BS, S, S, 1) * (BS, S, S, 1)
        targets_obj_conf = exist_box * targets[..., 20:21]  # (BS, S, S, 1) * (BS, S, S, 1)
        obj_conf_loss = self.mse(torch.flatten(predictions_obj_conf), torch.flatten(targets_obj_conf))

        # NO OBJECT LOSS
        targets_noobj_conf = (1-exist_box) * targets[..., 20:21]    # (BS, S, S, 1)
        predictions_noobj_conf1 = (1-exist_box) * predictions[..., 20:21]   # (BS, S, S, 1)
        predictions_noobj_conf2 = (1-exist_box) * predictions[..., 25:26]   # (BS, S, S, 1)

        noobj_conf_loss = self.mse(torch.flatten(predictions_noobj_conf1, start_dim=1), torch.flatten(targets_noobj_conf, start_dim=1))
        noobj_conf_loss += self.mse(torch.flatten(predictions_noobj_conf2, start_dim=1), torch.flatten(targets_noobj_conf, start_dim=1))

        # CLASS LOSS
        class_loss = self.mse(torch.flatten(exist_box * predictions[..., :20], end_dim=-2),      # (BS, S, S, 1)*(BS, S, S, 20)->(BS, S, S, 20)->(BS*S*S, 20)
                              torch.flatten(exist_box * targets[..., :20], end_dim=-2))        # (BS, S, S, 1)*(BS, S, S, 20)->(BS, S, S, 20)->(BS*S*S, 20)


        total_loss = ( self.lamda_coord * coord_loss + obj_conf_loss + self.lamda_noobj * noobj_conf_loss + class_loss )
        return total_loss


if __name__ == "__main__":
    loss_fn = YoloLoss()
    predictions = torch.randn((1, 7*7*30))
    targets = torch.randn((1, 7, 7, 25))
    loss = loss_fn(predictions, targets)
    # print(loss)
    print(loss.item())