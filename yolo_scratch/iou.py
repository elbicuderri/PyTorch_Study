import torch

def iou(box_pred, box_target, format="corner"):
    # box_pred : (N, 4)  N is the number of bboxes
    if format=="corner":
        box1_x1 = box_pred[..., 0:1] # maintain shape (N, 1)
        box1_y1 = box_pred[..., 1:2]
        box1_x2 = box_pred[..., 2:3]
        box1_y2 = box_pred[..., 3:4]
        
        box2_x1 = box_target[..., 0:1] # maintain shape (N, 1)
        box2_y1 = box_target[..., 1:2]
        box2_x2 = box_target[..., 2:3]
        box2_y2 = box_target[..., 3:4]
        
    elif format=="center":
        # (x, y, h, w)
        box1_x1 = box_pred[..., 0:1] - box_pred[..., 3:4] / 2
        box1_y1 = box_pred[..., 1:2] - box_pred[..., 2:3] / 2
        box1_x2 = box_pred[..., 0:1] + box_pred[..., 3:4] / 2
        box1_y2 = box_pred[..., 1:2] + box_pred[..., 2:3] / 2
        
        box2_x1 = box_target[..., 0:1] - box_target[..., 3:4] / 2
        box2_y1 = box_target[..., 1:2] - box_target[..., 2:3] / 2
        box2_x2 = box_target[..., 0:1] + box_target[..., 3:4] / 2
        box2_y2 = box_target[..., 1:2] + box_target[..., 2:3] / 2
        
        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)
        
        # .clamp(0) -> 
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        
        box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))
        
        return intersection / (box1_area + box2_area - intersection)

        
    