import torch
import torchvision
import numpy as np
import random
import time

def Area(box):
    # area of box
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    
    assert x2 > x1 and y2 > y1
    assert x1 >= 0 and y1 >= 0
    
    width = x2 - x1
    height = y2 - y1
    
    return width * height

def IOU(box1, box2):
    # IOU of box1 and box2
    box1_x1 = box1[0]
    box1_y1 = box1[1]
    box1_x2 = box1[2]
    box1_y2 = box1[3]

    box2_x1 = box2[0]
    box2_y1 = box2[1]
    box2_x2 = box2[2]
    box2_y2 = box2[3]
    
    assert box1_x2 > box1_x1 and box1_y2 > box1_y1
    assert box2_x2 > box2_x1 and box2_y2 > box2_y1 
    
    left_top_x = max(box1_x1, box2_x1)
    left_top_y = max(box1_y1, box2_y1)
    
    right_bottom_x = min(box1_x2, box2_x2)
    right_bottom_y = min(box1_y2, box2_y2)
    
    if (right_bottom_x - left_top_x < 0 or right_bottom_y - left_top_y < 0):
        return 0
        
    assert left_top_x >= 0 and left_top_y >= 0
    
    inter_area = Area([left_top_x, left_top_y, right_bottom_x, right_bottom_y])
    Area_of_box1 = Area(box1)
    Area_of_box2 = Area(box2)
    
    iou = inter_area / (Area_of_box1 + Area_of_box2 - inter_area)
    
    return iou

def my_nms_cpu(boxes, scores, threshold):
    
    assert len(boxes) == len(scores)
    
    origin_boxes = boxes.copy()
    
    Num_Boxes = len(scores)
    
    sorted_boxes = [[n, [scores[n], # (N, 1 + 4)
                     boxes[n][0],
                     boxes[n][1],
                     boxes[n][2],
                     boxes[n][3]]] for n in range(Num_Boxes)]
    
    sorted_boxes = sorted(sorted_boxes,
                          key=lambda bbox: bbox[1][0],
                          reverse=True)
    
    sorted_index = [sorted_boxes[n][0] for n in range(Num_Boxes)] # (N, ) // sorted_index
    
    keep_index = sorted_index.copy()
    meta_index_list = sorted_index.copy()
    
    while len(meta_index_list) > 0:
        idx_self = meta_index_list[0]
        for idx_other in meta_index_list[1:]:
            if IOU(origin_boxes[idx_self], origin_boxes[idx_other]) > threshold:
                meta_index_list.remove(idx_other)
                keep_index.remove(idx_other)
        meta_index_list.remove(idx_self)

    return keep_index

def generate_bbox(height, width):
    offset = random.uniform(0, 0.1)
    x1 = random.uniform(0, width-offset)
    y1 = random.uniform(0, height-offset)
    
    x2 = random.uniform(x1+offset, width)
    y2 = random.uniform(y1+offset, height)
    
    assert x1 < x2 and y1 < y2
    bbox = [x1, y1, x2, y2]
    
    return bbox

def generate_scores(length):
    return [ random.uniform(0.001, 0.999) for _ in range(length) ]

def check_nms(indices1, indices2):    
    print(f"len of indices1: {len(indices1)}, len of indicese2: {len(indices2)}")
    if len(indices1) != len(indices2):
        print("Wrong!! length!!")
        return False
    
    eqs = torch.equal(torch.tensor(indices1).to('cpu'), torch.tensor(indices2).to('cpu'))
    print(eqs)
    
    if not eqs:
        Num = len(indices1) 
        for n in range(Num):
            if indices1[n] not in indices2:
                print(f"indices1[{n}]: {indices1[n]} is not in indices2!!")
                return False
    return True

def nms_cpu_refer(boxes, confs, nms_thresh=0.5, min_mode=False):    
    # print(boxes.shape)    
    x1 = boxes[:, 0]    
    y1 = boxes[:, 1]    
    x2 = boxes[:, 2]    
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)    
    order = confs.argsort()[::-1]
    
    keep = []   
     
    while order.size > 0:        
        idx_self = order[0]        
        idx_other = order[1:]
        keep.append(idx_self)
        xx1 = np.maximum(x1[idx_self], x1[idx_other])        
        yy1 = np.maximum(y1[idx_self], y1[idx_other])        
        xx2 = np.minimum(x2[idx_self], x2[idx_other])        
        yy2 = np.minimum(y2[idx_self], y2[idx_other])
        w = np.maximum(0.0, xx2 - xx1)        
        h = np.maximum(0.0, yy2 - yy1)        
        inter = w * h
        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:            
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)
        inds = np.where(over <= nms_thresh)[0]        
        order = order[inds + 1]
                 
    return np.array(keep)
        
test_data = []
counts_of_boxes = 5000
image_size = (10240, 10240)
for i in range(counts_of_boxes):
    test_data.append(generate_bbox(image_size[1], image_size[0]))
    
test_data_scores = generate_scores(counts_of_boxes)
print("=======================")
print(f"Size of image: {image_size}")
print(f"Counts of bounding boxes in the image: {counts_of_boxes}\n")
print("Test data ready.")

my_nms_cpu_start = time.time()
cpu_out = my_nms_cpu(test_data, test_data_scores, threshold=0.5)
my_nms_cpu_end = time.time()
# print(cpu_out[-30:])

refer_nms_start = time.time()
cpu_refer_out= nms_cpu_refer(np.array(test_data), np.array(test_data_scores), nms_thresh=0.5)
refer_nms_end = time.time()
# print(list(cpu_refer_out)[-30:])


torch_cpu_start = time.time()
torch_out = torchvision.ops.nms(torch.tensor(test_data), torch.tensor(test_data_scores), iou_threshold=0.5)
torch_cpu_end = time.time()
# print(list(torch_out.numpy())[-30:])

torch_gpu_start = time.time()
torch_gpu_out = torchvision.ops.nms(torch.tensor(test_data).to('cuda'), torch.tensor(test_data_scores).to('cuda'), iou_threshold=0.5)
torch_gpu_end = time.time()
# print(list(torch_gpu_out.cpu().numpy())[:40])

print("=======================")
check_nms(list(cpu_out), list(cpu_refer_out))
print("=======================")
check_nms(list(cpu_out), list(torch_out))
print("=======================")
check_nms(list(cpu_refer_out), list(torch_out))
print("=======================")
check_nms(list(cpu_out), list(torch_gpu_out.to('cpu')))
print("=======================")
check_nms(list(torch_out), list(torch_gpu_out.to('cpu')))
print("=======================")

print("time of nms_reference(numpy): {}s".format(refer_nms_end - refer_nms_start))
print("time of my_nms_cpu: {}s".format(my_nms_cpu_end - my_nms_cpu_start))
print("time of torch_cpu: {}s".format(torch_cpu_end - torch_cpu_start))
print("time of torch_gpu: {}s".format(torch_gpu_end - torch_gpu_start))
