import torch
import torchvision
import random

def Area(box):
    # area of box
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    
    assert x2 > x1 and y2 > y1
    
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
    
    # if (box2_x1 >= box1_x2 or box1_x1 >= box2_x2):
    #     return 0

    left_top_x = max(box1_x1, box2_x1)
    left_top_y = max(box1_y1, box2_y1)
    
    right_bottom_x = min(box1_x2, box2_x2)
    right_bottom_y = min(box1_y2, box2_y2)
    
    if (right_bottom_x - left_top_x <= 0 or right_bottom_y - left_top_y):
        return 0
    
    # left_top_x = box1_x1 if box1_x1 >= box2_x1 else box2_x1
    # left_top_y = box1_y1 if box1_y1 >= box2_y1 else box2_y1
    
    # right_bottom_x = box1_x2 if box1_x2 <= box2_x2 else box2_x2
    # right_bottom_y = box1_y2 if box1_y2 <= box2_y2 else box2_y2
    
    
    inter_area = Area([left_top_x, left_top_y, right_bottom_x, right_bottom_y])
    
    iou = inter_area / (Area(box1) + Area(box2) - inter_area)
    
    return iou

def nms_cpu(boxes,
            scores,
            threshold):
    
    assert len(boxes) == len(scores)
    
    Num_Boxes = len(scores)
    
    sorted_boxes = [[n, [scores[n],
                     boxes[n][0],
                     boxes[n][1],
                     boxes[n][2],
                     boxes[n][3]]] for n in range(Num_Boxes)]
    
    # print("before sort:")
    # print(sorted_boxes)
    # rarange boxes by score
    sorted_boxes = sorted(sorted_boxes,
                          key=lambda bbox: bbox[1][0],
                          reverse=True)
    
    # print("after sort:")
    # print(sorted_boxes)
    
    boxes = [[sorted_boxes[n][1][1],
              sorted_boxes[n][1][2],
              sorted_boxes[n][1][3],
              sorted_boxes[n][1][4]]
              for n in range(Num_Boxes)]
    
    # print("final data:")
    # print(boxes)
    
    keep = [ sorted_boxes[n][0] for n in range(Num_Boxes)]
    
    # print("keep index:")
    # print(keep)
    # keep = [keep[0]]
    for box1_idx, box1 in enumerate(boxes):
        for box2_idx ,box2 in enumerate(boxes[box1_idx+1:]):
            if IOU(box1, box2) > threshold:
                # keep.append(box1_idx + box2_idx)
                keep[box1_idx+box2_idx] = -10000
                
    # print(keep)
    keep = [ e for e in keep if e > 0 ]
                
    return keep

def generate_bbox(height, width):
    x1 = random.uniform(0, width-0.0001)
    y1 = random.uniform(0, height-0.0001)
    
    x2 = random.uniform(x1+0.0001, width)
    y2 = random.uniform(y1+0.0001, height)
    
    assert x1 < x2 and y1 < y2
    bbox = [x1, y1, x2, y2]
    return bbox

def generate_scores(length):
    return [ random.uniform(0.001, 0.999) for _ in range(length)]

def check_nms(indices1, indices2):
    if len(indices1) != len(indices2):
        print(f"len(indices1): {len(indices1)}, len(indicese2): {len(indices2)}")
        print("Wrong!! length!!")
        return False
    
    Num = len(indices1)
    for n in range(Num):
        if indices1[n] != indices2:
            print(f"indices1: {indices1[n]}, but indices2: {indices2[n]}")
            print("Wrong!! index!!")
            return False
        
    print("Correct!!")
    return True


test_data = []
for i in range(5000):
    test_data.append(generate_bbox(500, 500))
    
test_data_scores = generate_scores(5000)
    
cpu_out = nms_cpu(test_data, test_data_scores, 0.5)

print(cpu_out[:50])

torch_out = torchvision.ops.nms(torch.tensor(test_data), torch.tensor(test_data_scores), iou_threshold=0.5)

print(torch_out[:50])

torch_gpu_out = torchvision.ops.nms(torch.tensor(test_data).to('cuda'), torch.tensor(test_data_scores).to('cuda'), iou_threshold=0.5)

print(torch_gpu_out[:50])

check_nms(cpu_out, list(torch_out))
print("=======================")
check_nms(cpu_out, list(torch_gpu_out))
print("=======================")
check_nms(list(torch_out), list(torch_gpu_out))
