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
    
    if (box2_x1 >= box1_x2 or box1_x1 >= box2_x2):
        return 0

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
                keep[box1_idx+box2_idx] += -100
                
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

def check_nms(indices1, indices2):
    if len(indices1) != len(indices2):
        print(f"len(indices1): {len(indices1)}, len(indicese2): {len(indices2)}")
        print("Wrong!! length!!")
        return False
    
    Num = len(indices1)
    for n in range(Num):
        if indices1[n] != indices2:
            print(f"my_nms: {indices1[n]}, but pytorch_nms: {indices2[n]}")
            print("Wrong!! index!!")
            return False
        
    print("Correct!!")
    return True

# test_data = [[350.9821, 161.8200, 369.9685, 205.2372], 
#              [250.5236, 154.2844, 274.1773, 204.9810], 
#              [471.4920, 160.4118, 496.0094, 213.4244], 
#              [352.0421, 164.5933, 366.4458, 205.9624], 
#              [166.0765, 169.7707, 183.0102, 232.6606], 
#              [252.3000, 183.1449, 269.6541, 210.6747], 
#              [469.7862, 162.0192, 482.1673, 187.0053], 
#              [168.4862, 174.2567, 181.7437, 232.9379], 
#              [470.3290, 162.3442, 496.4272, 214.6296], 
#              [251.0450, 155.5911, 272.2693, 203.3675], 
#              [252.0326, 154.7950, 273.7404, 195.3671], 
#              [351.7479, 161.9567, 370.6432, 204.3047],
#              [496.3306, 161.7157, 515.0573, 210.7200], 
#              [471.0749, 162.6143, 485.3374, 207.3448], 
#              [250.9745, 160.7633, 264.1924, 206.8350]]

# test_data_scores = [0.1919, 0.3293, 0.0860, 0.1600, 0.1885, 
#                     0.4297, 0.0974, 0.2711, 0.1483, 0.1173, 
#                     0.1034, 0.2915, 0.1993, 0.0677, 0.3217]

test_data = []
for i in range(53):
    test_data.append(generate_bbox(500, 500))
    
# print(test_data)

test_data_scores = [0.1919, 0.3293, 0.0860, 0.1600, 0.1885, 0.4297, 0.0974, 0.2711,
      0.1483, 0.1173, 0.1034, 0.2915, 0.1993, 0.0677, 0.3217, 0.0966, 0.0526,
      0.5675, 0.3130, 0.1592, 0.1353, 0.0634, 0.1557, 0.1512, 0.0699, 0.0545,
      0.2692, 0.1143, 0.0572, 0.1990, 0.0558, 0.1500, 0.2214, 0.1878, 0.2501,
      0.1343, 0.0809, 0.1266, 0.0743, 0.0896, 0.0781, 0.0983, 0.0557, 0.0623,
      0.5808, 0.3090, 0.1050, 0.0524, 0.0513, 0.4501, 0.4167, 0.0623, 0.1749]

box_indices = nms_cpu(test_data, test_data_scores, 0.5)

print(box_indices)

out = torchvision.ops.nms(torch.tensor(test_data).to('cuda'), torch.tensor(test_data_scores).to('cuda'), iou_threshold=0.5)

print(out)

check_nms(box_indices, list(out))