import cv2
import os
import json
import numpy as np
import pandas as pd
from glob import glob
import random
import shutil
import matplotlib.pyplot as plt

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

# Before
# - node1: Head
# - node2: Eye_R
# - node3: Eye_L  
# - node4: Neck
# - node5: Sholuder_R
# - node6: Elbow_high_R  ##
# - node7: Elbow_low_R  ##
# - node8: Wrist_high_R ##
# - node9: Wrist_low_R ##
# - node10: Hand_R 
# - node11: Sholuder_L 
# - node12: Elbow_high_L  ##
# - node13: Elbow_low_L  ##
# - node14: Wrist_high_L  ##
# - node15: Wrist_low_L  ##
# - node16: Hand_L 
# - node17: Pelvis 
# - node18: Hip_R 
# - node19: Knee_right_R 
# - node20: Knee_left_R 
# - node21: Ankle_right_R 
# - node22: Ankle_left_R 
# - node23: Foot_R 
# - node24: Hip_L 
# - node25: Knee_right_L 
# - node26: Knee_left_L 
# - node27: Ankle_right_L 
# - node28: Ankle_left_L 
# - node29: Foot_L


# After 
# -  Head(node1)
# -  Eye_R(node2)
# -  Eye_L(node3)
# -  Neck(node4)
# -  Sholuder_R(node5)
# -  Elbow_high_R(node6)
# -  Wrist_high_R(node7)
# -  Hand_R (node8)
# -  Sholuder_L (node9)
# - : Elbow_high_L(node10)
# - : Wrist_high_L(node11)
# - : Hand_L(node12)
# - : Pelvis(node13)
# - : Hip_R(node14)
# - : Knee_right_R(node15)
# - : Ankle_right_R(node16)
# - : Foot_R(node17)
# - : Hip_L(node18)
# - : Knee_right_L(node19)
# - : Ankle_right_L(node20)
# - : Foot_L(node21)


# cur_dir = os.path.dirname(os.path.abspath('openpose'))
seed_num = 18
cur_dir = 'E:/22.9.21/GM/openpose'
clients = ['jes', 'khb', 'ljh', 'njw']
skeleton = [
    [1,2],[2,4],[4,3],[3,1], # Head
    [4,5],[4,9],[4,13],[13,14],[13,18], # Body
    [5,6],[6,7],[7,8], # Right arm
    [9,10],[10,11],[11,12], # Left arm
    [14,15],[15,16],[16,17], # Right leg
    [18,19],[19,20],[20,21] # Left leg
]

def init_coco():
    coco = {

        "info": {
            "description": "",
            "url": "http://bilab.ai/",
            "version": "1.0",
            "year": 2022,
            "contributor": "",
            "data_created": "2022/07/09"
        },

        "licenses": [
            # For example
            # {
            # "url": "http://ycreativecommons.org/licenses/by-nc-sa/2.0/",
            # "id": 1,
            # "name": "Attribution-NonCommercial-ShareAlike License"
            # }
        ],

        "images": [
            # ***Format***
            # {
            #     "license": None,data_dir
            #     "height": 480,
            #     "width": 640,
            #     "date_captured": "2020-08-31T12:43:47.223Z"
            #     "flickr_url": None,
            #     "classId": 1496904
            #     "id": 506412614
        ],

        "categories": [
            {
                "supercategory": "bab",
                "id": 1,
                "name": "GM_Left",
                "keypoints": [
                    "Head", 
                    "Eye_R", 
                    "Eye_L", 
                    "Neck", 
                    "Sholuder_R", 
                    "Elbow_R", 
                    "Wrist_R", 
                    "Hand_R", 
                    "Sholuder_L", 
                    "Elbow_L", 
                    "Wrist_L",
                    "Hand_L", 
                    "Pelvis", 
                    "Hip_R",
                    "Knee_R",
                    "Ankle_R", 
                    "Foot_R", 
                    "Hip_L",
                    "Knee_L",   
                    "Ankle_L", 
                    "Foot_L"
                ],
                "skeleton": skeleton
            },
            {
                "supercategory": "baby",
                "id": 2,
                "name": "GM_Right",
                "keypoints": [
                    "Head", # 0
                    "Eye_R", # 1
                    "Eye_L", # 2
                    "Neck", # 3
                    "Sholuder_R", # 4
                    # "Elbow_high_R", "Elbow_low_R", 
                    "Elbow_R", # 5, 6
                    # "Wrist_high_R", "Wrist_low_R", 
                    "Wrist_R", # 7, 8
                    "Hand_R", # 9
                    "Sholuder_L", # 10
                    # "Elbow_high_L", "Elbow_low_L", 
                    "Elbow_L", # 11, 12 
                    # "Wrist_high_L", "Wrist_low_L", 
                    "Wrist_L", # 13, 14
                    "Hand_L", # 15
                    "Pelvis", # 16
                    "Hip_R",# 17
                    # "Knee_right_R", "Knee_left_R", 
                    "Knee_R", # 18, 19
                    # "Ankle_right_R", "Ankle_left_R", 
                    "Ankle_R", # 20, 21
                    "Foot_R", # 22
                    "Hip_L", # 23
                    # "Knee_right_L", "Knee_left_L", 
                    "Knee_L", # 24, 25
                    # "Ankle_right_L", "Ankle_left_L", 
                    "Ankle_L", # 26, 27
                    "Foot_L" # 28
                ],
                "skeleton": skeleton
            }
        ],

        "annotations": [
            # ***Format***
            # {
            #     "segmentation": None,
            #     "num_keypoints": 29,
            #     "area": None,
            #     "iscrowd": None,
            #     "keypoints": [],
            #     "image_id": 506412614,
            #     "bbox": [],
            #     "category_id": int,
            #     "id": None
            # }
        ]

    }

    return coco


def find_bbox(img, old_bbox):
# Finds bounding box from keypoints using canny edge detection

    h, w = img.shape[:2]
    gap = 14

    old_xmin, old_xmax, old_ymin, old_ymax = old_bbox
    xmin = max(0, old_xmin-gap)
    xmax = min(w, old_xmax-gap)
    ymin = max(0, old_ymin-gap)
    ymax = min(h, old_ymax+gap)

    # Erase out the background far from the object's bounding box
    img_cropped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use Otsu's threshold for canny edge detection
    th, ret = cv2.threshold(img_cropped, 0, 255, cv2.THRESH_OTSU)

    canny_output = cv2.Canny(img_cropped, th, th*2)
    canny_output[:int(ymin), :] = 0
    canny_output[int(ymax):, :] = 0
    canny_output[:, :int(xmin)] = 0
    canny_output[:, int(xmax):] = 0

    # Find the bounding box
    x,y,w,h = cv2.boundingRect(canny_output)
    x = min(x, old_xmin)
    y = min(y, old_ymin)
    w = max(w, old_xmax-x)
    h = max(h, old_ymax-y)

    # Visualize edges with bounding box
    # cv2.rectangle(canny_output, (int(x),int(y)), (int(x+w),int(y+h)), 255, 5)
    # cv2.imwrite(os.path.join(cur_dir, 'contour.jpg'), canny_output)

    return [x,y,w,h]

def datalist_split(data, rate=0.1):
    length = int(len(data) * rate)
    return data[:length], data[length:]

def preprocessing(data_dir, datalist_name='train', anno_id=0):
    idx = 0
    coco = init_coco()
    
    height = 256 
    width = 192 
    
    for ann in data_dir:

        img = cv2.imread(ann.replace('ann', 'img').rstrip('.json'))

        with open(ann) as f:
            data = json.load(f)
            if not data["objects"]:
                print("No object")
                continue

            obj = data["objects"][0]
            file_name = str(idx).zfill(6) + '.png'
            id = obj["id"] # image id 
            classId = obj["classId"]

            classTitle = obj["classTitle"]
            if classTitle == "GM_Left":
                # category_id = 1
                
                seq = iaa.Sequential([iaa.PadToSquare(position ='center', random_state=seed_num),
                                      iaa.CenterCropToAspectRatio(1.0),
                                      iaa.Affine(rotate=90),
                                      iaa.Resize({"height": height, "width": width})
                                      ])          
                
            elif classTitle == "GM_Right":
                # category_id = 2
                seq = iaa.Sequential([iaa.PadToSquare(position ='center', random_state=seed_num),
                                      iaa.CenterCropToAspectRatio(1.0),
                                      iaa.Affine(rotate=270),    
                                      iaa.Resize({"height": height, "width": width})
                                      ])
            else:
                continue

            keypointsX = []
            keypointsY = []
            keypoints = []
            num_nodes = len(obj["nodes"])
            # assert num_nodes == num_keypoints, f"{ann_file} has {num_nodes} nodes"

            nodes = obj["nodes"]
            first_key = list(nodes.keys())[0]
            node_table = []
            for df in GM_node_excel:
                if first_key in df.nodes.values:
                    node_table = df.nodes.values
                    break


            if len(node_table) == 0:
                print("Node key is not in the node table list")
                continue

            sum_list = np.array([5, 7, 11, 13, 18, 20, 24, 26])
            
            try:
                point_x = 0 
                point_y = 0
                for node_idx, key in enumerate(node_table):

                    x = nodes[key]["loc"][0]
                    y = nodes[key]["loc"][1]

                    if node_idx in sum_list:
                        point_x = x 
                        point_y = y 
                        continue

                    elif node_idx in (sum_list + 1):
                        point_x = (point_x + x) / 2 
                        point_y = (point_y + y) / 2 
                    else: 
                        point_x = x 
                        point_y = y 

                    keypointsX.append(point_x)
                    keypointsY.append(point_y)

                    # keypoints.extend([point_x, point_y])
                    keypoints.append((int(point_x), int(point_y)))
                    """
                    Annotations for keypoints is specified in (x, y, v)
                    v indicates visibility
                    v=0: not labeled (x=y=0)
                    v=1: labeled but not visible
                    v=2: labeled and visible
                    """

                    # keypoints.append(2)
                    
            except KeyError:
                continue
            
            xmin = min(keypointsX)
            xmax = max(keypointsX)
            ymin = min(keypointsY)
            ymax = max(keypointsY)

            # bbox = [xmin, xmax, ymin, ymax] 
            bbox = find_bbox(img, [xmin, xmax, ymin, ymax]) 

            # img_norm, keypoints_norm, bbox_norm = seq(image=img, keypoints=keypoints, bounding_boxes=tuple(map(float, bbox)))
            img_norm, keypoints_norm = seq(image=img, keypoints=keypoints)
            
            keypoints = [] 
            keypointsX = []
            keypointsY = []
            for key in keypoints_norm:
                key = list(map(float, key))
                
                # print('=' * 20)
                # print(key)
                # print('=' * 20)
                
                keypoints.extend([key[0], key[1]])
                keypoints.append(2)
                
                keypointsX.append(key[0])
                keypointsY.append(key[1])
    
            # bbox = list(map(float, bbox_norm))
            
            xmin = min(keypointsX)
            xmax = max(keypointsX)
            ymin = min(keypointsY)
            ymax = max(keypointsY)

            bbox = find_bbox(img_norm, [xmin, xmax, ymin, ymax]) 


            coco["images"].append(
                {   
                    "license": None,
                    "file_name": file_name,
                    "coco_url": None,
                    # "height": data["size"]["height"],
                    # "width": data["size"]["width"],
                    "height" : height, 
                    "width" : width, 
                    "date_captured": obj["createdAt"],
                    "flickr_url": None,
                    # "id": idx
                    "id": id
                }
            )

            coco["annotations"].append(
                {
                    "segmentation": [[None]],
                    # "segmentation": {
                    #     "counts" : None, 
                    #     "size" : None
                    # },
                    "num_keypoints": [],
                    # "area": None,
                    "area": data["size"]["height"] * data["size"]["width"],
                    # "iscrowd": None,
                    # "iscrowd": 'false',
                    "iscrowd": 0,
                    "keypoints": keypoints,
                    "image_id": id,
                    "bbox": bbox,
                    "category_id": 1,
                    "id": anno_id
                }
            )                    
                
            anno_id += 1

            # Save images and annotated images
            bbox = list(map(int, bbox))
            keypointsX = list(map(int, keypointsX))
            keypointsY = list(map(int, keypointsY))

            if not os.path.exists(os.path.join(cur_dir, 'custom_dataset/images', datalist_name)):
                os.mkdir(os.path.join(cur_dir, 'custom_dataset/images', datalist_name))

            if not os.path.exists(os.path.join(cur_dir, 'custom_dataset/annotated images', datalist_name)):
                os.mkdir(os.path.join(cur_dir, 'custom_dataset/annotated images', datalist_name))


            # cv2.imwrite(os.path.join(cur_dir, 'custom_dataset/images', datalist_name, file_name), img)
            cv2.imwrite(os.path.join(cur_dir, 'custom_dataset/images', datalist_name, file_name), img_norm)
            cv2.rectangle(img_norm, (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (0,255,0), 3)
            # cv2.rectangle(img_norm, (bbox_norm[0],bbox_norm[1]), (bbox_norm[0]+bbox_norm[2],bbox_norm[1]+bbox_norm[3]), (0,255,0), 3)
            
            for start, end in skeleton:
                cv2.line(img_norm, (keypointsX[start-1], keypointsY[start-1]), (keypointsX[end-1], keypointsY[end-1]), (255,0,0), 3)
            cv2.imwrite(os.path.join(cur_dir, 'custom_dataset/annotated images', datalist_name, file_name), img_norm)

        idx += 1

    # return coco
    with open(os.path.join(cur_dir, 'custom_dataset/annotations', f'{datalist_name}_baby_keypoints.json'), 'w') as f:
        json.dump(coco, f)

    
    return anno_id


# Read GM_node.xlsx
GM_node_excel = pd.read_excel(os.path.join(cur_dir, 'Dataset/GM_node.xlsx'), sheet_name=None).values()

# last_idx = 23528
num_keypoints = 21
anno_id = 10000 # annotation unique id (https://github.com/cocodataset/cocoapi/issues/95), no clue

if os.path.exists('/home/test/GM/openpose/custom_dataset/images/train'):
    shutil.rmtree(r'/home/test/GM/openpose/custom_dataset/images/train')

if os.path.exists('/home/test/GM/openpose/custom_dataset/images/valid'):
    shutil.rmtree(r'/home/test/GM/openpose/custom_dataset/images/valid')

if os.path.exists('/home/test/GM/openpose/custom_dataset/images/test'):
    shutil.rmtree(r'/home/test/GM/openpose/custom_dataset/images/test')

if os.path.exists('/home/test/GM/openpose/custom_dataset/annotated images/train'):
    shutil.rmtree(r'/home/test/GM/openpose/custom_dataset/annotated images/train')

if os.path.exists('/home/test/GM/openpose/custom_dataset/annotated images/valid'):
    shutil.rmtree(r'/home/test/GM/openpose/custom_dataset/annotated images/valid')

if os.path.exists('/home/test/GM/openpose/custom_dataset/annotated images/test'):
    shutil.rmtree(r'/home/test/GM/openpose/custom_dataset/annotated images/test')

if os.path.exists('/home/test/GM/openpose/custom_dataset/annotations/train_baby_keypoints.json'):    
    os.remove('/home/test/GM/openpose/custom_dataset/annotations/train_baby_keypoints.json')

if os.path.exists('/home/test/GM/openpose/custom_dataset/annotations/valid_baby_keypoints.json'):    
    os.remove('/home/test/GM/openpose/custom_dataset/annotations/valid_baby_keypoints.json')

if os.path.exists('/home/test/GM/openpose/custom_dataset/annotations/test_baby_keypoints.json'):    
    os.remove('/home/test/GM/openpose/custom_dataset/annotations/test_baby_keypoints.json')


train_list = [] 
valid_list = [] 
test_list = [] 

for c in clients:
    client_dir = os.path.join(cur_dir, 'Dataset', c)

    client_list = glob(os.path.join(client_dir, '*/*/ann/*.json'))

    train, test = datalist_split(client_list, 0.6)    
    train_list.extend(train)

    test, valid = datalist_split(test, 0.4)
    test_list.extend(test)
    valid_list.extend(valid)

    # print(f'train -----------> {len(train_list)}')
    # print(f'test -----------> {len(test_list)}')
    # print(f'valid -----------> {len(valid_list)}')

# preprocessing
# anno_id = preprocessing(train_list, 'train', anno_id)
# anno_id = preprocessing(valid_list, 'valid', anno_id)
anno_id = preprocessing(test_list, 'test', anno_id)

