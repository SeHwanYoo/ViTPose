dataset_info = dict(
    dataset_name='coco',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
        dict(
            name='head', 
            id=0, 
            color=[255, 128, 0], 
            type='upper', 
            swap=''),
        1:
        dict(
            name='right_eye',
            id=1,
            color=[255, 128, 0],
            type='upper',
            swap='left_eye'),
        2:
        dict(
            name='left_eye',
            id=2,
            color=[255, 128, 0],
            type='upper',
            swap='right_eye'),
        3:
        dict(
            name='neck',
            id=3,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        4:
        dict(
            name='right_shoulder',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_shoulder'),
        5:
        dict(
            name='right_elbow',
            id=5,
            color=[51, 153, 255],
            type='upper',
            swap='left_elbow'),
        6:
        dict(
            name='right_wrist',
            id=6,
            color=[51, 153, 255],
            type='upper',
            swap='left_wrist'),
        7:
        dict(
            name='right_hand',
            id=7,
            color=[51, 153, 255],
            type='upper',
            swap='right_hand'),
        8:
        dict(
            name='left_shoulder',
            id=8,
            color=[0, 255, 0],
            type='upper',
            swap='left_shoulder'),
        9:
        dict(
            name='left_elbow',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        10:
        dict(
            name='left_wrist',
            id=10,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        11:
        dict(
            name='left_hand',
            id=11,
            color=[0, 255, 0],
            type='upper',
            swap='right_hand'),
        12:
        dict(
            name='pelvis',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap=''),
        13:
        dict(
            name='right_hip',
            id=13,
            color=[51, 153, 255],
            type='lower',
            swap='left_hip'),
        14:
        dict(
            name='right_knee',
            id=14,
            color=[51, 153, 255],
            type='lower',
            swap='left_knee'),
        15:
        dict(
            name='right_ankle',
            id=15,
            color=[51, 153, 255],
            type='lower',
            swap='left_ankle'),
        16:
        dict(
            name='right_foot',
            id=16,
            color=[51, 153, 255],
            type='lower',
            swap='left_ankle'),
        17:
        dict(
            name='left_hip',
            id=17,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        18:
        dict(
            name='left_knee',
            id=18,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        19:
        dict(
            name='left_ankle',
            id=19,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        20:
        dict(
            name='left_foot',
            id=20,
            color=[0, 255, 0],
            type='lower',
            swap='right_foot')
        
    },
    skeleton_info={
        # head 
        0:
        dict(
            link=('head', 'right_eye'), 
            id=0, 
            color=[255, 128, 0]),
        1:
        dict(
            link=('right_eye', 'neck'), 
            id=1, 
            color=[255, 128, 0]),
        2:
        dict(
            link=('neck', 'left_eye'), 
            id=2, 
            color=[255, 128, 0]),
        3:
        dict(
            link=('left_eye', 'head'), 
            id=3, 
            color=[255, 128, 0]),
        # body
        4:
        dict(
            link=('neck', 'right_shoulder'), 
            id=4, 
            color=[255, 128, 0]),
        5:
        dict(
            link=('neck', 'left_shoulder'), 
            id=5, 
            color=[255, 128, 0]),
        6:
        dict(
            link=('neck', 'pelvis'), 
            id=6, 
            color=[255, 128, 0]),
        7:
        dict(
            link=('pelvis', 'right_hip'), 
            id=7, 
            color=[255, 128, 0]),
        8:
        dict(
            link=('pelvis', 'left_hip'), 
            id=8, 
            color=[255, 128, 0]),
        # right arm 
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), 
            id=9, 
            color=[51, 153, 255]),
        10:
        dict(
            link=('right_elbow', 'right_wrist'), 
            id=10, 
            color=[51, 153, 255]),
        11:
        dict(
            link=('right_wrist', 'right_hand'), 
            id=11, 
            color=[51, 153, 255]),
        # left arm
        12:
        dict(
            link=('left_shoulder', 'left_elbow'), 
            id=12, 
            color=[0, 255, 0]),
        13:
        dict(
            link=('left_elbow', 'left_wrist'), 
            id=13, 
            color=[0, 255, 0]),
        15:
        dict(
            link=('left_wrist', 'left_hand'), 
            id=15, 
            color=[0, 255, 0]),
        # right leg 
        16:
        dict(
            link=('right_hip', 'right_knee'), 
            id=16, 
            color=[51, 153, 255]),
        17:
        dict(
            link=('right_knee', 'right_ankle'), 
            id=17, 
            color=[51, 153, 255]),
        18:
        dict(
            link=('right_ankle', 'right_foot'), 
            id=18, 
            color=[51, 153, 255]),
        # left leg
        19:
        dict(
            link=('left_hip', 'left_knee'), 
            id=19, 
            color=[0, 255, 0]),
        20:
        dict(
            link=('left_knee', 'left_ankle'), 
            id=20, 
            color=[0, 255, 0]),
        21:
        dict(
            link=('left_ankle', 'left_foot'), 
            id=21, 
            color=[0, 255, 0]),
        
    },
    joint_weights=[
        1., # 0
        1., # 1
        1., # 2
        1., # 3
        1., # 4
        1., # 5
        1., # 6
        1.2, # 7
        1.2, # 8
        1.5, # 9
        1.5, # 10
        1., # 11
        1., # 12
        1.2, # 13
        1.2, # 14
        1.5, # 15
        1.5, # 16
        1.2, # 17
        1.2, # 18
        1.5,# 19
        1.5, # 20
        # 1.5 # 21
    ],
    sigmas=[
        0.026, # 0
        0.025, # 1
        0.025, # 2
        0.035, # 3
        0.035, # 4
        0.079, # 5
        0.079, # 6
        0.072, # 7
        0.072, # 8
        0.062, # 9
        0.062, # 10
        0.107, # 11
        0.107, # 12
        0.087, # 13
        0.087, # 14
        0.089, # 15
        0.089, # 16
        0.107, # 17
        0.087, # 18
        0.087, # 19
        0.089, # 20
        # 0.089, # 21
    ])
