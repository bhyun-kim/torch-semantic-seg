ROOT_DIRS = [
    '/home/user/UOS-SSaS Dropbox/05. Data/02. Training&Test/013. General Concrete Damage/v0.1.1',
    '/home/user/UOS-SSaS Dropbox/05. Data/02. Training&Test/024. Korean Electricity/CityScapes_v0.1.1'
]
MODEL = dict(
    encoder = dict(type='CGNet', classes=6),
    decoder = None,
    head = dict(type='Interpolate', scale_factor=8, mode='bilinear')
)
LOSS = dict(type='CrossEntropyLoss', weight=[
                0.5959933, 10.396974, 3.5354059, 6.3927646, 4.3097677, 0, 
            ],
            ignore_idx=6
)
BATCH_SIZE = 2
CLASSES = ('background', 'crack', 'efflorescence', 'rebar_exposure', 'spalling', 'leakage')
PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 0, 255], [0, 255, 0]]

TRAIN_PIPELINES = [
            dict(type='RandomRescale', output_range=(2800, 3800)),
            dict(type='RandomCrop', output_size=(2560, 2560)),
            dict(type='Normalization', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            dict(type='ToTensor')
        ]

VAL_PIPELINES = [
            dict(type='Rescale', output_size=(2560, 2560)),
            dict(type='Normalization', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            dict(type='ToTensor')
        ]

TEST_PIPELINES = [
            dict(type='Rescale', output_size=(2560, 2560)),
            dict(type='Normalization', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            dict(type='ToTensor')
        ]

SEG_SUFFIX = '_gtFine_labelIds.png'

DATA_LOADERS = dict(
    train=dict(
        dataset=dict(
            type='ConcatDataset',
            datasets=[
                dict(
                    type='CityscapesDataset', 
                    root_dir=ROOT_DIRS[0],
                    split='train',
                    classes=CLASSES, palette=PALETTE,
                    seg_suffix=SEG_SUFFIX,
                ),
                dict(
                    type='CityscapesDataset', 
                    root_dir=ROOT_DIRS[1],
                    split='train',
                    classes=CLASSES, palette=PALETTE,
                    seg_suffix=SEG_SUFFIX                
                ),
            ]      
        ),    
        pipelines=TRAIN_PIPELINES,
        loader=dict(
            shuffle=True,
            batch_size=BATCH_SIZE,
        )
    ),
    val = dict(
        dataset=dict(
            type='ConcatDataset',
            datasets=[
                dict(
                    type='CityscapesDataset', 
                    root_dir=ROOT_DIRS[0],
                    split='val',
                    classes=CLASSES, palette=PALETTE,
                    seg_suffix=SEG_SUFFIX                
                ),
                dict(
                    type='CityscapesDataset', 
                    root_dir=ROOT_DIRS[1],
                    split='val',
                    classes=CLASSES, palette=PALETTE,
                    seg_suffix=SEG_SUFFIX                
                ),
            ]      
        ),    
        pipelines=VAL_PIPELINES,
        loader=dict(
            shuffle=None,
            batch_size=1
        )
    ),
    test=dict(
        dataset= dict(
                    type='CityscapesDataset', 
                    root_dir=ROOT_DIRS[0],
                    split='test',
                    classes=CLASSES, palette=PALETTE,
                    seg_suffix=SEG_SUFFIX                
                ), 
        pipelines=TEST_PIPELINES,
        loader=dict(
            shuffle=None,
            batch_size=1
        )
    )
)

OPTIMIZER = dict(type='Adam', lr=0.001)
EPOCHS = 10 
LOGGER = dict(loop='iteration', interval=50)
EVALUATION = dict(loop='epoch', interval=1, metric='mIoU')
RUNNER = dict(type='SupervisedLearner')
DEVICES = ['cuda:0']
WORK_DIR = '/home/user/UOS-SSaS Dropbox/05. Data/03. Checkpoints/024. Korean Electricity/10.31.2022'