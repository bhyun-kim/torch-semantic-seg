ROOT_DIR = '/home/user/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01-1. cityscapes_mini'
MODEL = dict(
    encoder = dict(type='CGNet'),
    decoder = None,
    head = dict(type='Interpolate', scale_factor=8, mode='bilinear')
)
LOSS = dict(type='CrossEntropyLoss')
BATCH_SIZE = 2
DATA_LOADERS = dict(
    train=dict(
        dataset=dict(
            type='CityscapesDataset', 
            root_dir=ROOT_DIR,
            split='train'), 
        pipelines=[
            dict(type='RandomRescale', output_range=(600, 800)),
            dict(type='RandomCrop', output_size=(400, 800)),
            dict(type='Normalization', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            dict(type='ToTensor')
        ],
        loader=dict(
            shuffle=True,
            batch_size=BATCH_SIZE
        )
    ),
    val = dict(
        dataset=dict(
            type='CityscapesDataset', 
            root_dir=ROOT_DIR,
            split='val'), 
        pipelines=[
            dict(type='RandomRescale', output_range=(600, 800)),
            dict(type='RandomCrop', output_size=(400, 800)),
            dict(type='Normalization', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            dict(type='ToTensor')
        ],
        loader=dict(
            shuffle=True,
            batch_size=1
        )
    ),
    test=dict(
        dataset=dict(
            type='CityscapesDataset', 
            root_dir=ROOT_DIR,
            split='test'), 
        pipelines=[
            dict(type='Rescale', output_range=(600, 800)),
            # dict(type='RandomCrop', output_size=(400, 800)),
            dict(type='Normalization', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            dict(type='ToTensor')
        ],
        loader=dict(
            shuffle=True,
            batch_size=3
        )
    )
)
OPTIMIZER = dict(type='Adam', lr=0.001)
EPOCHS = 10 
LOGGER = dict(loop='iteration', interval=50)
EVALUATION = dict(loop='epoch', interval=1, metric='mIoU')
RUNNER = dict(type='SupervisedLearner')
DEVICES = ['cuda:0']
WORK_DIR = '/home/user/#checkpoints/temp'