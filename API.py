import torch,torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import pandas as pd
import wandb
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor,DefaultTrainer
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from tqdm import tqdm
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
PROJECT_NAME = 'Car-Object-Detection-V10-Learning-Detectron2-V2'
model = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
labels = ['Car']
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.DATASETS.TRAIN = ('data')
cfg.DATASETS.TEST = ()
cfg.SOLVER.STEPS = []
cfg.SOLVER.MAX_ITER = 625
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(labels)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
predictor = DefaultPredictor(cfg)
img = cv2.imread('./img.png')
v = Visualizer(img[:,:,::-1])
v = v.draw_instance_predictions(predictor(img)['instances'].to('cpu'))
v = v.get_image()[:,:,::-1]

