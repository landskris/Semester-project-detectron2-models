from os import getcwd, makedirs
from os.path import join

from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg
from detectron2.model_zoo import get_checkpoint_url
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np
import requests


class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


def register_datasets():
    data_dir = join(getcwd(), 'Grini_nc_merged')
    train_imgs = join(data_dir, 'train_imgs')
    val_imgs = join(data_dir, 'val_imgs')
    test_imgs = join(data_dir, 'test_imgs')

    train_annos = join(data_dir, 'grini_nc_merged_no_masks_train.json')
    val_annos = join(data_dir, 'grini_nc_merged_no_masks_val.json')
    test_annos = join(data_dir, 'grini_nc_merged_no_masks_test.json')

    # Register dataset configs
    register_coco_instances('grini_nc_merged_bbox_only_train', {}, train_annos, train_imgs)
    register_coco_instances('grini_nc_merged_bbox_only_val', {}, val_annos, val_imgs)
    register_coco_instances('grini_nc_merged_bbox_only_test', {}, test_annos, test_imgs)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml'))

register_datasets()
cfg.DATASETS.TRAIN = ('grini_nc_merged_bbox_only_train',)
cfg.DATASETS.TEST = ('grini_nc_merged_bbox_only_val',)

cfg.MODEL.WEIGHTS = get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml')
cfg.MODEL.DEVICE = "cpu"  # cpu or cuda
cfg.MASK_ON = False

# todo find out how rescale images and annotations first...
# Parameters fixed
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500  # adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

cfg.TEST.EVAL_PERIOD = 500

makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()