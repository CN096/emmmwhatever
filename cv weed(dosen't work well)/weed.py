import os
import json
import torch
import cv2
import random
import numpy as np
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.transforms import RandomFlip, Resize,RandomRotation
from detectron2.data import detection_utils as utils

# Step 1: Define custom Trainer with data augmentation
class AugmentedTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        # Define augmentations
        augmentations = [
            RandomFlip(prob=0.5),
            Resize(shortest_edge=(500, 600), max_size=800),
            RandomRotation(angle=[-5, 5])
            #RandomBrightnessContrast(brightness_lim=(0.9, 1.1), contrast_lim=(0.9, 1.1))
        ]
        
        # Use DatasetMapper with augmentations
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
        
        return build_detection_train_loader(cfg, mapper=mapper)
    
# Helper function to convert custom JSON to COCO format
def convert_labelme_to_coco(labelme_annotations_dir, output_coco_file):
    images = []
    annotations = []
    categories = [{"id": 1, "name": "weed"}, {"id": 2, "name": "crop"}]  # Define categories
    annotation_id = 1

    for idx, file_name in enumerate(os.listdir(labelme_annotations_dir)):
        if not file_name.endswith(".json"):
            continue

        with open(os.path.join(labelme_annotations_dir, file_name), 'r') as f:
            data = json.load(f)

        # Add image info
        image_id = idx + 1
        image_info = {
            "id": image_id,
            "file_name": data["imagePath"],
            "height": data["imageHeight"],
            "width": data["imageWidth"]
        }
        images.append(image_info)

        # Add annotations
        for shape in data["shapes"]:
            label = shape["label"]
            if shape["shape_type"] != "circle":
                continue  # Ignore non-circle shapes

            x1, y1 = shape["points"][0]
            x2, y2 = shape["points"][1]
            radius = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            # Convert circle to bounding box
            bbox = [x1 - radius, y1 - radius, 2 * radius, 2 * radius]

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 2 if label == "mq" else 1,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "segmentation": []  # Not used here
            }
            annotations.append(annotation)
            annotation_id += 1

    # Write to COCO JSON file
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(output_coco_file, 'w') as f:
        json.dump(coco_format, f, indent=4)

# Convert LabelMe annotations to COCO format
labelme_annotations_dir = "./data/labelme_annotations"
output_coco_file = "./data/annotations/instances_train.json"
os.makedirs(os.path.dirname(output_coco_file), exist_ok=True)
convert_labelme_to_coco(labelme_annotations_dir, output_coco_file)

# Step 1: Register dataset in Detectron2
data_dir = "./data"
image_dir = os.path.join(data_dir, "images")
register_coco_instances(
    "weed_dataset_train", {}, output_coco_file, image_dir
)

# Step 2: Configure the Model
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("weed_dataset_train",)
cfg.DATASETS.TEST = ()  # No test dataset for now
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.WARMUP_FACTOR = 0.001
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 12000  # Adjust this depending on your dataset size
cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5
cfg.MODEL.ROI_BOX_HEAD.NUM_FC=3
cfg.MODEL.ROI_BOX_HEAD.FC_DIM=1024
cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION=14
cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT=1.2
cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "giou"
cfg.MODEL.ANCHOR_GENERATOR.SIZES=[[8,16,32,64,128]]
cfg.SOLVER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.OPTIMIZER = "AdamW"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Crop and Weed
cfg.MODEL.ROI_HEADS.CLASSIFICATION_LOSS_WEIGHT = [1.84, 2.19]  # [weed, crop] 对应类别 1 和 2


# Output directory
cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Step 3: Train the Model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Step 4: Inference and Save Predictions
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # Set threshold for this model
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST=0.4
predictor = DefaultPredictor(cfg)

# Predict on test images and save results
def predict_test_images(test_images_dir, output_predictions_file):
    results = []
    for file_name in os.listdir(test_images_dir):
        if not (file_name.endswith(".jpg") or file_name.endswith(".png")):
            continue

        file_path = os.path.join(test_images_dir, file_name)
        img = cv2.imread(file_path)
        outputs = predictor(img)

        instances = outputs["instances"].to("cpu")
        bboxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        for bbox, score, cls in zip(bboxes, scores, classes):
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            results.append({
                "file_name": file_name,
                "class": "weed" if cls == 0 else "crop",
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "center": [float(x_center), float(y_center)],
                "size": [float(width), float(height)],
                "score": float(score)
            })

    with open(output_predictions_file, 'w') as f:
        json.dump(results, f, indent=4)

# Specify test images directory and output file
test_images_dir = "./data/test_images"
output_predictions_file = "./data/predictions.json"
os.makedirs(os.path.dirname(output_predictions_file), exist_ok=True)
predict_test_images(test_images_dir, output_predictions_file)

print(f"Predictions saved to {output_predictions_file}")
