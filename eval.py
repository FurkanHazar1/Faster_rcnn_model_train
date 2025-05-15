import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    metric = MeanAveragePrecision()
 

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [image.to(device) for image in images]
            predictions = model(images)
            
            # Process each image in the batch
            for i, (target, pred) in enumerate(zip(targets, predictions)):
                # Get ground truth boxes and labels
                true_boxes = []
                true_labels = []
                
                for annotation in target:
                    # COCO bbox format: [x_min, y_min, width, height]
                    bbox = annotation['bbox']
                    # Convert to [x_min, y_min, x_max, y_max]
                    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                    true_boxes.append(bbox)
                    true_labels.append(annotation['category_id'])
                
                if not true_boxes:  # Skip if no objects in image
                    continue
                
                true_boxes = torch.tensor(true_boxes, dtype=torch.float32).to(device)
                true_labels = torch.tensor(true_labels, dtype=torch.int64).to(device)
                
                # Get predictions
                pred_boxes = pred['boxes']
                pred_labels = pred['labels']
                pred_scores = pred['scores']
                
                # Filter out low-confidence predictions
                keep = pred_scores > 0.3  # You can adjust this threshold
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep]
                
                if len(pred_boxes) == 0:
                    continue
                
                # Update metrics
                metric.update(
                    [dict(boxes=pred_boxes, scores=pred_scores, labels=pred_labels)],
                    [dict(boxes=true_boxes, labels=true_labels)]
                )

    # Compute metrics
    map_score = metric.compute()
    


    return {
        "mAP_50": map_score['map_50'].item(),
        "mAP_75": map_score['map_75'].item()
    }