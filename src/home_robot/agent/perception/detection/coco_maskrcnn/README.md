# Mask R-CNN Trained on COCO

### Relevant Indoor Categories
```
coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    "dining table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14,
    "no-category": 15,
}
```

### Inference Usage
```
from home_robot.agent.perception.detection.coco_maskrcnn.coco_maskrcnn import COCOMaskRCNN

segmentation = COCOMaskRCNN(
    sem_pred_prob_thr=0.9,  # prediction threshold
    sem_gpu_id=0,           # prediction GPU id (-1 for CPU)
    visualize=True          # if True, visualize predictions
)

# rgb of shape (batch_size, H, W, 3) in BGR order
# depth optional if want depth filtering, of shape (batch_size, H, W)
# one hot segmentation predictions of shape (batch_size, H, W, 16)
# visualization_images of shape (batch_size, H, W, 3) 
one_hot_predictions, visualization_images = segmentation.get_prediction(rgb, depth)
```