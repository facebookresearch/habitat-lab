# Language-driven Semantic Segmentation (LSeg)

### Download Pre-trained Checkpoint
```
mkdir checkpoints; cd checkpoints

# This will download a "demo_e200.ckpt" checkpoint
gdown 1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb
```

### Inference Usage
```
from home_robot.agent.perception.detection.lseg import load_lseg_for_inference

# Load pre-trained model
checkpoint_path = "checkpoints/demo_e200.ckpt"
device = torch.device("cuda:0")
model = load_lseg_for_inference(checkpoint_path, device)

# Encode pixels to CLIP features
# rgb of shape (batch_size, H, W, 3) in RGB order
# pixel_features of shape (batch_size, 512, H, W)
pixel_features = model.encode(rgb)

# Decode pixel CLIP features to text labels - we can introduce new labels
# at inference time
# one_hot_predictions of shape (batch_size, H, W, len(labels))
# visualizations of shape (batch_size, H, W, 3)
labels = ["tree", "chair", "clock", "couch", "other",
          "cushion", "lamp", "cabinet"]
one_hot_predictions, visualizations = model.decode(pixel_features, labels)
```
