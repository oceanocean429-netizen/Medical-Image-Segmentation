import torch
import segmentation_models_pytorch as smp
from thop import profile

# 1. Define the model exactly as you did in your main function
# Update these two variables to match your config
INPUT_CHANNELS = 3   # e.g., 3 for RGB, 1 for Grayscale
NUM_CLASSES = 1      # e.g., 1 for Binary Segmentation
INPUT_SIZE = 512     # The Height/Width of your images (e.g., 224, 256, 512)

model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,      # Weights don't affect parameter count, only values
    in_channels=INPUT_CHANNELS,
    classes=NUM_CLASSES,
)

# 2. Create a dummy input tensor (Batch Size 1)
# Shape: (Batch_Size, Channels, Height, Width)
dummy_input = torch.randn(1, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE)

# 3. Calculate MACs (Multiply-Accumulate Operations) and Params
macs, params = profile(model, inputs=(dummy_input, ), verbose=False)

# 4. Convert to GFLOPs and Millions
# Standard convention: 1 MAC ≈ 2 FLOPs
gflops = (macs * 2) / 1e9 
params_in_millions = params / 1e6

print(f"Model: Unet++ (ResNet34)")
print(f"Input Size: {INPUT_SIZE}x{INPUT_SIZE}")
print("-" * 30)
print(f"Parameters: {params_in_millions:.2f} M")
print(f"GFLOPs:     {gflops:.2f}")
print("-" * 30)