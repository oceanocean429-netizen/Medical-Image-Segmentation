import torch
import torch.nn as nn
from thop import profile, clever_format
from einops import rearrange, repeat
import numpy as np
from UNet import TransUNet
# --- [PASTE YOUR MODEL CLASSES HERE] --- 
# (I have verified your provided classes; they are ready to run. 
# Ensure all classes: up_conv, conv_block, MultiHeadAttention, MLP, 
# TransformerEncoderBlock, TransformerEncoder, ViT, EncoderBottleneck, 
# DecoderBottleneck, Encoder, Decoder, TransUNet are defined before running this.)

# ... [Assuming the classes from your prompt are defined above] ...

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # 1. Configuration
    # We use standard medical image segmentation settings
    img_size = 512        # Standard input resolution
    in_channels = 3         # RGB
    out_channels = 128      # Base channel width (controls model width)
    head_num = 4            # Transformer heads
    mlp_dim = 512           # Transformer MLP hidden dim
    block_num = 8           # Number of Transformer layers
    patch_dim = 16          # Patch size for ViT input calculations
    class_num = 1           # Binary segmentation

    # 2. Instantiate Model
    # Note: Ensure the device is CPU for accurate profiling without CUDA overhead, 
    # though thop works on GPU as well.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TransUNet(
        img_dim=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        head_num=head_num,
        mlp_dim=mlp_dim,
        block_num=block_num,
        patch_dim=patch_dim,
        class_num=class_num
    ).to(device)

    # 3. Create Dummy Input
    # Shape: (Batch_Size, Channels, Height, Width)
    input_tensor = torch.randn(1, in_channels, img_size, img_size).to(device)

    # 4. Profile
    print(f"Profiling model with Input Shape: {input_tensor.shape}")
    
    # Calculate MACs (Multiply-Accumulate Operations) and Parameters
    macs, params = profile(model, inputs=(input_tensor, ), verbose=False)
    
    # Format the output
    macs_fmt, params_fmt = clever_format([macs, params], "%.3f")

    print("-" * 30)
    print(f"Parameters: {params_fmt}")
    print(f"MACs:       {macs_fmt}")
    # Note: 1 MAC is roughly equal to 2 FLOPs
    print(f"GFLOPs:     {float(macs/1e9) * 2:.3f} G (approx)")
    print("-" * 30)