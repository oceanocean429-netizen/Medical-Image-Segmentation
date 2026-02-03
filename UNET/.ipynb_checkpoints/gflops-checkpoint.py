import torch
import torch.nn as nn
from thop import profile, clever_format

# ==========================================
# 1. YOUR MODEL DEFINITIONS
# ==========================================

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

# ==========================================
# 2. CALCULATION SCRIPT
# ==========================================

if __name__ == "__main__":
    # Settings (MATCH THESE TO YOUR CONFIG)
    INPUT_CHANNELS = 3    # e.g., 3 for RGB
    NUM_CLASSES = 1       # e.g., 1 for Binary Segmentation
    INPUT_SIZE = 512      # Change this to 256 or whatever you use
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize Model
    model = U_Net(img_ch=INPUT_CHANNELS, output_ch=NUM_CLASSES).to(device)
    model.eval()

    # Create Dummy Input (Batch Size 1)
    input_tensor = torch.randn(1, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE).to(device)

    print(f"Profiling U-Net with input shape: {input_tensor.shape}")

    # Use THOP to calculate MACs and Params
    macs, params = profile(model, inputs=(input_tensor, ), verbose=False)
    
    # Format readable strings (e.g., "1.5G", "30M")
    macs_formatted, params_formatted = clever_format([macs, params], "%.3f")

    print("-" * 30)
    print(f"Parameters: {params_formatted}")
    print(f"MACs:       {macs_formatted}")
    print(f"GFLOPs:     {macs / 1e9 * 2:.3f} G (Approx. 2 * MACs)")
    print("-" * 30)