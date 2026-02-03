import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as resnet_model
import timm

# --- Import Analysis Libraries ---
try:
    from thop import profile
except ImportError:
    pass
try:
    from ptflops import get_model_complexity_info
except ImportError:
    pass

# ==========================================
# 1. CORRECTED MODEL DEFINITION
# ==========================================

class FAMBlock(nn.Module):
    def __init__(self, channels):
        super(FAMBlock, self).__init__()
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.relu3(self.conv3(x))
        x1 = self.relu1(self.conv1(x))
        return x3 + x1

class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)
        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.up(x)
        x = self.relu3(self.norm3(self.conv3(x)))
        return x

class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FAT_Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(FAT_Net, self).__init__()
        transformer = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=True, img_size=256)
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv, self.firstbn, self.firstrelu = resnet.conv1, resnet.bn1, resnet.relu
        self.encoder1, self.encoder2, self.encoder3, self.encoder4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        self.patch_embed = transformer.patch_embed
        self.transformers = nn.ModuleList([transformer.blocks[i] for i in range(12)])
        
        self.conv_seq_img = nn.Conv2d(192, 512, 3, 1, 1)
        self.se = SEBlock(1024)
        # Fixed: Explicit stride=1 to avoid errors
        self.conv2d = nn.Conv2d(1024, 512, 1, stride=1, padding=0)

        # === FIX: UNIQUE INSTANCES ===
        # Use 'for _ in range(x)' to create NEW instances for each block
        self.FAM1 = nn.ModuleList([FAMBlock(64) for _ in range(6)])
        self.FAM2 = nn.ModuleList([FAMBlock(128) for _ in range(4)])
        self.FAM3 = nn.ModuleList([FAMBlock(256) for _ in range(2)])
        # =============================

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        e0 = self.firstrelu(self.firstbn(self.firstconv(x)))
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        emb = self.patch_embed(x)
        for block in self.transformers:
            emb = block(emb)
        
        grid_h, grid_w = h // 16, w // 16
        feature_tf = emb.permute(0, 2, 1).view(b, 192, grid_h, grid_w)
        feature_tf = self.conv_seq_img(feature_tf)

        if feature_tf.shape[2:] != feature_cnn.shape[2:]:
            feature_tf = F.interpolate(feature_tf, size=feature_cnn.shape[2:], mode='bilinear', align_corners=True)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for layer in self.FAM3: e3 = layer(e3)
        for layer in self.FAM2: e2 = layer(e2)
        for layer in self.FAM1: e1 = layer(e1)
            
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out = self.final_conv3(self.final_relu2(self.final_conv2(self.final_relu1(self.final_conv1(d2)))))
        return out

# ==========================================
# 2. RUNNER
# ==========================================

def run_analysis():
    input_size = (1, 3, 512, 512)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Analyzing FAT_Net | Input: {input_size} | Device: {device}\n")

    # --- 1. THOP ---
    try:
        model = FAT_Net(n_channels=3, n_classes=1).to(device)
        dummy_input = torch.randn(*input_size).to(device)
        print("--- THOP Analysis ---")
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        print(f"Params: {params / 1e6:.2f} M")
        print(f"GMACs:  {macs / 1e9:.2f} G")
        print(f"GFLOPs: {macs * 2 / 1e9:.2f} G")
    except Exception as e:
        print(f"THOP Error: {e}")

    print("-" * 20)

    # --- 2. PTFLOPS ---
    try:
        # Re-instantiate model to clear hooks
        model = FAT_Net(n_channels=3, n_classes=1) 
        print("--- PTFLOPS Analysis ---")
        macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=False, print_per_layer_stat=False, verbose=False)
        print(f"Params: {params / 1e6:.2f} M")
        print(f"GMACs:  {macs / 1e9:.2f} G")
        print(f"GFLOPs: {macs * 2 / 1e9:.2f} G")
    except Exception as e:
        print(f"PTFLOPS Error: {e}")

if __name__ == "__main__":
    run_analysis()