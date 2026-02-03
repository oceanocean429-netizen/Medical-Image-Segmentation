import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from thop import profile, clever_format
import math
from functools import partial

# ==========================================
# 1. HELPER: FLOPs Counter for Scan
# ==========================================
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    Calculates FLOPs for the selective scan operation.
    """
    # Simply returning the calculation logic provided in your prompt
    # Simplified for float inputs (non-complex)
    
    # 1. Input projections (B, D, L) and (D, N) -> (B, D, L, N)
    # 2 ops (mul/add) per element * B * D * L * N
    flops = 0
    
    # Logic derived from the reference function provided:
    # einsum("bdl,dn->bdln")
    flops += B * D * L * N 
    
    # einsum("bdl,bnl,bdl->bdln") (Gate step)
    if with_Group:
        flops += B * D * L * N
    else:
        flops += B * D * L * N # Simplified approximation

    # Recurrent step
    # 2 ops per step * L * B * D * N
    flops += L * (B * D * N) 
    
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
        
    return flops

# ==========================================
# 2. MODEL DEFINITIONS (Simplified Imports)
# ==========================================
# [Note: I am pasting the necessary classes from your prompt here 
# to ensure the script is standalone and runnable]

from timm.models.layers import DropPath, trunc_normal_

# Mocking selective_scan if not installed to prevent crash during forward pass
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    # Dummy implementation just to allow shape tracing
    def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
        # Return a tensor of the expected shape (B, D, L)
        return u 

class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        # Pad if odd resolution
        if (W % 2 != 0) or (H % 2 != 0):
             x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
             B, H, W, C = x.shape

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, H//2, W//2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = x.view(B, H, W, self.dim_scale, self.dim_scale, C//self.dim_scale)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H*self.dim_scale, W*self.dim_scale, C//self.dim_scale)
        x= self.norm(x)
        return x

class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = x.view(B, H, W, self.dim_scale, self.dim_scale, C//self.dim_scale)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H*self.dim_scale, W*self.dim_scale, C//self.dim_scale)
        x= self.norm(x)
        return x

class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto", dropout=0., **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            groups=self.d_inner, bias=True,
            kernel_size=d_conv, padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        # Projections (Parameter based in original code, simplified logic here for profiling structure)
        # Note: In the original code provided, x_proj and dt_projs are Parameters, not layers
        # We define attributes for the custom profiler to read
        self.x_proj_weight = nn.Parameter(torch.zeros(4, self.dt_rank + self.d_state * 2, self.d_inner))
        self.dt_projs_weight = nn.Parameter(torch.zeros(4, self.d_inner, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.zeros(4, self.d_inner))
        self.A_logs = nn.Parameter(torch.zeros(4 * self.d_inner, self.d_state)) 
        self.Ds = nn.Parameter(torch.zeros(4 * self.d_inner))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        
        # Reference for forward core
        self.forward_core = self.forward_corev0

    def forward_corev0(self, x: torch.Tensor):
        # Simplified forward core for shape tracing
        B, C, H, W = x.shape
        L = H * W
        K = 4
        # Return dummy outputs for shape flow
        out_y = torch.randn(B, self.d_inner, L).to(x.device)
        return out_y, out_y, out_y, out_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlock(nn.Module):
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0, norm_layer = nn.LayerNorm, attn_drop_rate: float = 0, d_state: int = 16, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x

class VSSLayer(nn.Module):
    def __init__(self, dim, depth, attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, d_state=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=dim, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, attn_drop_rate=attn_drop, d_state=d_state)
            for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class VSSLayer_up(nn.Module):
    def __init__(self, dim, depth, attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False, d_state=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=dim, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, attn_drop_rate=attn_drop, d_state=d_state)
            for i in range(depth)])
        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x

class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim, norm_layer=norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(dim=dims[i_layer], depth=depths[i_layer], d_state=d_state, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None, use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(dim=dims_decoder[i_layer], depth=depths_decoder[i_layer], d_state=d_state, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])], norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None, use_checkpoint=use_checkpoint)
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1]//4, num_classes, 1)

    def forward(self, x):
        skip_list = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x+skip_list[-inx])
        x = self.final_up(x)
        x = x.permute(0,3,1,2)
        x = self.final_conv(x)
        return x

class VMUNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2], drop_path_rate=0.2, load_ckpt_path=None):
        super().__init__()
        self.vmunet = VSSM(in_chans=input_channels, num_classes=num_classes, depths=depths, depths_decoder=depths_decoder, drop_path_rate=drop_path_rate)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.vmunet(x)
        return logits

# ==========================================
# 3. CUSTOM HANDLER FOR SS2D
# ==========================================
def count_ss2d_custom(m: SS2D, x, y):
    """
    Custom handler for thop to count operations in SS2D, 
    including the custom einsums and selective scan.
    """
    # x is a tuple of inputs, x[0] is the tensor (B, H, W, C)
    inp = x[0]
    B, H, W, C = inp.shape
    L = H * W
    
    total_ops = 0
    
    # 1. in_proj: Linear(d_model, d_inner*2)
    # Ops: B * L * d_model * (d_inner * 2)
    total_ops += B * L * m.d_model * (m.d_inner * 2)
    
    # 2. conv2d: Depthwise (groups=d_inner)
    # Ops: B * d_inner * L * k * k
    total_ops += B * m.d_inner * L * m.d_conv * m.d_conv
    
    # 3. Custom Mamba Internal Operations (The ones thop usually misses)
    # These happen in forward_core
    K = 4
    D = m.d_inner
    R = m.dt_rank
    N = m.d_state
    
    # A) x_proj (Einsum): (B, K, D, L) @ (K, R+2N, D) -> (B, K, R+2N, L)
    # Note: The code creates 'x_dbl', which projects from D to (dt_rank + 2*d_state)
    # Ops: B * K * L * D * (R + 2*N)
    total_ops += B * K * L * D * (R + 2 * N)
    
    # B) dt_proj (Einsum): (B, K, R, L) @ (K, D, R) -> (B, K, D, L)
    # Projects from rank R back to D
    # Ops: B * K * L * R * D
    total_ops += B * K * L * R * D
    
    # C) Selective Scan
    # The scan runs on effective batch size B*K
    # We use the reference function provided by the user
    scan_flops = flops_selective_scan_ref(B=B*K, L=L, D=D, N=N, with_D=True, with_Z=False)
    total_ops += scan_flops
    
    # 4. Out Norm (LayerNorm) - Negligible but usually counted as L * D
    total_ops += B * L * D
    
    # 5. Out Proj: Linear(d_inner, d_model)
    # Ops: B * L * d_inner * d_model
    total_ops += B * L * D * m.d_model

    # 6. Gating (y * silu(z))
    # Ops: B * L * D
    total_ops += B * L * D

    # Assign to module
    m.total_ops += torch.DoubleTensor([int(total_ops)])

# ==========================================
# 4. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Settings (Standard Medical Image Seg Config)
    IMG_SIZE = 512
    IN_CHANNELS = 3
    NUM_CLASSES = 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate Model
    model = VMUNet(input_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model.eval().to(device)
    
    # Dummy Input
    input_tensor = torch.randn(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE).to(device)
    
    print(f"Profiling VMUNet with Input: {input_tensor.shape}")
    
    # Define custom handlers for thop
    # When we provide a handler for SS2D, thop will NOT recurse into it.
    # This is why our handler calculates EVERYTHING inside SS2D (Linear + Conv + Scan).
    custom_ops = {
        SS2D: count_ss2d_custom
    }
    
    macs, params = profile(model, inputs=(input_tensor, ), custom_ops=custom_ops, verbose=False)
    macs_fmt, params_fmt = clever_format([macs, params], "%.3f")
    
    print("-" * 40)
    print(f"Parameters: {params_fmt}")
    print(f"MACs:       {macs_fmt}")
    print(f"GFLOPs:     {float(macs/1e9) * 2:.3f} G (approx)")
    print("-" * 40)