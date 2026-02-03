import torch
from torch import nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from einops import rearrange, repeat
import timm

# This is the custom operation that thop doesn't know how to handle by default
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    print("Warning: mamba_ssm not found. Mocking selective_scan_fn for profiling.")
    # Mock function if mamba_ssm is not installed, to allow profiling
    def selective_scan_fn(
        u, delta, A, B, C, D=None, z=None, delta_bias=None,
        delta_softplus=True, return_last_state=False
    ):
        # This mock returns a tensor of the correct expected shape
        B, D, L = u.shape
        return torch.randn(B, D, L, device=u.device, dtype=u.dtype)


# Import the profiling tool
from thop import profile, clever_format


# --- MODIFICATION 1: Create a wrapper for the custom function ---
# We wrap selective_scan_fn in an nn.Module so thop can "see" it.
class SelectiveScanFnWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        u,
        delta,
        A,
        B,
        C,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=True,
        return_last_state=False,
    ):
        # Call the original imported function
        return selective_scan_fn(
            u,
            delta,
            A,
            B,
            C,
            D,
            z=z,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=return_last_state,
        )


class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=3, # Note: expand was hardcoded to 3 in your original init
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = 3 # Hardcoded in original code
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_model, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_model, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_model, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_model,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_model, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj1 = nn.Linear(self.d_model, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            bias=conv_bias, # Fixed: was conv_bias//3 which is not boolean
            kernel_size=1,
            groups=self.d_model,
            padding='same',
            **factory_kwargs,
        )
        self.conv1d_y = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            bias=conv_bias,
            kernel_size=3,
            groups=self.d_model,
            padding='same',
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            bias=conv_bias,
            kernel_size=5,
            groups=self.d_model,
            padding='same',
            **factory_kwargs,
        )

        # --- MODIFICATION 2: Instantiate the wrapper ---
        self.selective_scan_fn = SelectiveScanFnWrapper()


    def forward(self, hidden_states):
        _, seqlen, _ = hidden_states.shape
        xyz = self.in_proj(hidden_states)
        xyz = rearrange(xyz, "b l d -> b d l")
        x, y, z = xyz.chunk(3, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(self.conv1d_x(x))
        y = F.silu(self.conv1d_y(y))
        z = F.silu(self.conv1d_z(z))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
        # --- MODIFICATION 3: Call the wrapper instead of the raw function ---
        p = self.selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=None,
        )
        p = rearrange(p, "b d l -> b l d")
        y = rearrange(y, "b d l -> b l d")
        z = rearrange(z, "b d l -> b l d")
        l1 = p * y
        l1 = self.out_proj1(l1)
        l2 = l1 * z
        l2 = rearrange(l2, "b l d -> b d l")
        l2 = self.conv1d_x(l2) # Reusing conv1d_x as in the original code
        l2 = rearrange(l2, "b d l -> b l d")
        return l2


class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = MambaVisionMixer(
                d_model=input_dim//4, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1), 
                            self.avgpool(t2), 
                            self.avgpool(t3), 
                            self.avgpool(t4), 
                            self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)
            
        return att1, att2, att3, att4, att5
    
    
class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                            nn.Sigmoid())
    
    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]

    
class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        
        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()
        
    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5

        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5

        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_
    

class UltraLight_VM_UNet(nn.Module):
    
    def __init__(self, num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64],
                 split_att='fc', bridge=True):
        super().__init__()

        self.bridge = bridge
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        ) 
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.encoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4])
        )
        self.encoder6 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5])
        )

        if bridge: 
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')
        
        self.decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4])
        ) 
        self.decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3])
        ) 
        self.decoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2])
        )  
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )  
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )  
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out # b, c1, H/4, W/4 

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out # b, c2, H/8, W/8
        
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = out # b, c3, H/16, W/16
        
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        t5 = out # b, c4, H/32, W/32

        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)
        
        out = F.gelu(self.encoder6(out)) # b, c5, H/32, W/32
        
        out5 = F.gelu(self.dbn1(self.decoder1(out))) # b, c4, H/32, W/32
        out5 = torch.add(out5, t5) # b, c4, H/32, W/32
        
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16
        out4 = torch.add(out4, t4) # b, c3, H/16, W/16
        
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
        out3 = torch.add(out3, t3) # b, c2, H/8, W/8
        
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4
        out2 = torch.add(out2, t2) # b, c1, H/4, W/4 
        
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2
        out1 = torch.add(out1, t1) # b, c0, H/2, W/2
        
        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W
        
        return torch.sigmoid(out0)


# --- PROFILING SCRIPT ---

# --- THIS IS THE UPDATED HANDLER ---
def count_selective_scan(m, x, y):
    """
    This function calculates the approximate MACs for the selective_scan_fn.
    It's an approximation based on the tensor operations that *would* happen
    in a recurrent implementation.
    
    Args:
        m (nn.Module): The SelectiveScanFnWrapper module
        x (tuple): The tuple of inputs to the module's forward method
        y (tensor): The output tensor
    """
    # m: the SelectiveScanFnWrapper module
    # x: tuple of input tensors
    # y: the output tensor
    
    # --- Get Tensors ---
    # x = (u, delta, A, B, C, D, z, delta_bias, ...)
    try:
        u = x[0]
        A = x[2]
    except IndexError:
        print("Warning: Could not parse inputs for selective_scan_fn profiling.")
        m.total_ops = torch.tensor([0], dtype=torch.float64)
        return

    # --- Get Shapes ---
    # u.shape = (B, D, L)
    B, D, L = u.shape
    
    # A.shape = (D, N)
    # Handle case where A might be on a different device or dtype for profiling
    N = A.shape[1] # This is d_state (e.g., 16)
    
    # --- Calculate MACs (Multiply-Accumulate Operations) ---
    # This is an approximation of the operations in the selective scan kernel.
    
    # 1. Discretization of delta: delta_t = softplus(delta + delta_bias)
    #    - Add: B*D*L ops
    #    - Softplus: B*D*L ops
    #    - Approx MACs (1 MAC ~ 1 add + 1 mult): 2 * B * D * L
    delta_macs = 2 * B * D * L
    
    # 2. Discretization of A_bar: A_bar = exp(delta_t * A)
    #    - Outer product mult: (B,D,L,1) * (1,D,1,N) -> B*D*L*N mults
    #    - Exp: B*D*L*N ops
    #    - Approx MACs: 2 * B * D * L * N
    A_bar_macs = 2 * B * D * L * N
    
    # 3. Discretization of B_bar: B_bar = delta_t * B
    #    - Outer product mult: (B,D,1,L) * (B,1,N,L) -> B*D*L*N mults
    #    - Approx MACs: B * D * L * N
    B_bar_macs = B * D * L * N
    
    # 4. Recurrent Scan (h_t = A_bar * h + B_bar * u)
    #    - This is the parallel scan kernel. We approximate its cost as
    #    - the recurrent equivalent: L steps of (A*h + B*u)
    #    - A*h: B*D*N MACs (mult + add)
    #    - B*u: B*D*N MACs (mult + add)
    #    - Total per step: 2 * B*D*N MACs
    #    - Total over L steps: 2 * B * D * L * N
    scan_macs = 2 * B * D * L * N

    # 5. Output calculation: y_t = C * h_t
    #    - C * h: (B,1,N,L) * (B,D,N,L) -> B*D*N*L mults
    #    - Sum over N: B*D*L*N adds (approx)
    #    - Approx MACs: 2 * B * D * L * N
    C_macs = 2 * B * D * L * N
    
    # 6. D term: y = y + D * u
    #    - D * u: (D) * (B,D,L) -> B*D*L mults
    #    - Add: B*D*L adds
    #    - Approx MACs: B * D * L
    D_macs = B * D * L
    
    # --- Sum total MACs ---
    # Total = (delta_macs) + (A_bar_macs) + (B_bar_macs) + (scan_macs) + (C_macs) + (D_macs)
    # Total = (2*BDL) + (2*BDLN) + (BDLN) + (2*BDLN) + (2*BDLN) + (BDL)
    total_macs = (3 * B * D * L) + (7 * B * D * L * N)
    
    # Assign to the module's 'total_ops'
    # We use float64 for precision with large numbers
    m.total_ops = torch.tensor([total_macs], dtype=torch.float64)


if __name__ == "__main__":
    
    # 1. Instantiate the model
    # Using default parameters from the __init__
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UltraLight_VM_UNet(num_classes=1, input_channels=3)
    model.eval()
    model.to(device)
    
    # 2. Create a dummy input tensor
    # Using a common segmentation size, e.g., 256x256
    # (Batch size 1, 3 channels, 256 height, 256 width)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)

    # 3. Define the custom_ops dictionary
    # This tells thop to use our 'count_selective_scan' function
    # whenever it encounters the 'SelectiveScanFnWrapper' module.
    custom_ops = {
        SelectiveScanFnWrapper: count_selective_scan
    }

    # 4. Run the profile
    # We pass the model, inputs, and our custom handler
    # verbose=False to suppress detailed per-layer output
    macs, params = profile(model, inputs=(dummy_input, ), custom_ops=custom_ops, verbose=False)

    # 5. Format and print the results
    # 'macs' from thop are G-MACs (Multiply-Accumulate Operations)
    # GFLOPs is often approximated as 2 * G-MACs
    gmacs, mparams = clever_format([macs, params], "%.3f")
    gflops = (macs * 2) / 1e9 # Calculate GFLOPs

    print(f"--- Profiling Results for UltraLight_VM_UNet ---")
    print(f"Input size: (1, 3, 256, 256)")
    print(f"Total Parameters: {mparams}")
    print(f"Total G-MACs: {gmacs}")
    print(f"Total GFLOPs (approx. 2 * G-MACs): {gflops:.3f} G")
    print("\nNote: GFLOPs/G-MACs for 'selective_scan_fn' is an *approximation*")
    print("based on an analysis of its recurrent equivalent.")