import torch

class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, strides=None, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, strides, padding, ceil_mode, count_include_pad, divisor_override)
        self.epsilon = 1e-6 # Define epsilon for stability

    def forward(self, x):
        # --- Start: Numerically Stable Implementation ---
        
        # 1. Get max value from tensor to prevent overflow
        # We detach it so it's treated as a constant and doesn't affect gradients
        x_max = torch.max(x).detach()
        
        # 2. Subtract max from x for stable exponentiation
        x_stable = x - x_max
        
        # 3. Calculate stable exponential
        x_exp = torch.exp(x_stable)
        
        # 4. Calculate numerator: avg(x * exp(x-max(x)))
        #    We must use the *original* x here for the weighted average
        numerator = self.avgpool(x * x_exp)
        
        # 5. Calculate denominator: avg(exp(x-max(x)))
        denominator = self.avgpool(x_exp)
        
        # 6. Divide, adding epsilon to denominator to prevent division by zero
        return numerator / (denominator + self.epsilon)
        # --- End: Numerically Stable Implementation ---


def downsample_soft():
    return torch.nn.MaxPool2d(kernel_size=2, stride=2)
    # return SoftPooling2D(2, 2)