import torch
import torch.nn as nn

try:
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
except ImportError:
    LigerRMSNormFunction = None


class RMSNorm(nn.RMSNorm):
    """RMSNorm：把最后一维缩放到均方根为 1，再逐维乘一个可学的系数（初始为 1）。

    系数只有 dim 个，推理时可以并进后面的线性层，不花额外的算力。
    """

    def __init__(self, dim: int):
        super().__init__(dim)

    def forward(self, x):
        if use_liger(x):
            eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
            return LigerRMSNormFunction.apply(x, self.weight, eps, 0.0, "gemma", False)
        return super().forward(x)


def use_liger(x: torch.Tensor) -> bool:
    """只在不编译的 CUDA 前向里用 Liger 的 RMSNorm 内核。

    torch.compile 里改用 PyTorch 自带的 RMSNorm：Inductor 自己会把它融合进前后的运算，不需要 Liger；
    而 liger-kernel 0.8.5 的 norm 内核在 torch 2.8–2.11 的编译路径下会编译失败、静默算出 NaN
    （eps、offset 被当成普通 Python float 传进 Triton，见 linkedin/Liger-Kernel#1601 和本仓库 issue #1）。"""
    return (x.is_cuda and x.dtype != torch.float64 and LigerRMSNormFunction is not None
            and not torch.compiler.is_compiling())
