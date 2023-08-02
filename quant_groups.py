import torch
import torch.nn as nn

def quantize(x, scale, zero, maxq, eps=1e-9):
    q = torch.clamp(torch.round(x / scale.clamp_min(eps) + zero), 0, maxq)
    return scale * (q - zero)


class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        norm=2.0,
        grid=100,
        maxshrink=0.8,
        round_zero: bool = False,
        qq_scale_bits=None,
        qq_zero_bits=None,
        qq_groupsize=16,
        qq_zero_sym=False,
        reserved_bins: int = 0,
        qqq_params=None,
    ):
        self.bits = bits
        self.maxq = torch.tensor(2**bits - 1 - reserved_bins)
        self.perchannel = perchannel
        self.sym = sym
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.round_zero = round_zero

        self.qq_scale_bits = qq_scale_bits
        self.qq_zero_bits = qq_zero_bits
        self.qq_zero_sym = qq_zero_sym
        self.qq_groupsize = qq_groupsize
        self.qqq_params = qqq_params or {}

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)
        maybe_round_zero = torch.round if self.round_zero else lambda x: x

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        xmin = x.min(1).values # min per line
        xmax = x.max(1).values # max per line

        if self.sym: # symmetric quantization ([min of range] = -[max of range]), reduces real precision
            xmax = torch.maximum(torch.abs(xmin), xmax) # gets absolute maximum component of x
            tmp = xmin < 0 
            if torch.any(tmp): # tests if any element in tmp evaluates to True
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == xmax)
        xmin[tmp] = -1 # treat case where a line has the same max and min values, and default range to [-1; +1]
        xmax[tmp] = +1

        # compute quantization scale and zero-point (per line!)
        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = maybe_round_zero(-xmin / self.scale)

        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if self.qq_scale_bits is not None:
            scale_groups = self.scale.reshape(-1, self.qq_groupsize)
            self.qq_scale = Quantizer(shape=scale_groups.shape)
            self.qq_scale.configure(self.qq_scale_bits, perchannel=True, sym=False, round_zero=False, **self.qqq_params)
            self.qq_scale.find_params(scale_groups, weight=True)
            assert self.qq_scale.scale.shape == (scale_groups.shape[0], 1), self.qq_scale.scale.shape
            self.scale = self.qq_scale.quantize(scale_groups).reshape_as(self.scale) # further restrict self.scale due to 2nd-order quantization

        if self.qq_zero_bits is not None and ((not self.round_zero) or self.qq_zero_bits < self.bits):
            zero_groups = self.zero.reshape(-1, self.qq_groupsize)
            self.qq_zero = Quantizer(shape=zero_groups.shape)
            self.qq_zero.configure(self.qq_zero_bits, perchannel=True, sym=self.qq_zero_sym, round_zero=False, **self.qqq_params)
            self.qq_zero.find_params(zero_groups, weight=True)
            assert self.qq_zero.scale.shape == (zero_groups.shape[0], 1), self.qq_zero.scale.shape
            self.zero = self.qq_zero.quantize(zero_groups).reshape_as(self.zero)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)
