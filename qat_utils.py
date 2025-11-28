import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

import torch
import math




def convert_tensor_to_fixed_point(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 1. Input Validation
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if torch.any(input_tensor <= 0):
        raise ValueError("All elements in the input tensor must be positive numbers.")

    max_val = 65535.0
    if torch.any(input_tensor >= max_val):
        raise ValueError(f"Input tensor contains values >= {max_val}, which is not allowed.")

    min_val = 32768.0 * (2.0 ** -63)

    clamped_input = torch.clamp(input_tensor, min=min_val)
    log2_max_val = math.log2(max_val)
    shifts = torch.floor(log2_max_val - torch.log2(clamped_input))

    shifts.clamp_(0, 63)
    scale_factors = torch.pow(2.0, shifts)
    integers = torch.round(clamped_input * scale_factors)
    integers.clamp_(0, 65535)
    return integers.to(torch.int32), shifts.to(torch.uint8)


class RoundSTE(Function):
    @staticmethod
    def forward(ctx, x):
        # Forward pass: regular rounding
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass: pretend round(x) = x (identity gradient)
        return grad_output


# convenient alias
round_ste = RoundSTE.apply


class SimpleChannelAddition_Q(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return torch.clamp(x1 + x2, min=-2047.0 / 16, max=2047.0 / 16)


class Conv2dQ(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True,
                 q_a_bit=12, q_w_bit=12, force_scale=False):
        super().__init__()

        assert q_a_bit < 16, "quantization bits must be less than 16"

        # It's better to pass all conv parameters to the constructor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)

        self.register_buffer("weight_max", torch.zeros((out_channels, 1, 1, 1)))

        self.register_buffer("weight_scales", torch.ones((out_channels, 1, 1, 1)))
        self.force_scale = force_scale
        self.fixed_scale = 1 / 16.0

        self.register_buffer("momentum", torch.tensor(0.5))
        self.quantization_flag = False

        self.output_stages = pow(2, q_a_bit - 1) - 1
        self.register_buffer("output_scale", torch.tensor(1.0 / 16))
        self.register_buffer("output_max", self.output_stages * torch.tensor(1.0 / 16))
        self.weight_stages = pow(2, q_w_bit - 1) - 1

        self.clamp_min = 1.0 / (pow(2, 31))
        self.clamp_max = pow(2, 31)
        self.q_a_bit = q_a_bit

    def update_output_scale(self, x):
        with torch.no_grad():
            max_value = torch.max(torch.abs(x))
            cur_momentum = torch.clamp(self.momentum * 1.01, 0.5, 0.999)
            if max_value < self.output_max:
                new_value = self.output_max * cur_momentum + max_value * (1 - cur_momentum)
            else:
                new_value = self.output_max * 0.99 + max_value * 0.01
            new_value = torch.clamp(new_value, self.clamp_min, self.clamp_max)
            self.output_max.copy_(new_value)
            self.output_scale.copy_(new_value / self.output_stages)
            self.momentum.copy_(cur_momentum)

    def update_weight_scale(self):
        with torch.no_grad():
            weight_max = torch.amax(torch.abs(self.conv.weight.data), dim=(1, 2, 3), keepdim=True)
            weight_max = torch.clamp(weight_max, self.clamp_min, self.clamp_max)
            self.weight_max.copy_(weight_max)
            self.weight_scales.copy_(weight_max / self.weight_stages)

    def toggle_quantization(self, qat_flag):
        self.quantization_flag = qat_flag

        if qat_flag:
            self.update_weight_scale()

    def get_output_scale(self):
        return self.output_scale

    def get_quantized_params_for_hls(self, input_scale):
        # output ro=qo*so
        # input ri=qi*si
        # weights rw=qw*sw
        # qo=(si*sw/so)*sum(qi*qw)+(bias)/so
        # bias/so is quantized to 16bits
        # calculation:(si*sw/so)*sum(qi*qw) quantized to 16 bit first,added with bias/so, then right shift to q_a_bit
        with torch.no_grad():
            # add bias at 16 bitwidth, then shift to a bitw
            res_bits = 16 - self.q_a_bit
            res_scale = 1 << res_bits

            if self.conv.bias is not None:
                quantized_bias = torch.round(
                    (res_scale * self.conv.bias.data) / self.output_scale)
            else:
                quantized_bias = None

            quantized_weight = torch.round(self.conv.weight.data / self.weight_scales)

            total_float_scale = (res_scale * input_scale * self.weight_scales) / self.output_scale

            scale_param, scale_shift = convert_tensor_to_fixed_point(total_float_scale)

            return {
                'weight': quantized_weight.to(torch.int32),
                'bias': quantized_bias.to(torch.int32).view(-1) if quantized_bias is not None else None,
                'scale_param': scale_param.view(-1),
                'scale_shift': scale_shift.view(-1),
                'float_output_scale': self.output_scale
            }

        # 'float_weight': self.conv.weight.data,
        # 'float_bias': self.conv.bias.data,
        # 'float_output_scale': self.output_scale

    def forward(self, x):
        if self.quantization_flag or (not self.training):
            quantized_weights = torch.clamp(self.conv.weight, -self.weight_max, self.weight_max)
            quantized_weights = round_ste(quantized_weights / self.weight_scales) * self.weight_scales

            output = F.conv2d(x, quantized_weights, self.conv.bias, self.conv.stride,
                              padding=self.conv.padding,
                              groups=self.conv.groups)

            clamped_output = torch.clamp(output, -self.output_max, self.output_max)
            out = round_ste(clamped_output / self.output_scale) * self.output_scale
        else:
            raw_out = self.conv(x)
            if self.force_scale:
                clamped_output = torch.clamp(raw_out, -self.output_max, self.output_max)
                out = round_ste(clamped_output / self.output_scale) * self.output_scale
            else:
                self.update_output_scale(raw_out)
                out = raw_out

        return out


class Conv2dUpsample_Q(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, bias=True,
                 q_a_bit=12, q_w_bit=12, force_scale=False):
        super().__init__()
        self.conv = Conv2dQ(in_channels, 4 * out_channels, kernel_size=2, stride=1,
                            padding=0, groups=groups, bias=bias, q_a_bit=q_a_bit, q_w_bit=q_w_bit,
                            force_scale=force_scale)

    def get_output_scale(self):
        return self.conv.output_scale

    def get_quantized_params_for_hls(self, input_scale):
        return self.conv.get_quantized_params_for_hls(input_scale)

    def forward(self, input):
        output = F.pixel_shuffle(self.conv(F.pad(input, (1, 1, 1, 1))), 2)[:, :, 1:-1, 1:-1]
        return output


class LAGC_Q(nn.Module):
    def __init__(self, channels, q_a_bit=12, q_w_bit=12):
        super().__init__()

        self.affine = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.gain_reshape = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True,
                                      groups=channels)

        self.register_buffer("output_max", torch.tensor(0.1))
        self.register_buffer("output_scale", torch.tensor(0.1))

        self.register_buffer("affine_weight_max", torch.zeros((channels, 1, 1, 1)))
        self.register_buffer("affine_weight_scale", torch.ones((channels, 1, 1, 1)))

        self.register_buffer("tanh_in_scale", torch.tensor(1.0 / 256.0))  # Fixed Quant Scle 1/256
        self.register_buffer("tanh_in_max", torch.tensor(2047.0 / 256.0))  # QUANTIZED VALUE:[-2047,2047]
        self.register_buffer("tanh_out_scale", torch.tensor(1.0 / 1024.0))  # FIX SCALE TO 1/1024

        self.register_buffer("gain_max", torch.ones((1, channels, 1, 1)))
        self.register_buffer("gain_scale", torch.ones((1, channels, 1, 1)))

        self.register_buffer("momentum", torch.tensor(0.5))
        self.quantization_flag = False

        self.weight_stages = pow(2, q_w_bit - 1) - 1
        self.output_stages = pow(2, q_a_bit - 1) - 1
        self.gain_stages = pow(2, q_a_bit - 1) - 1

        self.clamp_min = 1.0 / (pow(2, 31))
        self.clamp_max = pow(2, 31)

        self.q_a_bit = q_a_bit
        self.q_w_bit = q_w_bit

    def update_output_scale(self, x):
        with torch.no_grad():
            max_value = torch.max(torch.abs(x))
            cur_momentum = torch.clamp(self.momentum * 1.01, 0.1, 0.99)
            if max_value < self.output_max:
                new_value = self.output_max * cur_momentum + max_value * (1 - cur_momentum)
            else:
                new_value = self.output_max * 0.9 + max_value * 0.1

            new_value = torch.clamp(new_value, self.clamp_min, self.clamp_max)
            self.output_max.copy_(new_value)
            self.momentum.copy_(cur_momentum)
            self.output_scale.copy_(new_value / self.output_stages)

    def update_affine_weight_scale(self):
        with torch.no_grad():
            weight_max = torch.amax(torch.abs(self.affine.weight.data), dim=(1, 2, 3), keepdim=True)
            weight_max = torch.clamp(weight_max, self.clamp_min, self.clamp_max)
            self.affine_weight_max.copy_(weight_max)
            self.affine_weight_scale.copy_(weight_max / self.weight_stages)

            gain_range = torch.abs(self.gain_reshape.weight.data.reshape(1, -1, 1, 1))
            gain_bias = torch.abs(1 + self.gain_reshape.bias.data.reshape(1, -1, 1, 1))
            gain_max = gain_bias + gain_range
            self.gain_max.copy_(gain_max)
            self.gain_scale.copy_(gain_max / self.gain_stages)

    def toggle_quantization(self, qat_flag: bool):
        self.quantization_flag = qat_flag

        if qat_flag:
            self.update_affine_weight_scale()

    def get_output_scale(self):
        return self.output_scale

    def get_quantized_params_for_hls(self, input_scale):
        # quantization process is similar to Conv2dQ, but with multiple steps:
        with torch.no_grad():
            res_bits = 16 - self.q_a_bit
            res_scale = 1 << res_bits

            # reverse gain_weights to make sure they are positive, and reverse affine_weight and bias to ensure that the output is the same
            gain_weights = self.gain_reshape.weight.data
            flip_mask = (gain_weights < 0).view(-1)
            multiplier = torch.ones_like(flip_mask, dtype=torch.float32)
            multiplier[flip_mask] = -1.0
            if self.affine.bias is not None:
                quantized_affine_bias = torch.round(
                    (res_scale * self.affine.bias.data) / self.tanh_in_scale)
                quantized_affine_bias = quantized_affine_bias * multiplier
            else:
                quantized_affine_bias = None
            quantized_affine_weight = torch.round(self.affine.weight.data / self.affine_weight_scale)
            quantized_affine_weight = quantized_affine_weight * multiplier.view(-1, 1, 1, 1)

            float_affine_scale = res_scale * input_scale * self.affine_weight_scale.view(-1) / self.tanh_in_scale
            affine_scale_param, affine_scale_shift = convert_tensor_to_fixed_point(float_affine_scale)

            positive_gain_weights = torch.abs(gain_weights)
            float_score_scale = res_scale * self.tanh_out_scale * positive_gain_weights.view(-1) / self.gain_scale.view(
                -1)
            score_scale_param, score_scale_shift = convert_tensor_to_fixed_point(float_score_scale)

            score_bias = torch.round(
                res_scale * (self.gain_reshape.bias.data.view(-1) + 1.0) / self.gain_scale.view(-1))

            float_output_scale = input_scale * self.gain_scale.view(-1) / self.output_scale
            output_scale_param, output_scale_shift = convert_tensor_to_fixed_point(float_output_scale)

            return {
                'affine_weight': quantized_affine_weight.to(torch.int32),
                'affine_bias': quantized_affine_bias.to(torch.int32).view(
                    -1),
                'affine_scale_param': affine_scale_param.view(-1),
                'affine_scale_shift': affine_scale_shift.view(-1),
                'score_scale_param': score_scale_param.view(-1),
                'score_scale_shift': score_scale_shift.view(-1),
                'score_bias': score_bias.to(torch.int32),
                'output_scale_param': output_scale_param.view(-1),
                'output_scale_shift': output_scale_shift.view(-1),
            }

        # 'float_affine_weight':self.affine.weight.data,
        # 'float_affine_bias':self.affine.bias.data,
        # 'float_reshape_weight':self.gain_reshape.weight.data,
        # 'float_reshape_bias':self.gain_reshape.bias.data,
        # 'float_input_scale': input_scale,
        # 'float_output_scale':self.output_scale

    def forward(self, x):
        if self.quantization_flag or (not self.training):
            q_affine_w = torch.clamp(self.affine.weight, -self.affine_weight_max, self.affine_weight_max)
            q_affine_w = round_ste(q_affine_w / self.affine_weight_scale) * self.affine_weight_scale
            tanh_in = F.conv2d(x, q_affine_w, self.affine.bias, stride=1,
                               padding=0, groups=1)
            tanh_in = torch.clamp(tanh_in, -self.tanh_in_max, self.tanh_in_max)
            tanh_in = round_ste(tanh_in / self.tanh_in_scale) * self.tanh_in_scale
            tanh_out = torch.tanh(tanh_in)
            tanh_out = round_ste(tanh_out / self.tanh_out_scale) * self.tanh_out_scale
            gain = self.gain_reshape(tanh_out) + 1
            clamped_gain = torch.clamp(gain, -self.gain_max, self.gain_max)
            quantized_gain = round_ste(clamped_gain / self.gain_scale) * self.gain_scale
            y = x * quantized_gain
            clamped_output = torch.clamp(y, -self.output_max, self.output_max)
            out = round_ste(clamped_output / self.output_scale) * self.output_scale

        else:
            score = self.affine(x)
            score = torch.tanh(score)
            gain = self.gain_reshape(score) + 1
            raw_out = x * gain
            self.update_output_scale(raw_out)
            out = raw_out
        return out


class FGPConv_Q(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=True,
                 PI=12, PO=12, group_num=12, flop_level=1, q_a_bit=12, q_w_bit=12, force_scale=False, pdf_index_flag=False):
        super().__init__()

        # --- Original FGPConv Properties ---
        self.pi = PI
        self.po = PO
        self.group_num = group_num
        self.float_level = float(flop_level)
        self.stride = stride
        self.padding = padding
        self.apply_mask = False

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)

        span = float(in_channels * math.prod(kernel_size))
        self.sparse_weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size) / span)

        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels) * 0.1)
        else:
            self.register_parameter('bias', None)

        self.register_buffer("mask", torch.zeros_like(self.sparse_weight.data))
        self.register_buffer("square_means", torch.ones(1, in_channels, 1, 1))
        self.register_buffer("in_means", torch.zeros(1, in_channels, 1, 1))
        self.register_buffer("momentum_prune", torch.tensor(0.1))  # Renamed to avoid conflict
        self.total_group_num = int(math.prod(kernel_size) * group_num)

        # --- New QAT-Specific Properties ---
        self.quantization_flag = False

        # Buffers for output activation quantization
        self.force_scale = force_scale

        # Buffers for sparse_weight quantization (per-output-channel)
        self.register_buffer("weight_max", torch.zeros((out_channels, 1, 1, 1)))
        self.register_buffer("weight_scales", torch.ones((out_channels, 1, 1, 1)))

        self.register_buffer("momentum_qat", torch.tensor(0.5))  # QAT momentum

        self.fixed_scale = 1.0 / 16
        self.output_stages = pow(2, q_a_bit - 1) - 1
        self.register_buffer("output_scale", torch.tensor(1.0 / 16))
        self.register_buffer("output_max", self.output_stages * torch.tensor(1.0 / 16.0))

        self.weight_stages = pow(2, q_w_bit - 1) - 1

        self.clamp_min = 1.0 / (pow(2, 31))
        self.clamp_max = pow(2, 31)
        self.q_a_bit = q_a_bit
        self.pdf_index_flag = pdf_index_flag

    # --- New QAT Methods ---
    def update_output_scale(self, x):
        """Tracks the running max of the convolution output for QAT."""
        with torch.no_grad():
            max_value = torch.max(torch.abs(x))
            cur_momentum = torch.clamp(self.momentum_qat * 1.01, 0.5, 0.999)
            if max_value < self.output_max:
                new_value = self.output_max * cur_momentum + max_value * (1 - cur_momentum)
            else:
                new_value = self.output_max * 0.99 + max_value * 0.01
            new_value = torch.clamp(new_value, self.clamp_min, self.clamp_max)
            self.output_max.copy_(new_value)
            self.output_scale.copy_(new_value / self.output_stages)
            self.momentum_qat.copy_(cur_momentum)

    def update_weight_scale(self):
        """Calculates and freezes the weight scales for QAT."""
        with torch.no_grad():
            weight_max = torch.amax(torch.abs(self.sparse_weight.data), dim=(1, 2, 3), keepdim=True)
            weight_max = torch.clamp(weight_max, self.clamp_min, self.clamp_max)
            self.weight_max.copy_(weight_max)
            self.weight_scales.copy_(weight_max / self.weight_stages)

    def toggle_quantization(self, qat_flag: bool):
        """Switches the module between float training and QAT."""
        self.quantization_flag = qat_flag
        if qat_flag:
            self.update_weight_scale()

    # --- Original FGPConv Methods (Unchanged) ---
    def record_scale(self, x):
        momentum = torch.clamp(self.momentum_prune * 1.001, min=0, max=0.999)
        self.in_means.copy_(self.in_means * momentum + (1 - momentum) * x.detach().mean(dim=(0, 2, 3), keepdim=True))
        self.square_means.copy_(
            self.square_means * momentum + (1 - momentum) * (x.detach() ** 2).mean(dim=(0, 2, 3), keepdim=True))
        self.momentum_prune.copy_(momentum)

    def set_mask(self):
        c_out, c_in, h, w = self.sparse_weight.size()
        pi, po = self.pi, self.po
        in_scale = torch.sqrt(self.square_means - self.in_means ** 2)
        cur_norm = self.sparse_weight * in_scale
        self.mask.zero_()
        for j in range(c_out // po):
            mag_list = torch.zeros(h * w * c_in // pi)
            for k in range(c_in // pi):
                for n in range(h):
                    for m in range(w):
                        mag_list[k * h * w + n * w + m] = torch.norm(
                            cur_norm[j * po:po + j * po, k * pi:pi + k * pi, n, m], p=2)
            sorted_index = torch.argsort(mag_list, dim=0, descending=True)
            for r in range(self.total_group_num):
                idx = sorted_index[r]
                w_idx, h_idx, ic_idx = idx % w, (idx // w) % h, idx // (h * w)
                self.mask[j * po:po + j * po, ic_idx * pi:pi + ic_idx * pi, h_idx, w_idx].fill_(1.0)
        self.apply_mask = True
        print("Sparsity mask has been set.")

    def get_reg_loss(self):
        c_out, c_in, h, w = self.sparse_weight.size()
        pi, po = self.pi, self.po
        in_scale = torch.sqrt(self.square_means - self.in_means ** 2)
        in_scale = in_scale / (in_scale.max() + torch.finfo(torch.float32).eps)
        weighted_weight_groups = (self.sparse_weight * in_scale).view(c_out // po, po, c_in // pi, pi, h * w)
        group_norm = torch.norm(weighted_weight_groups, p=2, dim=(1, 3))
        group_norm = group_norm.view(c_out // po, (c_in // pi) * h * w)
        top_k, _ = torch.topk(group_norm, self.total_group_num, dim=1)
        thresholds = top_k[:, -1:]
        filtered_norm = group_norm * (group_norm < thresholds)
        return torch.sum(filtered_norm) * self.float_level

    def get_output_scale(self):
        return self.output_scale

    def get_quantized_params_for_hls(self, input_scale):
        # Similar to Conv2dQ, with masks
        with torch.no_grad():

            res_bits = 16 - self.q_a_bit
            res_scale = 1 << res_bits

            if self.bias is not None:
                if (self.pdf_index_flag):

                    quantized_bias = ((self.bias.data) / self.output_scale).view(-1)
                    quantized_bias[1::2] = (quantized_bias[1::2] / 3.0)+37
                    quantized_bias = torch.round(quantized_bias*res_scale)
                    print("Bias force to pdf index")

                else:
                    quantized_bias = torch.round(
                        (res_scale * self.bias.data) / self.output_scale)

                if ((torch.abs(quantized_bias) > 32767).any()):
                    print("Bias Overflow")

            else:
                quantized_bias = None

            quantized_weight = torch.round(self.sparse_weight.data * self.mask / self.weight_scales)

            total_float_scale = ((res_scale * input_scale * self.weight_scales) / self.output_scale).view(-1)

            if self.pdf_index_flag:
                total_float_scale[1::2] = total_float_scale[1::2] / 3.0
                print("Scale force to pdf index")

            scale_param, scale_shift = convert_tensor_to_fixed_point(total_float_scale)

            return {
                'weight': quantized_weight.to(torch.int32),
                'bias': quantized_bias.view(-1).to(torch.int32) if quantized_bias is not None else None,
                'scale_param': scale_param.view(-1).to(torch.int32),
                'scale_shift': scale_shift.view(-1).to(torch.int32),
                'mask': self.mask.to(torch.int32),
                'float_output_scale': self.output_scale
                # 'float_weight':self.sparse_weight.data,
                # 'float_bias':self.bias.data,

            }

    def forward(self, x):
        if self.quantization_flag or (not self.training):
            q_weight = torch.clamp(self.sparse_weight, -self.weight_max, self.weight_max)
            q_weight = round_ste(q_weight / self.weight_scales) * self.weight_scales

            effective_weight = q_weight * self.mask if self.apply_mask else q_weight

            output = F.conv2d(x, weight=effective_weight, bias=self.bias, stride=self.stride,
                              padding=self.padding)
            clamped_output = torch.clamp(output, -self.output_max.item(), self.output_max.item())
            out = round_ste(clamped_output / self.output_scale) * self.output_scale
        else:
            effective_weight = self.sparse_weight * self.mask if self.apply_mask else self.sparse_weight
            raw_out = F.conv2d(x, weight=effective_weight, bias=self.bias, stride=self.stride,
                               padding=self.padding)

            if not self.apply_mask:
                self.record_scale(x)

            if self.force_scale:
                clamped_output = torch.clamp(raw_out, -self.output_max, self.output_max)
                out = round_ste(clamped_output / self.output_scale) * self.output_scale
            else:
                self.update_output_scale(raw_out)
                out = raw_out

        return out
