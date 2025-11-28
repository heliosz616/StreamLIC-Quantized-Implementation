
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .qat_utils import Conv2dQ, LAGC_Q, FGPConv_Q, Conv2dUpsample_Q,SimpleChannelAddition_Q
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
import math

from .base import (
    CompressionModel,
    get_scale_table
)

# from layers import CheckerboardContext


# From Balle's tensorflow compression examples

__all__ = ["ARLiteSparse_Q"]


class unevenPadding(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self, x):
        return F.pad(x, self.params)

class RectifiedPDF(nn.Module):
    def __init__(self, cdf_num: int, cdf_len: int,scale_min=0.1,scale_max=256):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(cdf_num, cdf_len), requires_grad=True)
        self.register_buffer("q_pdf", torch.ones_like(self.logits))
        self.levels = cdf_num
        self.cdf_len=cdf_len
        self.scale_min = scale_min
        self.scale_max = scale_max


    def force_pdf_table(self, pdf_table):
        self.logits.requires_grad_(False)
        self.logits.copy_(torch.log(pdf_table.detach()))
        self.logits.requires_grad_(True)

    def build_indexes(self, scales: Tensor) -> Tensor:
        scales = torch.clamp(scales, self.scale_min, self.scale_max)
        step=(math.log(self.scale_max)-math.log(self.scale_min))/(self.levels-1)
        indexes=torch.ceil((torch.log(scales)-math.log(self.scale_min))/step)
        indexes=(torch.clamp(indexes, 0, self.levels-1)).int()
        return indexes

        # scales = torch.clamp(scales, 1.0 / 16.0, 96)
        # scale_table = torch.exp(torch.linspace(math.log(1.0 / 16), math.log(96), self.levels))
        # indexes = scales.new_full(scales.size(), len(scale_table) - 1).int()
        # for s in scale_table[:-1]:
        #     indexes -= (scales <= s).int()
        # return indexes

    def forward(self, x, scales, means, scale2index_flag=False):
        if scale2index_flag:
            pdf_index = self.build_indexes(scales.detach())
        else:
            pdf_index = scales.detach().int()
        probs = torch.softmax(self.logits - self.logits.max(dim=1, keepdim=True).values, dim=1)
        symbols = torch.round(x.detach() - means.detach() + 127).to(torch.int32)
        assert (symbols < 255).all() and (symbols >= 0).all(), "Symbol out of bounds"
        assert (pdf_index < self.levels).all() and (pdf_index >= 0).all(), "Scale index out of bounds"
        likelihoods = probs[pdf_index, symbols]
        return likelihoods


class ARLiteSparse_Q(CompressionModel):
    def __init__(self, N=96, M=192, **kwargs):
        super().__init__(**kwargs)
        c1 = 48
        c2 = 96
        c3 = 192
        self.g_a = nn.Sequential(  # input scale 1.0/255
            Conv2dQ(3, c1, 4, 2, padding=1),
            LAGC_Q(c1),
            Conv2dQ(c1, c2, 4, 2, padding=1),
            LAGC_Q(c2),
            Conv2dQ(c2, c3, 4, 2, padding=1),
            LAGC_Q(c3),
            Conv2dQ(c3, M, 4, 2, padding=1, force_scale=True)
        )

        self.g_s = nn.Sequential(  # input scale 1.0/16
            Conv2dUpsample_Q(192, c3),
            LAGC_Q(c3),
            Conv2dUpsample_Q(c3, c2),
            LAGC_Q(c2),
            Conv2dUpsample_Q(c2, c1),
            LAGC_Q(c1),
            Conv2dUpsample_Q(c1, 3)
        )

        self.h_a = nn.Sequential(  # input scale 1.0/16
            Conv2dQ(M, M, 4, 2, padding=1),
            LAGC_Q(M),
            Conv2dQ(M, N, 4, 2, padding=1, force_scale=True),
        )

        self.h_s = nn.Sequential(  # input scale 1.0
            Conv2dUpsample_Q(N, M),
            nn.LeakyReLU(negative_slope=1 / 64.0),
            Conv2dUpsample_Q(M, 2 * M, force_scale=True),
        )

        self.context_prediction = nn.Sequential(  # input scale 1.0/16
            unevenPadding((0, 0, 0, 0)),
            FGPConv_Q(M, 2 * M, 3, 1, 0, PI=12, PO=12, group_num=3),
            nn.LeakyReLU(negative_slope=1 / 64.0),
            FGPConv_Q(2 * M, 2 * M, (1, 3), 1, (0, 1), PI=12, PO=12, group_num=3, force_scale=True),
        )

        self.entropy_parameters = nn.Sequential(  # input scale 1.0/16
            SimpleChannelAddition_Q(),
            nn.LeakyReLU(negative_slope=1 / 64.0),
            # input scale 1.0/16
            FGPConv_Q(2 * M, 2 * M, (1, 3), 1, (0, 1), PI=12, PO=12, group_num=4, force_scale=True,pdf_index_flag=True),
        )

        self.scale_lower_bound = pow(2, -111.0 / 32)
        self.scale_upper_bound = pow(2, 270.0 / 32)
        print(self.scale_lower_bound)
        print(self.scale_upper_bound)

        self.rectified_cdf_y = RectifiedPDF(128, 256, scale_min=self.scale_lower_bound,
                                            scale_max=self.scale_upper_bound)
        self.rectified_cdf_z = RectifiedPDF(N, 256, scale_min=self.scale_lower_bound, scale_max=self.scale_upper_bound)
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None, scale_bound=self.scale_lower_bound)
        self.N = int(N)
        self.M = int(M)
        self.register_buffer("z_channel_index", torch.ones(N))
        self.register_buffer("y_channel_index", torch.ones(M))
        self.register_buffer("z_channel_energy", torch.ones(N))
        self.register_buffer("y_channel_energy", torch.ones(M))
        self.ar_training_flag = False
        self.pdf_training_flag = False

    def toggle_quantization(self, qat_flag: bool):
        # print(f"Enabling QAT: {qat_flag}")
        for module in self.modules():
            if isinstance(module, Conv2dQ):
                module.toggle_quantization(qat_flag)

            if isinstance(module, LAGC_Q):
                module.toggle_quantization(qat_flag)

            if isinstance(module, FGPConv_Q):
                module.toggle_quantization(qat_flag)

    def regularization_loss(self, lbd=1e-6):
        reg_loss = 0.0
        for module in self.modules():
            if isinstance(module, FGPConv_Q):
                reg_loss = reg_loss + module.get_reg_loss()
        return reg_loss * lbd

    def toggle_mask(self, flag=True):
        print(f"Apply Mask {flag}")
        for module in self.modules():
            if isinstance(module, FGPConv_Q):
                module.apply_mask = flag

    def set_all_mask(self):
        print(f"Mask Set")
        for module in self.modules():
            if isinstance(module, FGPConv_Q):
                module.set_mask()

    def fix_encoder_flag(self, flag):
        for param in self.g_a.parameters():
            param.requires_grad = not flag
        for param in self.h_a.parameters():
            param.requires_grad = not flag

        self.ar_training_flag = flag

    def toggle_pdf_training_flag(self, flag):
        for param in self.g_a.parameters():
            param.requires_grad = not flag
        for param in self.g_s.parameters():
            param.requires_grad = not flag

        for param in self.h_a.parameters():
            param.requires_grad = not flag

        for param in self.h_s.parameters():
            param.requires_grad = not flag

        for param in self.entropy_parameters.parameters():
            param.requires_grad = not flag

        for param in self.context_prediction.parameters():
            param.requires_grad = not flag

        for param in self.rectified_cdf_y.parameters():
            param.requires_grad = True
        for param in self.rectified_cdf_z.parameters():
            param.requires_grad = True
        self.pdf_training_flag = flag

        if flag:
            print("PDF training enabled")

    def init_pdf_table(self):
        device = next(self.parameters()).device
        y_sample = torch.arange(-127, 129, 1).reshape(1, 1, 256, 1).repeat(1, 128, 1, 1).to(torch.float32).to(device)
        z_sample = torch.arange(-127, 129, 1).reshape(1, 1, 256, 1).repeat(1, self.N, 1, 1).to(torch.float32).to(device)
        scale_sample = (get_scale_table(self.scale_lower_bound, self.scale_upper_bound, 128).
                        reshape(1, 128, 1, 1).repeat(1, 1, 256, 1).to(torch.float32).to(device))

        _, y_pdf = self.gaussian_conditional(y_sample, scale_sample, means=None)
        _, z_pdf = self.entropy_bottleneck(z_sample)
        y_pdf = y_pdf.reshape(128, 256)
        z_pdf = z_pdf.reshape(self.N, 256)

        self.rectified_cdf_y.force_pdf_table(y_pdf)
        self.rectified_cdf_z.force_pdf_table(z_pdf)
        print("PDF table initiated")

    def forward(self, x):
        latent_quant_flag = (not self.training) or self.ar_training_flag
        y = self.g_a(x)
        z = self.h_a(y)

        b, c, h, w = y.size()
        if self.pdf_training_flag:
            z_mean = torch.tensor(0, device=z.device)
            z_hat = torch.round(torch.clamp(z, -127, 127))
            indexes = torch.arange(0, self.N, 1).reshape(1, self.N, 1, 1).repeat(b, 1, h // 4, w // 4)
            z_likelihoods = self.rectified_cdf_z(z_hat, indexes, means=z_mean, scale2index_flag=False)
        else:
            z_hat, z_likelihoods = self.entropy_bottleneck(z, training=not latent_quant_flag)
        params = self.h_s(z_hat)


        if latent_quant_flag:
            b, c, height, width = y.size()
            y_hat_pad = F.pad(y.detach(), (1, 1, 3, 0))
            for h in range(height):
                ctx_p = self.context_prediction(y_hat_pad[:, :, h:h + 3, :].detach())
                p = params[:, :, h:h + 1, :].detach()

                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1)).detach()
                means_hat = gaussian_params[:, 0::2, :, :].detach()
                y_hat_pad[:, :, h + 3:h + 4, 1:-1] = torch.round(
                    torch.clamp(y[:, :, h:h + 1].detach() - means_hat, min=-127, max=127)) + means_hat

            y_hat = y_hat_pad[:, :, 3:, 1:-1].detach()

        else:
            y_hat = self.gaussian_conditional.quantize(
                y, "noise" if self.training else "dequantize"
            )

        ctx_params = self.context_prediction(F.pad(y_hat[:, :, :-1], (1, 1, 3, 0)))
        cat_params = torch.cat((params, ctx_params), dim=1)
        gaussian_params = self.entropy_parameters(cat_params)
        means_hat = gaussian_params[:, 0::2, :, :]
        scales_param = gaussian_params[:, 1::2, :, :] / 2

        scales_param = torch.clamp(scales_param, math.log2(self.scale_lower_bound), math.log2(self.scale_upper_bound))
        scales_param = torch.pow(2, scales_param)

        if self.pdf_training_flag:
            y_likelihoods = self.rectified_cdf_y(y_hat, scales_param, means_hat, scale2index_flag=True)
        else:
            _, y_likelihoods = self.gaussian_conditional(y, scales_param, means=means_hat)

        x_hat = self.g_s(y_hat) + 0.5

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def export_quantized_parameters(self):
        all_params = {}

        def traverse_and_export(sequential_block, initial_scale, block_name):
            nonlocal all_params
            print(f"\n--- Processing Block: {block_name} (Initial Scale: {initial_scale:.6f}) ---")
            current_scale = torch.tensor(initial_scale, dtype=torch.float32)

            for name, layer in sequential_block.named_children():
                layer_path = f"{block_name}.{name}"
                layer_type_name = layer.__class__.__name__

                if hasattr(layer, 'get_quantized_params_for_hls') and hasattr(layer, 'get_output_scale'):
                    print(f"Exporting: {layer_path} ({layer_type_name}) with input scale {current_scale.item():.6f}")
                    params = layer.get_quantized_params_for_hls(current_scale)
                    all_params[layer_path] = params

                    new_scale = layer.get_output_scale()
                    print(f"  -> New output scale for next layer: {new_scale.item():.6f}")
                    current_scale = new_scale

                elif isinstance(layer, (nn.LeakyReLU,  unevenPadding, SimpleChannelAddition_Q)):
                    print(f"Skipping:  {layer_path} ({layer_type_name}) - Scale preserved.")

                else:
                    print(f"Warning:   {layer_path} ({layer_type_name}) is an unknown type and was skipped.")

        # input g_a float range:-0.5-0.5; integer range:-255,-255
        model_blocks = {
            "g_a": (self.g_a, 1.0 / 510.0),
            "g_s": (self.g_s, 1.0 / 16.0),
            "h_a": (self.h_a, 1.0 / 16.0),
            "h_s": (self.h_s, 1.0),
            "context_prediction": (self.context_prediction, 1.0 / 16.0),
            "entropy_parameters": (self.entropy_parameters, 1.0 / 16.0),
        }

        for name, (block, initial_scale) in model_blocks.items():
            traverse_and_export(block, initial_scale, name)

        return all_params
