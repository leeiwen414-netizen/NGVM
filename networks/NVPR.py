import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vmamba import selective_scan_fn
from .vmamba import SoftmaxSpatial, Permute
from .vmamba import mamba_init
from .vmamba import Linear
from .vmamba import LayerNorm
from .vmamba import cross_scan_fn, cross_merge_fn
import math
from functools import partial


class NPRPreprocessor(nn.Module):
    """Extract pooled, normalized NPR features from 2x2 pixel differences."""

    def __init__(self, grid_size=2):
        super().__init__()
        self.grid_size = grid_size
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(9)

    def forward(self, x):
        B, C, H, W = x.shape
        grid_h, grid_w = H // self.grid_size, W // self.grid_size
        x_unfold = F.unfold(x, kernel_size=self.grid_size, stride=self.grid_size)

        x_unfold = x_unfold.reshape(B, 3, 4, grid_h, grid_w)
        w1 = x_unfold[:, :, 0, :, :]
        w2 = x_unfold[:, :, 1, :, :]
        w3 = x_unfold[:, :, 2, :, :]
        w4 = x_unfold[:, :, 3, :, :]
        npr_features = torch.cat([w1 - w2, w1 - w3, w1 - w4], dim=1)

        npr_pooled = self.pool(npr_features)

        npr_pooled = npr_pooled.permute(0, 2, 3, 1)
        npr_pooled = self.norm(npr_pooled)
        return npr_pooled.permute(0, 3, 1, 2)


class NPRVSSBlock(nn.Module):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        forward_type="v04",
        channel_first=False,
        use_npr_diff=True,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()

        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first

        self.with_dconv = d_conv > 1
        self.forward = self.forwardv2

        self.use_npr_diff = use_npr_diff

        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z = True
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        FORWARD_TYPES = dict(
            v04=partial(self.forward_corev2_ys, force_fp32=False),
        )
        self.forward_core_ys = FORWARD_TYPES.get(forward_type, None)

        self.ln_1 = LayerNorm(self.d_model, channel_first=channel_first)
        self.in_proj = Linear(self.d_model, self.d_inner, bias=bias, channel_first=channel_first)
        self.act: nn.Module = act_layer()

        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        self.npr_proj = Linear(9, self.d_model, bias=bias, channel_first=channel_first)

        self.x_proj = Linear(
            self.d_inner,
            self.k_group * (self.dt_rank + self.d_state * 2),
            groups=self.k_group,
            bias=False,
            channel_first=True,
        )
        self.dt_projs = Linear(
            self.dt_rank,
            self.k_group * self.d_inner,
            groups=self.k_group,
            bias=False,
            channel_first=True,
        )

        # Allocate local difference fusion only for blocks that use it.
        if self.use_npr_diff:
            self.diff_proj = nn.Conv2d(9, self.d_inner, kernel_size=1, **factory_kwargs)
            self.fusion_weights = nn.Parameter(torch.ones(4, 4) * 0.2)

        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias, channel_first=channel_first)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.norm2 = LayerNorm(self.d_model, channel_first=channel_first)
        self.ffn = nn.Sequential(
            Linear(self.d_model, 4 * self.d_model, bias=bias, channel_first=channel_first),
            nn.GELU(),
            Linear(4 * self.d_model, self.d_model, bias=bias, channel_first=channel_first),
        )

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state,
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank))
            )
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))
        elif initialize in ["v2"]:
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank))
            )
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))
        self.dt_projs.weight.data = self.dt_projs_weight.data.view(self.dt_projs.weight.shape)
        del self.dt_projs_weight

    def forward_corev2_ys(
        self,
        x: torch.Tensor,
        force_fp32=False,
        ssoflex=True,
        selective_scan_backend=None,
        scan_mode="cross2d",
        scan_force_torch=False,
        **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = (
            dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None)
            if isinstance(scan_mode, str)
            else scan_mode
        )
        delta_softplus = True

        B, D, H, W = x.shape
        N, K, R, L = self.d_state, self.k_group, self.dt_rank, H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(
                u,
                delta,
                A,
                B,
                C,
                D,
                delta_bias,
                delta_softplus,
                ssoflex,
                backend=selective_scan_backend,
            )

        xs = cross_scan_fn(
            x,
            in_channel_first=True,
            out_channel_first=True,
            scans=_scan_mode,
            force_torch=scan_force_torch,
        )
        x_dbl = self.x_proj(xs.view(B, -1, L))
        dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
        dts = self.dt_projs(dts.contiguous().view(B, -1, L))
        xs = xs.view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp()
        Ds = self.Ds.to(torch.float)
        Bs, Cs = Bs.contiguous().view(B, K, N, L), Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        ys = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus)
        return ys.view(B, K, -1, H, W).to(x.dtype)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        # Keep the nine NPR channels separate from the main feature stream.
        npr_feat = x[:, :, :, self.d_model :]
        x_main = x[:, :, :, : self.d_model]

        npr_proj = self.npr_proj(npr_feat)

        x_main_norm = self.ln_1(x_main)
        x_mapped = self.in_proj(x_main_norm)

        if not self.channel_first:
            x_mapped = x_mapped.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x_mapped = self.conv2d(x_mapped)
        x_mapped = self.act(x_mapped)

        ys = self.forward_core_ys(x_mapped)

        npr_cf = npr_proj.permute(0, 3, 1, 2).contiguous()

        var = torch.var(npr_cf, dim=1, keepdim=True)
        fused = torch.zeros_like(ys[:, 0])

        # Shallow blocks inject directional differences; deep blocks use variance gating.
        if self.use_npr_diff:
            npr_cf_raw = npr_feat.permute(0, 3, 1, 2).contiguous()

            diff_h = torch.abs(
                F.pad(npr_cf_raw[:, :, :, 1:] - npr_cf_raw[:, :, :, :-1], (0, 1, 0, 0))
            )
            diff_v = torch.abs(
                F.pad(npr_cf_raw[:, :, 1:, :] - npr_cf_raw[:, :, :-1, :], (0, 0, 0, 1))
            )
            diff_d1 = torch.abs(
                F.pad(npr_cf_raw[:, :, 1:, 1:] - npr_cf_raw[:, :, :-1, :-1], (0, 1, 0, 1))
            )
            diff_d2 = torch.abs(
                F.pad(npr_cf_raw[:, :, 1:, :-1] - npr_cf_raw[:, :, :-1, 1:], (1, 0, 0, 1))
            )

            diff_h = self.diff_proj(diff_h)
            diff_v = self.diff_proj(diff_v)
            diff_d1 = self.diff_proj(diff_d1)
            diff_d2 = self.diff_proj(diff_d2)

            norm_weights = torch.softmax(self.fusion_weights, dim=1)

            attn = torch.sigmoid(var)

            for i in range(4):
                enh = (
                    ys[:, i]
                    + norm_weights[i, 0] * diff_h
                    + norm_weights[i, 1] * diff_v
                    + norm_weights[i, 2] * diff_d1
                    + norm_weights[i, 3] * diff_d2
                )
                fused = fused + attn * enh
        else:
            var_spatial_mean = var.mean(dim=[-2, -1], keepdim=True)

            alpha = 1.0
            T = 50.0

            threshold = var_spatial_mean * alpha

            hard_mask = torch.sigmoid((var - threshold) * T)

            for i in range(4):
                fused = fused + hard_mask * ys[:, i]

        fused = fused.permute(0, 2, 3, 1).contiguous()
        out = self.out_proj(fused)
        out = out + x_main

        out2 = self.norm2(out)
        out2 = self.ffn(out2)
        out = out + out2

        out = torch.cat([out, npr_feat], dim=-1)
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag) :] == tag
            if ret:
                value = value[: -len(tag)]
            return ret, value

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner, channel_first=channel_first),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner, channel_first=channel_first)
        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag) :] == tag
        if ret:
            value = value[: -len(tag)]
        return ret, value


class NPRVMamba(nn.Module):
    def __init__(self, num_classes=1, img_size=224):
        super().__init__()
        self.npr_preprocessor = NPRPreprocessor()

        self.vmamba_stem = nn.Conv2d(3, 96, kernel_size=4, stride=4)

        self.stem_norm = nn.LayerNorm(96)
        self.down_norm1 = nn.LayerNorm(96)
        self.down_norm2 = nn.LayerNorm(192)
        self.down_norm3 = nn.LayerNorm(384)

        self.downsample1 = nn.Conv2d(96, 192, kernel_size=2, stride=2)
        self.downsample2 = nn.Conv2d(192, 384, kernel_size=2, stride=2)
        self.downsample3 = nn.Conv2d(384, 768, kernel_size=2, stride=2)

        self.npr_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Stages 1-2 use local NPR difference injection.
        self.stage1 = nn.Sequential(
            NPRVSSBlock(d_model=96, use_npr_diff=True), NPRVSSBlock(d_model=96, use_npr_diff=True)
        )
        self.stage2 = nn.Sequential(
            NPRVSSBlock(d_model=192, use_npr_diff=True), NPRVSSBlock(d_model=192, use_npr_diff=True)
        )

        # Stages 3-4 retain variance gating without local difference injection.
        self.stage3 = nn.Sequential(
            NPRVSSBlock(d_model=384, use_npr_diff=False),
            NPRVSSBlock(d_model=384, use_npr_diff=False),
        )
        self.stage4 = nn.Sequential(
            NPRVSSBlock(d_model=768, use_npr_diff=False),
            NPRVSSBlock(d_model=768, use_npr_diff=False),
        )

        self.compress = nn.Conv2d(768, 256, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.npr_head = nn.Sequential(nn.Linear(9, 32), nn.SiLU(), nn.Linear(32, 64))

        self.norm_main = nn.LayerNorm(256)
        self.norm_npr = nn.LayerNorm(64)

        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 128), nn.SiLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )

    def _downsample_fusion(self, feat, downsample_layer, norm_layer):
        feat = feat.permute(0, 3, 1, 2).contiguous()
        main_feat, npr_feat = feat[:, :-9, :, :], feat[:, -9:, :, :]

        main_feat = main_feat.permute(0, 2, 3, 1).contiguous()
        main_feat = norm_layer(main_feat)
        main_feat = main_feat.permute(0, 3, 1, 2).contiguous()

        main_feat = downsample_layer(main_feat)
        npr_feat = self.npr_pool(npr_feat)

        feat = torch.cat([main_feat, npr_feat], dim=1)
        return feat.permute(0, 2, 3, 1).contiguous()

    def forward(self, x):
        B = x.shape[0]
        npr_feat = self.npr_preprocessor(x)
        stem_feat = self.vmamba_stem(x)

        stem_feat = stem_feat.permute(0, 2, 3, 1).contiguous()
        stem_feat = self.stem_norm(stem_feat)
        stem_feat = stem_feat.permute(0, 3, 1, 2).contiguous()

        init_npr = npr_feat.clone()

        fused_input = torch.cat([stem_feat, npr_feat], dim=1)
        fused_input = fused_input.permute(0, 2, 3, 1).contiguous()

        feat = self.stage1(fused_input)
        feat = self._downsample_fusion(feat, self.downsample1, self.down_norm1)

        feat = self.stage2(feat)
        feat = self._downsample_fusion(feat, self.downsample2, self.down_norm2)

        feat = self.stage3(feat)
        feat = self._downsample_fusion(feat, self.downsample3, self.down_norm3)

        feat = self.stage4(feat)

        feat = feat.permute(0, 3, 1, 2).contiguous()  # (B, 777, 7, 7)
        main_feat = feat[:, :-9, ...]  # (B, 768, 7, 7)

        compressed = self.compress(main_feat)  # (B, 256, 7, 7)
        assert not torch.isnan(compressed).any(), "compressed has NaN"

        B, C, H, W = compressed.shape
        N = H * W

        spatial_main = compressed.view(B, C, N).permute(0, 2, 1).contiguous()

        spatial_main = self.norm_main(spatial_main)

        npr_global_var = torch.var(init_npr, dim=[2, 3])  # (B, 9)
        assert not torch.isnan(npr_global_var).any(), "npr_global_var has NaN"

        npr_feat_64 = self.npr_head(npr_global_var)  # -> (B, 64)
        npr_feat_64 = self.norm_npr(npr_feat_64)  # -> (B, 64)

        npr_feat_expanded = npr_feat_64.unsqueeze(1).expand(B, N, -1)

        final_spatial_feat = torch.cat([spatial_main, npr_feat_expanded], dim=-1)
        assert not torch.isnan(final_spatial_feat).any(), "final_spatial_feat has NaN"

        spatial_logits = self.classifier(final_spatial_feat).squeeze(-1)
        assert not torch.isnan(spatial_logits).any(), "spatial_logits contain NaN"
        assert not torch.isinf(spatial_logits).any(), "spatial_logits contain Inf"

        # Mean the five highest patch logits to obtain the image-level logit.
        k = 5
        topk_logits, _ = torch.topk(spatial_logits, k=k, dim=1)  # (B, k)

        logits = topk_logits.mean(dim=1, keepdim=True)  # (B, 1)

        return logits
