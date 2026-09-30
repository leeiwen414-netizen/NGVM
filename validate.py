import torch
import numpy as np
from networks.NVPR import NPRVMamba
from sklearn.metrics import average_precision_score, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader
from collections import OrderedDict
import math
import os


def load_pretrained_vmamba_weights(model, ckpt_path):
    """Map VMamba pretrained weights onto the NPRVMamba model."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"VMamba pretrained checkpoint not found: {ckpt_path}\n"
            "Set --pretrain_path or NGVM_PRETRAIN_PATH to its actual location."
        )
    print(f"Loading pre-trained VMamba weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" in ckpt:
        vmamba_weights = ckpt["model"]
    else:
        vmamba_weights = ckpt

    nvpr_state_dict = model.state_dict()
    mapped_weights = {}

    print("--- 开始映射权重 ---")
    # Map stem, SS2D, and downsampling tensors by their pretrained keys.
    for k, v in vmamba_weights.items():
        if k == 'patch_embed.proj.weight':
            mapped_weights['vmamba_stem.weight'] = v
            continue
        elif k == 'patch_embed.proj.bias':
            mapped_weights['vmamba_stem.bias'] = v
            continue

        elif k == 'patch_embed.norm.weight':
            mapped_weights['stem_norm.weight'] = v
            continue
        elif k == 'patch_embed.norm.bias':
            mapped_weights['stem_norm.bias'] = v
            continue

        elif k.startswith('layers.'):
            parts = k.split('.')
            layer_idx = int(parts[1])  # 0, 1, 2, 3
            stage_idx = layer_idx + 1

            if parts[2] == 'blocks':
                block_idx = int(parts[3])

                # The target uses only the first two blocks of each pretrained stage.
                if block_idx >= 2:
                    continue

                sub_module = parts[4]
                prefix = f'stage{stage_idx}.{block_idx}.'

                if sub_module == 'norm':
                    param_name = parts[5]
                    mapped_weights[prefix + 'ln_1.' + param_name] = v
                    continue

                elif sub_module == 'self_attention':
                    param_name = parts[5]

                    if param_name == 'x_proj_weight':
                        mapped_weights[prefix + 'x_proj.weight'] = v
                    elif param_name == 'dt_projs_weight':
                        mapped_weights[prefix + 'dt_projs.weight'] = v
                    elif param_name == 'dt_projs_bias':
                        mapped_weights[prefix + 'dt_projs_bias'] = v
                    elif param_name == 'A_logs':
                        mapped_weights[prefix + 'A_logs'] = v
                    elif param_name == 'Ds':
                        mapped_weights[prefix + 'Ds'] = v
                    elif param_name == 'in_proj':
                        half_dim = v.shape[0] // 2
                        if parts[6] == 'weight':
                            mapped_weights[prefix + 'in_proj.weight'] = v[:half_dim, :]
                        elif parts[6] == 'bias':
                            mapped_weights[prefix + 'in_proj.bias'] = v[:half_dim]
                    elif param_name in ['conv2d', 'out_norm', 'out_proj']:
                        mapped_weights[prefix + param_name + '.' + parts[6]] = v

            elif parts[2] == 'downsample':

                # Preserve patch order when reshaping reduction weights to a 2x2 kernel.
                if parts[3] == 'reduction' and parts[4] == 'weight':
                    out_c, in_c4 = v.shape
                    in_c = in_c4 // 4
                    v_reshaped = v.view(out_c, 4, in_c)

                    w_conv = torch.zeros(out_c, in_c, 2, 2, dtype=v.dtype, device=v.device)
                    w_conv[:, :, 0, 0] = v_reshaped[:, 0, :]  # Top-Left
                    w_conv[:, :, 1, 0] = v_reshaped[:, 1, :]  # Bottom-Left
                    w_conv[:, :, 0, 1] = v_reshaped[:, 2, :]  # Top-Right
                    w_conv[:, :, 1, 1] = v_reshaped[:, 3, :]  # Bottom-Right

                    mapped_weights[f'downsample{stage_idx}.weight'] = w_conv
                    mapped_weights[f'downsample{stage_idx}.bias'] = torch.zeros(
                        out_c, dtype=v.dtype, device=v.device
                    )

    # Unmapped NPR-specific parameters keep their initialized values.
    nvpr_state_dict.update(mapped_weights)

    missing_keys, unexpected_keys = model.load_state_dict(nvpr_state_dict, strict=False)

    print(f"\n✅ 成功映射 {len(mapped_weights)} 个张量！")
    print("✅ 预训练权重加载完毕，缺失的键(即新增的NPR层，它们将被随机初始化)如下：")

    custom_layers = [k for k in model.state_dict().keys() if k not in mapped_weights]
    for k in custom_layers:
        print(f"  - {k}")

    return model


def validate(model, opt):
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = NPRVMamba(num_classes=1)

    load_pretrained_vmamba_weights(model, opt.pretrain_path)

    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)
    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
