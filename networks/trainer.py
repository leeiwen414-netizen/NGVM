import functools
import torch
import torch.nn as nn
from networks.NVPR import NPRVMamba
from networks.base_model import BaseModel, init_weights
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from validate import load_pretrained_vmamba_weights


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        if getattr(opt, 'continue_train', False):
            raise ValueError(
                'Cannot resume training from final-only model weights. '
                'Start a new run without --continue_train.'
            )

        # Fresh training initializes the model from the VMamba checkpoint.
        self.model = NPRVMamba(num_classes=1)
        if self.isTrain:
            print("正在将 VMamba 预训练权重注入 NPRVMamba 模型主干...")
            load_pretrained_vmamba_weights(self.model, opt.pretrain_path)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()

            if opt.optim == 'adam':
                new_params, pretrained_base, ssm_core_params = [], [], []

                # Assign distinct learning rates to new layers, backbone, and SSM core.
                for name, param in self.model.named_parameters():
                    if any(
                        key in name
                        for key in [
                            'classifier',
                            'compress',
                            'npr_proj',
                            'diff_proj',
                            'fusion_weights',
                            'ffn',
                            'norm2',
                            'down_norm',
                            'npr_head',
                            'norm_main',
                            'norm_npr',
                        ]
                    ):
                        new_params.append(param)
                    elif any(key in name for key in ['A_logs', 'Ds', 'x_proj', 'dt_projs']):
                        ssm_core_params.append(param)
                    else:
                        pretrained_base.append(param)

                optim_groups = [
                    {'params': new_params, 'lr': 2e-4, 'weight_decay': 1e-4},
                    {'params': pretrained_base, 'lr': 2e-5, 'weight_decay': 1e-4},
                    {'params': ssm_core_params, 'lr': 2e-6, 'weight_decay': 0.0},
                ]
                self.optimizer = torch.optim.AdamW(optim_groups, betas=(opt.beta1, 0.999))

                # Warm up for three epochs before cosine decay.
                warmup_epochs = 3
                total_epochs = opt.niter

                warmup_scheduler = LinearLR(
                    self.optimizer, start_factor=0.01, total_iters=warmup_epochs
                )

                cosine_scheduler = CosineAnnealingLR(
                    self.optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
                )

                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs],
                )

            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=opt.lr,
                    momentum=0.0,
                    weight_decay=0,
                )
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])

    def update_learning_rate(self, min_lr=1e-6):
        self.scheduler.step()
        self.lr = self.optimizer.param_groups[0]['lr']
        print('*' * 25)
        print(f'Scheduler step. Current Base LR updated to: {self.lr:.2e}')
        print('*' * 25)

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
