import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger
from validate import load_pretrained_vmamba_weights

import random
def seed_torch(seed=1099): #设置随机种子
    random.seed(seed) # Python随机数生成器
    os.environ['PYTHONHASHSEED'] = str(seed) # 设置环境变量PYTHONHASHSEED
    np.random.seed(seed) # Numpy随机数生成器
    torch.manual_seed(seed) # PyTorch CPU随机数生成器
    torch.cuda.manual_seed(seed) # PyTorch当前GPU随机数生成器
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU. 
    torch.backends.cudnn.benchmark = False # 禁用CuDNN的自动调优功能
    torch.backends.cudnn.deterministic = True # 确保CuDNN的确定性行为
    torch.backends.cudnn.enabled = False # 禁用CuDNN


# test config
vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake'] 
multiclass = [1, 1, 1, 0, 1, 0, 0, 0] 
# 1：表示该数据集是多类别的（有多个子文件夹/类别）; 
# 0：表示该数据集是单类别的（所有图像在一个文件夹中）


def get_val_opt(): # 创建并配置验证集选项
    val_opt = TrainOptions().parse(print_options=False) # 解析训练选项参数
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split) # 设置验证集数据路径
    val_opt.isTrain = False # 设置为测试模式
    val_opt.no_resize = False # 启用图像调整大小
    val_opt.no_crop = False # 启用图像裁剪
    val_opt.serial_batches = True # 使用顺序批处理

    return val_opt # 返回配置好的验证集选项


if __name__ == '__main__':

    # torch.autograd.set_detect_anomaly(True) 

    opt = TrainOptions().parse() # 解析训练参数（命令行+配置文件）
    seed_torch(100) #设置随机种子
    Testdataroot = os.path.join(opt.dataroot, 'test') # 测试集数据路径
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split) # 训练集数据路径
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log')) # 将控制台的print输出同时保存到日志文件
    print('  '.join(list(sys.argv)) ) # 打印命令行参数
    val_opt = get_val_opt() # 获取验证集选项
    Testopt = TestOptions().parse(print_options=False) # 解析测试选项参数
    data_loader = create_dataloader(opt) # 创建训练数据加载器（迭代返回batch数据）

    # 创建TensorBoard写入器（记录训练/验证指标，用于可视化）
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train")) 
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val")) 
    
    model = Trainer(opt) # 初始化训练模型

    # 预测试阶段（训练前先测试初始模型）
    def testmodel(): 
        print('*'*25);accs = [];aps = [] # 初始化准确率和平均精度列表
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())) # 打印当前时间
        for v_id, val in enumerate(vals): # 遍历测试集
            Testopt.dataroot = '{}/{}'.format(Testdataroot, val) # 设置当前测试数据集的路径
            Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else [''] # 多类别数据集获取子文件夹
            Testopt.no_resize = False # 启用resize
            Testopt.no_crop = True # 禁用crop
            acc, ap, _, _, _, _ = validate(model.model, Testopt)  # 调用validate函数评估模型
            accs.append(acc);aps.append(ap) # 保存当前数据集的指标
            print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
        print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) # 打印平均准确率和平均精度
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())) # 打印当前时间
        
        
    # 执行预测试：先切换模型到评估模式，测试初始模型（未训练的随机权重），再切回训练模式    
    model.eval(); testmodel(); # 评估模型并测试
    model.train() # 切换回训练模式
    print(f'cwd: {os.getcwd()}') # 打印当前工作目录

    for epoch in range(opt.niter): # 遍历训练轮次
        epoch_start_time = time.time() # 记录 epoch 开始时间

        iter_data_time = time.time() # 记录数据迭代开始时间
        epoch_iter = 0 # 初始化 epoch 迭代计数器
        print('Epoch %d/%d' % (epoch+1, opt.niter)) # 打印当前 epoch 信息

        # 遍历训练数据加载器，逐batch训练
        for i, data in enumerate(data_loader): 
            model.total_steps += 1 # 全局训练步数+1
            epoch_iter += opt.batch_size # 增加 epoch 迭代计数器

            model.set_input(data) # 将batch数据输入模型
            model.optimize_parameters() # 前向→损失→反向→参数更新

            # 每隔opt.loss_freq步，打印并记录损失
            if model.total_steps % opt.loss_freq == 0: 
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), "Train loss: {} at step: {} lr {}".format(model.loss, model.total_steps, model.lr))
                train_writer.add_scalar('loss', model.loss, model.total_steps) 

        # 每个epoch结束后，执行验证+测试
        model.eval() # 切换到评估模式
        acc, ap = validate(model.model, val_opt)[:2] # 在验证集上评估模型

        # 将验证集指标写入TensorBoard
        val_writer.add_scalar('accuracy', acc, model.total_steps) 
        val_writer.add_scalar('ap', ap, model.total_steps) 

        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap)) # 打印验证结果
        
        testmodel()

        model.update_learning_rate()

        model.train() # 切换回训练模式

    model.eval(); testmodel() # 训练结束后，最终测试所有数据集
    # Save only the final epoch weights.
    model.save_networks('last')