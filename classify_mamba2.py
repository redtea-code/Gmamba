import argparse
import os
import torch
from torch import nn
from tqdm import tqdm
import sys;
from Graph_Mamba.model import Graph_mamba_
from Graph_Mamba.src.test import build_graph_from_table
from utils.loss import FocalLossBinary

sys.path.append('./')
from pytorch3dunet.unet3d.model import Residual_mid_UNet3D_vit
from classify.classifier import Combine_classfier_vit_mid
from utils.common import copy_yaml_to_folder_auto, count_parameters, load_config, save_plot_data
from dataloader.pic_table_loader import classi_dataloader
from os.path import join as j
from torch.nn import functional as F
from cross_atten.mamba_transformer import Cross_mamba_both
from torchmetrics import Recall, F1Score, Accuracy, AUROC


def main(args):
    cf = load_config(args.config_path)
    device = 'cuda'
    is_debug = False
    if not is_debug:
        dir = copy_yaml_to_folder_auto(args.config_path, cf['project_dir'])
        cf['project_dir'] = dir
    train_dataloader = classi_dataloader(cf['train_path'],
                                         cf['img_sz'],
                                         cf['train_bc'],
                                         cf['table_path'],
                                         True,
                                         days_threshold=cf['days_threshold'],use_OASIS=True)
    val_dataloader = classi_dataloader(cf['eval_path'],
                                       cf['img_sz'],
                                       cf['eval_bc'],
                                       cf['table_path'],
                                       True,
                                       days_threshold=cf['days_threshold'],use_OASIS=True)
    ref_model = Residual_mid_UNet3D_vit(1, 1, is_segmentation=False, f_maps=(64, 128, 256))
    ref_model.load_state_dict(torch.load(
        'weights/model.pt'))

    # table_df = train_dataloader.dataset.table_df
    table_df = train_dataloader.dataset.clinical_data

    ft_model = Cross_mamba_both(
        categories=table_df['num_cat'],  # tuple containing the number of unique values within each category
        num_continuous=table_df['num_cont'],  # number of continuous values
        dim=cf['dim'],  # dimension, paper set at 32
        dim_out=cf['dim_out'],  # binary prediction, but could be anything
        depth=cf['depth'],  # depth, paper recommended 6
        heads=cf['heads'],  # heads, paper recommends 8
        attn_dropout=cf['attn_dropout'],  # post-attention dropout
        ff_dropout=cf['ff_dropout'],
        dim_head=cf['dim'] // cf['heads']
    )

    ref_model.eval()
    ref_model = ref_model.to(device)

    model = Combine_classfier_vit_mid(seq_length=cf['seq_length'])
    model_params = list(model.parameters())
    ft_model_params = list(ft_model.parameters())

    # 将两个模型的参数连接成一个单一的张量列表
    all_params = model_params + ft_model_params

    # 创建优化器并传递连接后的参数列表
    optimizer = torch.optim.Adam(all_params, lr=1e-4)
    val_interval = cf['val_inter']
    save_interval = cf['save_inter']
    loss_fn = nn.BCELoss()

    model, ft_model = model.to(device), ft_model.to(device)
    global_step = 0

    log_dir = j(dir, 'train_loss.txt') if not is_debug else 'debug.txt'
    file = open(log_dir, "w")
    file.write(f'The size of model: {count_parameters(ft_model)} \n')
    best_accuracy = 0.0
    best_losses = float('inf')

    if not is_debug:
        os.makedirs(j(dir, 'model_best'), exist_ok=True)
        os.makedirs(j(dir, 'model_current'), exist_ok=True)
        os.makedirs(j(dir, 'val_data'), exist_ok=True)

    for epoch in range(cf['num_epochs']):
        model.train()
        ft_model.train()
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch + 1}")
        for step, batch in enumerate(train_dataloader):
            x, x_cat, x_num, y = batch['image'], batch['cate_x'], batch['conti_x'], batch['label']
            x, x_cat, x_num, y = x.to(device), x_cat.to(device), x_num.to(device), y.to(device)
            if x.shape[1] != 1:
                print("Wrong! Got the first channel!", batch['name'])
                x = x[:, :1, ...]
            with torch.no_grad():
                mid_input, mid_output, pet = ref_model(x, output_vit_mid=True)
            mid_feature = model(mid_input, mid_output)
            pred = ft_model(x_cat, x_num, mid_feature, [x, pet])
            cla_loss = loss_fn(F.sigmoid(pred.squeeze(1)), y.float())
            cla_loss.backward()
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"cla_loss": cla_loss.detach().item()}
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)
            global_step += 1

        val_model, val_ft_model = model.eval(), ft_model.eval()
        recall_metric = Recall(average='macro', task='binary').to(device)
        f1_metric = F1Score(average='macro', task='binary').to(device)
        test_accuracy = Accuracy(average='macro', task='binary').to(device)
        all_predictions = []
        all_targets = []
        if (epoch + 1) % val_interval == 0 or epoch == cf['num_epochs'] - 1:
            correct = 0
            losses = 0
            total = 0
            for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                x, x_cat, x_num, y = batch['image'], batch['cate_x'], batch['conti_x'], batch['label'].float()
                x, x_cat, x_num, y = x.to(device), x_cat.to(device), x_num.to(device), y.to(device)
                with torch.no_grad():
                    mid_input, mid_output, pet = ref_model(x, output_vit_mid=True)
                    mid_feature = val_model(mid_input, mid_output)
                    pred = val_ft_model(x_cat, x_num, mid_feature, [x, pet])
                pred = F.sigmoid(pred)
                loss = loss_fn(pred.squeeze(1), y)
                pred_labels = pred.round()
                total += y.size(0)
                correct += (pred_labels.squeeze(1) == y).sum().item()
                losses += loss.item()
                all_predictions.append(y)
                all_targets.append(pred_labels.squeeze(1))
            all_predictions = torch.cat(all_predictions)
            all_targets = torch.cat(all_targets)
            recall_metric.update(all_predictions, all_targets)
            f1_metric.update(all_predictions, all_targets)
            test_accuracy.update(all_predictions, all_targets)
            accuracy = test_accuracy.compute() * 100
            recall = recall_metric.compute()
            f1 = f1_metric.compute()
            validation_loss = (losses / total)
            save_plot_data(epoch + 1, all_predictions, all_targets, j(dir, 'val_data'))
            # save the best model
            if accuracy > best_accuracy or (accuracy == best_accuracy and validation_loss < best_losses):
                best_accuracy = accuracy
                best_losses = validation_loss
                # we
                torch.save(model.state_dict(), j(dir, 'model_best/best_model.pth'))
                torch.save(ft_model.state_dict(), j(dir, 'model_best/best_ft_model.pth'))
                print('Saved best model')
            print(f'Acc: {accuracy: .4f}')
            print(f'Recall: {recall: .4f}')
            print(f'F1: {f1:.4f}')
            print(f'Val_loss: {validation_loss: .4f} \n')
            logs = dict(accuracy=accuracy, recall=recall, f1=f1, validation_loss=validation_loss)
            file.write(
                f"Epoch {epoch + 1}: Acc: {accuracy: .4f}% Recall: {recall: .4f} F1: {f1:.4f} Val_loss: {validation_loss: .4f} \n")
            file.flush()

        # save the model
        if (epoch + 1) % cf['save_inter'] == 0 or epoch == cf['num_epochs'] - 1:
            torch.save(model.state_dict(), j(dir, f'model_current/model_current.pth'))
            torch.save(ft_model.state_dict(), j(dir, f'model_current/ft_model_current.pth'))
            print('Saved model')
    file.close()


def main2(args):
    cf = load_config(args.config_path)
    device = 'cuda'
    is_debug = False
    if not is_debug:
        dir = copy_yaml_to_folder_auto(args.config_path, cf['project_dir'])
        cf['project_dir'] = dir
    train_dataloader = classi_dataloader(cf['train_path'],
                                         cf['img_sz'],
                                         cf['train_bc'],
                                         cf['table_path'],
                                         True,
                                         days_threshold=cf['days_threshold'])
    val_dataloader = classi_dataloader(cf['eval_path'],
                                       cf['img_sz'],
                                       cf['eval_bc'],
                                       cf['table_path'],
                                       True,
                                       days_threshold=cf['days_threshold'])
    ref_model = Residual_mid_UNet3D_vit(1, 1, is_segmentation=False, f_maps=(64, 128, 256))
    ref_model.load_state_dict(torch.load(
        r'C:\Users\cyh\Downloads\AD_proj\GFE-Mamba\weights\model.pt'))
    ref_model.eval()
    ref_model = ref_model.to(device)

    table_df = train_dataloader.dataset.table_df
    # 修改成自己的模型
    ft_model = Graph_mamba_(
        categories=table_df['num_cat'],  # tuple containing the number of unique values within each category
        num_continuous=table_df['num_cont'],  # number of continuous values
        dim=cf['dim'],  # dimension, paper set at 32
        dim_out=cf['dim_out'],  # binary prediction, but could be anything
        depth=cf['depth'],  # depth, paper recommended 6
        heads=cf['heads'],  # heads, paper recommends 8

    )
    # model 用于处理ref的潜在MRI和PET，此处不用
    # model = Combine_classfier_vit_mid(seq_length=cf['seq_length'])
    # model_params = list(model.parameters())
    ft_model_params = list(ft_model.parameters())

    all_params = ft_model_params

    # 创建优化器并传递连接后的参数列表
    optimizer = torch.optim.Adam(all_params, lr=cf['lr'])
    val_interval = cf['val_inter']
    loss_fn = nn.BCELoss()

    ft_model = ft_model.to(device)
    global_step = 0

    log_dir = j(dir, 'train_loss.txt') if not is_debug else 'debug.txt'
    file = open(log_dir, "w")
    file.write(f'The size of model: {count_parameters(ft_model)} \n')
    best_accuracy = 0.0
    best_losses = float('inf')

    if not is_debug:
        os.makedirs(j(dir, 'model_best'), exist_ok=True)
        os.makedirs(j(dir, 'model_current'), exist_ok=True)
        os.makedirs(j(dir, 'val_data'), exist_ok=True)

    for epoch in range(cf['num_epochs']):
        ft_model.train()
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch + 1}")
        for step, batch in enumerate(train_dataloader):
            x, x_cat, x_num, y = batch['image'], batch['cate_x'], batch['conti_x'], batch['label']
            x, x_cat, x_num, y = x.to(device), x_cat.to(device), x_num.to(device), y.to(device)
            if x.shape[1] != 1:
                print("Wrong! Got the first channel!", batch['name'])
                x = x[:, :1, ...]
            with torch.no_grad():
                mid_input, mid_output, pet = ref_model(x, output_vit_mid=True)

            pred = ft_model(x_cat, x_num, [x, pet])
            cla_loss = loss_fn(F.sigmoid(pred.squeeze(1)), y.float())
            cla_loss.backward()
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"cla_loss": cla_loss.detach().item()}
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)
            global_step += 1

        val_ft_model = ft_model.eval()
        recall_metric = Recall(average='macro', task='binary').to(device)
        f1_metric = F1Score(average='macro', task='binary').to(device)
        test_accuracy = Accuracy(average='macro', task='binary').to(device)
        all_predictions = []
        all_targets = []
        if (epoch + 1) % val_interval == 0 or epoch == cf['num_epochs'] - 1:
            correct = 0
            losses = 0
            total = 0
            for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                x, x_cat, x_num, y = batch['image'], batch['cate_x'], batch['conti_x'], batch['label'].float()
                x, x_cat, x_num, y = x.to(device), x_cat.to(device), x_num.to(device), y.to(device)
                with torch.no_grad():
                    mid_input, mid_output, pet = ref_model(x, output_vit_mid=True)

                    pred = val_ft_model(x_cat, x_num, [x, pet])
                pred = F.sigmoid(pred)
                loss = loss_fn(pred.squeeze(1), y)
                pred_labels = pred.round()
                total += y.size(0)
                correct += (pred_labels.squeeze(1) == y).sum().item()
                losses += loss.item()

                all_predictions.append(y)
                all_targets.append(pred_labels.squeeze(1))
            all_predictions = torch.cat(all_predictions)
            all_targets = torch.cat(all_targets)
            recall_metric.update(all_predictions, all_targets)
            f1_metric.update(all_predictions, all_targets)
            test_accuracy.update(all_predictions, all_targets)
            accuracy = test_accuracy.compute() * 100
            recall = recall_metric.compute()
            f1 = f1_metric.compute()
            validation_loss = (losses / total)
            save_plot_data(epoch + 1, all_predictions, all_targets, j(dir, 'val_data'))
            # save the best model
            if accuracy > best_accuracy or (accuracy == best_accuracy and validation_loss < best_losses):
                best_accuracy = accuracy
                best_losses = validation_loss
                # torch.save(model.state_dict(), j(dir, 'model_best/best_model.pth'))
                torch.save(ft_model.state_dict(), j(dir, 'model_best/best_ft_model.pth'))
                print('Saved best model')
            print(f'Acc: {accuracy: .4f}')
            print(f'Recall: {recall: .4f}')
            print(f'F1: {f1:.4f}')
            print(f'Val_loss: {validation_loss: .4f} \n')
            logs = dict(accuracy=accuracy, recall=recall, f1=f1, validation_loss=validation_loss)
            file.write(
                f"Epoch {epoch + 1}: Acc: {accuracy: .4f}% Recall: {recall: .4f} F1: {f1:.4f} Val_loss: {validation_loss: .4f} \n")
            file.flush()

        # save the model
        if (epoch + 1) % cf['save_inter'] == 0 or epoch == cf['num_epochs'] - 1:
            torch.save(ft_model.state_dict(), j(dir, f'model_current/ft_model_current.pth'))
            print('Saved model')
    file.close()


def main3(args):
    """use GCondNet"""
    cf = load_config(args.config_path)
    device = 'cuda'
    is_debug = False
    if not is_debug:
        dir = copy_yaml_to_folder_auto(args.config_path, cf['project_dir'])
        cf['project_dir'] = dir
    train_dataloader = classi_dataloader(cf['train_path'],
                                         cf['img_sz'],
                                         cf['train_bc'],
                                         cf['table_path'],
                                         True,
                                         days_threshold=cf['days_threshold'])
    val_dataloader = classi_dataloader(cf['eval_path'],
                                       cf['img_sz'],
                                       cf['eval_bc'],
                                       cf['table_path'],
                                       True,
                                       days_threshold=cf['days_threshold'])
    ref_model = Residual_mid_UNet3D_vit(1, 1, is_segmentation=False, f_maps=(64, 128, 256))
    ref_model.load_state_dict(torch.load(
        './weights/model.pt'))
    ref_model.eval()
    ref_model = ref_model.to(device)

    table_df = train_dataloader.dataset.table_df
    X = torch.cat([torch.tensor(table_df['cate_x'].values, dtype=torch.float32),
                   torch.tensor(table_df['conti_x'].values, dtype=torch.float32)], dim=1)
    graphs_dataset_all = build_graph_from_table(X)

    ft_model = Graph_mamba_(
        categories=table_df['num_cat'],  # tuple containing the number of unique values within each category
        num_continuous=table_df['num_cont'],  # number of continuous values
        dim=cf['dim'],  # dimension, paper set at 32
        dim_out=cf['dim_out'],  # binary prediction, but could be anything
        depth=cf['depth'],  # depth, paper recommended 6
        heads=cf['heads'],  # heads, paper recommends 8
        X=X,
        graphs_dataset_all=graphs_dataset_all
    )
    # model 用于处理ref的潜在MRI和PET，此处不用
    # model = Combine_classfier_vit_mid(seq_length=cf['seq_length'])
    # model_params = list(model.parameters())
    ft_model_params = list(ft_model.parameters())

    all_params = ft_model_params

    # 创建优化器并传递连接后的参数列表
    optimizer = torch.optim.Adam(all_params, lr=eval(cf['lr']))
    val_interval = cf['val_inter']
    # loss_fn = nn.BCELoss()
    loss_fn = FocalLossBinary()

    ft_model = ft_model.to(device)
    global_step = 0

    log_dir = j(dir, 'train_loss.txt') if not is_debug else 'debug.txt'
    file = open(log_dir, "w")
    file.write(f'The size of model: {count_parameters(ft_model)} \n')
    best_accuracy = 0.0
    best_losses = float('inf')

    if not is_debug:
        os.makedirs(j(dir, 'model_best'), exist_ok=True)
        os.makedirs(j(dir, 'model_current'), exist_ok=True)
        os.makedirs(j(dir, 'val_data'), exist_ok=True)

    for epoch in range(cf['num_epochs']):
        ft_model.train()
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch + 1}")
        for step, batch in enumerate(train_dataloader):
            x, x_cat, x_num, y = batch['image'], batch['cate_x'], batch['conti_x'], batch['label']
            x, x_cat, x_num, y = x.to(device), x_cat.to(device), x_num.to(device), y.to(device)
            if x.shape[1] != 1:
                print("Wrong! Got the first channel!", batch['name'])
                x = x[:, :1, ...]
            with torch.no_grad():
                mid_input, mid_output, pet = ref_model(x, output_vit_mid=True)

            pred = ft_model(x_cat, x_num, [x, pet])
            cla_loss = loss_fn(F.sigmoid(pred.squeeze(1)), y.float())
            cla_loss.backward()
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"cla_loss": cla_loss.detach().item()}
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)
            global_step += 1

        val_ft_model = ft_model.eval()
        recall_metric = Recall(average='macro', task='binary').to(device)
        f1_metric = F1Score(average='macro', task='binary').to(device)
        test_accuracy = Accuracy(average='macro', task='binary').to(device)
        AUC = AUROC(average='macro', task='binary').to(device)
        all_predictions = []
        all_targets = []
        if (epoch + 1) % val_interval == 0 or epoch == cf['num_epochs'] - 1:
            correct = 0
            losses = 0
            total = 0
            for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                x, x_cat, x_num, y = batch['image'], batch['cate_x'], batch['conti_x'], batch['label'].float()
                x, x_cat, x_num, y = x.to(device), x_cat.to(device), x_num.to(device), y.to(device)
                with torch.no_grad():
                    mid_input, mid_output, pet = ref_model(x, output_vit_mid=True)

                    pred = val_ft_model(x_cat, x_num, [x, pet])
                pred = F.sigmoid(pred)
                loss = loss_fn(pred.squeeze(1), y)
                pred_labels = pred.round()
                total += y.size(0)
                correct += (pred_labels.squeeze(1) == y).sum().item()
                losses += loss.item()

                all_predictions.append(y)
                all_targets.append(pred_labels.squeeze(1))
            all_predictions = torch.cat(all_predictions)
            all_targets = torch.cat(all_targets)
            recall_metric.update(all_predictions, all_targets)
            f1_metric.update(all_predictions, all_targets)
            test_accuracy.update(all_predictions, all_targets)
            AUC.update(all_predictions, all_targets)

            accuracy = test_accuracy.compute() * 100
            recall = recall_metric.compute()
            f1 = f1_metric.compute()
            AUC_ = AUC.compute()

            validation_loss = (losses / total)
            save_plot_data(epoch + 1, all_predictions, all_targets, j(dir, 'val_data'))
            # save the best model
            if accuracy > best_accuracy or (accuracy == best_accuracy and validation_loss < best_losses):
                best_accuracy = accuracy
                best_losses = validation_loss
                # we
                # torch.save(model.state_dict(), j(dir, 'model_best/best_model.pth'))
                # torch.save(ft_model.state_dict(), j(dir, 'model_best/best_ft_model.pth'))
                print('Saved best model')
            print(f'Acc: {accuracy: .4f}')
            print(f'Recall: {recall: .4f}')
            print(f'F1: {f1:.4f}')
            print(f'Val_loss: {validation_loss: .4f} \n')
            print(f'AUC: {AUC_:.4f}')
            logs = dict(accuracy=accuracy, recall=recall, f1=f1, validation_loss=validation_loss)
            file.write(
                f"Epoch {epoch + 1}: Acc: {accuracy: .4f}% Recall: {recall: .4f} F1: {f1:.4f} Val_loss: {validation_loss: .4f} AUC: {AUC_: .4f} \n")
            file.flush()

        # save the model
        if (epoch + 1) % cf['save_inter'] == 0 or epoch == cf['num_epochs'] - 1:
            torch.save(ft_model.state_dict(), j(dir, f'model_current/ft_model_current.pth'))
            print('Saved model')
    file.close()

def main4(args):
    """use GCondNet
    OASIS"""
    cf = load_config(args.config_path)
    device = 'cuda'
    is_debug = False
    if not is_debug:
        dir = copy_yaml_to_folder_auto(args.config_path, cf['project_dir'])
        cf['project_dir'] = dir
    train_dataloader = classi_dataloader(cf['train_path'],
                                         cf['img_sz'],
                                         cf['train_bc'],
                                         cf['table_path'],
                                         True,
                                         days_threshold=cf['days_threshold'],
                                         use_OASIS=True)
    val_dataloader = classi_dataloader(cf['eval_path'],
                                       cf['img_sz'],
                                       cf['eval_bc'],
                                       cf['table_path'],
                                       True,
                                       days_threshold=cf['days_threshold'],
                                       use_OASIS=True)
    ref_model = Residual_mid_UNet3D_vit(1, 1, is_segmentation=False, f_maps=(64, 128, 256))
    ref_model.load_state_dict(torch.load(
        './weights/model.pt'))
    ref_model.eval()
    ref_model = ref_model.to(device)

    table_df = train_dataloader.dataset.clinical_data
    X = torch.cat([torch.tensor(table_df['cate_x'].values, dtype=torch.float32),
                   torch.tensor(table_df['conti_x'].values, dtype=torch.float32)], dim=1)
    graphs_dataset_all = build_graph_from_table(X)

    ft_model = Graph_mamba_(
        categories=table_df['num_cat'],  # tuple containing the number of unique values within each category
        num_continuous=table_df['num_cont'],  # number of continuous values
        dim=cf['dim'],  # dimension, paper set at 32
        dim_out=cf['dim_out'],  # binary prediction, but could be anything
        depth=cf['depth'],  # depth, paper recommended 6
        heads=cf['heads'],  # heads, paper recommends 8
        X=X,
        graphs_dataset_all=graphs_dataset_all
    )
    # model 用于处理ref的潜在MRI和PET，此处不用
    # model = Combine_classfier_vit_mid(seq_length=cf['seq_length'])
    # model_params = list(model.parameters())
    ft_model_params = list(ft_model.parameters())

    all_params = ft_model_params

    # 创建优化器并传递连接后的参数列表
    optimizer = torch.optim.Adam(all_params, lr=eval(cf['lr']))
    val_interval = cf['val_inter']
    # loss_fn = nn.BCELoss()
    loss_fn = FocalLossBinary()

    ft_model = ft_model.to(device)
    global_step = 0

    log_dir = j(dir, 'train_loss.txt') if not is_debug else 'debug.txt'
    file = open(log_dir, "w")
    file.write(f'The size of model: {count_parameters(ft_model)} \n')
    best_accuracy = 0.0
    best_losses = float('inf')

    if not is_debug:
        os.makedirs(j(dir, 'model_best'), exist_ok=True)
        os.makedirs(j(dir, 'model_current'), exist_ok=True)
        os.makedirs(j(dir, 'val_data'), exist_ok=True)

    for epoch in range(cf['num_epochs']):
        ft_model.train()
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch + 1}")
        for step, batch in enumerate(train_dataloader):
            x, x_cat, x_num, y = batch['image'], batch['cate_x'], batch['conti_x'], batch['label']
            x, x_cat, x_num, y = x.to(device), x_cat.to(device), x_num.to(device), y.to(device)
            if x.shape[1] != 1:
                print("Wrong! Got the first channel!", batch['name'])
                x = x[:, :1, ...]
            with torch.no_grad():
                mid_input, mid_output, pet = ref_model(x, output_vit_mid=True)

            pred = ft_model(x_cat, x_num, [x, pet])
            cla_loss = loss_fn(F.sigmoid(pred.squeeze(1)), y.float())
            cla_loss.backward()
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"cla_loss": cla_loss.detach().item()}
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)
            global_step += 1

        val_ft_model = ft_model.eval()
        recall_metric = Recall(average='macro', task='binary').to(device)
        f1_metric = F1Score(average='macro', task='binary').to(device)
        test_accuracy = Accuracy(average='macro', task='binary').to(device)
        AUC = AUROC(average='macro', task='binary').to(device)
        all_predictions = []
        all_targets = []
        if (epoch + 1) % val_interval == 0 or epoch == cf['num_epochs'] - 1:
            correct = 0
            losses = 0
            total = 0
            for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                x, x_cat, x_num, y = batch['image'], batch['cate_x'], batch['conti_x'], batch['label'].float()
                x, x_cat, x_num, y = x.to(device), x_cat.to(device), x_num.to(device), y.to(device)
                with torch.no_grad():
                    mid_input, mid_output, pet = ref_model(x, output_vit_mid=True)
                    pred = val_ft_model(x_cat, x_num, [x, pet])
                pred = F.sigmoid(pred)
                loss = loss_fn(pred.squeeze(1), y)
                pred_labels = pred.round()
                total += y.size(0)
                correct += (pred_labels.squeeze(1) == y).sum().item()
                losses += loss.item()

                all_predictions.append(y)
                all_targets.append(pred_labels.squeeze(1))
            all_predictions = torch.cat(all_predictions)
            all_targets = torch.cat(all_targets)
            recall_metric.update(all_predictions, all_targets)
            f1_metric.update(all_predictions, all_targets)
            test_accuracy.update(all_predictions, all_targets)
            AUC.update(all_predictions, all_targets)

            accuracy = test_accuracy.compute() * 100
            recall = recall_metric.compute()
            f1 = f1_metric.compute()
            AUC_ = AUC.compute()

            validation_loss = (losses / total)
            save_plot_data(epoch + 1, all_predictions, all_targets, j(dir, 'val_data'))
            # save the best model
            if accuracy > best_accuracy or (accuracy == best_accuracy and validation_loss < best_losses):
                best_accuracy = accuracy
                best_losses = validation_loss
                # torch.save(model.state_dict(), j(dir, 'model_best/best_model.pth'))
                torch.save(ft_model.state_dict(), j(dir, 'model_best/best_ft_model.pth'))
                print('Saved best model')
            print(f'Acc: {accuracy: .4f}')
            print(f'Recall: {recall: .4f}')
            print(f'F1: {f1:.4f}')
            print(f'Val_loss: {validation_loss: .4f} \n')
            print(f'AUC: {AUC_:.4f}')
            logs = dict(accuracy=accuracy, recall=recall, f1=f1, validation_loss=validation_loss)
            file.write(
                f"Epoch {epoch + 1}: Acc: {accuracy: .4f}% Recall: {recall: .4f} F1: {f1:.4f} Val_loss: {validation_loss: .4f} AUC: {AUC_: .4f} \n")
            file.flush()

        # save the model
        if (epoch + 1) % cf['save_inter'] == 0 or epoch == cf['num_epochs'] - 1:
            torch.save(ft_model.state_dict(), j(dir, f'model_current/ft_model_current.pth'))
            print('Saved model')
    file.close()

def save_result(args,dataset=False):
    cf = load_config(args.config_path)
    device = 'cuda'
    use_best = True


    val_dataloader = classi_dataloader(cf['eval_path'],
                                       cf['img_sz'],
                                       cf['eval_bc'],
                                       cf['table_path'],
                                       False,
                                       days_threshold=cf['days_threshold'],use_OASIS=dataset)
    ref_model = Residual_mid_UNet3D_vit(1, 1, is_segmentation=False, f_maps=(64, 128, 256))
    ref_model.load_state_dict(torch.load('weights/model.pt'))

    if dataset:
        table_df = val_dataloader.dataset.clinical_data
    else:
        table_df = val_dataloader.dataset.table_df
    X = torch.cat([torch.tensor(table_df['cate_x'].values, dtype=torch.float32),
                   torch.tensor(table_df['conti_x'].values, dtype=torch.float32)], dim=1)
    graphs_dataset_all = build_graph_from_table(X)
    ft_model = Graph_mamba_(
        categories=table_df['num_cat'],  # tuple containing the number of unique values within each category
        num_continuous=table_df['num_cont'],  # number of continuous values
        dim=cf['dim'],  # dimension, paper set at 32
        dim_out=cf['dim_out'],  # binary prediction, but could be anything
        depth=cf['depth'],  # depth, paper recommended 6
        heads=cf['heads'],  # heads, paper recommends 8
        X=X,
        graphs_dataset_all=graphs_dataset_all
    )

    ref_model.eval()
    ref_model = ref_model.to(device)
    # model = Combine_classfier_vit_mid()
    if use_best:
        ft_model.load_state_dict(torch.load(j(cf['pth_dir'],'model_best', 'best_ft_model.pth')))
        # model.load_state_dict(torch.load(j(cf['pth_dir'], 'best_model.pth')))
    else:
        ft_model.load_state_dict(torch.load(j(cf['pth_dir'],'model_current', 'ft_model_current.pth')))
        # model.load_state_dict(torch.load(j(cf['pth_dir'], 'model_current.pth')))

    loss_fn = nn.BCELoss()
    ft_model = ft_model.to(device)
    
    log_dir = j(cf['pth_dir'], 'result.txt')
    file = open(log_dir, "w")
    
    for epoch in range(1):
    
        val_ft_model = ft_model.eval()
        all_predictions = []
        all_targets = []
        correct = 0
        losses = 0
        total = 0
        for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            x, x_cat, x_num, y = batch['image'], batch['cate_x'], batch['conti_x'], batch['label'].float()
            x, x_cat, x_num, y = x.to(device), x_cat.to(device), x_num.to(device), y.to(device)
            with torch.no_grad():
                mid_input, mid_output, PET = ref_model(x, output_vit_mid=True)
                pred = val_ft_model(x_cat, x_num, [x, PET])
            pred = F.sigmoid(pred)
            loss = loss_fn(pred.squeeze(1), y)
            pred_labels = pred.round()
            total += y.size(0)
            correct += (pred_labels.squeeze(1) == y).sum().item()
            losses += loss.item()
            all_predictions.append(y)
            all_targets.append(pred_labels.squeeze(1))
        lines = [f"{int(l1)},{int(l2)}\n" for l1, l2 in zip(all_predictions, all_targets)]
        file.writelines(lines)
        file.flush()
    
    file.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/classify_mamba_config_oasis.yaml')

    args = parser.parse_args()
    main(args)
