import argparse
import os
import sys
import torch
from torch import nn
from tqdm import tqdm
import sys;

from Graph_Mamba.model import Graph_mamba_, ITCFN
from Graph_Mamba.src.test import build_graph_from_table
from utils.loss import joint_loss

sys.path.append('./')
from pytorch3dunet.unet3d.model import Residual_mid_UNet3D_vit
from classify.classifier import Combine_classfier_vit_mid
from utils.common import copy_yaml_to_folder_auto, load_config
from dataloader.pic_table_loader import classi_dataloader
from os.path import join as j
from torch.nn import functional as F
from cross_atten.mamba_transformer import Cross_mamba_both
from torchmetrics import Recall, F1Score, Accuracy, Precision, MatthewsCorrCoef, AUROC


def main(args, dataset):
    cf = load_config(args.config_path)
    device = 'cuda'
    is_debug = True
    use_best = False
    if not is_debug:
        dir = copy_yaml_to_folder_auto(args.config_path, cf['project_dir'])
        cf['project_dir'] = dir

    val_dataloader = classi_dataloader(cf['eval_path'],
                                       cf['img_sz'],
                                       cf['eval_bc'],
                                       cf['table_path'],
                                       True,
                                       days_threshold=cf['days_threshold'], use_OASIS=dataset)
    ref_model = Residual_mid_UNet3D_vit(1, 1, is_segmentation=False, f_maps=(64, 128, 256))
    ref_model.load_state_dict(torch.load('weights/model.pt'))

    # table_df = val_dataloader.dataset.table_df
    table_df = val_dataloader.dataset.clinical_data

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
    if use_best:
        ft_model.load_state_dict(torch.load(j(cf['pth_dir'], 'model_best', 'best_ft_model.pth')))
        model.load_state_dict(torch.load(j(cf['pth_dir'], 'model_best', 'best_model.pth')))
    else:
        ft_model.load_state_dict(torch.load(j(cf['pth_dir'], 'model_current', 'ft_model_current.pth')))
        model.load_state_dict(torch.load(j(cf['pth_dir'], 'model_current', 'model_current.pth')))

    loss_fn = nn.BCELoss()

    model, ft_model = model.to(device), ft_model.to(device)

    log_dir = j(dir, 'test_loss.txt') if not is_debug else 'debug.txt'
    file = open(log_dir, "w")

    for epoch in range(cf['num_epochs']):

        val_model, val_ft_model = model.eval(), ft_model.eval()
        recall_metric = Recall(average='macro', task='binary').to(device)
        f1_metric = F1Score(average='macro', task='binary').to(device)
        test_accuracy = Accuracy(average='macro', task='binary').to(device)
        test_pre = Precision(average='macro', task='binary').to(device)
        test_mcc = MatthewsCorrCoef(task='binary').to(device)
        AUC = AUROC(average='macro', task='binary').to(device)
        all_predictions = []
        all_targets = []
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
        test_pre.update(all_predictions, all_targets)
        test_mcc.update(all_predictions, all_targets)
        AUC.update(all_predictions, all_targets)
        accuracy = test_accuracy.compute() * 100
        recall = recall_metric.compute()
        f1 = f1_metric.compute()
        pre = test_pre.compute()
        mcc = test_mcc.compute()
        AUC_ = AUC.compute()
        validation_loss = (losses / total)

        print(f'Acc: {accuracy: .4f}')
        print(f'Recall: {recall: .4f}')
        print(f'F1: {f1:.4f}')
        print(f'Pre: {pre:.4f}')
        print(f'MCC: {mcc:.4f}')
        print(f'AUC: {AUC_:.4f}')

        print(f'Val_loss: {validation_loss: .4f} \n')
        logs = dict(accuracy=accuracy, recall=recall, f1=f1, validation_loss=validation_loss)
        file.write(
            f"Epoch {epoch + 1}: Acc: {accuracy: .4f}% Recall: {recall: .4f} F1: {f1:.4f} AUC: {AUC_:.4f} Val_loss: {validation_loss: .4f} \n")
        file.flush()
    file.close()


def main2(args, dataset=False):
    cf = load_config(args.config_path)
    device = 'cuda'
    is_debug = True
    use_best = True
    if not is_debug:
        dir = copy_yaml_to_folder_auto(args.config_path, cf['project_dir'])
        cf['project_dir'] = dir

    val_dataloader = classi_dataloader(cf['eval_path'],
                                       cf['img_sz'],
                                       cf['eval_bc'],
                                       cf['table_path'],
                                       True,
                                       days_threshold=cf['days_threshold'], use_OASIS=dataset)
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

    if use_best:
        ft_model.load_state_dict(torch.load(j(cf['pth_dir'], 'model_best', 'best_ft_model.pth')))
    else:
        ft_model.load_state_dict(torch.load(j(cf['pth_dir'], 'model_current', 'ft_model_current.pth')))

    loss_fn = nn.BCELoss()

    ft_model = ft_model.to(device)

    log_dir = j(dir, 'test_loss.txt') if not is_debug else 'debug.txt'
    file = open(log_dir, "w")

    for epoch in range(1):

        val_ft_model = ft_model.eval()
        recall_metric = Recall(average='macro', task='binary').to(device)
        f1_metric = F1Score(average='macro', task='binary').to(device)
        test_accuracy = Accuracy(average='macro', task='binary').to(device)
        test_pre = Precision(average='macro', task='binary').to(device)
        test_mcc = MatthewsCorrCoef(task='binary').to(device)
        AUC = AUROC(average='macro', task='binary').to(device)
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
            all_predictions.append(pred_labels.squeeze(1))
            all_targets.append(y)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        recall_metric.update(all_predictions, all_targets)
        f1_metric.update(all_predictions, all_targets)
        test_accuracy.update(all_predictions, all_targets)
        test_pre.update(all_predictions, all_targets)
        test_mcc.update(all_predictions, all_targets)
        AUC.update(all_predictions, all_targets)

        accuracy = test_accuracy.compute() * 100
        recall = recall_metric.compute()
        f1 = f1_metric.compute()
        pre = test_pre.compute()
        mcc = test_mcc.compute()
        AUC_ = AUC.compute()

        validation_loss = (losses / total)

        print(f'Acc: {accuracy: .4f}')
        print(f'Recall: {recall: .4f}')
        print(f'F1: {f1:.4f}')
        print(f'Pre: {pre:.4f}')
        print(f'MCC: {mcc:.4f}')
        print(f'AUC: {AUC_:.4f}')

        print(f'Val_loss: {validation_loss: .4f} \n')
        logs = dict(accuracy=accuracy, recall=recall, f1=f1, validation_loss=validation_loss)
        file.write(
            f"Epoch {epoch + 1}: Acc: {accuracy: .4f}% Recall: {recall: .4f} F1: {f1:.4f} Val_loss: {validation_loss: .4f} AUC: {AUC_: .4f} \n")
        file.flush()

    file.close()


def main3(args, dataset=False):
    "ICTFN"
    cf = load_config(args.config_path)
    device = 'cuda'
    is_debug = True
    use_best = True
    if not is_debug:
        dir = copy_yaml_to_folder_auto(args.config_path, cf['project_dir'])
        cf['project_dir'] = dir

    val_dataloader = classi_dataloader(cf['eval_path'],
                                       cf['img_sz'],
                                       cf['eval_bc'],
                                       cf['table_path'],
                                       True,
                                       days_threshold=cf['days_threshold'], use_OASIS=dataset)
    ref_model = Residual_mid_UNet3D_vit(1, 1, is_segmentation=False, f_maps=(64, 128, 256))
    ref_model.load_state_dict(torch.load('weights/model.pt'))

    ft_model = ITCFN()

    ref_model.eval()
    ref_model = ref_model.to(device)

    if use_best:
        ft_model.load_state_dict(torch.load(j(cf['pth_dir'], 'model_best', 'best_ft_model.pth')))
    else:
        ft_model.load_state_dict(torch.load(j(cf['pth_dir'], 'model_current', 'ft_model_current.pth')))

    loss_fn = joint_loss()

    ft_model = ft_model.to(device)

    log_dir = j(dir, 'test_loss.txt') if not is_debug else 'debug.txt'
    file = open(log_dir, "w")

    for epoch in range(cf['num_epochs']):

        val_ft_model = ft_model.eval()
        recall_metric = Recall(average='macro', task='binary').to(device)
        f1_metric = F1Score(average='macro', task='binary').to(device)
        test_accuracy = Accuracy(average='macro', task='binary').to(device)
        test_pre = Precision(average='macro', task='binary').to(device)
        test_mcc = MatthewsCorrCoef(task='binary').to(device)
        AUC = AUROC(average='macro', task='binary').to(device)
        all_predictions = []
        all_targets = []
        correct = 0
        losses = 0
        total = 0
        for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            x, x_cat, x_num, y = batch['image'], batch['cate_x'], batch['conti_x'], batch['label']
            x, x_cat, x_num, y = x.to(device), x_cat.to(device), x_num.to(device), y.to(device)
            with torch.no_grad():
                mid_input, mid_output, pet = ref_model(x, output_vit_mid=True)
                cli = torch.cat([x_cat, x_num], dim=1)
                mri_feature, pet_feature, cli_feature, output = ft_model(x, pet, cli)

            cla_loss = loss_fn(mri_feature, pet_feature, cli_feature, y, output, )
            pred_labels = torch.argmax(output, dim=1)
            total += y.size(0)
            correct += (pred_labels == y).sum().item()

            losses += cla_loss.item()
            all_predictions.append(y)
            all_targets.append(pred_labels)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        recall_metric.update(all_predictions, all_targets)
        f1_metric.update(all_predictions, all_targets)
        test_accuracy.update(all_predictions, all_targets)
        test_pre.update(all_predictions, all_targets)
        test_mcc.update(all_predictions, all_targets)
        AUC.update(all_predictions, all_targets)

        accuracy = test_accuracy.compute() * 100
        recall = recall_metric.compute()
        f1 = f1_metric.compute()
        pre = test_pre.compute()
        mcc = test_mcc.compute()
        AUC_ = AUC.compute()

        validation_loss = (losses / total)

        print(f'Acc: {accuracy: .4f}')
        print(f'Recall: {recall: .4f}')
        print(f'F1: {f1:.4f}')
        print(f'Pre: {pre:.4f}')
        print(f'MCC: {mcc:.4f}')
        print(f'AUC: {AUC_:.4f}')

        print(f'Val_loss: {validation_loss: .4f} \n')
        logs = dict(accuracy=accuracy, recall=recall, f1=f1, validation_loss=validation_loss)
        file.write(
            f"Epoch {epoch + 1}: Acc: {accuracy: .4f}% Recall: {recall: .4f} F1: {f1:.4f} Val_loss: {validation_loss: .4f} AUC: {AUC_: .4f} \n")
        file.flush()

    file.close()


def save_result(args, dataset=False):
    cf = load_config(args.config_path)
    device = 'cuda'
    use_best = True

    val_dataloader = classi_dataloader(cf['eval_path'],
                                       cf['img_sz'],
                                       cf['eval_bc'],
                                       cf['table_path'],
                                       False,
                                       days_threshold=cf['days_threshold'], use_OASIS=dataset)
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
    # ft_model = ITCFN()

    ref_model.eval()
    ref_model = ref_model.to(device)
    # model = Combine_classfier_vit_mid()
    if use_best:
        ft_model.load_state_dict(torch.load(j(cf['pth_dir'], 'model_best', 'best_ft_model.pth')))
        # model.load_state_dict(torch.load(j(cf['pth_dir'], 'best_model.pth')))
        # ft_model.load_state_dict(torch.load(j(cf['pth_dir'], 'val_data', 'epoch_4_data.pth')))
    else:
        ft_model.load_state_dict(torch.load(j(cf['pth_dir'], 'model_current', 'ft_model_current.pth')))
        # model.load_state_dict(torch.load(j(cf['pth_dir'], 'model_current.pth')))

    ft_model = ft_model.to(device)

    log_dir = j(cf['pth_dir'], 'result.txt')
    file = open(log_dir, "w")

    for epoch in range(1):

        val_ft_model = ft_model.eval()
        all_predictions = []
        all_targets = []
        correct = 0
        total = 0
        for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            x, x_cat, x_num, y = batch['image'], batch['cate_x'], batch['conti_x'], batch['label'].float()
            x, x_cat, x_num, y = x.to(device), x_cat.to(device), x_num.to(device), y.to(device)
            with torch.no_grad():
                mid_input, mid_output, PET = ref_model(x, output_vit_mid=True)
                pred = val_ft_model(x_cat, x_num, [x, PET])
                # mid_input, mid_output, pet = ref_model(x, output_vit_mid=True)
                # cli = torch.cat([x_cat, x_num], dim=1)
                # mri_feature, pet_feature, cli_feature, output = ft_model(x, pet, cli)
            pred = F.sigmoid(pred)
            # pred = torch.argmax(output, dim=1).reshape(1,1)
            # pred = output[:,1].reshape(1,1)
            total += y.size(0)

            all_predictions.append(y)
            all_targets.append(pred.squeeze(1))
        lines = [f"{int(l1)},{float(l2):.4f}\n" for l1, l2 in zip(all_predictions, all_targets)]
        file.writelines(lines)
        file.flush()

    file.close()


def main4(args, dataset=False):
    cf = load_config(args.config_path)
    device = 'cuda'
    use_best = True

    val_dataloader = classi_dataloader(cf['eval_path'],
                                       cf['img_sz'],
                                       cf['eval_bc'],
                                       cf['table_path'],
                                       False,
                                       days_threshold=cf['days_threshold'], use_OASIS=dataset)
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

    all_predictions = torch.load(j(cf['pth_dir'], 'val_data', 'epoch_5_data.pth'))["targets"]
    all_targets = torch.load(j(cf['pth_dir'], 'val_data', 'epoch_5_data.pth'))["predictions"]
    ft_model = ft_model.to(device)

    log_dir = j(cf['pth_dir'], 'result.txt')
    file = open(log_dir, "w")
    recall_metric = Recall(average='macro', task='binary').to(device)
    f1_metric = F1Score(average='macro', task='binary').to(device)
    test_accuracy = Accuracy(average='macro', task='binary').to(device)
    test_pre = Precision(average='macro', task='binary').to(device)
    test_mcc = MatthewsCorrCoef(task='binary').to(device)
    AUC = AUROC(average='macro', task='binary').to(device)

    recall_metric.update(all_predictions, all_targets)
    f1_metric.update(all_predictions, all_targets)
    test_accuracy.update(all_predictions, all_targets)
    test_pre.update(all_predictions, all_targets)
    test_mcc.update(all_predictions, all_targets)
    AUC.update(all_predictions, all_targets)
    accuracy = test_accuracy.compute() * 100
    recall = recall_metric.compute()
    f1 = f1_metric.compute()
    pre = test_pre.compute()
    mcc = test_mcc.compute()
    AUC_ = AUC.compute()

    print(f'Acc: {accuracy: .4f}')
    print(f'Recall: {recall: .4f}')
    print(f'F1: {f1:.4f}')
    print(f'Pre: {pre:.4f}')
    print(f'MCC: {mcc:.4f}')
    print(f'AUC: {AUC_:.4f}')
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/classify_mamba_config_test.yaml')

    args = parser.parse_args()
    main4(args, dataset=True)
    # save_result(args, dataset=False)
