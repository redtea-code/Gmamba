import shutil
import sys;

import numpy as np

from dataloader.NACC_table_loade import NACC_classify

sys.path.append('./')
import nibabel as nib
from scipy.ndimage import zoom
from torch.utils import data
import torch
import os
from glob import glob
import re
from monai.utils import first
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized, RandFlip, RandAffine,
)
from table.deal_table import prepare_table, prepare_table2, prepare_table3
from utils.common import date_difference
from utils.data_normalization import adaptive_normal
import pandas as pd


def read_nii(ni_path, desired_shape=(160, 160, 96)):
    img = nib.load(ni_path)
    data = img.get_fdata()
    desired_depth = desired_shape[2]
    desired_width = desired_shape[1]
    desired_height = desired_shape[0]

    current_depth = data.shape[2]
    current_width = data.shape[1]
    current_height = data.shape[0]
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height

    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    return zoom(data, (height_factor, width_factor, depth_factor), order=1)


class MRI_classify(data.Dataset):
    def __init__(self, data_path, table_path='',
                 desired_shape=(160, 160, 96), days_threshold=-1):
        super(MRI_classify, self).__init__()
        self.mri_nii = glob(os.path.join(data_path, '*.nii.gz'))
        self.start_transformer = LoadImaged(keys=['image'])
        self.transformer = Compose(
            [
                EnsureChannelFirstd(keys=['image']),
                Resized(keys=['image'], spatial_size=desired_shape),
                ToTensord(keys=['image'])
                # ScaleIntensityRanged(keys=['image'], a_min=0.0, a_max=1000, b_min=-1.0, b_max=1.0, clip=True),
                # 空间变换（概率执行）
                # RandFlip(prob=0.5, spatial_axis=0),  # 左右翻转 (50%概率)
                # RandAffine(
                #     prob=0.8,
                #     rotate_range=(np.pi / 36, np.pi / 18, 0),  # 绕X/Y轴小角度旋转 (5-10度)
                #     scale_range=(0.1, 0.1, 0.0),  # 轻微缩放
                #     translate_range=(10, 10, 5),  # 平移（体素单位）
                #     padding_mode='border'
                # ),
            ])
        self.import_table = len(table_path)
        if self.import_table:
            self.table_df = pd.read_csv(table_path)
            print(f"Num before filter: {len(self.mri_nii)}")
            to_remove = []
            for i, path in enumerate(self.mri_nii):
                search_result = self.find_index(mri_path=path.replace("\\", "/").split(r'/')[-1],
                                                to_find_table=self.table_df)
                min_index = search_result[1]
                if search_result[0] == False:
                    to_remove.append(i)
                    # shutil.move(path,"E:/ADNI_Dataset/drop/" + os.path.basename(path))
                if self.table_df.iloc[min_index]['date_diff'] <= days_threshold:
                    # print(f'{ID} with {to_find_table.iloc[min_index]["LABEL"]} date_diff too small: {to_find_table.iloc[min_index]["date_diff"]}')
                    to_remove.append(i)
                    # shutil.move(path,"E:/ADNI_Dataset/drop/" + os.path.basename(path))

            for i in reversed(to_remove):
                self.mri_nii.pop(i)
            self.table_df = prepare_table(self.table_df)
            print(f"Num after filter: {len(self.mri_nii)}")

    def find_row(self, ID, current_datetime, ischanged, to_find_table):
        subset = to_find_table[(to_find_table['PTID'] == ID)]
        min = 31
        min_index = -1
        for index, data in subset.iterrows():
            dateInCsv = data['EXAMDATE']
            # print(dateInCsv + ' ' + str(data[10]) + ' ' +  str(ischanged))
            # 首先判断LABEL是否和 ischanged一致
            if (pd.isna(data['LABEL']) == False and (
                    (ischanged == '1' and int(data['LABEL']) == 1) or (ischanged == '0' and int(data['LABEL']) == 0))):
                if min > date_difference(dateInCsv, current_datetime):  # 保留最小的date_diff,返回查找表的索引
                    min = date_difference(dateInCsv, current_datetime)
                    min_index = index
            if min == 0:
                break

        if min != 31:
            return (True, min_index)
        else:
            print("找不到日期误差小于30天ischanged= " + str(ischanged) + " 对应数据(匹配数据信息：ID:" + str(
                ID) + "||date：" + str(current_datetime) + ")！")
            return (False, min_index)  # index should be -1

    def __getitem__(self, index):
        mri_path = self.mri_nii[index]
        # print("Name: ", mri_path.split('/')[-1])
        batch = self.start_transformer(dict(image=mri_path))
        batch['image'] = adaptive_normal(batch['image'])
        batch = self.transformer(batch)
        batch['image'] = batch["image"][:1, ...]
        batch['label'] = int(re.findall('-(\d).nii.gz', mri_path)[0])

        if self.import_table:
            _, date_index = self.find_index(mri_path.replace("\\", "/").split('/')[-1], self.table_df['info'])
            batch['cate_x'] = torch.tensor(self.table_df['cate_x'].iloc[date_index].values, dtype=torch.int64)
            batch['conti_x'] = torch.tensor(self.table_df['conti_x'].iloc[date_index].values, dtype=torch.float32)
        batch['name'] = mri_path.split('/')[-1]
        return batch

    def find_index(self, mri_path, to_find_table=None):
        ID, date, ischanged = mri_path.split('-')
        ischanged = str(ischanged.split('.')[0])
        date = date.split('_')[0] + '-' + date.split('_')[1] + '-' + date.split('_')[2]
        status, min_index = self.find_row(ID, date, ischanged, to_find_table)
        return (status, min_index)

    def __len__(self):
        return len(self.mri_nii)


def classi_dataloader(updir, image_size, batch_size, table_path, shuffle=True, Dataset="ADNI", **kwargs):
    if Dataset == "OASIS":
        dataset = OASIS_classify(updir, table_path, image_size)
    elif Dataset == "ADNI":
        dataset = MRI_classify(updir, table_path, image_size, **kwargs)
    elif Dataset == "NACC":
        dataset = NACC_classify(updir, table_path, image_size, **kwargs)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=True)


class OASIS_classify(data.Dataset):
    def __init__(self, data_path, table_path, desired_shape=(160, 160, 96), days_threshold=365):
        self.clinical_data = pd.read_csv(table_path)
        self.clinical_data['OASISID'] = self.clinical_data['OASISID'].astype(str).str.strip()
        self.clinical_data['days_to_visit'] = pd.to_numeric(self.clinical_data['days_to_visit'], errors='coerce')
        self.nii_files = glob(os.path.join(data_path, '*.nii.gz'))
        self.image_to_clinical = []
        self.days_threshold = days_threshold

        self.start_transformer = LoadImaged(keys=['image'])
        self.transformer = Compose(
            [
                EnsureChannelFirstd(keys=['image']),
                Resized(keys=['image'], spatial_size=desired_shape),
                ToTensord(keys=['image'])
            ])
        self.import_table = len(table_path)
        self.map_images_to_clinical_data()
        self.clinical_data = prepare_table2(self.clinical_data)

    def parse_filename(self, filename):
        match = re.match(r'(OAS\d+)_MR_d(\d+)_\d\.nii\.gz', filename)
        if match:
            subject_id = match.group(1)
            days_to_visit = int(match.group(2))
            return subject_id, days_to_visit
        return None, None

    def find_row(self, OASISID, target_days, table_data=None):
        if table_data is None:
            subset = self.clinical_data[self.clinical_data['OASISID'] == OASISID].copy()
        else:
            subset = table_data[table_data['OASISID'] == OASISID].copy()
        if subset.empty:
            print(f"No records found for OASISID: {OASISID}")
            return (False, -1)

        # print(f"Available days_to_visit for {OASISID}: {subset['days_to_visit'].tolist()}")
        subset['date_diff'] = abs(subset['days_to_visit'] - target_days)
        min_index = subset['date_diff'].idxmin()
        min_diff = subset.loc[min_index, 'date_diff']

        if min_diff <= self.days_threshold:
            return (True, min_index)
        else:
            print(f"No close match found for OASISID: {OASISID} (min date difference = {min_diff})")
            return (False, -1)

    def find_index(self, mri_filename, table_data=None):
        subject_id, days_to_visit = self.parse_filename(mri_filename)
        if (subject_id is not None) and (days_to_visit is not None):
            status, index = self.find_row(subject_id, days_to_visit, table_data=table_data)
            return (status, index)
        else:
            print(f"Failed to parse filename: {mri_filename}")
            return (False, -1)

    def map_images_to_clinical_data(self):
        to_remove = []
        for i, nii_file in enumerate(self.nii_files):
            filename = os.path.basename(nii_file)
            status, index = self.find_index(filename)
            if status == False:
                to_remove.append(i)
        for j in reversed(to_remove):
            self.nii_files.pop(j)

    def __getitem__(self, index):
        mri_path = self.nii_files[index]
        # print("Name: ", mri_path.split('/')[-1])
        batch = self.start_transformer(dict(image=mri_path))
        batch['image'] = adaptive_normal(batch['image'])
        batch = self.transformer(batch)
        batch['image'] = batch["image"][:1, ...]
        batch['label'] = int(re.findall('_(\d).nii.gz', mri_path)[0])

        if self.import_table:
            _, date_index = self.find_index(os.path.basename(mri_path), table_data=self.clinical_data['info'])
            batch['cate_x'] = torch.tensor(self.clinical_data['cate_x'].iloc[date_index].values, dtype=torch.int64)
            batch['conti_x'] = torch.tensor(self.clinical_data['conti_x'].iloc[date_index].values, dtype=torch.float32)
        batch['name'] = mri_path.split('/')[-1]
        return batch

    def __len__(self):
        return len(self.nii_files)


class NACC_classify(data.Dataset):
    def __init__(self, data_path, table_path, desired_shape=(160, 160, 96), days_threshold=365):
        self.clinical_data = pd.read_csv(table_path)

        self.nii_files = glob(os.path.join(data_path, '*.nii.gz'))
        self.image_to_clinical = []
        self.days_threshold = days_threshold

        self.start_transformer = LoadImaged(keys=['image'])
        self.transformer = Compose(
            [
                EnsureChannelFirstd(keys=['image']),
                Resized(keys=['image'], spatial_size=desired_shape),
                ToTensord(keys=['image'])
            ])
        self.import_table = len(table_path)
        self.map_images_to_clinical_data()
        self.clinical_data = prepare_table3(self.clinical_data)

    def parse_filename(self, filename):
        match = re.match(r'(NACC\d+)-(\d+_\d+_\d+)_\d+_\d+_\d+.0-\d\.nii\.gz', filename)
        if match:
            subject_id = match.group(1)
            visit_date = match.group(2).replace('_','-')
            return subject_id, visit_date
        return None, None

    def find_row(self, NACCID, target_days, table_data=None):
        if table_data is None:
            subset = self.clinical_data[self.clinical_data['NACCID'] == NACCID].copy()
        else:
            subset = table_data[table_data['NACCID'] == NACCID].copy()
        if subset.empty:
            print(f"No records found for NACCID: {NACCID}")
            return (False, -1)

        # print(f"Available days_to_visit for {OASISID}: {subset['days_to_visit'].tolist()}")

        subset['date_diff'] = date_difference(subset['date'], target_days,format1='%Y-%m-%d')
        min_index = subset['date_diff'].idxmin()
        min_diff = subset.loc[min_index, 'date_diff']

        if min_diff <= self.days_threshold:
            return (True, min_index)
        else:
            print(f"No close match found for NACCID: {NACCID} (min date difference = {min_diff})")
            return (False, -1)

    def find_index(self, mri_filename, table_data=None):
        subject_id, visit_date = self.parse_filename(mri_filename)
        if (subject_id is not None) and (visit_date is not None):
            status, index = self.find_row(subject_id, visit_date, table_data=table_data)
            return (status, index)
        else:
            print(f"Failed to parse filename: {mri_filename}")
            return (False, -1)

    def map_images_to_clinical_data(self):
        to_remove = []
        for i, nii_file in enumerate(self.nii_files):
            filename = os.path.basename(nii_file)
            status, index = self.find_index(filename)
            if status == False:
                to_remove.append(i)
                shutil.move(nii_file,"E:/NACC/Dataset/drop" + os.path.basename(nii_file))
        for j in reversed(to_remove):
            self.nii_files.pop(j)

    def __getitem__(self, index):
        mri_path = self.nii_files[index]
        # print("Name: ", mri_path.split('/')[-1])
        batch = self.start_transformer(dict(image=mri_path))
        batch['image'] = adaptive_normal(batch['image'])
        batch = self.transformer(batch)
        batch['image'] = batch["image"][:1, ...]
        batch['label'] = int(re.findall('-(\d).nii.gz', mri_path)[0])

        if self.import_table:
            _, date_index = self.find_index(os.path.basename(mri_path), table_data=self.clinical_data['info'])
            batch['cate_x'] = torch.tensor(self.clinical_data['cate_x'].iloc[date_index].values, dtype=torch.int64)
            batch['conti_x'] = torch.tensor(self.clinical_data['conti_x'].iloc[date_index].values, dtype=torch.float32)
        batch['name'] = mri_path.split('/')[-1]
        return batch

    def __len__(self):
        return len(self.nii_files)
if __name__ == "__main__":
    import sys;

    sys.path.append('./')
    import time
    import torch
    from utils.common import see_mri_pet
    from torchvision.utils import save_image

    data_path = r'E:\OASIS\OASIS3\classification2'
    table_path = r'E:\OASIS\OASIS3\table\table3.csv'
    # train_dataloader = classi_dataloader(r'E:\ADNI_Dataset\train',
    #                                      (160, 160, 96), batch_size=16,
    #                                      table_path='C:/Users/cyh/Downloads/AD_proj/GMamba/GEF-Mamba_ADNI_Dataset/train_data/ct_2&5_3year.csv')
    train_dataloader = classi_dataloader(data_path,
                                         (160, 160, 96), batch_size=16,
                                         table_path=table_path, use_OASIS=True)

    start_time = time.time()
    batch = first(train_dataloader)
    end_time = time.time()
    print("Time: ", end_time - start_time)
    print("Shape: ", batch['image'].shape)
    image = batch['image']
    # label = batch['label'][0, 0,...]
    save_image(see_mri_pet(image), 'combine.png')
    # plt_mri_pet(torch.cat((image, label), dim=-2), 'combine.png')
