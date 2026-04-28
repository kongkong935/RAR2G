import os
import json
import re

import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor

class FieldParser:
    def __init__(self, args, split='train'):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.split = split
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model,use_fast=True)


    def combine_reports(self, reports_list):
        """
        将reports_pure列表中的多个段落组合成一个字符串
        
        Args:
            reports_list: List[str] 多个段落组成的列表，每个元素是一段话
        
        Returns:
            str: 组合后的字符串，段落之间用空格或句号分隔
        """
        if reports_list is None:
            return ""
        if isinstance(reports_list, str):
            return reports_list
        if not isinstance(reports_list, list):
            return str(reports_list)
        
        # 过滤空字符串，并用空格连接
        reports_list = [r.strip() for r in reports_list if r and r.strip()]
        if not reports_list:
            return ""
        
        # 组合成单个字符串，段落之间用空格分隔
        combined = " ".join(reports_list)
        return combined

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # clean MIMIC-CXR reports
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # report = ' '.join(report.split()[:self.args.max_txt_len])
        return report
# #第二阶段专用
#     def parse(self, features):
#         to_return = {'id': features['id']}
#         to_return['count'] = features.get("count")
#         # 将reports_pure列表组合成字符串，然后清理
#         reports_pure_list = features.get("reports_pure")
#         reports_pure_combined = self.combine_reports(reports_pure_list)
#         to_return['reports_pure'] = self.clean_report(reports_pure_combined)
#         # 训练集和 sam split 优先使用带分割标注的报告；其它集使用原 report
#         to_return['input_text'] = self.clean_report(features.get("report"))
#         # 提取disease字段
#         to_return['disease'] = features.get("disease", "")
#         images = []
#         image = Image.open(os.path.join(self.args.base_dir, features['image_path'][0])).convert("RGB")
#         inputs = self.vit_feature_extractor(image, return_tensors="pt")["pixel_values"].squeeze(0)
#         images.append(inputs)
#         to_return["image"] = images
#         return to_return


    def parse(self, features):
        to_return = {'id': features['id']}
        to_return['count'] = features.get("count")
        # 将reports_pure列表组合成字符串，然后清理
        reports_pure_list = features.get("reports_pure")
        reports_pure_combined = self.combine_reports(reports_pure_list)
        to_return['reports_pure'] = self.clean_report(reports_pure_combined)
        # 训练集和 sam split 优先使用带分割标注的报告；其它集使用原 report
        # reports:finding,report:impression+finding
        to_return['input_text'] = self.clean_report(features.get("reports"))
        # 提取disease字段
        to_return['disease'] = features.get("disease", "")
        images = []
        for image_path in features['image_path']:
            image = Image.open(os.path.join(self.args.base_dir, image_path)).convert("RGB")
            inputs = self.vit_feature_extractor(image, return_tensors="pt")["pixel_values"].squeeze(0)
            images.append(inputs)
        if(features.get("count")==1):
            images.append(images[0])
        to_return["image"] = images
        return to_return


    def transform_with_parse(self, inputs):
        return self.parse(inputs)

class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.split = split
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args,split=split)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset



