from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import xml.etree.ElementTree as ET
from PIL import Image, ImageEnhance, ImageFilter
import os
import random
import numpy as np
import torchvision.ops as ops

from augmentation import CustomColorJitter, CustomHorizontalFlip, CustomMixup, CustomRandomCrop, CustomRandomRotation, CustomRandomScale
from voc_classes import VOC_CLASSES


class CustomDataset(Dataset):
    def __init__(self, root, year="2012", img_set=None, transform=None, S=7, B=2, C=20, augment=True):
        self.root = root
        self.year = year
        self.img_set = img_set
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.augment = augment
        self.current_idx = 0

        self.anno_path = os.path.join(root, 'VOCdevkit', f'VOC{self.year}', 'Annotations')
        self.img_path = os.path.join(root, 'VOCdevkit', f'VOC{self.year}', 'JPEGImages')
        self.filter_path = os.path.join(root, 'VOCdevkit', f'VOC{self.year}', 'ImageSets', 'Main')

        self.horizontal_flip = CustomHorizontalFlip(p=0.5)
        self.color_jitter = CustomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
        self.random_scale = CustomRandomScale(scale_range=(0.8, 1.2), p=0.5)
        self.random_rotation = CustomRandomRotation(degrees=10, p=0.3)
        self.random_crop = CustomRandomCrop(min_crop_ratio=0.6, max_crop_ratio=1.0, p=0.5)

        try:
            all_annotations = os.listdir(self.anno_path)
        except Exception as e:
            all_annotations = []
            print(f"VOC{self.year} Anno 폴더 열람 실패: {e}")

        self.filtered_annotations = []
        self.filtered_images = []

        # 클래스별 필터 확률 설정
        class_prob_filter = {
            "person": 0.3,
            "chair": 0.6,
            "car": 0.6
        }

        for anno_file in all_annotations:
            try:
                tree = ET.parse(os.path.join(self.anno_path, anno_file))
                root_elem = tree.getroot()
                objects = root_elem.findall("object")
                labels = [obj.find("name").text for obj in objects]

                should_skip = False
                for label in labels:
                    if label in class_prob_filter:
                        if random.random() > class_prob_filter[label]: # 0~1 사이의 난수, 설정한 퍼센트보다 크면 생략 -> 특정 라벨 이미지 줄이기
                            should_skip = True
                            break

                if should_skip:
                    continue

                img_file = anno_file.replace(".xml", ".jpg")
                self.filtered_annotations.append(anno_file)
                self.filtered_images.append(img_file)

            except Exception as e:
                continue


        # img_set 적용
        if img_set is not None:
            self.filtered_annotations, self.filtered_images = self.img_set_mode()

        print(f"VOC{self.year} {img_set} 데이터셋 로드: {len(self.filtered_images)}개 이미지")
            # 수평 반전 (Horizontal Flip)


    def img_set_mode(self):
        img_set_path = os.path.join(self.filter_path, f'{self.img_set}.txt')
        try:
            with open(img_set_path, 'r') as f:
                file_ids = f.read().strip().split()
        except Exception as e:
            print(f"VOC{self.year} {self.img_set}.txt 파일을 열 수 없습니다: {e}")
            file_ids = []

        filtered_annotations = []
        filtered_images = []
        for file_id in file_ids:
            anno_file = f'{file_id}.xml'
            img_file = f'{file_id}.jpg'

            # Check if both annotation and image files exist before adding them to the lists
            if anno_file in self.filtered_annotations and img_file in self.filtered_images:
                filtered_annotations.append(os.path.join(self.anno_path, anno_file)) # Add full path
                filtered_images.append(os.path.join(self.img_path, img_file)) # Add full path

        return filtered_annotations, filtered_images

    def __len__(self):
        return len(self.filtered_images)

    def __str__(self):
        if self.annotation_error:
            anno_cnt = f"Anno폴더 열람 실패 : {self.annotation_error}"
        else:
            anno_cnt = f"Anno폴더에서 찾은 *.xml파일 개수 : {len(self.filtered_annotations)}"

        if self.image_error:
            img_cnt = f"Img폴더 열람 실패 : {self.image_error}"
        else:
            img_cnt = f"Img폴더 찾은 *.JPG파일 개수 : {len(self.filtered_images)}"

        return f"{anno_cnt}\n{img_cnt} "

    def parse_voc_xml(self, node):
        boxes = []
        for obj in node.findall('object'):
            cls_name = obj.find('name').text
            if cls_name in VOC_CLASSES:
                cls_idx = VOC_CLASSES.index(cls_name)
                xml_box = obj.find('bndbox')
                x1 = int(xml_box.find('xmin').text)
                y1 = int(xml_box.find('ymin').text)
                x2 = int(xml_box.find('xmax').text)
                y2 = int(xml_box.find('ymax').text)
                boxes.append((x1, y1, x2, y2, cls_idx))
        return boxes

    def __getitem__(self, idx):
        self.current_idx = idx
        image_path = self.filtered_images[idx]
        annotation_path = self.filtered_annotations[idx]

        image = Image.open(image_path).convert("RGB")

        tree = ET.parse(annotation_path)
        root = tree.getroot()
        boxes = self.parse_voc_xml(root)

        image_width = int(root.find("size/width").text)
        image_height = int(root.find("size/height").text)

        # 기본 데이터 증강 (좌우 반전)
        image, boxes = self.horizontal_flip(image, boxes, image_width)

        # 추가 데이터 증강 적용
        if self.augment and boxes:  # 박스가 있는 경우에만 증강 적용
            # 1. 색상 조정
            image, boxes = self.color_jitter(image, boxes, image_width, image_height)

            # 2. 랜덤 스케일링
            image, boxes = self.random_scale(image, boxes, image_width, image_height)

            # 3. 랜덤 회전 (각도가 작을 때만)
            image, boxes = self.random_rotation(image, boxes, image_width, image_height)

            # 4. 랜덤 크롭
            image, boxes = self.random_crop(image, boxes, image_width, image_height)

            # 5. 믹스업
            # image, boxes = self.mixup(image, boxes, image_width, image_height)

        if self.transform:
            image = self.transform(image)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            x1, y1, x2, y2, class_label = box
            # 중심좌표, 너비, 높이 계산
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            box_width = x2 - x1
            box_height = y2 - y1

            # 정규화
            x = x_center / image_width
            y = y_center / image_height
            width = box_width / image_width
            height = box_height / image_height

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:

            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)

            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix