import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random

class CustomColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, image, boxes, image_width, image_height):
        if random.random() < self.p:
            # 박스 좌표는 변경되지 않음
            if self.brightness > 0:
                brightness_factor = random.uniform(1-self.brightness, 1+self.brightness)
                image = ImageEnhance.Brightness(image).enhance(brightness_factor)

            if self.contrast > 0:
                contrast_factor = random.uniform(1-self.contrast, 1+self.contrast)
                image = ImageEnhance.Contrast(image).enhance(contrast_factor)

            if self.saturation > 0:
                saturation_factor = random.uniform(1-self.saturation, 1+self.saturation)
                image = ImageEnhance.Color(image).enhance(saturation_factor)

            if self.hue > 0:
                hue_factor = random.uniform(-self.hue, self.hue)
                image = TF.adjust_hue(image, hue_factor)

        return image, boxes


class CustomRandomScale:
    def __init__(self, scale_range=(0.8, 1.2), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, image, boxes, image_width, image_height):
        if random.random() < self.p:
            scale_factor = random.uniform(*self.scale_range)

            # 새 이미지 크기 계산
            new_width = int(image_width * scale_factor)
            new_height = int(image_height * scale_factor)

            # 이미지 리사이즈
            image = image.resize((new_width, new_height), Image.BILINEAR)

            # 원래 크기로 다시 조정 (패딩 또는 크롭)
            if scale_factor > 1.0:  # 확대된 경우 중앙 크롭
                left = (new_width - image_width) // 2
                top = (new_height - image_height) // 2
                image = image.crop((left, top, left + image_width, top + image_height))

                # 박스 좌표 조정
                adjusted_boxes = []
                for box in boxes:
                    xmin, ymin, xmax, ymax, cls_idx = box
                    # 확대 후 크롭에 맞게 좌표 조정
                    new_xmin = max(0, (xmin * scale_factor) - left)
                    new_ymin = max(0, (ymin * scale_factor) - top)
                    new_xmax = min(image_width, (xmax * scale_factor) - left)
                    new_ymax = min(image_height, (ymax * scale_factor) - top)

                    # 유효한 박스인지 확인
                    if new_xmax > new_xmin and new_ymax > new_ymin:
                        adjusted_boxes.append((new_xmin, new_ymin, new_xmax, new_ymax, cls_idx))
                boxes = adjusted_boxes

            else:  # 축소된 경우 패딩
                padded_image = Image.new('RGB', (image_width, image_height), (0, 0, 0))
                paste_x = (image_width - new_width) // 2
                paste_y = (image_height - new_height) // 2
                padded_image.paste(image, (paste_x, paste_y))
                image = padded_image

                # 박스 좌표 조정
                adjusted_boxes = []
                for box in boxes:
                    xmin, ymin, xmax, ymax, cls_idx = box
                    # 축소 후 패딩에 맞게 좌표 조정
                    new_xmin = (xmin * scale_factor) + paste_x
                    new_ymin = (ymin * scale_factor) + paste_y
                    new_xmax = (xmax * scale_factor) + paste_x
                    new_ymax = (ymax * scale_factor) + paste_y
                    adjusted_boxes.append((new_xmin, new_ymin, new_xmax, new_ymax, cls_idx))
                boxes = adjusted_boxes

        return image, boxes


class CustomRandomRotation:
    def __init__(self, degrees=10, p=0.3):
        self.degrees = degrees
        self.p = p

    def __call__(self, image, boxes, image_width, image_height):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)

            # 이미지 회전
            image = image.rotate(angle, expand=False, resample=Image.BILINEAR)

            # 이미지 중심점
            cx, cy = image_width / 2, image_height / 2

            adjusted_boxes = []
            for box in boxes:
                xmin, ymin, xmax, ymax, cls_idx = box

                # 박스의 네 모서리 좌표
                corners = [
                    (xmin, ymin), (xmax, ymin),
                    (xmin, ymax), (xmax, ymax)
                ]

                # 회전 변환 행렬 생성
                angle_rad = -angle * np.pi / 180  # 시계 반대 방향이 양수
                cos_theta, sin_theta = np.cos(angle_rad), np.sin(angle_rad)

                # 각 모서리를 회전
                rotated_corners = []
                for x, y in corners:
                    # 중심을 기준으로 좌표 이동
                    x -= cx
                    y -= cy

                    # 회전 적용
                    new_x = cx + (x * cos_theta - y * sin_theta)
                    new_y = cy + (x * sin_theta + y * cos_theta)

                    rotated_corners.append((new_x, new_y))

                # 새로운 바운딩 박스 계산 (회전된 모서리를 포함하는 최소 직사각형)
                min_x = max(0, min([c[0] for c in rotated_corners]))
                min_y = max(0, min([c[1] for c in rotated_corners]))
                max_x = min(image_width, max([c[0] for c in rotated_corners]))
                max_y = min(image_height, max([c[1] for c in rotated_corners]))

                # 유효한 박스인지 확인
                if max_x > min_x and max_y > min_y:
                    adjusted_boxes.append((min_x, min_y, max_x, max_y, cls_idx))

            boxes = adjusted_boxes

        return image, boxes


class CustomRandomCrop:
    def __init__(self, min_crop_ratio=0.6, max_crop_ratio=1.0, p=0.5):
        self.min_crop_ratio = min_crop_ratio
        self.max_crop_ratio = max_crop_ratio
        self.p = p

    def __call__(self, image, boxes, image_width, image_height):
        if random.random() < self.p:
            # 크롭 크기 결정
            crop_ratio = random.uniform(self.min_crop_ratio, self.max_crop_ratio)
            crop_width = int(image_width * crop_ratio)
            crop_height = int(image_height * crop_ratio)

            # 크롭 위치 결정 (박스가 잘리지 않도록 노력)
            if boxes and random.random() < 0.7:  # 70% 확률로 객체 중심 크롭
                # 모든 박스의 중심점 계산
                box_centers = []
                for box in boxes:
                    xmin, ymin, xmax, ymax, _ = box
                    center_x = (xmin + xmax) / 2
                    center_y = (ymin + ymax) / 2
                    box_centers.append((center_x, center_y))

                # 무작위로 중심점 선택
                center_x, center_y = random.choice(box_centers)

                # 크롭 좌표 계산
                left = max(0, int(center_x - crop_width / 2))
                top = max(0, int(center_y - crop_height / 2))

                # 이미지 경계 확인
                if left + crop_width > image_width:
                    left = image_width - crop_width
                if top + crop_height > image_height:
                    top = image_height - crop_height
            else:
                # 무작위 위치에서 크롭
                left = random.randint(0, image_width - crop_width)
                top = random.randint(0, image_height - crop_height)

            # 이미지 크롭
            image = image.crop((left, top, left + crop_width, top + crop_height))

            # 박스 좌표 조정
            adjusted_boxes = []
            for box in boxes:
                xmin, ymin, xmax, ymax, cls_idx = box

                # 크롭 영역과의 교차 계산
                new_xmin = max(0, xmin - left)
                new_ymin = max(0, ymin - top)
                new_xmax = min(crop_width, xmax - left)
                new_ymax = min(crop_height, ymax - top)

                # 유효한 박스인지 확인 (일정 비율 이상 포함되는지)
                if new_xmax > new_xmin and new_ymax > new_ymin:
                    # 원래 박스와 겹치는 비율 계산
                    original_area = (xmax - xmin) * (ymax - ymin)
                    new_area = (new_xmax - new_xmin) * (new_ymax - new_ymin)
                    overlap_ratio = new_area / original_area

                    if overlap_ratio > 0.5:  # 50% 이상 포함되는 박스만 유지
                        adjusted_boxes.append((new_xmin, new_ymin, new_xmax, new_ymax, cls_idx))

            # 리사이즈: 크롭된 이미지를 원본 크기로 복원
            image = image.resize((image_width, image_height), Image.BILINEAR)

            # 박스 좌표도 원본 크기에 맞게 스케일 조정
            scale_x = image_width / crop_width
            scale_y = image_height / crop_height

            final_boxes = []
            for box in adjusted_boxes:
                xmin, ymin, xmax, ymax, cls_idx = box
                final_boxes.append((
                    xmin * scale_x,
                    ymin * scale_y,
                    xmax * scale_x,
                    ymax * scale_y,
                    cls_idx
                ))

            boxes = final_boxes

        return image, boxes


class CustomMixup:
    def __init__(self, dataset, alpha=1.0, p=0.3):
        self.dataset = dataset  # 전체 데이터셋 참조
        self.alpha = alpha
        self.p = p

    def __call__(self, image, boxes, image_width, image_height):
        if random.random() < self.p:
            # 다른 이미지 무작위 선택
            mix_idx = random.randint(0, len(self.dataset) - 1)
            while mix_idx == self.current_idx:  # 현재 이미지와 다른 이미지 선택
                mix_idx = random.randint(0, len(self.dataset) - 1)

            # 믹스업 이미지 로드
            mix_image_path = self.dataset.images[mix_idx]
            mix_annotation_path = self.dataset.annotations[mix_idx]

            mix_image = Image.open(mix_image_path).convert("RGB")

            # 믹스업 이미지의 박스 정보 가져오기
            mix_tree = ET.parse(mix_annotation_path)
            mix_root = mix_tree.getroot()
            mix_boxes = self.dataset.parse_voc_xml(mix_root)

            # 혼합 비율(람다) 샘플링
            lam = np.random.beta(self.alpha, self.alpha)

            # 이미지 혼합
            image = Image.blend(image, mix_image, lam)

            # 박스 정보 합치기 (가중치 적용)
            combined_boxes = boxes + [(xmin, ymin, xmax, ymax, cls_idx) for xmin, ymin, xmax, ymax, cls_idx in mix_boxes
                                     if random.random() < lam]  # 람다 값에 비례하여 박스 선택

            boxes = combined_boxes

        return image, boxes

class CustomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, boxes, image_width):
        if random.random() < self.p:
            image = TF.hflip(image)
            flipped_boxes = []
            for box in boxes:
                xmin, ymin, xmax, ymax, cls_idx = box
                new_xmin = image_width - xmax
                new_xmax = image_width - xmin
                flipped_boxes.append((new_xmin, ymin, new_xmax, ymax, cls_idx))
            return image, flipped_boxes
        else:
            return image, boxes
