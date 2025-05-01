import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from custom_dataset import CustomDataset



# 데이터 변환 정의
transform_train = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# VOC 데이터셋 경로
voc_root = "/content/path_to_voc"

# VOC2007 trainval 데이터셋 (CustomDataset 사용)
voc2007_trainval = CustomDataset(
    root=voc_root,
    year="2007",  # 2007년 데이터 사용
    img_set="trainval",
    transform=transform_train,
    augment=True
)

# VOC2012 train 데이터셋 (CustomDataset 사용)
voc2012_train = CustomDataset(
    root=voc_root,
    year="2012",  # 2012년 데이터 사용
    img_set="train",
    transform=transform_train,
    augment=True
)

# VOC2012 val 데이터셋 (CustomDataset 사용)
voc2012_val = CustomDataset(
    root=voc_root,
    year="2012",  # 2012년 데이터 사용
    img_set="val",
    transform=transform_train,
    augment=True
)

# 두 데이터셋 합치기
train_dataset = ConcatDataset([voc2007_trainval, voc2012_train, voc2012_val])
print(f"통합 훈련 데이터셋: {len(train_dataset)}개 이미지")

# 테스트 데이터셋은 VOC2007 test 사용
voc2007_valtest = CustomDataset(
    root=voc_root,
    year="2007",
    img_set="test",
    transform=transform_test,
    augment=False
)

from torch.utils.data import random_split
from torch.utils.data import DataLoader

# Train/Validation 데이터 분리 (80% train, 20% validation)
test_size = int(0.5 * len(voc2007_valtest))
val_size = len(voc2007_valtest) - test_size

test_dataset, val_dataset = random_split(voc2007_valtest, [test_size, val_size])

# def collate_fn(batch):
#     images = [item[0] for item in batch]
#     targets = [item[1] for item in batch]
#     return torch.stack(images), targets

print(len(train_dataset))
print(len(test_dataset))
print(len(val_dataset))


# DataLoader 설정
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
