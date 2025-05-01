from torchvision import datasets
from torch.utils.data import ConcatDataset

voc_root = "/content/path_to_voc"

# VOC2007 train+val
voc2007 = datasets.VOCDetection(
    root=voc_root,
    year="2007",
    image_set="trainval",
    download=True,
    # transform=transform_train
)

# VOC2012 train
voc2012 = datasets.VOCDetection(
    root=voc_root,
    year="2012",
    image_set="train",
    download=True,
    # transform=transform_train
)


# VOC2012 train
voc2012_val = datasets.VOCDetection(
    root=voc_root,
    year="2012",
    image_set="val",
    download=True,
    # transform=transform_train
)

# 합친 학습 데이터셋
train_dataset = ConcatDataset([voc2007, voc2012, voc2012_val])

# VOC2012 val은 그대로 테스트셋으로 사용
test_dataset = datasets.VOCDetection(
    root=voc_root,
    year="2007",
    image_set="test",
    download=True,
    # transform=transform_test
)
