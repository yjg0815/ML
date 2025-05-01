from dataset import custom_dataset
from dataset.voc_classes import VOC_CLASSES
from collections import defaultdict

def extract_label_types(dataset):
    label_set = set()
    for i in range(len(dataset)):
        _, label_matrix = dataset[i]
        C = 20  # 클래스 개수 (VOC 기준)
        class_one_hot = label_matrix[:, :, :C]
        # (S, S, C)에서 각 클래스가 한 번이라도 등장했는지 확인
        classes_in_sample = torch.any(class_one_hot > 0, dim=(0,1))
        for class_idx in range(C):
            if classes_in_sample[class_idx]:
                label_set.add(class_idx)
    return label_set

# train_labels = extract_label_types(train_dataset)
# test_labels = extract_label_types(test_dataset)
# val_labels = extract_label_types(val_dataset)

# train_label_names = [VOC_CLASSES[i] for i in sorted(train_labels)]
# test_label_names = [VOC_CLASSES[i] for i in sorted(test_labels)]
# val_label_names = [VOC_CLASSES[i] for i in sorted(val_labels)]

# print("Train 라벨 종류:", train_label_names)
# print("Test 라벨 종류:", test_label_names)
# print("Val 라벨 종류:", val_label_names)


def count_images_per_label(dataset):
    label_image_count = defaultdict(int)
    for i in range(len(dataset)):
        _, label_matrix = dataset[i]
        C = len(VOC_CLASSES)
        # (S, S, C + 5*B)에서 클래스 one-hot 부분만 추출
        class_one_hot = label_matrix[:, :, :C]
        # 각 클래스가 이미지 내에 하나라도 존재하는지 확인
        classes_in_image = (class_one_hot > 0).any(axis=(0, 1))
        for class_idx in range(C):
            if classes_in_image[class_idx]:
                label_image_count[class_idx] += 1
    return label_image_count

def print_label_image_distribution(label_image_count, class_names, total_images):
    print("라벨별 이미지 분포:")
    for class_idx in range(len(class_names)):
        count = label_image_count.get(class_idx, 0)
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        class_name = class_names[class_idx]
        print(f"{class_name}: {count}개 이미지 ({percentage:.2f}%)")


# label_image_count = count_images_per_label(train_dataset)
# print_label_image_distribution(label_image_count, VOC_CLASSES, len(train_dataset))