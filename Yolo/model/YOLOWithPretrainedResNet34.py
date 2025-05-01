import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet34
# from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet34, ResNet34_Weights

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)


import torch.nn.init as init
import torch.nn.utils as nn_utils

# def initialize_weights(model):
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d):
#             init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             if m.bias is not None:
#                 init.constant_(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             init.constant_(m.weight, 1)
#             init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             init.normal_(m.weight, 0, 0.01)
#             init.constant_(m.bias, 0)
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # LeakyReLU에 맞게 negative_slope를 추가
            init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.01)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)
# 모델 가중치 초기화 방식 변경
# def initialize_weights(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.01)
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)




class YOLOHead(nn.Module):
    def __init__(self, in_channels=512, grid_size=7, num_classes=20, B=2):
        super(YOLOHead, self).__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.B = B

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.5),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size))  # 항상 7x7로 맞추기

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.grid_size * self.grid_size, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, self.grid_size * self.grid_size * (self.B * 5 + self.num_classes))
        )
        # self.fc =  nn.Sequential(
        #       nn.Conv2d(1024, 512, kernel_size=3, padding=1),  # [1, 512, 7, 7]
        #       # nn.MaxPool2d(2),                                 # [1, 512, 3, 3]
        #       nn.Flatten(),                                    # [1, 4608]
        #       nn.Linear(25088, 1024),
        #       nn.LeakyReLU(),
        #       nn.Linear(1024, 1470)
        #   )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_layers(x)
        # x = self.adaptive_pool(x)  # 강제로 7x7 크기로 만들어줌
        # print(x.shape)
        x = self.fc(x)
        return x.view(batch_size, self.grid_size, self.grid_size, self.B * 5 + self.num_classes)



class YOLOWithPretrainedResNet34(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()

        # Load pretrained resnet34
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)


        # Extract features only up to layer4
        self.backbone = create_feature_extractor(
            resnet,
            return_nodes={"layer4": "feat"}
        )

        # Freeze early layers
        self._freeze_layers(self.backbone)

        # YOLO Detection Head
        self.yolo_head = YOLOHead()
        initialize_weights(self.yolo_head)

    def _freeze_layers(self, backbone):
        # Freeze up to layer2 (layer1, layer2)
        for name, param in backbone.named_parameters():
            if "layer1" in name or "layer2" in name:
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)["feat"]  # shape: (batch, 512, 14, 14)
        # print(features.shape)
        output = self.yolo_head(features)

        return output
