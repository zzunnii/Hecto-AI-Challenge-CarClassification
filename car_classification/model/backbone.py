import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import torchvision.models as models


class TorchVisionBackbone(nn.Module):
    """TorchVision 모델을 위한 백본 (ResNet, EfficientNet 등)"""

    def __init__(self, model_name, pretrained=True, trainable=True):
        super().__init__()
        self.model_name = model_name
        self.model_type = self._detect_model_type()

        # TorchVision 모델 로드
        self.model = self._load_torchvision_model(pretrained)

        # 특징 추출을 위해 분류 헤드 제거
        self._remove_classifier()

        # 백본의 훈련 가능 여부 설정
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

        print(f"Loaded TorchVision {self.model_type}: {model_name}")
        print(f"Hidden size: {self.get_hidden_size()}")

    def _detect_model_type(self):
        """모델 타입 자동 감지"""
        model_name_lower = self.model_name.lower()

        if 'resnet' in model_name_lower:
            return 'resnet'
        elif 'efficientnet' in model_name_lower:
            return 'efficientnet'
        elif 'mobilenet' in model_name_lower:
            return 'mobilenet'
        elif 'densenet' in model_name_lower:
            return 'densenet'
        elif 'resnext' in model_name_lower:
            return 'resnext'
        elif 'wide_resnet' in model_name_lower:
            return 'wide_resnet'
        else:
            return 'unknown'

    def _load_torchvision_model(self, pretrained):
        """TorchVision 모델 로드 (최신 weights 방식 사용)"""
        model_name_lower = self.model_name.lower()

        # ResNet 계열 - 최신 ImageNet 가중치 사용
        if model_name_lower == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            return models.resnet18(weights=weights)
        elif model_name_lower == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            return models.resnet34(weights=weights)
        elif model_name_lower == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None  # V2가 최신/최고성능
            return models.resnet50(weights=weights)
        elif model_name_lower == 'resnet101':
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            return models.resnet101(weights=weights)
        elif model_name_lower == 'resnet152':
            weights = models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
            return models.resnet152(weights=weights)
        elif model_name_lower == 'resnext50_32x4d':
            weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2 if pretrained else None
            return models.resnext50_32x4d(weights=weights)
        elif model_name_lower == 'resnext101_32x8d':
            weights = models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2 if pretrained else None
            return models.resnext101_32x8d(weights=weights)
        elif model_name_lower == 'wide_resnet50_2':
            weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V2 if pretrained else None
            return models.wide_resnet50_2(weights=weights)
        elif model_name_lower == 'wide_resnet101_2':
            weights = models.Wide_ResNet101_2_Weights.IMAGENET1K_V2 if pretrained else None
            return models.wide_resnet101_2(weights=weights)

        # EfficientNet 계열
        elif model_name_lower == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            return models.efficientnet_b0(weights=weights)
        elif model_name_lower == 'efficientnet_b1':
            weights = models.EfficientNet_B1_Weights.IMAGENET1K_V2 if pretrained else None
            return models.efficientnet_b1(weights=weights)
        elif model_name_lower == 'efficientnet_b2':
            weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
            return models.efficientnet_b2(weights=weights)
        elif model_name_lower == 'efficientnet_b3':
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            return models.efficientnet_b3(weights=weights)
        elif model_name_lower == 'efficientnet_b4':
            weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            return models.efficientnet_b4(weights=weights)
        elif model_name_lower == 'efficientnet_b5':
            weights = models.EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None
            return models.efficientnet_b5(weights=weights)
        elif model_name_lower == 'efficientnet_b6':
            weights = models.EfficientNet_B6_Weights.IMAGENET1K_V1 if pretrained else None
            return models.efficientnet_b6(weights=weights)
        elif model_name_lower == 'efficientnet_b7':
            weights = models.EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
            return models.efficientnet_b7(weights=weights)

        # MobileNet 계열
        elif model_name_lower == 'mobilenet_v2':
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
            return models.mobilenet_v2(weights=weights)
        elif model_name_lower == 'mobilenet_v3_large':
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
            return models.mobilenet_v3_large(weights=weights)
        elif model_name_lower == 'mobilenet_v3_small':
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            return models.mobilenet_v3_small(weights=weights)

        # DenseNet 계열
        elif model_name_lower == 'densenet121':
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            return models.densenet121(weights=weights)
        elif model_name_lower == 'densenet169':
            weights = models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
            return models.densenet169(weights=weights)
        elif model_name_lower == 'densenet201':
            weights = models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
            return models.densenet201(weights=weights)
        elif model_name_lower == 'densenet161':
            weights = models.DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None
            return models.densenet161(weights=weights)

        else:
            raise ValueError(f"Unsupported TorchVision model: {self.model_name}")

    def _remove_classifier(self):
        """분류 헤드 제거하여 특징 추출기로 변환"""
        if self.model_type in ['resnet', 'resnext', 'wide_resnet']:
            # ResNet 계열: fc layer 제거
            self.model.fc = nn.Identity()

        elif self.model_type == 'efficientnet':
            # EfficientNet: classifier 제거
            self.model.classifier = nn.Identity()

        elif self.model_type == 'mobilenet':
            # MobileNet: classifier 제거
            if hasattr(self.model, 'classifier'):
                self.model.classifier = nn.Identity()
            elif hasattr(self.model, 'head'):
                self.model.head = nn.Identity()

        elif self.model_type == 'densenet':
            # DenseNet: classifier 제거
            self.model.classifier = nn.Identity()

    def get_hidden_size(self):
        """모델의 hidden size 반환"""
        if self.model_type in ['resnet', 'resnext', 'wide_resnet']:
            # ResNet 계열: inplanes 또는 직접 계산
            if hasattr(self.model, 'fc') and hasattr(self.model.fc, 'in_features'):
                return self.model.fc.in_features

            # fc가 Identity로 바뀐 경우, layer4의 출력 채널 수 계산
            if hasattr(self.model, 'layer4'):
                last_block = self.model.layer4[-1]
                if hasattr(last_block, 'conv3'):  # Bottleneck
                    return last_block.conv3.out_channels
                elif hasattr(last_block, 'conv2'):  # BasicBlock
                    return last_block.conv2.out_channels

            # 모델명으로 추정
            if '18' in self.model_name or '34' in self.model_name:
                return 512
            else:  # 50, 101, 152 등
                return 2048

        elif self.model_type == 'efficientnet':
            # EfficientNet: 원래 classifier의 입력 크기
            if hasattr(self.model, '_avg_pooling') and hasattr(self.model, '_dropout'):
                # 임시로 입력을 통과시켜 크기 확인
                with torch.no_grad():
                    # 각 EfficientNet 버전별 특징 크기 (미리 정의)
                    efficientnet_sizes = {
                        'efficientnet_b0': 1280,
                        'efficientnet_b1': 1280,
                        'efficientnet_b2': 1408,
                        'efficientnet_b3': 1536,
                        'efficientnet_b4': 1792,
                        'efficientnet_b5': 2048,
                        'efficientnet_b6': 2304,
                        'efficientnet_b7': 2560,
                    }
                    return efficientnet_sizes.get(self.model_name.lower(), 1280)

        elif self.model_type == 'mobilenet':
            # MobileNet: last_channel
            if hasattr(self.model, 'last_channel'):
                return self.model.last_channel
            elif 'v3_large' in self.model_name.lower():
                return 960
            elif 'v3_small' in self.model_name.lower():
                return 576
            else:  # v2
                return 1280

        elif self.model_type == 'densenet':
            # DenseNet: classifier의 입력 크기
            densenet_sizes = {
                'densenet121': 1024,
                'densenet169': 1664,
                'densenet201': 1920,
                'densenet161': 2208,
            }
            return densenet_sizes.get(self.model_name.lower(), 1024)

        # 기본값
        return 2048

    def forward(self, pixel_values):
        """순전파"""
        # TorchVision 모델은 3차원 입력을 기대 (batch, channel, height, width)
        if len(pixel_values.shape) == 4:
            x = pixel_values
        else:
            # transformers 형태에서 변환
            x = pixel_values.squeeze(1) if pixel_values.shape[1] == 1 else pixel_values

        features = self.model(x)

        # 출력이 4차원인 경우 (CNN의 feature map) Global Average Pooling
        if len(features.shape) == 4:
            features = torch.mean(features, dim=[2, 3])

        return features


class UniversalBackbone(nn.Module):
    """ViT, Swin, ConvNeXt, ResNet 등 다양한 백본을 지원하는 범용 백본"""

    def __init__(self, model_name, output_hidden_states=False, trainable=True):
        super().__init__()
        self.model_name = model_name

        # TorchVision 모델인지 확인
        if self._is_torchvision_model():
            self.backbone = TorchVisionBackbone(model_name, pretrained=True, trainable=trainable)
            self.model_type = self.backbone.model_type
        else:
            # Transformers 모델 로드
            try:
                self.config = AutoConfig.from_pretrained(model_name)
                self.backbone = AutoModel.from_pretrained(
                    model_name,
                    output_hidden_states=output_hidden_states
                )
                self.model_type = self._detect_transformers_model_type()

                # 백본의 훈련 가능 여부 설정
                if not trainable:
                    for param in self.backbone.parameters():
                        param.requires_grad = False

            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                raise

        print(f"Loaded backbone: {self.model_type} - {model_name}")

    def _is_torchvision_model(self):
        """TorchVision 모델인지 확인"""
        torchvision_models = [
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'resnext50_32x4d', 'resnext101_32x8d',
            'wide_resnet50_2', 'wide_resnet101_2',
            'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
            'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
            'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
            'densenet121', 'densenet169', 'densenet201', 'densenet161'
        ]
        return self.model_name.lower() in torchvision_models

    def _detect_transformers_model_type(self):
        """Transformers 모델 타입 자동 감지"""
        model_name_lower = self.model_name.lower()

        if 'convnext' in model_name_lower:
            return 'convnext'
        elif 'swin' in model_name_lower:
            return 'swin'
        elif 'vit' in model_name_lower:
            return 'vit'
        elif 'beit' in model_name_lower:
            return 'beit'
        elif 'deit' in model_name_lower:
            return 'deit'
        elif 'cvt' in model_name_lower:
            return 'cvt'
        else:
            return 'vit'  # 기본값

    def get_hidden_size(self):
        """모델의 hidden size 반환"""
        if hasattr(self, 'backbone') and hasattr(self.backbone, 'get_hidden_size'):
            # TorchVision 백본
            return self.backbone.get_hidden_size()
        else:
            # Transformers 백본
            if hasattr(self.config, 'hidden_size'):
                return self.config.hidden_size
            elif hasattr(self.config, 'd_model'):
                return self.config.d_model
            elif hasattr(self.config, 'embed_dim'):
                return self.config.embed_dim
            else:
                # 모델명으로 추정
                if 'large' in self.model_name.lower():
                    return 1024
                elif 'huge' in self.model_name.lower():
                    return 1280
                else:
                    return 768

    def forward(self, pixel_values):
        if hasattr(self, 'backbone') and isinstance(self.backbone, TorchVisionBackbone):
            # TorchVision 백본 사용
            return self.backbone(pixel_values)
        else:
            # Transformers 백본 사용
            outputs = self.backbone(pixel_values=pixel_values)

            # Transformers 모델별 처리
            if self.model_type == 'swin':
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    return outputs.pooler_output
                else:
                    return torch.mean(outputs.last_hidden_state, dim=1)

            elif self.model_type in ['vit', 'beit', 'deit', 'cvt']:
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    return outputs.pooler_output
                else:
                    return outputs.last_hidden_state[:, 0]  # [CLS] 토큰

            elif self.model_type == 'convnext':
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    return outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    last_hidden = outputs.last_hidden_state
                    if len(last_hidden.shape) == 4:
                        return torch.mean(last_hidden, dim=[2, 3])
                    else:
                        return torch.mean(last_hidden, dim=1)

            else:
                # 기본 처리
                try:
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        return outputs.pooler_output
                    elif hasattr(outputs, 'last_hidden_state'):
                        if len(outputs.last_hidden_state.shape) == 3:
                            return outputs.last_hidden_state[:, 0]
                        elif len(outputs.last_hidden_state.shape) == 4:
                            return torch.mean(outputs.last_hidden_state, dim=[2, 3])
                    else:
                        return outputs
                except:
                    if hasattr(outputs, 'last_hidden_state'):
                        return torch.mean(outputs.last_hidden_state, dim=1)
                    else:
                        return outputs


# 기존 ViTBackbone과의 호환성을 위한 별칭
class ViTBackbone(UniversalBackbone):
    """기존 코드와의 호환성을 위한 ViTBackbone 별칭"""
    pass