import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class UniversalBackbone(nn.Module):
    """ViT, Swin, ConvNeXt, ResNet 등 다양한 백본을 지원하는 범용 백본"""

    def __init__(self, model_name, output_hidden_states=False, trainable=True):
        super().__init__()
        self.model_name = model_name

        # AutoModel로 자동 감지 및 로드
        try:
            self.config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(
                model_name,
                output_hidden_states=output_hidden_states
            )
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

        # 모델 타입 감지
        self.model_type = self._detect_model_type()
        print(f"Detected model type: {self.model_type}")

        # 백본의 훈련 가능 여부 설정
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

    def _detect_model_type(self):
        """모델 타입 자동 감지"""
        model_name_lower = self.model_name.lower()

        # CNN 계열
        if 'convnext' in model_name_lower:
            return 'convnext'
        elif 'resnet' in model_name_lower:
            return 'resnet'
        elif 'regnet' in model_name_lower:
            return 'regnet'
        elif 'efficientnet' in model_name_lower:
            return 'efficientnet'
        # Transformer 계열
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
            # 기본값은 ViT 스타일로 처리
            return 'vit'

    def get_hidden_size(self):
        """모델의 hidden size 반환"""
        # CNN 계열 모델들
        if self.model_type in ['convnext', 'resnet', 'regnet', 'efficientnet']:
            if hasattr(self.config, 'hidden_sizes') and self.config.hidden_sizes:
                return self.config.hidden_sizes[-1]  # 마지막 레이어의 채널 수
            elif hasattr(self.model, 'num_features'):
                return self.model.num_features
            elif hasattr(self.config, 'num_channels'):
                return self.config.num_channels
            else:
                # ConvNeXt 특정 설정
                if 'tiny' in self.model_name.lower():
                    return 768
                elif 'small' in self.model_name.lower():
                    return 768
                elif 'base' in self.model_name.lower():
                    return 1024
                elif 'large' in self.model_name.lower():
                    return 1536
                elif 'xlarge' in self.model_name.lower():
                    return 2048
                else:
                    return 768  # 기본값

        # Transformer 계열 모델들
        elif hasattr(self.config, 'hidden_size'):
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
                return 768  # 기본값

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)

        # CNN 계열 모델 처리
        if self.model_type in ['convnext', 'resnet', 'regnet', 'efficientnet']:
            # CNN 모델들은 보통 pooler_output을 제공
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                # Global Average Pooling 적용
                last_hidden = outputs.last_hidden_state
                # CNN의 경우: [batch, channels, height, width]
                if len(last_hidden.shape) == 4:
                    return torch.mean(last_hidden, dim=[2, 3])  # spatial dimensions을 평균
                else:
                    return torch.mean(last_hidden, dim=1)
            else:
                # 일부 모델은 직접 특징을 반환
                return outputs

        # Transformer 계열 모델 처리
        elif self.model_type == 'swin':
            # Swin: pooler_output 또는 last_hidden_state의 평균
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            else:
                # last_hidden_state의 평균 풀링 사용
                last_hidden = outputs.last_hidden_state
                return torch.mean(last_hidden, dim=1)

        elif self.model_type in ['vit', 'beit', 'deit', 'cvt']:
            # ViT 계열: [CLS] 토큰 사용
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            else:
                return outputs.last_hidden_state[:, 0]  # [CLS] 토큰

        else:
            # 기본: [CLS] 토큰 또는 첫 번째 토큰
            try:
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    return outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    if len(outputs.last_hidden_state.shape) == 3:
                        # Transformer: [batch, seq_len, hidden]
                        return outputs.last_hidden_state[:, 0]
                    elif len(outputs.last_hidden_state.shape) == 4:
                        # CNN: [batch, channels, height, width]
                        return torch.mean(outputs.last_hidden_state, dim=[2, 3])
                else:
                    return outputs
            except:
                # 최후의 수단: 평균 풀링
                if hasattr(outputs, 'last_hidden_state'):
                    return torch.mean(outputs.last_hidden_state, dim=1)
                else:
                    return outputs


# 기존 ViTBackbone과의 호환성을 위한 별칭
class ViTBackbone(UniversalBackbone):
    """기존 코드와의 호환성을 위한 ViTBackbone 별칭"""
    pass