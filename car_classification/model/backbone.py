import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class UniversalBackbone(nn.Module):
    """ViT, Swin, 기타 Vision Transformer 백본을 지원하는 범용 백본"""

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

        if 'swin' in model_name_lower:
            return 'swin'
        elif 'vit' in model_name_lower:
            return 'vit'
        elif 'beit' in model_name_lower:
            return 'beit'
        elif 'deit' in model_name_lower:
            return 'deit'
        else:
            # 기본값은 ViT 스타일로 처리
            return 'vit'

    def get_hidden_size(self):
        """모델의 hidden size 반환"""
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
                return 768  # 기본값

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)

        # 모델 타입에 따라 다른 출력 처리
        if self.model_type == 'swin':
            # Swin: pooler_output 또는 last_hidden_state의 평균
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            else:
                # last_hidden_state의 평균 풀링 사용
                last_hidden = outputs.last_hidden_state
                return torch.mean(last_hidden, dim=1)  # [batch_size, seq_len, hidden] -> [batch_size, hidden]

        elif self.model_type in ['vit', 'beit', 'deit']:
            # ViT 계열: [CLS] 토큰 사용
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            else:
                return outputs.last_hidden_state[:, 0]  # [CLS] 토큰

        else:
            # 기본: [CLS] 토큰 또는 첫 번째 토큰
            try:
                return outputs.last_hidden_state[:, 0]
            except:
                # 최후의 수단: 평균 풀링
                return torch.mean(outputs.last_hidden_state, dim=1)


# 기존 ViTBackbone과의 호환성을 위한 별칭
class ViTBackbone(UniversalBackbone):
    """기존 코드와의 호환성을 위한 ViTBackbone 별칭"""
    pass