import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
class ViTBackbone(nn.Module):
    """Vision Transformer 백본"""

    def __init__(self, model_name, output_hidden_states=False, trainable=True):
        super().__init__()
        self.model = ViTModel.from_pretrained(
            model_name,
            output_hidden_states=output_hidden_states
        )

        # 백본의 훈련 가능 여부 설정
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0]  # [CLS] 토큰 출력 반환