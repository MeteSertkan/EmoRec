import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import AdditiveAttention
#from models.utils import PersonalizedAttentivePooling


class EmotionEncoder(torch.nn.Module):
    def __init__(self, config):
        super(EmotionEncoder, self).__init__()
        self.config = config
        self.mh_selfattention = nn.MultiheadAttention(
            config.emotion_dim,
            config.num_attention_heads_emotions)
        self.additive_attention = AdditiveAttention(config.query_vector_dim, config.emotion_dim)

    def forward(self, emotions):
        # multi-head self-attention
        emotions = emotions.permute(1, 0, 2)
        emotions, _ = self.mh_selfattention(
            emotions,
            emotions,
            emotions)
        # additive-attention
        emotions = emotions.permute(1, 0, 2)
        emotion_vector = self.additive_attention(emotions)
        return emotion_vector