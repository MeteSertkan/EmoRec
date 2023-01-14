import torch
import torch.nn as nn
from models.utils import AdditiveAttention


class EmotionPooler(torch.nn.Module):
    def __init__(self, config):
        super(EmotionPooler, self).__init__()
        self.config = config
        self.additive_attention = AdditiveAttention(config.query_vector_dim, config.emotion_dim)

    def forward(self, title_emotions, abstract_emotions, category_emotions, subcategory_emotions):
        # stack all vectors
        all_vectors = torch.stack(
                [
                    title_emotions,
                    abstract_emotions,
                    category_emotions,
                    subcategory_emotions
                ],
                dim=2
            )
        # squash batch and history/candidates axis
        all_vectors = all_vectors.contiguous().view(-1, all_vectors.size(-2), all_vectors.size(-1))
        emotion_vector = self.additive_attention(all_vectors)
        # split back to batch x sample x encoding dim
        emotion_vector = emotion_vector.contiguous().view(-1, title_emotions.size(-2), emotion_vector.size(-1))
        return emotion_vector