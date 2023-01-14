import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import AdditiveAttention
from models.utils import TimeDistributed


class TextEncoder(nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(TextEncoder, self).__init__()
        self.config = config
        self.word_embedding = pretrained_word_embedding
        self.mh_selfattention = nn.MultiheadAttention(
            config.word_embedding_dim,
            config.num_attention_heads)
        self.additive_attention = AdditiveAttention(config.query_vector_dim, config.word_embedding_dim)

    def forward(self, text):
        # word embedding
        text_vector = F.dropout(self.word_embedding(
            text),
            p=self.config.dropout_probability,
            training=self.training)
        # multi-head self attention
        text_vector = text_vector.permute(1, 0, 2)
        text_vector, _ = self.mh_selfattention(
            text_vector,
            text_vector,
            text_vector)
        # additive attention
        text_vector = text_vector.permute(1, 0, 2)
        text_vector = F.dropout(text_vector,
                                           p=self.config.dropout_probability,
                                           training=self.training)
        text_vector = self.additive_attention(text_vector)

        return text_vector


class SemanticEncoder(nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(SemanticEncoder, self).__init__()
        self.config = config
        word_embedding = nn.Embedding.from_pretrained(
            pretrained_word_embedding,
            freeze=config.freeze_word_embeddings,
            padding_idx=0)
        self.text_encoder = TimeDistributed(
            TextEncoder(
                config,
                word_embedding
                ),
            batch_first=True)
        self.additive_attention = AdditiveAttention(
            config.query_vector_dim,
            config.word_embedding_dim)

    def forward(self, title, abstract):
        # text encoding
        title_vector = self.text_encoder(title)
        # abstract encoding
        abstract_vector = self.text_encoder(abstract)
        # stack all vectors
        all_vectors = torch.stack(
                [
                    title_vector,
                    abstract_vector
                ],
                dim=2
            )
        # squash batch and sample (n-negative+1) axis
        all_vectors = all_vectors.contiguous().view(-1, all_vectors.size(-2), all_vectors.size(-1))
        semantic_vector = self.additive_attention(all_vectors)
        # split back to batch x sample x encoding dim
        semantic_vector = semantic_vector.contiguous().view(-1, title.size(-2), semantic_vector.size(-1))
        return semantic_vector
