import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from models.emorec.semantic_encoder import SemanticEncoder
from models.emorec.user_encoder import UserEncoder
from models.emorec.emotion_encoder import EmotionEncoder
from models.emorec.emotion_pooling import EmotionPooler
from models.metrics import NDCG, MRR, AUC


class EMOREC(pl.LightningModule):
    def __init__(self, config=None, pretrained_word_embedding=None):
        super(EMOREC, self).__init__()
        self.config = config
        self.semantic_encoder = SemanticEncoder(config, pretrained_word_embedding)
        self.user_encoder = UserEncoder(config)
        self.emotion_pooler = EmotionPooler(config)
        self.emotion_encoder = EmotionEncoder(config)
        # val metrics
        self.val_performance_metrics = MetricCollection({
            'val_auc': AUC(),
            'val_mrr': MRR(),
            'val_ndcg@5': NDCG(k=5),
            'val_ndcg@10': NDCG(k=10)
        })
        # test metrics
        self.test_performance_metrics = MetricCollection({
            'test_auc': AUC(),
            'test_mrr': MRR(),
            'test_ndcg@5': NDCG(k=5),
            'test_ndcg@10': NDCG(k=10)
        })

    def forward(self, batch):
        # encode candidate news
        candidate_news_emotion_vector = self.emotion_pooler(
            batch["c_title_" + self.config.emotion_taxonomy],
            batch["c_abstract_" + self.config.emotion_taxonomy],
            batch["c_category_" + self.config.emotion_taxonomy],
            batch["c_subcategory_" + self.config.emotion_taxonomy]
        )
        candidate_news_semantic_vector = self.semantic_encoder(
            batch["c_title"],
            batch["c_abstract"]
            )
        candidate_news_vector = torch.cat((
            candidate_news_semantic_vector, candidate_news_emotion_vector), 
            dim=-1)
        # encode clicked emotions
        clicked_news_emotion_vector = self.emotion_pooler(
            batch["h_title_" + self.config.emotion_taxonomy],
            batch["h_abstract_" + self.config.emotion_taxonomy],
            batch["h_category_" + self.config.emotion_taxonomy],
            batch["h_subcategory_" + self.config.emotion_taxonomy]
        )
        user_emotion_vector = self.emotion_encoder(
            clicked_news_emotion_vector
        )
        # encode clicked news 
        clicked_news_semantic_vector = self.semantic_encoder(
            batch["h_title"],
            batch["h_abstract"]
            )
        # encode user
        user_semantic_vector = self.user_encoder(clicked_news_semantic_vector)
        # concat user and emotion vector
        user_vector = torch.cat((
            user_semantic_vector, user_emotion_vector), 
            dim=-1)
        # compute scores for each candidate news
        clicks_score = torch.bmm(candidate_news_vector,
                                user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        clicks_score = F.softmax(clicks_score, dim=1)
        return clicks_score, user_emotion_vector

    def training_step(self, batch, batch_idx):
        y_pred, _ = self(batch)
        #y_pred = torch.sigmoid(y_pred)
        y = torch.zeros(len(y_pred), dtype=torch.long, device=self.device)
        loss = F.cross_entropy(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, _ = self(batch)
        y = batch["labels"]
        # compute metrics
        self.val_performance_metrics(y_pred, y)
        # log metric
        self.log_dict(self.val_performance_metrics, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        y_pred, _ = self(batch)
        y = batch["labels"]
        # compute metrics
        self.test_performance_metrics(y_pred, y)
        # log metric
        self.log_dict(self.test_performance_metrics, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.config.learning_rate)