from re import M
from torch.utils.data import Dataset
import torch
from tqdm import tqdm


class BaseDataset(Dataset):
    def __init__(self, behavior_path, news_path, config):
        super(BaseDataset, self).__init__()
        self.config = config
        self.behaviors_parsed = []
        news_parsed = {}
        #
        # loading and preparing news collection
        #       
        with open(news_path, 'r') as file:
            news_collection = file.readlines()
            for news in tqdm(news_collection):
                nid, cat, subcat, title, abstract, vader_sent, bert_sent, \
                title_emotions, abstract_emotions, title_abstract_emotions, \
                title_emotions_grouped, abstract_emotions_grouped, title_abstract_emotions_grouped, \
                title_emotions_ekman, abstract_emotions_ekman, title_abstract_emotions_ekman, \
                category_emotions, subcategory_emotions, \
                category_emotions_grouped, subcategory_emotions_grouped, \
                category_emotions_ekman, subcategory_emotions_ekman = news.split("\t")
                news_parsed[nid] = {
                    # 'nid': nid,
                    'category': torch.tensor(int(cat)),
                    'subcategory': torch.tensor((int(subcat))),
                    'title': torch.tensor([int(i) for i in title.split(" ")]), 
                    'abstract': torch.tensor([int(i) for i in abstract.split(" ")]),
                    'vader_sentiment': torch.tensor(float(vader_sent)),
                    'bert_sentiment': torch.tensor(float(bert_sent)),
                    'title_emotions': torch.tensor([float(i) for i in title_emotions.split(" ")]),
                    'title_emotions_grouped': torch.tensor([float(i) for i in title_emotions_grouped.split(" ")]),
                    'title_emotions_ekman': torch.tensor([float(i) for i in title_emotions_ekman.split(" ")]),
                    'abstract_emotions': torch.tensor([float(i) for i in abstract_emotions.split(" ")]),
                    'abstract_emotions_grouped': torch.tensor([float(i) for i in abstract_emotions_grouped.split(" ")]),
                    'abstract_emotions_ekman': torch.tensor([float(i) for i in abstract_emotions_ekman.split(" ")]),
                    'title_abstract_emotions': torch.tensor([float(i) for i in title_abstract_emotions.split(" ")]),
                    'title_abstract_emotions_grouped': torch.tensor([float(i) for i in title_abstract_emotions_grouped.split(" ")]),
                    'title_abstract_emotions_ekman': torch.tensor([float(i) for i in title_abstract_emotions_ekman.split(" ")]),
                    'category_emotions': torch.tensor([float(i) for i in category_emotions.split(" ")]),
                    'category_emotions_grouped': torch.tensor([float(i) for i in category_emotions_grouped.split(" ")]),
                    'category_emotions_ekman': torch.tensor([float(i) for i in category_emotions_ekman.split(" ")]),
                    'subcategory_emotions': torch.tensor([float(i) for i in subcategory_emotions.split(" ")]),
                    'subcategory_emotions_grouped': torch.tensor([float(i) for i in subcategory_emotions_grouped.split(" ")]),
                    'subcategory_emotions_ekman': torch.tensor([float(i) for i in subcategory_emotions_ekman.split(" ")])
                    }
        #
        # loading and preparing behaviors
        #
        # padding for news
        padding = {
            #'nid': torch.tensor(0),
            'category': torch.tensor(0),
            'subcategory': torch.tensor(0),
            'title': torch.tensor([0] * config.num_words_title),
            'abstract': torch.tensor([0] * config.num_words_abstract),
            'vader_sentiment': torch.tensor(0.0), 
            'bert_sentiment': torch.tensor(0.0),
            'title_emotions': torch.tensor([0.0] * config.num_emotions),
            'title_emotions_grouped': torch.tensor([0.0] * config.num_emotions_grouped),
            'title_emotions_ekman': torch.tensor([0.0] * config.num_emotions_ekman),
            'abstract_emotions': torch.tensor([0.0] * config.num_emotions),
            'abstract_emotions_grouped': torch.tensor([0.0] * config.num_emotions_grouped),
            'abstract_emotions_ekman': torch.tensor([0.0] * config.num_emotions_ekman),
            'title_abstract_emotions': torch.tensor([0.0] * config.num_emotions),
            'title_abstract_emotions_grouped': torch.tensor([0.0] * config.num_emotions_grouped),
            'title_abstract_emotions_ekman': torch.tensor([0.0] * config.num_emotions_ekman),
            'category_emotions': torch.tensor([0.0] * config.num_emotions),
            'category_emotions_grouped': torch.tensor([0.0] * config.num_emotions_grouped),
            'category_emotions_ekman': torch.tensor([0.0] * config.num_emotions_ekman),
            'subcategory_emotions': torch.tensor([0.0] * config.num_emotions),
            'subcategory_emotions_grouped': torch.tensor([0.0] * config.num_emotions_grouped),
            'subcategory_emotions_ekman': torch.tensor([0.0] * config.num_emotions_ekman)
        }

        with open(behavior_path, 'r') as file:
            behaviors = file.readlines()
            for behavior in tqdm(behaviors):
                uid, hist, candidates, clicks = behavior.split("\t")
                user = torch.tensor(int(uid))
                if hist:
                    history = [news_parsed[i] for i in hist.split(" ")]
                    if len(history) > config.max_history: 
                        history = history[:config.max_history]
                    else:
                        repeat = config.max_history - len(history)
                        history = [padding]*repeat + history
                else:
                    history = [padding]*config.max_history
                candidates = [news_parsed[i] for i in candidates.split(" ")]
                labels = torch.tensor([int(i) for i in clicks.split(" ")])
                self.behaviors_parsed.append(
                    {
                        'user': user,
                        # 'h_nids': [h['nid'] for h in history],
                        'h_title': torch.stack([h['title'] for h in history]),
                        'h_abstract': torch.stack([h['abstract'] for h in history]),
                        'h_category': torch.stack([h['category'] for h in history]),
                        'h_subcategory': torch.stack([h['subcategory'] for h in history]),
                        'h_vader_sentiment': torch.stack([h['vader_sentiment'] for h in history]),
                        'h_bert_sentiment': torch.stack([h['bert_sentiment'] for h in history]),
                        'h_title_emotions': torch.stack([h['title_emotions'] for h in history]),
                        'h_title_emotions_grouped': torch.stack([h['title_emotions_grouped'] for h in history]),
                        'h_title_emotions_ekman': torch.stack([h['title_emotions_ekman'] for h in history]),
                        'h_abstract_emotions': torch.stack([h['abstract_emotions'] for h in history]),
                        'h_abstract_emotions_grouped': torch.stack([h['abstract_emotions_grouped'] for h in history]),
                        'h_abstract_emotions_ekman': torch.stack([h['abstract_emotions_ekman'] for h in history]),
                        'h_title_abstract_emotions': torch.stack([h['title_abstract_emotions'] for h in history]),
                        'h_title_abstract_emotions_grouped': torch.stack([h['title_abstract_emotions_grouped'] for h in history]),
                        'h_title_abstract_emotions_ekman': torch.stack([h['title_abstract_emotions_ekman'] for h in history]),
                        'h_category_emotions': torch.stack([h['category_emotions'] for h in history]),
                        'h_category_emotions_grouped': torch.stack([h['category_emotions_grouped'] for h in history]),
                        'h_category_emotions_ekman': torch.stack([h['category_emotions_ekman'] for h in history]),
                        'h_subcategory_emotions': torch.stack([h['subcategory_emotions'] for h in history]),
                        'h_subcategory_emotions_grouped': torch.stack([h['subcategory_emotions_grouped'] for h in history]),
                        'h_subcategory_emotions_ekman': torch.stack([h['subcategory_emotions_ekman'] for h in history]),
                        'history_length': torch.tensor(len(history)),
                        # 'c_nids': [c['nid'] for c in candidates],
                        'c_title': torch.stack([c['title'] for c in candidates]),
                        'c_abstract': torch.stack([c['abstract'] for c in candidates]),
                        'c_category': torch.stack([c['category'] for c in candidates]),
                        'c_subcategory': torch.stack([c['subcategory'] for c in candidates]),
                        'c_vader_sentiment': torch.stack([c['vader_sentiment'] for c in candidates]),
                        'c_bert_sentiment': torch.stack([c['bert_sentiment'] for c in candidates]),
                        'c_title_emotions': torch.stack([c['title_emotions'] for c in candidates]),
                        'c_title_emotions_grouped': torch.stack([c['title_emotions_grouped'] for c in candidates]),
                        'c_title_emotions_ekman': torch.stack([c['title_emotions_ekman'] for c in candidates]),
                        'c_abstract_emotions': torch.stack([c['abstract_emotions'] for c in candidates]),
                        'c_abstract_emotions_grouped': torch.stack([c['abstract_emotions_grouped'] for c in candidates]),
                        'c_abstract_emotions_ekman': torch.stack([c['abstract_emotions_ekman'] for c in candidates]),
                        'c_title_abstract_emotions': torch.stack([c['title_abstract_emotions'] for c in candidates]),
                        'c_title_abstract_emotions_grouped': torch.stack([c['title_abstract_emotions_grouped'] for c in candidates]),
                        'c_title_abstract_emotions_ekman': torch.stack([c['title_abstract_emotions_ekman'] for c in candidates]),
                        'c_category_emotions': torch.stack([c['category_emotions'] for c in candidates]),
                        'c_category_emotions_grouped': torch.stack([c['category_emotions_grouped'] for c in candidates]),
                        'c_category_emotions_ekman': torch.stack([c['category_emotions_ekman'] for c in candidates]),
                        'c_subcategory_emotions': torch.stack([c['subcategory_emotions'] for c in candidates]),
                        'c_subcategory_emotions_grouped': torch.stack([c['subcategory_emotions_grouped'] for c in candidates]),
                        'c_subcategory_emotions_ekman': torch.stack([c['subcategory_emotions_ekman'] for c in candidates]),
                        'labels': labels
                    }
                )

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        return self.behaviors_parsed[idx]
