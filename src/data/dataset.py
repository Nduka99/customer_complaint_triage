"""
CFPBDataset — PyTorch Dataset for CFPB complaint classification.

Handles on-the-fly tokenisation and multi-task label packaging.
Used by NB05-08 fine-tuning notebooks.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd


class CFPBDataset(Dataset):
    """
    PyTorch Dataset for CFPB consumer complaints.

    Tokenises on-the-fly to support swapping tokenisers
    between experiments without re-processing data.

    Args:
        df: DataFrame with 'narrative', 'product_id', 'issue_id', 'response_id' columns
        tokeniser: HuggingFace tokeniser instance
        max_length: Maximum token sequence length (default 512)
        metadata_cols: List of metadata feature column names (optional)
    """
    def __init__(self, df, tokeniser, max_length=512, metadata_cols=None):
        self.texts = df['narrative'].tolist()
        self.product_ids = torch.tensor(df['product_id'].values, dtype=torch.long)
        self.issue_ids = torch.tensor(df['issue_id'].values, dtype=torch.long)
        self.response_ids = torch.tensor(df['response_id'].values, dtype=torch.long)
        self.rewards = torch.tensor(df['reward'].values, dtype=torch.float32)
        self.tokeniser = tokeniser
        self.max_length = max_length

        if metadata_cols and all(c in df.columns for c in metadata_cols):
            self.metadata = torch.tensor(
                df[metadata_cols].values, dtype=torch.float32
            )
        else:
            self.metadata = None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokeniser(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label_product': self.product_ids[idx],
            'label_issue': self.issue_ids[idx],
            'label_response': self.response_ids[idx],
            'reward': self.rewards[idx],
        }

        # Include token_type_ids if the tokeniser produces them (BERT-family)
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].squeeze(0)

        if self.metadata is not None:
            item['metadata'] = self.metadata[idx]

        return item


def load_split(split_name, processed_dir='data/processed'):
    """Load a preprocessed split by name ('train', 'val', 'test')."""
    path = f'{processed_dir}/{split_name}.parquet'
    return pd.read_parquet(path)
