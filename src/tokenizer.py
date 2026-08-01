import torch
import torch.nn as nn

class TabularFeatureTokenizer(nn.Module):
    def __init__(self, num_numerical, cat_cardinalities, d_model):
        """
        Args:
            num_numerical (int): 수치형 변수의 개수
            cat_cardinalities (list of int): 각 범주형 변수의 고유 카테고리 개수 (Label Encoding 범위)
            d_model (int): 출력 토큰의 차원 수
        """
        super().__init__()
        self.num_numerical = num_numerical
        self.has_categorical = len(cat_cardinalities) > 0

        # 1. 수치형 토크나이저 가중치 (기존 코드 유지)
        if num_numerical > 0:
            self.num_weight = nn.Parameter(torch.randn(num_numerical, d_model))
            self.num_bias = nn.Parameter(torch.randn(num_numerical, d_model))

        # 2. 범주형 토크나이저 추가 (각 컬럼마다 독립된 nn.Embedding 생성)
        if self.has_categorical:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(num_embeddings=cardinality, embedding_dim=d_model, padding_idx=0)
                for cardinality in cat_cardinalities
            ])

    def forward(self, x_num=None, x_cat=None):
        tokens = []

        # 수치형 변수 토큰화: (batch_size, num_numerical, d_model)
        if x_num is not None and self.num_numerical > 0:
            num_tokens = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
            tokens.append(num_tokens)

        # 범주형 변수 토큰화: (batch_size, num_categorical, d_model)
        if x_cat is not None and self.has_categorical:
            # 각 컬럼의 embedding 결과 리스트 생성 -> 크기: (batch_size, 1, d_model)
            cat_tokens_list = [
                emb(x_cat[:, i]).unsqueeze(1) 
                for i, emb in enumerate(self.cat_embeddings)
            ]
            # 컬럼 방향(dim=1)으로 결합 -> 크기: (batch_size, num_categorical, d_model)
            cat_tokens = torch.cat(cat_tokens_list, dim=1)
            tokens.append(cat_tokens)

        # 전체 토큰 결합: (batch_size, total_features, d_model)
        return torch.cat(tokens, dim=1)
        