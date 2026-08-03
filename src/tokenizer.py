import torch
import torch.nn as nn

class TabularFeatureTokenizer(nn.Module):
    def __init__(self, num_numerical : int, cat_cardinalities : list[int], d_model : int, num_emmedding_method : str = "periodical", **kwargs):
        """
        Args:
            num_numerical (int): 수치형 변수의 개수
            cat_cardinalities (list of int): 각 범주형 변수의 고유 카테고리 개수 (Label Encoding 범위)
            d_model (int): 출력 토큰의 차원 수
        """
        super().__init__()
        self.num_numerical = num_numerical
        self.num_categorical = len(cat_cardinalities)
        self.has_categorical = self.num_categorical > 0

        self.num_emmedding_method = num_emmedding_method

        # 1. 수치형 토크나이저 
        if num_numerical > 0:
            if num_emmedding_method == "periodical":
                n_frequencies = kwargs.get("n_frequencies",16)
                sigma = kwargs.get("sigma",0.01)
                self.num_periodical_embedding = PeriodicEmbedding(num_numerical, d_model, n_frequencies = n_frequencies, sigma = sigma)
            
            elif num_emmedding_method == "PLE":
                bin_edges_dict = kwargs.get("bin_edges_dict",None)
                if bin_edges_dict is not None:
                    self.num_ple = PiecewiseLinearEncoding(bin_edges_dict, d_model)
            
            elif num_emmedding_method == "linear":
                self.num_weight = nn.Parameter(torch.randn(num_numerical, d_model))
                self.num_bias = nn.Parameter(torch.randn(num_numerical, d_model))
            
            elif num_emmedding_method == "PLE_linear":
                bin_edges_dict = kwargs.get("bin_edges_dict",None)
                if bin_edges_dict is not None:
                    self.num_ple = PiecewiseLinearEncoding(bin_edges_dict, d_model)
                self.num_weight = nn.Parameter(torch.randn(num_numerical, d_model))
                self.num_bias = nn.Parameter(torch.randn(num_numerical, d_model))
            
            self.num_feature_identifier = nn.Parameter(torch.randn(1,self.num_numerical,d_model))
            nn.init.normal_(self.num_feature_identifier, std = 0.01)

        # 2. 범주형 토크나이저
        if self.has_categorical:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(num_embeddings=cardinality, embedding_dim=d_model, padding_idx=0)
                for cardinality in cat_cardinalities
            ])
            self.cat_feature_identifier = nn.Parameter(torch.randn(1,self.num_categorical,d_model))
            nn.init.normal_(self.num_feature_identifier, std = 0.01)


    def forward(self, x_num=None, x_cat=None):
        tokens = []

        # 수치형 변수 토큰화: (batch_size, num_numerical, d_model)
        if x_num is not None and self.num_numerical > 0:
            if self.num_emmedding_method == "periodical":
                x_num_val = self.num_periodical_embedding(x_num)
            elif self.num_emmedding_method == "PLE":
                x_num_val = self.num_ple(x_num)
            elif self.num_emmedding_method == "linear":
                x_num_val = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
            elif self.num_emmedding_method == "PLE_linear":
                x_num_ple_val = self.num_ple(x_num)
                x_num_linear_val = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
                x_num_val = x_num_ple_val + x_num_linear_val
            num_tokens = x_num_val + self.num_feature_identifier
            tokens.append(num_tokens)

        # 범주형 변수 토큰화: (batch_size, num_categorical, d_model)
        if x_cat is not None and self.has_categorical:
            # 각 컬럼의 embedding 결과 리스트 생성 -> 크기: (batch_size, 1, d_model)
            cat_tokens_list = [
                emb(x_cat[:, i]).unsqueeze(1) 
                for i, emb in enumerate(self.cat_embeddings)
            ]
            # 컬럼 방향(dim=1)으로 결합 -> 크기: (batch_size, num_categorical, d_model)
            x_cat_val = torch.cat(cat_tokens_list, dim=1)
            cat_tokens = x_cat_val + self.cat_feature_identifier
            tokens.append(cat_tokens)

        # 전체 토큰 결합: (batch_size, total_features, d_model)
        return torch.cat(tokens, dim=1)

class PeriodicEmbedding(nn.Module):
    """    
    Args:
        num_features (int): 수치형 피처(컬럼)의 개수
        d_embedding (int): FT-Transformer의 메인 임베딩 차원 (d_model)
        n_frequencies (int): 인코딩할 주파수의 개수 (k). 기본값 16 -> 32차원 삼각함수 벡터 생성
        sigma (float): 주파수 가중치(w) 초기화 시 사용할 표준편차
    """
    def __init__(
        self, 
        num_features: int, 
        d_embedding: int, 
        n_frequencies: int = 16, 
        sigma: float = 0.01
    ):
        super().__init__()
        self.num_features = num_features
        self.d_embedding = d_embedding
        self.n_frequencies = n_frequencies

        # 1. 학습 가능한 주파수(w) 및 위상(b) 파라미터 선언
        # Shape: [num_features, n_frequencies]
        self.coefficients = nn.Parameter(torch.randn(num_features, n_frequencies) * sigma)
        self.biases = nn.Parameter(torch.zeros(num_features, n_frequencies))

        # 2. 피처별 선형 프로젝션 가중치
        # 2k 차원(sin + cos) -> d_embedding 차원으로 변환
        # Shape: [num_features, n_frequencies * 2, d_embedding]
        self.linear_weight = nn.Parameter(
            torch.randn(num_features, n_frequencies * 2, d_embedding) * 0.02
        )
        self.linear_bias = nn.Parameter(torch.zeros(num_features, d_embedding))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 전처리된 수치형 입력 텐서 [Batch_Size, Num_Features]
        Returns:
            torch.Tensor: 임베딩 텐서 [Batch_Size, Num_Features, d_embedding]
        """
        B, N = x.shape
        assert N == self.num_features, f"입력 피처 수({N})가 설정값({self.num_features})과 다릅니다."

        # 차원 확장: [B, N] -> [B, N, 1]
        x_expanded = x.unsqueeze(-1)

        # Step 1: 2 * pi * w * x + b 계산 -> Shape: [B, N, n_frequencies]
        angles = 2 * torch.pi * self.coefficients.unsqueeze(0) * x_expanded + self.biases.unsqueeze(0)

        # Step 2: sin과 cos을 구해 Concatenate -> Shape: [B, N, n_frequencies * 2]
        periodic_features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        # Step 3: 피처별 Linear Projection 연산
        # einsum 연산 설명:
        # b: Batch, n: Num_Features, f: Frequencies*2, d: d_embedding
        # periodic_features (b, n, f) x linear_weight (n, f, d) -> out (b, n, d)
        out = torch.einsum('bnf,nfd->bnd', periodic_features, self.linear_weight) + self.linear_bias.unsqueeze(0)

        return out

class PiecewiseLinearEncoding(nn.Module):
    def __init__(self, bin_edges_dict: dict, d_embedding: int):
        """
        Args:
            bin_edges_dict (dict): {컬럼명: 경계값_텐서} 형태의 딕셔너리
            d_embedding (int): FT-Transformer의 메인 임베딩 차원 (d_model)
        """
        super().__init__()
        self.num_features = len(bin_edges_dict)
        self.d_embedding = d_embedding
        
        self.feature_names = list(bin_edges_dict.keys())
        for i, (col, edges) in enumerate(bin_edges_dict.items()):
            self.register_buffer(f"edges_{i}", edges)
            
        # 각 피처의 Bin 개수 구하기
        self.num_bins = [len(edges) - 1 for edges in bin_edges_dict.values()]
        
        # [수정 포인트 1] 각 Bin 구간(k)마다 d_embedding 크기의 학습 가능한 가중치 벡터 생성
        # ModuleList 안에 각 피처별 Parameter(Bin_Size, d_embedding)를 보관
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(n_bins, d_embedding))
            for n_bins in self.num_bins
        ])
        
        # [수정 포인트 2] 편향(Bias)도 피처별로 추가 (선택적이지만 논문 권장)
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(d_embedding))
            for _ in self.num_bins
        ])
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Kaiming / Xavier 초깃값 설정
        for w in self.weights:
            nn.init.kaiming_uniform_(w, a=1)

    def _encode_single_feature(self, x_i: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """
        x_i: [Batch_Size]
        edges: [Num_Edges]
        returns: [Batch_Size, Bin_Size]
        """
        left = edges[:-1]   # [Bin_Size]
        right = edges[1:]   # [Bin_Size]
        
        x_expanded = x_i.unsqueeze(-1)  # [Batch_Size, 1]
        
        # (x - left) / (right - left) 계산 후 0.0 ~ 1.0 범위로 클리핑
        encoded = (x_expanded - left) / (right - left + 1e-8)
        encoded = torch.clamp(encoded, min=0.0, max=1.0) # [Batch, Bin_Size]
        
        return encoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [Batch_Size, Num_Features]
        Returns:
            torch.Tensor: [Batch_Size, Num_Features, d_embedding]
        """
        encoded_features = []
        
        for i in range(self.num_features):
            x_i = x[:, i]
            edges = getattr(self, f"edges_{i}")
            
            # 1. 구간별 충전 비율 인코딩 -> [Batch_Size, Bin_Size]
            ple_val = self._encode_single_feature(x_i, edges)
            
            # 2. [수정 포인트 3] 정석 선형 결합 (Einsum 연산)
            # ple_val: [Batch, Bin_Size]
            # weight:  [Bin_Size, d_embedding]
            # 결과:     [Batch, d_embedding]
            feat_emb = torch.einsum('bk, kd -> bd', ple_val, self.weights[i]) + self.biases[i]
            
            encoded_features.append(feat_emb)
            
        # [Batch, Num_Features, d_embedding]
        return torch.stack(encoded_features, dim=1)