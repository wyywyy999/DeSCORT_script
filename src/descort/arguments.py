"""Training and model arguments for DeSCORT."""

from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    # ---- common ----
    hidden_size: int = field(default=256)
    bias: bool = field(default=True)
    pretraining_tp: int = field(default=1)
    hidden_act: str = field(default="silu")

    # ---- AutoDiscretizationEmbedding ----
    max_seq_len: int = field(default=1024)
    bin_num: int = field(default=10)
    bin_alpha: int = field(default=1)
    temperature: float = field(default=1.0)
    mask_token_value: int = field(default=-100)
    pad_token_value: int = field(default=100)

    # ---- DescortConditionEmbeddings ----
    feature_name_vocab_size: int = field(default=7366)
    pad_token_id: int = field(default=0)

    # ---- MultiHeadAttention ----
    num_attention_heads: int = field(default=8)
    attention_dropout: float = field(default=0.0)
    use_sdpa: bool = field(default=False)

    # ---- DeepseekMLP ----
    intermediate_size: int = field(default=1024)

    # ---- MoE ----
    n_routed_experts: int = field(default=8)
    num_experts_per_tok: int = field(default=4)
    first_k_dense_replace: int = field(default=0)
    moe_layer_freq: int = field(default=1)
    moe_intermediate_size: int = field(default=1024)
    n_shared_experts: int = field(default=1)
    scoring_func: str = field(default="softmax")
    seq_aux: bool = field(default=True)
    aux_loss_alpha: float = field(default=0.001)
    norm_topk_prob: bool = field(default=True)

    # ---- RMSNorm ----
    rms_norm_eps: float = field(default=1e-6)

    # ---- Encoder ----
    n_layers: int = field(default=8)

    # ---- Initialization ----
    initializer_range: float = field(default=0.02)

    # ---- DeSCORTPerturbModel ----
    hidden_dropout: float = field(default=0.1)
    esm2_input_dim: int = field(
        default=5120,
        metadata={"help": "Input dimension of ESM2/conditioning embeddings."},
    )
    esm2_projection_dim: int = field(default=0)
    normalize_esm2_embedding: bool = field(default=False)

    # ---- DeSCORTForPerturbPrediction ----
    mean_weight: float = field(default=1.0)
    direction_weight: float = field(default=1.0)
    marginal_quantile_weight: float = field(default=1.0)
    centered_quantile_weight: float = field(default=0.0)
    sliced_wasserstein_weight: float = field(default=0.0)
    sliced_wasserstein_num_projections: int = field(default=64)
    sliced_wasserstein_projection_seed: int = field(default=42)
    variance_weight: float = field(default=0.0)
    gene_contrastive_weight: float = field(default=0.0)
    gene_contrastive_temperature: float = field(default=0.1)
    predict_residual: bool = field(default=False)
    use_global_delta_prior: bool = field(default=False)
    freeze_global_delta_prior: bool = field(default=False)
    freeze_morphology_backbone: bool = field(default=False)
    prediction_head_type: str = field(default="mlm")

    def __post_init__(self):
        if self.gene_contrastive_temperature <= 0:
            raise ValueError('gene_contrastive_temperature must be positive')
        if self.sliced_wasserstein_num_projections < 1:
            raise ValueError('sliced_wasserstein_num_projections must be positive')
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by "
                f"num_attention_heads {self.num_attention_heads}"
            )
        if self.esm2_input_dim <= 0:
            raise ValueError("esm2_input_dim must be positive")
        if self.esm2_projection_dim < 0:
            raise ValueError("esm2_projection_dim must be non-negative")
        loss_weights = {
            "centered_quantile_weight": self.centered_quantile_weight,
            "sliced_wasserstein_weight": self.sliced_wasserstein_weight,
            "variance_weight": self.variance_weight,
            "gene_contrastive_weight": self.gene_contrastive_weight,
            "mean_weight": self.mean_weight,
            "direction_weight": self.direction_weight,
            "marginal_quantile_weight": self.marginal_quantile_weight,
        }
        negative = {name: value for name, value in loss_weights.items() if value < 0}
        if negative:
            raise ValueError(f"Perturbation loss weights must be non-negative: {negative}")
        if not any(loss_weights.values()):
            raise ValueError("At least one perturbation loss weight must be positive")
        valid_prediction_heads = {"mlm", "gene_film"}
        if self.prediction_head_type not in valid_prediction_heads:
            raise ValueError(
                "prediction_head_type must be one of "
                f"{sorted(valid_prediction_heads)}, got {self.prediction_head_type!r}"
            )


@dataclass
class DataArguments:
    train_dataset_path: str = field(default=None)
    mlm_probability: float = 0.15
    sampling: bool = True
    keep_first_n_tokens: int = 0
    do_mlm: bool = True
    mask_value: int = -100
    max_length: int = 1024
    pad_value: int = 100


@dataclass
class FinetuneDataArguments:
    pretrianed_model_path: str = field(default=None)
