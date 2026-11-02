"""DeSCORT model configuration."""

from transformers import PretrainedConfig


class DeSCORTConfig(PretrainedConfig):
    """
    Configuration for the DeSCORT model.

    Parameter usage by class:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Class                          │ Config parameters used                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ ContinuousValueEncoder         │ hidden_size, bias, hidden_act           │
    │ AutoDiscretizationEmbedding    │ hidden_size, max_seq_len, bin_num,      │
    │                                │ bin_alpha, temperature, mask_token_value,│
    │                                │ pad_token_value                         │
    │ MultiHeadAttention             │ hidden_size, bias, num_attention_heads, │
    │                                │ attention_dropout, pretraining_tp       │
    │ DeepseekMLP                    │ hidden_size, bias, hidden_act,          │
    │                                │ intermediate_size, pretraining_tp       │
    │ MoEGate                        │ hidden_size, n_routed_experts,          │
    │                                │ num_experts_per_tok, scoring_func,      │
    │                                │ aux_loss_alpha, seq_aux, norm_topk_prob │
    │ DeepseekMoE                    │ num_experts_per_tok, n_routed_experts,  │
    │                                │ moe_intermediate_size, n_shared_experts │
    │ DeepseekRMSNorm                │ rms_norm_eps                            │
    │ DeSCORTLayer                   │ hidden_size, attention_dropout,         │
    │                                │ n_routed_experts, first_k_dense_replace,│
    │                                │ moe_layer_freq, rms_norm_eps            │
    │ DeSCORTEncoder                 │ hidden_size, n_layers, rms_norm_eps     │
    │ DeSCORTPreTrainedModel         │ initializer_range                       │
    │ DescortConditionEmbeddings     │ hidden_size, feature_name_vocab_size,   │
    │                                │ pad_token_id                            │
    │ DeSCORTPerturbModel            │ hidden_size, max_seq_len, bin_num,      │
    │                                │ bin_alpha, temperature, mask_token_value,│
    │                                │ pad_token_value, hidden_dropout,        │
    │                                │ output_attentions*, output_hidden_states*│
    │ DeSCORTModel                   │ hidden_size, max_seq_len, bin_num,      │
    │                                │ bin_alpha, temperature, mask_token_value,│
    │                                │ pad_token_value, initializer_range,     │
    │                                │ output_attentions*, output_hidden_states*│
    │ MLMHead                        │ hidden_size, bias, hidden_act           │
    │ DeSCORTForMaskedLM             │ output_attentions*, output_hidden_states*│
    │ DeSCORTForContrastiveLM        │ hidden_size, output_attentions*,        │
    │                                │ output_hidden_states*                   │
    │ DeSCORTForPerturbPrediction    │ energy_weight, mse_weight,              │
    │                                │ output_attentions*, output_hidden_states*│
    └─────────────────────────────────────────────────────────────────────────┘
    * inherited from PretrainedConfig, not defined here
    """

    model_type = "DeSCORT"

    def __init__(
        self,
        # ═══════════════════════════════════════════════════════════════════
        # common — used by 3+ independent classes
        # ═══════════════════════════════════════════════════════════════════
        # ContinuousValueEncoder, MultiHeadAttention, DeepseekMLP, MoEGate,
        # DeSCORTLayer, DeSCORTEncoder, DeSCORTPerturbModel, DeSCORTModel,
        # DescortConditionEmbeddings, MLMHead, DeSCORTForContrastiveLM,
        # DeSCORTPreTrainedModel
        hidden_size: int = 256,
        # ContinuousValueEncoder, MultiHeadAttention, DeepseekMLP, MLMHead
        bias: bool = True,
        # MultiHeadAttention, DeepseekMLP
        pretraining_tp: int = 1,
        # ContinuousValueEncoder, DeepseekMLP, MLMHead
        hidden_act: str = "silu",

        # ═══════════════════════════════════════════════════════════════════
        # AutoDiscretizationEmbedding — gene expression value discretization
        # Used by: DeSCORTPerturbModel, DeSCORTModel (via embedding_layer)
        # ═══════════════════════════════════════════════════════════════════
        max_seq_len: int = 1024,
        bin_num: int = 10,
        bin_alpha: int = 1,
        temperature: float = 1.0,
        mask_token_value: int = -100,
        pad_token_value: int = 100,

        # ═══════════════════════════════════════════════════════════════════
        # DescortConditionEmbeddings — gene feature name lookup
        # Used by: DescortConditionEmbeddings
        # ═══════════════════════════════════════════════════════════════════
        feature_name_vocab_size: int = 7366,
        pad_token_id: int = 0,

        # ═══════════════════════════════════════════════════════════════════
        # MultiHeadAttention
        # Used by: MultiHeadAttention, DeSCORTLayer (stores attention_dropout)
        # ═══════════════════════════════════════════════════════════════════
        num_attention_heads: int = 8,
        attention_dropout: float = 0.0,
        use_sdpa: bool = False,

        # ═══════════════════════════════════════════════════════════════════
        # DeepseekMLP — dense feed-forward network
        # Used by: DeepseekMLP
        # ═══════════════════════════════════════════════════════════════════
        intermediate_size: int = 1024,

        # ═══════════════════════════════════════════════════════════════════
        # MoE (Mixture of Experts) — expert routing & architecture
        # Used by: MoEGate, DeepseekMoE, DeSCORTLayer
        # ═══════════════════════════════════════════════════════════════════
        n_routed_experts: int = 8,
        num_experts_per_tok: int = 4,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        moe_intermediate_size: int = 1024,
        n_shared_experts: int = 1,

        # MoE gate — expert selection & load balancing
        # Used by: MoEGate
        scoring_func: str = "softmax",
        seq_aux: bool = True,
        aux_loss_alpha: float = 0.001,
        norm_topk_prob: bool = True,

        # ═══════════════════════════════════════════════════════════════════
        # DeepseekRMSNorm — layer normalization
        # Used by: DeSCORTLayer, DeSCORTEncoder
        # ═══════════════════════════════════════════════════════════════════
        rms_norm_eps: float = 1e-6,

        # ═══════════════════════════════════════════════════════════════════
        # DeSCORTEncoder — transformer stack
        # Used by: DeSCORTEncoder
        # ═══════════════════════════════════════════════════════════════════
        n_layers: int = 8,

        # ═══════════════════════════════════════════════════════════════════
        # Weight initialization
        # Used by: DeSCORTPreTrainedModel._init_weights, DeSCORTModel (cls_token)
        # ═══════════════════════════════════════════════════════════════════
        initializer_range: float = 0.02,

        # ═══════════════════════════════════════════════════════════════════
        # DeSCORTPerturbModel — ESM2 protein embedding MLP
        # Used by: DeSCORTPerturbModel (ESM2_mlp input dimension and dropout)
        # ═══════════════════════════════════════════════════════════════════
        hidden_dropout: float = 0.1,
        esm2_input_dim: int = 5120,
        esm2_projection_dim: int = 0,
        normalize_esm2_embedding: bool = False,

        # ═══════════════════════════════════════════════════════════════════
        # DeSCORTForPerturbPrediction — perturbation prediction loss weights
        # Used by: DeSCORTForPerturbPrediction
        # ═══════════════════════════════════════════════════════════════════
        energy_weight: float = 1.0,
        mse_weight: float = 0.0,
        mean_weight: float = 1.0,
        direction_weight: float = 1.0,
        sign_weight: float = 0.0,
        variance_weight: float = 0.0,
        marginal_quantile_weight: float = 0.0,
        centered_quantile_weight: float = 0.0,
        sliced_wasserstein_weight: float = 0.0,
        sliced_wasserstein_num_projections: int = 64,
        sliced_wasserstein_projection_seed: int = 42,
        standardized_effect_weight: float = 0.0,
        confidence_direction_weight: float = 0.0,
        gene_contrastive_weight: float = 0.0,
        gene_contrastive_temperature: float = 0.1,
        gene_residual_weight: float = 0.0,
        effect_confidence_threshold: float = 2.5,
        effect_confidence_temperature: float = 0.5,
        predict_residual: bool = False,
        use_global_delta_prior: bool = False,
        prediction_head_type: str = "mlm",

        **kwargs,
    ):
        super().__init__(**kwargs)
        # common
        self.hidden_size = hidden_size
        self.bias = bias
        self.pretraining_tp = pretraining_tp
        self.hidden_act = hidden_act

        # AutoDiscretizationEmbedding
        self.max_seq_len = max_seq_len
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha
        self.temperature = temperature
        self.mask_token_value = mask_token_value
        self.pad_token_value = pad_token_value

        # DescortConditionEmbeddings
        self.feature_name_vocab_size = feature_name_vocab_size
        self.pad_token_id = pad_token_id

        # MultiHeadAttention
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.use_sdpa = use_sdpa

        # DeepseekMLP
        self.intermediate_size = intermediate_size

        # MoE
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.moe_intermediate_size = moe_intermediate_size
        self.n_shared_experts = n_shared_experts

        # MoE gate
        self.scoring_func = scoring_func
        self.seq_aux = seq_aux
        self.aux_loss_alpha = aux_loss_alpha
        self.norm_topk_prob = norm_topk_prob

        # Norm
        self.rms_norm_eps = rms_norm_eps

        # Encoder
        self.n_layers = n_layers

        # Initialization
        self.initializer_range = initializer_range

        # DeSCORTPerturbModel
        self.hidden_dropout = hidden_dropout
        if esm2_input_dim <= 0:
            raise ValueError("esm2_input_dim must be positive")
        self.esm2_input_dim = esm2_input_dim
        self.esm2_projection_dim = esm2_projection_dim
        self.normalize_esm2_embedding = normalize_esm2_embedding

        # DeSCORTForPerturbPrediction
        self.energy_weight = energy_weight
        self.mse_weight = mse_weight
        self.mean_weight = mean_weight
        self.direction_weight = direction_weight
        self.sign_weight = sign_weight
        self.variance_weight = variance_weight
        self.marginal_quantile_weight = marginal_quantile_weight
        self.centered_quantile_weight = centered_quantile_weight
        self.sliced_wasserstein_weight = sliced_wasserstein_weight
        self.sliced_wasserstein_num_projections = sliced_wasserstein_num_projections
        self.sliced_wasserstein_projection_seed = sliced_wasserstein_projection_seed
        self.standardized_effect_weight = standardized_effect_weight
        self.confidence_direction_weight = confidence_direction_weight
        self.gene_contrastive_weight = gene_contrastive_weight
        self.gene_contrastive_temperature = gene_contrastive_temperature
        self.gene_residual_weight = gene_residual_weight
        self.effect_confidence_threshold = effect_confidence_threshold
        self.effect_confidence_temperature = effect_confidence_temperature
        self.predict_residual = predict_residual
        self.use_global_delta_prior = use_global_delta_prior
        self.prediction_head_type = prediction_head_type
