> 原实验历史审查记录，保留供追溯；本次整理未重新验证其中全部结论。链接已指向新副本。

# DeSCORT Config Parameter Audit Report

> Generated: 2026-08-19
>
> Source files:
> - [configuration_descort.py](../../src/descort/model/configuration_descort.py)
> - [modeling_descort.py](../../src/descort/model/modeling_descort.py)

---

## 1. Summary

| Category | Count |
|---|---|
| Parameters defined in `DeSCORTConfig.__init__` signature | 36 (with 1 duplicate) |
| Parameters assigned in `DeSCORTConfig.__init__` body | 35 |
| Parameters actually used in `modeling_descort.py` | 33 |
| **Missing from config but used in modeling** | **3** |
| **In config but unused in modeling** | **7** |
| **Critical bugs (will cause runtime error)** | **2** |

---

## 2. Parameters Missing from Config (Need to be Added)

These parameters are referenced in `modeling_descort.py` via `self.config.XXX` but are NOT defined in `DeSCORTConfig`:

### 2.1 `energy_weight` (CRITICAL)

- **Used in**: `DeSCORTForPerturbPrediction.__init__` at [modeling_descort.py:1177](../../src/descort/model/modeling_descort.py#L1177)
- **Meaning**: Weight coefficient for the energy distance loss (Sinkhorn/earth mover distance) in perturbation prediction. Balances the distribution-matching loss against the per-position MSE loss.
- **Suggested default**: `float = 0.1`
- **Suggested signature**: `energy_weight: float = 0.1`

### 2.2 `mse_weight` (CRITICAL)

- **Used in**: `DeSCORTForPerturbPrediction.__init__` at [modeling_descort.py:1178](../../src/descort/model/modeling_descort.py#L1178)
- **Meaning**: Weight coefficient for the per-position MSE loss in perturbation prediction. Controls how much the model is penalized for element-wise reconstruction error.
- **Suggested default**: `float = 0.9`
- **Suggested signature**: `mse_weight: float = 0.9`

### 2.3 `value_pad_token_id` (CRITICAL)

- **Used in**: `AutoDiscretizationEmbedding` instantiation at [modeling_descort.py:648](../../src/descort/model/modeling_descort.py#L648) and [modeling_descort.py:729](../../src/descort/model/modeling_descort.py#L729)
- **Meaning**: The token ID used for padding in the continuous value (gene expression) input. Passed to `AutoDiscretizationEmbedding` as `pad_token_id`, which uses it to identify and replace padding positions with a learned padding embedding.
- **Current status**: `self.value_pad_token_id = value_pad_token_id` is in the `__init__` body (line 73) but `value_pad_token_id` is NOT in the function signature → **Runtime NameError**.
- **Suggested default**: `int = 100`
- **Suggested signature**: `value_pad_token_id: int = 100`

---

## 3. Parameters in Config but NOT Used in Modeling

These parameters are defined in `DeSCORTConfig` but are not referenced anywhere in `modeling_descort.py`. They may be:
- Dead code / legacy parameters
- Intended for future use
- Used only in external training scripts

### 3.1 `embedding_bias`

- **Defined**: `bool = True` at [configuration_descort.py:34](../../src/descort/model/configuration_descort.py#L34)
- **Meaning**: Controls whether the embedding layers use bias terms.
- **Status**: Not used in any `nn.Embedding` instantiation within modeling.

### 3.2 `batch_norm_continuous_input`

- **Defined**: `bool = False` at [configuration_descort.py:35](../../src/descort/model/configuration_descort.py#L35)
- **Meaning**: Intended to control whether batch normalization is applied to continuous input values.
- **Status**: Not used. The `ContinuousValueEncoder` class is defined but never instantiated in the current model architecture.

### 3.3 `embedding_dropout`

- **Defined**: `float = 0.1` at [configuration_descort.py:36](../../src/descort/model/configuration_descort.py#L36)
- **Meaning**: Dropout rate applied to embeddings after they are computed.
- **Status**: Not used. No dropout is applied after embedding computation in the current code.

### 3.4 `cell_line_vocab_size`

- **Defined**: `int = 3` at [configuration_descort.py:28](../../src/descort/model/configuration_descort.py#L28)
- **Meaning**: Vocabulary size for cell line type embeddings. Was likely used in a previous version that had cell line conditional embeddings.
- **Status**: Not used. The `DescortConditionEmbeddings` class only uses `feature_name_vocab_size`; cell line embeddings are commented out in the forward pass (see [modeling_descort.py:685-687](../../src/descort/model/modeling_descort.py#L685-L687)).

### 3.5 `perturb_type_vocab_size`

- **Defined**: `int = 3` at [configuration_descort.py:29](../../src/descort/model/configuration_descort.py#L29)
- **Meaning**: Vocabulary size for perturbation type embeddings (e.g., overexpression, knockout, compound).
- **Status**: Not used. Same as above — the conditional embedding code is commented out.

### 3.6 `solvent_vocab_size`

- **Defined**: `int = 3` at [configuration_descort.py:30](../../src/descort/model/configuration_descort.py#L30)
- **Meaning**: Vocabulary size for solvent/vehicle type embeddings.
- **Status**: Not used. Same as above.

### 3.7 `emb_hidden_size`

- **Defined**: `int = 256` at [configuration_descort.py:32](../../src/descort/model/configuration_descort.py#L32)
- **Meaning**: Intended hidden size for embedding projections (separate from the main `hidden_size`).
- **Status**: Not used. All embeddings use `config.hidden_size` directly.

---

## 4. Bugs Found in configuration_descort.py

### 4.1 `pad_token_id` is duplicated (BUG)

At [configuration_descort.py:25](../../src/descort/model/configuration_descort.py#L25) and [configuration_descort.py:31](../../src/descort/model/configuration_descort.py#L31), `pad_token_id` is defined twice in the same function signature:

```python
# Line 25
pad_token_id: int = 100,    # ← intended for value_pad_token_id (AutoDiscretizationEmbedding)

# Line 31
pad_token_id: int = 0,      # ← intended for feature_name_embedding padding_idx
```

Python uses the **last** definition, so `pad_token_id` is always `0`. The value `100` is lost. The first definition (line 25) should be renamed to `value_pad_token_id`.

### 4.2 `d_numerical` is referenced but never defined (CRITICAL BUG)

At [configuration_descort.py:68](../../src/descort/model/configuration_descort.py#L68):

```python
self.d_numerical = d_numerical
```

`d_numerical` is not in the function signature → **Runtime NameError** when instantiating `DeSCORTConfig()`.

**Fix**: Either add `d_numerical` to the signature or remove this line. Based on context, it likely represents the number of numerical features (gene expression values), and the default should be set according to the dataset.

### 4.3 `temperature` has wrong type annotation (MINOR BUG)

At [configuration_descort.py:21](../../src/descort/model/configuration_descort.py#L21):

```python
temperature: int = 1.0,
```

The type annotation is `int` but the default value is `1.0` (a float). Should be:

```python
temperature: float = 1.0,
```

---

## 5. Complete Parameter Reference

### 5.1 Common Parameters

| Parameter | Type | Default | Used in Modeling | Meaning |
|---|---|---|---|---|
| `hidden_size` | `int` | `256` | ✅ | Dimensionality of all hidden states throughout the model (embeddings, attention, MLP). |
| `bias` | `bool` | `True` | ✅ | Whether linear layers (QKV projections, MLP, output) include bias terms. |
| `pretraining_tp` | `int` | `1` | ✅ | Tensor parallelism degree for pretraining. >1 splits QKV and MLP computations across tensors. Use 1 for single-GPU. |
| `hidden_act` | `str` | `"silu"` | ✅ | Activation function used in MLP layers and MLM head. Must be a key in `transformers.activations.ACT2FN`. |

### 5.2 AutoDiscretizationEmbedding Parameters

| Parameter | Type | Default | Used in Modeling | Meaning |
|---|---|---|---|---|
| `max_seq_len` | `int` | `1024` | ✅ | Maximum sequence length (number of genes/features). Determines the maximum input length. |
| `bin_num` | `int` | `10` | ✅ | Number of discretization bins for continuous gene expression values. Higher values give finer granularity. |
| `bin_alpha` | `int` | `1` | ✅ | Weight for the cross-layer residual connection in the auto-discretization MLP: `x = alpha * mlp1(x) + mlp2(mlp1(x))`. |
| `temperature` | `float` | `1.0` | ✅ | Temperature for the softmax over discretization bins. Lower values make the distribution sharper (harder assignment). |
| `mask_token_id` | `int` | `-100` | ✅ | Token ID used to identify masked positions in gene expression input. Positions with this value get a learned mask embedding. |
| `value_pad_token_id` | `int` | `100` | ❌ **MISSING** | Token ID for padding in gene expression values. Positions with this value get a learned padding embedding. |

### 5.3 Embedding Parameters

| Parameter | Type | Default | Used in Modeling | Meaning |
|---|---|---|---|---|
| `feature_name_vocab_size` | `int` | `3312` | ✅ | Vocabulary size for gene feature name embeddings. Should match the number of unique gene names/IDs in the dataset. |
| `pad_token_id` | `int` | `0` | ✅ | Padding index for `nn.Embedding` layers (feature name embeddings). Tokens with this ID are ignored in embedding lookup. |
| `emb_hidden_size` | `int` | `256` | ❌ | (Unused) Intended hidden size for embedding projections. |
| `embedding_bias` | `bool` | `True` | ❌ | (Unused) Intended bias flag for embedding layers. |
| `batch_norm_continuous_input` | `bool` | `False` | ❌ | (Unused) Intended flag for batch normalization on continuous inputs. |
| `embedding_dropout` | `float` | `0.1` | ❌ | (Unused) Intended dropout rate after embeddings. |

### 5.4 Conditional Embedding Vocabularies

| Parameter | Type | Default | Used in Modeling | Meaning |
|---|---|---|---|---|
| `cell_line_vocab_size` | `int` | `3` | ❌ | (Unused) Vocabulary size for cell line type embeddings. |
| `perturb_type_vocab_size` | `int` | `3` | ❌ | (Unused) Vocabulary size for perturbation type embeddings. |
| `solvent_vocab_size` | `int` | `3` | ❌ | (Unused) Vocabulary size for solvent/vehicle embeddings. |

### 5.5 Multi-Head Attention Parameters

| Parameter | Type | Default | Used in Modeling | Meaning |
|---|---|---|---|---|
| `num_attention_heads` | `int` | `8` | ✅ | Number of attention heads. `hidden_size` must be divisible by this value. |
| `attention_dropout` | `float` | `0.0` | ✅ | Dropout probability applied to attention weights after softmax. |

### 5.6 MLP Parameters

| Parameter | Type | Default | Used in Modeling | Meaning |
|---|---|---|---|---|
| `intermediate_size` | `int` | `1024` | ✅ | Hidden size of the dense MLP's intermediate (gate/up) projection. Typically 4× `hidden_size`. |

### 5.7 Mixture of Experts (MoE) Parameters

| Parameter | Type | Default | Used in Modeling | Meaning |
|---|---|---|---|---|
| `n_routed_experts` | `int` | `8` | ✅ | Number of routed experts in each MoE layer. Each expert is an independent DeepseekMLP. |
| `num_experts_per_tok` | `int` | `4` | ✅ | Top-K: number of experts selected per token. Each token is routed to this many experts. |
| `hidden_dropout` | `float` | `0.1` | ✅ | Dropout rate in the ESM2 protein embedding projection MLP. |
| `first_k_dense_replace` | `int` | `0` | ✅ | Number of initial layers that use dense MLP instead of MoE. Layers 0 to `first_k_dense_replace-1` use dense MLP. |
| `moe_layer_freq` | `int` | `1` | ✅ | Frequency of MoE layers. `1` = every layer from `first_k_dense_replace` onward is MoE. `2` = every other layer is MoE. |
| `moe_intermediate_size` | `int` | `1024` | ✅ | Intermediate size for each expert's MLP. Usually smaller than `intermediate_size` since experts are specialized. |
| `n_shared_experts` | `int` | `1` | ✅ | Number of shared experts (always active, applied to all tokens). Output is added to the routed expert output. Set to `None` to disable. |

### 5.8 MoE Gate Parameters

| Parameter | Type | Default | Used in Modeling | Meaning |
|---|---|---|---|---|
| `scoring_func` | `str` | `"softmax"` | ✅ | Scoring function for expert selection. Currently only `"softmax"` is supported. |
| `seq_aux` | `bool` | `True` | ✅ | Whether to compute sequence-level auxiliary load balancing loss. `True` = sequence-level, `False` = token-level. |
| `aux_loss_alpha` | `float` | `0.001` | ✅ | Weight coefficient for the auxiliary load balancing loss. Higher values encourage more balanced expert usage. |
| `norm_topk_prob` | `bool` | `True` | ✅ | Whether to normalize top-K expert probabilities to sum to 1. Recommended to keep `True`. |

### 5.9 Normalization Parameters

| Parameter | Type | Default | Used in Modeling | Meaning |
|---|---|---|---|---|
| `rms_norm_eps` | `float` | `1e-6` | ✅ | Epsilon for RMSNorm to avoid division by zero. Used in both input and post-attention layer norms. |

### 5.10 Encoder Parameters

| Parameter | Type | Default | Used in Modeling | Meaning |
|---|---|---|---|---|
| `n_layers` | `int` | `8` | ✅ | Number of transformer layers (DeSCORTLayer) in the encoder. |

### 5.11 Initialization Parameters

| Parameter | Type | Default | Used in Modeling | Meaning |
|---|---|---|---|---|
| `initializer_range` | `float` | `0.02` | ✅ | Standard deviation for normal initialization of linear and embedding weights. Also used to scale the CLS token. |

### 5.12 Loss Parameters (MISSING)

| Parameter | Type | Suggested Default | Used in Modeling | Meaning |
|---|---|---|---|---|
| `energy_weight` | `float` | `0.1` | ✅ (but missing) | Weight for the energy distance (Sinkhorn divergence) loss in perturbation prediction. |
| `mse_weight` | `float` | `0.9` | ✅ (but missing) | Weight for the per-position MSE loss in perturbation prediction. |

### 5.13 Inherited Parameters (from `PretrainedConfig`)

| Parameter | Type | Inherited Default | Used in Modeling | Meaning |
|---|---|---|---|---|
| `output_attentions` | `bool` | `False` | ✅ | Whether to return attention weights from all layers. Inherited from `PretrainedConfig`. |
| `output_hidden_states` | `bool` | `False` | ✅ | Whether to return hidden states from all layers. Inherited from `PretrainedConfig`. |

---

## 6. Suggested Fixes

### 6.1 Critical: Fix `configuration_descort.py` signature

```python
def __init__(
    self,
    ## common
    hidden_size: int = 256,
    bias: bool = True,
    pretraining_tp: int = 1,
    hidden_act: str = "silu",

    # AutoDiscretizationEmbedding
    max_seq_len: int = 1024,
    bin_num: int = 10,
    bin_alpha: int = 1,
    temperature: float = 1.0,        # FIX: int → float
    mask_token_id: int = -100,
    value_pad_token_id: int = 100,   # FIX: was pad_token_id (duplicate)

    # Embeddings
    feature_name_vocab_size: int = 3312,
    pad_token_id: int = 0,

    # Conditional vocabularies (currently unused)
    cell_line_vocab_size: int = 3,
    perturb_type_vocab_size: int = 3,
    solvent_vocab_size: int = 3,

    emb_hidden_size: int = 256,
    embedding_bias: bool = True,
    batch_norm_continuous_input: bool = False,
    embedding_dropout: float = 0.1,

    # MultiHeadAttention
    num_attention_heads: int = 8,
    attention_dropout: float = 0.0,

    # MLP
    intermediate_size: int = 1024,

    # MoE
    n_routed_experts: int = 8,
    num_experts_per_tok: int = 4,
    hidden_dropout: float = 0.1,
    first_k_dense_replace: int = 0,
    moe_layer_freq: int = 1,
    moe_intermediate_size: int = 1024,
    n_shared_experts: int = 1,

    # MoE gate
    scoring_func: str = "softmax",
    seq_aux: bool = True,
    aux_loss_alpha: float = 0.001,
    norm_topk_prob: bool = True,

    # Norm
    rms_norm_eps: float = 1e-6,

    # Encoder
    n_layers: int = 8,

    # Initialization
    initializer_range: float = 0.02,

    # Loss weights (NEW)
    energy_weight: float = 0.1,       # ADD
    mse_weight: float = 0.9,          # ADD

    # Numerical features (was missing from signature)
    d_numerical: int = 3312,          # ADD (or remove from __init__ body)

    **kwargs,
):
```

### 6.2 Summary of Required Changes

| # | Change | Severity |
|---|---|---|
| 1 | Add `value_pad_token_id: int = 100` to signature, replace duplicate `pad_token_id` | **Critical** (runtime error) |
| 2 | Add `d_numerical: int = 3312` to signature (or remove `self.d_numerical = d_numerical` from body) | **Critical** (runtime error) |
| 3 | Add `energy_weight: float = 0.1` to signature | **Critical** (runtime error when using DeSCORTForPerturbPrediction) |
| 4 | Add `mse_weight: float = 0.9` to signature | **Critical** (runtime error when using DeSCORTForPerturbPrediction) |
| 5 | Fix `temperature: int = 1.0` → `temperature: float = 1.0` | Minor (type annotation) |
| 6 | Consider removing unused parameters or adding them back to the model | Advisory |