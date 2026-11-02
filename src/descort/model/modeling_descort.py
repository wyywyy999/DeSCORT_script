import sys
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import nn
from typing import Optional, Tuple
from .configuration_descort import DeSCORTConfig
from .perturbation_loss import perturbation_loss
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from .model_output import DeSCORTLayerOutput,DeSCORTEncoderOutput,DeSCORTModelOutput
from .model_output import DeSCORTForMaskedLMOutput,DeSCORTForPerturbPredictionOutput,DeSCORTForContrastiveLMOutput


class ContinuousValueEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(1, config.hidden_size, bias=config.bias)
        self.act_fn = ACT2FN[config.hidden_act]
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        x = x.to(self.linear1.weight.dtype)
        # expand last dimension
        x = x.unsqueeze(-1)
        x = self.act_fn(self.linear1(x))
        x = self.linear2(x)
        return x

class AutoDiscretizationEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, bin_num, bin_alpha, temperature = 1.0,
                 mask_token_value = None, pad_token_value = None):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha
        self.temperature = temperature
        
        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(self.bin_num, self.dim)
        
        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)
        
        self.bin_num_idx = torch.tensor(range(self.bin_num))
        self.mask_token_value = mask_token_value
        self.pad_token_value = pad_token_value

        self.tensor0 = torch.tensor(0, dtype=torch.long)

    def forward(self, x, output_weight=0):
        x_mask_idx = (x==self.mask_token_value).nonzero()
        x_pad_idx = (x==self.pad_token_value).nonzero()
        x = x.unsqueeze(-1)
        x = self.mlp(x) # [B,N,1] -> [B,N,H]
        x = self.LeakyReLU(x) # [B,N,H]
        x_crosslayer = self.mlp2(x) # [B,N,H]
        x = self.bin_alpha * x + x_crosslayer # [B,N,H]
        weight = self.Softmax(x/self.temperature ) # [B, N, H]
        
        bin_num_idx = self.bin_num_idx.to(x.device) # [H,]
        
        token_emb = self.emb(bin_num_idx) # [H, D]
        x = torch.matmul(weight, token_emb) #[B, N, D]
    
        
        tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)

        mask_token_emb = self.emb_mask(tensor0).to(x.device).type(x.dtype)

        x[x_mask_idx[:,0],x_mask_idx[:,1],:] = mask_token_emb.repeat(x_mask_idx.shape[0],1)

        pad_token_emb = self.emb_pad(tensor0).to(x.device).type(x.dtype)
        x[x_pad_idx[:,0],x_pad_idx[:,1],:] = pad_token_emb.repeat(x_pad_idx.shape[0],1)
    
        if output_weight:
            return x,weight
        return x
        
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.bias)

        self.dropout = nn.Dropout(config.attention_dropout)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        if self.config.pretraining_tp > 1:
            qkv_slicing = self.all_head_size // self.config.pretraining_tp
            query_slices = self.query.weight.split(qkv_slicing, dim=0)
            key_slices = self.key.weight.split(qkv_slicing, dim=0)
            value_slices = self.value.weight.split(qkv_slicing, dim=0)

            query_layer = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_layer = self.transpose_for_scores(torch.cat(query_layer, dim=-1))

            key_layer = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_layer = self.transpose_for_scores(torch.cat(key_layer, dim=-1))

            value_layer = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_layer = self.transpose_for_scores(torch.cat(value_layer, dim=-1))
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            query_layer = self.transpose_for_scores(self.query(hidden_states))

        if getattr(self.config, "use_sdpa", False) and not output_attentions:
            context_layer = F.scaled_dot_product_attention(
                query_layer, key_layer, value_layer, attn_mask=attention_mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
            )
        else:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        if self.config.pretraining_tp > 1:
            context_layer = context_layer.split(
                self.all_head_size // self.config.pretraining_tp, dim=2
            )
            Wo_slices = self.Wo.weight.split(
                self.config.hidden_size // self.config.pretraining_tp, dim=1
            )
            context_layer = self.dropout(
                sum(
                    [
                        F.linear(context_layer[i], Wo_slices[i])
                        for i in range(self.config.pretraining_tp)
                    ]
                )
            )
        else:
            context_layer = self.dropout(self.Wo(context_layer))

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs
    

class DeepseekRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DeepseekMLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [
                    F.linear(x, gate_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
            up_proj = torch.cat(
                [
                    F.linear(x, up_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.contiguous().view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss
class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss

class DeepseekMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList(
            [
                DeepseekMLP(config, intermediate_size=config.moe_intermediate_size)
                for i in range(config.n_routed_experts)
            ]
        )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekMLP(
                config=config, intermediate_size=intermediate_size
            )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.contiguous().view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            hidden_states = hidden_states.repeat_interleave(
                self.num_experts_per_tok, dim=0
            )
            # print(f'repeated_hidden_states.shape:{hidden_states.shape}')
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                expert_output = expert(hidden_states[flat_topk_idx == i])
                
                y = y.to(expert_output.dtype)
                y[flat_topk_idx == i] = expert_output
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(
                hidden_states, flat_topk_idx, topk_weight.view(-1, 1)
            ).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y
    
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_out = expert_out.to(expert_cache.dtype)
            expert_cache.scatter_reduce_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )
        return expert_cache

class DeSCORTLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden = config.hidden_size
        self.attention_dropout = config.attention_dropout
        self.self_attn = MultiHeadAttention(config)
        self.mlp = (
            DeepseekMoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq ==0
            )
            else DeepseekMLP(config)
        )
        
        self.input_layernorm = DeepseekRMSNorm(
            config.hidden_size, eps = config.rms_norm_eps
        )
        
        self.post_attention_layernorm = DeepseekRMSNorm(
            config.hidden_size, eps = config.rms_norm_eps
        )
        
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                output_attentions: Optional[bool] = True,
                ):
        
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        
        
        # Self Attention
        if output_attentions:
            hidden_states, attention_scores = self.self_attn(
                hidden_states, attention_mask, output_attentions
            )
        else:
            (hidden_states,) = self.self_attn(
                hidden_states, attention_mask, output_attentions
            )
        
        ## Residual connection
        hidden_states = residual.to(hidden_states.dtype) + hidden_states
        
        ## Fully connection
        residual = hidden_states
        
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual.to(hidden_states.dtype) + hidden_states
        
        return DeSCORTLayerOutput(
            hidden_states=hidden_states,
            attention_scores=attention_scores if output_attentions else None,
        )
                 
        

class DeSCORTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList(
            DeSCORTLayer(config, layer_idx) for layer_idx in range(config.n_layers)
        )
        
        self.norm = DeepseekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ) :
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer_module,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions += (layer_outputs[1],)

        hidden_states = self.norm(
            hidden_states
        )  # conduct rmsnorm on last layer hidden_states

        if output_hidden_states:
            all_hidden_states += (hidden_states,)  # last layer hidden_states

        return DeSCORTEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        
        
class DeSCORTPreTrainedModel(PreTrainedModel):
    config_class = DeSCORTConfig
    base_model_prefix = 'DeSCORT'
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = False
    
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
    
    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, ...],
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        """
        Creates an attention mask where 0.0 means "attend" and a large negative value
        means "do not attend". Compatible with additive attention (added to scores).

        Args:
            attention_mask: [batch_size, seq_len] with 1.0 for valid tokens, 0.0 for padding.
            input_shape: Shape of the input tensor (batch_size, seq_len).
            dtype: Data type for the output mask. If None, uses the model's dtype.
        """
        if attention_mask is None:
            return None

        if dtype is None:
            dtype = next(self.parameters()).dtype

        # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
      

class DescortConditionEmbeddings(nn.Module):
    """Construct the embeddings from gene_symbol and gene_expression embeddings."""

    def __init__(self, config):
        super().__init__()
        
        self.feature_name_embedding = nn.Embedding(
            config.feature_name_vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )

    def forward(
        self,
        Feature_Name_Index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feature_name_embeddings = self.feature_name_embedding(Feature_Name_Index)   
        return feature_name_embeddings  


class DeSCORTPerturbModel(DeSCORTPreTrainedModel):
    def __init__(self, config, INDEX2ESM2_Dict):
        super().__init__(config)
        self.config = config
        self.ESM2 = INDEX2ESM2_Dict
        projection_dim = getattr(self.config, "esm2_projection_dim", 0)
        if projection_dim > 0:
            self.ESM2_mlp = nn.Sequential(
                nn.Linear(self.config.esm2_input_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.SiLU(),
                nn.Dropout(self.config.hidden_dropout),
                nn.Linear(projection_dim, self.config.hidden_size),
                nn.LayerNorm(self.config.hidden_size),
                nn.SiLU(),
            )
        else:
            self.ESM2_mlp = nn.Sequential(
                nn.Linear(self.config.esm2_input_dim, self.config.hidden_size),
                nn.LayerNorm(self.config.hidden_size),
                nn.SiLU(),
                nn.Dropout(self.config.hidden_dropout),
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.LayerNorm(self.config.hidden_size),
                nn.SiLU(),
            )
        
        self.encoder = DeSCORTEncoder(self.config)
        self.conditional_embeddings = DescortConditionEmbeddings(self.config)
        
        self.embedding_layer = AutoDiscretizationEmbedding(dim = self.config.hidden_size,
                                                            max_seq_len=self.config.max_seq_len,
                                                            bin_num=self.config.bin_num,
                                                            bin_alpha=self.config.bin_alpha,
                                                            mask_token_value=self.config.mask_token_value,
                                                            pad_token_value=self.config.pad_token_value,
                                                            temperature=self.config.temperature )
        
    
    def get_embeddings(self, Perturb_Gene_Index):
        indices = Perturb_Gene_Index.reshape(-1).detach().cpu().tolist()
        missing = sorted(set(indices).difference(self.ESM2))
        if missing:
            raise KeyError(f'Missing ESM2 embeddings for gene indices: {missing[:10]}')
        # Copy indices once; per-row .item() would repeatedly synchronize CUDA.
        return torch.stack([torch.as_tensor(self.ESM2[i]) for i in indices], dim=0)

    def project_gene_embeddings(self, Perturb_Gene_Index, device):
        """Return trainable ESM2 projections in the model's compute dtype."""
        esm2_weight = self.ESM2_mlp[0].weight
        esm2_embedding = self.get_embeddings(Perturb_Gene_Index).to(
            device=device,
            dtype=esm2_weight.dtype,
        )
        if getattr(self.config, "normalize_esm2_embedding", False):
            esm2_embedding = F.layer_norm(
                esm2_embedding, (esm2_embedding.shape[-1],)
            )
        return self.ESM2_mlp(esm2_embedding)
    
    def forward(
        self,
        masked_expr: Optional[torch.Tensor] = None,
        Feature_Name_Index: Optional[torch.Tensor] = None,
        Perturb_Gene_Index:Optional[torch.Tensor] = None,
        gene_embedding: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        input_shape = Feature_Name_Index.size()
        
        out_put = self.embedding_layer(masked_expr)
        out_put = nn.functional.layer_norm(out_put, out_put.shape[-1:])
        # feature_name_embedding,  conditional_embedding = self.conditional_embeddings(Feature_Name_Index,
        #                                                                             Cell_Line_Index,
        #                                                                             Perturb_Type_Index,)
        feature_name_embedding = self.conditional_embeddings(Feature_Name_Index)
        feature_name_embedding = nn.functional.layer_norm(feature_name_embedding, feature_name_embedding.shape[-1:])
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )
        
        if gene_embedding is None:
            gene_embedding = self.project_gene_embeddings(
                Perturb_Gene_Index, out_put.device
            )
        gene_embedding = gene_embedding.unsqueeze(1)
        
        # total_embeds = out_put+ feature_name_embedding + conditional_embedding
        # total_embeds = out_put+ gene_embedding + feature_name_embedding + conditional_embedding
        total_embeds = out_put+ feature_name_embedding +gene_embedding
        encoder_outputs = self.encoder(total_embeds,
                                       extended_attention_mask,
                                       output_attentions,
                                       output_hidden_states)

        sequence_output = encoder_outputs[0]  
        
        return DeSCORTModelOutput(last_hidden_state=sequence_output, 
                                       hidden_states = encoder_outputs.hidden_states, 
                                       attentions = encoder_outputs.attentions)
    


class DeSCORTModel(DeSCORTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.encoder = DeSCORTEncoder(self.config)
        self.conditional_embeddings = DescortConditionEmbeddings(self.config)
        
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_size) * config.initializer_range
        )

        self.embedding_layer = AutoDiscretizationEmbedding(dim = self.config.hidden_size,
                                                            max_seq_len=self.config.max_seq_len,
                                                            bin_num=self.config.bin_num,
                                                            bin_alpha=self.config.bin_alpha,
                                                            mask_token_value=self.config.mask_token_value,
                                                            pad_token_value=self.config.pad_token_value,
                                                            temperature=self.config.temperature )
    
    def forward(
        self,
        masked_expr: Optional[torch.Tensor] = None,
        Feature_Name_Index: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        input_shape = Feature_Name_Index.size()
        
        out_put = self.embedding_layer(masked_expr)
        out_put = nn.functional.layer_norm(out_put, out_put.shape[-1:])
        feature_name_embedding = self.conditional_embeddings(Feature_Name_Index)
        feature_name_embedding = nn.functional.layer_norm(feature_name_embedding, feature_name_embedding.shape[-1:])
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )
        
        total_embeds = out_put+ feature_name_embedding 
        
        # ---- prepend [CLS] token ----
        cls_tokens = self.cls_token.expand(input_shape[0], -1, -1)  # [B, 1, D]
        total_embeds = torch.cat([cls_tokens, total_embeds], dim=1)  # [B, 1+N, D]
        
        cls_mask = torch.ones(input_shape[0], 1, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask_with_cls = torch.cat([cls_mask, attention_mask], dim=1)  # [B, 1+N]

        input_shape_with_cls = attention_mask_with_cls.size()
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask_with_cls, input_shape_with_cls
        )
        
        encoder_outputs = self.encoder(total_embeds,
                                       extended_attention_mask,
                                       output_attentions,
                                       output_hidden_states)

        sequence_output = encoder_outputs[0]  
        
        return DeSCORTModelOutput(last_hidden_state=sequence_output, 
                                       hidden_states = encoder_outputs.hidden_states, 
                                       attentions = encoder_outputs.attentions)
    
    

class MLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        self.decoder = nn.Linear(config.hidden_size, 1, bias=config.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(hidden_states)
        x = self.dense(x)
        x = self.transform_act_fn(x)
        x = self.decoder(x)
        return x


class GeneFiLMPredictionHead(nn.Module):
    """Decode each feature with an explicit projected-ESM2 condition.

    The encoder already receives the gene embedding additively. This head
    supplies a direct path from the gene to every output feature so unseen
    gene information does not have to survive all encoder layers. A bounded
    FiLM transform keeps the newly initialized residual head stable.
    """

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.gene_to_mod = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=config.bias),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2, bias=config.bias),
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 3)
        self.dense = nn.Linear(hidden_size * 3, hidden_size, bias=config.bias)
        self.transform_act_fn = (
            ACT2FN[config.hidden_act]
            if isinstance(config.hidden_act, str)
            else config.hidden_act
        )
        self.decoder = nn.Linear(hidden_size, 1, bias=config.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        gene_embedding: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        scale, bias = self.gene_to_mod(gene_embedding).chunk(2, dim=-1)
        modulated = (
            hidden_states * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(1))
            + 0.1 * bias.unsqueeze(1)
        )
        expanded_gene = gene_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat(
            (hidden_states, expanded_gene, modulated), dim=-1
        )
        x = self.layer_norm(combined)
        x = self.dense(x)
        x = self.transform_act_fn(x)
        return self.decoder(x).squeeze(-1)
    

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        
        features = F.normalize(features, p=2, dim=-1)
        # features = features.unsqueeze(1)
        
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class DeSCORTForContrastiveLM(DeSCORTPreTrainedModel):
    # Mask Gene Expression Bin
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.DeSCORT = DeSCORTModel(config)
        self.mlmhead = MLMHead(config)
        
        self.SupLoss = SupConLoss()
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_size, 128) 
        )
        self.post_init()
        
        self.last_mlm_loss_1 = 0.0
        self.last_mlm_loss_2 = 0.0
        self.last_contrastive_loss = 0.0
        
    
    def forward(
        self, 
        Perturb_Gene_Index: Optional[torch.Tensor] = None,
        Feature_Name_Index_1: Optional[torch.Tensor] = None,
        Feature_Name_Index_2: Optional[torch.Tensor] = None,
        Masked_Expr_1: Optional[torch.Tensor] = None,
        Masked_Expr_2: Optional[torch.Tensor] = None,
        Label_1: Optional[torch.Tensor] = None,
        Label_2: Optional[torch.Tensor] = None,
        Attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    )-> DeSCORTForContrastiveLMOutput:
        
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        
        outputs_1 = self.DeSCORT(
            masked_expr = Masked_Expr_1,
            Feature_Name_Index = Feature_Name_Index_1,
            attention_mask=Attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        outputs_2 = self.DeSCORT(
            masked_expr = Masked_Expr_2,
            Feature_Name_Index = Feature_Name_Index_2,
            attention_mask=Attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        hidden_states_1 = outputs_1[0]  # last_hidden_states

        mlm_logits_1 = self.mlmhead(hidden_states_1[:, 1:, :]).squeeze(-1) # pooled output
        
        hidden_states_2 = outputs_2[0]  # last_hidden_states

        mlm_logits_2 = self.mlmhead(hidden_states_2[:, 1:, :]).squeeze(-1) # pooled output
        
        
        
        if Label_1 is not None:
            mask_1 = Label_1 != -100
            mask_1 = mask_1.to(mlm_logits_1.device)
            valid_logits_1 = mlm_logits_1[mask_1]
            valid_labels_1 = Label_1[mask_1].to(device=mlm_logits_1.device, dtype=mlm_logits_1.dtype)
            
            mlm_loss_1 = torch.nn.functional.mse_loss(valid_logits_1, valid_labels_1)

        else:
            valid_labels_1 = Masked_Expr_1
            mlm_loss_1 = torch.nn.functional.mse_loss(mlm_logits_1, valid_labels_1)
            
            
        if Label_2 is not None:
            mask_2 = Label_2 != -100
            mask_2 = mask_2.to(mlm_logits_2.device)
            valid_logits_2 = mlm_logits_2[mask_2]
            valid_labels_2 = Label_2[mask_2].to(device=mlm_logits_2.device, dtype=mlm_logits_2.dtype)
            
            mlm_loss_2 = torch.nn.functional.mse_loss(valid_logits_2, valid_labels_2)

        else:
            valid_labels_2 = Masked_Expr_2
            mlm_loss_2 = torch.nn.functional.mse_loss(mlm_logits_2, valid_labels_2)  
        
        contrastive_loss_val = 0.0  
        if Perturb_Gene_Index is not None :
            z_i = self.projection_head(hidden_states_1[:, 0, :])
            z_j = self.projection_head(hidden_states_2[:, 0, :])
            contrastive_embedding = torch.stack([z_i,z_j], dim =1)
            contrastive_loss_val = self.SupLoss(contrastive_embedding,Perturb_Gene_Index)
          
        
        self.last_mlm_loss_1 = (mlm_loss_1*0.5).detach().item()
        self.last_mlm_loss_2 = (mlm_loss_2*0.5).detach().item()
        self.last_contrastive_loss = contrastive_loss_val.detach().item()
               
        total_loss = mlm_loss_2*0.5 + mlm_loss_1*0.5 + contrastive_loss_val 
        
        return DeSCORTForContrastiveLMOutput(
            loss=total_loss,
            logits=None,
            hidden_states=None,
            attentions=None,
        )

    def get_latent_embedding(
        self, 
        Perturb_Gene_Index: Optional[torch.Tensor] = None,
        Feature_Name_Index_1: Optional[torch.Tensor] = None,
        Feature_Name_Index_2: Optional[torch.Tensor] = None,
        Attention_mask: Optional[torch.Tensor] = None,
        Masked_Expr: Optional[torch.Tensor] = None,
        Label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    )-> DeSCORTForContrastiveLMOutput:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        
        outputs = self.DeSCORT(
            masked_expr = Masked_Expr,
            Feature_Name_Index = Feature_Name_Index_1,
            attention_mask=Attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        hidden_states = outputs[0]  # last_hidden_states
        cls_token_embedding = hidden_states[:,0,:]
        
        return DeSCORTForContrastiveLMOutput(
            loss=None,
            logits=cls_token_embedding, # 返回处理后的 [Batch_Size, Hidden_Dim]
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class DeSCORTForMaskedLM(DeSCORTPreTrainedModel):
    def __init__(self, config, INDEX2ESM2_Dict):
        super().__init__(config)
        self.config = config
        self.DeSCORT = DeSCORTModel(config, INDEX2ESM2_Dict)
        self.mlmhead = MLMHead(config)
        self.post_init()
        
    
    def forward(
        self, 
        Feature_Name_Index: Optional[torch.Tensor] = None,
        Masked_Expr: Optional[torch.Tensor] = None,
        Label: Optional[torch.Tensor] = None,
        Attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    )-> DeSCORTForMaskedLMOutput:
        
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        
        outputs = self.DeSCORT(
            masked_expr = Masked_Expr,
            Feature_Name_Index = Feature_Name_Index,
            attention_mask=Attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        hidden_states = outputs[0]  # last_hidden_states

        mlm_logits = self.mlmhead(hidden_states[:,1:,:]).squeeze(-1) # pooled output
        
        
        if Label is not None:
            mask = Label != -100
            mask = mask.to(mlm_logits.device)
            valid_logits = mlm_logits[mask]
            valid_labels = Label[mask].to(device=mlm_logits.device, dtype=mlm_logits.dtype)
            
            mlm_loss = torch.nn.functional.mse_loss(valid_logits, valid_labels)

        else:
            valid_labels = Masked_Expr
            mlm_loss = torch.nn.functional.mse_loss(mlm_logits, valid_labels)
        
        return DeSCORTForMaskedLMOutput(
            loss=mlm_loss,
            logits=mlm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_latent_embedding(
        self, 
        Feature_Name_Index: Optional[torch.Tensor] = None,
        Masked_Expr: Optional[torch.Tensor] = None,
        Label: Optional[torch.Tensor] = None,
        Attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    )-> DeSCORTForMaskedLMOutput:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        
        outputs = self.DeSCORT(
            masked_expr = Masked_Expr,
            Feature_Name_Index = Feature_Name_Index,
            attention_mask=Attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        hidden_states = outputs[0]  # last_hidden_states [Batch_Size, Seq_Len, Hidden_Dim]

        return DeSCORTForMaskedLMOutput(
            loss=None,
            logits=hidden_states[:,0,:], # 返回处理后的 [Batch_Size, Hidden_Dim]
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    
class DeSCORTForPerturbPrediction(DeSCORTPreTrainedModel):
    def __init__(self, config, INDEX2ESM2_Dict):
        super().__init__(config)
        self.config = config
        self.DeSCORT = DeSCORTPerturbModel(config, INDEX2ESM2_Dict)
        self.prediction_head_type = getattr(config, "prediction_head_type", "mlm")
        if self.prediction_head_type == "mlm":
            self.pphead = MLMHead(config)
        elif self.prediction_head_type == "gene_film":
            self.pphead = GeneFiLMPredictionHead(config)
        else:
            raise ValueError(
                f"Unsupported prediction_head_type: {self.prediction_head_type!r}"
            )
        self.mean_weight = config.mean_weight
        self.direction_weight = config.direction_weight
        self.marginal_quantile_weight = getattr(
            config, "marginal_quantile_weight", 0.0
        )
        # Backward compatible: checkpoints created before residual prediction
        # do not contain this field and therefore retain absolute prediction.
        self.predict_residual = getattr(config, "predict_residual", False)
        self.use_global_delta_prior = getattr(
            config, "use_global_delta_prior", False
        )
        if self.use_global_delta_prior:
            # Feature order is fixed by Feature_Name_Index in the processed
            # datasets.  This parameter represents the morphology response
            # shared across training perturbations; the FiLM head learns only
            # the gene-conditioned correction around it.
            self.global_delta = nn.Parameter(torch.zeros(config.max_seq_len))
        else:
            self.register_parameter("global_delta", None)
        self.last_loss_components = {}
        self.post_init()
        if self.prediction_head_type == "gene_film" and self.predict_residual:
            # Only residual mode starts from identity (prediction = control).
            # Direct-absolute FiLM must retain the normal decoder
            # initialization; zero initialization would make every initial
            # perturbed profile equal to zero.
            nn.init.zeros_(self.pphead.decoder.weight)
            if self.pphead.decoder.bias is not None:
                nn.init.zeros_(self.pphead.decoder.bias)

    def _predict_values(
        self,
        hidden_states: torch.Tensor,
        gene_embedding: torch.Tensor,
        control_expr: torch.Tensor,
    ) -> torch.Tensor:
        if self.prediction_head_type == "gene_film":
            perturb_delta = self.pphead(hidden_states, gene_embedding)
        else:
            perturb_delta = self.pphead(hidden_states).squeeze(-1)
        if self.use_global_delta_prior:
            sequence_length = perturb_delta.shape[1]
            if sequence_length > self.global_delta.numel():
                raise ValueError(
                    "Prediction length exceeds global-delta capacity: "
                    f"{sequence_length} > {self.global_delta.numel()}"
                )
            perturb_delta = perturb_delta + self.global_delta[
                :sequence_length
            ].to(dtype=perturb_delta.dtype).unsqueeze(0)
        return (
            control_expr + perturb_delta
            if self.predict_residual
            else perturb_delta
        )
        
        
    def forward(
        self, 
        Perturb_Gene_Index: Optional[torch.Tensor] = None,
        Feature_Name_Index: Optional[torch.Tensor] = None,
        Control_Expr: Optional[torch.Tensor] = None,
        Perturb_Expr: Optional[torch.Tensor] = None,
        Attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    )-> DeSCORTForPerturbPredictionOutput:
        
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        
        gene_embedding = self.DeSCORT.project_gene_embeddings(
            Perturb_Gene_Index, Control_Expr.device
        )
        outputs = self.DeSCORT(
            masked_expr = Control_Expr,
            Perturb_Gene_Index = Perturb_Gene_Index,
            gene_embedding = gene_embedding,
            Feature_Name_Index = Feature_Name_Index,
            attention_mask=Attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        hidden_states = outputs[0]  # last_hidden_states

        perturb_pred = self._predict_values(
            hidden_states, gene_embedding, Control_Expr
        )
        mlm_loss = perturb_pred.sum() * 0.0
        if Perturb_Expr is not None:
            # Preserve the public weight attributes used by older experiment scripts.
            self.config.mean_weight = self.mean_weight
            self.config.direction_weight = self.direction_weight
            self.config.marginal_quantile_weight = self.marginal_quantile_weight
            mlm_loss, self.last_loss_components = perturbation_loss(
                perturb_pred, Perturb_Expr, Control_Expr, Perturb_Gene_Index,
                Attention_mask, self.config, Feature_Name_Index,
            )
            self.last_loss_components["total"] = float(mlm_loss.detach())

        
        return DeSCORTForPerturbPredictionOutput(
            loss=mlm_loss,
            logits=hidden_states[:, 0, :],
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    @torch.no_grad()
    def inference(
        self, 
        Perturb_Gene_Index: Optional[torch.Tensor] = None,
        Feature_Name_Index: Optional[torch.Tensor] = None,
        Control_Expr: Optional[torch.Tensor] = None,
        Perturb_Expr: Optional[torch.Tensor] = None,
        Attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    )-> DeSCORTForPerturbPredictionOutput:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        
        gene_embedding = self.DeSCORT.project_gene_embeddings(
            Perturb_Gene_Index, Control_Expr.device
        )
        outputs = self.DeSCORT(
            masked_expr = Control_Expr,
            Perturb_Gene_Index = Perturb_Gene_Index,
            gene_embedding = gene_embedding,
            Feature_Name_Index = Feature_Name_Index,
            attention_mask=Attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        hidden_states = outputs[0]  # last_hidden_states [Batch_Size, Seq_Len, Hidden_Dim]

        perturb_pred = self._predict_values(
            hidden_states, gene_embedding, Control_Expr
        )
        if Attention_mask is not None:
        # 方法1：返回 list of tensors（推荐，因为每个样本长度可能不同）
            batch_logits = []
            for i in range(perturb_pred.shape[0]):
                valid_len = Attention_mask[i].sum().int().item()
                valid_logits = perturb_pred[i, :valid_len]  # 只取有效长度部分
                batch_logits.append(valid_logits)
            
            # 或者方法2：如果需要保持 tensor 格式，可以用 packed sequence
            # 但通常 list of tensors 更灵活
            final_logits = torch.stack(batch_logits,dim = 0)
        else:
            # 如果没有 mask，假设全部有效
            final_logits = perturb_pred
        
        return DeSCORTForPerturbPredictionOutput(
            loss=None,
            logits=final_logits, # 返回处理后的 [Batch_Size, Hidden_Dim]
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float = 0.0,
    activation: nn.Module = nn.ReLU,
) -> nn.Sequential:
    """Build an MLP of `n_layers` from `in_dim` to `out_dim`."""
    layers = []
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")

    if n_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim))
    else:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation())
        layers.append(nn.Dropout(dropout))

        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, out_dim))

    return nn.Sequential(*layers)


class LinearESM2PerturbPrediction(nn.Module):
    """Control_Expr + ESM2(gene) → MLP → Perturb_Expr

    input:  Control_Expr [B, N]  cat  ESM2_embedding [B, 5120]
    output: Perturb_Expr [B, N]
    """
    def __init__(
        self,
        input_dim: int,
        esm2_dim: int = 5120,
        esm2_hidden: int = 256,
        mlp_hidden: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        # ESM2 投影：5120 → esm2_hidden
        self.esm2_proj = build_mlp(
            in_dim=esm2_dim,
            out_dim=esm2_hidden,
            hidden_dim=esm2_hidden * 2,
            n_layers=2,
            dropout=dropout,
            activation=nn.SiLU,
        )
        # 主干：[N_features + esm2_hidden] → N_features
        self.mlp = build_mlp(
            in_dim=input_dim + esm2_hidden,
            out_dim=input_dim,
            hidden_dim=mlp_hidden,
            n_layers=n_layers,
            dropout=dropout,
            activation=nn.SiLU,
        )

    def forward(self, control_expr, esm2_emb):
        """
        Args:
            control_expr: [B, N]  control cell expression
            esm2_emb:     [B, 5120] ESM2 gene embedding
        Returns:
            perturb_pred: [B, N]  predicted perturbed expression
        """
        gene_feat = self.esm2_proj(esm2_emb.to(control_expr.dtype))  # [B, H]
        combined = torch.cat([control_expr, gene_feat], dim=-1)      # [B, N+H]
        return self.mlp(combined)                                     # [B, N]
