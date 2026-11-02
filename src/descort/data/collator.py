"""Data collators for DeSCORT training tasks."""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Sequence padding / sampling helpers
# ---------------------------------------------------------------------------

def _sample_or_truncate_plus_pad(expression, 
                                 feature_ids, 
                                 max_length, 
                                 sampling=True,
                                 keep_first_n_tokens=0, pad_value=100):
    """Pad or sample a sequence to max_length, keeping expression and feature_ids aligned."""
    seq_len = expression.shape[0]
    assert feature_ids.shape[0] == seq_len, "expression and feature_ids must have same length"

    if seq_len == max_length:
        return expression, feature_ids
    elif seq_len > max_length:
        if sampling:
            return _sample(expression, feature_ids, max_length, keep_first_n_tokens)
        else:
            return expression[:max_length], feature_ids[:max_length]
    else:
        return _pad(expression, feature_ids, max_length, pad_value)


def _sample(expression, feature_ids, max_length, keep_first_n_tokens=0):
    device = expression.device
    seq_len = expression.shape[0]
    if keep_first_n_tokens == 0:
        indices = torch.randperm(seq_len, device=device)[:max_length]
        return expression[indices], feature_ids[indices]
    else:
        _n = keep_first_n_tokens
        if seq_len <= _n:
            return expression, feature_ids
        if max_length <= _n:
            return expression[:max_length], feature_ids[:max_length]
        remaining_indices = torch.randperm(seq_len - _n, device=device)[:max_length - _n]
        indices = torch.cat([torch.arange(_n, device=device), remaining_indices + _n], dim=0)
        return expression[indices], feature_ids[indices]


def _pad(expression, feature_ids, max_length, pad_value=100):
    seq_len = expression.shape[0]
    pad_len = max_length - seq_len

    if len(expression.shape) == 1:
        expr_padding = torch.full((pad_len,), pad_value, dtype=expression.dtype, device=expression.device)
        new_expression = torch.cat([expression, expr_padding], dim=0)
    else:
        padding_shape = (pad_len,) + expression.shape[1:]
        expr_padding = torch.full(padding_shape, pad_value, dtype=expression.dtype, device=expression.device)
        new_expression = torch.cat([expression, expr_padding], dim=0)

    id_padding = torch.full((pad_len,), 0, dtype=feature_ids.dtype, device=feature_ids.device)
    new_feature_ids = torch.cat([feature_ids, id_padding], dim=0)
    return new_expression, new_feature_ids


def _build_attention_mask(feature_name_index_list):
    masks = [(expr != 0).float() for expr in feature_name_index_list]
    return torch.stack(masks, dim=0)


# ---------------------------------------------------------------------------
# Base collator: shared _call_cell_painting logic
# ---------------------------------------------------------------------------

class DeSCORTForMaskedLMDataCollator:
    """Base collator for MLM training on Cell Painting data."""

    def __init__(self, do_mlm: bool = True,
                 mlm_probability: float = 0.15, mask_value: int = -100,
                 pad_value: int = 100, max_length: Optional[int] = 1024,
                 sampling: bool = True, keep_first_n_tokens: int = 0):
        self.do_mlm = do_mlm
        self.mlm_probability = mlm_probability
        self.mask_value = mask_value
        self.pad_value = pad_value
        self.max_length = max_length
        self.sampling = sampling
        self.keep_first_n_tokens = keep_first_n_tokens

    def __call__(self, examples: Dict) -> Dict[str, torch.Tensor]:
        if 'Feature_expression' not in examples[0]:
            raise ValueError("Feature_expression missing!")
        data_dict = self._call_cell_painting(examples)
        del data_dict['Feature_Expression']
        return data_dict

    def _call_cell_painting(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not isinstance(examples[0], Mapping):
            return NotImplementedError

        device = examples[0]["Feature_expression"].device
        Feature_Expressions_list = []
        Feature_Name_Index_list = []

        for i in range(len(examples)):
            Feature_Expression = examples[i]['Feature_expression']
            Feature_name_Index = examples[i]['Feature_name_index']

            if self.max_length is not None:
                New_Feature_Expression, New_Feature_name_Index = _sample_or_truncate_plus_pad(
                    Feature_Expression, Feature_name_Index, self.max_length,
                    sampling=self.sampling, keep_first_n_tokens=self.keep_first_n_tokens,
                    pad_value=self.pad_value,
                )
            Feature_Expressions_list.append(New_Feature_Expression)
            Feature_Name_Index_list.append(New_Feature_name_Index)

        Feature_Expressions = torch.stack(Feature_Expressions_list, dim=0).to(device)
        Feature_Name_Indexs = torch.stack(Feature_Name_Index_list, dim=0).to(device)
        attention_masks = _build_attention_mask(Feature_Name_Index_list).to(device)

        data_dict = {
            'Feature_Expression': Feature_Expressions,
            'Feature_Name_Index': Feature_Name_Indexs,
            'Attention_mask': attention_masks,
        }

        if self.do_mlm:
            masked_expressions, labels = self._mask(Feature_Expressions)
        else:
            masked_expressions = Feature_Expressions
            labels = None

        data_dict["Masked_Expr"] = masked_expressions
        data_dict['Label'] = labels
        return data_dict

    def _mask(self, expressions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = expressions.device
        shape = expressions.shape
        probability_matrix = torch.full(shape, self.get_mlm_probability())

        if self.max_length is not None and expressions.dim() == 2:
            padding_mask = (expressions != self.pad_value)
            probability_matrix = probability_matrix * padding_mask.float()

        mask = torch.bernoulli(probability_matrix).bool().to(device)
        masked_expressions = expressions.masked_fill(mask, self.mask_value)
        labels = expressions.clone()
        labels[~mask] = -100
        return masked_expressions, labels

    def get_mlm_probability(self) -> float:
        if isinstance(self.mlm_probability, float):
            return self.mlm_probability
        elif isinstance(self.mlm_probability, list):
            return np.random.choice(self.mlm_probability)
        else:
            raise ValueError(f"mlm_probability must be float or list, got {self.mlm_probability}")


# ---------------------------------------------------------------------------
# Contrastive collator: generates two views
# ---------------------------------------------------------------------------

class DeSCORTForContrastiveLMDataCollator(DeSCORTForMaskedLMDataCollator):
    """Collator for contrastive learning: generates two random views per sample."""

    def _call_cell_painting(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not isinstance(examples[0], Mapping):
            return NotImplementedError

        device = examples[0]["Feature_expression"].device
        Feature_Expressions_list_1, Feature_Name_Index_list_1 = [], []
        Feature_Expressions_list_2, Feature_Name_Index_list_2 = [], []
        Perturb_Gene_Index_list = []

        for i in range(len(examples)):
            Feature_Expression = examples[i]['Feature_expression']
            Perturb_Gene_Index = examples[i]['Perturb_Gene_Index']
            Feature_name_Index = examples[i]['Feature_name_index']

            if self.max_length is not None:
                New_Feature_Expression_1, New_Feature_name_Index_1 = _sample_or_truncate_plus_pad(
                    Feature_Expression, Feature_name_Index, self.max_length,
                    sampling=self.sampling, keep_first_n_tokens=self.keep_first_n_tokens,
                    pad_value=self.pad_value,
                )
                New_Feature_Expression_2, New_Feature_name_Index_2 = _sample_or_truncate_plus_pad(
                    Feature_Expression, Feature_name_Index, self.max_length,
                    sampling=self.sampling, keep_first_n_tokens=self.keep_first_n_tokens,
                    pad_value=self.pad_value,
                )

            Perturb_Gene_Index_list.append(Perturb_Gene_Index)
            Feature_Expressions_list_1.append(New_Feature_Expression_1)
            Feature_Name_Index_list_1.append(New_Feature_name_Index_1)
            Feature_Expressions_list_2.append(New_Feature_Expression_2)
            Feature_Name_Index_list_2.append(New_Feature_name_Index_2)

        Feature_Expressions_1 = torch.stack(Feature_Expressions_list_1, dim=0).to(device)
        Feature_Expressions_2 = torch.stack(Feature_Expressions_list_2, dim=0).to(device)
        Feature_Name_Indexs_1 = torch.stack(Feature_Name_Index_list_1, dim=0).to(device)
        Feature_Name_Indexs_2 = torch.stack(Feature_Name_Index_list_2, dim=0).to(device)
        Perturb_Gene_Indexs = torch.stack(Perturb_Gene_Index_list, dim=0).to(device)
        attention_masks = _build_attention_mask(Feature_Name_Index_list_1).to(device)

        data_dict = {
            'Feature_Expression': New_Feature_Expression_1,
            'Perturb_Gene_Index': Perturb_Gene_Indexs,
            'Feature_Name_Index_1': Feature_Name_Indexs_1,
            'Feature_Name_Index_2': Feature_Name_Indexs_2,
            'Attention_mask': attention_masks,
        }

        if self.do_mlm:
            masked_expressions_1, labels_1 = self._mask(Feature_Expressions_1)
            masked_expressions_2, labels_2 = self._mask(Feature_Expressions_2)
            data_dict["Masked_Expr_1"] = masked_expressions_1
            data_dict['Label_1'] = labels_1
            data_dict["Masked_Expr_2"] = masked_expressions_2
            data_dict['Label_2'] = labels_2
        else:
            data_dict["Masked_Expr"] = Feature_Expressions_1
            data_dict['Label'] = None

        return data_dict


# ---------------------------------------------------------------------------
# Perturb prediction collator (base MLM variant)
# ---------------------------------------------------------------------------

class DeSCORTPerturbForMaskedLMDataCollator(DeSCORTForMaskedLMDataCollator):
    """Collator for perturbation prediction with MLM."""

    def _call_cell_painting(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not isinstance(examples[0], Mapping):
            return NotImplementedError

        device = examples[0]["Feature_expression"].device
        Feature_Expressions_list, Perturb_Gene_Index_list, Feature_Name_Index_list = [], [], []

        for i in range(len(examples)):
            Feature_Expression = examples[i]['Feature_expression']
            Perturb_Gene_Index = examples[i]['Perturb_Gene_Index']
            Feature_name_Index = examples[i]['Feature_name_index']

            if self.max_length is not None:
                New_Feature_Expression, New_Feature_name_Index = _sample_or_truncate_plus_pad(
                    Feature_Expression, Feature_name_Index, self.max_length,
                    sampling=self.sampling, keep_first_n_tokens=self.keep_first_n_tokens,
                    pad_value=self.pad_value,
                )
            Perturb_Gene_Index_list.append(Perturb_Gene_Index)
            Feature_Expressions_list.append(New_Feature_Expression)
            Feature_Name_Index_list.append(New_Feature_name_Index)

        Feature_Expressions = torch.stack(Feature_Expressions_list, dim=0).to(device)
        Perturb_Gene_Indexs = torch.stack(Perturb_Gene_Index_list, dim=0).to(device)
        Feature_Name_Indexs = torch.stack(Feature_Name_Index_list, dim=0).to(device)
        attention_masks = _build_attention_mask(Feature_Name_Index_list).to(device)

        data_dict = {
            'Feature_Expression': Feature_Expressions,
            'Perturb_Gene_Index': Perturb_Gene_Indexs,
            'Feature_Name_Index': Feature_Name_Indexs,
            'Attention_mask': attention_masks,
        }

        if self.do_mlm:
            masked_expressions, labels = self._mask(Feature_Expressions)
        else:
            masked_expressions = Feature_Expressions
            labels = None

        data_dict["Masked_Expr"] = masked_expressions
        data_dict['Label'] = labels
        return data_dict


# ---------------------------------------------------------------------------
# PerturbPrediction collator: contrastive control/perturb
# ---------------------------------------------------------------------------

class PerturbPredictionDataCollator:
    """Collate unpaired control/perturbation populations with aligned features."""

    def __init__(self, max_length: int = 1024, sampling: bool = False,
                 pad_value: int = 100, keep_first_n_tokens: int = 0,
                 feature_order: Optional[Sequence[int]] = None,
                 feature_alignment: str = 'pad'):
        self.max_length = max_length
        self.sampling = sampling
        self.pad_value = pad_value
        self.keep_first_n_tokens = keep_first_n_tokens
        if feature_alignment not in ('pad', 'intersection'):
            raise ValueError('feature_alignment must be pad or intersection')
        if feature_alignment == 'intersection' and feature_order is None:
            raise ValueError('intersection alignment requires a global feature_order')
        self.feature_alignment = feature_alignment
        self.feature_order = None
        if feature_order is not None:
            self.feature_order = torch.as_tensor(feature_order, dtype=torch.long)
            if (self.feature_order.ndim != 1 or self.feature_order.numel() == 0
                    or (self.feature_order <= 0).any()
                    or (self.feature_order[1:] <= self.feature_order[:-1]).any()):
                raise ValueError('feature_order must contain sorted, unique, positive feature IDs')

    def _align_features(self, example):
        """Place values on a shared feature axis; missing features remain masked."""
        ids = example['Feature_name_index']
        target, control = example['Perturb_expression'], example['Control_expression']
        if ids.ndim != 1 or target.shape != ids.shape or control.shape != ids.shape:
            raise ValueError('Feature IDs, perturbation values and control values must have equal lengths')
        valid = ids > 0
        if (ids < 0).any() or ids[valid].unique().numel() != int(valid.sum()):
            raise ValueError('Feature IDs must be non-negative and non-padding IDs must be unique')
        reference = self.feature_order.to(ids.device)
        if self.feature_alignment == 'intersection':
            # Select the global intersection, never a per-batch intersection.
            sorted_ids, order = torch.sort(ids)
            positions = torch.searchsorted(sorted_ids, reference)
            if (positions >= len(sorted_ids)).any() or not torch.equal(sorted_ids[positions], reference):
                raise ValueError('Sample is missing required intersection features')
            selection = order[positions]
            return dict(example, Feature_name_index=ids[selection],
                        Perturb_expression=target[selection], Control_expression=control[selection])
        positions = torch.searchsorted(reference, ids[valid])
        if (positions >= len(reference)).any() or not torch.equal(reference[positions], ids[valid]):
            raise ValueError('Sample contains feature IDs outside feature_order')
        aligned_ids = torch.zeros_like(reference)
        aligned_target = target.new_full(reference.shape, self.pad_value)
        aligned_control = control.new_full(reference.shape, self.pad_value)
        aligned_ids[positions] = ids[valid]
        aligned_target[positions] = target[valid]
        aligned_control[positions] = control[valid]
        return dict(example, Feature_name_index=aligned_ids,
                    Perturb_expression=aligned_target, Control_expression=aligned_control)

    def __call__(self, examples: Dict) -> Dict[str, torch.Tensor]:
        if 'Perturb_expression' not in examples[0]:
            raise ValueError("Perturb_expression missing!")
        return self._call_cell_painting(examples)

    def _call_cell_painting(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not isinstance(examples[0], Mapping):
            return NotImplementedError

        if self.feature_order is not None:
            examples = [self._align_features(example) for example in examples]

        device = examples[0]["Perturb_expression"].device
        Perturb_Expressions_list, Control_Expressions_list = [], []
        Feature_Name_Index_list, Perturb_Genes_Index_list = [], []

        shared_selection = None
        reference_ids = examples[0]['Feature_name_index']
        if self.sampling and self.max_length is not None and len(reference_ids) > self.max_length:
            if not all(torch.equal(e['Feature_name_index'], reference_ids) for e in examples):
                raise ValueError('Population feature sampling requires matching feature orders')
            shared_selection, _ = _sample(
                torch.arange(len(reference_ids), device=reference_ids.device),
                reference_ids, self.max_length, self.keep_first_n_tokens,
            )

        for i in range(len(examples)):
            Perturb_Expression = examples[i]['Perturb_expression']
            Control_Expression = examples[i]['Control_expression']
            Feature_name_Index = examples[i]['Feature_name_index']
            Perturb_Gene_Index = examples[i]['Perturb_Gene_Index']

            if shared_selection is not None:
                Perturb_Expression = Perturb_Expression[shared_selection]
                Control_Expression = Control_Expression[shared_selection]
                Feature_name_Index = Feature_name_Index[shared_selection]

            if self.max_length is not None:
                New_Perturb_Expression, New_Feature_name_Index = _sample_or_truncate_plus_pad(
                    Perturb_Expression, Feature_name_Index, self.max_length,
                    sampling=False, keep_first_n_tokens=self.keep_first_n_tokens,
                    pad_value=self.pad_value,
                )
                New_Control_Expression, _ = _sample_or_truncate_plus_pad(
                    Control_Expression, Feature_name_Index, self.max_length,
                    sampling=False, keep_first_n_tokens=self.keep_first_n_tokens,
                    pad_value=self.pad_value,
                )
            else:
                New_Perturb_Expression = Perturb_Expression
                New_Control_Expression = Control_Expression
                New_Feature_name_Index = Feature_name_Index

            Perturb_Genes_Index_list.append(Perturb_Gene_Index)
            Perturb_Expressions_list.append(New_Perturb_Expression)
            Control_Expressions_list.append(New_Control_Expression)
            Feature_Name_Index_list.append(New_Feature_name_Index)

        Perturb_Expressions = torch.stack(Perturb_Expressions_list, dim=0)
        Control_Expressions = torch.stack(Control_Expressions_list, dim=0)
        Feature_Name_Indexs = torch.stack(Feature_Name_Index_list, dim=0)
        Perturb_Gene_Index = torch.stack(Perturb_Genes_Index_list, dim=0)
        attention_masks = _build_attention_mask(Feature_Name_Index_list)

        return {
            'Control_Expr': Control_Expressions,
            'Perturb_Expr': Perturb_Expressions,
            'Attention_mask': attention_masks,
            'Feature_Name_Index': Feature_Name_Indexs,
            'Perturb_Gene_Index': Perturb_Gene_Index,
        }
