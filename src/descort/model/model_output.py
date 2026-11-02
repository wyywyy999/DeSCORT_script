"""Model output dataclasses for DeSCORT."""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple

from transformers.utils import ModelOutput


@dataclass
class DeSCORTLayerOutput(ModelOutput):
    hidden_states: torch.FloatTensor = None
    attention_scores: Optional[torch.FloatTensor] = None


@dataclass
class DeSCORTEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class DeSCORTModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class DeSCORTForMaskedLMOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    
@dataclass
class DeSCORTForContrastiveLMOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class DeSCORTForPerturbPredictionOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
