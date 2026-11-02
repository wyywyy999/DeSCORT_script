"""Shared training log callback."""
from transformers import TrainerCallback

class LossLoggingCallback(TrainerCallback):
    """Inject mlm_loss and contrastive_loss into the Trainer's log output."""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if model is not None and hasattr(model, "module"):
            actual_model = model.module
        else:
            actual_model = model

        if actual_model is not None and hasattr(actual_model, "last_mlm_loss_1"):
            logs["mlm_loss_1"] = actual_model.last_mlm_loss_1
            logs["mlm_loss_2"] = actual_model.last_mlm_loss_2
            logs["contrastive_loss"] = actual_model.last_contrastive_loss
