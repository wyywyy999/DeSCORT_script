# Support both an installed package and direct execution from this checkout.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path(__file__).resolve().parents[2] / "src"
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))

import os
import sys
import logging
from pathlib import Path


from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    TrainerCallback,
)
from datasets import load_from_disk

from descort.model.configuration_descort import DeSCORTConfig
from descort.data.collator import DeSCORTForContrastiveLMDataCollator
from descort.arguments import ModelArguments, DataArguments, FinetuneDataArguments
from descort.model.modeling_descort import DeSCORTForContrastiveLM

logger = logging.getLogger(__name__)


from descort.training import LossLoggingCallback


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, FinetuneDataArguments, TrainingArguments)
    )
    model_args, dataset_args, finetune_args, training_args = (
        parser.parse_args_into_dataclasses()
    )
    model_args: ModelArguments
    dataset_args: DataArguments
    finetune_args: FinetuneDataArguments
    training_args: TrainingArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and "
            "is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, "
        "16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16 or training_args.bf16,
    )

    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Dataset parameters %s", dataset_args)

    # Set seed
    set_seed(training_args.seed)

    # ---- Build config from CLI args (consistent with DeSCORTConfig) ----
    config = DeSCORTConfig(
        # common
        hidden_size=model_args.hidden_size,
        bias=model_args.bias,
        pretraining_tp=model_args.pretraining_tp,
        hidden_act=model_args.hidden_act,
        # AutoDiscretizationEmbedding
        max_seq_len=model_args.max_seq_len,
        bin_num=model_args.bin_num,
        bin_alpha=model_args.bin_alpha,
        temperature=model_args.temperature,
        mask_token_value=model_args.mask_token_value,
        pad_token_value=model_args.pad_token_value,
        # DescortConditionEmbeddings
        feature_name_vocab_size=model_args.feature_name_vocab_size,
        pad_token_id=model_args.pad_token_id,
        # MultiHeadAttention
        num_attention_heads=model_args.num_attention_heads,
        attention_dropout=model_args.attention_dropout,
        # DeepseekMLP
        intermediate_size=model_args.intermediate_size,
        # MoE
        n_routed_experts=model_args.n_routed_experts,
        num_experts_per_tok=model_args.num_experts_per_tok,
        first_k_dense_replace=model_args.first_k_dense_replace,
        moe_layer_freq=model_args.moe_layer_freq,
        moe_intermediate_size=model_args.moe_intermediate_size,
        n_shared_experts=model_args.n_shared_experts,
        scoring_func=model_args.scoring_func,
        seq_aux=model_args.seq_aux,
        aux_loss_alpha=model_args.aux_loss_alpha,
        norm_topk_prob=model_args.norm_topk_prob,
        # RMSNorm
        rms_norm_eps=model_args.rms_norm_eps,
        # Encoder
        n_layers=model_args.n_layers,
        # Initialization
        initializer_range=model_args.initializer_range,
        # DeSCORTPerturbModel
        hidden_dropout=model_args.hidden_dropout,
        # DeSCORTForPerturbPrediction
    )

    # ---- Collator ----
    collator = DeSCORTForContrastiveLMDataCollator(
        do_mlm=dataset_args.do_mlm,
        mlm_probability=dataset_args.mlm_probability,
        pad_value=dataset_args.pad_value,
        mask_value=dataset_args.mask_value,
        max_length=dataset_args.max_length,
        sampling=dataset_args.sampling,
        keep_first_n_tokens=dataset_args.keep_first_n_tokens,
    )

    logger.info("Config: %s", config)

    # ---- Model (finetune from pretrained) ----
    model = DeSCORTForContrastiveLM.from_pretrained(
        finetune_args.pretrianed_model_path
    )

    # ---- Dataset ----
    logger.info("%s", dataset_args.train_dataset_path)
    train_dataset = load_from_disk(dataset_args.train_dataset_path)
    train_dataset = train_dataset.shuffle(seed=training_args.seed)

    logger.info("%s", train_dataset)
    logger.info(
        "Load train dataset success! Total training cells (%s,d)!",
        len(train_dataset),
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[LossLoggingCallback()],
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(os.path.join(training_args.output_dir, "models"))
    trainer.save_state()


if __name__ == "__main__":
    main()