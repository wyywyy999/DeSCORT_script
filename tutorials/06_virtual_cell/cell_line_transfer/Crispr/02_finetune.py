# Support both an installed package and direct execution from this checkout.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path(__file__).resolve().parents[4] / "src"
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))

import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch



from datasets import load_from_disk
from transformers import HfArgumentParser, TrainingArguments, set_seed

from descort.arguments import DataArguments, FinetuneDataArguments, ModelArguments
from descort.data.collator import PerturbPredictionDataCollator
from descort.data.sampler import GeneGroupedTrainer
from descort.model.configuration_descort import DeSCORTConfig
from descort.model.modeling_descort import DeSCORTForPerturbPrediction


logger = logging.getLogger(__name__)

from descort.vc.finetune import VirtualCellArguments


def main():
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            FinetuneDataArguments,
            VirtualCellArguments,
            TrainingArguments,
        )
    )
    (
        model_args,
        dataset_args,
        finetune_args,
        virtual_cell_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    model_args: ModelArguments
    dataset_args: DataArguments
    finetune_args: FinetuneDataArguments
    virtual_cell_args: VirtualCellArguments
    training_args: TrainingArguments

    if model_args.gene_contrastive_weight > 0 and virtual_cell_args.genes_per_batch < 2:
        raise ValueError('gene_contrastive_weight requires genes_per_batch >= 2')
    if virtual_cell_args.genes_per_batch > 1:
        expected_batch = virtual_cell_args.genes_per_batch * virtual_cell_args.wells_per_gene
        if training_args.per_device_train_batch_size != expected_batch:
            raise ValueError('per_device_train_batch_size must equal genes_per_batch * wells_per_gene')

    # This tutorial predicts the absolute post-perturbation morphology profile.
    # Both the original MLM head and the gene-conditioned FiLM head are valid;
    # FiLM is a conditioning mechanism and does not imply residual prediction.
    # Keep residual/global-delta branches loadable for old checkpoints, but do
    # not allow this entry point to silently switch to delta semantics.
    incompatible_absolute_prediction_options = []
    if model_args.predict_residual:
        incompatible_absolute_prediction_options.append("predict_residual=True")
    if model_args.prediction_head_type not in {"mlm", "gene_film"}:
        incompatible_absolute_prediction_options.append(
            f"prediction_head_type={model_args.prediction_head_type!r}"
        )
    if model_args.use_global_delta_prior:
        incompatible_absolute_prediction_options.append("use_global_delta_prior=True")
    if incompatible_absolute_prediction_options:
        raise ValueError(
            "06_vc requires direct absolute prediction; disable: "
            + ", ".join(incompatible_absolute_prediction_options)
        )

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

    # ---- Logging ----
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
    logger.info("Virtual-cell parameters %s", virtual_cell_args)
    logger.info(
        "Prediction semantics: %s directly outputs absolute perturbed features",
        model_args.prediction_head_type,
    )

    set_seed(training_args.seed)

    # ---- Model config ----
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
        use_sdpa=model_args.use_sdpa,
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
        esm2_input_dim=model_args.esm2_input_dim,
        esm2_projection_dim=model_args.esm2_projection_dim,
        normalize_esm2_embedding=model_args.normalize_esm2_embedding,
        # DeSCORTForPerturbPrediction
        mean_weight=model_args.mean_weight,
        direction_weight=model_args.direction_weight,
        marginal_quantile_weight=model_args.marginal_quantile_weight,
        centered_quantile_weight=model_args.centered_quantile_weight,
        sliced_wasserstein_weight=model_args.sliced_wasserstein_weight,
        sliced_wasserstein_num_projections=model_args.sliced_wasserstein_num_projections,
        sliced_wasserstein_projection_seed=model_args.sliced_wasserstein_projection_seed,
        variance_weight=model_args.variance_weight,
        gene_contrastive_weight=model_args.gene_contrastive_weight,
        gene_contrastive_temperature=model_args.gene_contrastive_temperature,
        predict_residual=model_args.predict_residual,
        use_global_delta_prior=model_args.use_global_delta_prior,
        prediction_head_type=model_args.prediction_head_type,
    )
    logger.info("Config: %s", config)

    # ---- Collator ----
    collator = PerturbPredictionDataCollator(
        max_length=dataset_args.max_length,
        sampling=dataset_args.sampling,
        pad_value=dataset_args.pad_value,
        keep_first_n_tokens=dataset_args.keep_first_n_tokens,
    )

    # ---- ESM2 embeddings and pretrained model ----
    with open(virtual_cell_args.index2esm2_pkl, "rb") as file:
        index2esm2_dict = pickle.load(file)

    model = DeSCORTForPerturbPrediction.from_pretrained(
        finetune_args.pretrianed_model_path,
        config=config,
        INDEX2ESM2_Dict=index2esm2_dict,
    )
    if virtual_cell_args.esm2_pca_path:
        pca_data = np.load(virtual_cell_args.esm2_pca_path)
        components = torch.from_numpy(pca_data["components"])
        pca_mean = torch.from_numpy(pca_data["mean"])
        input_projection = model.DeSCORT.ESM2_mlp[0]
        expected_shape = tuple(input_projection.weight.shape)
        if tuple(components.shape) != expected_shape:
            raise ValueError(
                f"PCA components have shape {tuple(components.shape)}, "
                f"expected {expected_shape}"
            )
        with torch.no_grad():
            input_projection.weight.copy_(
                components.to(dtype=input_projection.weight.dtype)
            )
            input_projection.bias.copy_(
                (-components @ pca_mean).to(dtype=input_projection.bias.dtype)
            )
        input_projection.requires_grad_(
            not virtual_cell_args.freeze_esm2_pca_projection
        )
        model.config.esm2_pca_path = virtual_cell_args.esm2_pca_path
        model.config.freeze_esm2_pca_projection = (
            virtual_cell_args.freeze_esm2_pca_projection
        )
        logger.info(
            "Initialized ESM2 projection from %s (frozen=%s)",
            virtual_cell_args.esm2_pca_path,
            virtual_cell_args.freeze_esm2_pca_projection,
        )

    # ---- Dataset ----
    logger.info("Loading train dataset from %s", dataset_args.train_dataset_path)
    train_dataset = load_from_disk(dataset_args.train_dataset_path)

    # For strict unseen-gene prediction, start from the shared morphology
    # response estimated only on the training split. The model then learns a
    # regularized ESM2-conditioned correction around this leakage-free prior.
    if config.use_global_delta_prior:
        numpy_dataset = train_dataset.with_format("numpy")
        global_delta_sum = None
        global_delta_count = 0
        for start in range(0, len(numpy_dataset), 2048):
            batch = numpy_dataset[start : min(start + 2048, len(numpy_dataset))]
            target = np.asarray(batch["Perturb_expression"], dtype=np.float64)
            control = np.asarray(batch["Control_expression"], dtype=np.float64)
            batch_sum = (target - control).sum(axis=0)
            global_delta_sum = (
                batch_sum
                if global_delta_sum is None
                else global_delta_sum + batch_sum
            )
            global_delta_count += len(target)
        if global_delta_sum is None or global_delta_count == 0:
            raise ValueError("Cannot initialize global delta from an empty dataset")
        global_delta = torch.from_numpy(
            (global_delta_sum / global_delta_count).astype(np.float32)
        )
        if global_delta.numel() > model.global_delta.numel():
            raise ValueError(
                f"Training global delta has {global_delta.numel()} features, "
                f"but model capacity is {model.global_delta.numel()}"
            )
        with torch.no_grad():
            model.global_delta.zero_()
            model.global_delta[: global_delta.numel()].copy_(global_delta)
        logger.info(
            "Initialized global delta from %s training rows (L2=%.6f)",
            f"{global_delta_count:,d}",
            float(global_delta.norm()),
        )
        if model_args.freeze_global_delta_prior:
            model.global_delta.requires_grad_(False)
            logger.info("Frozen training-derived global delta prior")

    if model_args.freeze_morphology_backbone:
        # Preserve the pretrained morphology representation for strict
        # unseen-gene transfer.  The ESM2 projection and GeneFiLM prediction
        # head remain trainable, so gene-conditioned residuals can still be
        # learned without rewriting the foundation encoder around 479 genes.
        for name, parameter in model.DeSCORT.named_parameters():
            if not name.startswith("ESM2_mlp"):
                parameter.requires_grad_(False)
        logger.info(
            "Frozen morphology backbone; ESM2 projection and prediction head remain trainable"
        )

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "Trainable parameters: %s / %s (%.2f%%)",
        f"{trainable_parameters:,d}",
        f"{total_parameters:,d}",
        100.0 * trainable_parameters / total_parameters,
    )

    train_dataset = train_dataset.shuffle(seed=training_args.seed)
    logger.info(
        "Load train dataset success! Total training cells: %s",
        f"{len(train_dataset):,d}",
    )

    # ---- Trainer ----
    trainer = GeneGroupedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        gene_column=virtual_cell_args.gene_column,
        genes_per_batch=virtual_cell_args.genes_per_batch,
        wells_per_gene=virtual_cell_args.wells_per_gene,
        gene_sampling_rounds=virtual_cell_args.gene_sampling_rounds,
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(os.path.join(training_args.output_dir, "models"))
    trainer.save_state()


if __name__ == "__main__":
    main()
