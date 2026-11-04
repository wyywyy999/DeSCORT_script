# Representation Fine-tuning and Embedding Extraction

Input and output: a pretrained model and task-specific Parquet files → a fine-tuned model and embedding Parquet files.

Run `01_process_data.py` → `02_finetune.py` → `03_calculate_emb.py` in order. The embedding extraction entry contains its own `parse_args()` and `main()` functions and supports CPU execution. When using an existing checkpoint, explicitly provide all vocabularies and the selected feature list. Use a representation model checkpoint for embedding extraction rather than a perturbation-prediction model.

```bash
bash tutorials/03_finetune/run.sh --help
```

To run the wrapper from any working directory, use its absolute path. Refer to each script's `--help` output for supported arguments. Update server-specific paths in the experiment Bash scripts to match your environment.
