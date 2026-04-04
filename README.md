# LLM Fine-Tuning with QLoRA and DPO

This project contains a single end-to-end notebook for instruction tuning a large language model using:
- QLoRA (4-bit quantization + LoRA adapters)
- Supervised Fine-Tuning (SFT)
- Optional Direct Preference Optimization (DPO)

Code Notebook:
- LLM_FineTuning_Industry_Grade_(1).ipynb

## What the notebook does

1. Checks GPU availability
2. Installs required libraries
3. Loads experiment configuration
4. Loads and normalizes dataset format
5. Loads tokenizer and base model (4-bit)
6. Attaches LoRA adapters
7. Runs baseline generation
8. Trains with SFT
9. Optionally runs DPO
10. Computes evaluation metrics
11. Saves comparison report and plots
12. Optionally pushes model to Hugging Face Hub
13. Launches a Gradio side-by-side demo

## Default setup in notebook

- Model: HuggingFaceH4/zephyr-7b-beta
- Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset
- LoRA rank: 16
- Epochs: 1
- Max train samples: 200
- Max eval samples: 5

These can be changed in the Config cell.

## Main dependencies

- transformers
- peft
- trl
- bitsandbytes
- datasets
- accelerate
- evaluate
- rouge_score
- bert_score
- gradio
- huggingface_hub

## Outputs generated

- eval_results.json
  - Config snapshot
  - Baseline metrics and predictions
  - Post-SFT metrics and predictions
- metrics_comparison.png
- loss_curve.png

## How to use

1. Open the notebook.
2. Run cells from top to bottom.
3. Update the Config section before training if needed.
4. Review metrics and saved artifacts.
5. Enable push to Hub only after setting a valid HF token.

## Notes

- This workflow is designed for Colab or Kaggle T4 class GPUs.
- DPO is optional and controlled by RUN_DPO in config.
- For more reliable evaluation, increase eval sample size and use consistent settings across runs.
