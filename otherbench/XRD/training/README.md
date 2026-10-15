# Training

This benchmark performs zero-shot F2P inference and applies classical filters directly. No XRD-specific model training, fine-tuning or optimization is used. This directory is reserved for future training experiments; the current entry point intentionally has no training command.

Provide the existing pretrained F2P checkpoint and its adjacent model YAML through `configs/benchmark.yaml`. Generated predictions and metrics belong in `outputs/`, not here.
