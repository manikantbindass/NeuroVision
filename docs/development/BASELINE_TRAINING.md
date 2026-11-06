# Baseline segmentation training

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

**MRI benchmark status: Not yet benchmarked.** The bundled recipes train on generated ellipsoids with noise. They exercise the code, checkpointing, and reporting on CPU. Their scores describe synthetic inputs only.

## Run the examples

Use the virtual environment interpreter from the [README](../../README.md). The official CPU wheel is sufficient. Install a compatible PyTorch CUDA build separately if your experiment requires it. [PyTorch installation guide](https://pytorch.org/get-started/locally/)

```text
python -m pip install --no-deps "torch>=2.8,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[training]"
python -m neurovision.training.example --config configs/baseline-2d.json --output artifacts/baseline-2d
python -m neurovision.training.example --config configs/baseline-3d.json --output artifacts/baseline-3d
```

The first install command selects the CPU wheel. The second resolves its dependencies, including a patched setuptools version. Each recipe generates separate training and validation cases, sets one CPU thread, and runs two epochs. Start in an empty output directory. To continue from `last.pt`, increase only the epoch budget and repeat the command with `--resume`. Reuse the same dataset, environment, and configuration. A run that has already reached its early stopping threshold retains that decision. [Setuptools advisory](https://github.com/pypa/setuptools/security/advisories/GHSA-xmj3-9gf7-h52w)

## Model and training configuration

The 2D and 3D U-Nets use padded double convolutions with ReLU, max pooling, a bottleneck, transposed convolutions, and skip concatenation. A pointwise head produces one logit per class, including background at index zero. The 3D model uses volumetric operations throughout. [U-Net paper](https://arxiv.org/abs/1505.04597) · [3D U-Net paper](https://arxiv.org/abs/1606.06650)

Inputs use `(N, C, H, W)` or `(N, C, D, H, W)`. Each spatial dimension must be divisible by `2 ** (len(features) - 1)`. Labels use the same spatial grid without a channel axis and contain integer class IDs in `0 .. num_classes - 1`. `case_tensors` in `neurovision.training.mri` checks aligned MRI modalities and labels, makes axial slices for 2D or a volume for 3D, and returns spacing in tensor axis order. It requires the `imaging` extra. Prepare equal-sized batches with [preprocessing](PREPROCESSING.md) and use nearest-neighbor sampling for labels. Define an explicit mapping for any dataset with noncontiguous label IDs.

Use a subject-level split before creating slices or patches. Pass separate replayable PyTorch datasets yielding CPU `(float32_image, int64_label)` pairs to `fit`. Provide an opaque dataset version identifier that changes when the splits or preprocessing change. For 2D training, keep slices from the same subject in the same split. Keep augmentation in training only, and consider laterality before enabling spatial flips.

- **Loss.** Use soft Dice, cross-entropy, or a weighted sum. Set either loss weight to zero to select the other. Dice averages across samples and selected classes, excludes background by default, and uses configurable positive smoothing. Probability reductions run in float32.
- **Optimization.** AdamW updates the model. ReduceLROnPlateau and early stopping both monitor validation loss. Configure patience, minimum improvement, learning rate, factor, and floor in the JSON recipe. `best.pt` tracks improvement under that criterion.
- **Precision.** Use `fp32` on CPU or CUDA, `bf16` with autocast on supported CPU or CUDA, or `fp16` with autocast and gradient scaling on CUDA. CPU FP32 and BF16 were exercised locally. [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)
- **Reproducibility.** The runner seeds Python, NumPy, and PyTorch, uses single-process data loading, and seeds shuffle and augmentation separately by epoch. Its deterministic mode configures process-wide PyTorch settings. Repeat runs and epoch-boundary resume depend on unchanged data, hardware, software, and experiment settings. [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- **Artifacts.** `last.pt` stores model, optimizer, scheduler, scaler, and early stopping state. `history.json` records actual validation measurements, epoch losses, environment, and configuration. Use one writer per output directory and keep artifacts outside Git. Load checkpoints from trusted local runs. The reader uses `weights_only=True`.

## Metric conventions

The validation loop evaluates foreground classes by default. It averages defined per-sample values and reports the number of contributing samples per metric and class. In 2D, each sample is a slice. In 3D, each sample is the supplied volume or patch. Specify the aggregation unit when reporting results.

| Metric | Convention |
| --- | --- |
| Dice | Twice the true-positive count divided by the sum of predicted and reference positives. |
| IoU | Intersection divided by union. |
| Precision | True positives divided by predicted positives. |
| Recall / sensitivity | True positives divided by reference positives. |
| Specificity | True negatives divided by reference negatives. |
| Hausdorff | Maximum of the two directed nearest-boundary distances. |
| HD95 | Maximum of the two directed 95th percentiles. |
| ASSD | Mean of the pooled directed nearest-boundary distances, weighted by sampled boundary voxel count. |

Zero denominators return undefined. If either mask is empty, all surface distances are undefined. JSON represents undefined results as `null`. Surface measurements use face-connected voxel boundaries and Euclidean distances in millimetres on orthogonal grids. In the training loop, enable them with explicit `validation_spacing_mm` only when the same tensor-axis spacing applies to every validation sample. For variable spacing, call `segmentation_metrics` separately per case with the appropriate spacing. Surface metrics for 2D are in-plane distances.
