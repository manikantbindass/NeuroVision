# Model zoo and comparison protocol

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

**Benchmark pending.** MRI accuracy and comparative speed have not been measured. Architectures in this phase can be instantiated, trained, checkpointed, and used for tensor inference. Repository checks exercise generated data. Select an architecture through an experiment configuration and report results on a documented research cohort before making performance claims.

## Architecture choices

| Architecture | Family and structure | Memory and compute considerations | Research trade-offs to evaluate |
| --- | --- | --- | --- |
| Native U-Net | CNN encoder and decoder with skip concatenation, in 2D or 3D | Feature width, depth, batch size, and grid size determine activation storage. Volumetric convolutions process many more voxels than slices. | An auditable baseline for assessing whether added complexity helps a specific task. |
| SegResNet | Residual CNN with a convolutional encoder and decoder | Residual feature maps and the number of blocks affect activation storage. Downsampling reduces the grids processed by deeper blocks. | Residual learning offers a different optimization path. Measure its effect under the same data split and training budget. |
| Attention U-Net | CNN U-Net with attention gates on skip features | Gates introduce additional operations and feature maps. The effect on runtime depends on the chosen widths and input. | Gating can change how skip features contribute. Compare against a matched CNN baseline. |
| UNETR | Hybrid with a transformer encoder and convolutional decoding | Patch tokens attend globally. Attention cost can grow quadratically with the token count. Hidden width, patch grid, and decoder activations affect memory. | Global token interactions add context while changing the amount of data and optimization needed. |
| Swin UNETR | Hybrid with hierarchical shifted-window transformer encoding and convolutional decoding | Windowed attention restricts each attention calculation. Hierarchical stages, window size, feature width, and the 3D grid still affect memory. Activation checkpointing trades recomputation for memory. | Local windows and cross-window shifts provide a different context mechanism. Measure results and runtime on the same task. |

Advanced networks use MONAI 1.6.1 and its architecture implementations. Einops supports transformer tensor rearrangements. The native U-Net stays available through the factory when MONAI is absent. [MONAI source and architecture implementations](https://github.com/Project-MONAI/MONAI/tree/1.6.1/monai/networks/nets) · [UNETR paper](https://arxiv.org/abs/2103.10504) · [Swin UNETR paper](https://arxiv.org/abs/2201.01266) · [Attention U-Net paper](https://arxiv.org/abs/1804.03999)

## Configure and train

Install the desired PyTorch build first, then the project extras. For CPU:

```text
python -m pip install --upgrade pip
python -m pip install --no-deps "torch>=2.8,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[training,zoo]"
python -m neurovision.training.example --config configs/zoo-segresnet.json --output artifacts/zoo-segresnet
python -m neurovision.training.example --config configs/zoo-attention-unet.json --output artifacts/zoo-attention
python -m neurovision.training.example --config configs/zoo-unetr.json --size 32 --output artifacts/zoo-unetr
python -m neurovision.training.example --config configs/zoo-swin-unetr.json --size 64 --output artifacts/zoo-swin
```

The bundled configurations reduce widths or transformer dimensions to make synthetic checks practical. They are starting points for software experiments. The UNETR recipe uses a fixed 32-cubed patch, and the Swin recipe uses 64-cubed input. The latter can still consume substantial CPU memory and time.

The experiment `model` section supplies `architecture`, `spatial_dims`, `in_channels`, `num_classes`, and architecture-specific `parameters`. Supported architecture names are `unet`, `segresnet`, `attention_unet`, `unetr`, and `swin_unetr`. Unknown architecture names and unsupported parameter keys fail configuration validation. The original baseline configuration format remains accepted.

| Architecture | Exposed parameters | Input contract |
| --- | --- | --- |
| U-Net | `features` | Axes divisible by the encoder pooling factor. |
| SegResNet | `init_filters`, `blocks_down`, `blocks_up`, `dropout_prob` | Axes divisible by the encoder stride product. Default MONAI group normalization also constrains filter width. |
| Attention U-Net | `channels`, `strides`, `dropout` | Axes divisible by the stride product and at least twice that product for batch-one normalization. |
| UNETR | `img_size`, `feature_size`, `hidden_size`, `mlp_dim`, `num_heads`, `dropout_rate`, `qkv_bias` | The configured grid, with each axis a multiple of 16 and at least 32. |
| Swin UNETR | `feature_size`, `depths`, `num_heads`, `window_size`, `drop_rate`, `attn_drop_rate`, `dropout_path_rate`, `use_checkpoint` | Axes divisible by 32 and at least 64 for the supported instance-normalized path. |

All factory models return same-grid class logits through `BaseSegmentationModel`. The existing training loop handles the selected network, losses, AMP, checkpoints, and validation. `ModelRegistry` registers explicit builders for local extensions. Pass a factory using that registry to `fit(..., factory=factory)` and to checkpoint reconstruction when using a custom architecture. Configuration cannot import arbitrary code. This registry describes architectures. Artifact storage and model promotion remain later MLOps work.

## Inference and checkpoint identity

```python
from pathlib import Path
import torch

from neurovision.inference.engine import InferenceEngine
from neurovision.models.checkpoints import CheckpointManager
from neurovision.models.factory import ModelFactory
from neurovision.training.config import load_experiment

experiment = load_experiment(Path("configs/zoo-unetr.json"))
model = ModelFactory().create(experiment.model)
inputs = torch.zeros(1, 1, 32, 32, 32)
prediction = InferenceEngine(model).predict(inputs)

digest = CheckpointManager.export(Path("artifacts/untrained-unetr.pt"), model)
restored = CheckpointManager.load_model(Path("artifacts/untrained-unetr.pt"), sha256=digest)
```

The example exports randomly initialized weights. For a trained model, load a `best.pt` or `last.pt` produced by the current trainer, or export the trained instance. A model artifact declares its construction configuration and stores a state dictionary. The manager uses restricted checkpoint loading, validates tensor names, shapes, dtypes, and finite values, and optionally verifies the file digest. Use trusted local artifacts and compatible backend versions. A matching architecture configuration does not define modality order or label meaning, which must accompany the experiment record.

`InferenceEngine` returns CPU logits, probabilities, and integer labels. It owns the model's selected device for its lifetime, uses evaluation mode during prediction, and restores prior training flags afterward. Give each worker its own engine. Inputs are finite float32 tensors, with configurable input and class-wise output element limits. Use `predict(inputs, roi_size=(32, 32, 32))` for a larger UNETR input. Sliding-window inference uses MONAI Gaussian blending with one window per device batch and accumulates logits on CPU. The ROI must meet the model contract. Element limits cover the input and logits grid, while temporary activations and additional probability buffers also need memory.

## Compare resource use

`parameter_report(model)` counts total and trainable parameters and exact parameter and buffer bytes for their current dtypes. These values cover stored tensors. Training also needs gradients, optimizer state, activations, temporary operations, and allocator overhead.

These counts were obtained from the bundled configurations with MONAI 1.6.1, PyTorch 2.15.0, and float32 parameters. The configurations deliberately differ in width and depth. Counts describe those configurations and provide no accuracy or speed ranking.

| Configuration | Parameters | Parameter bytes | Buffer bytes |
| --- | --- | --- | --- |
| `baseline-2d` | 1,650 | 6,600 | 0 |
| `baseline-3d` | 4,730 | 18,920 | 0 |
| `zoo-segresnet` | 108,346 | 433,384 | 0 |
| `zoo-attention-unet` | 22,496 | 89,984 | 768 |
| `zoo-unetr` | 600,490 | 2,401,960 | 0 |
| `zoo-swin-unetr` | 3,746,549 | 14,986,196 | 131,072 |

`profile_forward(model, example, warmup=2, repeats=5, count_flops=True)` measures resident-input evaluation forwards. It records the shape, dtype, device, threads, PyTorch version, platform, and median elapsed time. CUDA timings synchronize the device. PyTorch's FLOP counter covers its supported operator formulas, so the report labels the count as partial and lists counted operators. A graph that cannot be counted reports the estimate as unavailable.

```python
from neurovision.models.reporting import parameter_report, profile_forward

storage = parameter_report(model)
profile = profile_forward(model, inputs, warmup=2, repeats=5, count_flops=True)
print(storage)
print(profile.as_dict())
```

For a fair comparison, use the same cohort, spatial grid, modality channels, labels, preprocessing, batch size, dtype, device, and evaluation protocol. Record parameter counts, measured peak memory, warm-up behavior, forward time, and end-to-end latency separately. Compare quality under declared training budgets and subject-level splits. The repository currently publishes no architecture ranking or measured memory/speed table.
