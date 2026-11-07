# Ensemble intelligence and NeuroConfidence Map

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

**Experimental.** The ensemble subsystem supports weighted soft voting, reversible flip test-time augmentation (TTA), and Monte Carlo dropout experiments for supported modules. Calculations are verified with synthetic probabilities. **Benchmark pending.** MRI performance and calibration have not been measured.

## Configure compatible members

Install the [training environment](../development/BASELINE_TRAINING.md). Include the `zoo` extra for MONAI models or sliding-window inference.

```python
from pathlib import Path

from neurovision.inference.engine import InferenceEngine
from neurovision.inference.ensemble import EnsembleConfig, EnsembleEngine, EnsembleMember
from neurovision.models.checkpoints import CheckpointManager

members = tuple(
    EnsembleMember(
        identifier=identifier,
        engine=InferenceEngine(CheckpointManager.load_model(Path(checkpoint))),
        channels=("t1", "t1ce", "t2", "flair"),
        labels=("background", "region_1", "region_2"),
    )
    for identifier, checkpoint in (
        ("experiment-a/checkpoint-digest", "artifacts/experiment-a/best.pt"),
        ("experiment-b/checkpoint-digest", "artifacts/experiment-b/best.pt"),
    )
)
config = EnsembleConfig.from_json(Path("configs/ensemble-research.json").read_text())
ensemble = EnsembleEngine(members, config)
# Supply an aligned finite float32 (N, C, D, H, W) tensor in the declared channel order.
# prediction = ensemble.predict(inputs)
```

Adapt the declared channel and label schemas to the trained checkpoints. Each member needs a unique research identifier, ideally tied to an immutable checkpoint digest. The engine verifies tensor dimensions and matching schema declarations. The caller must supply the same registered physical grid and label meaning for every member. Checkpoint metadata alone cannot establish anatomical alignment or correct channel semantics.

Weights align with member order, must be finite and nonnegative, and are normalized internally. Zero-weight members are excluded from inference. With equal weights, soft voting is probability averaging. Weighted soft voting averages probabilities and selects the largest class probability. Ties select the first class index. Weights should be chosen on separate development data, then fixed before evaluating the research test set.

## Sampling and uncertainty definitions

| Output | Definition |
| --- | --- |
| Mean probability | Average each member's aligned TTA/dropout samples, then take the weighted mean across members. |
| Segmentation | Argmax of that mean probability. |
| Voxel confidence | Maximum mean class probability, between zero and one. |
| Predictive entropy | Shannon entropy of mean class probabilities divided by `log(number_of_classes)`, between zero and one. |
| Ensemble variance | Weighted population variance across member mean probabilities, separately for each class and voxel. |
| Sample variance | Weighted population variance across all aligned model, TTA, and dropout samples. It combines these sources of variation. |
| Model disagreement | One minus the largest weighted hard-vote fraction across member mean predictions. |
| High-confidence mask | Confidence is at least its threshold, entropy is at most its threshold, and model disagreement is at most its threshold. |

The configuration specifies sorted spatial-axis tuples in `tta_flips`, with the identity `[]` always present. For tensor `(N, C, D, H, W)`, spatial axis zero is D. Each flip is inverted on its probability map before averaging. A flip preserves array shape but may alter anatomical meaning. Use label-preserving augmentations for the task, especially when left and right are different classes. Class permutations for laterality-specific flips are outside this phase.

Set `mc_samples` to 2 through 128 for Monte Carlo dropout, or zero for deterministic inference. The engine enables explicit PyTorch Dropout/AlphaDropout modules with probabilities strictly between zero and one while keeping other modules in evaluation mode. It restores module flags and the relevant random-generator states afterward. Native baseline U-Nets have no dropout, so MC requests for them fail clearly. Functional dropout and custom stochastic layers are outside this adapter. Use models trained with appropriate dropout and assess the experiment's behavior. [MC dropout research](https://arxiv.org/abs/1506.02142) · [PyTorch evaluation-mode guidance](https://docs.pytorch.org/docs/stable/notes/autograd.html)

The output records configuration, normalized weights, member model definitions, declared schemas, device and precision, and the exact dropout seeds. Defaults limit the ensemble to 16 declared members, 256 model/TTA/dropout inference calls, and 16 million class-probability elements per output grid. Each sliding-window inference call can run multiple network forwards. Streaming moments retain accumulators instead of all samples, while accumulators, outputs, active model parameters, and temporary activations still require memory.

These statistics are research signals. Low variance or agreement can occur when all models make the same error. A high maximum probability can be poorly calibrated, especially after distribution shifts. Thresholds are adjustable experiment settings and do not certify correctness or clinical certainty. [Deep ensemble research](https://arxiv.org/abs/1612.01474)

## NeuroConfidence Map

Generate an interactive synthetic preview:

```text
python -m neurovision.uncertainty.example --output artifacts/neuroconfidence-synthetic
```

Open `artifacts/neuroconfidence-synthetic/neuroconfidence.html` locally. The example uses two analytic probability phantoms with different intensity thresholds. It uses no learned model or human MRI.

For an ensemble result, call `write_neuroconfidence_html(inputs, prediction, path)` from `neurovision.uncertainty.visual`. The self-contained viewer shows the image, segmentation, and NeuroConfidence overlay with a slice slider, probability/entropy/disagreement thresholds, and class legend. Cyan marks the high-confidence category. Amber marks the low-confidence or uncertain category. Background visibility is configurable. The viewer is a local research utility. A hosted web application is planned for later phases.

The preview samples a bounded grid for display and uses float32 map values for thresholding. It retains the original tensor-axis slice indices and avoids anatomical orientation claims. The generated HTML includes image pixels and must remain in approved research storage. It uses no external scripts or network requests. Offline JavaScript tests verify controls and canvas data. Browser layout verification remains pending because the available browser policy blocks local-file previews.
