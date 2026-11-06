"""Shared interfaces for the NeuroVision Intelligence Pipeline."""

from dataclasses import dataclass
from enum import StrEnum
from random import Random
from typing import Protocol, TypeVar

RESEARCH_NOTICE = (
    "Research and educational use only. Not a medical device and not intended "
    "for clinical diagnosis or treatment."
)


class Stage(StrEnum):
    """Stage identifiers in the order of the reference pipeline."""

    VALIDATION = "validation"
    DEIDENTIFICATION_SAFEGUARDS = "deidentification-safeguards"
    PREPROCESSING = "preprocessing"
    REGISTRATION = "registration"
    NORMALIZATION = "normalization"
    QUALITY_CONTROL = "quality-control"
    SEGMENTATION = "segmentation"
    ENSEMBLE_MODEL_SELECTION = "ensemble-model-selection"
    UNCERTAINTY_ESTIMATION = "uncertainty-estimation"
    ANATOMICAL_MAPPING = "anatomical-mapping"
    QUANTIFICATION = "quantification"
    EXPLAINABILITY = "explainability"
    VISUALIZATION = "visualization"
    LONGITUDINAL_COMPARISON = "longitudinal-comparison"
    STRUCTURED_RESEARCH_REPORT = "structured-research-report"


@dataclass(frozen=True, slots=True)
class RunContext:
    """Shared experiment settings. The run ID is generated independently of subjects."""

    run_id: str
    seed: int

    def rng(self) -> Random:
        """Create a fresh deterministic Python random stream for this run."""
        return Random(self.seed)


StageInput = TypeVar("StageInput", contravariant=True)
StageOutput = TypeVar("StageOutput", covariant=True)


class PipelineComponent(Protocol[StageInput, StageOutput]):
    """A replaceable implementation of one stage with typed inputs and outputs."""

    @property
    def stage(self) -> Stage:
        """Identify the stage this component implements."""
        ...

    def run(self, inputs: StageInput, *, context: RunContext) -> StageOutput:
        """Produce the stage result for the given experiment context."""
        ...
