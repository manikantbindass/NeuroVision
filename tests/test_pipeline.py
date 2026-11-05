import subprocess
import sys
import unittest

from neurovision.pipeline import PipelineComponent, RunContext, Stage

NOTICE = (
    "Research and educational use only. Not a medical device and not intended "
    "for clinical diagnosis or treatment."
)


class PipelineTests(unittest.TestCase):
    def test_replacement_uses_same_component_contract(self) -> None:
        class Identity:
            stage = Stage.NORMALIZATION

            def run(self, inputs: float, *, context: RunContext) -> float:
                return inputs

        class Scale:
            stage = Stage.NORMALIZATION

            def run(self, inputs: float, *, context: RunContext) -> float:
                return inputs / 100

        context = RunContext(run_id="synthetic-run", seed=0)
        components: tuple[PipelineComponent[float, float], ...] = (Identity(), Scale())
        results = [component.run(25.0, context=context) for component in components]
        self.assertEqual(results, [25.0, 0.25])

    def test_module_prints_notice_and_reference_order(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "neurovision"],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = result.stdout.splitlines()
        self.assertEqual(lines[0], NOTICE)
        self.assertEqual(
            [line.split(". ", 1)[1] for line in lines[2:]],
            [
                "validation",
                "deidentification-safeguards",
                "preprocessing",
                "registration",
                "normalization",
                "quality-control",
                "segmentation",
                "ensemble-model-selection",
                "uncertainty-estimation",
                "anatomical-mapping",
                "quantification",
                "explainability",
                "visualization",
                "longitudinal-comparison",
                "structured-research-report",
            ],
        )

    def test_help_displays_notice(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "neurovision", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn(NOTICE, " ".join(result.stdout.split()))
