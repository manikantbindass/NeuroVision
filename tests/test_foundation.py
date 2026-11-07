import json
import os
import random
import subprocess
import sys
import unittest

from neurovision.core.errors import ConfigurationError, NeuroVisionError
from neurovision.core.settings import LogLevel, Settings
from neurovision.pipeline import RunContext


class SettingsTests(unittest.TestCase):
    def test_defaults_and_overrides(self) -> None:
        self.assertEqual(Settings.from_env({}), Settings())
        settings = Settings.from_env(
            {"NEUROVISION_SEED": "4294967295", "NEUROVISION_LOG_LEVEL": " info "}
        )
        self.assertEqual(settings, Settings(seed=4294967295, log_level=LogLevel.INFO))
        self.assertEqual(Settings.from_env({"NEUROVISION_SEED": "000"}).seed, 0)

    def test_invalid_seed_and_level_raise_configuration_error(self) -> None:
        for seed in ("", "-1", "1.5", "4294967296", "9" * 5000, "４"):
            with self.subTest(length=len(seed)), self.assertRaises(ConfigurationError):
                Settings.from_env({"NEUROVISION_SEED": seed})
        with self.assertRaises(NeuroVisionError):
            Settings.from_env({"NEUROVISION_LOG_LEVEL": "verbose"})

    def test_seed_repeats_without_mutating_the_global_random_stream(self) -> None:
        seed = Settings.from_env({"NEUROVISION_SEED": "7"}).seed
        state = random.getstate()
        first = RunContext("first", seed).rng()
        second = RunContext("second", seed).rng()
        self.assertEqual(
            [first.random() for _ in range(4)], [second.random() for _ in range(4)]
        )
        self.assertEqual(random.getstate(), state)


class CliConfigurationTests(unittest.TestCase):
    def run_cli(
        self, *, seed: str = "42", level: str = "WARNING"
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "neurovision"],
            capture_output=True,
            text=True,
            check=False,
            env=os.environ | {"NEUROVISION_SEED": seed, "NEUROVISION_LOG_LEVEL": level},
        )

    def test_info_emits_json_to_stderr_only(self) -> None:
        baseline = self.run_cli()
        verbose = self.run_cli(level="INFO")
        self.assertEqual(baseline.returncode, 0)
        self.assertEqual(baseline.stderr, "")
        self.assertEqual(verbose.returncode, 0)
        self.assertEqual(verbose.stdout, baseline.stdout)
        event = json.loads(verbose.stderr)
        self.assertEqual(set(event), {"timestamp", "level", "logger", "message"})
        self.assertEqual(event["level"], "INFO")
        self.assertEqual(event["message"], "pipeline_catalog_listed")

    def test_invalid_config_exits_without_echoing_value(self) -> None:
        bad_seed = "sensitive-test-value"
        result = self.run_cli(seed=bad_seed)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertNotIn(bad_seed, result.stderr)
        event = json.loads(result.stderr)
        self.assertEqual(event["level"], "ERROR")
        self.assertIn("NEUROVISION_SEED", event["message"])
