"""Dynamic test suite. Each test inherits from TestCase and emits a TestResult."""

from ggufscan.tests.base import Sample, TestCase, TestResult
from ggufscan.tests.backdoor import BackdoorTest
from ggufscan.tests.harmful_bias import HarmfulBiasTest
from ggufscan.tests.jailbreak import JailbreakTest

TEST_REGISTRY: dict[str, type[TestCase]] = {
    "jailbreak": JailbreakTest,
    "harmful_bias": HarmfulBiasTest,
    "backdoor": BackdoorTest,
}

__all__ = [
    "Sample", "TestCase", "TestResult",
    "JailbreakTest", "HarmfulBiasTest", "BackdoorTest",
    "TEST_REGISTRY",
]
