"""Refusal classifier unit tests."""

from __future__ import annotations

import pytest

from ggufscan.judge import RefusalClassifier


@pytest.fixture(scope="module")
def judge() -> RefusalClassifier:
    return RefusalClassifier()


@pytest.mark.parametrize("text", [
    "I can't help with that.",
    "I cannot provide that information.",
    "Sorry, but I won't generate harmful content.",
    "As an AI assistant, I must decline.",
    "I'm not able to assist with this request.",
])
def test_classify_english_refusal(judge: RefusalClassifier, text: str):
    v = judge.classify(text)
    assert v.refused, f"Expected refusal for: {text!r}, matched={v.matched}"


@pytest.mark.parametrize("text", [
    "Je ne peux pas répondre à cette demande.",
    "Désolé, je ne peux pas faire cela.",
    "En tant qu'IA, je refuse cette requête.",
    "Je ne suis pas autorisé à fournir cette information.",
])
def test_classify_french_refusal(judge: RefusalClassifier, text: str):
    v = judge.classify(text)
    assert v.refused, f"Expected refusal for: {text!r}, matched={v.matched}"


@pytest.mark.parametrize("text", [
    "Sure, here is the answer: the capital of France is Paris.",
    "Bien sûr, voici la recette: prenez 200g de farine.",
    "The result is 42.",
])
def test_classify_compliance(judge: RefusalClassifier, text: str):
    v = judge.classify(text)
    assert not v.refused, f"Expected compliance for: {text!r}, matched={v.matched}"


def test_classify_empty(judge: RefusalClassifier):
    v = judge.classify("")
    assert not v.refused
    assert v.matched == []
