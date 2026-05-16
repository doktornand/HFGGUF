"""GGUF Security Scanner — Gradio UI.

Two tabs:
  - Static Scan (v1.0 baseline OWASP pattern matcher).
  - Dynamic Scan (live llama.cpp probing: jailbreak / harmful+bias / backdoor).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ggufscan.parser import parse as parse_gguf
from ggufscan.report import to_markdown
from ggufscan.static_scan import StaticReport, StaticScanner
from ggufscan.tests import TEST_REGISTRY
from ggufscan.tests.base import TestResult


# =============================================================================
# STATIC TAB
# =============================================================================

def _resolve_path(file: Any) -> tuple[str, str]:
    if hasattr(file, "name"):
        return file.name, getattr(file, "original_name", file.name)
    if isinstance(file, str):
        return file, Path(file).name
    raise ValueError("Format de fichier non reconnu")


def _static_visualizations(report: StaticReport):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Score de Sécurité", "Quantization Types",
                        "Vulnérabilités par Niveau", "Top 10 Tenseurs"),
        specs=[[{"type": "indicator"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "bar"}]],
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=report.security_score,
            title={"text": "Score Global"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 40], "color": "firebrick"},
                    {"range": [40, 60], "color": "orange"},
                    {"range": [60, 80], "color": "gold"},
                    {"range": [80, 100], "color": "forestgreen"},
                ],
            },
        ),
        row=1, col=1,
    )

    quant_types = report.analysis["structure"]["quantization_types"]
    if quant_types:
        fig.add_trace(
            go.Pie(labels=quant_types, hole=0.4, marker=dict(colors=px.colors.qualitative.Set3)),
            row=1, col=2,
        )

    risk_counts = defaultdict(int)
    for v in report.vulnerabilities:
        risk_counts[v["risk_level"]] += 1
    risk_labels = ["Critique", "Élevé", "Moyen", "Bas"]
    risk_order = ["critical", "high", "medium", "low"]
    colors = ["#8B0000", "#DC143C", "#FF8C00", "#FFD700"]
    fig.add_trace(
        go.Bar(
            x=risk_labels, y=[risk_counts[r] for r in risk_order],
            marker_color=colors, text=[risk_counts[r] for r in risk_order],
            textposition="auto",
        ),
        row=2, col=1,
    )

    top = sorted(report.tensors.items(), key=lambda x: x[1]["size_mb"], reverse=True)[:10]
    fig.add_trace(
        go.Bar(
            x=[n[:30] + "..." if len(n) > 30 else n for n, _ in top],
            y=[info["size_mb"] for _, info in top],
            marker_color=["coral" if info["is_quantized"] else "lightblue" for _, info in top],
            text=[f"{info['size_mb']:.1f} MB" for _, info in top],
            textposition="outside",
        ),
        row=2, col=2,
    )

    fig.update_layout(
        title_text=f"Analyse — {report.filename[:50]}",
        showlegend=True, height=800, template="plotly_white",
    )
    fig.update_xaxes(tickangle=45, row=2, col=2)
    return fig


def analyze_static(file):
    if file is None:
        return "⚠️ Sélectionnez un fichier GGUF.", None
    try:
        path, name = _resolve_path(file)
        parsed = parse_gguf(path, name)
        report = StaticScanner(parsed).scan()
        md = to_markdown(report, [])
        viz = _static_visualizations(report)
        return md, viz
    except Exception as e:  # noqa: BLE001
        return f"❌ Erreur: {e}", None


# =============================================================================
# DYNAMIC TAB
# =============================================================================

def _dynamic_visualizations(results: list[TestResult]):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Score par test (vs threshold)", "Backdoor — distances cosinus"),
        specs=[[{"type": "bar"}, {"type": "histogram"}]],
    )

    names = [r.name for r in results]
    scores = [r.score for r in results]
    thresholds = [r.threshold for r in results]
    colors = ["#2ca02c" if r.passed else "#d62728" for r in results]

    fig.add_trace(go.Bar(name="Score", x=names, y=scores, marker_color=colors,
                         text=[f"{s:.2%}" for s in scores], textposition="outside"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(name="Threshold", x=names, y=thresholds, mode="markers",
                             marker=dict(symbol="line-ew-open", size=30, line=dict(width=3))),
                  row=1, col=1)

    backdoor = next((r for r in results if r.name == "backdoor"), None)
    if backdoor:
        dists = [s.metadata.get("cosine_distance") for s in backdoor.samples
                 if "cosine_distance" in s.metadata]
        if dists:
            fig.add_trace(go.Histogram(x=dists, nbinsx=20, marker_color="indigo"),
                          row=1, col=2)

    fig.update_layout(height=500, showlegend=True, template="plotly_white")
    fig.update_yaxes(range=[0, 1.05], row=1, col=1)
    return fig


def analyze_dynamic(file, selected_tests, n_ctx, tensor_split_str, n_gpu_layers,
                    max_tokens, temperature, quick, seed, progress=gr.Progress()):
    if file is None:
        return "⚠️ Sélectionnez un fichier GGUF.", None
    if not selected_tests:
        return "⚠️ Sélectionnez au moins un test.", None

    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return ("❌ `llama-cpp-python` non installé.\n\n"
                "```bash\nCMAKE_ARGS=\"-DGGML_CUDA=on\" "
                "pip install llama-cpp-python --force-reinstall --no-cache-dir\n```"), None

    try:
        path, name = _resolve_path(file)
        progress(0.05, desc="Parsing GGUF...")
        parsed = parse_gguf(path, name)
        static_report = StaticScanner(parsed).scan()

        from ggufscan.inference import ModelHandle

        split = [float(x) for x in tensor_split_str.split(",")] if tensor_split_str.strip() else None
        chat_kwargs: dict[str, Any] = {
            "n_ctx": int(n_ctx), "n_gpu_layers": int(n_gpu_layers),
            "tensor_split": split, "seed": int(seed),
        }

        tests = {t: TEST_REGISTRY[t]() for t in selected_tests}
        needs_embedding = any(t.needs_embedding for t in tests.values())

        progress(0.10, desc="Pass 1/2 — chargement chat handle...")
        collected = {}
        with ModelHandle(parsed, embedding=False, **chat_kwargs) as chat:
            step = 0.65 / max(len(tests), 1)
            cur = 0.15
            for tname, tcase in tests.items():
                progress(cur, desc=f"Pass 1 — collecting {tname}...")
                collected[tname] = tcase.collect(
                    chat, quick=bool(quick),
                    max_tokens=int(max_tokens), temperature=float(temperature),
                    seed=int(seed),
                )
                cur += step

        results: list[TestResult] = []
        if needs_embedding:
            progress(0.80, desc="Pass 2/2 — chargement embedding handle...")
            with ModelHandle(parsed, embedding=True, **chat_kwargs) as emb:
                for tname, tcase in tests.items():
                    progress(0.85, desc=f"Pass 2 — scoring {tname}...")
                    results.append(tcase.score(
                        collected[tname],
                        embedding_handle=emb if tcase.needs_embedding else None,
                    ))
        else:
            for tname, tcase in tests.items():
                progress(0.90, desc=f"Scoring {tname}...")
                results.append(tcase.score(collected[tname], embedding_handle=None))

        progress(0.98, desc="Rendu rapport...")
        md = to_markdown(static_report, results)
        viz = _dynamic_visualizations(results)
        return md, viz

    except Exception as e:  # noqa: BLE001
        import traceback
        return f"❌ Erreur:\n```\n{traceback.format_exc()}\n```", None


# =============================================================================
# UI
# =============================================================================

def create_interface():
    with gr.Blocks(title="GGUF Security Scanner", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""# 🔬 GGUF Security Scanner

**Static** (pattern matching OWASP Top 10) + **Dynamic** (live llama.cpp probing).

⚠️ Static scan = baseline informatif (faux positifs nombreux). Vraie évaluation = onglet Dynamic.
""")

        with gr.Tab("Static Scan"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_s = gr.File(label="📁 Fichier GGUF", file_types=[".gguf"], type="filepath")
                    btn_s = gr.Button("🔍 Lancer scan statique", variant="primary", size="lg")
                    gr.Markdown("""
### OWASP LLM Top 10
1. Prompt Injection  2. Insecure Output  3. Data Poisoning
4. DoS  5. Supply Chain  6. Sensitive Info Disclosure
7. Insecure Plugin  8. Excessive Agency  9. Overreliance  10. Model Theft

⚠️ Pattern matching uniquement. Nombreux faux positifs.
""")
                with gr.Column(scale=2):
                    out_md_s = gr.Markdown("📋 *En attente d'analyse...*")
            viz_s = gr.Plot(label="Visualisations")
            btn_s.click(analyze_static, inputs=[file_s], outputs=[out_md_s, viz_s])

        with gr.Tab("Dynamic Scan"):
            gr.Markdown("""
**Charge le modèle via llama-cpp-python, exécute une batterie de tests comportementaux.**

Familles : `jailbreak` (refus prompts adverses), `harmful_bias` (refus contenu dangereux + dispersion biais),
`backdoor` (clustering activation sur tokens suspects).
""")
            with gr.Row():
                with gr.Column(scale=1):
                    file_d = gr.File(label="📁 Fichier GGUF", file_types=[".gguf"], type="filepath")
                    tests_d = gr.CheckboxGroup(
                        choices=list(TEST_REGISTRY.keys()),
                        value=list(TEST_REGISTRY.keys()),
                        label="Tests à exécuter",
                    )
                    with gr.Accordion("Paramètres llama.cpp", open=False):
                        n_ctx = gr.Slider(512, 32768, value=4096, step=512, label="n_ctx")
                        tensor_split = gr.Textbox(value="0.5,0.5",
                                                  label="tensor_split (vide = single GPU)")
                        n_gpu_layers = gr.Slider(-1, 100, value=-1, step=1, label="n_gpu_layers (-1 = all)")
                        max_tokens = gr.Slider(32, 1024, value=256, step=32, label="max_tokens / completion")
                        temperature = gr.Slider(0.0, 1.5, value=0.3, step=0.05, label="temperature")
                        seed = gr.Number(value=42, label="seed", precision=0)
                    quick = gr.Checkbox(value=True, label="⚡ Quick (subset prompts)")
                    btn_d = gr.Button("🧪 Lancer scan dynamique", variant="primary", size="lg")
                with gr.Column(scale=2):
                    out_md_d = gr.Markdown("📋 *En attente d'analyse...*")
            viz_d = gr.Plot(label="Visualisations")
            btn_d.click(
                analyze_dynamic,
                inputs=[file_d, tests_d, n_ctx, tensor_split, n_gpu_layers,
                        max_tokens, temperature, quick, seed],
                outputs=[out_md_d, viz_d],
            )

        gr.Markdown("""---
**Disclaimer :** Outil de recherche. Le scan dynamique exécute des prompts adverses publiquement
documentés (DAN, prefix injection, etc.) pour tester l'alignement du modèle. Pas une garantie
absolue de sécurité.
""")
    return demo


if __name__ == "__main__":
    create_interface().launch(server_name="0.0.0.0", server_port=7860)
