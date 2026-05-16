"""Static OWASP LLM Top 10 scanner.

Baseline pattern-matching analysis preserved from v1.0 for backward compatibility.
Known limitations (many false positives) — see plan / project docs.
The real signal is produced by `ggufscan.tests.*` (dynamic probing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ggufscan.parser import ParsedGGUF


@dataclass
class StaticReport:
    filename: str
    file_size_mb: float
    metadata: dict[str, Any]
    tensors: dict[str, dict[str, Any]]
    analysis: dict[str, Any]
    vulnerabilities: list[dict[str, Any]]
    security_score: int
    recommendations: list[str]
    capabilities: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "file_size_mb": self.file_size_mb,
            "metadata": self.metadata,
            "tensors": self.tensors,
            "analysis": self.analysis,
            "vulnerabilities": self.vulnerabilities,
            "security_score": self.security_score,
            "recommendations": self.recommendations,
            "capabilities": self.capabilities,
        }


class StaticScanner:
    """OWASP LLM Top 10 static analyzer. Pattern-matching only."""

    def __init__(self, parsed: ParsedGGUF):
        self.parsed = parsed
        self.metadata = parsed.metadata
        self.tensors_info = parsed.tensors_info
        self.analysis_results: dict[str, Any] = {}
        self.vulnerabilities: list[dict[str, Any]] = []
        self.security_score = 100
        self.recommendations: list[str] = []
        self.capabilities: dict[str, bool] = {}

    def scan(self) -> StaticReport:
        self._analyze_structure()
        self._check_llm01_prompt_injection()
        self._check_llm02_insecure_output()
        self._check_llm03_training_data_poisoning()
        self._check_llm04_model_denial_service()
        self._check_llm05_supply_chain()
        self._check_llm06_sensitive_info_disclosure()
        self._check_llm07_insecure_plugin_design()
        self._check_llm08_excessive_agency()
        self._check_llm09_overreliance()
        self._check_llm10_model_theft()
        self._detect_capabilities()
        self._generate_recommendations()
        self._calculate_final_score()

        return StaticReport(
            filename=self.parsed.filename,
            file_size_mb=self.parsed.file_size_mb,
            metadata=self.metadata,
            tensors=self.tensors_info,
            analysis=self.analysis_results,
            vulnerabilities=self.vulnerabilities,
            security_score=self.security_score,
            recommendations=self.recommendations,
            capabilities=self.capabilities,
        )

    def _analyze_structure(self):
        arch = self.metadata.get("general.architecture", "unknown")
        self.analysis_results["structure"] = {
            "architecture": arch,
            "tensor_count": len(self.tensors_info),
            "total_size_mb": sum(t["size_mb"] for t in self.tensors_info.values()),
            "model_name": self.metadata.get("general.name", "Unknown"),
            "file_type": self.metadata.get("general.file_type", "Unknown"),
            "quantization_types": list(set(t["type_name"] for t in self.tensors_info.values())),
        }

    def _check_llm01_prompt_injection(self):
        findings = []
        risk_level = "low"
        model_name = str(self.metadata.get("general.name", "")).lower()
        arch_name = str(self.metadata.get("general.architecture", "")).lower()
        for pattern in ["instruct", "chat", "assistant", "uncensored", "unfiltered", "jailbreak"]:
            if pattern in model_name or pattern in arch_name:
                findings.append(f"Nom suggérant une surface d'attaque: '{pattern}'")
                risk_level = "medium"
        total_params = sum(t["element_count"] for t in self.tensors_info.values())
        if total_params < 1e9:
            findings.append("Petit modèle (< 1B) - plus vulnérable au fine-tuning malveillant")
            risk_level = "medium"
        if findings:
            self.vulnerabilities.append({
                "id": "LLM01", "name": "Prompt Injection", "risk_level": risk_level,
                "findings": findings,
                "mitigation": "Filtrage des prompts, séparateurs de contexte, validation des entrées",
            })

    def _check_llm02_insecure_output(self):
        findings = []
        risk_level = "low"
        model_desc = str(self.metadata.get("general.description", "")).lower()
        for keyword in ["code", "coding", "developer", "programming", "sql", "python", "javascript", "html"]:
            if keyword in model_desc:
                findings.append(f"Génération de code détectée ({keyword})")
                risk_level = "high"
                break
        if "sql" in model_desc and risk_level != "high":
            findings.append("Génération SQL - risque d'injection")
            risk_level = "high"
        if findings:
            self.vulnerabilities.append({
                "id": "LLM02", "name": "Insecure Output Handling", "risk_level": risk_level,
                "findings": findings,
                "mitigation": "Sanitization des sorties, encodage selon contexte, validation stricte",
            })

    def _check_llm03_training_data_poisoning(self):
        findings = []
        risk_level = "low"
        author = str(self.metadata.get("general.author", "")).lower()
        trusted = ["meta", "microsoft", "google", "mistral", "llama", "qwen", "deepseek"]
        if author and author not in trusted:
            findings.append(f"Auteur non officiel: {author}")
            risk_level = "medium"
        if not self.metadata.get("general.quantization_version"):
            findings.append("Version de quantification non spécifiée")
            risk_level = "medium"
        if findings:
            self.vulnerabilities.append({
                "id": "LLM03", "name": "Training Data Poisoning", "risk_level": risk_level,
                "findings": findings,
                "mitigation": "Utiliser des sources officielles, vérifier les checksums",
            })

    def _check_llm04_model_denial_service(self):
        findings = []
        risk_level = "low"
        total_size_mb = sum(t["size_mb"] for t in self.tensors_info.values())
        if total_size_mb > 10000:
            findings.append(f"Modèle très volumineux ({total_size_mb:.0f} MB)")
            risk_level = "high"
        elif total_size_mb > 5000:
            findings.append(f"Modèle volumineux ({total_size_mb:.0f} MB)")
            risk_level = "medium"
        ctx = self.metadata.get("llama.context_length", 0)
        if ctx and ctx > 32768:
            findings.append(f"Contexte très long ({ctx})")
            risk_level = "high"
        elif ctx and ctx > 16384:
            findings.append(f"Contexte long ({ctx})")
            if risk_level == "low":
                risk_level = "medium"
        if findings:
            self.vulnerabilities.append({
                "id": "LLM04", "name": "Model Denial of Service", "risk_level": risk_level,
                "findings": findings,
                "mitigation": "Rate limiting, monitoring, timeout, batch processing",
            })

    def _check_llm05_supply_chain(self):
        findings = []
        risk_level = "low"
        if len(self.tensors_info) == 0:
            findings.append("Aucun tenseur détecté - fichier corrompu")
            risk_level = "high"
        suspicious = [n for n in self.tensors_info if any(p in n.lower()
                      for p in ["backdoor", "trojan", "malware", "exploit", "payload"])]
        if suspicious:
            findings.append(f"Tenseurs suspects: {', '.join(suspicious[:3])}")
            risk_level = "critical"
        if findings:
            self.vulnerabilities.append({
                "id": "LLM05", "name": "Supply Chain Vulnerabilities", "risk_level": risk_level,
                "findings": findings,
                "mitigation": "Vérifier les signatures, sources officielles, audits",
            })

    def _check_llm06_sensitive_info_disclosure(self):
        findings = []
        risk_level = "low"
        sensitive = ["password", "token", "key", "secret", "api", "credential", "auth", "private"]
        for key, value in self.metadata.items():
            value_str = str(value).lower()
            for p in sensitive:
                if p in key.lower() or p in value_str:
                    findings.append(f"Métadonnée sensible possible: {key}")
                    risk_level = "high"
                    break
            if risk_level == "high":
                break
        if findings:
            self.vulnerabilities.append({
                "id": "LLM06", "name": "Sensitive Information Disclosure", "risk_level": risk_level,
                "findings": findings, "mitigation": "Redaction des logs, contrôle des sorties",
            })

    def _check_llm07_insecure_plugin_design(self):
        findings = []
        risk_level = "low"
        metadata_str = str(self.metadata).lower()
        for cap in ["execute", "run", "shell", "command", "system", "subprocess", "eval"]:
            if cap in metadata_str:
                findings.append(f"Capacité d'exécution: {cap}")
                risk_level = "high"
                break
        for kw in ["file", "read", "write", "open", "path", "directory", "delete"]:
            if kw in metadata_str and risk_level != "high":
                findings.append(f"Accès fichier: {kw}")
                risk_level = "medium"
                break
        if findings:
            self.vulnerabilities.append({
                "id": "LLM07", "name": "Insecure Plugin Design", "risk_level": risk_level,
                "findings": findings, "mitigation": "Sandboxing, moindre privilège, validation",
            })

    def _check_llm08_excessive_agency(self):
        findings = []
        risk_level = "low"
        metadata_str = str(self.metadata).lower()
        for kw in ["agent", "autonomous", "auto", "tool", "function", "call", "action", "api"]:
            if kw in metadata_str:
                findings.append(f"Capacité agentique: {kw}")
                risk_level = "medium"
                break
        if "instruct" in str(self.metadata.get("general.name", "")).lower():
            findings.append("Modèle Instruct - potentiel agentique")
            if risk_level == "low":
                risk_level = "medium"
        if findings:
            self.vulnerabilities.append({
                "id": "LLM08", "name": "Excessive Agency", "risk_level": risk_level,
                "findings": findings, "mitigation": "Limiter actions, validation humaine, audit trail",
            })

    def _check_llm09_overreliance(self):
        findings = []
        risk_level = "low"
        total = sum(t["element_count"] for t in self.tensors_info.values())
        if total < 1e9:
            findings.append("Très petit modèle - risque d'hallucinations élevé")
            risk_level = "high"
        elif total < 3e9:
            findings.append("Petit modèle - risque d'hallucinations modéré")
            risk_level = "medium"
        quants = [t["type_name"] for t in self.tensors_info.values()]
        if any(q in ["IQ1_S", "IQ1_M", "IQ2_XXS", "Q2_K"] for q in quants):
            findings.append("Quantification très agressive - qualité réduite")
            if risk_level == "low":
                risk_level = "high"
        if findings:
            self.vulnerabilities.append({
                "id": "LLM09", "name": "Overreliance", "risk_level": risk_level,
                "findings": findings, "mitigation": "Validation humaine, monitoring des hallucinations",
            })

    def _check_llm10_model_theft(self):
        findings = []
        risk_level = "low"
        if not self.metadata.get("general.author"):
            findings.append("Auteur non spécifié")
            risk_level = "medium"
        if not self.metadata.get("general.license"):
            findings.append("Licence non spécifiée")
            risk_level = "medium"
        findings.append("Modèle non chiffré - vulnérable au vol")
        self.vulnerabilities.append({
            "id": "LLM10", "name": "Model Theft", "risk_level": risk_level,
            "findings": findings, "mitigation": "Chiffrement, contrôles d'accès, watermarking",
        })

    def _detect_capabilities(self):
        self.capabilities = {k: False for k in
            ["code_generation", "sql_generation", "system_commands", "file_access", "web_access", "tool_use"]}
        s = str(self.metadata).lower()
        if any(kw in s for kw in ["code", "programming", "python", "javascript", "java", "c++"]):
            self.capabilities["code_generation"] = True
        if "sql" in s:
            self.capabilities["sql_generation"] = True
        if any(kw in s for kw in ["command", "shell", "execute", "system", "terminal"]):
            self.capabilities["system_commands"] = True
        if any(kw in s for kw in ["file", "read", "write", "open", "path"]):
            self.capabilities["file_access"] = True
        if any(kw in s for kw in ["web", "http", "url", "browse", "internet"]):
            self.capabilities["web_access"] = True
        if any(kw in s for kw in ["tool", "function", "action", "plugin", "api"]):
            self.capabilities["tool_use"] = True

    def _generate_recommendations(self):
        self.recommendations = []
        crit = [v for v in self.vulnerabilities if v["risk_level"] == "critical"]
        high = [v for v in self.vulnerabilities if v["risk_level"] == "high"]
        med = [v for v in self.vulnerabilities if v["risk_level"] == "medium"]
        if crit:
            self.recommendations.append("🚨 **URGENT** - Vulnérabilités critiques détectées. NE PAS UTILISER en production.")
            for v in crit:
                self.recommendations.append(f"  - {v['name']}: {v['mitigation']}")
        if high:
            self.recommendations.append("⚠️ **HAUTE PRIORITÉ** - Vulnérabilités majeures:")
            for v in high:
                self.recommendations.append(f"  - {v['name']}: {v['mitigation']}")
        if med:
            self.recommendations.append("📋 **RECOMMANDATIONS** - Améliorations suggérées:")
            for v in med:
                self.recommendations.append(f"  - {v['name']}: {v['mitigation']}")
        if self.capabilities["code_generation"] or self.capabilities["system_commands"]:
            self.recommendations.append("🔒 Isoler dans un environnement sandboxé")
        if self.capabilities["file_access"]:
            self.recommendations.append("📁 Restreindre l'accès au système de fichiers")
        if self.security_score < 50:
            self.recommendations.append("❌ Score critique - Remplacer ce modèle")

    def _calculate_final_score(self):
        penalties = {"critical": 30, "high": 20, "medium": 10, "low": 5}
        total = sum(penalties.get(v["risk_level"], 5) for v in self.vulnerabilities)
        self.security_score = max(0, min(100, 100 - total))
        if self.metadata.get("general.license"):
            self.security_score = min(100, self.security_score + 5)
        if self.metadata.get("general.author"):
            self.security_score = min(100, self.security_score + 5)
