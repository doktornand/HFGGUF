# app.py
"""
GGUF Security Scanner v1.0
===========================
Application Gradio pour l'analyse de sécurité des modèles GGUF
Intègre les contrôles OWASP Top 10 pour LLM Applications v1.1
"""

import gradio as gr
import struct
import numpy as np
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# =============================================================================
# GGUF FORMAT SPECIFICATION
# =============================================================================

GGUF_ALIGNMENT = 32

class GGMLType:
    """Types de données GGML/GGUF"""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29


GGML_TYPE_INFO = {
    GGMLType.F32: {"name": "F32", "block_size": 1, "type_size": 4, "bits": 32},
    GGMLType.F16: {"name": "F16", "block_size": 1, "type_size": 2, "bits": 16},
    GGMLType.Q4_0: {"name": "Q4_0", "block_size": 32, "type_size": 18, "bits": 4.5},
    GGMLType.Q4_1: {"name": "Q4_1", "block_size": 32, "type_size": 20, "bits": 5},
    GGMLType.Q5_0: {"name": "Q5_0", "block_size": 32, "type_size": 22, "bits": 5.5},
    GGMLType.Q5_1: {"name": "Q5_1", "block_size": 32, "type_size": 24, "bits": 6},
    GGMLType.Q8_0: {"name": "Q8_0", "block_size": 32, "type_size": 34, "bits": 8.5},
    GGMLType.Q8_1: {"name": "Q8_1", "block_size": 32, "type_size": 36, "bits": 9},
    GGMLType.Q2_K: {"name": "Q2_K", "block_size": 256, "type_size": 82, "bits": 2.5625},
    GGMLType.Q3_K: {"name": "Q3_K", "block_size": 256, "type_size": 110, "bits": 3.4375},
    GGMLType.Q4_K: {"name": "Q4_K", "block_size": 256, "type_size": 144, "bits": 4.5},
    GGMLType.Q5_K: {"name": "Q5_K", "block_size": 256, "type_size": 176, "bits": 5.5},
    GGMLType.Q6_K: {"name": "Q6_K", "block_size": 256, "type_size": 210, "bits": 6.5625},
    GGMLType.Q8_K: {"name": "Q8_K", "block_size": 256, "type_size": 292, "bits": 9.125},
    GGMLType.IQ2_XXS: {"name": "IQ2_XXS", "block_size": 256, "type_size": 66, "bits": 2.0625},
    GGMLType.IQ2_XS: {"name": "IQ2_XS", "block_size": 256, "type_size": 74, "bits": 2.3125},
    GGMLType.IQ3_XXS: {"name": "IQ3_XXS", "block_size": 256, "type_size": 98, "bits": 3.0625},
    GGMLType.IQ1_S: {"name": "IQ1_S", "block_size": 256, "type_size": 50, "bits": 1.5625},
    GGMLType.IQ4_NL: {"name": "IQ4_NL", "block_size": 32, "type_size": 18, "bits": 4.5},
    GGMLType.IQ3_S: {"name": "IQ3_S", "block_size": 256, "type_size": 110, "bits": 3.4375},
    GGMLType.IQ2_S: {"name": "IQ2_S", "block_size": 256, "type_size": 82, "bits": 2.5625},
    GGMLType.IQ4_XS: {"name": "IQ4_XS", "block_size": 256, "type_size": 136, "bits": 4.25},
    GGMLType.I8: {"name": "I8", "block_size": 1, "type_size": 1, "bits": 8},
    GGMLType.I16: {"name": "I16", "block_size": 1, "type_size": 2, "bits": 16},
    GGMLType.I32: {"name": "I32", "block_size": 1, "type_size": 4, "bits": 32},
    GGMLType.I64: {"name": "I64", "block_size": 1, "type_size": 8, "bits": 64},
    GGMLType.F64: {"name": "F64", "block_size": 1, "type_size": 8, "bits": 64},
    GGMLType.IQ1_M: {"name": "IQ1_M", "block_size": 256, "type_size": 56, "bits": 1.75},
}


# =============================================================================
# GGUF SECURITY SCANNER
# =============================================================================

class GGUFSecurityScanner:
    """Analyseur de sécurité complet pour modèles GGUF"""
    
    def __init__(self, file_path: str, filename: str):
        self.file_path = file_path
        self.filename = filename
        self.file_size = os.path.getsize(file_path)
        self.metadata = {}
        self.tensors_info = {}
        self.analysis_results = {}
        self.vulnerabilities = []
        self.security_score = 100
        self.recommendations = []
        self.capabilities = {}
        self._parse_gguf()
    
    def _parse_gguf(self):
        """Parse la structure GGUF depuis le fichier"""
        try:
            with open(self.file_path, 'rb') as f:
                magic = f.read(4)
                if magic != b'GGUF':
                    raise ValueError("Fichier GGUF invalide: magic number incorrect")
                
                version = struct.unpack('<I', f.read(4))[0]
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
                
                self._read_metadata(f, metadata_kv_count)
                self._read_tensor_info(f, tensor_count)
                
        except Exception as e:
            raise ValueError(f"Erreur parsing GGUF: {e}")
    
    def _read_metadata(self, f, count):
        for i in range(count):
            try:
                key_length = struct.unpack('<Q', f.read(8))[0]
                key = f.read(key_length).decode('utf-8', errors='replace')
                value_type = struct.unpack('<I', f.read(4))[0]
                value = self._read_value_by_type(f, value_type)
                self.metadata[key] = value
            except Exception:
                continue
    
    def _read_value_by_type(self, f, value_type):
        try:
            if value_type == 0:    # UINT8
                return struct.unpack('<B', f.read(1))[0]
            elif value_type == 1:  # INT8
                return struct.unpack('<b', f.read(1))[0]
            elif value_type == 2:  # UINT16
                return struct.unpack('<H', f.read(2))[0]
            elif value_type == 3:  # INT16
                return struct.unpack('<h', f.read(2))[0]
            elif value_type == 4:  # UINT32
                return struct.unpack('<I', f.read(4))[0]
            elif value_type == 5:  # INT32
                return struct.unpack('<i', f.read(4))[0]
            elif value_type == 6:  # FLOAT32
                return struct.unpack('<f', f.read(4))[0]
            elif value_type == 7:  # BOOL
                return struct.unpack('<B', f.read(1))[0] != 0
            elif value_type == 8:  # STRING
                length = struct.unpack('<Q', f.read(8))[0]
                return f.read(length).decode('utf-8', errors='replace')
            elif value_type == 9:  # ARRAY
                atype = struct.unpack('<I', f.read(4))[0]
                alen = struct.unpack('<Q', f.read(8))[0]
                return [self._read_value_by_type(f, atype) for _ in range(alen)]
            elif value_type == 10: # UINT64
                return struct.unpack('<Q', f.read(8))[0]
            elif value_type == 11: # INT64
                return struct.unpack('<q', f.read(8))[0]
            elif value_type == 12: # FLOAT64
                return struct.unpack('<d', f.read(8))[0]
            else:
                return None
        except Exception:
            return None
    
    def _read_tensor_info(self, f, count):
        """Lit les informations des tenseurs"""
        for i in range(count):
            try:
                name_length = struct.unpack('<Q', f.read(8))[0]
                name = f.read(name_length).decode('utf-8', errors='replace')
                ndim = struct.unpack('<I', f.read(4))[0]
                shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndim)]
                ggml_type = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]
                
                element_count = int(np.prod(shape)) if shape else 0
                type_info = GGML_TYPE_INFO.get(ggml_type, {"name": f"Unknown", "block_size": 1, "type_size": 1, "bits": 32})
                
                if type_info["block_size"] > 1:
                    n_blocks = (element_count + type_info["block_size"] - 1) // type_info["block_size"]
                    size_bytes = n_blocks * type_info["type_size"]
                else:
                    size_bytes = element_count * type_info["type_size"]
                
                self.tensors_info[name] = {
                    'shape': shape,
                    'type_name': type_info["name"],
                    'size_bytes': size_bytes,
                    'size_mb': size_bytes / (1024 * 1024),
                    'element_count': element_count,
                    'is_quantized': type_info["block_size"] > 1,
                    'bits_per_element': type_info.get("bits", 32)
                }
            except Exception:
                continue
    
    def scan(self) -> Dict[str, Any]:
        """Exécute tous les scans de sécurité"""
        
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
        
        return {
            'filename': self.filename,
            'file_size_mb': self.file_size / (1024 * 1024),
            'metadata': self.metadata,
            'tensors': self.tensors_info,
            'analysis': self.analysis_results,
            'vulnerabilities': self.vulnerabilities,
            'security_score': self.security_score,
            'recommendations': self.recommendations,
            'capabilities': self.capabilities
        }
    
    def _analyze_structure(self):
        """Analyse structurelle de base"""
        arch = self.metadata.get('general.architecture', 'unknown')
        self.analysis_results['structure'] = {
            'architecture': arch,
            'tensor_count': len(self.tensors_info),
            'total_size_mb': sum(t['size_mb'] for t in self.tensors_info.values()),
            'model_name': self.metadata.get('general.name', 'Unknown'),
            'file_type': self.metadata.get('general.file_type', 'Unknown'),
            'quantization_types': list(set(t['type_name'] for t in self.tensors_info.values()))
        }
    
    def _check_llm01_prompt_injection(self):
        """LLM01: Prompt Injection"""
        findings = []
        risk_level = "low"
        
        model_name = self.metadata.get('general.name', '').lower()
        arch_name = self.metadata.get('general.architecture', '').lower()
        
        suspicious_patterns = ['instruct', 'chat', 'assistant', 'uncensored', 'unfiltered', 'jailbreak']
        for pattern in suspicious_patterns:
            if pattern in model_name or pattern in arch_name:
                findings.append(f"Nom suggérant une surface d'attaque: '{pattern}'")
                risk_level = "medium"
        
        total_params = sum(t['element_count'] for t in self.tensors_info.values())
        if total_params < 1e9:
            findings.append("Petit modèle (< 1B) - plus vulnérable au fine-tuning malveillant")
            risk_level = "medium"
        
        if findings:
            self.vulnerabilities.append({
                'id': 'LLM01',
                'name': 'Prompt Injection',
                'risk_level': risk_level,
                'findings': findings,
                'mitigation': "Filtrage des prompts, séparateurs de contexte, validation des entrées"
            })
            self.security_score -= 10 if risk_level == "medium" else 5
    
    def _check_llm02_insecure_output(self):
        """LLM02: Insecure Output Handling"""
        findings = []
        risk_level = "low"
        
        model_desc = str(self.metadata.get('general.description', '')).lower()
        
        code_keywords = ['code', 'coding', 'developer', 'programming', 'sql', 'python', 'javascript', 'html']
        for keyword in code_keywords:
            if keyword in model_desc:
                findings.append(f"Génération de code détectée ({keyword})")
                risk_level = "high"
                break
        
        if 'sql' in model_desc and risk_level != "high":
            findings.append("Génération SQL - risque d'injection")
            risk_level = "high"
        
        if findings:
            self.vulnerabilities.append({
                'id': 'LLM02',
                'name': 'Insecure Output Handling',
                'risk_level': risk_level,
                'findings': findings,
                'mitigation': "Sanitization des sorties, encodage selon contexte, validation stricte"
            })
            self.security_score -= 15 if risk_level == "high" else 8
    
    def _check_llm03_training_data_poisoning(self):
        """LLM03: Training Data Poisoning"""
        findings = []
        risk_level = "low"
        
        author = str(self.metadata.get('general.author', '')).lower()
        trusted_authors = ['meta', 'microsoft', 'google', 'mistral', 'llama', 'qwen', 'deepseek']
        
        if author and author not in trusted_authors:
            findings.append(f"Auteur non officiel: {author}")
            risk_level = "medium"
        
        if not self.metadata.get('general.quantization_version'):
            findings.append("Version de quantification non spécifiée")
            risk_level = "medium"
        
        if findings:
            self.vulnerabilities.append({
                'id': 'LLM03',
                'name': 'Training Data Poisoning',
                'risk_level': risk_level,
                'findings': findings,
                'mitigation': "Utiliser des sources officielles, vérifier les checksums"
            })
            self.security_score -= 12 if risk_level == "medium" else 6
    
    def _check_llm04_model_denial_service(self):
        """LLM04: Model Denial of Service"""
        findings = []
        risk_level = "low"
        
        total_size_mb = sum(t['size_mb'] for t in self.tensors_info.values())
        
        if total_size_mb > 10000:
            findings.append(f"Modèle très volumineux ({total_size_mb:.0f} MB)")
            risk_level = "high"
        elif total_size_mb > 5000:
            findings.append(f"Modèle volumineux ({total_size_mb:.0f} MB)")
            risk_level = "medium"
        
        context_length = self.metadata.get('llama.context_length', 0)
        if context_length and context_length > 32768:
            findings.append(f"Contexte très long ({context_length})")
            risk_level = "high"
        elif context_length and context_length > 16384:
            findings.append(f"Contexte long ({context_length})")
            if risk_level == "low":
                risk_level = "medium"
        
        if findings:
            self.vulnerabilities.append({
                'id': 'LLM04',
                'name': 'Model Denial of Service',
                'risk_level': risk_level,
                'findings': findings,
                'mitigation': "Rate limiting, monitoring, timeout, batch processing"
            })
            self.security_score -= 15 if risk_level == "high" else 8
    
    def _check_llm05_supply_chain(self):
        """LLM05: Supply Chain Vulnerabilities"""
        findings = []
        risk_level = "low"
        
        if len(self.tensors_info) == 0:
            findings.append("Aucun tenseur détecté - fichier corrompu")
            risk_level = "high"
        
        suspicious_tensors = []
        for name in self.tensors_info.keys():
            if any(pattern in name.lower() for pattern in ['backdoor', 'trojan', 'malware', 'exploit', 'payload']):
                suspicious_tensors.append(name)
        
        if suspicious_tensors:
            findings.append(f"Tenseurs suspects: {', '.join(suspicious_tensors[:3])}")
            risk_level = "critical"
        
        if findings:
            self.vulnerabilities.append({
                'id': 'LLM05',
                'name': 'Supply Chain Vulnerabilities',
                'risk_level': risk_level,
                'findings': findings,
                'mitigation': "Vérifier les signatures, sources officielles, audits"
            })
            self.security_score -= 20 if risk_level == "critical" else 10
    
    def _check_llm06_sensitive_info_disclosure(self):
        """LLM06: Sensitive Information Disclosure"""
        findings = []
        risk_level = "low"
        
        sensitive_patterns = ['password', 'token', 'key', 'secret', 'api', 'credential', 'auth', 'private']
        
        for key, value in self.metadata.items():
            value_str = str(value).lower()
            for pattern in sensitive_patterns:
                if pattern in key.lower() or pattern in value_str:
                    findings.append(f"Métadonnée sensible possible: {key}")
                    risk_level = "high"
                    break
            if risk_level == "high":
                break
        
        if findings:
            self.vulnerabilities.append({
                'id': 'LLM06',
                'name': 'Sensitive Information Disclosure',
                'risk_level': risk_level,
                'findings': findings,
                'mitigation': "Redaction des logs, contrôle des sorties"
            })
            self.security_score -= 18 if risk_level == "high" else 9
    
    def _check_llm07_insecure_plugin_design(self):
        """LLM07: Insecure Plugin Design"""
        findings = []
        risk_level = "low"
        
        metadata_str = str(self.metadata).lower()
        
        dangerous_capabilities = ['execute', 'run', 'shell', 'command', 'system', 'subprocess', 'eval']
        for cap in dangerous_capabilities:
            if cap in metadata_str:
                findings.append(f"Capacité d'exécution: {cap}")
                risk_level = "high"
                break
        
        fs_keywords = ['file', 'read', 'write', 'open', 'path', 'directory', 'delete']
        for kw in fs_keywords:
            if kw in metadata_str and risk_level != "high":
                findings.append(f"Accès fichier: {kw}")
                risk_level = "medium"
                break
        
        if findings:
            self.vulnerabilities.append({
                'id': 'LLM07',
                'name': 'Insecure Plugin Design',
                'risk_level': risk_level,
                'findings': findings,
                'mitigation': "Sandboxing, moindre privilège, validation"
            })
            self.security_score -= 15 if risk_level == "high" else 8
    
    def _check_llm08_excessive_agency(self):
        """LLM08: Excessive Agency"""
        findings = []
        risk_level = "low"
        
        metadata_str = str(self.metadata).lower()
        
        agency_keywords = ['agent', 'autonomous', 'auto', 'tool', 'function', 'call', 'action', 'api']
        for kw in agency_keywords:
            if kw in metadata_str:
                findings.append(f"Capacité agentique: {kw}")
                risk_level = "medium"
                break
        
        if 'instruct' in self.metadata.get('general.name', '').lower():
            findings.append("Modèle Instruct - potentiel agentique")
            if risk_level == "low":
                risk_level = "medium"
        
        if findings:
            self.vulnerabilities.append({
                'id': 'LLM08',
                'name': 'Excessive Agency',
                'risk_level': risk_level,
                'findings': findings,
                'mitigation': "Limiter actions, validation humaine, audit trail"
            })
            self.security_score -= 12 if risk_level == "medium" else 6
    
    def _check_llm09_overreliance(self):
        """LLM09: Overreliance"""
        findings = []
        risk_level = "low"
        
        total_params = sum(t['element_count'] for t in self.tensors_info.values())
        if total_params < 1e9:
            findings.append("Très petit modèle - risque d'hallucinations élevé")
            risk_level = "high"
        elif total_params < 3e9:
            findings.append("Petit modèle - risque d'hallucinations modéré")
            risk_level = "medium"
        
        quant_types = [t['type_name'] for t in self.tensors_info.values()]
        aggressive_quants = ['IQ1_S', 'IQ1_M', 'IQ2_XXS', 'Q2_K']
        if any(q in aggressive_quants for q in quant_types):
            findings.append("Quantification très agressive - qualité réduite")
            if risk_level == "low":
                risk_level = "high"
        
        if findings:
            self.vulnerabilities.append({
                'id': 'LLM09',
                'name': 'Overreliance',
                'risk_level': risk_level,
                'findings': findings,
                'mitigation': "Validation humaine, monitoring des hallucinations"
            })
            self.security_score -= 10 if risk_level == "high" else 5
    
    def _check_llm10_model_theft(self):
        """LLM10: Model Theft"""
        findings = []
        risk_level = "low"
        
        if not self.metadata.get('general.author'):
            findings.append("Auteur non spécifié")
            risk_level = "medium"
        
        if not self.metadata.get('general.license'):
            findings.append("Licence non spécifiée")
            risk_level = "medium"
        
        findings.append("Modèle non chiffré - vulnérable au vol")
        
        if findings:
            self.vulnerabilities.append({
                'id': 'LLM10',
                'name': 'Model Theft',
                'risk_level': risk_level,
                'findings': findings,
                'mitigation': "Chiffrement, contrôles d'accès, watermarking"
            })
            self.security_score -= 8
    
    def _detect_capabilities(self):
        """Détecte les capacités spéciales du modèle"""
        self.capabilities = {
            'code_generation': False,
            'sql_generation': False,
            'system_commands': False,
            'file_access': False,
            'web_access': False,
            'tool_use': False
        }
        
        metadata_str = str(self.metadata).lower()
        
        if any(kw in metadata_str for kw in ['code', 'programming', 'python', 'javascript', 'java', 'c++']):
            self.capabilities['code_generation'] = True
        
        if 'sql' in metadata_str:
            self.capabilities['sql_generation'] = True
        
        if any(kw in metadata_str for kw in ['command', 'shell', 'execute', 'system', 'terminal']):
            self.capabilities['system_commands'] = True
        
        if any(kw in metadata_str for kw in ['file', 'read', 'write', 'open', 'path']):
            self.capabilities['file_access'] = True
        
        if any(kw in metadata_str for kw in ['web', 'http', 'url', 'browse', 'internet']):
            self.capabilities['web_access'] = True
        
        if any(kw in metadata_str for kw in ['tool', 'function', 'action', 'plugin', 'api']):
            self.capabilities['tool_use'] = True
    
    def _generate_recommendations(self):
        """Génère des recommandations"""
        self.recommendations = []
        
        critical = [v for v in self.vulnerabilities if v['risk_level'] == 'critical']
        high = [v for v in self.vulnerabilities if v['risk_level'] == 'high']
        medium = [v for v in self.vulnerabilities if v['risk_level'] == 'medium']
        
        if critical:
            self.recommendations.append("🚨 **URGENT** - Vulnérabilités critiques détectées. NE PAS UTILISER en production.")
            for v in critical:
                self.recommendations.append(f"  - {v['name']}: {v['mitigation']}")
        
        if high:
            self.recommendations.append("⚠️ **HAUTE PRIORITÉ** - Vulnérabilités majeures:")
            for v in high:
                self.recommendations.append(f"  - {v['name']}: {v['mitigation']}")
        
        if medium:
            self.recommendations.append("📋 **RECOMMANDATIONS** - Améliorations suggérées:")
            for v in medium:
                self.recommendations.append(f"  - {v['name']}: {v['mitigation']}")
        
        if self.capabilities['code_generation'] or self.capabilities['system_commands']:
            self.recommendations.append("🔒 Isoler dans un environnement sandboxé")
        
        if self.capabilities['file_access']:
            self.recommendations.append("📁 Restreindre l'accès au système de fichiers")
        
        if self.security_score < 50:
            self.recommendations.append("❌ Score critique - Remplacer ce modèle")
    
    def _calculate_final_score(self):
        """Calcule le score final"""
        risk_penalties = {
            'critical': 30,
            'high': 20,
            'medium': 10,
            'low': 5
        }
        
        total_penalty = 0
        for v in self.vulnerabilities:
            total_penalty += risk_penalties.get(v['risk_level'], 5)
        
        self.security_score = max(0, min(100, 100 - total_penalty))
        
        if self.metadata.get('general.license'):
            self.security_score = min(100, self.security_score + 5)
        if self.metadata.get('general.author'):
            self.security_score = min(100, self.security_score + 5)


# =============================================================================
# GRADIO UI FUNCTIONS
# =============================================================================

def create_security_report(scanner: GGUFSecurityScanner) -> str:
    """Crée un rapport de sécurité formaté Markdown"""
    
    results = scanner.scan()
    
    report = []
    report.append("# 🔬 GGUF Security Scanner - Rapport d'Analyse\n")
    report.append(f"**Fichier:** `{results['filename']}`")
    report.append(f"**Taille:** `{results['file_size_mb']:.2f} MB`")
    report.append(f"**Score de sécurité:** `{results['security_score']}/100`\n")
    
    score = results['security_score']
    if score >= 80:
        grade = "🟢 **Excellent**"
    elif score >= 60:
        grade = "🟡 **Moyen**"
    elif score >= 40:
        grade = "🟠 **Risqué**"
    else:
        grade = "🔴 **Critique**"
    report.append(f"**Niveau:** {grade}\n")
    
    # Métriques
    report.append("## 📊 Métriques du Modèle\n")
    report.append(f"- **Architecture:** {results['analysis']['structure']['architecture']}")
    report.append(f"- **Nom:** {results['analysis']['structure']['model_name']}")
    report.append(f"- **Tenseurs:** {results['analysis']['structure']['tensor_count']:,}")
    report.append(f"- **Types de quantification:** {', '.join(results['analysis']['structure']['quantization_types'])}\n")
    
    # Capacités
    report.append("## 🎯 Capacités Détectées\n")
    cap_icons = {
        'code_generation': '💻',
        'sql_generation': '📊',
        'system_commands': '🐚',
        'file_access': '📁',
        'web_access': '🌐',
        'tool_use': '🔧'
    }
    detected = False
    for cap, detected_flag in results['capabilities'].items():
        if detected_flag:
            icon = cap_icons.get(cap, '✓')
            report.append(f"- {icon} **{cap.replace('_', ' ').title()}**")
            detected = True
    if not detected:
        report.append("- Aucune capacité spéciale détectée")
    report.append("")
    
    # Vulnérabilités
    if results['vulnerabilities']:
        report.append("## 🚨 Vulnérabilités Détectées (OWASP LLM Top 10)\n")
        
        for v in results['vulnerabilities']:
            risk_icon = {
                'critical': '💀',
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }.get(v['risk_level'], '⚪')
            
            report.append(f"### {risk_icon} {v['id']}: {v['name']}")
            report.append(f"**Niveau de risque:** `{v['risk_level'].upper()}`\n")
            report.append("**Constats:**")
            for finding in v['findings']:
                report.append(f"- {finding}")
            report.append("")
            report.append("**Recommandation:**")
            report.append(f"> {v['mitigation']}\n")
    else:
        report.append("## ✅ Sécurité\n")
        report.append("Aucune vulnérabilité majeure détectée selon OWASP LLM Top 10.\n")
    
    # Recommandations
    if results['recommendations']:
        report.append("## 💡 Recommandations\n")
        for rec in results['recommendations']:
            report.append(rec)
        report.append("")
    
    # Top tenseurs
    report.append("## 📦 Top 10 Plus Gros Tenseurs\n")
    report.append("| Nom | Type | Taille (MB) | Éléments | Quantifié |")
    report.append("|-----|------|-------------|----------|-----------|")
    
    top_tensors = sorted(results['tensors'].items(), key=lambda x: x[1]['size_mb'], reverse=True)[:10]
    for name, info in top_tensors:
        short_name = name[:40] + "..." if len(name) > 40 else name
        report.append(f"| `{short_name}` | {info['type_name']} | {info['size_mb']:.1f} | {info['element_count']:,} | {'✓' if info['is_quantized'] else '✗'} |")
    
    return "\n".join(report)


def create_visualizations(scanner: GGUFSecurityScanner):
    """Crée des visualisations Plotly interactives"""
    
    results = scanner.scan()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Score de Sécurité", "Distribution des Types de Quantification", "Vulnérabilités par Niveau", "Top 10 Plus Gros Tenseurs"),
        specs=[[{"type": "indicator"}, {"type": "pie"}], [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Score gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=results['security_score'],
            title={"text": "Score Global"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 40], "color": "firebrick"},
                    {"range": [40, 60], "color": "orange"},
                    {"range": [60, 80], "color": "gold"},
                    {"range": [80, 100], "color": "forestgreen"}
                ]
            }
        ),
        row=1, col=1
    )
    
    # Distribution des types
    quant_types = results['analysis']['structure']['quantization_types']
    if quant_types:
        fig.add_trace(
            go.Pie(labels=quant_types, hole=0.4, marker=dict(colors=px.colors.qualitative.Set3)),
            row=1, col=2
        )
    
    # Vulnérabilités par niveau
    risk_counts = defaultdict(int)
    for v in results['vulnerabilities']:
        risk_counts[v['risk_level']] += 1
    
    risk_labels = ['Critique', 'Élevé', 'Moyen', 'Bas']
    risk_order = ['critical', 'high', 'medium', 'low']
    risk_colors = {'critical': '#8B0000', 'high': '#DC143C', 'medium': '#FF8C00', 'low': '#FFD700'}
    
    counts = [risk_counts[r] for r in risk_order]
    colors = [risk_colors[r] for r in risk_order]
    
    fig.add_trace(
        go.Bar(
            x=risk_labels,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # Top tenseurs
    top_tensors = sorted(results['tensors'].items(), key=lambda x: x[1]['size_mb'], reverse=True)[:10]
    tensor_names = [name[:30] + "..." if len(name) > 30 else name for name, _ in top_tensors]
    tensor_sizes = [info['size_mb'] for _, info in top_tensors]
    tensor_colors = ['coral' if info['is_quantized'] else 'lightblue' for _, info in top_tensors]
    
    fig.add_trace(
        go.Bar(
            x=tensor_names,
            y=tensor_sizes,
            marker_color=tensor_colors,
            text=[f"{s:.1f} MB" for s in tensor_sizes],
            textposition='outside'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text=f"Analyse de Sécurité - {results['filename'][:50]}",
        showlegend=True,
        height=800,
        template="plotly_white",
        title_font_size=16
    )
    
    fig.update_xaxes(title_text="Niveau de risque", row=2, col=1)
    fig.update_yaxes(title_text="Nombre de vulnérabilités", row=2, col=1)
    fig.update_xaxes(title_text="Tenseur", tickangle=45, row=2, col=2)
    fig.update_yaxes(title_text="Taille (MB)", row=2, col=2)
    
    return fig


def analyze_gguf_file(file):
    """
    Fonction principale appelée par Gradio
    """
    if file is None:
        return "⚠️ Veuillez sélectionner un fichier GGUF", None
    
    try:
        if hasattr(file, 'name'):
            file_path = file.name
            original_filename = getattr(file, 'original_name', file.name)
        elif isinstance(file, str):
            file_path = file
            original_filename = Path(file).name
        else:
            return "❌ Format de fichier non reconnu", None
        
        scanner = GGUFSecurityScanner(file_path, original_filename)
        report = create_security_report(scanner)
        visualizations = create_visualizations(scanner)
        
        return report, visualizations
        
    except Exception as e:
        error_msg = f"""❌ **Erreur lors de l'analyse**
        {str(e)}
        **Solutions possibles:**
        1. Vérifiez que le fichier est bien au format GGUF
        2. Assurez-vous que le fichier n'est pas corrompu"""
        return error_msg, None


# =============================================================================
# INTERFACE GRADIO
# =============================================================================

def create_interface():
    with gr.Blocks(title="GGUF Security Scanner", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🔬 GGUF Security Scanner
        
        **Analyse de sécurité complète pour modèles LLM au format GGUF**
        
        Cet outil analyse les fichiers GGUF pour détecter les vulnérabilités selon l'**OWASP Top 10 for LLM Applications v1.1**.
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="📁 Sélectionner un fichier GGUF",
                    file_types=[".gguf"],
                    type="filepath"
                )
                
                analyze_btn = gr.Button("🔍 Lancer l'analyse", variant="primary", size="lg")
                
                gr.Markdown("""
                ### 📚 OWASP LLM Top 10 Vérifiés
                
                1. **LLM01** - Prompt Injection
                2. **LLM02** - Insecure Output Handling
                3. **LLM03** - Training Data Poisoning
                4. **LLM04** - Model Denial of Service
                5. **LLM05** - Supply Chain Vulnerabilities
                6. **LLM06** - Sensitive Information Disclosure
                7. **LLM07** - Insecure Plugin Design
                8. **LLM08** - Excessive Agency
                9. **LLM09** - Overreliance
                10. **LLM10** - Model Theft
                
                ### ⚠️ Avertissement
                Analyse statique uniquement. Ne garantit pas l'absence de toutes vulnérabilités.
                """)
            
            with gr.Column(scale=2):
                report_output = gr.Markdown("📋 **En attente d'analyse...**\n\nSélectionnez un fichier GGUF et cliquez sur 'Lancer l'analyse'.")
        
        with gr.Row():
            viz_output = gr.Plot(label="📊 Visualisations interactives")
        
        analyze_btn.click(
            fn=analyze_gguf_file,
            inputs=[file_input],
            outputs=[report_output, viz_output]
        )
        
        gr.Markdown("""
        ---
        **Disclaimer:** Outil fourni à titre éducatif. Toujours vérifier les modèles avant utilisation en production.
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
