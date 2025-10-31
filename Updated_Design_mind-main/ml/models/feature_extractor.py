"""
Feature Extraction Module
Extracts 38 features from HLD text content
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np


@dataclass
class HLDFeatures:
    """Container for extracted HLD features"""
    word_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    avg_word_length: float = 0.0
    header_count: int = 0
    code_block_count: int = 0
    table_count: int = 0
    list_count: int = 0
    diagram_count: int = 0
    completeness_score: float = 0.0
    security_mentions: int = 0
    scalability_mentions: int = 0
    api_mentions: int = 0
    database_mentions: int = 0
    performance_mentions: int = 0
    monitoring_mentions: int = 0
    duplicate_headers: int = 0
    header_coverage: float = 0.0
    code_coverage: float = 0.0
    keyword_density: float = 0.0
    section_density: float = 0.0
    has_architecture_section: int = 0
    has_security_section: int = 0
    has_scalability_section: int = 0
    has_deployment_section: int = 0
    has_monitoring_section: int = 0
    has_api_spec: int = 0
    has_data_model: int = 0
    service_count: int = 0
    entity_count: int = 0
    api_endpoint_count: int = 0
    readability_score: float = 0.0
    completeness_index: float = 0.0
    consistency_index: float = 0.0
    documentation_quality: float = 0.0
    technical_terms_density: float = 0.0
    acronym_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert features to dictionary"""
        return {
            'word_count': self.word_count,
            'sentence_count': self.sentence_count,
            'avg_sentence_length': self.avg_sentence_length,
            'avg_word_length': self.avg_word_length,
            'header_count': self.header_count,
            'code_block_count': self.code_block_count,
            'table_count': self.table_count,
            'list_count': self.list_count,
            'diagram_count': self.diagram_count,
            'completeness_score': self.completeness_score,
            'security_mentions': self.security_mentions,
            'scalability_mentions': self.scalability_mentions,
            'api_mentions': self.api_mentions,
            'database_mentions': self.database_mentions,
            'performance_mentions': self.performance_mentions,
            'monitoring_mentions': self.monitoring_mentions,
            'duplicate_headers': self.duplicate_headers,
            'header_coverage': self.header_coverage,
            'code_coverage': self.code_coverage,
            'keyword_density': self.keyword_density,
            'section_density': self.section_density,
            'has_architecture_section': self.has_architecture_section,
            'has_security_section': self.has_security_section,
            'has_scalability_section': self.has_scalability_section,
            'has_deployment_section': self.has_deployment_section,
            'has_monitoring_section': self.has_monitoring_section,
            'has_api_spec': self.has_api_spec,
            'has_data_model': self.has_data_model,
            'service_count': self.service_count,
            'entity_count': self.entity_count,
            'api_endpoint_count': self.api_endpoint_count,
            'readability_score': self.readability_score,
            'completeness_index': self.completeness_index,
            'consistency_index': self.consistency_index,
            'documentation_quality': self.documentation_quality,
            'technical_terms_density': self.technical_terms_density,
            'acronym_count': self.acronym_count
        }

    def to_array(self) -> np.ndarray:
        """Convert features to numpy array"""
        values = list(self.to_dict().values())
        return np.array(values).reshape(1, -1)


class FeatureExtractor:
    """Extract features from HLD text content"""

    # Keywords for semantic analysis
    SECURITY_KEYWORDS = {
        'authentication', 'authorization', 'encryption', 'ssl', 'tls', 'oauth', 'jwt',
        'password', 'token', 'secret', 'certificate', 'firewall', 'vpn', 'secure',
        'identity', 'access control', 'audit', 'compliance', 'security', 'threat'
    }

    SCALABILITY_KEYWORDS = {
        'scalable', 'scalability', 'horizontal', 'vertical', 'load balancing',
        'distributed', 'cluster', 'partition', 'shard', 'cache', 'elastic',
        'auto-scale', 'throughput', 'latency', 'capacity', 'growth', 'concurrent'
    }

    API_KEYWORDS = {
        'api', 'endpoint', 'rest', 'graphql', 'http', 'request', 'response',
        'method', 'get', 'post', 'put', 'delete', 'json', 'xml', 'payload'
    }

    DATABASE_KEYWORDS = {
        'database', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis',
        'table', 'schema', 'query', 'index', 'transaction', 'data', 'storage'
    }

    PERFORMANCE_KEYWORDS = {
        'performance', 'optimization', 'caching', 'compression', 'bottleneck',
        'throughput', 'latency', 'response time', 'benchmark', 'profiling'
    }

    MONITORING_KEYWORDS = {
        'monitoring', 'logging', 'metrics', 'alerting', 'dashboard', 'observability',
        'tracing', 'health check', 'uptime', 'sla', 'error rate'
    }

    TECHNICAL_TERMS = {
        'microservices', 'container', 'kubernetes', 'docker', 'aws', 'azure', 'gcp',
        'lambda', 'serverless', 'api gateway', 'message queue', 'event driven',
        'pipeline', 'deployment', 'ci/cd', 'agile', 'devops', 'infrastructure'
    }

    def __init__(self):
        """Initialize feature extractor"""
        pass

    def extract(self, hld_content: str) -> HLDFeatures:
        """
        Extract features from HLD text

        Parameters:
            hld_content (str): HLD document text

        Returns:
            HLDFeatures: Extracted features
        """
        features = HLDFeatures()

        # Text metrics
        features.word_count = self._count_words(hld_content)
        features.sentence_count = self._count_sentences(hld_content)
        if features.sentence_count > 0:
            features.avg_sentence_length = features.word_count / features.sentence_count
        if features.word_count > 0:
            features.avg_word_length = len(hld_content) / features.word_count

        # Structure
        features.header_count = len(re.findall(r'^#+\s', hld_content, re.MULTILINE))
        features.code_block_count = len(re.findall(r'```', hld_content))
        features.table_count = len(re.findall(r'\|.*\|', hld_content))
        features.list_count = len(re.findall(r'^\s*[-*]\s', hld_content, re.MULTILINE))
        features.diagram_count = len(re.findall(r'```(?:mermaid|diagram|graphviz)', hld_content))

        # Semantic indicators
        features.completeness_score = self._calculate_completeness_score(hld_content)
        features.security_mentions = self._count_mentions(hld_content, self.SECURITY_KEYWORDS)
        features.scalability_mentions = self._count_mentions(hld_content, self.SCALABILITY_KEYWORDS)
        features.api_mentions = self._count_mentions(hld_content, self.API_KEYWORDS)
        features.database_mentions = self._count_mentions(hld_content, self.DATABASE_KEYWORDS)
        features.performance_mentions = self._count_mentions(hld_content, self.PERFORMANCE_KEYWORDS)
        features.monitoring_mentions = self._count_mentions(hld_content, self.MONITORING_KEYWORDS)

        # Consistency
        headers = re.findall(r'^#+\s(.+)$', hld_content, re.MULTILINE)
        features.duplicate_headers = len(headers) - len(set(headers))
        features.header_coverage = min(features.header_count / 20, 1.0) if features.header_count > 0 else 0
        features.code_coverage = min(features.code_block_count / 10, 1.0) if features.code_block_count > 0 else 0

        # Density metrics
        features.keyword_density = self._calculate_keyword_density(hld_content)
        if features.header_count > 0:
            features.section_density = min(features.header_count / max(features.word_count / 100, 1), 1.0)

        # Document properties
        features.has_architecture_section = 1 if re.search(r'architecture|design|structure', hld_content, re.I) else 0
        features.has_security_section = 1 if re.search(r'security|authentication|authorization', hld_content, re.I) else 0
        features.has_scalability_section = 1 if re.search(r'scalability|scalable|performance', hld_content, re.I) else 0
        features.has_deployment_section = 1 if re.search(r'deployment|infrastructure|deployment|devops', hld_content, re.I) else 0
        features.has_monitoring_section = 1 if re.search(r'monitoring|logging|metrics|observability', hld_content, re.I) else 0
        features.has_api_spec = 1 if re.search(r'api|endpoint|rest|graphql', hld_content, re.I) else 0
        features.has_data_model = 1 if re.search(r'data model|entity|schema|database', hld_content, re.I) else 0

        # Complexity
        features.service_count = len(re.findall(r'service|microservice', hld_content, re.I))
        features.entity_count = len(re.findall(r'entity|class|object', hld_content, re.I))
        features.api_endpoint_count = len(re.findall(r'/api|endpoint|get|post|put|delete', hld_content, re.I))

        # Quality indicators
        features.readability_score = self._calculate_readability_score(hld_content)
        features.completeness_index = self._calculate_completeness_index(hld_content)
        features.consistency_index = self._calculate_consistency_index(hld_content)
        features.documentation_quality = self._calculate_documentation_quality(hld_content)

        # Text features
        features.technical_terms_density = self._calculate_technical_terms_density(hld_content)
        features.acronym_count = len(re.findall(r'\b[A-Z]{2,}\b', hld_content))

        return features

    def features_to_array(self, features: HLDFeatures) -> np.ndarray:
        """
        Convert features to ML-compatible format

        Parameters:
            features (HLDFeatures): Extracted features

        Returns:
            ndarray: Feature array suitable for model input
        """
        return features.to_array()

    # Helper methods
    @staticmethod
    def _count_words(text: str) -> int:
        """Count words in text"""
        return len(text.split())

    @staticmethod
    def _count_sentences(text: str) -> int:
        """Count sentences in text"""
        return len(re.split(r'[.!?]+', text)) - 1

    @staticmethod
    def _count_mentions(text: str, keywords: set) -> int:
        """Count keyword mentions"""
        count = 0
        text_lower = text.lower()
        for keyword in keywords:
            count += len(re.findall(rf'\b{keyword}\b', text_lower))
        return count

    @staticmethod
    def _calculate_keyword_density(text: str) -> float:
        """Calculate density of important keywords"""
        important_keywords = {
            'architecture', 'security', 'scalability', 'performance', 'api',
            'database', 'monitoring', 'deployment', 'service', 'entity'
        }
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        keyword_count = FeatureExtractor._count_mentions(text, important_keywords)
        return min(keyword_count / total_words, 1.0)

    @staticmethod
    def _calculate_readability_score(text: str) -> float:
        """Calculate readability score (0-100)"""
        sentences = len(re.split(r'[.!?]+', text)) - 1
        words = len(text.split())

        if sentences == 0 or words == 0:
            return 0.0

        # Flesch Reading Ease adapted
        avg_sentence_length = words / sentences
        avg_word_length = len(text) / words if words > 0 else 0

        score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_word_length
        return max(0, min(100, score))

    @staticmethod
    def _calculate_completeness_score(text: str) -> float:
        """Calculate content completeness score (0-100)"""
        required_sections = [
            r'architecture', r'design', r'structure',
            r'security', r'scalability', r'performance',
            r'api', r'database', r'deployment'
        ]

        found = sum(1 for section in required_sections if re.search(section, text, re.I))
        return (found / len(required_sections)) * 100

    @staticmethod
    def _calculate_completeness_index(text: str) -> float:
        """Calculate completeness index (0-1)"""
        score = FeatureExtractor._calculate_completeness_score(text)
        return score / 100

    @staticmethod
    def _calculate_consistency_index(text: str) -> float:
        """Calculate internal consistency (0-1)"""
        headers = re.findall(r'^#+\s(.+)$', text, re.MULTILINE)
        if not headers:
            return 0.0

        unique_headers = len(set(headers))
        duplicates = len(headers) - unique_headers

        return max(0, 1 - (duplicates / len(headers)))

    @staticmethod
    def _calculate_documentation_quality(text: str) -> float:
        """Calculate overall documentation quality (0-100)"""
        factors = [
            (len(text) > 1000, 20),  # Sufficient length
            (len(re.findall(r'```', text)) > 0, 20),  # Has code examples
            (len(re.findall(r'^#+\s', text, re.MULTILINE)) > 5, 20),  # Has structure
            (FeatureExtractor._count_sentences(text) > 20, 20),  # Has detail
            (len(re.findall(r'\|.*\|', text)) > 0, 20),  # Has tables
        ]

        quality = sum(points for condition, points in factors if condition)
        return min(quality, 100)

    @staticmethod
    def _calculate_technical_terms_density(text: str) -> float:
        """Calculate density of technical terms"""
        terms = FeatureExtractor.TECHNICAL_TERMS
        total_words = len(text.split())
        if total_words == 0:
            return 0.0

        term_count = FeatureExtractor._count_mentions(text, terms)
        return min(term_count / total_words, 1.0)
