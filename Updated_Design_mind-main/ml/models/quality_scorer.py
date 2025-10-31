"""
Rule-Based Quality Scorer
Provides heuristic quality assessment without ML models
"""

import re
from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass
class QualityScore:
    """Quality assessment results"""
    overall_score: float = 0.0  # 0-100
    completeness: float = 0.0  # 0-100
    clarity: float = 0.0  # 0-100
    consistency: float = 0.0  # 0-100
    security: float = 0.0  # 0-100
    scalability: float = 0.0  # 0-100
    recommendations: List[str] = None
    missing_elements: List[str] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        if self.missing_elements is None:
            self.missing_elements = []


class RuleBasedQualityScorer:
    """Rule-based quality scoring without ML models"""

    def __init__(self):
        """Initialize scorer"""
        self.sections = {
            'architecture': r'architecture|design|structure|overview',
            'security': r'security|authentication|authorization|encryption',
            'scalability': r'scalability|scaling|horizontal|vertical',
            'performance': r'performance|optimization|throughput|latency',
            'deployment': r'deployment|infrastructure|devops|cloud',
            'monitoring': r'monitoring|logging|metrics|observability',
            'api': r'api|endpoint|rest|graphql|http',
            'database': r'database|sql|nosql|storage|persistence'
        }

    def score(self, hld_content: str) -> QualityScore:
        """
        Calculate quality score from HLD text

        Parameters:
            hld_content (str): HLD document text

        Returns:
            QualityScore: Quality assessment
        """
        score = QualityScore()

        # Calculate individual metrics
        score.completeness = self._assess_completeness(hld_content)
        score.clarity = self._assess_clarity(hld_content)
        score.consistency = self._assess_consistency(hld_content)
        score.security = self._assess_security(hld_content)
        score.scalability = self._assess_scalability(hld_content)

        # Calculate overall score (weighted average)
        score.overall_score = (
            score.completeness * 0.25 +
            score.clarity * 0.20 +
            score.consistency * 0.20 +
            score.security * 0.15 +
            score.scalability * 0.20
        )

        # Generate recommendations
        score.recommendations = self._generate_recommendations(score, hld_content)
        score.missing_elements = self._identify_missing_elements(hld_content)

        return score

    def _assess_completeness(self, text: str) -> float:
        """
        Assess content completeness (0-100)

        Parameters:
            text (str): HLD text

        Returns:
            float: Completeness score
        """
        sections_found = 0
        total_sections = len(self.sections)

        for section_name, pattern in self.sections.items():
            if re.search(pattern, text, re.IGNORECASE):
                sections_found += 1

        completeness = (sections_found / total_sections) * 100

        # Bonus for document length
        word_count = len(text.split())
        if word_count > 2000:
            completeness = min(100, completeness + 10)
        elif word_count < 500:
            completeness = max(0, completeness - 15)

        return completeness

    def _assess_clarity(self, text: str) -> float:
        """
        Assess documentation clarity (0-100)

        Parameters:
            text (str): HLD text

        Returns:
            float: Clarity score
        """
        clarity = 50  # Base score

        # Check for headers (good structure)
        headers = len(re.findall(r'^#+\s', text, re.MULTILINE))
        if headers > 10:
            clarity += 20
        elif headers > 5:
            clarity += 10
        else:
            clarity -= 10

        # Check for code examples (clarity)
        code_blocks = len(re.findall(r'```', text))
        if code_blocks > 5:
            clarity += 15
        elif code_blocks > 0:
            clarity += 5

        # Check for tables (clarity)
        tables = len(re.findall(r'\|.*\|', text))
        if tables > 2:
            clarity += 10
        elif tables > 0:
            clarity += 5

        # Check for diagrams
        diagrams = len(re.findall(r'```(?:mermaid|diagram)', text))
        if diagrams > 0:
            clarity += 10

        # Check readability (sentence length)
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 10:
            avg_sentence_length = len(text.split()) / len(sentences)
            if 10 < avg_sentence_length < 25:
                clarity += 10
            elif avg_sentence_length > 30:
                clarity -= 10

        return min(100, max(0, clarity))

    def _assess_consistency(self, text: str) -> float:
        """
        Assess internal consistency (0-100)

        Parameters:
            text (str): HLD text

        Returns:
            float: Consistency score
        """
        consistency = 70  # Base score

        # Check for duplicate headers
        headers = re.findall(r'^#+\s(.+)$', text, re.MULTILINE)
        if headers:
            unique_headers = len(set(headers))
            duplicate_ratio = (len(headers) - unique_headers) / len(headers)
            consistency -= duplicate_ratio * 30

        # Check for consistent terminology
        technical_terms = self._extract_technical_terms(text)
        if len(technical_terms) > 5:
            consistency += 10

        # Check for formatting consistency
        code_backticks = text.count('```')
        if code_backticks % 2 == 0 and code_backticks > 0:
            consistency += 5

        return min(100, max(0, consistency))

    def _assess_security(self, text: str) -> float:
        """
        Assess security considerations (0-100)

        Parameters:
            text (str): HLD text

        Returns:
            float: Security score
        """
        security = 0

        # Check for security section
        if re.search(r'security|authentication|authorization', text, re.IGNORECASE):
            security += 30

        # Check for specific security topics
        security_keywords = {
            'authentication': 15,
            'authorization': 15,
            'encryption': 15,
            'ssl': 10,
            'tls': 10,
            'oauth': 10,
            'jwt': 10,
            'token': 10,
            'password': 5,
            'secure': 10,
            'audit': 10,
            'compliance': 10
        }

        for keyword, points in security_keywords.items():
            if re.search(rf'\b{keyword}\b', text, re.IGNORECASE):
                security += points

        return min(100, security)

    def _assess_scalability(self, text: str) -> float:
        """
        Assess scalability considerations (0-100)

        Parameters:
            text (str): HLD text

        Returns:
            float: Scalability score
        """
        scalability = 0

        # Check for scalability section
        if re.search(r'scalability|scaling|horizontal|vertical', text, re.IGNORECASE):
            scalability += 30

        # Check for specific scalability topics
        scalability_keywords = {
            'scalable': 10,
            'horizontal': 10,
            'vertical': 10,
            'load balancing': 15,
            'distributed': 15,
            'cluster': 10,
            'cache': 10,
            'elastic': 10,
            'throughput': 10,
            'latency': 10
        }

        for keyword, points in scalability_keywords.items():
            if re.search(rf'\b{keyword}\b', text, re.IGNORECASE):
                scalability += points

        return min(100, scalability)

    def _generate_recommendations(self, score: QualityScore, text: str) -> List[str]:
        """
        Generate improvement recommendations

        Parameters:
            score (QualityScore): Current quality scores
            text (str): HLD text

        Returns:
            list: Recommendations
        """
        recommendations = []

        if score.completeness < 60:
            recommendations.append("Add more comprehensive coverage of all architectural aspects")

        if score.clarity < 60:
            recommendations.append("Improve documentation clarity with better structure and examples")

        if score.consistency < 70:
            recommendations.append("Review document for consistent terminology and formatting")

        if score.security < 50:
            recommendations.append("Expand security considerations and authentication mechanisms")

        if score.scalability < 50:
            recommendations.append("Add detailed scalability and performance optimization strategies")

        # Check for missing diagrams
        if len(re.findall(r'```(?:mermaid|diagram)', text)) == 0:
            recommendations.append("Include diagrams to visualize architecture and data flows")

        # Check for missing code examples
        if len(re.findall(r'```', text)) == 0:
            recommendations.append("Add code examples for clarity")

        if score.overall_score >= 80:
            recommendations.append("Documentation quality is good! Consider minor refinements.")

        return recommendations

    def _identify_missing_elements(self, text: str) -> List[str]:
        """
        Identify missing architectural elements

        Parameters:
            text (str): HLD text

        Returns:
            list: Missing elements
        """
        missing = []

        # Check for required sections
        required_sections = {
            'Architecture Overview': r'architecture|design|structure|overview',
            'Security Design': r'security|authentication',
            'Scalability Plan': r'scalability|scaling',
            'Performance Considerations': r'performance|optimization',
            'Deployment Strategy': r'deployment|infrastructure',
            'Monitoring & Observability': r'monitoring|logging',
            'API Specification': r'api|endpoint',
            'Data Model': r'data model|database|schema'
        }

        for section_name, pattern in required_sections.items():
            if not re.search(pattern, text, re.IGNORECASE):
                missing.append(section_name)

        return missing

    @staticmethod
    def _extract_technical_terms(text: str) -> set:
        """
        Extract technical terms from text

        Parameters:
            text (str): HLD text

        Returns:
            set: Unique technical terms
        """
        # Match multi-word terms and single technical terms
        terms = set()

        # Multi-word technical terms
        patterns = [
            r'(?:micro)?services',
            r'load balancing',
            r'api gateway',
            r'message queue',
            r'data model',
            r'database',
            r'kubernetes',
            r'docker',
            r'deployment',
            r'scalability'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.update(matches)

        return terms


def demo_scoring():
    """Demonstrate quality scoring"""
    scorer = RuleBasedQualityScorer()

    # Example HLD with good structure
    good_hld = """
    # System Architecture

    ## Architecture Overview
    The system is designed as a microservices architecture with the following components:
    - API Gateway
    - Authentication Service
    - User Service
    - Product Service
    - Order Service
    - Database

    ## Security Considerations
    We implement OAuth2 for authentication and encryption for sensitive data.
    All APIs use HTTPS/TLS.

    ## Scalability
    The system is horizontally scalable using Kubernetes for orchestration.
    Load balancing distributes traffic across services.

    ## Database Design
    - Users: id, name, email
    - Products: id, name, price
    - Orders: id, user_id, product_id, quantity

    ## API Specification
    ```
    GET /api/users/{id}
    POST /api/orders
    GET /api/products
    ```

    ## Deployment
    Services are containerized with Docker and deployed on Kubernetes.
    CI/CD pipeline handles automated testing and deployment.

    ## Monitoring
    Prometheus collects metrics. Grafana visualizes dashboards.
    ELK stack handles centralized logging.
    """

    # Score the HLD
    score = scorer.score(good_hld)

    print("="*60)
    print("HLD QUALITY ASSESSMENT")
    print("="*60)
    print(f"\nOverall Score: {score.overall_score:.2f}/100")
    print(f"  Completeness: {score.completeness:.2f}/100")
    print(f"  Clarity: {score.clarity:.2f}/100")
    print(f"  Consistency: {score.consistency:.2f}/100")
    print(f"  Security: {score.security:.2f}/100")
    print(f"  Scalability: {score.scalability:.2f}/100")

    print("\nMissing Elements:")
    for element in score.missing_elements:
        print(f"  - {element}")

    print("\nRecommendations:")
    for rec in score.recommendations:
        print(f"  - {rec}")


if __name__ == "__main__":
    demo_scoring()
