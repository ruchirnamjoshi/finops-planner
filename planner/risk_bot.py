from __future__ import annotations
from typing import List, Dict, Any, Optional
import json
from .schemas import RiskFinding, Estimate, ProjectSpec, Blueprint
import logging

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"Warning: Failed to load .env file: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangChain output schemas
class RiskAnalysisSchema(BaseModel):
    """Schema for risk analysis."""
    risks: List[Dict[str, Any]] = Field(description="List of identified risks")

class IntelligentRiskAssessmentAgent:
    """
    LLM-powered agent that intelligently assesses risks in cloud infrastructure plans.
    Capabilities:
    - Advanced risk pattern recognition
    - Multi-dimensional risk analysis (cost, security, performance, compliance)
    - Context-aware risk assessment
    - Detailed mitigation strategies
    - Risk scoring and prioritization
    - Compliance and security analysis
    """
    
    def __init__(self, openai_client: Optional[ChatOpenAI] = None):
        self._langchain_client = openai_client
        self.risk_patterns = self._load_risk_patterns()
        self.compliance_frameworks = self._load_compliance_frameworks()
        
    @property
    def client(self):
        """Lazy initialization of LangChain client to prevent import hangs"""
        if self._langchain_client is None:
            # Check if API key is available
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")
            
            self._langchain_client = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                openai_api_key=api_key
            )
            print("✅ Risk Assessment LangChain client initialized successfully with API key")
        return self._langchain_client
    
    def _load_risk_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive risk patterns and their characteristics."""
        return {
            "cost_risks": {
                "runaway_costs": {
                    "description": "Uncontrolled cost escalation",
                    "indicators": ["high_egress", "unused_resources", "over_provisioning"],
                    "severity": "high",
                    "mitigation": "Implement cost controls, budgets, and monitoring"
                },
                "vendor_lock_in": {
                    "description": "Difficulty migrating to other cloud providers",
                    "indicators": ["proprietary_services", "high_migration_costs"],
                    "severity": "medium",
                    "mitigation": "Use multi-cloud strategies and standard technologies"
                }
            },
            "security_risks": {
                "data_exposure": {
                    "description": "Unauthorized access to sensitive data",
                    "indicators": ["public_access", "weak_encryption", "inadequate_iam"],
                    "severity": "critical",
                    "mitigation": "Implement proper access controls and encryption"
                },
                "network_vulnerabilities": {
                    "description": "Network security weaknesses",
                    "indicators": ["open_ports", "public_subnets", "no_vpc"],
                    "severity": "high",
                    "mitigation": "Use VPCs, security groups, and network ACLs"
                }
            },
            "performance_risks": {
                "latency_issues": {
                    "description": "Performance degradation due to latency",
                    "indicators": ["cross_region", "high_egress", "poor_connectivity"],
                    "severity": "medium",
                    "mitigation": "Optimize data placement and use CDNs"
                },
                "scalability_limits": {
                    "description": "Inability to handle increased load",
                    "indicators": ["fixed_capacity", "no_auto_scaling"],
                    "severity": "medium",
                    "mitigation": "Implement auto-scaling and load balancing"
                }
            },
            "compliance_risks": {
                "data_residency": {
                    "description": "Data stored in non-compliant locations",
                    "indicators": ["wrong_region", "no_encryption"],
                    "severity": "high",
                    "mitigation": "Ensure data is stored in compliant regions"
                },
                "audit_trail": {
                    "description": "Insufficient logging and monitoring",
                    "indicators": ["no_cloudtrail", "limited_logging"],
                    "severity": "medium",
                    "mitigation": "Enable comprehensive logging and monitoring"
                }
            }
        }
    
    def _load_compliance_frameworks(self) -> Dict[str, Dict[str, Any]]:
        """Load compliance framework requirements."""
        return {
            "soc2": {
                "data_encryption": "Data must be encrypted at rest and in transit",
                "access_controls": "Strict access controls and authentication required",
                "audit_logging": "Comprehensive audit logging mandatory",
                "incident_response": "Documented incident response procedures required"
            },
            "iso27001": {
                "information_security": "Information security management system required",
                "risk_assessment": "Regular risk assessments mandatory",
                "business_continuity": "Business continuity planning required",
                "compliance_monitoring": "Ongoing compliance monitoring required"
            },
            "gdpr": {
                "data_protection": "Data protection by design and default",
                "user_consent": "Explicit user consent required",
                "data_portability": "Data portability rights",
                "breach_notification": "72-hour breach notification requirement"
            }
        }
    
    def _analyze_cost_risks(self, spec: ProjectSpec, bp: Blueprint, est: Estimate) -> List[RiskFinding]:
        """Analyze cost-related risks."""
        findings = []
        total_cost = est.monthly_cost
        
        if total_cost == 0:
            return findings
        
        # Egress cost analysis
        egress_cost = sum(li.cost for li in est.bom if "egress" in li.sku.lower())
        egress_ratio = egress_cost / total_cost
        
        if egress_ratio > 0.4:
            findings.append(RiskFinding(
                category="cost_egress",
                severity="high",
                evidence=f"Egress costs represent {round(egress_ratio*100, 1)}% of total monthly cost (${egress_cost:.2f})",
                fix="Implement data locality strategies, use CDN, reduce cross-region transfers, and consider data residency requirements."
            ))
        
        # Storage cost analysis
        storage_cost = sum(li.cost for li in est.bom if li.service in ["storage", "s3", "blob"])
        storage_ratio = storage_cost / total_cost
        
        if storage_ratio > 0.6:
            findings.append(RiskFinding(
                category="cost_storage",
                severity="medium",
                evidence=f"Storage costs represent {round(storage_ratio*100, 1)}% of total monthly cost (${storage_cost:.2f})",
                fix="Implement storage lifecycle policies, use appropriate storage tiers, and consider data archival strategies."
            ))
        
        # Budget constraint analysis
        if spec.constraints.max_monthly_cost and total_cost > spec.constraints.max_monthly_cost:
            findings.append(RiskFinding(
                category="cost_budget",
                severity="critical",
                evidence=f"Estimated cost (${total_cost:.2f}) exceeds budget constraint (${spec.constraints.max_monthly_cost:.2f})",
                fix="Reduce resource requirements, use cost-optimized services, or increase budget allocation."
            ))
        
        return findings
    
    def _analyze_security_risks(self, spec: ProjectSpec, bp: Blueprint, est: Estimate) -> List[RiskFinding]:
        """Analyze security-related risks."""
        findings = []
        
        # Check for public-facing services
        public_services = [svc for svc in bp.services if svc.get("service") in ["load_balancer", "api_gateway"]]
        if public_services:
            findings.append(RiskFinding(
                category="security_public",
                severity="medium",
                evidence=f"Public-facing services detected: {[svc.get('service') for svc in public_services]}",
                fix="Implement proper security groups, WAF, DDoS protection, and regular security audits."
            ))
        
        # Check for data sensitivity
        if spec.data.size_gb > 1000:  # Large datasets
            findings.append(RiskFinding(
                category="security_data",
                severity="high",
                evidence=f"Large dataset detected ({spec.data.size_gb} GB) - potential data exposure risk",
                fix="Implement encryption at rest and in transit, strict access controls, and data classification."
            ))
        
        # Check compliance requirements
        if spec.constraints.compliance:
            for compliance in spec.constraints.compliance:
                if compliance.lower() in self.compliance_frameworks:
                    findings.append(RiskFinding(
                        category=f"compliance_{compliance.lower()}",
                        severity="high",
                        evidence=f"Compliance requirement: {compliance}",
                        fix=f"Ensure all {compliance} requirements are met, including data encryption, access controls, and audit logging."
                    ))
        
        return findings
    
    def _analyze_performance_risks(self, spec: ProjectSpec, bp: Blueprint, est: Estimate) -> List[RiskFinding]:
        """Analyze performance-related risks."""
        findings = []
        
        # Latency analysis
        if spec.workload.latency_ms and spec.workload.latency_ms > 0:
            if spec.workload.batch:
                findings.append(RiskFinding(
                    category="performance_latency",
                    severity="low",
                    evidence=f"Batch workload with latency target of {spec.workload.latency_ms}ms",
                    fix="Consider removing latency constraints for batch workloads to enable more cost-effective options."
                ))
            else:
                # Check if region selection supports latency requirements
                if bp.region not in ["us-east-1", "us-west-2", "eu-west-1"]:  # Major regions
                    findings.append(RiskFinding(
                        category="performance_region",
                        severity="medium",
                        evidence=f"Non-major region selected: {bp.region}",
                        fix="Consider major regions for better connectivity and lower latency, or implement edge caching."
                    ))
        
        # Scalability analysis
        if spec.workload.inference_qps and spec.workload.inference_qps > 1000:
            findings.append(RiskFinding(
                category="performance_scalability",
                severity="medium",
                evidence=f"High inference QPS requirement: {spec.workload.inference_qps}",
                fix="Implement auto-scaling, load balancing, and consider using managed inference services."
            ))
        
        return findings
    
    def _get_llm_risk_analysis(self, spec: ProjectSpec, bp: Blueprint, est: Estimate, 
                               basic_findings: List[RiskFinding]) -> List[Dict[str, Any]]:
        """Get LLM-powered risk analysis for complex scenarios."""
        
        system_prompt = f"""You are an expert cloud security and risk assessment consultant.
        Analyze the following cloud infrastructure plan for additional risks and provide detailed analysis.
        
        Project Specification:
        {spec.model_dump()}
        
        Blueprint:
        {bp.model_dump()}
        
        Cost Estimate:
        {est.model_dump()}
        
        Basic Risk Findings:
        {[f.model_dump() for f in basic_findings]}
        
        Provide additional risk analysis in this JSON format:
        [
            {{
                "category": "risk_category",
                "severity": "low|medium|high|critical",
                "evidence": "detailed_evidence_description",
                "fix": "detailed_mitigation_strategy",
                "impact": "business_impact_description",
                "probability": "risk_probability_assessment",
                "mitigation_effort": "low|medium|high",
                "time_to_mitigate": "estimated_time_in_weeks"
            }}
        ]
        
        Focus on identifying risks that may not be obvious from basic analysis, including:
        - Business continuity risks
        - Vendor lock-in considerations
        - Compliance gaps
        - Operational risks
        - Integration risks
        - Future scalability concerns"""
        
        try:
            # Create LangChain prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert cloud security and risk assessment consultant.
                Analyze the cloud infrastructure plan for additional risks and provide detailed analysis.
                
                IMPORTANT: You must respond with ONLY valid JSON in exactly the specified format.
                Do not include any text before or after the JSON. Return ONLY the JSON object."""),
                ("human", """Analyze this cloud infrastructure plan for additional risks:

Project Specification: {spec}
Blueprint: {blueprint}
Cost Estimate: {estimate}
Basic Risk Findings: {findings}

Provide additional risk analysis in this JSON format:
{{
    "risks": [
        {{
            "category": "risk_category",
            "severity": "low|medium|high|critical",
            "evidence": "detailed_evidence_description",
            "fix": "detailed_mitigation_strategy",
            "impact": "business_impact_description",
            "probability": "risk_probability_assessment",
            "mitigation_effort": "low|medium|high",
            "time_to_mitigate": "estimated_time_in_weeks"
        }}
    ]
}}

Focus on identifying risks that may not be obvious from basic analysis, including:
- Business continuity risks
- Vendor lock-in considerations
- Compliance gaps
- Operational risks
- Integration risks
- Future scalability concerns""")
            ])
            
            # Create output parser
            parser = JsonOutputParser(pydantic_object=RiskAnalysisSchema)
            
            # Create the chain
            chain = prompt_template | self.client | parser
            
            # Prepare inputs
            inputs = {
                "spec": spec.model_dump(),
                "blueprint": bp.model_dump(),
                "estimate": est.model_dump(),
                "findings": [f.model_dump() for f in basic_findings]
            }
            
            # Invoke the chain
            result = chain.invoke(inputs)
            logger.info("✅ LangChain risk analysis generated successfully")
            return result.get("risks", [])
            
        except Exception as e:
            logger.error(f"LangChain risk analysis failed: {e}")
            # No fallbacks - raise error
            raise RuntimeError(f"LLM risk analysis failed - no fallback available: {e}")
    
    def assess_plan(self, spec: ProjectSpec, bp: Blueprint, est: Estimate,
    egress_share_threshold: float = 0.4,
                    compute_share_threshold: float = 0.6) -> List[RiskFinding]:
        """Comprehensive risk assessment using multiple analysis methods."""
        findings = []
        
        # Basic heuristic analysis (legacy)
    total = max(est.monthly_cost, 1e-9)
        
    def share(pred):
        s = sum(li.cost for li in est.bom if pred(li))
        return s / total

    egress_share = share(lambda li: "egress" in li.sku.lower() or "egress" in li.service.lower())
    compute_share = share(lambda li: any(k in (li.service.lower()+" "+li.sku.lower()) for k in ["compute","ec2","vm","gpu"]))

        # High egress share
    if egress_share > egress_share_threshold:
        findings.append(RiskFinding(
            category="egress",
            severity="high",
            evidence=f"Egress is {round(egress_share*100)}% of monthly cost.",
            fix="Co-locate compute with data, reduce cross-region traffic, or cache results.",
        ))

        # Low compute share
    if compute_share < (1 - egress_share_threshold) * 0.3:
        findings.append(RiskFinding(
            category="allocation_mismatch",
            severity="medium",
            evidence=f"Compute share is low at {round(compute_share*100)}%.",
            fix="Check if storage tiering/egress are dominating. Revisit data placement and lifecycle.",
        ))

        # Latency vs batch mismatch
    if spec.workload.batch and spec.workload.latency_ms and spec.workload.latency_ms > 0:
        findings.append(RiskFinding(
            category="latency_mismatch",
            severity="low",
            evidence=f"Batch workload but latency target set to {spec.workload.latency_ms}ms.",
            fix="If this is offline/batch, set latency to 0 to widen blueprint options.",
        ))

        # Storage tier sanity check
    if spec.data.hot_fraction < 0.1 and any("hot" in (li.sku.lower()+" "+li.service.lower()) for li in est.bom):
        findings.append(RiskFinding(
            category="storage_tier",
            severity="low",
            evidence=f"hot_fraction={spec.data.hot_fraction} but hot tier is provisioned.",
            fix="Push most data to cool/nearline tiers; keep only active working set hot.",
        ))

        # Enhanced analysis methods
        findings.extend(self._analyze_cost_risks(spec, bp, est))
        findings.extend(self._analyze_security_risks(spec, bp, est))
        findings.extend(self._analyze_performance_risks(spec, bp, est))
        
        # Get LLM-powered analysis for complex scenarios
        llm_risks = self._get_llm_risk_analysis(spec, bp, est, findings)
        for risk in llm_risks:
            findings.append(RiskFinding(
                category=risk["category"],
                severity=risk["severity"],
                evidence=risk["evidence"],
                fix=risk["fix"]
            ))
        
        # Sort findings by severity (critical, high, medium, low)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        findings.sort(key=lambda x: severity_order.get(x.severity, 4))

        return findings
    
    def get_risk_report(self, spec: ProjectSpec, bp: Blueprint, est: Estimate) -> Dict[str, Any]:
        """Generate a comprehensive risk assessment report."""
        findings = self.assess_plan(spec, bp, est)
        
        # Categorize findings by severity
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        category_counts = {}
        
        for finding in findings:
            severity_counts[finding.severity] += 1
            category_counts[finding.category] = category_counts.get(finding.category, 0) + 1
        
        # Calculate risk score
        risk_score = (
            severity_counts["critical"] * 10 +
            severity_counts["high"] * 5 +
            severity_counts["medium"] * 2 +
            severity_counts["low"] * 1
        )
        
        return {
            "blueprint_id": bp.id,
            "total_findings": len(findings),
            "risk_score": risk_score,
            "severity_distribution": severity_counts,
            "category_distribution": category_counts,
            "findings": [finding.model_dump() for finding in findings],
            "summary": {
                "critical_risks": severity_counts["critical"],
                "high_risks": severity_counts["high"],
                "overall_assessment": "high" if risk_score > 15 else "medium" if risk_score > 5 else "low",
                "recommendations": self._generate_risk_recommendations(findings)
            }
        }
    
    def _generate_risk_recommendations(self, findings: List[RiskFinding]) -> List[str]:
        """Generate actionable recommendations based on risk findings."""
        recommendations = []
        
        critical_findings = [f for f in findings if f.severity == "critical"]
        high_findings = [f for f in findings if f.severity == "high"]
        
        if critical_findings:
            recommendations.append("Address critical risks immediately before deployment")
        
        if high_findings:
            recommendations.append("Prioritize high-severity risks in your implementation plan")
        
        # Category-specific recommendations
        categories = set(f.category for f in findings)
        
        if "cost" in categories:
            recommendations.append("Implement cost monitoring and alerting systems")
        
        if "security" in categories:
            recommendations.append("Conduct security review and penetration testing")
        
        if "performance" in categories:
            recommendations.append("Implement performance monitoring and alerting")
        
        if "compliance" in categories:
            recommendations.append("Ensure compliance requirements are documented and implemented")
        
        return recommendations
    
    
# Backward compatibility function
def assess_plan(spec: ProjectSpec, bp: Blueprint, est: Estimate,
                egress_share_threshold: float = 0.4,
                compute_share_threshold: float = 0.6) -> List[RiskFinding]:
    """Legacy function for backward compatibility."""
    agent = IntelligentRiskAssessmentAgent()
    return agent.assess_plan(spec, bp, est, egress_share_threshold, compute_share_threshold)