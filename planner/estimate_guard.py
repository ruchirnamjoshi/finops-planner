from __future__ import annotations
from .schemas import Estimate, RiskFinding
from typing import List

def assess(estimate: Estimate) -> List[RiskFinding]:
    total = estimate.monthly_cost or 1.0
    egress = sum(li.cost for li in estimate.bom if "egress" in li.sku.lower())
    findings: List[RiskFinding] = []

    if egress / total > 0.4:
        findings.append(RiskFinding(
            category="egress",
            severity="high",
            evidence=f"Egress is {round(egress/total*100)}% of monthly cost.",
            fix="Co-locate compute with data or reduce cross-region traffic.",
        ))

    # add more simple checks here (MVP keeps it minimal)
    return findings