from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Workload(BaseModel):
    train_gpus: Optional[int] = 0
    train_steps: Optional[int] = 0
    inference_qps: Optional[float] = 0.0
    latency_ms: Optional[int] = 0
    batch: bool = True

class DataSpec(BaseModel):
    size_gb: float
    growth_gb_per_month: float = 0
    hot_fraction: float = 0.3
    retention_days: int = 90
    egress_gb_per_month: float = 0.0

class Constraints(BaseModel):
    clouds: List[str] = Field(default_factory=list)        # ["aws","gcp","azure"]
    regions: List[str] = Field(default_factory=list)
    compliance: List[str] = Field(default_factory=list)
    region_lock: Optional[str] = None
    managed_ok: bool = True
    serverless_ok: bool = True
    max_monthly_cost: Optional[float] = None

class ProjectSpec(BaseModel):
    name: str
    workload: Workload
    data: DataSpec
    constraints: Constraints

class Blueprint(BaseModel):
    id: str
    cloud: str
    region: str
    services: List[Dict[str, Any]]  # [{service, sku, qty_expr, unit}]
    assumptions: Dict[str, Any] = {}

class LineItem(BaseModel):
    service: str
    sku: str
    qty: float
    unit: str
    unit_price: float
    cost: float

class Estimate(BaseModel):
    blueprint_id: str
    monthly_cost: float
    bom: List[LineItem]
    p50: Optional[float] = None
    p90: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class OptimizationAction(BaseModel):
    type: str
    rationale: str
    delta_cost: float

class OptimizationResult(BaseModel):
    blueprint_id: str
    actions: List[OptimizationAction]
    estimate: Estimate
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class RiskFinding(BaseModel):
    category: str     # "egress","idle_gpu","storage_tier"
    severity: str     # "low","med","high"
    evidence: str
    fix: str