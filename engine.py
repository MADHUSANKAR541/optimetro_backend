from storage import load_csv
from typing import List

# Try to import other modules, fallback to mock responses if not found
try:
    from induction_api import get_induction_decision
except ImportError:
    def get_induction_decision(train_id):
        return "Run"

try:
    from stabling_api import get_stabling_bay
except ImportError:
    def get_stabling_bay(train_id):
        return "Bay-1"

try:
    from conflicts_api import get_conflicts
except ImportError:
    def get_conflicts(train_id):
        return ["Maintenance pending"]

try:
    from demand_api import get_predicted_demand
except ImportError:
    def get_predicted_demand(train_id):
        return 1200.0


def generate_reasons(
    train_id: str,
    induction_decision: str = None,
    stabling_bay: str = None,
    conflicts: List[str] = None,
    predicted_demand: float = None
) -> List[str]:
    reasons = []

    # Load CSV data
    fitness = load_csv("fitness_certificates.csv")
    job_cards = load_csv("job_cards.csv")
    branding = load_csv("branding_contracts.csv")

    # Fitness check
    if train_id in fitness['train_id'].values:
        row = fitness[fitness['train_id'] == train_id].iloc[0]
        reasons.append("Fitness OK" if row['is_valid'] else f"Fitness Invalid")

    # Job-card check
    if train_id in job_cards['train_id'].values:
        row = job_cards[job_cards['train_id'] == train_id].iloc[0]
        reasons.append("Job-card closed" if row['status'].lower() == 'completed' else f"Job-card {row['status']}")

    # Branding SLA
    if train_id in branding['train_id'].values:
        row = branding[branding['train_id'] == train_id].iloc[0]
        reasons.append(f"Branding SLA {row['sla_priority'].capitalize()}")

    # Conflicts
    if conflicts:
        for c in conflicts:
            reasons.append(f"Conflict: {c}")

    # Predicted demand
    if predicted_demand:
        reasons.append(f"Predicted Demand: {predicted_demand}")

    # Stabling info
    if stabling_bay:
        reasons.append(f"Assigned Bay: {stabling_bay}")

    # Induction decision
    if induction_decision:
        reasons.append(f"Induction Decision: {induction_decision}")

    return reasons
