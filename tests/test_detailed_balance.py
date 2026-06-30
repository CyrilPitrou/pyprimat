
import os
import csv
import pytest
from primat.config import PRIMATConfig
from primat.network_data import compute_detailed_balance_coefficients, reaction_species

def test_detailed_balance_consistency():
    """Verify that compute_detailed_balance_coefficients reproduces the values in detailed_balance.csv."""
    cfg = PRIMATConfig()
    db_csv_path = os.path.join(cfg._resolved_data_dir, "csv", "detailed_balance.csv")
    
    with open(db_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['reaction']
            ref_alpha = float(row['alpha'])
            ref_beta = float(row['beta'])
            ref_gamma = float(row['gamma'])
            
            reactants, products = reaction_species(name)
            alpha, beta, gamma = compute_detailed_balance_coefficients(reactants, products, cfg)
            
            # Check beta exactly (it's always 0, 1.5, -1.5, etc.)
            assert beta == pytest.approx(ref_beta), f"Beta mismatch for {name}"
            
            # Check alpha and gamma to 1% (reproducibility limit of the physics formula vs original PRIMAT tables)
            if ref_alpha != 0:
                assert abs(alpha - ref_alpha) / abs(ref_alpha) < 0.01, f"Alpha mismatch for {name}: {alpha} vs {ref_alpha}"
            if ref_gamma != 0:
                assert abs(gamma - ref_gamma) / abs(ref_gamma) < 0.01, f"Gamma mismatch for {name}: {gamma} vs {ref_gamma}"
