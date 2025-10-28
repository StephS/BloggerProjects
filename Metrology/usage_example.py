"""
Spindle Metrology Analysis - Usage Example

This shows the minimal steps to analyze your measurement data.
"""

import sympy as sp
from models import SpindleConfiguration, RawMeasurementData
from solver import run_analysis
from utils import create_standard_config

# ============================================================================
# STEP 1: Define Configuration
# ============================================================================

# Option A: Use helper for standard config
config = create_standard_config(
    indicator_angles='0, 180',      # Top and bottom
    spindle_angles='0, 180',        # 0° and 180° rotation
    z_start=0,
    z_stop=100,
    z_step=10,
    diameter=50
)

# Option B: Create manually for custom setup
# config = SpindleConfiguration(
#     indicator_angles='0, 90',
#     spindle_angles='0, 90, 180, 270',
#     nominal_diameter=50,
#     z_positions=[sp.Rational(i*10) for i in range(11)]
# )

# ============================================================================
# STEP 2: Input Measurement Data
# ============================================================================

# Paste your measurements here (comma-separated values)
# Format: {(indicator_angle, spindle_angle): "measurement1, measurement2, ..."}

measurements = {
    (0, 0):   "4.957592, 4.87905, 4.800509, 4.721967, 4.643426, 4.564885, 4.486343, 4.407802, 4.32926, 4.250719, 4.172177",
    (180, 0): "5.048135, 5.144131, 5.240127, 5.336123, 5.432119, 5.528115, 5.624111, 5.720108, 5.816104, 5.91210, 6.008096",
    (0, 180): "4.976310, 4.932676, 4.889043, 4.845409, 4.801776, 4.758142, 4.714509, 4.670875, 4.627242, 4.583608, 4.539975",
    (180, 180): "5.030635, 5.091722, 5.152810, 5.213897, 5.274984, 5.336071, 5.397159, 5.458246, 5.519333, 5.580421, 5.641508",
}

raw_data = RawMeasurementData.from_measurements(config, measurements)

# ============================================================================
# STEP 3: Run Analysis
# ============================================================================

tilt_result, form_result = run_analysis(config, raw_data)

# ============================================================================
# STEP 4: View Results
# ============================================================================

# Print detailed results
tilt_result.print()
form_result.print()

# Export for copy/paste (optional)
print("\n" + tilt_result.export_text())
print("\n" + form_result.export_text())