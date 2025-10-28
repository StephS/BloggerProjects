# ============================================================================
# File 4: tests.py
# ============================================================================
"""
Test infrastructure for spindle solver, using the LINEARIZED model.
Includes synthetic data generation with linear rigid body effect.
"""

import sympy as sp
from typing import Dict, List, Optional, Tuple, Callable, Set, Any
from dataclasses import dataclass

from models import (
    SpindleConfiguration,
    RawMeasurementData,
    MeasurementSeries,
    SolutionResult,
    ComparisonResult,
    EquationSystem,
    SystemType,
    ProfileCallable,
    Tilts,
    Forms,
    ConfigKey,
    CompareResult,
)

from solver import run_analysis, build_equation_system, assess_rigid_body_coupling

from utils import DEBUG, format_tilt_angle, get_surface_angle, slope_from_deg, _least_squares_core, remove_form_trend, convert_str_to_sp

# ============================================================================
# Surface Profile Functions
# ============================================================================


def flat_profile(value: sp.Rational = sp.S.Zero) -> ProfileCallable:
    """Create a flat (constant) profile."""
    return lambda z: sp.Rational(value)


def sinusoidal_profile(
    amplitude: sp.Rational, wavelength: sp.Rational, offset: sp.Rational = sp.S.Zero, phase: sp.Rational = sp.S.Zero
) -> ProfileCallable:
    """Create a sinusoidal profile: offset + amplitude·sin(2π·z/wavelength + phase)"""
    a, w, o, p = map(sp.Rational, [amplitude, wavelength, offset, phase])
    return lambda z: o + a * sp.sin(2 * sp.pi * z / w + p)  # pyright: ignore[reportReturnType, reportOperatorIssue]


def linear_profile(slope: sp.Rational, offset: sp.Rational = sp.S.Zero) -> ProfileCallable:
    """Create a linear profile: offset + slope·z"""
    s, o = map(sp.Rational, [slope, offset])
    return lambda z: o + s * z  # pyright: ignore[reportReturnType, reportOperatorIssue]


def quadratic_profile(a: sp.Rational, b: sp.Rational = sp.S.Zero, c: sp.Rational = sp.S.Zero) -> ProfileCallable:
    """Create a quadratic profile: c + b·z + a·z²"""
    a_r, b_r, c_r = map(sp.Rational, [a, b, c])
    return lambda z: c_r + b_r * z + a_r * z**2  # pyright: ignore[reportReturnType, reportOperatorIssue]


def composite_profile(*profiles: ProfileCallable) -> ProfileCallable:
    """Combine multiple profiles by addition."""
    if not profiles:
        raise ValueError("Requires at least one profile")
    return lambda z: sum(p(z) for p in profiles)  # pyright: ignore[reportArgumentType, reportCallIssue]


def noisy_profile(base_profile: Callable, noise_amplitude: sp.Rational, seed: Optional[int] = None) -> Callable:
    """Add Gaussian noise to a profile (numerical evaluation only)."""
    from sympy.stats import Normal, sample
    import random

    noise_amplitude = sp.Rational(noise_amplitude)
    noise_rv = Normal("noise_X", 0, noise_amplitude)

    if seed is not None:
        random.seed(seed)

    def noisy(z):
        base_value = base_profile(z)
        # If z is symbolic, return without noise
        if isinstance(z, sp.Symbol):
            return base_value
        # Numerical mode: add noise
        noise_sample = sp.Rational(sample(noise_rv))
        return (base_value + noise_sample).simplify()

    return noisy


def flat_with_noise(
    value: sp.Rational = sp.Rational(0), noise_amplitude: sp.Rational = sp.Rational("0.0001"), seed: Optional[int] = None
) -> Callable:
    """Create a flat profile with measurement noise."""
    return noisy_profile(flat_profile(value), noise_amplitude, seed)


def sinusoidal_with_noise(
    amplitude: sp.Rational,
    wavelength: sp.Rational,
    noise_amplitude: sp.Rational,
    offset: sp.Rational = sp.S.Zero,
    phase: sp.Rational = sp.S.Zero,
    seed: Optional[int] = None,
) -> Callable:
    """Create a sinusoidal profile with measurement noise."""
    base = sinusoidal_profile(amplitude, wavelength, offset, phase)
    return noisy_profile(base, noise_amplitude, seed)


def create_detrended_profile_callable_cached(profile_func: ProfileCallable, z_positions: List[sp.Rational]) -> ProfileCallable:
    """
    Creates a detrended version of the input profile.

    DETRENDING STRATEGY:
    This ensures forms don't contain hidden tilts, maintaining model validity.
    The linearized model requires: I = Form + Tilt·z

    Where:
    - Forms (M, S) should have NO linear trend with z
    - Tilts (α, β, γ) are the ONLY z-dependent terms

    By pre-detrending synthetic profiles:
    1. Forms contain only pure shape deviations
    2. All tilt effects come from α, β, γ parameters
    3. Model equation separation is maintained

    PERFORMANCE OPTIMIZATION:
    Pre-computes values at known z_positions and caches them.
    Falls back to computation for other z values (e.g., rigid body offset).

    Args:
        profile_func: Original profile function
        z_positions: Z positions where profile will be evaluated

    Returns:
        New profile function with trend removed
    """
    # Compute trend parameters
    values = [profile_func(z) for z in z_positions]
    slope, intercept = _least_squares_core(z_positions, values)
    DEBUG.info(f"Detrending profile: Found slope={float(slope):.6f}, intercept={float(intercept):.6f}")

    # Pre-compute detrended values at known positions for fast lookup
    detrended_cache: Dict[sp.Rational, sp.Rational] = {}
    for i, z in enumerate(z_positions):
        # Already computed original value, just subtract trend
        detrended_cache[z] = values[i] - (intercept + slope * z)  # pyright: ignore[reportOperatorIssue]

    # Return function with cache lookup and fallback
    def detrended_profile(z: sp.Rational) -> sp.Rational:
        # Fast path: lookup pre-computed value
        if z in detrended_cache:
            return detrended_cache[z]

        # Slow path: compute for new z value (e.g., rigid body offset)
        # Skip simplify() - not needed for correctness, only cleaner display
        raw_value = profile_func(z)
        return raw_value - (intercept + slope * z)  # pyright: ignore[reportOperatorIssue]

    return detrended_profile



def create_detrended_profile_callable(profile_func: ProfileCallable, z_positions: List[sp.Rational]) -> ProfileCallable:
    """
    Creates a detrended version of the input profile.

    DETRENDING STRATEGY:
    This ensures forms don't contain hidden tilts, maintaining model validity.
    The linearized model requires: I = Form + Tilt·z

    Where:
    - Forms (M, S) should have NO linear trend with z
    - Tilts (α, β, γ) are the ONLY z-dependent terms

    By pre-detrending synthetic profiles:
    1. Forms contain only pure shape deviations
    2. All tilt effects come from α, β, γ parameters
    3. Model equation separation is maintained

    Args:
        profile_func: Original profile function
        z_positions: Z positions where profile will be evaluated

    Returns:
        New profile function with trend removed
    """
    values = [profile_func(z) for z in z_positions]
    slope, intercept = _least_squares_core(z_positions, values)
    DEBUG.info(f"Detrending profile: Found slope={float(slope):.3e}, intercept={float(intercept):.3e}")

    # Return new function that subtracts the trend
    return lambda z: profile_func(z) - (intercept + slope * z)  # pyright: ignore[reportOperatorIssue]


# ============================================================================
# Synthetic Data Generation
# ============================================================================


def calculate_linear_rigid_body_z_offset(
    theta_ind_rad: sp.Expr,
    theta_surf_rad: sp.Expr,
    radius: sp.Rational,
    alpha_x: sp.Rational,
    alpha_y: sp.Rational,
    beta_x: sp.Rational,
    beta_y: sp.Rational,
) -> sp.Rational:
    """
    Calculates the LINEAR APPROXIMATION of Z offset due to rigid body tilt.

    COORDINATE SYSTEMS:
    - Z_machine: Fixed to guideway (commanded position)
    - Z_artifact: Along tilted artifact axis (actual measurement position)

    When artifact is tilted, indicator touches at different Z than commanded.
    First-order approximation: ΔZ ≈ R·effective_tilt_slope

    GAMMA EXCLUSION:
    Gamma (cone angle) is NOT included because:
    1. At first order, gamma changes effective radius, not Z position
    2. The effect of gamma is already captured in T equation
    3. Including gamma here would add second-order terms (gamma·R·tilt)
    4. Linearized model deliberately ignores second-order effects

    Args:
        theta_ind_rad: Indicator angle (radians)
        theta_surf_rad: Surface angle (radians)
        radius: Nominal artifact radius
        alpha_x, alpha_y: Machine tilt (doesn't rotate with spindle)
        beta_x, beta_y: Setup tilt (rotates with spindle)

    Returns:
        Z offset (ΔZ) due to rigid body geometry
    """
    # Machine tilt projected onto indicator direction
    T_alpha_eff = alpha_x * sp.sin(theta_ind_rad) + alpha_y * sp.cos(theta_ind_rad)  # pyright: ignore[reportOperatorIssue]

    # Setup tilt projected onto surface normal direction
    T_beta_eff = beta_x * sp.sin(theta_surf_rad) + beta_y * sp.cos(theta_surf_rad)  # pyright: ignore[reportOperatorIssue]

    # Combined tilt causing radial displacement
    effective_radial_tilt_slope = T_alpha_eff + T_beta_eff

    # Linear approximation: Offset = radius × slope
    delta_Z = radius * effective_radial_tilt_slope

    return sp.simplify(delta_Z)


def generate_synthetic_data(
    equations: EquationSystem,
    alpha_x: sp.Rational = sp.Rational(0),
    alpha_y: sp.Rational = sp.Rational(0),
    beta_x: sp.Rational = sp.Rational(0),
    beta_y: sp.Rational = sp.Rational(0),
    gamma: sp.Rational = sp.Rational(0),
    M_x_profile: Optional[ProfileCallable] = None,
    M_y_profile: Optional[ProfileCallable] = None,
    S_profiles: Optional[Dict[int, ProfileCallable]] = None,
    use_rigid_body: bool = False,
) -> Tuple[RawMeasurementData, Tilts, Forms]:
    """
    Generates synthetic data using the LINEARIZED model.

    Optionally includes LINEAR approximation of rigid body Z-offset effect.
    Uses PRE-DETRENDED profiles to ensure model validity (no hidden tilts in forms).

    Args:
        equations: Symbolic equation system
        alpha_x, alpha_y: Machine tilts
        beta_x, beta_y: Setup tilts
        gamma: Cone angle
        M_x_profile, M_y_profile: Machine straightness profiles
        S_profiles: Dict of surface straightness profiles by angle
        use_rigid_body: Whether to include rigid body Z-offset effect

    Returns:
        (raw_data, true_tilts, true_forms) tuple
    """
    DEBUG.enter_section(f"Generating Synthetic Data (Rigid Body: {use_rigid_body})")

    if equations.config.z_positions is None:
        raise ValueError("Config must have z_positions")

    # Create Tilts instance for ground truth
    true_tilts = Tilts(alpha_x=alpha_x, alpha_y=alpha_y, beta_x=beta_x, beta_y=beta_y, gamma=gamma)

    # Set up profile functions with defaults
    M_x_prof = M_x_profile or flat_profile()
    M_y_prof = M_y_profile or flat_profile()
    S_profs = S_profiles or {}

    # Create DETRENDED profile callables
    z_pos_list = equations.config.z_positions
    M_x_detrended = create_detrended_profile_callable(M_x_prof, z_pos_list)
    M_y_detrended = create_detrended_profile_callable(M_y_prof, z_pos_list)
    S_detrended: Dict[int, ProfileCallable] = {angle: create_detrended_profile_callable(S_profs[angle], z_pos_list) for angle in S_profs}

    # Store ground truth forms (evaluated DETRENDED profiles)
    true_forms = Forms(M_x=[M_x_detrended(z) for z in z_pos_list], M_y=[M_y_detrended(z) for z in z_pos_list], S={})

    # Add surface forms
    for var_symbol in equations.forms.surface_values(): # .to_list():
        var_name = str(var_symbol)
        # angle = sp.Rational(var_name[3:-1])
        angle = Forms._to_angle(var_name)
        if angle in S_detrended:
            values = [S_detrended[angle](z) for z in z_pos_list]
        else:
            # No profile given - assume flat zero
            values = [sp.S.Zero] * len(z_pos_list)
        #true_forms.S[f"S_{{{int(angle)}}}"] = values
        true_forms[angle] = values # pyright: ignore[reportArgumentType]

    # Generate measurements
    series_list: List[MeasurementSeries] = []
    tilt_subs = {equations.tilts[name]: val for name, val in true_tilts.to_dict().items()}

    for ind_pos in equations.config.indicator_positions:
        for spin_pos in equations.config.spindle_positions:
            key = ConfigKey(ind_pos, spin_pos)
            tilt_eq = equations.tilt_equations[key]
            theta_surf_deg = get_surface_angle(ind_pos, spin_pos)
            theta_ind_rad = sp.rad(ind_pos)
            theta_surf_rad = sp.rad(theta_surf_deg)

            data_points: List[Tuple[sp.Rational, sp.Rational]] = []

            for z_idx, z_pos in enumerate(z_pos_list):
                # Calculate Tilt Component (T value)
                T_value = tilt_eq.subs(tilt_subs)

                # Calculate Form Component (F value)
                Z_artifact = z_pos  # Default: sample at nominal Z

                if use_rigid_body:
                    # Apply linear Z-offset approximation
                    delta_Z = calculate_linear_rigid_body_z_offset(
                        theta_ind_rad, theta_surf_rad, equations.config.radius, alpha_x, alpha_y, beta_x, beta_y
                    )
                    Z_artifact = z_pos + delta_Z  # pyright: ignore[reportOperatorIssue]

                # Sample DETRENDED profiles at Z_artifact
                M_x_at_z = M_x_detrended(Z_artifact)
                M_y_at_z = M_y_detrended(Z_artifact)
                S_value = S_detrended.get(int(theta_surf_deg), flat_profile())(Z_artifact)

                # F_value = sp.simplify(M_x_at_z * sp.sin(theta_ind_rad) + M_y_at_z * sp.cos(theta_ind_rad) + S_value)  # pyright: ignore[reportOperatorIssue]
                F_value = M_x_at_z * sp.sin(theta_ind_rad) + M_y_at_z * sp.cos(theta_ind_rad) + S_value  # pyright: ignore[reportOperatorIssue]

                # Note: calling simplify only once should speed things up
                # Total measurement: I = F + T·z
                I_value = sp.simplify(F_value + T_value * z_pos)  # pyright: ignore[reportOperatorIssue]
                data_points.append((z_pos, I_value))

            series_list.append(MeasurementSeries(ind_pos, spin_pos, data_points))

    raw_data = RawMeasurementData(config=equations.config, series=series_list)
    DEBUG.success(f"Generated {len(series_list)} measurement series.")
    DEBUG.exit_section()

    return raw_data, true_tilts, true_forms


# ============================================================================
# Measurement Comparison (Replaces Projection Hack)
# ============================================================================


def compare_tilts_via_measurements(
    ground_truth: Tilts, solved: Tilts, equations: EquationSystem, tolerance: sp.Rational = sp.Rational(1, 1000000)
) -> bool:
    """
    Compare tilt solutions by checking if they produce the same measurements.

    MATHEMATICAL EQUIVALENCE:
    Two solutions are equivalent if and only if they produce identical measurements.
    For coupled variables, ground truth and min-norm solution differ only by
    nullspace components, which satisfy A·v = 0 (don't affect measurements).

    This eliminates the need for projection - we directly verify the fundamental
    equivalence criterion: A·x₁ = A·x₂

    Args:
        ground_truth: Expected tilt values
        solved: Solved tilt values from analysis
        equations: Equation system
        tolerance: Maximum acceptable measurement difference

    Returns:
        True if solutions produce equivalent measurements
    """
    DEBUG.info("Comparing tilts via measurement equivalence...")

    # Substitute ground truth into tilt equations
    measurements_from_truth = {}
    for key, eq in equations.tilt_equations.items():
        measurements_from_truth[key] = eq.subs(ground_truth.to_dict())

    # Substitute solved values into tilt equations
    measurements_from_solved = {}
    for key, eq in equations.tilt_equations.items():
        measurements_from_solved[key] = eq.subs(solved.to_dict())

    # Collect all differences
    differences = []
    all_match = True

    for key in measurements_from_truth:
        diff = abs(measurements_from_truth[key] - measurements_from_solved[key])
        differences.append(diff)

        if diff > tolerance:
            all_match = False
            DEBUG.warning(f"Measurement mismatch at {key}: {float(diff):.6e}")

    # Calculate statistics
    if differences:
        min_diff = min(differences)
        max_diff = max(differences)
        mean_diff = sum(differences) / len(differences)

        # Standard deviation: sqrt(mean((x - mean)^2))
        variance = sum((d - mean_diff) ** 2 for d in differences) / len(differences)
        std_diff = sp.sqrt(variance)

        DEBUG.info(f"Measurement difference statistics:")
        DEBUG.info(f"  Min:    {float(min_diff):.6e}", nocaller=True)
        DEBUG.info(f"  Max:    {float(max_diff):.6e}", nocaller=True)
        DEBUG.info(f"  Mean:   {float(mean_diff):.6e}", nocaller=True)
        DEBUG.info(f"  StdDev: {float(std_diff):.6e}", nocaller=True)
        DEBUG.info(f"  Count:  {len(differences)} measurements", nocaller=True)

    if all_match:
        DEBUG.success("All measurements match - solutions are equivalent!")

    return all_match


def compare_forms_via_measurements(
    ground_truth: Forms,
    solved: Forms,
    equations: EquationSystem,
    z_positions: List[sp.Rational],
    tolerance: sp.Rational = sp.Rational(1, 1000000),
) -> bool:
    """
    Compare form solutions by checking if they produce the same F measurements at each Z.

    Same principle as tilt comparison, applied independently at each Z position.
    Forms are solved separately at each Z, so we verify equivalence at each Z.

    Args:
        ground_truth: Expected form profiles
        solved: Solved form profiles from analysis
        equations: Equation system
        z_positions: Z positions where forms were solved
        tolerance: Maximum acceptable measurement difference

    Returns:
        True if solutions produce equivalent measurements
    """
    DEBUG.info("Comparing forms via measurement equivalence...")

    all_match = True
    differences = []

    for z_idx, z_pos in enumerate(z_positions):
        # Build substitution dicts for this Z position
        truth_at_z = {}
        solved_at_z = {}

        for var in equations.forms.to_list():
            var_name = str(var)

            # Get values from Forms instances
            try:
                truth_val = ground_truth[var_name]
                solved_val = solved[var_name]
            except KeyError as e:
                DEBUG.error(f"Missing form variable {var_name}: {e}")
                raise

            # Extract value at this Z index
            if isinstance(truth_val, list):
                if z_idx >= len(truth_val):
                    DEBUG.error(f"Z index {z_idx} out of range for {var_name} (len={len(truth_val)})")
                    raise IndexError(f"Z index out of range for {var_name}")
                truth_at_z[var] = truth_val[z_idx]
            else:
                truth_at_z[var] = truth_val

            if isinstance(solved_val, list):
                if z_idx >= len(solved_val):
                    DEBUG.error(f"Z index {z_idx} out of range for {var_name} (len={len(solved_val)})")
                    raise IndexError(f"Z index out of range for {var_name}")
                solved_at_z[var] = solved_val[z_idx]
            else:
                solved_at_z[var] = solved_val

        # Compare measurements at this Z
        for key, eq in equations.form_equations.items():
            measurement_from_truth = eq.subs(truth_at_z)
            measurement_from_solved = eq.subs(solved_at_z)

            # Check if substitution was complete (no remaining symbols)
            if measurement_from_truth.free_symbols:
                DEBUG.error(f"Incomplete substitution in truth at {key}, Z={z_pos}")
                DEBUG.error(f"  Remaining symbols: {measurement_from_truth.free_symbols}")
                DEBUG.error(f"  Expression: {measurement_from_truth}")
                raise ValueError("Incomplete substitution in ground truth measurement")

            if measurement_from_solved.free_symbols:
                DEBUG.error(f"Incomplete substitution in solved at {key}, Z={z_pos}")
                DEBUG.error(f"  Remaining symbols: {measurement_from_solved.free_symbols}")
                DEBUG.error(f"  Expression: {measurement_from_solved}")
                raise ValueError("Incomplete substitution in solved measurement")

            # Now safe to compute difference
            diff = abs(measurement_from_truth - measurement_from_solved) # pyright: ignore[reportOperatorIssue]

            # Simplify to get numerical value
            diff = sp.simplify(diff)
            differences.append(diff)

            if diff > tolerance:
                all_match = False
                DEBUG.warning(f"Form measurement mismatch at {key}, Z={z_pos}: {float(diff):.6e}")

    # Calculate statistics
    if differences:
        min_diff = min(differences)
        max_diff = max(differences)
        mean_diff = sum(differences) / len(differences)

        # Standard deviation: sqrt(mean((x - mean)^2))
        variance = sum((d - mean_diff) ** 2 for d in differences) / len(differences)
        std_diff = sp.sqrt(variance)

        DEBUG.info(f"Form measurement difference statistics:")
        DEBUG.info(f"  Min:    {float(min_diff):.6e}", nocaller=True)
        DEBUG.info(f"  Max:    {float(max_diff):.6e}", nocaller=True)
        DEBUG.info(f"  Mean:   {float(mean_diff):.6e}", nocaller=True)
        DEBUG.info(f"  StdDev: {float(std_diff):.6e}", nocaller=True)
        DEBUG.info(f"  Count:  {len(differences)} measurements (across {len(z_positions)} Z positions)", nocaller=True)

    if all_match:
        DEBUG.success("All form measurements match - solutions are equivalent!")

    return all_match


# ============================================================================
# Comparison and Verification
# ============================================================================


def compare_results(
    solved: SolutionResult,
    expected_tilts: Optional[Tilts] = None,
    expected_forms: Optional[Forms] = None,
    tolerance: sp.Rational = sp.Rational(1, 1000000),
) -> ComparisonResult:
    """
    Unified comparison using measurement equivalence for coupled variables.

    Strategy:
    1. For independent variables: direct value comparison
    2. For coupled variables: measurement equivalence check
    3. Verify coupling constraints if ground truth provided

    Args:
        solved: Solution result from analysis
        expected_tilts: Expected tilt values (Tilts instance)
        expected_forms: Expected form profiles (Forms instance)
        tolerance: Comparison tolerance

    Returns:
        ComparisonResult with detailed comparison information
    """
    DEBUG.enter_section(f"Comparing Results: {solved.system_type.value}")

    matched_vars: Dict[sp.Symbol, CompareResult] = {}
    mismatched_vars: Dict[sp.Symbol, CompareResult] = {}
    matched_forms: Dict[sp.Symbol, CompareResult] = {}
    mismatched_forms: Dict[sp.Symbol, CompareResult] = {}
    all_errors = []
    measurements_match = False

    # Helper to compare values
    def compare_values(var: sp.Symbol, solved_val, expected_dict: Dict[str, Any]):
        var_name = str(var)
        if var_name not in expected_dict:
            return

        true_val = expected_dict[var_name]
        is_series = isinstance(solved_val, list)

        if is_series:
            if not isinstance(true_val, list) or len(solved_val) != len(true_val):
                DEBUG.error(f"{var}: Length mismatch during comparison.")
                return

            # Compare form series
            errors = [abs(s - t) for s, t in zip(solved_val, true_val)]
            max_error = max(errors) if errors else sp.Rational(0)
            all_errors.extend(errors)

            if max_error <= tolerance:
                matched_forms[var] = CompareResult(true_val, solved_val, max_error)
            else:
                mismatched_forms[var] = CompareResult(true_val, solved_val, max_error)
        else:
            # Scalar comparison
            error = abs(solved_val - true_val)
            all_errors.append(error)

            if error <= tolerance:
                matched_vars[var] = CompareResult(true_val, solved_val, error)
            else:
                mismatched_vars[var] = CompareResult(true_val, solved_val, error)

    # Compare Independent Solutions
    if solved.system_type == SystemType.TILTS and expected_tilts:
        DEBUG.info("Comparing independent tilts...")
        for var, solved_val in solved.independent.items():
            compare_values(var, solved_val, expected_tilts.to_dict())

    elif solved.system_type == SystemType.FORMS and expected_forms and solved.forms_solution:
        DEBUG.info("Comparing independent forms...")
        for var in solved.independent.keys():
            if solved.forms_solution:
                solved_val = solved.forms_solution[str(var)]
                compare_values(var, solved_val, expected_forms.to_dict())

    # Check Measurement Equivalence for Coupled Variables
    if solved.system_type == SystemType.TILTS and expected_tilts and solved.dependent:
        DEBUG.info("Checking measurement equivalence for coupled tilts...")
        equations = build_equation_system(solved.config)

        # Build Tilts instance from complete solution
        solved_tilts = Tilts(
            alpha_x=solved.complete_solution[equations.tilts.alpha_x],
            alpha_y=solved.complete_solution[equations.tilts.alpha_y],
            beta_x=solved.complete_solution[equations.tilts.beta_x],
            beta_y=solved.complete_solution[equations.tilts.beta_y],
            gamma=solved.complete_solution[equations.tilts.gamma],
        )

        measurements_match = compare_tilts_via_measurements(expected_tilts, solved_tilts, equations, tolerance)

    elif solved.system_type == SystemType.FORMS and expected_forms and solved.dependent:
        DEBUG.info("Checking measurement equivalence for coupled forms...")
        equations = build_equation_system(solved.config)

        if solved.forms_solution and solved.config.z_positions:
            measurements_match = compare_forms_via_measurements(
                expected_forms, solved.forms_solution, equations, solved.config.z_positions, tolerance
            )

    # Final Assessment
    all_matched = not mismatched_vars and not mismatched_forms
    max_err = max(all_errors) if all_errors else None
    rms_err = sp.sqrt(sum(e**2 for e in all_errors) / len(all_errors)) if all_errors else None

    DEBUG.exit_section()

    return ComparisonResult(
        system_type=solved.system_type.value,
        matched_variables=matched_vars,
        mismatched_variables=mismatched_vars,
        matched_forms=matched_forms,
        mismatched_forms=mismatched_forms,
        coupled_variables=solved.dependent,
        free_variables=solved.free,
        measurements_match=measurements_match,
        all_matched=all_matched,
        max_error=max_err,
        rms_error=rms_err,
    )


# ============================================================================
# Test Case Infrastructure
# ============================================================================


@dataclass
class TestCase:
    """Single test case definition."""

    name: str
    description: str
    config: SpindleConfiguration
    raw_data: Optional[RawMeasurementData] = None
    expected_tilts: Optional[Tilts] = None
    expected_forms: Optional[Forms] = None
    tolerance: sp.Rational = sp.Rational(1, 1000000)

    def run(self) -> Tuple[SolutionResult, SolutionResult, Optional[ComparisonResult], Optional[ComparisonResult]]:
        """Runs the analysis and compares results if ground truth is provided."""
        print(f"\n{'='*70}\nRunning Test: {self.name}\nDescription: {self.description}\n{'='*70}")

        tilt_result, form_result = run_analysis(self.config, self.raw_data)
        tilt_result.print()
        form_result.print()

        tilt_comp, form_comp = None, None

        if self.expected_tilts and tilt_result.is_numerical:
            tilt_comp = compare_results(tilt_result, expected_tilts=self.expected_tilts, tolerance=self.tolerance)
            tilt_comp.print()

        if self.expected_forms and form_result.is_numerical:
            form_comp = compare_results(form_result, expected_forms=self.expected_forms, tolerance=self.tolerance)
            form_comp.print()

        return tilt_result, form_result, tilt_comp, form_comp


class TestLibrary:
    """Manages and runs a collection of predefined test cases."""

    def __init__(self):
        # Static variables for common tilt values
        self._0p1_deg = slope_from_deg("0.1")
        self._0p05_deg = slope_from_deg("0.05")
        self._0p025_deg = slope_from_deg("0.025")

        self.tests: Dict[str, Callable[[], TestCase]] = {}
        self._create_standard_tests()

    def _create_standard_tests(self):
        """Register all standard test cases."""
        self.tests["symbolic_180"] = self._create_symbolic_180
        self.tests["symbolic_120"] = self._create_symbolic_120
        self.tests["symbolic_120_coupled"] = self._create_symbolic_120_coupled
        self.tests["symbolic_90_coupled"] = self._create_symbolic_90_coupled
        self.tests["real_data_1"] = self._create_real_data_test_1
        self.tests["real_data_2"] = self._create_real_data_test_2
        self.tests["synthetic_simple_180"] = self._create_synthetic_simple_180
        self.tests["synthetic_complex_180"] = self._create_synthetic_complex_180
        self.tests["synthetic_rigid_body_180"] = self._create_synthetic_rigid_body_180
        self.tests["synthetic_complex_120"] = self._create_synthetic_complex_120
        self.tests["synthetic_coupled"] = self._create_synthetic_coupled_test
        self.tests["synthetic_coupled_complex"] = self._create_synthetic_coupled_complex_test
        self.tests["synthetic_coupled_complex_noise"] = self._create_synthetic_coupled_complex_noise_test

    # --- Symbolic Test Factories ---

    def _create_symbolic_180(self) -> TestCase:
        """Symbolic analysis of 180° setup to examine coupling."""
        config = SpindleConfiguration(indicator_positions=[0, 180], spindle_positions=[0, 180])
        return TestCase(name="Symbolic 180", description="Analyze coupling for 180° setup", config=config)

    def _create_symbolic_120(self) -> TestCase:
        """Symbolic analysis of 120° setup (3 indicators)."""
        config = SpindleConfiguration(indicator_positions=[0, 120, 240], spindle_positions=[0, 120, 240])
        return TestCase(name="Symbolic 120", description="Analyze solvability for 120° (3 indicators)", config=config)
    
    def _create_symbolic_120_coupled(self) -> TestCase:
        """Symbolic analysis of 120° setup (3 indicators)."""
        config = SpindleConfiguration(indicator_positions=[0, 120], spindle_positions=[0, 120, 240])
        return TestCase(name="Symbolic 120 Coupled", description="Analyze solvability for 120° (2 indicators)", config=config)

    def _create_symbolic_90_coupled(self) -> TestCase:
        """Symbolic analysis of 120° setup (3 indicators)."""
        config = SpindleConfiguration(indicator_positions=[0, 90], spindle_positions=[0, 90, 180, 270])
        return TestCase(name="Symbolic 90 Coupled", description="Analyze solvability for 90°", config=config)
    # --- Numerical Test Factories ---

    def _create_synthetic_simple_180(self) -> TestCase:
        """Simple synthetic test with only tilts."""
        z_pos = [sp.Rational(i * 10) for i in range(11)]
        config = SpindleConfiguration(indicator_positions=[0, 180], spindle_positions=[0, 180], z_positions=z_pos)
        eqs = build_equation_system(config)

        raw_data, expected_tilts, expected_forms = generate_synthetic_data(
            eqs, alpha_y=self._0p1_deg, beta_y=self._0p05_deg, gamma=self._0p025_deg
        )

        return TestCase(
            name="Synthetic Simple 180",
            description="Simple tilts, 180° setup (expects coupling)",
            config=config,
            raw_data=raw_data,
            expected_tilts=expected_tilts,
            expected_forms=expected_forms,
        )

    def _create_synthetic_complex_180(self) -> TestCase:
        """Complex synthetic test with tilts and forms."""
        z_pos = [sp.Rational(i * 10) for i in range(11)]
        config = SpindleConfiguration(indicator_positions=[0, 180], spindle_positions=[0, 180], z_positions=z_pos)
        eqs = build_equation_system(config)

        raw_data, expected_tilts, expected_forms = generate_synthetic_data(
            eqs,
            alpha_y=sp.Rational("0.001"),
            beta_y=sp.Rational("-0.0005"),
            gamma=sp.Rational("0.0002"),
            M_y_profile=sinusoidal_profile(sp.Rational("0.0001"), 50),
            S_profiles={0: linear_profile(sp.Rational("0.00001")), 180: quadratic_profile(sp.Rational("0.000001"))},
        )

        return TestCase(
            name="Synthetic Complex 180",
            description="Tilts + forms, 180° setup (expects coupling)",
            config=config,
            raw_data=raw_data,
            expected_tilts=expected_tilts,
            expected_forms=expected_forms,
        )

    def _create_synthetic_rigid_body_180(self) -> TestCase:
        """Synthetic test with rigid body Z-offset effect."""
        z_pos = [sp.Rational(i * 10) for i in range(11)]
        config = SpindleConfiguration(indicator_positions=[0, 180], spindle_positions=[0, 180], z_positions=z_pos, nominal_diameter=50)
        eqs = build_equation_system(config)

        raw_data, expected_tilts, expected_forms = generate_synthetic_data(
            eqs,
            alpha_y=sp.Rational("0.001"),
            beta_y=sp.Rational("-0.0005"),
            gamma=sp.Rational("0.0002"),
            M_y_profile=sinusoidal_profile(sp.Rational("0.0001"), 50),
            S_profiles={0: linear_profile(sp.Rational("0.00001")), 180: quadratic_profile(sp.Rational("0.000001"))},
            use_rigid_body=True,
        )

        return TestCase(
            name="Synthetic Rigid Body 180",
            description="Tilts+forms+linear rigid body, 180° setup",
            config=config,
            raw_data=raw_data,
            expected_tilts=expected_tilts,
            expected_forms=expected_forms,
        )

    def _create_synthetic_complex_120(self) -> TestCase:
        """Complex test with 120° setup (should be fully solvable)."""
        z_pos = [sp.Rational(i * 10) for i in range(11)]
        config = SpindleConfiguration(indicator_positions=[0, 120], spindle_positions=[0, 120, 240], z_positions=z_pos)
        eqs = build_equation_system(config)

        raw_data, expected_tilts, expected_forms = generate_synthetic_data(
            eqs,
            alpha_x=sp.Rational("0.0003"),
            alpha_y=sp.Rational("0.001"),
            beta_x=sp.Rational("-0.0002"),
            beta_y=sp.Rational("-0.0005"),
            gamma=sp.Rational("0.0002"),
            M_x_profile=linear_profile(sp.Rational("-0.00002")),
            M_y_profile=sinusoidal_profile(sp.Rational("0.0001"), 50),
            S_profiles={
                0: quadratic_profile(sp.Rational("0.0000005")),
                120: flat_profile(),
                240: sinusoidal_profile(sp.Rational("0.00005"), 75),
            },
        )

        return TestCase(
            name="Synthetic Complex 120",
            description="All tilts+forms, 120° setup (should be fully solvable)",
            config=config,
            raw_data=raw_data,
            expected_tilts=expected_tilts,
            expected_forms=expected_forms,
        )

    def _create_synthetic_coupled_test(self) -> TestCase:
        """Coupled synthetic test (simple)."""
        z_pos = [sp.Rational(i) for i in range(0, 201, 10)]
        config = SpindleConfiguration(
            indicator_positions=[sp.Rational(0), sp.Rational(90)],
            spindle_positions=[sp.Rational(0), sp.Rational(90), sp.Rational(180), sp.Rational(270)],
            nominal_diameter=sp.Rational(50),
            z_positions=z_pos,
        )

        equations = build_equation_system(config)

        raw_data, expected_tilts, expected_forms = generate_synthetic_data(
            equations, alpha_y=slope_from_deg(sp.Rational("0.2")), beta_y=slope_from_deg(sp.Rational("0.1"))
        )

        return TestCase(
            name="Synthetic Coupled Test",
            description=f"Simple tilts: α_y={format_tilt_angle(expected_tilts.alpha_y)}, β_y={format_tilt_angle(expected_tilts.beta_y)}",
            config=config,
            raw_data=raw_data,
            expected_tilts=expected_tilts,
            expected_forms=expected_forms,
        )

    def _create_synthetic_coupled_complex_test(self) -> TestCase:
        """Complex coupled synthetic test with forms."""
        z_pos = [sp.Rational(i) for i in range(0, 201, 10)]
        config = SpindleConfiguration(
            indicator_positions=[sp.Rational(0), sp.Rational(90)],
            spindle_positions=[sp.Rational(0), sp.Rational(90), sp.Rational(180), sp.Rational(270)],
            nominal_diameter=sp.Rational(50),
            z_positions=z_pos,
        )

        equations = build_equation_system(config)

        raw_data, expected_tilts, expected_forms = generate_synthetic_data(
            equations,
            alpha_y=slope_from_deg(sp.Rational("0.2")),
            beta_y=slope_from_deg(sp.Rational("0.1")),
            M_x_profile=sinusoidal_profile(amplitude=sp.Rational("0.001"), wavelength=sp.Rational(75)),
            M_y_profile=sinusoidal_profile(amplitude=sp.Rational("0.001"), wavelength=sp.Rational(50)),
            S_profiles={0: flat_profile(), 180: sinusoidal_profile(amplitude=sp.Rational("0.0005"), wavelength=sp.Rational(30))},
        )

        return TestCase(
            name="Synthetic Coupled Complex Test",
            description="Tilts + sinusoidal forms",
            config=config,
            raw_data=raw_data,
            expected_tilts=expected_tilts,
            expected_forms=expected_forms,
        )
    
    def _create_synthetic_coupled_complex_noise_test(self) -> TestCase:
        """Complex coupled synthetic test with forms."""
        z_pos = [sp.Rational(i) for i in range(0, 201, 10)]
        config = SpindleConfiguration(
            indicator_positions=[sp.Rational(0), sp.Rational(90)],
            spindle_positions=[sp.Rational(0), sp.Rational(90), sp.Rational(180), sp.Rational(270)],
            nominal_diameter=sp.Rational(50),
            z_positions=z_pos,
        )

        equations = build_equation_system(config)

        raw_data, expected_tilts, expected_forms = generate_synthetic_data(
            equations,
            alpha_y=slope_from_deg(sp.Rational("0.2")),
            beta_y=slope_from_deg(sp.Rational("0.1")),
            M_x_profile=flat_with_noise(value=sp.Rational(0), noise_amplitude=sp.Rational("0.0001"), seed=42),
            M_y_profile=sinusoidal_with_noise(
                amplitude=sp.Rational("0.001"), wavelength=sp.Rational(50), noise_amplitude=sp.Rational("0.00005"), seed=43
            ),
            S_profiles={0: flat_profile(), 180: sinusoidal_profile(amplitude=sp.Rational("0.0005"), wavelength=sp.Rational(30))},
        )

        return TestCase(
            name="Synthetic Coupled Complex Test",
            description="Tilts + sinusoidal forms with noise",
            config=config,
            raw_data=raw_data,
            expected_tilts=expected_tilts,
            expected_forms=expected_forms,
        )

    # --- Real Data Test Factories ---

    def _create_real_data_test_1(self) -> TestCase:
        """Real CAD data test 1."""
        z = [sp.Rational(i * 10) for i in range(11)]
        config = SpindleConfiguration(indicator_positions=[0, 180], spindle_positions=[0, 180], z_positions=z)

        I00 = convert_str_to_sp('4.98887, 4.98451, 4.98015, 4.97578, 4.97142, 4.96706, 4.96269, 4.95833, 4.95397, 4.94960, 4.94524')
        I1800 = convert_str_to_sp('5.01527, 5.02836, 5.04145, 5.05454, 5.06763, 5.08072, 5.09381, 5.10690, 5.11999, 5.13308, 5.14617')
        I0180 = convert_str_to_sp('4.99775, 5.01084, 5.02393, 5.03702, 5.05011, 5.06320, 5.07629, 5.08938, 5.10247, 5.11556, 5.12865')
        I180180 = convert_str_to_sp('5.00655, 5.00219, 4.99782, 4.99346, 4.98910, 4.98473, 4.98037, 4.97601, 4.97164, 4.96728, 4.96292')

        series = [
            MeasurementSeries(0, 0, list(zip(z, I00))),
            MeasurementSeries(180, 0, list(zip(z, I1800))),
            MeasurementSeries(0, 180, list(zip(z, I0180))),
            MeasurementSeries(180, 180, list(zip(z, I180180))),
        ]
        raw = RawMeasurementData(config, series)

        exp_t = Tilts(
            alpha_x=sp.Rational(0),
            alpha_y=slope_from_deg("-0.1"),
            beta_x=sp.Rational(0),
            beta_y=slope_from_deg("-0.05"),
            gamma=slope_from_deg("0.025"),
        )

        return TestCase(
            name="Real Data 1", description="CAD data: ay=-0.1°, by=-0.05°, g=0.025°", config=config, raw_data=raw, expected_tilts=exp_t
        )

    def _create_real_data_test_2(self) -> TestCase:
        """Real CAD data test 2."""
        z = [sp.Rational(i * 10) for i in range(11)]
        config = SpindleConfiguration(indicator_positions=[0, 180], spindle_positions=[0, 180], z_positions=z)

        I00 = convert_str_to_sp('4.957592, 4.87905, 4.800509, 4.721967, 4.643426, 4.564885, 4.486343, 4.407802, 4.32926, 4.250719, 4.172177')
        I1800 = convert_str_to_sp('5.048135, 5.144131, 5.240127, 5.336123, 5.432119, 5.528115, 5.624111, 5.720108, 5.816104, 5.91210, 6.008096')
        I0180 = convert_str_to_sp('4.976310, 4.932676, 4.889043, 4.845409, 4.801776, 4.758142, 4.714509, 4.670875, 4.627242, 4.583608, 4.539975')
        I180180 = convert_str_to_sp('5.030635, 5.091722, 5.152810, 5.213897, 5.274984, 5.336071, 5.397159, 5.458246, 5.519333, 5.580421, 5.641508')

        #string_parts = rational_string.split(',')

        series = [
            MeasurementSeries(0, 0, list(zip(z, I00))),
            MeasurementSeries(180, 0, list(zip(z, I1800))),
            MeasurementSeries(0, 180, list(zip(z, I0180))),
            MeasurementSeries(180, 180, list(zip(z, I180180))),
        ]
        raw = RawMeasurementData(config, series)

        exp_t = Tilts(
            alpha_x=sp.Rational(0),
            alpha_y=slope_from_deg("-0.4"),
            beta_x=sp.Rational(0),
            beta_y=slope_from_deg("-0.1"),
            gamma=slope_from_deg("0.05"),
        )

        return TestCase(
            name="Real Data 2", description="CAD data: ay=-0.4°, by=-0.1°, g=0.05°", config=config, raw_data=raw, expected_tilts=exp_t
        )

    # Test 1: Compare with and without rigid body coupling
    def test_rigid_body_coupling_effect(self):
        """See how much difference rigid body coupling makes."""
        z_pos = [sp.Rational(i) for i in range(0, 101, 10)]
        config = SpindleConfiguration(
            indicator_positions=[sp.Rational(0), sp.Rational(180)],
            spindle_positions=[sp.Rational(0), sp.Rational(180)],
            nominal_diameter=sp.Rational(50),
            z_positions=z_pos,
        )

        equations = build_equation_system(config)

        # Generate with reasonable tilts
        raw_no_coupling, _, _ = generate_synthetic_data(equations, beta_y=slope_from_deg(sp.Rational("0.1")), use_rigid_body=False)

        raw_with_coupling, _, _ = generate_synthetic_data(equations, beta_y=slope_from_deg(sp.Rational("0.1")), use_rigid_body=True)

        raw_no_coupling.print_comparison(raw_with_coupling, "Without Coupling", "With Coupling")

    # Test 2: Compare against real data
    def test_rigid_body_vs_real_data_1(self):
        """Generate synthetic data and compare to real_data_1."""
        # Use real_data_1 configuration and expected tilts
        z_positions = [sp.Rational(i) for i in range(0, 101, 10)]
        config = SpindleConfiguration(
            indicator_positions=[sp.Rational(0), sp.Rational(180)],
            spindle_positions=[sp.Rational(0), sp.Rational(180)],
            nominal_diameter=sp.Rational(50),
            z_positions=z_positions,
        )

        equations = build_equation_system(config)

        # Load real data
        real_data = self._create_real_data_test_1()

        # Generate with coupling
        synthetic_with_coupling, _, _ = generate_synthetic_data(
            equations,
            alpha_y=slope_from_deg(sp.Rational("-0.1")),
            beta_y=slope_from_deg(sp.Rational("-0.05")),
            gamma=slope_from_deg(sp.Rational("0.025")),
            use_rigid_body=True,
        )

        synthetic_with_coupling.print_comparison(real_data.raw_data, "Synthetic (with coupling)", "Real Data 1")

    def test_rigid_body_vs_real_data_2(self):
        """Generate synthetic data and compare to real_data_1."""
        # Use real_data_1 configuration and expected tilts
        z_positions = [sp.Rational(i) for i in range(0, 101, 10)]
        config = SpindleConfiguration(
            indicator_positions=[sp.Rational(0), sp.Rational(180)],
            spindle_positions=[sp.Rational(0), sp.Rational(180)],
            nominal_diameter=sp.Rational(50),
            z_positions=z_positions,
        )

        equations = build_equation_system(config)

        # Load real data
        real_data = self._create_real_data_test_2()

        # Generate with coupling
        synthetic_with_coupling, _, _ = generate_synthetic_data(
            equations,
            alpha_y=real_data.expected_tilts["alpha_y"],
            beta_y=real_data.expected_tilts["beta_y"],
            gamma=real_data.expected_tilts["gamma"],
            use_rigid_body=True,
        )

        synthetic_without_coupling, _, _ = generate_synthetic_data(
            equations,
            alpha_y=real_data.expected_tilts["alpha_y"],
            beta_y=real_data.expected_tilts["beta_y"],
            gamma=real_data.expected_tilts["gamma"],
            use_rigid_body=True,
        )

        synthetic_with_coupling.print_comparison(real_data.raw_data, "Synthetic (with coupling)", "Real Data 2")

        synthetic_without_coupling.print_comparison(real_data.raw_data, "Synthetic (without coupling)", "Real Data 2")


    # --- Test Management ---

    def add_test(self, name: str, test_func: Callable[[], TestCase]):
        """Add a custom test to the library."""
        self.tests[name] = test_func

    def run_test(self, name: str):
        """Run a specific test by name."""
        if name not in self.tests:
            print(f"Test '{name}' not found.")
            self.list_tests()
            return None
        return self.tests[name]().run()

    def run_all_tests(self):
        """Run all tests in the library."""
        return {name: self.tests[name]().run() for name in self.tests}

    def list_tests(self):
        """List all available tests."""
        print("\nAvailable Tests:\n" + "=" * 70)
        [print(f"  {name}") for name in self.tests]
        print("=" * 70)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    test_lib = TestLibrary()

    # Run specific test
    # test_lib.run_test("synthetic_coupled")
    test_lib.run_test("synthetic_coupled_complex")
    # test_lib.run_test("symbolic_90_coupled")
    # test_lib.run_test("real_data_2")
    # test_lib.test_rigid_body_coupling_effect()
    # test_lib.test_rigid_body_vs_real_data_2()
