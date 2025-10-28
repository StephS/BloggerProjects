# ============================================================================
# File 3: solver.py
# ============================================================================
"""
Main solving logic for spindle metrology system. Handles symbolic analysis
and numerical solving using a LINEARIZED model suitable for least squares.
"""

import sympy as sp
from typing import List, Dict, Optional, Tuple, Set

from models import (
    SpindleConfiguration,
    RawMeasurementData,
    EquationSystem,
    SolutionResult,
    MeasurementSeries,
    CouplingAssessment,
    CouplingSeverity,
    Tilts,
    Forms,
    SystemType,
    LinearSystemSolution,
    ConfigKey,
)

from utils import DEBUG, USE_UNICODE, _least_squares_core, get_surface_angle, format_tilt_angle

# ============================================================================
# Pure Linear Algebra (Domain-Agnostic)
# ============================================================================


def fit_linear(z_values: List[sp.Rational], I_values: List[sp.Rational]) -> Tuple[sp.Rational, sp.Rational]:
    """
    Fits I = F + T·z using exact least squares.

    This separates measurements into:
    - T (slope): Combined tilt effect
    - F (intercept): Combined form effect at z=0

    Returns:
        (T_slope, F_intercept) tuple
    """
    return _least_squares_core(z_values, I_values)


def _solve_linear_system(
    equations: Dict[ConfigKey, sp.Expr], variables: List[sp.Symbol], knowns: Dict[ConfigKey, sp.Rational | sp.Symbol]
) -> LinearSystemSolution:
    """
    Core linear system solver with coupling analysis using pseudo-inverse.

    COUPLING DETECTION LOGIC:
    1. Compute nullspace of A (directions where A·v = 0)
    2. Each nullspace vector represents a coupling constraint
    3. Classify variables:
       - Independent: Uniquely determined (not in any coupling)
       - Dependent: Coupled to others (can't separate, but combination is known)
       - Free: Unconstrained (column of A is zero - no effect on measurements)

    COMBINED SOLUTIONS CALCULATION:
    - Nullspace equation: v·x = 0 means lhs = rhs (e.g., α_x = γ)
    - We use lhs + rhs (NOT lhs - rhs) to get the combined value
    - Why? Because lhs - rhs = 0 always (useless for comparison)
    - Instead: lhs + rhs gives the actual sum that measurements constrain
    - Substitute min-norm solution to find this combined value

    Args:
        equations: Dict mapping ConfigKey to symbolic expressions
        variables: List of variables to solve for
        knowns: Dict mapping ConfigKey to known measurement values

    Returns:
        LinearSystemSolution with classified variables and coupling info
    """
    DEBUG.enter_section("Core Linear System Solver")

    # Build linear system Ax = b from equations
    system_eqs = [sp.Eq(equations[key], knowns[key]) for key in equations]

    try:
        A, b = sp.linear_eq_to_matrix(system_eqs, variables)
    except Exception as e:
        DEBUG.error(f"Failed to create matrix: {e}")
        DEBUG.info("Equations:", title="System Equations")
        [DEBUG.info(str(eq), nocaller=True) for eq in system_eqs]
        DEBUG.info("Variables:", title="Variables")
        DEBUG.info(str(variables), nocaller=True)
        raise

    rank = A.rank()
    DEBUG.info(f"Matrix rank: {rank}/{len(variables)}")

    # Compute minimum norm solution using Moore-Penrose pseudo-inverse
    DEBUG.info("Computing Moore-Penrose pseudo-inverse...")
    A_pinv = A.pinv()
    solution_vector = (A_pinv * b).applyfunc(sp.simplify)

    complete_solution = {var: sol for var, sol in zip(variables, solution_vector)}

    # Analyze nullspace for coupling
    null_space = A.nullspace()
    coupling_equations: List[sp.Eq] = [sp.Eq(vec.dot(sp.Matrix(variables)), 0).simplify() for vec in null_space]

    # Classify variables
    independent, dependent, free, combined_solutions = {}, set(), set(), []

    for eq in coupling_equations:
        eq_symbols = eq.free_symbols
        if not eq_symbols:
            continue

        # Check for trivial coupling: Eq(var, 0) where var has no effect
        is_trivial_coupling = eq.lhs.is_Symbol and eq.rhs == sp.S.Zero

        if is_trivial_coupling:
            # Verify this variable truly has no effect (zero column in A)
            var_index = variables.index(eq.lhs)
            if A.col(var_index).is_zero_matrix:
                free.update(eq_symbols)
                DEBUG.info(f"Identified free variable (no effect on measurements): {eq.lhs}")
            else:
                # Should not happen if nullspace is correct, but handle defensively
                dependent.update(eq_symbols)
                DEBUG.warning(f"Unexpected trivial coupling for non-zero column: {eq}")

        elif eq.lhs == sp.S.Zero or eq.rhs == sp.S.Zero:
            # One side is zero - might indicate free variable
            free.update(eq_symbols)

        else:
            # Real coupling between variables
            dependent.update(eq_symbols)

            # Compute combined value: lhs + rhs (NOT lhs - rhs)
            # The nullspace tells us lhs = rhs, so lhs + rhs is the constrained sum
            expr = eq.lhs + eq.rhs  # pyright: ignore[reportOperatorIssue]

            # Substitute min-norm solution to find the value
            try:
                combined_value = expr.subs(complete_solution).simplify()
                combined_solutions.append(sp.Eq(expr, combined_value))
            except Exception as e:
                DEBUG.error(f"Error evaluating combined expression {expr}: {e}")
                combined_solutions.append(sp.Eq(expr, sp.Symbol("ERROR")))

    # Variables not coupled or free are independent
    for var in variables:
        if var not in dependent and var not in free:
            independent[var] = complete_solution[var]

    DEBUG.info(f"Classified: {len(independent)} independent, {len(dependent)} coupled, {len(free)} free")
    DEBUG.exit_section()

    return LinearSystemSolution(
        variables=variables,
        rank=rank,
        complete_solution=complete_solution,
        independent=independent,
        dependent=dependent,
        free=free,
        coupling_equations=coupling_equations,
        combined_solutions=combined_solutions,
        matrix_A=A,
        matrix_b=b,
        solution_vector=solution_vector,
    )


def _calculate_residuals(
    solution: LinearSystemSolution, measurement_keys: List[ConfigKey]
) -> Tuple[Dict[ConfigKey, sp.Rational], sp.Rational]:
    """
    Calculate residuals for numerical solution.

    Args:
        solution: Solved linear system
        measurement_keys: Keys for organizing residuals by measurement

    Returns:
        (residuals_dict, residual_norm)
    """
    predicted = solution.matrix_A * solution.solution_vector
    residual_vector = predicted - solution.matrix_b

    residuals = {key: residual_vector[i, 0] for i, key in enumerate(measurement_keys)}
    residual_norm = residual_vector.norm()

    return residuals, residual_norm


# ============================================================================
# Domain-Specific Logic (Linearized Model)
# ============================================================================


def build_equation_system(config: SpindleConfiguration) -> EquationSystem:
    """
    Generates the symbolic equation system based on the LINEARIZED model.

    Model: I(θ_ind, θ_spin, z) = F + T·z

    Where:
        F = M_x·sin(θ_ind) + M_y·cos(θ_ind) + S_θ_surf
        T = α_x·sin(θ_ind) + α_y·cos(θ_ind) + β_x·sin(θ_surf) + β_y·cos(θ_surf) + γ

    And:
        θ_surf = (360° - (θ_ind - θ_spin)) mod 360°  [CCW to CW conversion]

    Returns:
        EquationSystem with symbolic equations for all configurations
    """
    DEBUG.enter_section("Building Equation System (Linearized Model)")

    # Determine which surface angles we'll need
    surface_angles = {int(get_surface_angle(ind, spin)) for ind in config.indicator_positions for spin in config.spindle_positions}

    # Create symbolic variables
    tilts = Tilts.create_symbolic()
    forms = Forms.create_symbolic(surface_angles)
    z_i = sp.Symbol("z_i", real=True)

    tilt_equations: Dict[ConfigKey, sp.Expr] = {}
    form_equations: Dict[ConfigKey, sp.Expr] = {}
    parametric_equations: Dict[ConfigKey, sp.Expr] = {}

    for ind_pos in config.indicator_positions:
        for spin_pos in config.spindle_positions:
            key = ConfigKey(ind_pos, spin_pos)

            # Calculate angles
            theta_ind_rad = sp.rad(ind_pos)
            theta_surf_deg = get_surface_angle(ind_pos, spin_pos)
            theta_surf_rad = sp.rad(theta_surf_deg)

            # LINEARIZED Tilt equation: T = α·ind + β·surf + γ
            T_alpha = tilts.alpha_x * sp.sin(theta_ind_rad) + tilts.alpha_y * sp.cos(theta_ind_rad)
            T_beta = tilts.beta_x * sp.sin(theta_surf_rad) + tilts.beta_y * sp.cos(theta_surf_rad)
            T_gamma = tilts.gamma
            tilt_eq = sp.trigsimp(T_alpha + T_beta + T_gamma)
            tilt_equations[key] = tilt_eq

            # Form equation: F = M·ind + S
            S_symbol = forms[int(theta_surf_deg)]
            form_eq = forms.M_x * sp.sin(theta_ind_rad) + forms.M_y * sp.cos(theta_ind_rad) + S_symbol
            form_equations[key] = sp.trigsimp(form_eq)

            # Parametric equation: I(z) = F + T·z
            parametric_equations[key] = form_eq + tilt_eq * z_i  # pyright: ignore[reportOperatorIssue]

    DEBUG.success(f"Generated {len(tilt_equations)} linearized equation sets")
    DEBUG.exit_section()

    return EquationSystem(
        config=config,
        tilts=tilts,
        forms=forms,
        tilt_equations=tilt_equations,
        form_equations=form_equations,
        parametric_equations=parametric_equations,
        z_param=z_i,
    )


def separate_measurements(raw_data: RawMeasurementData) -> Tuple[Dict[ConfigKey, sp.Rational], Dict[ConfigKey, sp.Rational]]:
    """
    Separates raw I(z) series into T (slope) and F (intercept).

    For each measurement configuration:
    - Fit: I = F + T·z
    - Extract T (tilt slope) and F (form at z=0)

    Returns:
        (tilt_measurements, form_measurements) tuple of dicts
    """
    DEBUG.enter_section("Separating Measurements via Linear Fit")

    tilt_measurements: Dict[ConfigKey, sp.Rational] = {}
    form_measurements: Dict[ConfigKey, sp.Rational] = {}

    for series in raw_data.series:
        key = series.get_configuration_key()
        z_values = series.get_z_values()
        I_values = series.get_I_values()

        # Fit I = F + T·z
        T_slope, F_intercept = fit_linear(z_values, I_values)

        tilt_measurements[key] = T_slope
        form_measurements[key] = F_intercept
        DEBUG.info(f"{key} -> Tilt={format_tilt_angle(T_slope)}")

    DEBUG.success(f"Separated {len(tilt_measurements)} configurations")
    DEBUG.exit_section()

    return tilt_measurements, form_measurements


def solve_tilts(equations: EquationSystem, raw_data: Optional[RawMeasurementData] = None) -> SolutionResult:
    """
    Solves the LINEAR tilt subsystem (symbolic or numerical).

    The tilt equation is: T = α·ind + β·surf + γ
    Where each configuration gives one equation, solving for 5 tilts.

    Args:
        equations: Symbolic equation system
        raw_data: Optional measurement data for numerical solve

    Returns:
        SolutionResult containing tilt solutions
    """
    DEBUG.enter_section("Solving Tilts (Linear System)")
    is_numerical = raw_data is not None

    knowns: Dict[ConfigKey, sp.Rational | sp.Symbol]
    if is_numerical:
        DEBUG.info("Mode: NUMERICAL")
        knowns, _ = separate_measurements(raw_data)
    else:
        DEBUG.info("Mode: SYMBOLIC")
        # Create symbolic measurement values
        knowns = {key: sp.Symbol(f"T_{key}") for key in equations.measurement_keys()}

    solution = _solve_linear_system(equations.tilt_equations, equations.tilts.to_list(), knowns)

    residuals, residual_norm = None, None
    if is_numerical:
        residuals, residual_norm = _calculate_residuals(solution, equations.measurement_keys())

    DEBUG.exit_section()

    return SolutionResult(
        system_type=SystemType.TILTS,
        config=equations.config,
        is_numerical=is_numerical,
        linear_solution=solution,
        residuals=residuals,
        residual_norm=residual_norm,
    )


def solve_forms(
    equations: EquationSystem, raw_data: RawMeasurementData, tilt_solution: SolutionResult, remove_offset: bool = True
) -> SolutionResult:
    """
    Solves the LINEAR forms subsystem numerically at each Z position.

    At each Z, we compute: F = I - T·z
    Then solve: F = M·ind + S for machine and surface straightness.

    Args:
        equations: Symbolic equation system
        raw_data: Measurement data
        tilt_solution: Previously solved tilts
        remove_offset: Whether to detrend forms after solving

    Returns:
        SolutionResult with Forms instance containing solved profiles
    """
    DEBUG.enter_section("Solving Forms at Each Z (Linear System)")

    if not tilt_solution.is_numerical:
        raise ValueError("Tilt solution must be numerical")
    if equations.config.z_positions is None:
        raise ValueError("Config must have z_positions")

    eq_forms_list = equations.forms.to_list()
    # Initialize lists for each form variable (use symbols from equations)
    form_series: Dict[sp.Symbol, List[sp.Expr]] = {var: [] for var in eq_forms_list}

    last_solution: Optional[LinearSystemSolution] = None

    for z_idx, z_pos in enumerate(equations.config.z_positions):
        knowns_at_z: Dict[ConfigKey, sp.Rational] = {}

        for series in raw_data.series:
            key = series.get_configuration_key()
            z_measured, I_measured = series.data_points[z_idx]

            if z_measured != z_pos:
                raise ValueError(f"Z position mismatch at index {z_idx}")

            # Compute F = I - T·z using complete tilt solution
            tilt_eq = equations.tilt_equations[key]
            T_value = tilt_eq.subs(tilt_solution.complete_solution)
            F_at_z = I_measured - T_value * z_pos  # pyright: ignore[reportOperatorIssue]
            knowns_at_z[key] = sp.simplify(F_at_z)

        # Solve for forms at this Z
        solution = _solve_linear_system(equations.form_equations, eq_forms_list, knowns_at_z)
        last_solution = solution

        # Store results using equations.forms symbols
        for var in eq_forms_list:
            form_series[var].append(solution.complete_solution[var])

    # Build Forms instance with solved series (using equations.forms structure)
    # This ensures we use the SAME symbols as in equations, not new ones
    forms_solution = Forms(M_x=form_series[equations.forms.M_x], M_y=form_series[equations.forms.M_y], S={})

    # Populate S dict using the symbols from equations.forms
    for var in equations.forms.surface_values():
        var_name = str(var)
        forms_solution.S[var_name] = form_series[var]

    # Detrend if requested
    if remove_offset:
        DEBUG.info("Removing linear trend from solved form series")
        forms_solution = forms_solution.remove_trends(equations.config.z_positions)

    DEBUG.success(f"Solved forms at {len(equations.config.z_positions)} Z positions")
    DEBUG.exit_section()

    assert last_solution is not None

    return SolutionResult(
        system_type=SystemType.FORMS,
        config=equations.config,
        is_numerical=True,
        linear_solution=last_solution,
        forms_solution=forms_solution,
    )


def run_analysis(config: SpindleConfiguration, raw_data: Optional[RawMeasurementData] = None) -> Tuple[SolutionResult, SolutionResult]:
    """
    Executes the complete LINEAR analysis pipeline.

    Workflow:
    1. Build symbolic equation system
    2. Solve tilts (extracts slopes from measurements)
    3. Solve forms (extracts shapes at each Z, accounting for tilts)

    Args:
        config: Spindle configuration
        raw_data: Optional measurement data (None for symbolic analysis)

    Returns:
        (tilt_result, form_result) tuple
    """
    mode = "NUMERICAL" if raw_data else "SYMBOLIC"
    DEBUG.enter_section(f"{mode} Analysis Pipeline (Linear Solver)")

    equations = build_equation_system(config)
    tilt_result = solve_tilts(equations, raw_data)

    form_result: SolutionResult
    if raw_data:
        form_result = solve_forms(equations, raw_data, tilt_result)
    else:
        # Symbolic form analysis
        form_knowns = {key: sp.Symbol(f"F_{key}") for key in equations.measurement_keys()}
        form_solution = _solve_linear_system(equations.form_equations, equations.forms.to_list(), form_knowns)
        form_result = SolutionResult(
            system_type=SystemType.FORMS, config=equations.config, is_numerical=False, linear_solution=form_solution
        )

    DEBUG.exit_section()
    return tilt_result, form_result


def assess_rigid_body_coupling(config: SpindleConfiguration, max_expected_tilt_deg: float = 0.1) -> CouplingAssessment:
    """
    Assess significance of rigid body coupling (LINEAR APPROXIMATION).

    RIGID BODY Z-OFFSET:
    When the artifact is tilted relative to the machine guideway, the indicator
    doesn't measure at the commanded Z position. Instead, it measures at:
        Z_artifact ≈ Z_commanded + R·tan(tilt) ≈ Z_commanded + R·tilt_slope

    This first-order approximation is valid for small tilts and is consistent
    with our linearized model.

    GAMMA EXCLUSION:
    Gamma (cone angle) is NOT included in Z-offset calculation because:
    1. Gamma affects effective radius, not Z position (at first order)
    2. Including gamma would introduce second-order terms (gamma·R·tilt)
    3. Our linearized model deliberately ignores second-order effects

    Args:
        config: Spindle configuration
        max_expected_tilt_deg: Maximum expected tilt in degrees

    Returns:
        CouplingAssessment with severity and warnings
    """
    diameter_mm = config.nominal_diameter
    max_tilt_rad = sp.rad(sp.Rational(max_expected_tilt_deg))

    # First-order Z-offset: ΔZ ≈ R·tan(tilt) ≈ R·tilt (for small angles)
    max_z_offset_mm = config.radius * sp.tan(max_tilt_rad)

    z_range_mm = config.get_z_range()
    coupling_ratio = None
    warnings = []

    if z_range_mm is not None and z_range_mm > 0:
        coupling_ratio = abs(max_z_offset_mm / z_range_mm)

    # Classify severity
    severity = CouplingSeverity.NEGLIGIBLE
    if coupling_ratio is not None:
        if coupling_ratio >= sp.Rational(1, 10):
            severity = CouplingSeverity.SEVERE
            warnings.append("Coupling ratio > 10% is SEVERE!")
        elif coupling_ratio >= sp.Rational(5, 100):
            severity = CouplingSeverity.SIGNIFICANT
            warnings.append("Coupling ratio > 5% is significant.")
        elif coupling_ratio >= sp.Rational(1, 100):
            severity = CouplingSeverity.MODERATE
            warnings.append("Coupling ratio > 1% is moderate.")
    elif abs(max_z_offset_mm) > sp.Rational(1, 10):
        severity = CouplingSeverity.SEVERE
    elif abs(max_z_offset_mm) > sp.Rational(5, 100):
        severity = CouplingSeverity.SIGNIFICANT
    elif abs(max_z_offset_mm) > sp.Rational(1, 100):
        severity = CouplingSeverity.MODERATE

    return CouplingAssessment(
        max_z_offset_mm=max_z_offset_mm,
        z_range_mm=z_range_mm,
        coupling_ratio=coupling_ratio,
        severity=severity,
        details={"diameter_mm": diameter_mm, "max_tilt_deg": max_expected_tilt_deg, "max_tilt_rad": max_tilt_rad},
        warnings=warnings,
    )
