# ============================================================================
# File 2: models.py
# ============================================================================
"""
Data models and type definitions for spindle solver.
All dataclasses and type structures used throughout the system.
"""

import sympy as sp
from typing import Callable, List, Dict, Set, Any, Tuple, Optional, Union, TypeAlias, NamedTuple
from dataclasses import InitVar, dataclass, field
from enum import Enum
from utils import (
    DEBUG,
    Colors,
    USE_UNICODE,
    format_tilt_angle,
    remove_form_trend,
    preview_series as util_preview_series,
    format_comparison_stats,
)

# ============================================================================
# Type Aliases
# ============================================================================

SolutionValue: TypeAlias = Union[sp.Expr, sp.Rational]
ProfileCallable: TypeAlias = Callable[[sp.Rational], sp.Rational]  # Type hint for profile functions


# ============================================================================
# Configuration Key
# ============================================================================


class ConfigKey(NamedTuple):
    """
    Configuration key for (indicator_position, spindle_position) pairs.

    Using NamedTuple provides:
    - Type safety (can't accidentally swap indicator/spindle)
    - Fast hashing (tuples are optimized in Python)
    - Clean string representation
    - IDE autocomplete support
    """

    indicator: sp.Rational
    spindle: sp.Rational

    def __str__(self) -> str:
        return f"({int(self.indicator)},{int(self.spindle)})"


class CompareResult(NamedTuple):
    """Result of comparing two values (true vs solved)."""

    true: List[sp.Rational] | sp.Rational
    solved: List[sp.Rational] | sp.Rational
    error: sp.Rational


# ============================================================================
# Enums
# ============================================================================


class SystemType(Enum):
    """Type of subsystem being analyzed."""

    TILTS = "Tilts"
    FORMS = "Forms"


class CouplingSeverity(Enum):
    """Classification levels for rigid body coupling significance."""

    NEGLIGIBLE = "NEGLIGIBLE"
    MODERATE = "MODERATE"
    SIGNIFICANT = "SIGNIFICANT"
    SEVERE = "SEVERE"


# ============================================================================
# Core Variable Classes
# ============================================================================


@dataclass
class Tilts:
    """
    Tilt variables - symbolic or numerical.

    Contains all 5 tilt parameters:
    - alpha_x, alpha_y: Machine tilt (machine frame reference, doesn't rotate with spindle)
    - beta_x, beta_y: Setup tilt (spindle/artifact reference, rotates with spindle)
    - gamma: Artifact cone angle (intrinsic geometry, doesn't change with rotation)

    Can be symbolic (sp.Symbol) or numerical (sp.Expr/sp.Rational).
    """

    alpha_x: sp.Symbol | sp.Expr
    alpha_y: sp.Symbol | sp.Expr
    beta_x: sp.Symbol | sp.Expr
    beta_y: sp.Symbol | sp.Expr
    gamma: sp.Symbol | sp.Expr

    @property
    def is_symbolic(self) -> bool:
        return isinstance(self.alpha_x, sp.Symbol)

    @property
    def is_numerical(self) -> bool:
        return not self.is_symbolic

    @classmethod
    def create_symbolic(cls) -> "Tilts":
        """Create symbolic tilt variables for equation building."""
        return cls(
            alpha_x=sp.Symbol("alpha_x", real=True),
            alpha_y=sp.Symbol("alpha_y", real=True),
            beta_x=sp.Symbol("beta_x", real=True),
            beta_y=sp.Symbol("beta_y", real=True),
            gamma=sp.Symbol("gamma", real=True),
        )

    def __iter__(self):
        yield self.alpha_x
        yield self.alpha_y
        yield self.beta_x
        yield self.beta_y
        yield self.gamma

    def __getitem__(self, key: str) -> sp.Symbol | sp.Expr:
        return getattr(self, key)

    def __setitem__(self, key: str, value: sp.Symbol | sp.Expr) -> None:
        setattr(self, key, value)

    def to_list(self) -> List[sp.Symbol | sp.Expr]:
        return [self.alpha_x, self.alpha_y, self.beta_x, self.beta_y, self.gamma]

    def to_dict(self) -> Dict[str, sp.Symbol | sp.Expr]:
        return {"alpha_x": self.alpha_x, "alpha_y": self.alpha_y, "beta_x": self.beta_x, "beta_y": self.beta_y, "gamma": self.gamma}

    def compare_to(self, other: "Tilts", tolerance: sp.Rational = sp.Rational(1, 1000000)) -> Dict[str, Tuple[bool, sp.Rational]]:
        """
        Compare this Tilts instance to another.

        Args:
            other: Another Tilts instance to compare against
            tolerance: Maximum acceptable difference for match

        Returns:
            Dict of {var_name: (matches, difference)} for each tilt variable
        """
        results = {}
        for name in ["alpha_x", "alpha_y", "beta_x", "beta_y", "gamma"]:
            val_self = self[name]
            val_other = other[name]
            diff = abs(val_self - val_other) # pyright: ignore[reportOperatorIssue]
            matches = diff <= tolerance
            results[name] = (matches, diff)
        return results

    def format_angles(self) -> str:
        """Format all tilts with both slope and angle representations."""
        lines = []
        for name in ["alpha_x", "alpha_y", "beta_x", "beta_y", "gamma"]:
            val = self[name]
            if isinstance(val, (sp.Rational, sp.Float, int, float)):
                lines.append(f"{name:8} = {format_tilt_angle(val)}")
            else:
                lines.append(f"{name:8} = {val}")
        return "\n".join(lines)


@dataclass
class Forms:
    """
    Form variables - symbolic or numerical.

    Contains:
    - M_x, M_y: Machine straightness errors (horizontal/vertical)
    - S: Dictionary of surface straightness errors by angle

    Can be:
    - Symbolic: sp.Symbol (for equation building)
    - Numerical scalar: sp.Expr/sp.Rational (single value)
    - Numerical series: List[sp.Expr] (form profile at multiple Z positions)
    """

    M_x: sp.Symbol | sp.Expr | List[sp.Expr] | List[sp.Rational]
    M_y: sp.Symbol | sp.Expr | List[sp.Expr] | List[sp.Rational]
    S: Dict[str, sp.Symbol | sp.Expr | List[sp.Expr] | List[sp.Rational]] = field(default_factory=dict)

    @property
    def is_symbolic(self) -> bool:
        return isinstance(self.M_x, sp.Symbol)

    @property
    def is_numerical(self) -> bool:
        return not self.is_symbolic

    @classmethod
    def create_symbolic(cls, surface_angles: Set[int]) -> "Forms":
        """Create symbolic form variables for equation building."""
        forms = cls(M_x=sp.Symbol("M_x", real=True), M_y=sp.Symbol("M_y", real=True))
        for angle in sorted(surface_angles):
            forms.add_surface_symbol(angle)
        return forms

    @classmethod
    def _to_surf_key(cls, angle: int) -> str:
        """Convert angle to surface key string."""
        return f"S_{{{int(angle)}}}"

    @classmethod
    def _to_angle(cls, var_name: str) -> int:
        """Extract angle from surface key string."""
        return int(var_name.split("_")[1].strip("{}"))

    def add_surface_symbol(self, angle: int) -> None:
        """Add a surface variable at the specified angle."""
        key = self._to_surf_key(angle)
        if key not in self.S:
            self.S[key] = sp.Symbol(key, real=True)

    def __getitem__(self, key: str | int | sp.Symbol) -> sp.Symbol | sp.Expr | List[sp.Expr] | List[sp.Rational]:
        # Handle Symbol keys (allows passing symbols directly)
        if isinstance(key, sp.Symbol):
            key = key.name

        if isinstance(key, int):
            return self.S[self._to_surf_key(key)]
        elif key in ["M_x", "M_y"]:
            return getattr(self, key)
        else:
            return self.S[key]

    def __setitem__(self, key: str | int, value: sp.Symbol | sp.Expr | List[sp.Expr] | List[sp.Rational]) -> None:
        if isinstance(key, int):
            self.S[self._to_surf_key(key)] = value
        elif key in ["M_x", "M_y"]:
            setattr(self, key, value)
        else:
            self.S[key] = value

    def to_list(self) -> List[sp.Symbol | sp.Expr | List[sp.Expr] | List[sp.Rational]]:
        """Return all form variables as ordered list: [M_x, M_y, S_0, S_90, ...]"""
        result: List[sp.Symbol | sp.Expr | List[sp.Expr] | List[sp.Rational]] = [self.M_x, self.M_y]
        result.extend(self.surface_values())
        return result

    def surface_values(self) -> List[sp.Symbol | sp.Expr | List[sp.Expr] | List[sp.Rational]]:
        """Return sorted list of surface form values."""
        # Always sort to ensure consistent ordering
        surface_keys = sorted(self.S.keys(), key=self._to_angle)
        return [self.S[k] for k in surface_keys]

    def to_dict(self) -> Dict[str, sp.Symbol | sp.Expr | List[sp.Expr] | List[sp.Rational]]:
        """Return all form variables as dictionary."""
        return {"M_x": self.M_x, "M_y": self.M_y, **self.S}

    def compare_to(self, other: "Forms", tolerance: sp.Rational = sp.Rational(1, 1000000)) -> Dict[str, Tuple[bool, Optional[sp.Rational]]]:
        """
        Compare this Forms instance to another.

        For numerical forms (lists), compares entire series and returns max difference.
        For symbolic/scalar forms, compares values directly.

        Args:
            other: Another Forms instance to compare against
            tolerance: Maximum acceptable difference for match

        Returns:
            Dict of {var_name: (matches, max_diff)} for each form variable
        """
        results = {}
        all_vars = ["M_x", "M_y"] + list(self.S.keys())

        for var_name in all_vars:
            val_self = self[var_name]
            val_other = other[var_name]

            # Handle list (numerical series) vs symbol (symbolic)
            if isinstance(val_self, list) and isinstance(val_other, list):
                if len(val_self) != len(val_other):
                    results[var_name] = (False, None)
                    continue
                diffs = [abs(v1 - v2) for v1, v2 in zip(val_self, val_other)] # pyright: ignore[reportOperatorIssue]
                max_diff = max(diffs) if diffs else sp.Rational(0)
                matches = max_diff <= tolerance
                results[var_name] = (matches, max_diff)
            else:
                # Symbolic or scalar comparison
                diff = abs(val_self - val_other) if isinstance(val_self, sp.Expr) else None # pyright: ignore[reportOperatorIssue]
                matches = diff <= tolerance if diff is not None else val_self == val_other
                results[var_name] = (matches, diff)

        return results

    def remove_trends(self, z_positions: List[sp.Rational]) -> "Forms":
        """
        Return new Forms instance with linear trends removed from all form series.

        This ensures forms contain only shape deviations, not hidden tilts.
        Only works on numerical forms (lists).

        Args:
            z_positions: Z positions corresponding to form values

        Returns:
            New Forms instance with detrended series

        Raises:
            ValueError: If forms are symbolic or not lists
        """
        if self.is_symbolic:
            raise ValueError("Cannot remove trends from symbolic forms - forms must be numerical (lists)")

        if not isinstance(self.M_x, list) or not isinstance(self.M_y, list):
            raise ValueError("Cannot remove trends from non-list forms - M_x and M_y must be numerical series (lists)")

        new_forms = Forms(M_x=remove_form_trend(self.M_x, z_positions), M_y=remove_form_trend(self.M_y, z_positions), S={})

        for key, values in self.S.items():
            if isinstance(values, list):
                new_forms.S[key] = remove_form_trend(values, z_positions)
            else:
                new_forms.S[key] = values

        return new_forms


# ============================================================================
# Configuration and Data Classes
# ============================================================================


@dataclass
class SpindleConfiguration:
    """Complete physical configuration including geometry."""

    indicator_angles: InitVar[str]
    spindle_angles: InitVar[str]
    indicator_positions: List[sp.Rational] = field(init=False)
    spindle_positions: List[sp.Rational] = field(init=False)
    nominal_diameter: sp.Rational = sp.Rational(50)
    z_positions: Optional[List[sp.Rational]] = None

    def __post_init__(self, indicator_positions_str, spindle_positions_str) -> None:
        from utils import convert_str_to_sympy  # Import here to avoid circular dependency
        self.indicator_positions = convert_str_to_sympy(indicator_positions_str) #[sp.Rational(p) for p in self.indicator_positions]
        self.spindle_positions = convert_str_to_sympy(spindle_positions_str) #[sp.Rational(p) for p in self.spindle_positions]
        # self.indicator_positions = [sp.Rational(p) for p in self.indicator_positions]
        # self.spindle_positions = [sp.Rational(p) for p in self.spindle_positions]
        self.nominal_diameter = sp.Rational(self.nominal_diameter)
        if self.z_positions is not None:
            self.z_positions = [sp.Rational(z) for z in self.z_positions]

    def get_z_range(self) -> Optional[sp.Rational]:
        if self.z_positions is None or len(self.z_positions) < 2:
            return None
        return max(self.z_positions) - min(self.z_positions)  # pyright: ignore[reportOperatorIssue]

    @property
    def radius(self) -> sp.Rational:
        return self.nominal_diameter / 2  # pyright: ignore[reportOperatorIssue]


@dataclass(frozen=True)
class MeasurementSeries:
    """Single measurement series I(z) for one configuration."""

    indicator_position: sp.Rational
    spindle_position: sp.Rational
    data_points: List[Tuple[sp.Rational, sp.Rational]]  # List of (z, I) tuples
    key: ConfigKey = field(init=False)

    def __post_init__(self) -> None:
        # Because frozen=True, must use object.__setattr__
        object.__setattr__(self, "indicator_position", sp.Rational(self.indicator_position))
        object.__setattr__(self, "spindle_position", sp.Rational(self.spindle_position))
        object.__setattr__(self, "key", ConfigKey(self.indicator_position, self.spindle_position))

        # Validate data points
        if len(self.data_points) < 2:
            raise ValueError(f"MeasurementSeries requires at least 2 data points at {self.key}, " f"but got {len(self.data_points)}")

        if not isinstance(self.data_points[0][1], sp.Basic):
            raise ValueError(
                f"MeasurementSeries data_points must be sympy types at {self.key}. "
                f"Use convert_str_to_sp() to convert lists of floats/strings."
            )

        if len(set(self.get_z_values())) < 2:
            raise ValueError(f"All Z values are identical in MeasurementSeries at {self.key}")

    def get_configuration_key(self) -> ConfigKey:
        """Return ConfigKey for this measurement configuration."""
        return self.key

    def get_z_values(self) -> List[sp.Rational]:
        return [z for z, I in self.data_points]

    def get_I_values(self) -> List[sp.Rational]:
        return [I for z, I in self.data_points]


@dataclass
class RawMeasurementData:
    """Collection of raw I(z) measurement series."""

    config: SpindleConfiguration
    series: List[MeasurementSeries]

    def __post_init__(self) -> None:
        expected_configs = {(ind, spin) for ind in self.config.indicator_positions for spin in self.config.spindle_positions}
        actual_configs = {(s.indicator_position, s.spindle_position) for s in self.series}

        if actual_configs != expected_configs:
            missing = expected_configs - actual_configs
            extra = actual_configs - expected_configs
            errors = []
            if missing:
                errors.append(f"Missing configurations: {missing}")
            if extra:
                errors.append(f"Extra configurations: {extra}")
            raise ValueError(". ".join(errors))

        self.series.sort(key=lambda s: (s.indicator_position, s.spindle_position))

    @classmethod
    def from_measurements(cls, config: SpindleConfiguration, measurements: Dict[Tuple[int, int], str]) -> "RawMeasurementData":
        """
        Create RawMeasurementData from measurement dictionary.

        This is the recommended way to load measurement data for analysis.

        Args:
            config: Configuration with z_positions defined
            measurements: Dict mapping (indicator_deg, spindle_deg) to comma-separated measurement string

        Returns:
            RawMeasurementData ready for analysis

        Raises:
            ValueError: If config missing z_positions or measurement count mismatch

        Example:
            >>> config = create_standard_config([0, 180], [0, 180], z_count=11)
            >>> measurements = {
            ...     (0, 0): "4.95, 4.87, 4.80, ...",
            ...     (180, 0): "5.04, 5.14, 5.24, ...",
            ...     (0, 180): "4.99, 5.01, 5.02, ...",
            ...     (180, 180): "5.00, 5.00, 4.99, ...",
            ... }
            >>> raw_data = RawMeasurementData.from_measurements(config, measurements)
        """
        from utils import convert_str_to_sympy  # Import here to avoid circular dependency

        if config.z_positions is None:
            raise ValueError("Configuration must have z_positions defined")

        series = []
        for (ind, spin), measurement_string in measurements.items():
            I_values = convert_str_to_sympy(measurement_string)

            if len(I_values) != len(config.z_positions):
                raise ValueError(
                    f"Measurement count mismatch at ({ind},{spin}): " f"got {len(I_values)} values, expected {len(config.z_positions)}"
                )

            series.append(MeasurementSeries(ind, spin, list(zip(config.z_positions, I_values))))

        return cls(config, series)

    def get_series_by_config(self, ind_pos: sp.Rational, spin_pos: sp.Rational) -> Optional[MeasurementSeries]:
        """Get measurement series for specific indicator/spindle configuration."""
        for series in self.series:
            if series.indicator_position == ind_pos and series.spindle_position == spin_pos:
                return series
        return None

    def get_series_by_key(self, key: ConfigKey) -> Optional[MeasurementSeries]:
        """Get measurement series by ConfigKey."""
        for series in self.series:
            if series.get_configuration_key() == key:
                return series
        return None

    def compare_to(self, other: "RawMeasurementData", tolerance: sp.Rational = sp.Rational(1, 1000000)) -> Dict[str, Any]:
        """Compare two measurement datasets (used for validation)."""
        if len(self.series) != len(other.series):
            raise ValueError("Cannot compare - different number of series")

        results = {"max_difference": sp.Rational(0), "rms_difference": sp.Rational(0), "differences_by_config": {}, "summary": []}
        all_diffs = []

        for s1 in self.series:
            key = s1.get_configuration_key()
            s2 = other.get_series_by_key(key)
            if s2 is None:
                raise ValueError(f"Config {key} not found in other dataset")

            z1_positions = s1.get_z_values()
            I1_values = s1.get_I_values()
            z2_positions = s2.get_z_values()
            I2_values = s2.get_I_values()

            if z1_positions != z2_positions:
                raise ValueError(f"Z position mismatch at {key}")

            # Detrend before comparison (removes bias from different tilt assumptions)
            I1_detrended = remove_form_trend(I1_values, z1_positions)
            I2_detrended = remove_form_trend(I2_values, z1_positions)

            config_diffs = [abs(v1 - v2) for v1, v2 in zip(I1_detrended, I2_detrended)] # pyright: ignore[reportOperatorIssue]
            all_diffs.extend(config_diffs)

            max_diff = max(config_diffs)
            rms_diff = sp.sqrt(sum(d**2 for d in config_diffs) / len(config_diffs))

            results["differences_by_config"][str(key)] = {"max": max_diff, "rms": rms_diff, "all_diffs": config_diffs}

            if max_diff > tolerance:
                results["summary"].append(f"{key}: max_diff = {float(max_diff):.6e}, rms = {float(rms_diff):.6e}")

        if all_diffs:
            results["max_difference"] = max(all_diffs)
            results["rms_difference"] = sp.sqrt(sum(d**2 for d in all_diffs) / len(all_diffs))

        return results

    def print_comparison(self, other: "RawMeasurementData", name1: str = "Dataset 1", name2: str = "Dataset 2"):
        """Print comparison results in readable format."""
        results = self.compare_to(other)

        print(f"\n{'='*70}\nCOMPARISON: {name1} vs {name2}\n{'='*70}")
        print(f"Overall max difference (detrended): {float(results['max_difference']):.8f}  ({float(results['max_difference']):.4e})")
        print(f"Overall RMS difference (detrended): {float(results['rms_difference']):.8f}  ({float(results['rms_difference']):.4e})")

        print(f"\nBy configuration (detrended):")
        for key, stats in results["differences_by_config"].items():
            print(format_comparison_stats(stats, label=f"  {key}"))

        if results["summary"]:
            print(f"\nConfigs with significant differences:\n  " + "\n  ".join(results["summary"]))

        print(f"{'='*70}")


# ============================================================================
# Equation System
# ============================================================================


@dataclass
class EquationSystem:
    """Complete symbolic equation system."""

    config: SpindleConfiguration
    tilt_symbols: Tilts  # Authoritative symbolic tilts
    form_symbols: Forms  # Authoritative symbolic forms
    tilt_equations: Dict[ConfigKey, sp.Expr]  # T equations by config
    form_equations: Dict[ConfigKey, sp.Expr]  # F equations by config
    parametric_equations: Dict[ConfigKey, sp.Expr]  # I(z) = F + T*z
    z_symbol: sp.Symbol  # Symbolic z parameter

    def measurement_keys(self) -> List[ConfigKey]:
        """Return sorted list of all measurement configuration keys."""
        return sorted(self.tilt_equations.keys())


# ============================================================================
# Solution Classes
# ============================================================================


@dataclass
class LinearSystemSolution:
    """
    Result from core linear algebra solver (domain-agnostic).

    This represents the pure mathematical solution to Ax = b using
    Moore-Penrose pseudo-inverse, including coupling analysis.

    COUPLING DETECTION LOGIC:
    The nullspace of A contains vectors v where A·v = 0.
    These represent directions in solution space that don't affect measurements.

    Variables are classified as:
    - Independent: Can be solved uniquely
    - Dependent (Coupled): Constrained together (can't separate, but sum is known)
    - Free: Completely unconstrained (no effect on measurements)

    The pseudo-inverse gives the MINIMUM NORM solution, which is unique.
    Other valid solutions differ only by nullspace components.
    """

    variables: List[sp.Symbol]
    rank: int
    complete_solution: Dict[sp.Symbol, sp.Expr]  # All variables -> min-norm solutions
    independent: Dict[sp.Symbol, sp.Expr]  # Uniquely solvable variables
    dependent: Set[sp.Symbol]  # Coupled variables
    free: Set[sp.Symbol]  # Unconstrained variables
    coupling_equations: List[sp.Eq]  # Nullspace constraints
    combined_solutions: List[sp.Eq]  # Combined values of coupled groups
    # Keep matrices for residual calculation
    matrix_A: sp.Matrix
    matrix_b: sp.Matrix
    solution_vector: sp.Matrix


@dataclass
class SolutionResult:
    """
    High-level solution result for user consumption (domain-specific).

    This wraps LinearSystemSolution with spindle-specific context and presentation.
    """

    system_type: SystemType
    config: SpindleConfiguration
    is_numerical: bool
    linear_solution: LinearSystemSolution  # Embed instead of duplicate
    forms_solution: Optional[Forms] = None  # Forms instance with solved values
    residuals: Optional[Dict[ConfigKey, sp.Rational]] = None
    residual_norm: Optional[sp.Rational] = None

    # Delegate properties to embedded linear_solution
    @property
    def independent(self) -> Dict[sp.Symbol, sp.Expr]:
        return self.linear_solution.independent

    @property
    def dependent(self) -> Set[sp.Symbol]:
        return self.linear_solution.dependent

    @property
    def free(self) -> Set[sp.Symbol]:
        return self.linear_solution.free

    @property
    def coupling_equations(self) -> List[sp.Eq]:
        return self.linear_solution.coupling_equations

    @property
    def combined_solutions(self) -> List[sp.Eq]:
        return self.linear_solution.combined_solutions

    @property
    def complete_solution(self) -> Dict[sp.Symbol, sp.Expr]:
        return self.linear_solution.complete_solution

    @property
    def rank(self) -> int:
        return self.linear_solution.rank

    @property
    def num_unknowns(self) -> int:
        return len(self.linear_solution.variables)

    @property
    def is_fully_solvable(self) -> bool:
        return len(self.free) == 0 and len(self.dependent) == 0

    def print(self) -> None:
        """Print formatted solution result."""
        mode = "Numerical" if self.is_numerical else "Symbolic"
        print(f"\n--- {mode} Analysis: {self.system_type.value} ---")
        print(f"Equations: {self.num_unknowns}")
        print(f"Rank:      {self.rank}")

        if self.is_fully_solvable:
            DEBUG.success("FULL RANK - all variables independently solvable")
        else:
            DEBUG.warning("RANK-DEFICIENT - contains coupled or free variables")

        if self.independent:
            print(f"\n--- Independent Solutions ({self.system_type.value}) ---")
            for var, val in self.independent.items():
                if self.is_numerical and self.system_type == SystemType.TILTS:
                    print(f"  {var} = {format_tilt_angle(val)}")
                elif self.is_numerical and self.system_type == SystemType.FORMS:
                    # Use utility function for form series
                    if self.forms_solution:
                        series = self.forms_solution[var]
                        print("  " + util_preview_series(series, str(var)))
                else:
                    sp.pprint(sp.Eq(var, sp.factor(val)), use_unicode=USE_UNICODE)

        if self.forms_solution is not None:
            print("\n--- Solved Form Series (Trend Removed) ---")
            for var_name, series in self.forms_solution.to_dict().items():
                print("  " + util_preview_series(series, var_name))

        if self.combined_solutions:
            print("\n--- Combined Solutions (Coupled Variables) ---")
            for eq in self.combined_solutions:
                sp.pprint(eq, use_unicode=USE_UNICODE)

        if self.free:
            print("\n--- Free Variables (Underconstrained) ---")
            for var in sorted(self.free, key=str):
                sp.pprint(var, use_unicode=USE_UNICODE)

        if self.residuals and self.residual_norm is not None:
            print("\n--- Fit Quality ---")
            print(f"  Residual norm: {float(self.residual_norm):.6e}")
            max_residual = max(abs(r) for r in self.residuals.values()) if self.residuals else 0
            print(f"  Max residual:  {float(max_residual):.6e}")
            if max_residual < sp.Rational(1, 1000000):
                DEBUG.success("Excellent fit")
            elif max_residual < sp.Rational(1, 1000):
                DEBUG.success("Good fit")
            else:
                DEBUG.warning("Residuals may indicate model mismatch")

        print("-" * 70)

    def export_text(self) -> str:
        """
        Export results as plain text for copy/paste.

        Returns formatted text with all solution values, suitable for
        pasting into spreadsheets, reports, or documentation.

        Returns:
            Multi-line string with formatted results
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"{self.system_type.value.upper()} RESULTS")
        lines.append("=" * 70)

        if self.system_type == SystemType.TILTS and self.is_numerical:
            lines.append("\nTilt Parameters:")
            for var_name in ["alpha_x", "alpha_y", "beta_x", "beta_y", "gamma"]:
                var = sp.Symbol(var_name)
                if var in self.independent:
                    val = self.independent[var]
                    lines.append(f"  {var_name:8} = {float(val):12.6e}")

            if self.dependent:
                lines.append(f"\nCoupled Variables:")
                lines.append(f"  {', '.join(str(v) for v in sorted(self.dependent, key=str))}")

            if self.combined_solutions:
                lines.append("\nCombined Values:")
                for eq in self.combined_solutions:
                    lines.append(f"  {eq}")

        elif self.system_type == SystemType.FORMS and self.forms_solution:
            lines.append(f"\nForm Errors ({len(self.config.z_positions)} Z positions, detrended):")
            for var_name, series in self.forms_solution.to_dict().items():
                if isinstance(series, list):
                    # show all of the values when exporting
                    lines.append(util_preview_series(series, var_name, show_all = True))
                    # lines.append(f"  {var_name}: {len(series)} values")
                    # Could optionally include statistics here
                    # max_val = float(max(abs(v) for v in series))
                    # lines.append(f"    Max deviation: {max_val:.6e}")

        if self.residual_norm is not None:
            lines.append(f"\nFit Quality:")
            lines.append(f"  Residual norm: {float(self.residual_norm):.6e}")
            if self.residuals:
                max_residual = max(abs(r) for r in self.residuals.values())
                lines.append(f"  Max residual:  {float(max_residual):.6e}")

        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# Assessment and Comparison Classes
# ============================================================================


@dataclass
class CouplingAssessment:
    """Assessment of rigid body coupling significance."""

    max_z_offset_mm: sp.Rational
    z_range_mm: Optional[sp.Rational]
    coupling_ratio: Optional[sp.Rational]
    severity: CouplingSeverity
    details: Dict[str, Any]
    warnings: List[str]


@dataclass
class ComparisonResult:
    """Results from comparing analysis output against ground truth."""

    system_type: str
    matched_variables: Dict[sp.Symbol, CompareResult] = field(default_factory=dict)
    mismatched_variables: Dict[sp.Symbol, CompareResult] = field(default_factory=dict)
    matched_forms: Dict[sp.Symbol, CompareResult] = field(default_factory=dict)
    mismatched_forms: Dict[sp.Symbol, CompareResult] = field(default_factory=dict)
    coupled_variables: Set[sp.Symbol] = field(default_factory=set)
    free_variables: Set[sp.Symbol] = field(default_factory=set)
    satisfied_coupling: List[sp.Eq] = field(default_factory=list)
    violated_coupling: List[Tuple[sp.Eq, sp.Rational]] = field(default_factory=list)
    measurements_match: bool = False  # Do solutions produce same measurements?
    all_matched: bool = False
    max_error: Optional[sp.Rational] = None
    rms_error: Optional[sp.Rational] = None

    def print(self) -> None:
        """Print formatted comparison results."""
        print(f"\n{'='*70}\nVERIFICATION REPORT: {self.system_type}\n{'='*70}")

        if self.matched_variables:
            print(f"\n{Colors.GREEN}✓ EXACT MATCHES ({len(self.matched_variables)}):{Colors.ENDC} (true ~ solved)")
            for symbol, var in self.matched_variables.items():
                print(f"  {symbol}: {float(var.true):.8f} ~ {float(var.solved):.8f} (err={float(var.error):.2e})")

        if self.matched_forms:
            print(f"\n{Colors.GREEN}✓ MATCHED FORMS ({len(self.matched_forms)}):{Colors.ENDC}")
            for symbol, var in self.matched_forms.items():
                print(f"  {symbol}: max err = {float(var.error):.2e}")

        if self.mismatched_variables:
            print(f"\n{Colors.FAIL}✗ MISMATCHES ({len(self.mismatched_variables)}):{Colors.ENDC}")
            for symbol, var in self.mismatched_variables.items():
                print(f"  {symbol}:\n    Exp: {var.true}\n    Sol: {var.solved}\n    Err: {var.error} ({float(var.error):.6e})")

        if self.mismatched_forms:
            print(f"\n{Colors.FAIL}✗ MISMATCHED FORMS ({len(self.mismatched_forms)}):{Colors.ENDC}")
            for symbol, var in self.mismatched_forms.items():
                print(f"  {symbol}: max err = {float(var.error):.6e}")

        if self.measurements_match:
            print(f"\n{Colors.GREEN}✓ MEASUREMENT EQUIVALENCE:{Colors.ENDC}")
            print("  Solutions produce identical measurements (coupled variables differ by nullspace only)")

        if self.satisfied_coupling:
            print(f"\n{Colors.GREEN}✓ SATISFIED COUPLING CONSTRAINTS ({len(self.satisfied_coupling)}):{Colors.ENDC}")
            for eq in self.satisfied_coupling:
                print(f"  {eq}")

        if self.violated_coupling:
            print(f"\n{Colors.FAIL}✗ VIOLATED COUPLING CONSTRAINTS ({len(self.violated_coupling)}):{Colors.ENDC}")
            for eq, e in self.violated_coupling:
                print(f"  {eq}\n    Error: {float(e):.6e}")

        if self.coupled_variables:
            print(f"\n{Colors.WARNING}⚠ COUPLED VARIABLES ({len(self.coupled_variables)}):{Colors.ENDC}")
            print(f"  Cannot solve independently:\n    {', '.join(map(str, sorted(self.coupled_variables, key=str)))}")

        if self.free_variables:
            print(f"\n{Colors.WARNING}⚠ FREE VARIABLES ({len(self.free_variables)}):{Colors.ENDC}")
            print(f"  Underconstrained:\n    {', '.join(map(str, sorted(self.free_variables, key=str)))}")

        print(f"\n{'='*70}")
        if self.all_matched or self.measurements_match:
            print(f"{Colors.GREEN}VERIFICATION: PASSED ✓{Colors.ENDC}")
            if self.all_matched:
                print("All independently solvable variables matched exactly!")
            if self.measurements_match:
                print("Solutions are equivalent (produce same measurements)!")
        else:
            print(f"{Colors.FAIL}VERIFICATION: FAILED ✗{Colors.ENDC}")
            if self.max_error:
                print(f"Maximum error: {float(self.max_error):.6e}")
            if self.rms_error:
                print(f"RMS error:     {float(self.rms_error):.6e}")
        print(f"{'='*70}")
