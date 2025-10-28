# ============================================================================
# File 1: utils.py
# ============================================================================
"""
Utility functions, core math operations, and debug infrastructure.
"""

import inspect
import textwrap
from enum import StrEnum
from typing import Any, Dict, List, Optional, Tuple
import sympy as sp
import math

def convert_str_to_sympy(rational_string, type = sp.Rational):
    """
    Converts a comma-separated string of rational numbers into a list of sympy.Rational objects.

    Args:
        rational_string (str): A string containing comma-separated rational numbers (e.g., "1/2,3/4,5").

    Returns:
        list: A list of sympy.Rational objects.
    """
    return [type(part.strip()) for part in rational_string.split(",")]


class Colors(StrEnum):
    """ANSI terminal color codes for formatted console output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


class DebugPrinter:
    """Centralized debug output manager with hierarchical section tracking."""

    def __init__(self, info: bool = True, success: bool = True, warning: bool = True, error: bool = True):
        self.levels: Dict[str, bool] = {"info": info, "success": success, "warning": warning, "error": error}
        self._indent_level: int = 0
        self._section_stack: List[str] = []

    def _get_caller_info(self, stack_count: int) -> str:
        try:
            frame = inspect.stack()[stack_count]
            frame_info = inspect.getframeinfo(frame[0])
            if "self" in frame[0].f_locals:
                class_name = frame[0].f_locals["self"].__class__.__name__
                return f"[{class_name}.{frame_info.function}:{frame_info.lineno}]"
            return f"[{frame_info.function}:{frame_info.lineno}]"
        except IndexError:
            return "[unknown location]"

    def enter_section(self, name: str) -> None:
        self.info(f"Entering {name}...", stack_count=4)
        self._section_stack.append(name)
        self._indent_level += 1

    def exit_section(self) -> None:
        if self._section_stack:
            self._indent_level -= 1
            section_name = self._section_stack.pop()
            self.info(f"Exiting {section_name}...", stack_count=4)

    def print(self, message: str, color: Colors, icon: str = "", stack_count: int = 2, nocaller: bool = False) -> None:
        if self.levels.get(color.name.lower(), True):  # Basic level check
            caller_info = "" if nocaller else " " + self._get_caller_info(stack_count)
            indent_message = textwrap.indent(f"{color}{icon}{caller_info} {message}{Colors.ENDC}", "  " * self._indent_level)
            print(indent_message)

    def info(self, message: str, stack_count: int = 3, title: str = "", nocaller: bool = False) -> None:
        if self.levels["info"]:
            if title:
                self.print(title, Colors.CYAN, "ℹ", stack_count, nocaller)
                self.print(message, Colors.CYAN, "ℹ", stack_count, True)
            else:
                self.print(message, Colors.CYAN, "ℹ", stack_count, nocaller)

    def success(self, message: str, stack_count: int = 3, nocaller: bool = False) -> None:
        if self.levels["success"]:
            self.print(message, Colors.GREEN, "✓", stack_count, nocaller)

    def warning(self, message: str, stack_count: int = 3) -> None:
        if self.levels["warning"]:
            self.print(message, Colors.WARNING, "⚠ ", stack_count)

    def error(self, message: str, stack_count: int = 3) -> None:
        if self.levels["error"]:
            self.print(message, Colors.FAIL, "✗", stack_count)


# ============================================================================
# Core Math Utilities
# ============================================================================

# Speed optimization: pre-create commonly used constants
RATIONAL_360 = sp.Rational(360)
INTEGER_2 = sp.Integer(2)


def get_surface_angle(ind_deg: sp.Rational, spin_deg: sp.Rational) -> sp.Rational:
    """
    Calculate clockwise surface angle from indicator and spindle positions (CCW).

    COORDINATE SYSTEM CONVERSION:
    - Indicator positions are measured CCW from +Y (top)
    - Surface labels are measured CW from top (when spindle at 0°)
    - This function converts between the two reference frames

    Formula: θ_surf = (360° - (θ_ind - θ_spin)) mod 360°

    Args:
        ind_deg: Indicator angular position (CCW from +Y axis)
        spin_deg: Spindle rotation angle (CCW)

    Returns:
        Surface angle in CW reference frame
    """
    return (RATIONAL_360 - (ind_deg - spin_deg)) % RATIONAL_360  # pyright: ignore[reportOperatorIssue]


def _least_squares_core(x_values: List[sp.Rational], y_values: List[sp.Rational]) -> Tuple[sp.Rational, sp.Rational]:
    """
    Core least squares fit: y = intercept + slope * x using exact SymPy arithmetic.

    This is the foundation for:
    1. Separating measurements into tilt (slope) and form (intercept)
    2. Detrending form profiles to isolate shape from linear trends

    Uses normal equations: minimizes Σ(y_i - (intercept + slope·x_i))²
    All arithmetic is exact (SymPy rationals), no floating point errors.

    Args:
        x_values: Independent variable values (e.g., Z positions)
        y_values: Dependent variable values (e.g., measurements or form values)

    Returns:
        (slope, intercept) tuple
    """
    n = sp.Rational(len(x_values))
    if n < INTEGER_2:
        raise ValueError("Least squares requires at least 2 points.")

    sum_x = sum(x_values)  # pyright: ignore[reportArgumentType, reportCallIssue]
    sum_y = sum(y_values)  # pyright: ignore[reportArgumentType, reportCallIssue]
    sum_xy: sp.Rational = sum(x * y for x, y in zip(x_values, y_values))  # pyright: ignore[reportAssignmentType, reportOperatorIssue]
    sum_x2 = sum(x**INTEGER_2 for x in x_values)  # pyright: ignore[reportArgumentType, reportCallIssue]

    denominator = n * sum_x2 - sum_x**INTEGER_2
    if denominator == 0:
        # All x values identical - slope undefined (or zero), intercept is mean y
        if len(set(x_values)) == 1:
            DEBUG.warning("All x values identical in least squares; slope is zero.")
            return sp.S.Zero, sum_y / n
        else:
            raise ValueError("Least squares denominator is zero unexpectedly.")

    slope = (n * sum_xy - sum_x * sum_y) / denominator  # pyright: ignore[reportOperatorIssue]
    intercept = (sum_y - slope * sum_x) / n

    return sp.simplify(slope), sp.simplify(intercept)


def remove_form_trend(form_series: List[sp.Expr] | List[sp.Rational], z_positions: List[sp.Rational]) -> List[sp.Rational]:
    """
    Removes linear trend (slope + offset) from form series using least squares.

    DETRENDING STRATEGY:
    This is critical for maintaining model validity. The linearized model assumes:
    - Tilts (α, β, γ) contribute ONLY to z-dependent terms (slope)
    - Forms (M, S) should have NO linear trend with z

    By detrending, we ensure:
    1. In synthetic data: forms don't accidentally contain "hidden tilts"
    2. In solved results: forms represent pure shape deviations
    3. Model equation separation is maintained: I = Form + Tilt·z

    Args:
        form_series: List of form values at each z position
        z_positions: Corresponding z positions

    Returns:
        Detrended form series (mean removed, slope removed)
    """
    if not form_series or len(form_series) != len(z_positions):
        raise ValueError("Series and Z positions must be non-empty and equal length")

    slope, intercept = _least_squares_core(z_positions, form_series)

    return [
        sp.simplify(form_series[i] - (intercept + slope * z_positions[i])) for i in range(len(form_series))
    ]  # pyright: ignore[reportOperatorIssue]


# ============================================================================
# Formatting Utilities
# ============================================================================


def preview_series(series, series_name: str, max_preview: int = 5, show_all = False) -> str:
    """
    Format series for display with optional truncation.

    Handles both list series (forms) and scalar values (tilts).

    Args:
        series: Either a list of values or a single value
        series_name: Name/label for the series
        max_preview: Maximum number of values to show before truncating

    Returns:
        Formatted string for display

    Examples:
        >>> preview_series([1.0, 2.0, 3.0], "M_x", max_preview=5)
        'M_x = [1.000000e+00, 2.000000e+00, 3.000000e+00] (3 values)'

        >>> preview_series(0.001745, "alpha_y", max_preview=5)
        'alpha_y = 0.001745'
    """
    if not isinstance(series, list):
        # Scalar value - just return formatted string
        return f"{series_name} = {series}"

    # List series - format with truncation if needed
    if len(series) <= max_preview or show_all:
        preview = [f"{float(v):.6e}" for v in series]
    else:
        _first = int(math.ceil(max_preview/2))
        _last = max_preview - _first

        # Show first 3, ellipsis, last 2
        preview = [f"{float(v):.6e}" for v in series[:_first]] + ["..."] + [f"{float(v):.6e}" for v in series[-_last:]]

    return f"{series_name} = [{', '.join(preview)}] ({len(series)} values)"


def format_comparison_stats(stats: Dict[str, Any], label: str = "", indent: int = 2) -> str:
    """
    Format comparison statistics for display.

    Helper for RawMeasurementData.print_comparison() and similar functions.
    Handles both scalar statistics and per-configuration breakdowns.

    Args:
        stats: Dictionary of statistics (max, rms, etc.)
        label: Optional label for the section
        indent: Number of spaces to indent

    Returns:
        Formatted string
    """
    lines = []
    prefix = " " * indent

    if label:
        lines.append(f"{prefix}{label}:")

    # Format main statistics
    if "max" in stats:
        max_val = float(stats["max"])
        lines.append(f"{prefix}  Max: {max_val:.8f}  ({max_val:.4e})")

    if "rms" in stats:
        rms_val = float(stats["rms"])
        lines.append(f"{prefix}  RMS: {rms_val:.8f}  ({rms_val:.4e})")

    # Format any additional fields
    for key, value in stats.items():
        if key not in ["max", "rms", "all_diffs"]:
            if isinstance(value, (int, float, sp.Rational)):
                lines.append(f"{prefix}  {key}: {float(value):.8f}")

    return "\n".join(lines)


def format_table(data: Dict[str, Any], title: Optional[str] = None) -> str:
    """Format dictionary as aligned table string."""
    if title:
        output = [f"\n{title}", "=" * len(title)]
    else:
        output = []

    if not data:
        output.append("(empty)")
        return "\n".join(output)

    max_key_len = max(len(str(k)) for k in data.keys()) if data else 0
    for key, value in data.items():
        output.append(f"  {str(key):<{max_key_len}} : {value}")

    return "\n".join(output)


def format_tilt_angle(slope: sp.Rational | sp.Float) -> str:
    """
    Format tilt slope as both slope value and angle in degrees.

    Converts small slope values (typical in metrology) to more intuitive angles.
    Example: slope = 0.001745 → "0.001745 (0.1000°)"

    Args:
        slope: Tilt slope value (unitless, small number)

    Returns:
        Formatted string with both slope and angle
    """
    try:
        angle_deg = sp.deg(sp.atan(slope))
        return f"{float(slope):.6f} ({float(angle_deg):.4f}°)"
    except Exception:  # Handle potential eval errors for complex expressions
        return f"{sp.N(slope):.6f} (angle calc error)"


def slope_from_deg(deg: sp.Rational | str) -> sp.Rational:
    """
    Calculate slope from angle in degrees.

    Inverse of format_tilt_angle's angle calculation.
    Used to specify expected tilts in intuitive degree units.

    Args:
        deg: Angle in degrees

    Returns:
        Slope value (unitless)
    """
    return sp.simplify(sp.tan(sp.rad(sp.Rational(deg))))


# ============================================================================
# Configuration Helpers
# ============================================================================


def create_standard_config(
    indicator_angles: str,
    spindle_angles: str,
    z_step: float,
    z_count: Optional[int] = None,
    z_start: float = 0,
    z_stop: Optional[float] = None,
    diameter: float = 50,
):
    """
    Create standard spindle configuration.

    Can specify Z positions either by count OR by range:
    - Count-based: z_count=11, z_step=10 → [0, 10, 20, ..., 100]
    - Range-based: z_start=0, z_stop=100, z_step=10 → [0, 10, 20, ..., 100]

    Args:
        indicator_angles: Indicator positions in degrees
        spindle_angles: Spindle positions in degrees
        z_count: Number of Z positions (if specified, uses count-based)
        z_start: Starting Z position (default 0)
        z_stop: Ending Z position (required if not using z_count)
        z_step: Z increment
        diameter: Nominal artifact diameter

    Returns:
        SpindleConfiguration ready for use

    Examples:
        >>> # Count-based (for tests)
        >>> config = create_standard_config([0, 180], [0, 180], z_count=11)

        >>> # Range-based (for custom measurements)
        >>> config = create_standard_config([0, 180], [0, 180],
        ...                                  z_start=5, z_stop=105, z_step=5)
    """
    from models import SpindleConfiguration  # Import here to avoid circular dependency

    if z_count is not None:
        # Count-based: generate z_count positions starting at z_start
        z_positions = [sp.Rational(z_start + i * z_step) for i in range(z_count)]
    elif z_stop is not None:
        # Range-based: generate from z_start to z_stop (inclusive)
        z_positions = [sp.Rational(z) for z in range(int(z_start), int(z_stop) + 1, int(z_step))]
    else:
        raise ValueError("Must specify either z_count or z_stop")

    return SpindleConfiguration(
        indicator_angles=indicator_angles, spindle_angles=spindle_angles, nominal_diameter=diameter, z_positions=z_positions
    )


# ============================================================================
# Global Configuration
# ============================================================================

DEBUG = DebugPrinter(info=True, success=True, warning=True, error=True)
USE_UNICODE: bool = False
