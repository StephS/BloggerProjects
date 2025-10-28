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


def convert_str_to_sp(rational_string):
    """
    Converts a comma-separated string of rational numbers into a list of sympy.Rational objects.

    Args:
        rational_string (str): A string containing comma-separated rational numbers (e.g., "1/2,3/4,5").

    Returns:
        list: A list of sympy.Rational objects.
    """
    return [sp.Rational(part.strip()) for part in rational_string.split(',')]


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

# speed things up a little
RATIONAL_360 = sp.Rational(360)
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

INTEGER_2 = sp.Integer(2)
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

    sum_x = sum(x_values) # pyright: ignore[reportArgumentType, reportCallIssue]
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


def remove_form_trend(form_series: List[sp.Rational], z_positions: List[sp.Rational]) -> List[sp.Rational]:
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


def format_tilt_angle(slope: sp.Rational) -> str:
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
# Global Configuration
# ============================================================================

DEBUG = DebugPrinter(info=True, success=True, warning=True, error=True)
USE_UNICODE: bool = False
