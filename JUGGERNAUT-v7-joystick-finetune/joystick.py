"""Joystick axis mapping shared by the preview and hardware integration."""
import math

def command_from_axes(horizontal, vertical, deadzone=0.08):
    """Pygame convention: left/up negative; return body vx (m/s), yaw (rad/s)."""
    def axis(value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Joystick axes must be finite")
        value = max(-1.0, min(1.0, value))
        if abs(value) <= deadzone:
            return 0.0
        return math.copysign((abs(value)-deadzone)/(1.0-deadzone), value)
    if not 0 <= deadzone < 1:
        raise ValueError("deadzone must be in [0, 1)")
    forward = -axis(vertical)
    return (forward * (0.15 if forward >= 0 else 0.10), -0.4 * axis(horizontal))
