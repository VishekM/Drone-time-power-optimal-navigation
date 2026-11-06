"""
Crazyflie 2.0 quadrotor model: parameters, mixer, aerodynamic drag, dynamics.

Sources
-------
- J. Förster, "System Identification of the Crazyflie 2.0 Nano Quadrocopter",
  ETH Zürich, 2015 (Appendix A: mass, inertia, motor maps, drag).
- Bitcraze, Crazyflie 2.0 Datasheet Rev 2 (geometry: 92 mm motor-to-motor).

Conventions
-----------
- World frame: z up.  Body frame: x forward, y left, z up (Crazyflie convention).
- Quaternion q = [w, x, y, z], rotates body -> world.
- State   x = [p (3), v (3), q (4), omega (3)]   -> 13 states
            p, v in world frame; omega = body angular rates.
- Input   u = [f1, f2, f3, f4]  individual motor thrusts [N], 0 <= f_i <= F_MAX.
  (Use wrench_to_motor_forces() if a controller outputs [T, tau_x, tau_y, tau_z].)
- Motor layout (X configuration, viewed from above):

            front (+x)
        M4 (+d,+d)   M1 (+d,-d)
                  \\ /
                  / \\
        M3 (-d,+d)   M2 (-d,-d)

- Motor lag is NOT modelled (thrust responds instantly).
"""

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CrazyflieParams:
    # --- Förster 2015, Appendix A -------------------------------------------
    m: float = 0.028                      # [kg] measured mass (datasheet: 27 g)
    g: float = 9.81                       # [m/s^2]

    # Full inertia matrix [kg m^2]
    J: np.ndarray = field(default_factory=lambda: np.array([
        [16.571710,  0.830806,  0.718277],
        [ 0.830806, 16.655602,  1.800197],
        [ 0.718277,  1.800197, 29.261652],
    ]) * 1e-6)

    # Input command (PWM, 0..65535) -> thrust per motor [N]
    #   f = a2*cmd^2 + a1*cmd + a0
    thrust_a2: float = 2.130295e-11
    thrust_a1: float = 1.032633e-6
    thrust_a0: float = 5.484560e-4
    cmd_max: float = 65535.0

    # Input command -> rotor angular velocity [rad/s]
    #   theta_dot = b1*cmd + b0
    rotor_b1: float = 0.04076521
    rotor_b0: float = 380.8359

    # Thrust -> reaction torque per motor [N m]
    #   tau_i = c_yaw * f_i (+ 1.563383e-5 offset, cancels across the 4 motors)
    c_yaw: float = 0.005964552

    # Aerodynamic drag (Förster simple model, eq. [13]):
    #   f_a (body) = K_aero * theta_dot_sum * v_body
    K_aero: np.ndarray = field(default_factory=lambda: np.diag(
        [-9.1785e-7, -9.1785e-7, -10.311e-7]))           # [kg / rad]

    # --- Bitcraze datasheet -------------------------------------------------
    # 92 mm motor-to-motor (diagonal, incl. motor mount feet) -> 46 mm
    # centre-to-motor. Matches the 0.046 m arm used in the Crazyflie literature.
    arm: float = 0.046                    # [m] centre to motor axis

    # Motor spin direction -> sign of reaction torque about body z.
    # Bitcraze: M1 & M3 spin CCW (react CW, -z), M2 & M4 spin CW (react +z).
    # Flip all signs if your yaw response comes out reversed.
    spin: np.ndarray = field(default_factory=lambda: np.array([-1., 1., -1., 1.]))

    # ---------------- Derived quantities ------------------------------------
    @property
    def d(self) -> float:
        """Roll/pitch moment arm for X configuration [m]."""
        return self.arm / np.sqrt(2.0)

    @property
    def J_inv(self) -> np.ndarray:
        return np.linalg.inv(self.J)

    @property
    def f_max(self) -> float:
        """Max thrust per motor [N] (cmd = 65535)."""
        return float(cmd_to_thrust(self.cmd_max, self))

    @property
    def f_min(self) -> float:
        return 0.0

    @property
    def f_hover(self) -> float:
        """Thrust per motor at hover [N]."""
        return self.m * self.g / 4.0

    @property
    def thrust_to_weight(self) -> float:
        return 4.0 * self.f_max / (self.m * self.g)

    @property
    def mixer(self) -> np.ndarray:
        """Allocation matrix B: [T, tau_x, tau_y, tau_z] = B @ [f1..f4]."""
        d, c = self.d, self.c_yaw
        x = np.array([ d, -d, -d,  d])     # motor x positions
        y = np.array([-d, -d,  d,  d])     # motor y positions
        return np.vstack([
            np.ones(4),        # total thrust
            y,                 # tau_x =  sum y_i f_i
            -x,                # tau_y = -sum x_i f_i
            c * self.spin,     # tau_z =  sum s_i c f_i
        ])

    @property
    def mixer_inv(self) -> np.ndarray:
        return np.linalg.inv(self.mixer)


# ---------------------------------------------------------------------------
# Motor maps
# ---------------------------------------------------------------------------
def cmd_to_thrust(cmd, P: CrazyflieParams):
    cmd = np.asarray(cmd, dtype=float)
    return P.thrust_a2 * cmd**2 + P.thrust_a1 * cmd + P.thrust_a0


def thrust_to_cmd(f, P: CrazyflieParams):
    """Invert the quadratic thrust map. Returns cmd in [0, cmd_max]."""
    f = np.asarray(f, dtype=float)
    a, b, c = P.thrust_a2, P.thrust_a1, P.thrust_a0 - f
    cmd = (-b + np.sqrt(np.maximum(b**2 - 4 * a * c, 0.0))) / (2 * a)
    return np.clip(cmd, 0.0, P.cmd_max)


def thrust_to_rotor_speed(f, P: CrazyflieParams):
    """Rotor angular velocity [rad/s] for thrust f [N]. Zero if motor is off."""
    f = np.asarray(f, dtype=float)
    omega = P.rotor_b1 * thrust_to_cmd(f, P) + P.rotor_b0
    return np.where(f > 0.0, omega, 0.0)


# ---------------------------------------------------------------------------
# Mixer helpers
# ---------------------------------------------------------------------------
def motor_forces_to_wrench(f, P: CrazyflieParams):
    """[f1..f4] -> [T, tau_x, tau_y, tau_z]."""
    return P.mixer @ np.asarray(f, dtype=float)


def wrench_to_motor_forces(w, P: CrazyflieParams, clip: bool = True):
    """[T, tau_x, tau_y, tau_z] -> [f1..f4], optionally clipped to limits."""
    f = P.mixer_inv @ np.asarray(w, dtype=float)
    return np.clip(f, P.f_min, P.f_max) if clip else f


# ---------------------------------------------------------------------------
# Quaternion utilities (q = [w, x, y, z])
# ---------------------------------------------------------------------------
def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def quat_derivative(q, omega):
    """q_dot = 0.5 * q ⊗ [0, omega]   (omega in body frame)."""
    w, x, y, z = q
    p, qq, r = omega
    return 0.5 * np.array([
        -x*p - y*qq - z*r,
         w*p + y*r  - z*qq,
         w*qq + z*p - x*r,
         w*r + x*qq - y*p,
    ])


# ---------------------------------------------------------------------------
# Aerodynamic drag
# ---------------------------------------------------------------------------
def drag_force_body(v_world, R, f, P: CrazyflieParams):
    """Förster drag model: f_a = K_aero * sum(theta_dot) * R^T v  (body frame)."""
    theta_dot_sum = np.sum(thrust_to_rotor_speed(f, P))
    return P.K_aero @ (theta_dot_sum * (R.T @ v_world))


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------
def dynamics(x, u, P: CrazyflieParams, use_drag: bool = True):
    """Continuous-time dynamics x_dot = f(x, u).

    x: (13,) state [p, v, q, omega]
    u: (4,)  motor thrusts [N]
    """
    v = x[3:6]
    q = x[6:10]
    omega = x[10:13]

    u = np.clip(u, P.f_min, P.f_max)
    T, tau_x, tau_y, tau_z = motor_forces_to_wrench(u, P)
    tau = np.array([tau_x, tau_y, tau_z])

    R = quat_to_rotmat(q)

    # Translational: m v_dot = -m g e3 + R (T e3 + f_drag)
    f_body = np.array([0.0, 0.0, T])
    if use_drag:
        f_body = f_body + drag_force_body(v, R, u, P)
    v_dot = np.array([0.0, 0.0, -P.g]) + (R @ f_body) / P.m

    # Rotational: J omega_dot = tau - omega x J omega
    omega_dot = P.J_inv @ (tau - np.cross(omega, P.J @ omega))

    return np.concatenate([v, v_dot, quat_derivative(q, omega), omega_dot])


def rk4_step(x, u, dt, P: CrazyflieParams, use_drag: bool = True):
    """One RK4 step with zero-order-hold input; renormalises the quaternion."""
    k1 = dynamics(x, u, P, use_drag)
    k2 = dynamics(x + 0.5 * dt * k1, u, P, use_drag)
    k3 = dynamics(x + 0.5 * dt * k2, u, P, use_drag)
    k4 = dynamics(x + dt * k3, u, P, use_drag)
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    x_next[6:10] /= np.linalg.norm(x_next[6:10])
    return x_next


def hover_state(position=(0.0, 0.0, 1.0)):
    x = np.zeros(13)
    x[0:3] = position
    x[6] = 1.0  # identity quaternion
    return x


def hover_input(P: CrazyflieParams):
    return np.full(4, P.f_hover)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    P = CrazyflieParams()

    print("=== Crazyflie 2.0 parameters ===")
    print(f"mass                 {P.m*1e3:.1f} g")
    print(f"arm (centre-motor)   {P.arm*1e3:.1f} mm   d = {P.d*1e3:.2f} mm")
    print(f"yaw coefficient      {P.c_yaw:.6f} m")
    print(f"hover thrust/motor   {P.f_hover:.4f} N")
    print(f"max thrust/motor     {P.f_max:.4f} N")
    print(f"thrust-to-weight     {P.thrust_to_weight:.2f}")
    print(f"hover cmd            {float(thrust_to_cmd(P.f_hover, P)):.0f} "
          f"({100*float(thrust_to_cmd(P.f_hover, P))/P.cmd_max:.1f}%)")
    print(f"hover rotor speed    {float(thrust_to_rotor_speed(P.f_hover, P)):.0f} rad/s")
    print(f"max rotor speed      {float(thrust_to_rotor_speed(P.f_max, P)):.0f} rad/s")
    print("J [1e-6 kg m^2]:\n", np.round(P.J * 1e6, 4))
    print("mixer:\n", np.round(P.mixer, 5))

    # 1) Mixer round-trip
    w = np.array([P.m * P.g, 1e-4, -2e-4, 5e-5])
    assert np.allclose(motor_forces_to_wrench(wrench_to_motor_forces(w, P, clip=False), P), w)

    # 2) Hover equilibrium: x_dot should be zero
    x0, u0 = hover_state(), hover_input(P)
    xdot = dynamics(x0, u0, P)
    print(f"\n|x_dot| at hover     {np.linalg.norm(xdot):.2e}")
    assert np.linalg.norm(xdot) < 1e-10

    # 3) Hover for 2 s at 500 Hz -> should not drift
    x, dt = x0.copy(), 0.002
    for _ in range(1000):
        x = rk4_step(x, u0, dt, P)
    print(f"drift after 2 s      {np.linalg.norm(x[:3] - x0[:3]):.2e} m")

    # 4) Drag check: free coast at 1 m/s forward with hover thrust
    x = hover_state()
    x[3] = 1.0
    a = dynamics(x, u0, P)[3:6]
    print(f"drag decel @1 m/s    {a[0]:.4f} m/s^2 (x), body-frame drag "
          f"{np.round(drag_force_body(x[3:6], np.eye(3), u0, P), 5)} N")

    # 5) Sign checks: roll/pitch/yaw torque from differential thrust
    df = 0.005
    print("\nsign checks (expected: +tau_x, +tau_y, +tau_z):")
    print("  more thrust on left  (M3,M4):", np.round(motor_forces_to_wrench(u0 + df*np.array([0,0,1,1]), P)[1], 6))
    print("  more thrust on rear  (M2,M3):", np.round(motor_forces_to_wrench(u0 + df*np.array([0,1,1,0]), P)[2], 6))
    print("  more thrust on CW    (M2,M4):", np.round(motor_forces_to_wrench(u0 + df*np.array([0,1,0,1]), P)[3], 6))
    print("\nAll checks passed.")