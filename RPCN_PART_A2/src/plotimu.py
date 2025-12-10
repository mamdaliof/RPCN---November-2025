"""IMU trajectory plotting helper.

The script reads IMU samples exported from rosbag (see Assignment.md),
optionally applies calibration parameters obtained in task A1, integrates the
sequence in 2D, and plots the resulting trajectory.

Usage example:
    python src/plotimu.py --data data/imu_data_1.csv \
        --accel-bias 0.01,-0.02,0.05 --gyro-bias 0.001,0.0,-0.002 \
        --output plot.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_vector(arg: str | None, fallback: Sequence[float]) -> np.ndarray:
    """Parses comma separated floats like '0.1,0.2,0.3'."""
    if arg is None:
        return np.asarray(fallback, dtype=float)
    parts = [value.strip() for value in arg.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected 3 comma separated values, received {arg!r}"
        )
    try:
        floats = [float(value) for value in parts]
    except ValueError as exc:  # pragma: no cover - argparse already reports it
        raise argparse.ArgumentTypeError(f"Could not parse {arg!r}") from exc
    return np.asarray(floats, dtype=float)


@dataclass
class ImuData:
    time_s: np.ndarray  # shape (N,)
    angular_velocity: np.ndarray  # shape (N, 3)
    linear_acceleration: np.ndarray  # shape (N, 3)


@dataclass
class Calibration:
    accel_bias: np.ndarray
    accel_scale: np.ndarray
    gyro_bias: np.ndarray
    gyro_scale: np.ndarray

    @classmethod
    def identity(cls) -> "Calibration":
        zeros = np.zeros(3, dtype=float)
        ones = np.ones(3, dtype=float)
        return cls(accel_bias=zeros, accel_scale=ones,
                   gyro_bias=zeros, gyro_scale=ones)

    def apply(self, accel: np.ndarray, gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns calibrated accelerometer and gyroscope readings."""
        corrected_accel = (accel - self.accel_bias) * self.accel_scale
        corrected_gyro = (gyro - self.gyro_bias) * self.gyro_scale
        return corrected_accel, corrected_gyro


def load_imu_csv(csv_path: Path) -> ImuData:
    """Loads rosbag exported CSV with IMU measurements."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} does not contain any measurements")

    time_ns = df["%time"].to_numpy(dtype=np.float64)
    time_s = (time_ns - time_ns[0]) * 1e-9

    angular_velocity = df[
        [
            "field.angular_velocity.x",
            "field.angular_velocity.y",
            "field.angular_velocity.z",
        ]
    ].to_numpy(dtype=np.float64)

    linear_acceleration = df[
        [
            "field.linear_acceleration.x",
            "field.linear_acceleration.y",
            "field.linear_acceleration.z",
        ]
    ].to_numpy(dtype=np.float64)

    return ImuData(time_s=time_s,
                   angular_velocity=angular_velocity,
                   linear_acceleration=linear_acceleration)


def integrate_planar_motion(imu: ImuData, initial_heading: float = 0.0) -> np.ndarray:
    """Performs a simple 2D mechanization and returns positions with shape (N, 2)."""
    n = imu.time_s.shape[0]
    print(f"The amount of data points: ${n}")
    positions = np.zeros((n, 2), dtype=float)
    velocities = np.zeros((n, 2), dtype=float)
    yaw = np.zeros(n, dtype=float)
    yaw[0] = initial_heading

    for i in range(1, n):
        dt = imu.time_s[i] - imu.time_s[i - 1]
        if dt <= 0.0:
            positions[i] = positions[i - 1]
            velocities[i] = velocities[i - 1]
            yaw[i] = yaw[i - 1]
            continue

        # Integrate yaw with the Z gyroscope component.
        yaw[i] = yaw[i - 1] + imu.angular_velocity[i - 1, 2] * dt
        yaw_mid = 0.5 * (yaw[i] + yaw[i - 1])

        accel_body_xy = imu.linear_acceleration[i - 1, :2]
        c = np.cos(yaw_mid)
        s = np.sin(yaw_mid)
        accel_nav = np.array(
            [c * accel_body_xy[0] - s * accel_body_xy[1],
             s * accel_body_xy[0] + c * accel_body_xy[1]]
        )

        velocities[i] = velocities[i - 1] + accel_nav * dt
        velocities[i] =  accel_nav * dt
        positions[i] = positions[i - 1] + velocities[i] * dt

    return positions


def plot_trajectories(trajectories: Iterable[Tuple[str, np.ndarray]],
                      title: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, traj in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], label=label)
    ax.scatter([0.0], [0.0], c="green", marker="o", label="Start")
    for label, traj in trajectories:
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="x")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    if title:
        ax.set_title(title)
    return fig


def format_vector(vec: np.ndarray) -> str:
    return ", ".join(f"{value:+.3f}" for value in vec)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrate and plot IMU trajectories exported from rosbag."
    )
    parser.add_argument("--data", type=Path, required=True,
                        help="CSV file with IMU readings (see data/*.csv).")
    parser.add_argument("--accel-bias", type=str, default=None,
                        help="Comma separated accelerometer bias (m/s^2).")
    parser.add_argument("--gyro-bias", type=str, default=None,
                        help="Comma separated gyroscope bias (rad/s).")
    parser.add_argument("--accel-scale", type=str, default=None,
                        help="Comma separated accelerometer scale factors.")
    parser.add_argument("--gyro-scale", type=str, default=None,
                        help="Comma separated gyroscope scale factors.")
    parser.add_argument("--initial-heading", type=float, default=0.0,
                        help="Initial yaw angle in radians.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional path to save the plot (PNG, PDF, ...).")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip showing the window (useful in headless mode).")
    parser.add_argument("--title", type=str, default=None,
                        help="Optional title for the plot.")

    args = parser.parse_args()

    accel_bias = parse_vector(args.accel_bias, fallback=[0.0, 0.0, 0.0])
    gyro_bias = parse_vector(args.gyro_bias, fallback=[0.0, 0.0, 0.0])
    accel_scale = parse_vector(args.accel_scale, fallback=[1.0, 1.0, 1.0])
    gyro_scale = parse_vector(args.gyro_scale, fallback=[1.0, 1.0, 1.0])

    calibration = Calibration(
        accel_bias=accel_bias,
        accel_scale=accel_scale,
        gyro_bias=gyro_bias,
        gyro_scale=gyro_scale,
    )

    imu = load_imu_csv(args.data)

    calibrated_accel, calibrated_gyro = calibration.apply(
        imu.linear_acceleration, imu.angular_velocity
    )

    calibrated_imu = ImuData(
        time_s=imu.time_s,
        angular_velocity=calibrated_gyro,
        linear_acceleration=calibrated_accel,
    )

    # Raw (uncalibrated) trajectory.
    raw_positions = integrate_planar_motion(imu, initial_heading=args.initial_heading)
    calibrated_positions = integrate_planar_motion(
        calibrated_imu, initial_heading=args.initial_heading
    )

    trajectories: List[Tuple[str, np.ndarray]] = [
        ("Raw data", raw_positions),
        ("Calibrated", calibrated_positions),
    ]

    drift_raw = np.linalg.norm(raw_positions[-1] - raw_positions[0])
    drift_calibrated = np.linalg.norm(calibrated_positions[-1] - calibrated_positions[0])

    print(f"Loaded {imu.time_s.size} samples from {args.data}")
    print(f"Accelerometer bias: [{format_vector(accel_bias)}] m/s^2")
    print(f"Gyroscope bias:     [{format_vector(gyro_bias)}] rad/s")
    print(f"Raw drift:          {drift_raw:.3f} m")
    print(f"Calibrated drift:   {drift_calibrated:.3f} m")

    fig = plot_trajectories(trajectories, title=args.title)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, bbox_inches="tight", dpi=200)
        print(f"Plot saved to {args.output}")

    if not args.no_display:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
