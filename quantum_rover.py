"""
quantum_rover.py

Simulated path-detector rover using a hybrid classical-quantum decision module (variational quantum circuit).
- Simulates a simple 2D environment with a path (center line).
- Generates synthetic sensor data (left, center, right distance-to-path).
- Trains a small variational quantum classifier to map sensor readings -> action {LEFT, FORWARD, RIGHT}.
- Runs a control loop where the quantum model selects actions for the rover.

Requirements:
  pip install qiskit numpy scipy matplotlib

Run:
  python quantum_rover.py
"""

import math
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.utils import algorithm_globals
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import random

# ---------------------------
# Environment & sensors
# ---------------------------
ACTION_LEFT = 0
ACTION_FORWARD = 1
ACTION_RIGHT = 2
ACTIONS = ['LEFT', 'FORWARD', 'RIGHT']

class SimpleRoverEnv:
    """Very small deterministic 2D environment where a 'path' is along y=0.
       Rover has (x,y,theta). Sensors return signed lateral distance from path at three probe angles."""
    def __init__(self, x=0.0, y=0.5, theta=0.0, sensor_range=1.5):
        self.x = x
        self.y = y
        self.theta = theta  # radians; 0 means facing +x
        self.sensor_range = sensor_range
        self.dt = 0.1   # step length for forward move
        self.turn_angle = math.radians(15)  # per turn action

    def get_sensors(self):
        """Simulate three range-like sensors (left, center, right).
           We project a probe line and return lateral distance to path (y=0), normalized [0,1].
           Smaller value => closer to path centerline."""
        # probe angles relative to heading: left +30°, center 0°, right -30°
        offsets = [math.radians(30), 0.0, -math.radians(30)]
        readings = []
        for ao in offsets:
            probe_theta = self.theta + ao
            # sample a point at sensor range (simple single-sample)
            probe_x = self.x + self.sensor_range * math.cos(probe_theta)
            probe_y = self.y + self.sensor_range * math.sin(probe_theta)
            # distance to path y=0 is abs(probe_y)
            d = abs(probe_y)
            # clamp and normalize: we map [0, sensor_range] -> [0,1]
            d_norm = min(d / self.sensor_range, 1.0)
            readings.append(d_norm)
        return np.array(readings, dtype=float)

    def step(self, action):
        """Apply action: LEFT/RIGHT rotates, FORWARD moves ahead."""
        if action == ACTION_LEFT:
            self.theta += self.turn_angle
        elif action == ACTION_RIGHT:
            self.theta -= self.turn_angle
        elif action == ACTION_FORWARD:
            self.x += self.dt * math.cos(self.theta)
            self.y += self.dt * math.sin(self.theta)
        else:
            raise ValueError("Unknown action")
        # normalize theta
        self.theta = (self.theta + math.pi) % (2*math.pi) - math.pi

# ---------------------------
# Quantum classifier (VQC)
# ---------------------------
def build_vqc(num_qubits, params):
    """Build parameterized quantum circuit.
       num_qubits = number of encoding qubits (=3 here).
       params = list/array of rotation parameters (size depends on ansatz).
       Encoding will NOT be in the same circuit builder; instead we build
       a function that accepts encoded angles and returns a circuit."""
    # We'll build a function returning a circuit given angles and param vector
    def circuit_for_angles(angles):
        # angles: list of length num_qubits with values in [0, pi]
        qc = QuantumCircuit(num_qubits, num_qubits)
        # Feature encoding: RX with sensor angle
        for i, a in enumerate(angles):
            qc.rx(a, i)
        # Variational layers: we'll interpret params as RY params in one layer plus entangling CZs
        # Expect len(params) == num_qubits for single layer; generalizable
        pvec = params.reshape(-1)
        # apply RY to each qubit
        for i in range(num_qubits):
            qc.ry(pvec[i], i)
        # simple entanglement: CZ chain
        for i in range(num_qubits - 1):
            qc.cz(i, i+1)
        # final rotations (reuse remaining params if present)
        if len(pvec) >= 2 * num_qubits:
            for i in range(num_qubits):
                qc.rz(pvec[num_qubits + i], i)
        # measure all qubits
        qc.measure(range(num_qubits), range(num_qubits))
        return qc
    return circuit_for_angles

# ---------------------------
# Utility: run circuit and interpret output
# ---------------------------
def predict_action_counts(circ_fn, angles, backend, shots=512):
    """Execute the parameterized circuit for given encoded angles and get counts.
       Interpretation: we measure all qubits; choose the action index corresponding to the measured qubit with highest probability of being '1'."""
    qc = circ_fn(angles)
    job = execute(qc, backend=backend, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)
    # convert counts to probabilities of each qubit being '1'
    num_qubits = qc.num_qubits
    probs_one = np.zeros(num_qubits)
    total_shots = shots
    for bitstring, c in counts.items():
        # qiskit returns bitstrings with qubit0 as least-significant by default ordering '...'
        # For mapping simplicity, reverse bitstring so index 0 corresponds to qubit 0
        bitstring_rev = bitstring[::-1]
        for i, ch in enumerate(bitstring_rev):
            if ch == '1':
                probs_one[i] += c
    probs_one = probs_one / total_shots
    # action selection: index with maximum prob of 1
    chosen = int(np.argmax(probs_one))
    return chosen, probs_one

# ---------------------------
# Dataset generation & training
# ---------------------------
def generate_synthetic_dataset(n_samples=200):
    """Generate sensor patterns and labels. The rule:
       - If center reading small (close to path), label FORWARD
       - Else if left reading < right reading => LEFT
       - Else RIGHT
       Add noise to readings to make problem nontrivial."""
    X = []
    y = []
    for _ in range(n_samples):
        # sample lateral offset of rover between -1.0 and +1.0
        lateral = random.uniform(-1.0, 1.0)  # y coordinate relative to path centerline
        heading_error = random.uniform(-math.pi/6, math.pi/6)  # small heading offset
        # compute ideal sensor readings by probing at sensor_range = 1.0
        sensor_range = 1.0
        offsets = [math.radians(30), 0.0, -math.radians(30)]
        readings = []
        for ao in offsets:
            probe_y = lateral + sensor_range * math.sin(heading_error + ao)
            d = abs(probe_y)
            d_norm = min(d / sensor_range, 1.0)
            readings.append(d_norm)
        readings = np.array(readings)
        # add gaussian noise
        readings += np.random.normal(0, 0.05, size=3)
        readings = np.clip(readings, 0.0, 1.0)
        # rule-based label
        if readings[1] < 0.25:
            label = ACTION_FORWARD
        elif readings[0] < readings[2]:
            label = ACTION_LEFT
        else:
            label = ACTION_RIGHT
        X.append(readings)
        y.append(label)
    return np.array(X), np.array(y)

def loss_on_dataset(params_vec, X, y, num_qubits, backend, shots=256):
    """Loss function to optimize: fraction of wrong predictions (simple)."""
    params = np.array(params_vec)
    circ_fn = build_vqc(num_qubits, params)
    n = len(X)
    wrong = 0
    for i in range(n):
        angles = X[i] * math.pi  # map normalized reading [0,1] -> [0, pi]
        pred, _ = predict_action_counts(circ_fn, angles, backend, shots=shots)
        if pred != y[i]:
            wrong += 1
    return wrong / n

# ---------------------------
# Main: train and run simple control loop
# ---------------------------
def main():
    seed = 42
    algorithm_globals.random_seed = seed
    np.random.seed(seed)
    random.seed(seed)

    # 1) make dataset
    X, y = generate_synthetic_dataset(n_samples=300)
    # split
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(0.8 * len(X))
    train_idx, val_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # 2) quantum model setup
    num_qubits = 3
    # parameter vector size: single-layer RY per qubit + RZ per qubit => 2*num_qubits
    param_size = 2 * num_qubits
    init_params = 0.1 * np.random.randn(param_size)

    backend = Aer.get_backend('qasm_simulator')

    # 3) training using classical optimizer (COBYLA through scipy)
    print("Starting parameter optimization (this may take a minute)...")
    def objective(p):
        return loss_on_dataset(p, X_train, y_train, num_qubits, backend, shots=256)

    res = minimize(objective, init_params, method='Nelder-Mead', options={'maxiter': 60, 'disp': True})
    params_opt = res.x
    print("Optimization complete. Final train loss (error fraction):", objective(params_opt))

    # validate
    train_err = loss_on_dataset(params_opt, X_train, y_train, num_qubits, backend, shots=384)
    val_err = loss_on_dataset(params_opt, X_val, y_val, num_qubits, backend, shots=384)
    print(f"Train error fraction: {train_err:.3f}  |  Val error fraction: {val_err:.3f}")

    # 4) run a simulated control episode
    env = SimpleRoverEnv(x=0.0, y=0.7, theta=0.0)
    circ_fn = build_vqc(num_qubits, params_opt)

    positions = []
    for step in range(80):
        sensors = env.get_sensors()
        angles = sensors * math.pi
        pred_action, probs = predict_action_counts(circ_fn, angles, backend, shots=384)
        env.step(pred_action)
        positions.append((env.x, env.y, pred_action))
        # stop early if close to path center
        if abs(env.y) < 0.05 and step > 10:
            break

    # 5) print trajectory summary
    print("Trajectory (x, y, action) sample:")
    for i, (px, py, a) in enumerate(positions[:20]):
        print(f"{i:02d}: x={px:.3f}, y={py:.3f}, action={ACTIONS[a]}")

    # 6) optional: plot path projection
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    acts = [p[2] for p in positions]
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker='o', label='rover path')
    plt.axhline(0.0, color='k', linestyle='--', label='path centerline (y=0)')
    for i, (xx, yy, a) in enumerate(positions):
        plt.text(xx, yy, ACTIONS[a][0], fontsize=8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Simulated rover trajectory under quantum controller')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
