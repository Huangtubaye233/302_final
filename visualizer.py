from argparse import ArgumentParser
import copy
import json
import time

import numpy as np
from flask import Flask, Response, render_template

from simulator import Simulator
from utils import load_config

app = Flask(
    __name__, 
    template_folder="visualizer/templates",
    static_folder="visualizer/static",
)

TARGET_FPS = 60.0

app_state = {
    "actual_fps": 0.0,
}
panels = []

@app.route("/")
def index():
    return render_template("index.html")

def ground_height_at(x, sim_config):
    return float(sim_config.get("ground_height", 0.02))


def build_terrain_profile(sim_config, x_min=-3.0, x_max=3.0, n=180):
    xs = np.linspace(x_min, x_max, n)
    return [[float(x), float(ground_height_at(float(x), sim_config))] for x in xs]


def step_panel(panel):
    t = panel["step_index"]
    max_steps = panel["max_steps"]
    simulator = panel["simulator"]

    if t >= max_steps:
        simulator.reinitialize_robots()
        panel["step_index"] = 0
        panel["coverage_min_x"] = np.inf
        panel["coverage_max_x"] = -np.inf
        t = 0

    simulator.compute_com(t)
    simulator.nn1(t)
    simulator.nn2(t)
    simulator.apply_spring_force(t)
    simulator.advance(t + 1)

    positions = simulator.x.to_numpy()[0, t + 1, :panel["n_masses"]]
    activations = simulator.act.to_numpy()[0, t, :panel["n_springs"]]
    center_of_mass = positions.mean(axis=0)

    panel["coverage_min_x"] = min(panel["coverage_min_x"], float(center_of_mass[0]))
    panel["coverage_max_x"] = max(panel["coverage_max_x"], float(center_of_mass[0]))
    coverage = panel["coverage_max_x"] - panel["coverage_min_x"]

    panel["step_index"] = t + 1
    sim_config = panel["sim_config"]
    return {
        "id": panel["id"],
        "title": panel["title"],
        "movement_style": panel.get("movement_style", "neutral"),
        "morphology_reward": panel.get("morphology_reward", "none"),
        "evolve_shape": panel.get("evolve_shape", False),
        "positions": positions.tolist(),
        "activations": activations.tolist(),
        "center_of_mass": center_of_mass.tolist(),
        "coverage": float(max(0.0, coverage)),
        "step": panel["step_index"],
    }

@app.route("/stream")
def stream():
    """Server-sent events stream for 2x2 comparison visualization."""
    def event_stream():
        topology = {
            "type": "topology_batch",
            "panels": [
                {
                    "id": panel["id"],
                    "title": panel["title"],
                    "movement_style": panel.get("movement_style", "neutral"),
                    "morphology_reward": panel.get("morphology_reward", "none"),
                    "evolve_shape": panel.get("evolve_shape", False),
                    "springs": panel["robot"]["springs"].tolist(),
                    "n_masses": int(panel["n_masses"]),
                    "n_springs": int(panel["n_springs"]),
                    "terrain_profile": build_terrain_profile(panel["sim_config"]),
                }
                for panel in panels
            ],
        }
        yield f"data: {json.dumps(topology)}\n\n"

        fps_samples = []
        last_fps_update = time.perf_counter()

        while True:
            frame_start = time.perf_counter()
            target_interval = 1.0 / TARGET_FPS

            panel_payloads = [step_panel(panel) for panel in panels]
            payload = {
                "type": "step_batch",
                "panels": panel_payloads,
                "fps": app_state["actual_fps"],
            }
            yield f"data: {json.dumps(payload)}\n\n"

            frame_end = time.perf_counter()
            work_time = frame_end - frame_start

            sleep_time = target_interval - work_time
            if sleep_time > 0.001:
                time.sleep(sleep_time)

            total_frame_time = time.perf_counter() - frame_start
            if total_frame_time > 0:
                fps_samples.append(1.0 / total_frame_time)

            current_time = time.perf_counter()
            if current_time - last_fps_update >= 0.5:
                if fps_samples:
                    app_state["actual_fps"] = sum(fps_samples) / len(fps_samples)
                    fps_samples = []
                    last_fps_update = current_time

    response = Response(event_stream(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=str,
        default="robot_00.npy,robot_01.npy,robot_10.npy,robot_11.npy",
        help="Comma-separated list of 4 robot .npy files",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    paths = [p.strip() for p in args.inputs.split(",") if p.strip()]
    if len(paths) != 4:
        raise ValueError("Expected exactly 4 input robot files.")

    base_config = load_config(args.config)
    default_panels = [
        {"title": "00 Normal Morph + Neutral", "movement_style": "neutral", "morphology_reward": "none", "evolve_shape": False},
        {"title": "01 Normal Morph + Jump", "movement_style": "jump", "morphology_reward": "none", "evolve_shape": False},
        {"title": "10 Sparse Morph + Neutral", "movement_style": "neutral", "morphology_reward": "spread", "evolve_shape": True},
        {"title": "11 Sparse Morph + Jump", "movement_style": "jump", "morphology_reward": "spread", "evolve_shape": True},
    ]
    panels = []

    for idx, path in enumerate(paths):
        robot = np.load(path, allow_pickle=True).item()
        scenario = robot.get("scenario", {})
        fallback = default_panels[idx]
        panel_title = scenario.get("title", fallback["title"])

        scenario_sim_config = copy.deepcopy(base_config["simulator"])
        for key, value in scenario.get("sim_config", {}).items():
            scenario_sim_config[key] = value
        scenario_sim_config["n_sims"] = 1
        if "max_n_masses" in robot and "max_n_springs" in robot:
            scenario_sim_config["n_masses"] = int(robot["max_n_masses"])
            scenario_sim_config["n_springs"] = int(robot["max_n_springs"])
        else:
            scenario_sim_config["n_masses"] = int(robot["n_masses"])
            scenario_sim_config["n_springs"] = int(robot["n_springs"])

        simulator = Simulator(
            sim_config=scenario_sim_config,
            taichi_config=base_config["taichi"],
            seed=base_config["seed"] + idx,
            needs_grad=False,
        )
        spring_types = [robot["spring_types"]] if "spring_types" in robot else None
        simulator.initialize([robot["masses"]], [robot["springs"]], spring_types)
        if "control_params" in robot:
            simulator.set_control_params([0], [robot["control_params"]])

        panels.append(
            {
                "id": f"{idx:02d}",
                "title": panel_title,
                "movement_style": scenario.get("movement_style", fallback["movement_style"]),
                "morphology_reward": scenario.get("morphology_reward", fallback["morphology_reward"]),
                "evolve_shape": scenario.get("evolve_shape", fallback["evolve_shape"]),
                "robot": robot,
                "sim_config": scenario_sim_config,
                "simulator": simulator,
                "max_steps": int(simulator.steps[None]),
                "n_masses": int(simulator.n_masses[0]),
                "n_springs": int(simulator.n_springs[0]),
                "step_index": 0,
                "coverage_min_x": np.inf,
                "coverage_max_x": -np.inf,
            }
        )

    print(f"\nVisualizer running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop\n")

    app.run(host="0.0.0.0", port=args.port, debug=args.debug, threaded=False, use_reloader=False)