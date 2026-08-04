"""Non-interactive command bridge between the OpenTUI app and the action layer.

The TypeScript TUI shells out to ``python -m core.tui_backend <command>``,
passing all decisions as a JSON ``--config-json`` payload. Long-running runner
output streams to stdout; a final ``[TUI_RESULT] <json>`` line carries the
structured result. Errors go to stderr prefixed with ``[TUI_ERROR] ``.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading

from core.pipeline_actions import (
    BeamNGOptions,
    BenchmarkRequest,
    CourseRequest,
    EvaluateRequest,
    HumanPlayRequest,
    MultiTrainRequest,
    RacerSpec,
    TrainRequest,
    TrajectoryRequest,
    catalog,
    run_benchmark,
    run_course,
    run_evaluate,
    run_human_play,
    run_multi_train,
    run_train,
    run_trajectory,
)
from core.stop_signal import request_stop
from environments import beamng_spec

RESULT_PREFIX = "[TUI_RESULT] "
ERROR_PREFIX = "[TUI_ERROR] "


def _beamng_from(payload: dict) -> BeamNGOptions | None:
    raw = payload.get("beamng")
    if not raw:
        return None
    return BeamNGOptions(**raw)


def _emit_result(result: object) -> None:
    print(RESULT_PREFIX + json.dumps(result))


def _cmd_catalog(_payload: dict) -> None:
    print(json.dumps(catalog()))


def _cmd_train(payload: dict) -> None:
    req = TrainRequest(
        algo_name=payload["algo_name"],
        env_name=payload["env_name"],
        n_episodes=payload["n_episodes"],
        save_path=payload["save_path"],
        agent_params=payload.get("agent_params", {}),
        beamng=_beamng_from(payload),
        reset_existing=payload.get("reset_existing", False),
    )
    _emit_result(run_train(req))


def _cmd_evaluate(payload: dict) -> None:
    req = EvaluateRequest(
        algo_name=payload["algo_name"],
        env_name=payload["env_name"],
        model_path=payload["model_path"],
        n_episodes=payload["n_episodes"],
        beamng=_beamng_from(payload),
    )
    _emit_result(run_evaluate(req))


def _cmd_benchmark(payload: dict) -> None:
    req = BenchmarkRequest(
        benchmark_name=payload["benchmark_name"],
        seeds=payload["seeds"],
        eval_episodes=payload["eval_episodes"],
        success_threshold=payload["success_threshold"],
        max_episodes=payload["max_episodes"],
        reward_threshold=payload.get("reward_threshold", 7.0),
        algo_name=payload.get("algo_name"),
        env_name=payload.get("env_name"),
        algos=payload.get("algos"),
        param_grid=payload.get("param_grid"),
        beamng=_beamng_from(payload),
    )
    _emit_result(run_benchmark(req))


def _cmd_human_play(payload: dict) -> None:
    req = HumanPlayRequest(
        map_name=payload["map_name"],
        sensor=payload.get("sensor", beamng_spec.DEFAULT_SENSOR),
        random_path=payload.get("random_path", False),
        track=payload.get("track", ""),
        road_info=payload.get("road_info", False),
        wheel_info=payload.get("wheel_info", False),
    )
    run_human_play(req)
    _emit_result({"status": "stopped"})


def _cmd_trajectory(payload: dict) -> None:
    req = TrajectoryRequest(
        map_name=payload["map_name"],
        overwrite=payload.get("overwrite", False),
    )
    _emit_result(run_trajectory(req))


def _cmd_multi_train(payload: dict) -> None:
    req = MultiTrainRequest(
        map_name=payload["map_name"],
        random_path=payload.get("random_path", False),
        specs=payload["specs"],
        n_episodes=payload["n_episodes"],
        time_limit_minutes=payload.get("time_limit_minutes", 0.0),
        reset_existing=payload.get("reset_existing", False),
        track=payload.get("track", ""),
    )
    _emit_result(run_multi_train(req))


def _cmd_course(payload: dict) -> None:
    req = CourseRequest(
        map_name=payload["map_name"],
        racers=[RacerSpec(**raw) for raw in payload["racers"]],
        laps=payload.get("laps", 1),
        races=payload.get("races", 1),
        learning=payload.get("learning", False),
        path_idx=payload.get("path_idx", 0),
        track=payload.get("track", ""),
    )
    _emit_result(run_course(req))


_COMMANDS = {
    "train": _cmd_train,
    "evaluate": _cmd_evaluate,
    "benchmark": _cmd_benchmark,
    "human-play": _cmd_human_play,
    "trajectory": _cmd_trajectory,
    "multi-train": _cmd_multi_train,
    "course": _cmd_course,
}


def _watch_for_stop() -> None:
    """Set the cooperative stop flag when the TUI sends "STOP" on stdin.

    The TUI cancels a run (or quits the app) by writing "STOP" to our stdin. The
    running command's loop polls ``stop_requested()`` and breaks at a safe point
    so its ``finally`` runs the usual cleanup — saving checkpoints and closing
    BeamNG — instead of us being hard-killed and orphaning the simulator. Only an
    explicit "STOP" trips it; a bare EOF is ignored so a run is never killed at
    startup.
    """
    for line in sys.stdin:
        if line.strip().upper() == "STOP":
            request_stop()
            return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="core.tui_backend")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("catalog")
    for name in _COMMANDS:
        p = sub.add_parser(name)
        p.add_argument("--config-json", required=True)

    args = parser.parse_args(argv)

    if args.command == "catalog":
        _cmd_catalog({})
        return 0

    threading.Thread(target=_watch_for_stop, daemon=True).start()
    try:
        payload = json.loads(args.config_json)
        _COMMANDS[args.command](payload)
    except KeyboardInterrupt:
        # Terminal Ctrl+C: the command's `finally` already ran its cleanup.
        pass
    except Exception as exc:  # noqa: BLE001 — bridge reports any failure to TUI
        print(f"{ERROR_PREFIX}{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
