"""Head-to-head racing on one shared track.

`BeamNGRaceEnv` extends :class:`~environments.beamng_multi.BeamNGMultiEnv`, which
already owns everything hard about running several cars: one scenario with N
vehicles, one physics step for all of them, per-slot sensors, observations and
reward, and a bounded shutdown. Racing changes four things:

1. **One shared path, offset spawns.** Multi-agent *training* deliberately gives
   each vehicle its own path (>= ``MIN_PATH_SEPARATION_M`` apart) so the cars never
   meet. A race needs the opposite: everyone on the same waypoint list, spread
   across a starting grid so they do not spawn inside one another.
2. **Collisions.** Nothing to enable — BeamNG vehicles in one scenario collide
   physically, and the reward's damage terms already price contact. Training only
   avoided contact because of the path separation above.
3. **A gap-aware reward.** Each step every slot's position along the shared path is
   measured in metres, and the best *other* slot's position is fed to the reward as
   the rival, which pays for metres gained. Solo training passes nothing and the
   term vanishes.
4. **Whole-race resets.** Training resets one finished vehicle while the others
   keep driving (``reset_vehicle``). A race is over for everyone at once, so all
   cars reset together.

A human entrant is a slot with ``human=True`` and no agent: it receives no control
input from us, because the player is driving it.
"""

import numpy as np

from environments import beamng_spec
from environments.beamng_multi import BeamNGMultiEnv, VehicleSlot


def build_race_slots(specs: list[dict]) -> list[VehicleSlot]:
    """Build the entrants' slots, including an optional human.

    Each spec: ``{"algo", "agent", "color", "save_path", "sensor",
    "trajectory_hints", "body_orientation", "road_info", "wheel_info", "human"}``.
    A spec with ``human=True`` needs no algorithm or agent — the player drives it,
    so it has no observation to size and no action head. Every other spec goes
    through the same sizing as multi-agent training.
    """
    from environments.beamng_multi import slot_n_states

    slots = []
    for i, spec in enumerate(specs):
        if spec.get("human"):
            slots.append(
                VehicleSlot(
                    name=f"human_{i}",
                    color=spec.get("color", "Red"),
                    agent=None,
                    save_path="",
                    human=True,
                )
            )
            continue
        sensor = spec.get("sensor", beamng_spec.DEFAULT_SENSOR)
        hints = spec.get("trajectory_hints", 0)
        body = spec.get("body_orientation", False)
        road = spec.get("road_info", False)
        wheel = spec.get("wheel_info", False)
        slots.append(
            VehicleSlot(
                name=f"racer_{i}",
                color=spec["color"],
                agent=spec["agent"],
                save_path=spec.get("save_path", ""),
                sensor=sensor,
                output=beamng_spec.output_for_algo(spec["algo"]),
                trajectory_hints=hints,
                body_orientation=body,
                road_info=road,
                wheel_info=wheel,
                n_states=slot_n_states(sensor, hints, body, road, wheel),
            )
        )
    return slots


class BeamNGRaceEnv(BeamNGMultiEnv):
    """N vehicles racing the same path, with collisions and a gap-based reward."""

    # Grid geometry (GRID_LATERAL_M / GRID_STAGGER_M) is inherited from
    # BeamNGMultiEnv, which also needs it when training on a shared game track.

    def __init__(
        self,
        slots: list[VehicleSlot],
        beamng_home: str,
        beamng_user: str = None,
        host: str = "localhost",
        port: int = 25252,
        headless: bool = False,
        map_name: str = "gridmap_v2",
        path_idx: int = 0,
        laps: int = 1,
        realtime: bool = False,
        track: str | None = None,
    ):
        """
        Args:
            path_idx: Which of the map's generated paths to race on. Ignored when
                ``track`` is set, since a game track is a single path.
            laps: Reserved. Must be 1. A generated path is an open road, so a
                second lap would mean driving back to the start; a game track
                marked ``closed`` does return to its start, but counting laps
                (rather than finishing at the last checkpoint) is not wired up yet.
            realtime: False runs lockstep (``bng.step``), which is deterministic and
                as fast as the sim allows — right for agent-vs-agent. True lets the
                simulator run freely so a human can drive at wall-clock speed.
            track: One of the game's own race tracks — a :mod:`core.quickrace` key
                such as ``"race_track"`` — raced instead of a generated path. These
                are authored racing lines and include real closed circuits.
        """
        if laps != 1:
            raise ValueError(
                f"laps={laps} is not supported yet: a race finishes at the last "
                "checkpoint rather than counting laps. Use laps=1 — on a closed game "
                "track that is one full lap."
            )
        # random_path is meaningless in a race: every car must run the same path.
        super().__init__(
            slots=slots,
            beamng_home=beamng_home,
            beamng_user=beamng_user,
            host=host,
            port=port,
            headless=headless,
            map_name=map_name,
            random_path=False,
            track=track,
        )
        self.path_idx = path_idx
        self.laps = laps
        self.realtime = realtime
        self._resumed = False

    # ------------------------------------------------------------------
    # Shared path + starting grid
    # ------------------------------------------------------------------

    def _assign_paths(self):
        """Put every entrant on the same path, spread across a starting grid.

        Overrides the training behaviour (one distinct path per vehicle, which
        errors when vehicles outnumber paths). Here the only limit is the grid.
        """
        paths = self.trajectories.paths
        if not paths:
            raise ValueError(f"map '{self.map_name}' produced no drivable path")
        idx = min(self.path_idx, len(paths) - 1)
        self._assign_shared_path(paths[idx], path_idx=idx)

    def _update_slot_marker(self, slot: VehicleSlot):
        """No per-slot target markers in a race.

        Every car aims at the same waypoint, so per-slot spheres would stack on top
        of each other in different colours. The scenario's checkpoint rings already
        show the line both cars are driving.
        """
        return

    # ------------------------------------------------------------------
    # Progress and the gap-aware reward
    # ------------------------------------------------------------------
    # progress_of is inherited from BeamNGMultiEnv — one definition of "how far
    # along am I", shared by training's pace reward and this env's gap term.

    def leader(self) -> VehicleSlot | None:
        """The slot furthest along the path, or None when there are no slots."""
        if not self.slots:
            return None
        return max(self.slots, key=self.progress_of)

    def standings(self) -> list[tuple[str, float]]:
        """(name, progress_m) for every entrant, leader first."""
        ranked = sorted(self.slots, key=self.progress_of, reverse=True)
        return [(s.name, self.progress_of(s)) for s in ranked]

    def compute_race_reward_for(self, slot: VehicleSlot, obs) -> tuple[float, bool]:
        """Reward one slot, with the best rival's progress feeding the gap term."""
        rivals = [s for s in self.slots if s is not slot]
        if not rivals:
            return self.compute_reward(slot, obs)

        best_rival = max(rivals, key=self.progress_of)
        return self.compute_reward(
            slot,
            obs,
            laps=self.laps,
            # progress_m / last_progress_m are not passed here: compute_reward
            # (the base env's) now supplies them itself, from the same
            # progress_of/last_progress_m the pace term reads. Passing them again
            # here would collide with that (TypeError: multiple values for
            # keyword argument).
            rival_progress_m=self.progress_of(best_rival),
            last_rival_progress_m=slot.last_rival_progress_m,
            rival_finished=any(s.finished for s in rivals),
        )

    def snapshot_progress(self) -> None:
        """Record every slot's own and best-rival progress for the next step's gap.

        Called once per tick *after* rewards are computed, so each slot's stored
        "last" values are the ones the next step must telescope against. Doing it
        per-slot inside the reward would let one slot's update change another's
        baseline mid-tick.
        """
        current = {s.name: self.progress_of(s) for s in self.slots}
        for slot in self.slots:
            rivals = [s for s in self.slots if s is not slot]
            slot.last_progress_m = current[slot.name]
            slot.last_rival_progress_m = (
                max(current[s.name] for s in rivals) if rivals else current[slot.name]
            )

    # ------------------------------------------------------------------
    # Race lifecycle
    # ------------------------------------------------------------------

    def agent_slots(self) -> list[VehicleSlot]:
        """Entrants we drive. Excludes the human, who drives themselves."""
        return [s for s in self.slots if not s.human]

    def human_slots(self) -> list[VehicleSlot]:
        return [s for s in self.slots if s.human]

    def race_over(self) -> bool:
        """True once someone has finished, or nobody is left running."""
        if any(s.finished for s in self.slots):
            return True
        return all(s.done for s in self.slots)

    def winner(self) -> VehicleSlot | None:
        """The entrant that finished, else whoever is furthest along."""
        finishers = [s for s in self.slots if s.finished]
        if finishers:
            return finishers[0]
        return self.leader()

    def result(self) -> dict:
        """Summarise the race: winner, per-entrant progress, margin and steps."""
        win = self.winner()
        ranked = sorted(self.slots, key=self.progress_of, reverse=True)
        margin = 0.0
        if len(ranked) >= 2:
            margin = self.progress_of(ranked[0]) - self.progress_of(ranked[1])
        return {
            "winner": win.name if win else None,
            "margin_m": round(float(margin), 2),
            "entrants": [
                {
                    "name": s.name,
                    "human": s.human,
                    "progress_m": round(float(self.progress_of(s)), 2),
                    "checkpoints": s.waypoint_idx,
                    "steps": s.steps,
                    "finished": s.finished,
                }
                for s in ranked
            ],
        }

    def reset_race(self) -> None:
        """Return every car to its grid slot and zero all episode state.

        Unlike training's per-vehicle reset, the whole field restarts together — a
        race that is over is over for everyone.
        """
        self.reset_all()
        for slot in self.slots:
            slot.last_progress_m = self.progress_of(slot)
        self.snapshot_progress()

    def advance(self) -> None:
        """Advance the simulation by one race tick.

        Lockstep mode steps a fixed number of physics ticks, matching training.
        Realtime mode resumes the simulator once and then lets wall-clock time pass,
        because a human cannot drive in lockstep — the caller paces the loop.
        """
        if not self.realtime:
            self.step_physics()
            return
        if not self._resumed:
            self.bng.resume()
            self._resumed = True
            # Realtime race: resuming lets the sim advance continuously on its own
            # from here on, so the road sensor is always safe to poll — the same
            # reasoning as BeamNGDrivingEnv.human_play (docs/romain.md, seventh
            # issue). Nothing else in a race ever calls step_physics() to reopen
            # this once realtime, so it must be opened explicitly here.
            self._road_pollable = True

    def observe_all(self) -> dict[str, np.ndarray]:
        """Poll **every** entrant, returning observations for the driven ones only.

        The human is polled too, and deliberately so: ``observe`` is what advances a
        slot's waypoint index and current position, which is what ``progress_of``
        reads. Skipping the human would freeze their progress at the grid and make
        the gap term meaningless for their rival.
        """
        obs = {}
        for slot in self.slots:
            value = self.observe(slot)
            if not slot.human:
                obs[slot.name] = value
        return obs

    def _create_slot_sensor(self, slot: VehicleSlot):
        """Skip perception for a human entrant — nothing consumes it.

        A LiDAR or dashcam for the player would cost GPU every frame to feed an
        observation vector no policy reads. Their position and checkpoints come from
        the vehicle state, which needs no extra sensor.
        """
        if slot.human:
            return
        super()._create_slot_sensor(slot)

    def _load_scenario(self):
        """Load the shared-track scenario, then give the human input focus.

        BeamNG routes keyboard input to the focused vehicle, so without this the
        player would be driving whichever car the simulator picked (usually the
        first one added) while we also sent it agent controls.
        """
        super()._load_scenario()
        self._focus_human()

    def _focus_human(self) -> None:
        """Point the simulator's camera and input at the human's car, if any.

        The beamngpy API for this moved between versions, so try the known spellings
        and carry on without focus rather than failing the race: a wrong camera is
        recoverable in-game, a crashed launch is not.
        """
        humans = self.human_slots()
        if not humans or self.bng is None:
            return
        vehicle = humans[0].vehicle
        for target, attr in (
            (getattr(self.bng, "vehicles", None), "switch_vehicle"),
            (self.bng, "switch_vehicle"),
        ):
            fn = getattr(target, attr, None) if target is not None else None
            if callable(fn):
                try:
                    fn(vehicle)
                    return
                except Exception as exc:  # noqa: BLE001 — focus is best-effort
                    print(f"[BeamNGRaceEnv] Could not focus the human vehicle: {exc}")
                    return
        print(
            "[BeamNGRaceEnv] Warning: this beamngpy build exposes no switch_vehicle; "
            "select your car in-game if the keyboard drives the wrong one."
        )
