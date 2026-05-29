# Human Play Sensor Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sensor selection prompt to the human play CLI — None, LiDAR, or Camera — each routing to a dedicated play method on the appropriate BeamNG env class.

**Architecture:** Strip the existing LiDAR print from `human_play()` so it becomes a clean no-sensor loop, add `human_play_lidar()` alongside it, and add `BeamNGCameraEnv.human_play_camera()` that renders the 16×16 frame as in-place ASCII art. The CLI prompt routes to the right env class and method.

**Tech Stack:** Python 3.11, beamngpy, numpy, ANSI escape codes for in-place terminal rendering.

---

## Files Changed

| File | Change |
|------|--------|
| `environments/beamng.py` | Add `import sys`; strip LiDAR print from `human_play()`; add `human_play_lidar()` to `BeamNGContinuousEnv`; add `human_play_camera()` to `BeamNGCameraEnv` |
| `core/cli.py` | Add sensor prompt + routing in `_human_play_menu()` |

---

### Task 1: Refactor `human_play()` and add `human_play_lidar()` in `BeamNGContinuousEnv`

**Files:**
- Modify: `environments/beamng.py:1` (add `import sys`)
- Modify: `environments/beamng.py:218-253` (strip LiDAR print from `human_play()`)
- Modify: `environments/beamng.py:254` (add `human_play_lidar()` after `human_play()`)

- [ ] **Step 1: Add `import sys` at the top of `environments/beamng.py`**

  The current first line is `import random`. Add `sys` to that block:

  ```python
  import random
  import sys
  import threading
  import time
  ```

- [ ] **Step 2: Strip the LiDAR polling and print from `human_play()`**

  Current `human_play()` body (lines 218–253) contains this block inside the `while True` loop:

  ```python
  lidar_data = (
      self.lidar.poll().get("pointCloud", None) if self.lidar is not None else None
  )
  lidar_bins = self._process_lidar(lidar_data, pos, vehicle_heading)
  print(f"[LiDAR bins] {' '.join(f'{v:.2f}' for v in lidar_bins)}")
  ```

  Remove those five lines. The final `human_play()` method should be:

  ```python
  def human_play(self):
      """Load the scenario and give control back to the human player (no sensor output)."""
      if self.bng is None:
          self._launch(human_control=True)
      else:
          self._load_scenario(human_control=True)

      self._waypoint_idx = 0
      self._update_active_marker(1)

      self.bng.resume()
      print("[BeamNGDrivingEnv] Human control active — drive in-game. Press Ctrl+C to stop.")

      try:
          while True:
              self.vehicle.poll_sensors()
              time.sleep(0.1)
      except KeyboardInterrupt:
          print("[BeamNGDrivingEnv] Human play stopped.")
  ```

- [ ] **Step 3: Add `human_play_lidar()` immediately after `human_play()`**

  Insert this method right after the closing line of `human_play()`, before `def close(self)`:

  ```python
  def human_play_lidar(self):
      """Human play with LiDAR bins printed to stdout each tick."""
      if self.bng is None:
          self._launch(human_control=True)
      else:
          self._load_scenario(human_control=True)

      self._waypoint_idx = 0
      self._update_active_marker(1)

      self.bng.resume()
      print("[BeamNGDrivingEnv] Human control active — drive in-game. Press Ctrl+C to stop.")

      try:
          while True:
              self.vehicle.poll_sensors()
              state = self.vehicle.state or {}
              pos = state.get("pos", (0.0, 0.0, 0.0))
              vel = state.get("vel", (1.0, 0.0, 0.0))
              dir_vec = state.get("dir", vel)
              vehicle_heading = float(np.arctan2(dir_vec[1], dir_vec[0]))

              lidar_data = (
                  self.lidar.poll().get("pointCloud", None) if self.lidar is not None else None
              )
              lidar_bins = self._process_lidar(lidar_data, pos, vehicle_heading)
              print(f"[LiDAR bins] {' '.join(f'{v:.2f}' for v in lidar_bins)}")

              time.sleep(0.1)
      except KeyboardInterrupt:
          print("[BeamNGDrivingEnv] Human play stopped.")
  ```

- [ ] **Step 4: Verify the file parses without errors**

  ```
  python -c "import environments.beamng"
  ```

  Expected: no output (clean import).

---

### Task 2: Add `human_play_camera()` to `BeamNGCameraEnv`

**Files:**
- Modify: `environments/beamng.py` — add method to `BeamNGCameraEnv` after its `close()` method (currently ends around line 896)

- [ ] **Step 1: Add `human_play_camera()` to `BeamNGCameraEnv`**

  Insert this method inside `BeamNGCameraEnv`, after `def close(self):` and before the `# Camera processing` comment block:

  ```python
  def human_play_camera(self):
      """Human play with the 16×16 dashcam frame rendered as ASCII art in-place."""
      if self.bng is None:
          self._launch(human_control=True)
      else:
          self._load_scenario(human_control=True)

      self._waypoint_idx = 0
      self._update_active_marker(1)

      self.bng.resume()
      print("[BeamNGCameraEnv] Human control active — drive in-game. Press Ctrl+C to stop.")

      ramp = " ░▒▓█"
      h = self.CAM_OUT_SIZE[0]
      first = True

      try:
          while True:
              pixels = self._process_camera().reshape(self.CAM_OUT_SIZE)
              rows = ["".join(ramp[min(int(v * 4), 4)] for v in row) for row in pixels]
              if not first:
                  sys.stdout.write(f"\033[{h}A")
              sys.stdout.write("\n".join(rows) + "\n")
              sys.stdout.flush()
              first = False
              time.sleep(0.1)
      except KeyboardInterrupt:
          print("[BeamNGCameraEnv] Human play stopped.")
  ```

  **How the in-place rendering works:**
  - First tick: print 16 rows normally.
  - Subsequent ticks: `\033[{h}A` moves the cursor up 16 lines, then the 16 new rows overwrite the previous frame — no scrolling.
  - `ramp[min(int(v * 4), 4)]`: maps `v ∈ [0, 1]` → index 0–4 → `' '`, `'░'`, `'▒'`, `'▓'`, `'█'`.

- [ ] **Step 2: Verify the file parses without errors**

  ```
  python -c "import environments.beamng"
  ```

  Expected: no output (clean import).

---

### Task 3: Update `_human_play_menu()` in `core/cli.py`

**Files:**
- Modify: `core/cli.py:280-296`

- [ ] **Step 1: Replace `_human_play_menu()` with the new version**

  Replace the entire function (lines 280–296) with:

  ```python
  def _human_play_menu():
      print("\n--- Human Play (BeamNG) ---")
      envs = registry.list_environments()
      if "beamng" not in envs:
          print("BeamNG environment not registered.")
          return

      map_name, vehicle_id = _pick_beamng_options()

      print("\nShow sensor during play?")
      sensor = _pick(["None", "LiDAR", "Camera"], "Sensor")

      if sensor == "Camera":
          if "beamng_camera" not in envs:
              print("BeamNG camera environment not registered.")
              return
          env = registry.get_environment("beamng_camera")["factory"](
              map_name=map_name, vehicle_id=vehicle_id
          )
      else:
          env = registry.get_environment("beamng")["factory"](
              map_name=map_name, vehicle_id=vehicle_id
          )

      print("Launching BeamNG for human play...")
      try:
          if sensor == "LiDAR":
              env.human_play_lidar()
          elif sensor == "Camera":
              env.human_play_camera()
          else:
              env.human_play()
          input("\nPress Enter when done playing...")
      finally:
          env.close()
  ```

- [ ] **Step 2: Verify the file parses without errors**

  ```
  python -c "import core.cli"
  ```

  Expected: no output (clean import).

---

### Task 4: Manual verification

- [ ] **Step 1: Launch and select None**

  ```
  python main.py
  ```

  Select `4. Human play (BeamNG)` → pick any map and vehicle → select `1. None`.
  Expected: BeamNG launches, no sensor output in the terminal.

- [ ] **Step 2: Launch and select LiDAR**

  Same as above → select `2. LiDAR`.
  Expected: `[LiDAR bins] 0.82 0.91 ...` printed each tick (~10 Hz).

- [ ] **Step 3: Launch and select Camera**

  Same as above → select `3. Camera`.
  Expected: A 16-row ASCII block (`█▓▒░ ` characters) updates in-place in the terminal without scrolling.
