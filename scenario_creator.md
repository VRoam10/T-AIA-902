# Scenario Creator — Getting Position & Rotation from BeamNG.drive

This guide explains how to find the exact world-space **position** and **rotation quaternion** of a vehicle while BeamNG.drive is running normally, so you can paste those values into `beamng_env.py` as `SPAWN_POS`, `SPAWN_ROT`, or `BASE_WAYPOINTS`.

---

## Method 1 — In-game Lua console (no Python needed)

1. In-game, press **F8** to open the Lua console.
2. Type the following and press **Enter**:

```lua
print(be:getObjectByID(be:getPlayerVehicleID(0)):getPosition())
print(be:getObjectByID(be:getPlayerVehicleID(0)):getRotation())   -- returns a quaternion
```

3. The output appears in the console overlay. Note the 3 position values and 4 rotation values.

> **Note:** BeamNG's Lua quaternion order is **(x, y, z, w)**, same as beamngpy's `rot_quat`.

---

## Method 2 — World Editor (for static waypoints)

1. Press **F11** to open the World Editor.
2. Click **Create → Sphere / Waypoint** and place it where you want.
3. In the Inspector panel on the right you will see the **Position** (`X Y Z`) fields.
4. Copy those values into `BASE_WAYPOINTS` in `beamng_env.py`.

> Rotation is not needed for waypoints — only position `(x, y, z)` matters.

---

## How the values map to `beamng_env.py`

```python
# beamng_env.py

# Vehicle spawn point — use Method 1 or 2
SPAWN_POS = (-391.0, -110.0, 1.0)      # (x, y, z) world coords
SPAWN_ROT = (0.0, 0.0, 1.0, 0.0)       # quaternion (x, y, z, w)

# Path the agent must follow — use Method 1, 2, or 3 for each point
BASE_WAYPOINTS = [
    (-370.0, -85.0, 1.0),
    (-320.0, -65.0, 1.0),
    # ... add as many as you need
]
```

### Quick rotation reference

| Heading (approx.) | `rot_quat` (x, y, z, w) |
|---|---|
| North (+Y) | `(0, 0, 0, 1)` |
| East (+X)  | `(0, 0, 0.707, 0.707)` |
| South (−Y) | `(0, 0, 1, 0)` |
| West (−X)  | `(0, 0, −0.707, 0.707)` |

For precise rotations, always read the value from the game rather than calculating it manually.

---

## Workflow summary

```
1. Launch BeamNG normally
2. Load the map you want (e.g. smallgrid, gridmap_v2, automation_test_track)
3. Drive/place the vehicle at the desired spawn or waypoint location
4. Run get_position.py (Method 1) OR use F8 console (Method 2)
5. Copy (x, y, z) → SPAWN_POS / BASE_WAYPOINTS
6. Copy (x, y, z, w) → SPAWN_ROT
7. Restart the training script
```
