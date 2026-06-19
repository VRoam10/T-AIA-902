"""Shared test configuration.

Force a non-interactive matplotlib backend so the benchmark/runner plotting
code never reaches for a GUI toolkit (Tk) during the test run. This keeps the
suite headless and order-independent on CI and on machines without Tk.
"""

import matplotlib

matplotlib.use("Agg")
