"""Colour tokens shared by the figure scripts (light and dark variants).

Series and chrome colours follow a validated palette: the series colour clears
3:1 contrast on its surface in both modes, and the sequential ramp is a single
hue running light to dark.
"""

THEMES = {
    "light": dict(surface="#fcfcfb", series="#2a78d6", ink="#0b0b0b", ink2="#52514e",
                  muted="#898781", grid="#e1e0d9", axis="#c3c2b7"),
    "dark": dict(surface="#1a1a19", series="#3987e5", ink="#f0efec", ink2="#c3c2b7",
                 muted="#898781", grid="#2c2c2a", axis="#383835"),
}

# Sequential blue, steps 100 -> 700. Low values sit near the surface colour:
# light steps first on a light surface, dark steps first on a dark one.
_BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQUENTIAL = {"light": _BLUE_RAMP, "dark": _BLUE_RAMP[::-1]}
