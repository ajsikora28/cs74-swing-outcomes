"""Feature lists shared by every model so all comparisons use identical inputs."""

PITCH_FEATURES = []    # pitch type, velocity, spin, spin axis, movement, release point, location
CONTEXT_FEATURES = []  # handedness, count, outs, baserunners (added on top of PITCH_FEATURES)
