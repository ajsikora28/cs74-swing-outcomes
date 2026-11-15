PITCH_FEATURES = ["in_zone","same_hand", "stand", "p_throws", "pitch_type", "release_speed", "release_spin_rate", "spin_axis",
                  "release_extension", "release_pos_x", "release_pos_z", "pfx_x", "pfx_z", "plate_x", "plate_z"]   

CONTEXT_FEATURES = PITCH_FEATURES + ["inning", "bat_score_diff", "count",
                                     "outs_when_up", "on_3b", "on_2b", "on_1b"]

PITCH_CATEGORICAL_FEATURES = ["in_zone", "same_hand", "stand", "p_throws", "pitch_type"]

CONTEXT_CATEGORICAL_FEATURES = PITCH_CATEGORICAL_FEATURES + ["count"]
