PITCH_FEATURES = ["in_zone", "same_hand", "stand", "p_throws", "pitch_type", "release_speed", "release_spin_rate", "spin_axis",
                  "release_extension", "release_pos_x", "release_pos_z", "pfx_x", "pfx_z", "plate_x", "plate_z"]   

BASE_FEATURES = ["same_hand", "in_zone", "pitch_type"]

PITCH_GROUPS = {"stuff": ["release_speed", "release_spin_rate", "spin_axis"],
                "release": ["release_pos_x", "release_pos_z", "release_extension"],
                "location": ["plate_x", "plate_z"],
                "movement": ["pfx_x", "pfx_z"],
                "matchup": ["stand", "p_throws"]}

CONTEXT_FEATURES = PITCH_FEATURES + ["inning", "bat_score_diff", "count",
                                     "outs_when_up", "on_3b", "on_2b", "on_1b"]

CONTEXT_GROUPS = {"base_out_state": ["on_3b", "on_2b", "on_1b", "outs_when_up"],
                  "game_state": ["inning", "bat_score_diff"],
                  "count": ["count"]}

PITCH_CATEGORICAL_FEATURES = ["in_zone", "same_hand", "stand", "p_throws", "pitch_type"]

CONTEXT_CATEGORICAL_FEATURES = PITCH_CATEGORICAL_FEATURES + ["count"]
