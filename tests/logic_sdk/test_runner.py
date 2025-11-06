from logic.runner import run_match


def noop_ai(round_idx, my_seat, state):
    # end turn immediately
    return [[8]]


def test_run_match_completes():
    winner, state = run_match(noop_ai, noop_ai, seed=0, max_rounds=3)
    # With immediate passes, winner decided by tiebreak
    assert winner in (0, 1)
    # Stored under default replays directory
    assert state.replay_file.startswith("replays/") or \
           state.replay_file.split("/")[-2] == "replays"
