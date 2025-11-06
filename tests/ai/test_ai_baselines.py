from logic.runner import run_match
from AI.ai_random_safe import policy as random_safe
from AI.ai_greedy import policy as greedy


def test_ai_interfaces_and_match():
    # Ensure both AIs can produce a match and replay saved under replays/
    winner, state = run_match(greedy, random_safe, max_rounds=5, p0_name="greedy", p1_name="random_safe")
    assert winner in (-1, 0, 1)
    assert state.replay_file.startswith("replays/") or \
           state.replay_file.split("/")[-2] == "replays"
