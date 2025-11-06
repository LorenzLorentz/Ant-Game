import os
import sys

# Ensure repo root is importable when running as a script
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from SDK import env


def run_random_steps(steps: int = 50):
    e = env(render_mode="ansi")
    e.reset()

    for t in range(steps):
        agent = e.agent_selection
        action = e.action_space(agent).sample()
        # encourage end turn sometimes so the game progresses
        if t % 4 == 0:
            action[0] = 0  # EndTurn
        e.step(action)
        if any(e.terminations.values()) or any(e.truncations.values()):
            break

    out = e.render()
    if out is not None:
        print(out)


if __name__ == "__main__":
    run_random_steps()
