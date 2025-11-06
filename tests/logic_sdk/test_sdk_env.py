from SDK import env


def test_sdk_env_basic_step():
    e = env(render_mode=None)
    e.reset()
    # Take a few steps forcing EndTurn to roll the round
    for _ in range(4):
        a = e.action_space(e.agent_selection).sample()
        a[0] = 0  # EndTurn
        e.step(a)
        if any(e.terminations.values()) or any(e.truncations.values()):
            break
    # Ensure render returns a string in ansi mode
    e.render_mode = "ansi"
    assert isinstance(e.render(), str)

