from __future__ import annotations

try:
    from ai import AI as PackagedAI
except ModuleNotFoundError as exc:  # pragma: no cover - repository layout
    if exc.name != "ai":
        raise
    PackagedAI = None

try:
    from common import BaseAgent, MatchSession
except ModuleNotFoundError as exc:  # pragma: no cover - repository layout
    if exc.name != "common":
        raise
    from AI.common import BaseAgent, MatchSession


def build_session(agent) -> MatchSession:
    factory = getattr(agent, "create_session", None)
    if callable(factory):
        session = factory()
        if not isinstance(session, MatchSession):
            raise TypeError("AI.create_session() must return a MatchSession")
        return session

    if isinstance(agent, BaseAgent):
        try:
            from protocol import ProtocolSession
        except ModuleNotFoundError as exc:  # pragma: no cover - repository layout
            if exc.name != "protocol":
                raise
            from AI.protocol import ProtocolSession
        return ProtocolSession(agent)

    raise TypeError("AI must inherit BaseAgent or expose create_session()")


def run_session(session: MatchSession) -> None:
    while True:
        if session.player == 0:
            session.perform_self_turn()
            if not session.receive_opponent_turn():
                break
            if not session.sync_round():
                break
        else:
            if not session.receive_opponent_turn():
                break
            session.perform_self_turn()
            if not session.sync_round():
                break


def main(ai_cls=None) -> None:
    agent_cls = ai_cls or PackagedAI
    if agent_cls is None:
        raise RuntimeError("main.py expects ai.py to export class AI, or an explicit ai_cls argument")
    run_session(build_session(agent_cls()))


if __name__ == "__main__":  # pragma: no cover - exercised in packaged layout
    main()
