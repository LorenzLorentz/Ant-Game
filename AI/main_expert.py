try:
    from AI_expert.ai import AI
    from main import main
except ModuleNotFoundError as exc:
    if exc.name not in {"AI_expert", "main"}:
        raise
    from AI.AI_expert.ai import AI
    from AI.main import main


if __name__ == "__main__":
    main(AI)
