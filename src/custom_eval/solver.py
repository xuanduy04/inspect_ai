from inspect_ai.model import ChatMessageSystem
from inspect_ai.solver import Generate, Solver, TaskState, solver


@solver
def system_prompt_generate(system_prompt: str | None, seed: int | None) -> Solver:
    """Insert the run's system prompt verbatim, then generate once.

    The system prompt is never formatted or read as a file (unlike Inspect's
    `system_message()`). Any system messages the sample already has are
    replaced, so every sample carries exactly the run's system prompt.

    Args:
        system_prompt: System prompt text, or None to send no system message.
        seed: Base request seed, or None to send no seed. Repeat `k` (1-based)
            uses `seed + k - 1` so repeats are distinct but reproducible.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages = [m for m in state.messages if m.role != "system"]
        if system_prompt is not None:
            state.messages.insert(0, ChatMessageSystem(content=system_prompt))
        if seed is None:
            return await generate(state, tool_calls="none")
        return await generate(state, tool_calls="none", seed=seed + state.epoch - 1)

    return solve
