from custom_eval.answer import extract_answer
from custom_eval.config import JudgeConfig
from custom_eval.model import connect_model, resolve_model_id
from inspect_ai.model import ChatMessageUser, Model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState


def judge_model(judge: JudgeConfig, max_generation_workers: int | None) -> Model:
    """Connect to the judge's vLLM server (same access rules as the evaluated model).

    Args:
        judge: Judge config.
        max_generation_workers: Maximum concurrent judge requests, or None for
            Inspect's default.

    Returns:
        The judge model.
    """
    model_id = resolve_model_id(
        judge["url"], judge["model"] if "model" in judge else None
    )
    sampling = judge["sampling"] if "sampling" in judge else {}
    return connect_model(judge["url"], model_id, sampling, max_generation_workers)


def judge_prompt(template: str, question: str, response: str, target: str) -> str:
    """Fill the judge prompt: replace `{question}`, `{response}`, `{target}` verbatim.

    Plain replacement (not `str.format`), so other braces in the template or in
    the substituted texts are left untouched.
    """
    return (
        template.replace("{question}", question)
        .replace("{response}", response)
        .replace("{target}", target)
    )


def judge_scorer(
    judge: JudgeConfig,
    model: Model,
    answer_open: str,
    answer_close: str,
    use_answer: bool,
) -> Scorer:
    """Build an LLM-judge scorer bound to the run's answer markers.

    Args:
        judge: Judge config.
        model: Judge model (see `judge_model`).
        answer_open: Run's opening answer marker.
        answer_close: Run's closing answer marker.
        use_answer: Judge the text between the run's answer markers (True;
            no markers scores 0 without calling the judge) or the whole final
            output (False).

    Returns:
        The scorer (`Score.value` is the verdict's score).
    """
    verdict_open, verdict_close = judge["answer"]

    @scorer(metrics=[mean(), stderr()], name="llm_judge")
    def _judge() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            completion = state.output.completion
            answer = extract_answer(completion, answer_open, answer_close)
            if use_answer and answer is None:
                return Score(
                    value=0.0, metadata={"marker_found": False, "verdict": None}
                )
            text = answer if use_answer and answer is not None else completion
            prompt = judge_prompt(judge["prompt"], state.input_text, text, target.text)
            output = await model.generate([ChatMessageUser(content=prompt)])
            verdict = extract_answer(output.completion, verdict_open, verdict_close)
            verdict = verdict.strip() if verdict is not None else None
            value = (
                float(judge["scores"][verdict])
                if verdict is not None and verdict in judge["scores"]
                else 0.0
            )
            return Score(
                value=value,
                answer=text,
                explanation=output.completion,
                metadata={
                    "marker_found": answer is not None,
                    "verdict": verdict,
                    "judge_prompt": prompt,
                },
            )

        return score

    return _judge()
