from typing import Any

import httpx
import httpx2
from openai import DefaultAsyncHttpxClient
from typing_extensions import override

from inspect_ai._util.http_defaults_httpx2 import (
    connect_timeout,
    default_client_kwargs,
    default_limits,
)
from inspect_ai.model import GenerateConfig, Model, get_model, modelapi
from inspect_ai.model._model import RetryDecision
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model._providers.vllm import VLLMAPI, _is_fatal_vllm_error

PROVIDER = "custom-eval-vllm"
"""Inspect model provider name registered by this module."""

REQUEST_TIMEOUT_S = 6 * 60 * 60
"""Per-request timeout. Long reasoning generations under load can take hours."""

DISCOVERY_TIMEOUT_S = 60.0
"""Timeout for the `/v1/models` lookup."""

GENERATE_CONFIG_KEYS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "max_tokens": "max_tokens",
    "seed": "seed",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "stop": "stop_seqs",
}
"""Sampling keys mapped onto Inspect's `GenerateConfig`; all others go in the request body."""


class ModelLookupError(RuntimeError):
    """Raised when the served model cannot be resolved."""


async def _drop_authorization(request: httpx2.Request) -> None:
    # The servers reject any non-empty API key, so no request carries one.
    if "Authorization" in request.headers:
        del request.headers["Authorization"]


def pool_limits(max_generation_workers: int | None) -> httpx2.Limits:
    """HTTP connection pool limits for a model client.

    Inspect's limits (a pool of 1,000 connections unless
    `INSPECT_HTTP_POOL_CONNECTIONS` sets another size), with the pool raised to
    `max_generation_workers` connections when that is larger. So the pool never
    caps in-flight requests below `max_generation_workers`; it is never lowered
    either.

    Args:
        max_generation_workers: Maximum concurrent generation requests, or None
            to keep Inspect's limits.

    Returns:
        The limits.
    """
    limits = default_limits()
    if (
        max_generation_workers is None
        or limits.max_connections is None
        or limits.max_connections >= max_generation_workers
    ):
        return limits
    return httpx2.Limits(
        max_connections=max_generation_workers,
        max_keepalive_connections=max_generation_workers,
        keepalive_expiry=limits.keepalive_expiry,
    )


class NoAuthVLLMAPI(VLLMAPI):
    """vLLM provider for an existing server: no API key, no TLS verification.

    Every request (generation, `/v1/models`, `/tokenize`) goes out without an
    `Authorization` header and without certificate verification. A server that
    cannot be reached is retried like any other transient error, also on
    localhost (Inspect's vLLM provider gives up at once on an unreachable
    localhost server). The connection pool holds at least `max_connections`
    (the run's `max_generation_workers`) connections, see `pool_limits`.
    """

    @override
    def should_retry(self, ex: BaseException) -> bool | RetryDecision:
        # Inspect's vLLM rules minus the unreachable-localhost fail-fast (the
        # server-process check only applies to servers Inspect launched)
        if _is_fatal_vllm_error(ex):
            return RetryDecision.no()
        return OpenAICompatibleAPI.should_retry(self, ex)

    @override
    def _create_http_client(self) -> DefaultAsyncHttpxClient:
        return DefaultAsyncHttpxClient(
            **default_client_kwargs(
                verify=False,
                limits=pool_limits(self._init_config.max_connections),
                timeout=httpx2.Timeout(
                    timeout=REQUEST_TIMEOUT_S,
                    connect=max(REQUEST_TIMEOUT_S, connect_timeout()),
                ),
                event_hooks={"request": [_drop_authorization]},
            )
        )


@modelapi(name=PROVIDER)
def _no_auth_vllm() -> type[NoAuthVLLMAPI]:
    return NoAuthVLLMAPI


def api_base(url: str) -> str:
    """Normalize a server URL to its OpenAI API base (ending in `/v1`).

    Args:
        url: Server URL, with or without a trailing `/v1`.

    Returns:
        The URL ending in `/v1`, without a trailing slash.
    """
    base = url.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def served_models(url: str) -> list[str]:
    """List the model ids served at `url` (`GET /v1/models`).

    Args:
        url: Server URL, with or without a trailing `/v1`.

    Returns:
        Served model ids, in server order.
    """
    response = httpx.get(
        f"{api_base(url)}/models", verify=False, timeout=DISCOVERY_TIMEOUT_S
    )
    response.raise_for_status()
    return [entry["id"] for entry in response.json()["data"]]


def resolve_model_id(url: str, model: str | None) -> str:
    """Resolve which served model to evaluate.

    Args:
        url: Server URL.
        model: Requested model id, or None to use the only served model.

    Returns:
        The model id.

    Raises:
        ModelLookupError: If `model` is not served, or `model` is None and the
            server does not serve exactly one model.
    """
    try:
        ids = served_models(url)
    except Exception as ex:
        raise ModelLookupError(
            f"Could not list models at {api_base(url)}/models: {ex}"
        ) from ex
    if model is not None:
        if model not in ids:
            raise ModelLookupError(
                f"Model {model!r} is not served at {url}. Served: {ids}"
            )
        return model
    if len(ids) != 1:
        raise ModelLookupError(
            f"{url} serves {len(ids)} models {ids}; set 'model' in the config "
            "(or pass --model) to pick one."
        )
    return ids[0]


def generate_config(
    sampling: dict[str, Any], max_generation_workers: int | None
) -> GenerateConfig:
    """Build the generation config from the config's sampling section.

    Keys Inspect knows are set on `GenerateConfig`; every other key (e.g.
    `top_k`, `min_p`, `repetition_penalty`, `chat_template_kwargs`) is sent
    verbatim in the request body. Keys not given are not sent.

    Args:
        sampling: The `sampling` section (may be empty).
        max_generation_workers: Maximum concurrent generation requests
            (Inspect's `max_connections`), or None for Inspect's default.

    Returns:
        The generation config.
    """
    fields: dict[str, Any] = {}
    extra_body: dict[str, Any] = {}
    for key, value in sampling.items():
        if key in GENERATE_CONFIG_KEYS:
            fields[GENERATE_CONFIG_KEYS[key]] = value
        else:
            extra_body[key] = value
    if "stop_seqs" in fields and isinstance(fields["stop_seqs"], str):
        fields["stop_seqs"] = [fields["stop_seqs"]]
    if extra_body:
        fields["extra_body"] = extra_body
    if max_generation_workers is not None:
        fields["max_connections"] = max_generation_workers
    return GenerateConfig(**fields)


def connect_model(
    url: str,
    model_id: str,
    sampling: dict[str, Any],
    max_generation_workers: int | None,
) -> Model:
    """Create the Inspect model for a served vLLM model.

    Args:
        url: Server URL.
        model_id: Served model id (see `resolve_model_id`).
        sampling: Sampling parameters (absent keys are not sent).
        max_generation_workers: Maximum concurrent generation requests, or None
            for Inspect's default.

    Returns:
        The model.
    """
    return get_model(
        f"{PROVIDER}/{model_id}",
        config=generate_config(sampling, max_generation_workers),
        base_url=api_base(url),
        client_timeout=REQUEST_TIMEOUT_S,
        memoize=False,
    )
