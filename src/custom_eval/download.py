import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

HFD_ENV = "PATH_TO_CUSTOM_HFD"
"""Environment variable holding the absolute path of the custom `hfd.sh`."""


@dataclass(frozen=True)
class DataSpec:
    """Dataset files a benchmark reads, in the Hub layout under the data dir.

    Attributes:
        repo_id: Hugging Face dataset repo, e.g. `HuggingFaceH4/MATH-500`.
        pattern: Glob (relative to the repo folder) the benchmark reads.
        include: `--include` patterns passed to `hfd.sh`, to fetch only part of
            a large repo; empty fetches the whole repo.
    """

    repo_id: str
    pattern: str
    include: tuple[str, ...] = ()

    def files(self, data_dir: str | Path) -> list[Path]:
        """Return the matching files under `data_dir`, sorted."""
        return sorted(
            p for p in (Path(data_dir) / self.repo_id).glob(self.pattern) if p.is_file()
        )


class DataMissingError(RuntimeError):
    """Raised when dataset files are missing and cannot be downloaded."""


def ensure_data(specs: list[DataSpec], data_dir: str | Path) -> None:
    """Make sure every spec's files exist, downloading missing ones with `hfd.sh`.

    Missing repos are fetched with `bash $PATH_TO_CUSTOM_HFD <repo_id>
    --dataset --local-dir <data_dir>/<repo_id>` (plus `--include ...` for
    partial repos).

    Args:
        specs: Dataset files needed by the selected benchmarks.
        data_dir: Root data directory.

    Raises:
        DataMissingError: If files are missing and `PATH_TO_CUSTOM_HFD` is not
            set, or still missing after the download.
    """
    missing = [spec for spec in specs if not spec.files(data_dir)]
    if not missing:
        return

    if HFD_ENV not in os.environ or os.environ[HFD_ENV] == "":
        lines = [
            f"  {s.repo_id}: no files matching {s.pattern!r} under "
            f"{Path(data_dir) / s.repo_id}"
            for s in missing
        ]
        raise DataMissingError(
            "Dataset files are missing and "
            f"{HFD_ENV} is not set, so they cannot be downloaded:\n" + "\n".join(lines)
        )
    hfd = os.environ[HFD_ENV]

    # one download per repo, with the union of the include patterns
    repos: dict[str, list[DataSpec]] = {}
    for spec in missing:
        repos.setdefault(spec.repo_id, []).append(spec)
    for repo_id, repo_specs in repos.items():
        dest = Path(data_dir) / repo_id
        cmd = ["bash", hfd, repo_id, "--dataset", "--local-dir", str(dest)]
        include = [p for s in repo_specs for p in s.include]
        if include and all(s.include for s in repo_specs):
            cmd += ["--include", *include]
        print(f"downloading {repo_id} -> {dest}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise DataMissingError(
                f"Download of {repo_id} failed (exit {result.returncode}): {cmd}"
            )

    still_missing = [spec for spec in missing if not spec.files(data_dir)]
    if still_missing:
        raise DataMissingError(
            "Files still missing after download: "
            + ", ".join(f"{s.repo_id}/{s.pattern}" for s in still_missing)
        )
