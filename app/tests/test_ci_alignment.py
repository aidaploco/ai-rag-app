"""What ships must be what is tested, and the CI gate must stay honest.

These tie the Docker image, both packages' declared Python floors and the CI matrix together,
check that every workflow action stays pinned to a commit SHA, and that `ci-success` (the one
check meant to be required on `main`) waits for every other job. A Dependabot bump of the base
image, or a new CI job that nobody added to `needs:`, fails here on the PR itself instead of
going unnoticed.
"""

import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def _ci() -> dict:
    return yaml.safe_load((WORKFLOWS / "ci.yml").read_text())


def _matrix() -> list[str]:
    return [str(v) for v in _ci()["jobs"]["test"]["strategy"]["matrix"]["python-version"]]


def _floor(pyproject: Path) -> str:
    requires = tomllib.loads(pyproject.read_text())["project"]["requires-python"]
    match = re.search(r">=\s*(\d+\.\d+)", requires)
    assert match, f"unexpected requires-python in {pyproject}: {requires}"
    return match.group(1)


def test_the_image_python_is_in_the_ci_matrix():
    match = re.search(r"^FROM python:(\d+\.\d+)", (ROOT / "Dockerfile").read_text(), re.MULTILINE)
    assert match, "Dockerfile has no `FROM python:X.Y` line"
    assert match.group(1) in _matrix(), (
        f"the Docker image runs Python {match.group(1)} but CI only tests {_matrix()}: "
        "add it to the test matrix in .github/workflows/ci.yml"
    )


@pytest.mark.parametrize("package", ["rag-sdk", "app"])
def test_each_declared_python_floor_is_tested(package):
    floor = _floor(ROOT / package / "pyproject.toml")
    assert floor in _matrix(), f"{package} declares >={floor} but CI does not test it"


def test_ruff_targets_the_lowest_tested_python():
    target = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["ruff"]["target-version"]
    lowest = min(_matrix(), key=lambda v: tuple(int(p) for p in v.split(".")))
    assert target == "py" + lowest.replace(".", ""), (target, lowest)


def test_the_compose_and_ci_pgvector_images_are_the_same():
    # Dependabot's docker-compose ecosystem bumps the compose image; CI must test what compose runs.
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    ci_image = _ci()["jobs"]["test"]["services"]["postgres"]["image"]
    assert compose["services"]["postgres"]["image"] == ci_image, (
        "docker-compose.yml and the CI `services:` pgvector image differ: bump both together"
    )


def test_ci_success_waits_for_every_other_job():
    jobs = _ci()["jobs"]
    needs = set(jobs["ci-success"]["needs"])
    assert needs == set(jobs) - {"ci-success"}, (
        f"ci-success needs {sorted(needs)} but the other jobs are "
        f"{sorted(set(jobs) - {'ci-success'})}: a job it does not wait for is not gated"
    )


def test_ci_success_always_runs_so_a_skipped_job_cannot_pass_it():
    assert _ci()["jobs"]["ci-success"]["if"] == "always()"


@pytest.mark.parametrize("workflow", sorted(WORKFLOWS.glob("*.yml")), ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_commit_sha(workflow):
    uses = re.findall(r"^\s*(?:-\s*)?uses:\s*(\S+)", workflow.read_text(), re.MULTILINE)
    unpinned = [u for u in uses if not re.search(r"@[0-9a-f]{40}$", u)]
    assert not unpinned, (
        f"pin these to a full commit SHA (with the version in a comment): {unpinned}"
    )


@pytest.mark.parametrize("workflow", sorted(WORKFLOWS.glob("*.yml")), ids=lambda p: p.name)
def test_workflows_run_steps_with_pipefail(workflow):
    # GitHub's default shell is `bash -e` with no pipefail, so `cmd | tee file` takes tee's exit
    # status and a failing command shows green. `shell: bash` adds `-o pipefail`.
    defaults = yaml.safe_load(workflow.read_text()).get("defaults", {})
    assert defaults.get("run", {}).get("shell") == "bash", (
        "set top-level `defaults: run: shell: bash` so a failing pipeline fails the step"
    )


@pytest.mark.parametrize("workflow", sorted(WORKFLOWS.glob("*.yml")), ids=lambda p: p.name)
def test_workflows_default_to_read_only_permissions(workflow):
    permissions = yaml.safe_load(workflow.read_text()).get("permissions")
    assert permissions == {"contents": "read"}, (
        "set top-level `permissions: contents: read`; grant more per job, not for the whole file"
    )
