"""Immutable contracts for the supported v1 dataset and serving artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import Any

from .manifests import sha256_file
from .paths import manifests_dir, models_dir, project_root


PROCESSED_CONTRACT_SCHEMA = "wind_forecast.v1_processed_contract.v1"
SERVING_CONTRACT_SCHEMA = "wind_forecast.v1_serving_contract.v1"
V1_PROCESSED_CONTRACT_PATH = manifests_dir() / "v1_processed_contract.json"
V1_SERVING_CONTRACT_PATH = models_dir() / "v1_serving_contract.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class V1ContractError(ValueError):
    """Raised when an immutable v1 contract or artifact is invalid."""


def canonical_sha256(value: object) -> str:
    """Hash a JSON value using the repository's deterministic encoding."""
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def contract_sha256(path: str | Path) -> str:
    """Hash a JSON contract independently of checkout line endings.

    Git may materialize tracked JSON as CRLF on Windows. Contract references
    describe the logical JSON snapshot, so normalize line endings before
    hashing while keeping byte-exact hashes for datasets and model artifacts.
    """
    raw = Path(path).read_bytes()
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256(normalized).hexdigest()


def load_processed_contract(
    path: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
    verify_dataset: bool = False,
) -> dict[str, Any]:
    """Load and validate the processed v1 contract."""
    root = Path(repository_root or project_root()).resolve()
    contract_path = _resolve_path(
        path or root / "data" / "manifests" / "v1_processed_contract.json", root
    )
    payload = _load_json(contract_path, PROCESSED_CONTRACT_SCHEMA)
    _validate_processed_contract(payload, root, verify_dataset=verify_dataset)
    return payload


def load_serving_contract(
    path: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Load and validate the v1 inference-only serving contract."""
    root = Path(repository_root or project_root()).resolve()
    contract_path = _resolve_path(
        path or root / "models" / "v1_serving_contract.json", root
    )
    payload = _load_json(contract_path, SERVING_CONTRACT_SCHEMA)
    _validate_serving_contract(payload, root, verify_files=verify_files)
    return payload


def serving_artifacts(
    target_type: str,
    *,
    contract: Mapping[str, Any] | None = None,
    repository_root: str | Path | None = None,
    verify_files: bool = True,
) -> Mapping[str, Any]:
    """Return the immutable model/scaler record for a target type."""
    payload = contract or load_serving_contract(
        repository_root=repository_root, verify_files=verify_files
    )
    models = payload["targets"]
    if target_type not in models:
        raise V1ContractError(f"Serving contract has no target: {target_type!r}.")
    return models[target_type]


def _load_json(path: Path, expected_schema: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise V1ContractError(f"Could not read contract {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise V1ContractError(f"Contract {path} must be a JSON object.")
    if payload.get("schema_version") != expected_schema:
        raise V1ContractError(
            f"Contract {path} has unsupported schema_version "
            f"{payload.get('schema_version')!r}."
        )
    return payload


def _resolve_path(path: str | Path, root: Path) -> Path:
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise V1ContractError("Contract path must be a non-empty path.")
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise V1ContractError(f"Contract path escapes repository root: {path!r}.") from exc
    if not resolved.is_file():
        raise V1ContractError(f"Contract file is missing: {resolved}.")
    return resolved


def _relative_artifact_path(value: Any, root: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise V1ContractError(f"{field} must be a non-empty repository-relative path.")
    normalized = value.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(normalized)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
    ):
        raise V1ContractError(f"{field} must be a safe repository-relative path.")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise V1ContractError(f"{field} escapes repository root: {value!r}.") from exc
    return path


def _require_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise V1ContractError(f"{field} must be a lowercase SHA-256 digest.")
    return value


def _validate_processed_contract(
    payload: Mapping[str, Any], root: Path, *, verify_dataset: bool
) -> None:
    if payload.get("transformation_version") != "v1_preprocessing.v1":
        raise V1ContractError("Processed contract has an unsupported transformation version.")
    source_hash = _require_sha(payload.get("source_contract_sha256"), "source_contract_sha256")
    _require_sha(payload.get("dataset_sha256"), "dataset_sha256")
    dataset_path = _relative_artifact_path(payload.get("dataset_path"), root, "dataset_path")
    row_count = payload.get("row_count")
    column_count = payload.get("column_count")
    if not isinstance(row_count, int) or row_count < 1:
        raise V1ContractError("Processed contract row_count must be a positive integer.")
    if not isinstance(column_count, int) or column_count < 1:
        raise V1ContractError("Processed contract column_count must be a positive integer.")
    columns = payload.get("columns")
    if (
        not isinstance(columns, list)
        or len(columns) != column_count
        or not columns
        or any(not isinstance(column, str) or not column for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise V1ContractError("Processed contract columns must be a unique ordered list.")
    if columns[:2] != ["Date", "Wind_Production"]:
        raise V1ContractError("Processed contract must start with Date and Wind_Production.")
    coverage = payload.get("coverage")
    if not isinstance(coverage, Mapping) or not all(
        isinstance(coverage.get(key), str) and coverage[key].strip()
        for key in ("start", "end")
    ):
        raise V1ContractError("Processed contract coverage must contain start and end.")
    if canonical_sha256(columns[2:]) != _require_sha(
        payload.get("feature_schema_sha256"), "feature_schema_sha256"
    ):
        raise V1ContractError("Processed contract feature schema hash does not match columns.")
    source_path = _relative_artifact_path(
        payload.get("source_contract_path"), root, "source_contract_path"
    )
    if contract_sha256(source_path) != source_hash:
        raise V1ContractError("Processed contract source contract hash does not match its file.")
    if verify_dataset:
        if not dataset_path.is_file():
            raise V1ContractError(f"Processed dataset is missing: {dataset_path}.")
        if sha256_file(dataset_path) != payload["dataset_sha256"]:
            raise V1ContractError("Processed dataset hash does not match its contract.")


def _validate_serving_contract(
    payload: Mapping[str, Any], root: Path, *, verify_files: bool
) -> None:
    if payload.get("status") != "legacy_inference_only":
        raise V1ContractError("v1 serving contract must be legacy_inference_only.")
    dataset_contract = payload.get("processed_contract")
    if not isinstance(dataset_contract, Mapping):
        raise V1ContractError("Serving contract must declare processed_contract.")
    processed_path = _relative_artifact_path(dataset_contract.get("path"), root, "processed_contract.path")
    processed_hash = _require_sha(dataset_contract.get("sha256"), "processed_contract.sha256")
    if contract_sha256(processed_path) != processed_hash:
        raise V1ContractError("Serving contract processed contract hash does not match its file.")
    processed = load_processed_contract(processed_path, repository_root=root, verify_dataset=False)
    if processed["dataset_sha256"] != payload.get("dataset_sha256"):
        raise V1ContractError("Serving contract dataset hash does not match processed contract.")
    feature_contract = payload.get("feature_contract")
    if not isinstance(feature_contract, Mapping):
        raise V1ContractError("Serving contract must declare feature_contract.")
    if feature_contract.get("columns_source") != "processed_contract.columns[2:]":
        raise V1ContractError("Serving contract must source feature order from processed contract.")
    features = processed["columns"][2:]
    if canonical_sha256(features) != _require_sha(
        feature_contract.get("sha256"), "feature_contract.sha256"
    ):
        raise V1ContractError("Serving contract feature hash does not match its columns.")
    targets = payload.get("targets")
    if not isinstance(targets, Mapping) or set(targets) != {"original", "log"}:
        raise V1ContractError("Serving contract must declare original and log targets.")
    for target_type, record in targets.items():
        _validate_target_record(target_type, record, root, verify_files=verify_files)


def _validate_target_record(
    target_type: str, record: Any, root: Path, *, verify_files: bool
) -> None:
    if not isinstance(record, Mapping) or record.get("model_name") != "ANN_Tuned":
        raise V1ContractError(f"Invalid v1 serving record for target {target_type!r}.")
    expected_transform = "log1p" if target_type == "log" else "identity"
    if record.get("target_transform") != expected_transform:
        raise V1ContractError(f"Invalid target transform for {target_type!r}.")
    for key in ("model", "scaler_x", "scaler_y"):
        item = record.get(key)
        if not isinstance(item, Mapping):
            raise V1ContractError(f"Serving record {target_type}.{key} is missing.")
        path = _relative_artifact_path(item.get("path"), root, f"{target_type}.{key}.path")
        digest = _require_sha(item.get("sha256"), f"{target_type}.{key}.sha256")
        if verify_files:
            if not path.is_file():
                raise V1ContractError(f"Serving artifact is missing: {path}.")
            if sha256_file(path) != digest:
                raise V1ContractError(f"Serving artifact hash mismatch: {path}.")


__all__ = [
    "PROCESSED_CONTRACT_SCHEMA",
    "SERVING_CONTRACT_SCHEMA",
    "V1ContractError",
    "V1_PROCESSED_CONTRACT_PATH",
    "V1_SERVING_CONTRACT_PATH",
    "canonical_sha256",
    "contract_sha256",
    "load_processed_contract",
    "load_serving_contract",
    "serving_artifacts",
]
