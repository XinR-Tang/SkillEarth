"""Generic controller bridge that forwards direct JSON tool observations."""
from __future__ import annotations

import uuid
import argparse
import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# FastMCP's client default is 60 seconds.  That is too short for GPU-backed
# perception tools such as RemoteSAM, and used to cut those calls off before
# the outer SkillEarth timeout was reached.
MCP_CALL_TIMEOUT_S = int(os.getenv("SKILLEARTH_MCP_CALL_TIMEOUT_S", "600"))


class ModelRequest(BaseModel):
    model: str


def _json_safe(value: Any) -> Any:
    """Convert FastMCP result objects without discarding structured content."""
    for attribute in ("data", "structured_content"):
        candidate = getattr(value, attribute, None)
        if candidate is not None:
            return _json_safe(candidate)
    content = getattr(value, "content", None)
    if isinstance(content, (list, tuple)):
        parts: list[Any] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                try:
                    parts.append(json.loads(text))
                except json.JSONDecodeError:
                    parts.append(text)
            else:
                parts.append(_json_safe(block))
        return parts[0] if len(parts) == 1 else parts
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def _compact_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _semantic_evidence(value: Any, *, max_items: int = 8, max_chars: int = 2000) -> Any:
    """Return bounded real tool evidence for the semantic judge.

    This is used only for the semantic judge; action results remain complete.
    """
    if isinstance(value, list):
        preview = [
            _semantic_evidence(item, max_items=max_items, max_chars=max_chars)
            for item in value[:max_items]
        ]
        return preview if len(value) <= max_items else {
            "summary": {"kind": "list", "count": len(value), "preview": preview}
        }
    if isinstance(value, dict):
        items = list(value.items())
        preview = {
            str(key): _semantic_evidence(item, max_items=max_items, max_chars=max_chars)
            for key, item in items[:max_items]
        }
        return preview if len(items) <= max_items else {
            "summary": {
                "kind": "object",
                "keys": [str(key) for key, _ in items[:max_items]],
                "preview": preview,
                "omitted": len(items) - max_items,
            }
        }
    if isinstance(value, str) and len(value) > max_chars:
        return {
            "summary": {
                "kind": "string",
                "chars": len(value),
                "preview": value[:max_chars],
            }
        }
    return value


_NUMBER_TEXT = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _coerce_numeric_strings(value: Any) -> Any:
    """Convert JSON-number text emitted by benchmark trajectories to numbers.

    FastMCP may expose untyped scalar parameters (notably arithmetic tools).
    Sending ``\"1.25\"`` unchanged then invokes Python string operations rather
    than numeric operations.  Restrict conversion to complete numeric literals,
    so paths and ordinary text remain untouched.
    """
    if isinstance(value, dict):
        return {key: _coerce_numeric_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_numeric_strings(item) for item in value]
    if isinstance(value, str) and _NUMBER_TEXT.fullmatch(value.strip()):
        normalized = value.strip()
        try:
            return int(normalized) if normalized.isdigit() else float(normalized)
        except ValueError:
            return value
    return value


def _schema_allows_array(schema: Any) -> bool:
    """Return whether a JSON schema accepts an array, including union schemas."""
    if not isinstance(schema, dict):
        return False
    if schema.get("type") == "array":
        return True
    for key in ("anyOf", "oneOf", "allOf"):
        if any(_schema_allows_array(item) for item in schema.get(key, [])):
            return True
    return False


_OUTPUT_NAME = re.compile(
    r"^(?:(?:output|save|destination|dest)(?:_.*)?|"
    r"target_(?:path|file|name))$",
    re.IGNORECASE,
)
def _is_output_parameter(name: str, schema: Any) -> bool:
    """Recognise only conventional output-path parameters, never tool names."""
    normalized = name.lower().replace("-", "_")
    if not _OUTPUT_NAME.fullmatch(normalized):
        return False
    return True


@dataclass(frozen=True)
class ToolContract:
    server: str
    input_schema: dict[str, Any]

    @property
    def properties(self) -> dict[str, Any]:
        properties = self.input_schema.get("properties", {})
        return properties if isinstance(properties, dict) else {}

    def accepts_array(self, parameter: str) -> bool:
        return _schema_allows_array(self.properties.get(parameter))

    def output_parameters(self) -> tuple[str, ...]:
        return tuple(
            key for key, schema in self.properties.items()
            if _is_output_parameter(key, schema)
        )


class MCPBridge:
    """Controller-protocol facade over live Earth-Agent FastMCP services."""

    def __init__(self, servers: list[str], *, work_dir: str | Path):
        self.servers = servers
        self.work_dir = Path(work_dir).expanduser().resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.contracts: dict[str, ToolContract] = {}
        self.last_listed_directory: Path | None = None
        # Logical benchmark names and basenames point to the real runtime
        # output.  Consumers should follow producer lineage, not reconstruct
        # paths under benchmark/out by string concatenation.
        self.produced_paths: dict[str, str] = {}

    async def discover(self) -> None:
        from fastmcp import Client

        for server in self.servers:
            async with Client(server) as client:
                for tool in await client.list_tools():
                    name = getattr(tool, "name", None)
                    if not isinstance(name, str) or not name:
                        continue
                    if name in self.contracts:
                        raise RuntimeError(f"duplicate_mcp_tool:{name}")
                    schema = (
                        getattr(tool, "inputSchema", None)
                        or getattr(tool, "input_schema", None)
                        or {}
                    )
                    if hasattr(schema, "model_dump"):
                        schema = schema.model_dump()
                    if not isinstance(schema, dict):
                        schema = {}
                    self.contracts[name] = ToolContract(server, schema)

        # Earth-Bench trajectories use scalar NBR/NDWI action names, while
        # the current official MCP services expose only their batch variants.
        # Advertise compatible scalar aliases and translate them at invocation
        # time rather than treating a catalog/service-version mismatch as a
        # missing worker.
        for legacy_name, batch_name in {
            "calculate_nbr": "calculate_batch_nbr",
            "calculate_ndwi": "calculate_batch_ndwi",
            "calculate_ndvi": "calculate_batch_ndvi",
            "calculate_ndti": "calculate_batch_ndti",
            # Earth-Bench calls this historical name, while the current
            # Perception service publishes the same skeleton-contour
            # implementation as ``count_connected_components(image_path)``.
            "count_skeleton_contours": "count_connected_components",
        }.items():
            if batch_name in self.contracts:
                self.contracts[legacy_name] = self.contracts[batch_name]

    def _supply_missing_output_paths(
        self,
        tool_name: str,
        contract: ToolContract,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        output_keys = contract.output_parameters()
        if not output_keys:
            return arguments

        output_dir = self.work_dir
        source_dir = self.last_listed_directory
        if (
            source_dir is not None
            and re.fullmatch(r"question\d+", source_dir.name, re.I)
        ):
            output_dir = self.work_dir / source_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        suffix = (
            ".png"
            if tool_name in {"SAM2", "ChangeOS", "grayscale_to_colormap"}
            else ".tif"
        )

        completed = dict(arguments)

        for key in output_keys:
            if completed.get(key) not in (None, "", []):
                continue

            filename = f"{tool_name}_{uuid.uuid4().hex}{suffix}"
            completed[key] = str(output_dir / filename)

        return completed

    def _model_visible_value(self, value: Any) -> Any:
        """Hide bridge-internal runtime directories from model-visible outputs."""
        if isinstance(value, list):
            return [self._model_visible_value(item) for item in value]
        if isinstance(value, dict):
            return {
                key: self._model_visible_value(item)
                for key, item in value.items()
            }
        if not isinstance(value, str):
            return value

        runtime_root = str(self.work_dir.resolve()) + "/"
        if value.startswith(runtime_root):
            return Path(value).name

        # Covers status strings such as: "Result saved at /.../runtime/foo.tif".
        return re.sub(
            re.escape(runtime_root) + r"[^\s,;\]\}\"']+",
            lambda match: Path(match.group(0)).name,
            value,
        )

    @staticmethod
    def _translate_compatibility_call(
        name: str, arguments: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Adapt benchmark-era action contracts to current MCP contracts."""
        scalar_to_batch = {
            "calculate_nbr": (
                "calculate_batch_nbr",
                ("input_nir_path", "input_swir_path", "output_path"),
                ("input_nir_paths", "input_swir_paths", "output_paths"),
            ),
            "calculate_ndwi": (
                "calculate_batch_ndwi",
                ("input_nir_path", "input_swir_path", "output_path"),
                ("input_nir_paths", "input_swir_paths", "output_paths"),
            ),
            "calculate_ndvi": (
                "calculate_batch_ndvi",
                ("input_nir_path", "input_red_path", "output_path"),
                ("input_nir_paths", "input_red_paths", "output_paths"),
            ),
            "calculate_ndti": (
                "calculate_batch_ndti",
                ("input_red_path", "input_green_path", "output_path"),
                ("input_red_paths", "input_green_paths", "output_paths"),
            ),
        }
        if name == "calculate_ndti" and "input_pairs" in arguments:
            pairs = arguments["input_pairs"]
            if not isinstance(pairs, list) or not pairs:
                raise ValueError("invalid_legacy_contract:calculate_ndti.input_pairs")
            reds: list[Any] = []
            greens: list[Any] = []
            outputs: list[Any] = []
            for pair in pairs:
                if isinstance(pair, dict):
                    red = pair.get("input_red_path", pair.get("red_path", pair.get("red")))
                    green = pair.get("input_green_path", pair.get("green_path", pair.get("green")))
                    output = pair.get("output_path", pair.get("output"))
                elif isinstance(pair, (list, tuple)) and len(pair) == 3:
                    red, green, output = pair
                else:
                    raise ValueError("invalid_legacy_contract:calculate_ndti.input_pairs")
                if red is None or green is None or output is None:
                    raise ValueError("invalid_legacy_contract:calculate_ndti.input_pairs")
                reds.append(red)
                greens.append(green)
                outputs.append(output)
            return "calculate_batch_ndti", {
                "input_red_paths": reds,
                "input_green_paths": greens,
                "output_paths": outputs,
            }

        if name in scalar_to_batch:
            batch_name, scalar_keys, batch_keys = scalar_to_batch[name]
            missing = [key for key in scalar_keys if key not in arguments]
            if missing:
                raise ValueError(f"missing_compatibility_arguments:{missing}")
            return batch_name, {
                batch_key: [arguments[scalar_key]]
                for scalar_key, batch_key in zip(scalar_keys, batch_keys)
            }

        if name == "count_skeleton_contours":
            # Fixtures use both ``image_path`` and the older
            # ``input_image_path`` spelling.  Normalize at the bridge edge;
            # the live service contract is ``count_connected_components``.
            image_path = arguments.get("image_path", arguments.get("input_image_path"))
            if image_path is None or image_path == "":
                raise ValueError("missing_compatibility_arguments:['image_path']")
            return "count_connected_components", {"image_path": image_path}

        # The plural ``bboxes`` form is handled in ``invoke`` once the live
        # contract is available: modern SAM2 accepts ``bboxes`` natively,
        # while the singular fallback is applied only for older services.
        return name, arguments

    def _source_directory(self, arguments: dict[str, Any]) -> Path | None:
        """Resolve the directory passed to a listing call without exposing it."""
        for key, value in arguments.items():
            normalized = key.lower().replace("-", "_")
            if normalized in {"dir", "dir_path", "directory", "directory_path", "input_dir", "folder"}:
                if not isinstance(value, str):
                    continue
                path = Path(value)
                if path.is_absolute():
                    return path
                # Earth-Agent resolves benchmark/data/... below its project
                # root.  Map that public relative contract to our local
                # Earth-Bench checkout only at invocation time.
                candidate = self.work_dir.parent.parent / "Earth-Bench" / path
                if candidate.is_dir():
                    return candidate
        return None

    def _resolve_listing_directory(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Send an absolute directory to the isolated MCP worker process."""
        source_dir = self._source_directory(arguments)
        if source_dir is None:
            return arguments
        return {
            key: str(source_dir) if key.lower().replace("-", "_")
            in {"dir", "dir_path", "directory", "directory_path", "input_dir", "folder"}
            else value
            for key, value in arguments.items()
        }

    @staticmethod
    def _path_values(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [path for item in value for path in MCPBridge._path_values(item)]
        return []

    @staticmethod
    def _paired_result_preview(
        executed: dict[str, Any], raw_result: list
    ) -> list[dict] | None:
        """Pair each scalar result with its aligned input file basename.

        A batch measurement is otherwise a bare numeric list, decoupled from
        the scenes it describes, which makes downstream per-file reasoning
        (grouping by calendar day, counting exceedance days) impossible.
        Keep the stored payload unchanged and expose this pairing only in the
        prompt-visible channel.
        """
        if not raw_result or not all(
            item is None or isinstance(item, (int, float, bool))
            for item in raw_result
        ):
            return None
        candidates = [
            key
            for key, value in executed.items()
            if isinstance(value, list)
            and value
            and all(isinstance(item, str) for item in value)
            and len(value) == len(raw_result)
        ]
        if not candidates:
            return None
        preferred = ("file_list", "image_paths", "image_path", "paths", "path")
        source_key = next(
            (name for name in preferred if name in candidates), candidates[0]
        )
        paths = executed[source_key]
        rows = [
            {"i": i, "file": Path(paths[i]).name, "value": raw_result[i]}
            for i in range(len(raw_result))
        ]
        # Bound prompt growth for very large batches while keeping both ends.
        cap = 120
        if len(rows) <= cap:
            return rows
        head, tail = 90, 30
        return (
            rows[:head]
            + [{"note": f"omitted {len(rows) - head - tail} middle rows"}]
            + rows[-tail:]
        )

    @staticmethod
    def _looks_like_file(value: str) -> bool:
        """Avoid treating labels and ordinary prose as relative file paths."""
        suffix = Path(value).suffix.lower()
        return suffix in {
            ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2", ".nc",
            ".npy", ".npz", ".csv", ".json", ".geojson", ".shp", ".zip",
        }

    def _normalize_observation_paths(
        self,
        value: Any,
        *,
        arguments: dict[str, Any],
        executed: dict[str, Any],
        contract: ToolContract,
    ) -> Any:
        """Return the same observation, with only file locations canonicalized.

        Directory-listing services return bare filenames.  Output-producing
        services may echo relative result paths.  Both forms are made absolute
        from explicit call context; numbers and unrelated text are unchanged.
        """
        source_dir = self._source_directory(arguments)
        output_map: dict[str, str] = {}
        for key in contract.output_parameters():
            for raw, absolute in zip(
                self._path_values(arguments.get(key)), self._path_values(executed.get(key))
            ):
                output_map[raw] = absolute
                if not Path(raw).is_absolute():
                    output_map[f"benchmark/out/{raw}"] = absolute
                    output_map[f"out/{raw}"] = absolute

        def normalize_string(item: str) -> str:
            if item in output_map:
                return output_map[item]
            path = Path(item)
            if path.is_absolute():
                return str(path)
            parts = path.parts
            if len(parts) >= 2 and parts[:2] == ("benchmark", "out"):
                return str(self.work_dir / Path(*parts[2:]))
            if parts and parts[0] == "out":
                return str(self.work_dir / Path(*parts[1:]))
            if source_dir is not None and self._looks_like_file(item):
                return str(source_dir / path)
            return item

        if isinstance(value, str):
            return normalize_string(value)
        if isinstance(value, list):
            return [self._normalize_observation_paths(
                item, arguments=arguments, executed=executed, contract=contract
            ) for item in value]
        if isinstance(value, dict):
            return {key: self._normalize_observation_paths(
                item, arguments=arguments, executed=executed, contract=contract
            ) for key, item in value.items()}
        return value

    def _resolve_listed_input_paths(
        self, contract: ToolContract, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve bare input filenames against the last listed directory.

        Directory listings remain concise for the executor, while the bridge
        deterministically restores absolute paths only at tool-invocation time.
        Output parameters are excluded so their existing runtime redirection is
        preserved.
        """
        source_dir = self.last_listed_directory
        if source_dir is None:
            return arguments

        output_keys = set(contract.output_parameters())

        def resolve(value: Any) -> Any:
            if isinstance(value, list):
                return [resolve(item) for item in value]
            if not isinstance(value, str) or not self._looks_like_file(value):
                return value
            path = Path(value)
            if path.is_absolute():
                return value
            # Reference fixtures use several logical spellings.  Never append
            # a full benchmark/data/... path to the already sample-scoped
            # listing directory, which produced .../Earth-Bench/benchmark/data
            # /questionN/... in live calls.
            parts = path.parts
            candidates: list[Path] = []
            if len(parts) >= 3 and parts[:2] == ("benchmark", "data"):
                relative = Path(*parts[2:])
                candidates.extend((source_dir / path.name, source_dir.parent / relative))
            elif len(parts) >= 2 and parts[0] == "data":
                relative = Path(*parts[1:])
                candidates.extend((source_dir / path.name, source_dir.parent / relative))
            elif parts and re.fullmatch(r"question\d+", parts[0], re.I):
                candidates.extend((source_dir / path.name, source_dir.parent / path))
            else:
                candidates.append(source_dir / path)
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate)
            # A producer result is authoritative even before a downstream
            # consumer checks its existence.
            if value in self.produced_paths:
                return self.produced_paths[value]
            if path.name in self.produced_paths:
                return self.produced_paths[path.name]
            return value

        return {
            key: value if key in output_keys else resolve(value)
            for key, value in arguments.items()
        }

    def _redirect_outputs(
        self, contract: ToolContract, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Keep every generated file below the dedicated runtime directory.

        Input paths are never altered.  For an output parameter, however, the
        caller must not be able to select a location outside ``work_dir``:
        absolute paths are reduced to their filename, while conventional
        ``benchmark/out/...`` paths retain their suffix below ``work_dir``.
        This keeps the benchmark data tree read-only even when a model echoes
        an input filename as its output path.
        """
        output_keys = set(contract.output_parameters())
        output_dir = self.work_dir
        source_dir = self.last_listed_directory
        if source_dir is not None and re.fullmatch(r"question\d+", source_dir.name, re.I):
            output_dir = self.work_dir / source_dir.name

        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_work_dir = output_dir.resolve()

        def redirect(value: Any) -> Any:
            if isinstance(value, list):
                return [redirect(item) for item in value]
            if not isinstance(value, str):
                return value
            path = Path(value)
            if path.is_absolute():
                # An absolute path already inside the runtime root is the
                # intended destination (the dataset loader builds
                # work_dir/question<id>/...).  Keep it verbatim so the
                # per-question subdirectory -- and the downstream data flow
                # -- is not destroyed.
                try:
                    resolved_path = path.resolve()
                    resolved_path.relative_to(resolved_work_dir)
                    return str(resolved_path)
                except (OSError, ValueError):
                    pass
                # An absolute output path anywhere else (including one under
                # benchmark/data) is never permission to write there.
                # Preserve only its basename so it cannot escape the runtime
                # root or leak arbitrary host directory names.
                path = Path(path.name)
            parts = path.parts
            if len(parts) >= 2 and parts[:2] in {("benchmark", "out"), ("benchmark", "data")}:
                path = Path(*parts[2:])
            elif parts and parts[0] == "out":
                path = Path(*parts[1:])
            return str(output_dir / path)

        return {
            key: redirect(value) if key in output_keys else value
            for key, value in arguments.items()
        }

    @staticmethod
    def _vectorized_calls(
        contract: ToolContract, arguments: dict[str, Any]
    ) -> list[dict[str, Any]] | None:
        """Lift parallel lists only when the live schema declares scalars.

        Native MCP list parameters are submitted once.  If a task supplies
        lists for scalar parameters, this is the generic equivalent of an
        Earth-Agent batch wrapper: zip aligned values, invoke the scalar tool
        per row, and retain each raw tool observation.
        """
        vector_keys = [
            key for key, value in arguments.items()
            if isinstance(value, list) and not contract.accepts_array(key)
        ]
        # Older SAM2 services expose only the singular bbox argument.  On those
        # services, expand a nested list of boxes into one call per box.
        # Current services declare bboxes as a native array and must receive
        # the complete list in one call to produce a combined mask.
        vector_keys.extend(
            key for key, value in arguments.items()
            if key == "bboxes"
            and isinstance(value, list)
            and value
            and all(isinstance(item, (list, tuple)) for item in value)
            and key not in vector_keys
            and not contract.accepts_array("bboxes")
        )
        # ``centroid_distance_extremes`` declares ``centroids`` as an untyped
        # array (same situation as bboxes), but consumes the WHOLE point set in
        # one call to compute pairwise distances.  A nested N x 2 coordinate
        # list is therefore a native array, not a batch of scalar calls:
        # vectorizing it would send one flat pair per call (a 1-D array) and
        # crash the pairwise computation.
        native_centroid_keys = {
            key for key, value in arguments.items()
            if key == "centroids"
            and isinstance(value, list)
            and value
            and all(
                isinstance(item, (list, tuple))
                and len(item) >= 2
                and all(isinstance(coord, (int, float)) for coord in list(item)[:2])
                for item in value
            )
        }
        vector_keys = [key for key in vector_keys if key not in native_centroid_keys]
        if not vector_keys:
            return None
        output_keys = set(contract.output_parameters())
        input_vector_keys = [key for key in vector_keys if key not in output_keys]

        def _pairing_key(value: Any) -> str | None:
            # Align parallel groups by observation date even when the groups
            # contain different numbers of items.  Accept both ``YYYY-MM-DD``
            # and ``YYYY_MM_DD`` spellings (and ignore an HHMM acquisition
            # token, which can differ between regenerated benchmark snapshots).
            if not isinstance(value, str):
                return None
            normalized = value.replace("_", "-")
            match = re.search(r"\d{4}-\d{2}-\d{2}", normalized)
            return match.group(0) if match else None

        def _keyed_maps(keys: list[str]) -> dict[str, dict[str, int]] | None:
            maps: dict[str, dict[str, int]] = {}
            for key in keys:
                items = arguments[key]
                extracted = [_pairing_key(item) for item in items]
                if (
                    not items
                    or any(token is None for token in extracted)
                    or len(set(extracted)) != len(extracted)
                ):
                    return None
                maps[key] = {token: idx for idx, token in enumerate(extracted)}
            return maps

        input_lengths = {len(arguments[key]) for key in input_vector_keys}
        keyed: dict[str, dict[str, int]] | None = None
        ordered_keys: list[str] = []
        call_count = 0
        if len(input_lengths) != 1 or not input_lengths or next(iter(input_lengths)) == 0:
            # Do not infer a many-to-one acquisition grouping from dates.
            # Multiple scenes may share a day and output cardinality can be
            # lower than input cardinality.  Such legacy contracts need an
            # explicit adapter/planner; a generic zip would silently call the
            # wrong scenes.
            raise ValueError(
                "UNSUPPORTED_CONTRACT_DRIFT:generic_vectorizer_requires_explicit_n_to_n_alignment"
            )
        else:
            call_count = next(iter(input_lengths))
            # The benchmark may provide more output names than executable input
            # rows (for example, a missing source-band scene).  Inputs determine
            # the number of legal tool calls; surplus output names are dropped
            # at the output boundary.  Never truncate inputs, which could
            # silently misalign bands.
            for key in vector_keys:
                if key in output_keys and len(arguments[key]) < call_count:
                    raise ValueError(
                        "vectorized_scalar_arguments_must_be_nonempty_and_equal_length"
                    )
        scalar_output_keys = [
            key
            for key in contract.output_parameters()
            if key in arguments and not isinstance(arguments[key], list)
        ]
        if scalar_output_keys:
            raise ValueError(
                "vectorized_scalar_calls_require_aligned_output_lists:"
                + ",".join(scalar_output_keys)
            )

        def _row(selection: dict[str, int]) -> dict[str, Any]:
            row_arguments: dict[str, Any] = {}
            for key, value in arguments.items():
                if key in vector_keys:
                    item = value[selection[key]]
                    # Geometry workers expect one *list containing one bbox*;
                    # the bridge's vector dimension is the surrounding list of
                    # boxes supplied by an Earth-Bench trajectory.
                    if key == "bboxes":
                        item = [item]
                    row_arguments[key] = item
                else:
                    row_arguments[key] = value
            return row_arguments

        if keyed is not None:
            # Only complete sets are executable: every input AND every output
            # group must name the same date.  Extra output dates are dropped.
            output_vector_keys = [key for key in vector_keys if key in output_keys]
            selections = [
                {key: keyed[key][token] for key in vector_keys}
                for token in ordered_keys
                if all(token in keyed[key] for key in output_vector_keys)
            ]
            if not selections:
                raise ValueError(
                    "vectorized_scalar_arguments_must_be_nonempty_and_equal_length"
                )
            return [_row(selection) for selection in selections]

        return [
            _row({key: index for key in vector_keys})
            for index in range(call_count)
        ]

    @staticmethod
    def _contains_error(value: Any) -> str | None:
        if isinstance(value, str) and re.match(r"^\s*(?:error|exception|failed)\b", value, re.I):
            return value
        if isinstance(value, list):
            return next((error for item in value if (error := MCPBridge._contains_error(item))), None)
        if isinstance(value, dict):
            # Application services conventionally return {"status": "error",
            # "message": "..."}.  Preserve the diagnostic message instead
            # of recursing into ``status`` first and reducing it to "error".
            status = value.get("status")
            if isinstance(status, str) and status.lower() in {"error", "failed", "failure"}:
                message = value.get("message") or value.get("error")
                return str(message or status)
            return next((error for item in value.values() if (error := MCPBridge._contains_error(item))), None)
        return None

    async def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        requested_name = name
        contract = self.contracts.get(name)
        if contract is None:
            raise KeyError(name)

        model_arguments = arguments
        bound_benchmark_arguments = dict(arguments)
        name, arguments = self._translate_compatibility_call(name, arguments)
        contract = self.contracts.get(name)
        if contract is None:
            raise KeyError(name)
        if name == "SAM2" and "bboxes" in arguments and not contract.accepts_array("bboxes"):
            arguments["bbox"] = arguments.pop("bboxes")
        arguments = self._resolve_listing_directory(arguments) if name == "get_filelist" else arguments
        resolved_inputs = self._resolve_listed_input_paths(contract, arguments)
        executed = self._redirect_outputs(contract, resolved_inputs)
        executed = self._supply_missing_output_paths(
            name,
            contract,
            executed,
        )
        
        # Arithmetic utility tools are sometimes published without JSON
        # numeric types.  Normalize only complete numeric literals at the
        # final call boundary, after path processing is complete.
        executed = _coerce_numeric_strings(executed)
        calls = self._vectorized_calls(contract, executed)
        print(f"[bridge] model arguments: {model_arguments}")
        print(f"[bridge] resolved absolute arguments: {executed}")

        from fastmcp import Client
        async with Client(contract.server) as client:
            if calls is None:
                raw_result = _json_safe(
                    await client.call_tool(
                        name,
                        executed,
                        timeout=MCP_CALL_TIMEOUT_S,
                    )
                )
            else:
                raw_result = [
                    _json_safe(
                        await client.call_tool(
                            name,
                            call_arguments,
                            timeout=MCP_CALL_TIMEOUT_S,
                        )
                    )
                    for call_arguments in calls
                ]

        # Match Earth-Agent's compact list contract: a directory listing is
        # bare names only.  The bridge remembers the directory and resolves
        # those names to absolute paths when a later tool is invoked.  Other
        # outputs retain canonical locations for safe cross-process access.
        if requested_name != "get_filelist":
            raw_result = self._normalize_observation_paths(
                raw_result,
                arguments=arguments,
                executed=executed,
                contract=contract,
            )

        if error := self._contains_error(raw_result):
            return {
                "error_code": 1, "text": error, "outputs": {},
                "bound_benchmark_arguments": bound_benchmark_arguments,
                "live_arguments": executed,
                "executed_arguments": executed,
            }

        if requested_name == "get_filelist":
            source_dir = self._source_directory(arguments)
            if source_dir is not None:
                self.last_listed_directory = source_dir.resolve()

        # Return the live observation directly so the model can use it in the
        # next action without a bridge-specific reference token.
        model_result = self._model_visible_value(raw_result)
        outputs = {
            "result": model_result,
            "text": _compact_text(model_result),
        }

        emitted_paths: list[str] = []
        # For vectorized batches, validate only the output paths actually
        # passed to an executed row (``calls``); surplus declared names that
        # were paired to a missing source scene are never produced and must
        # not be reported as phantom failures.
        output_sources = calls if calls is not None else [executed]
        for source in output_sources:
            for key in contract.output_parameters():
                emitted_paths.extend(self._path_values(source.get(key)))
        if emitted_paths:
            missing_outputs = [path for path in emitted_paths if not Path(path).exists()]
            if missing_outputs:
                # A successful RPC response is not sufficient evidence that a
                # producer completed.  Do not publish a phantom output path that
                # a later consumer will fail to open.
                return {
                    "error_code": 1,
                    "text": "declared_output_missing:" + ",".join(missing_outputs[:4]),
                    "outputs": {},
                    "executed_arguments": executed,
                    "model_arguments": model_arguments,
                    "bound_benchmark_arguments": bound_benchmark_arguments,
                    "live_arguments": executed,
                    "root_cause": "PRODUCER_OUTPUT_MISSING",
                    "attribution": "live_service",
                }
            for path in emitted_paths:
                self.produced_paths[path] = path
                self.produced_paths[Path(path).name] = path
            paths_value = emitted_paths[0] if len(emitted_paths) == 1 else emitted_paths
            model_paths = (
                Path(paths_value).name
                if isinstance(paths_value, str)
                else [Path(path).name for path in paths_value]
            )

            outputs["output_paths"] = model_paths
            outputs["text"] = json.dumps(
                {"declared_output_paths": model_paths},
                ensure_ascii=False,
            )

        response_text = outputs["text"]

        return {
            "error_code": 0,
            "text": response_text,
            "outputs": outputs,
            # Kept bounded for the semantic judge; action models receive the
            # complete observation through ``outputs.result``.
            "semantic_evidence": _semantic_evidence(raw_result),
            "bound_benchmark_arguments": bound_benchmark_arguments,
            "live_arguments": executed,
            "executed_arguments": executed,
            "model_arguments": model_arguments,
            "execution_mode": "vectorized_scalar" if calls is not None else "native_schema",
        }


def create_app(bridge: MCPBridge, advertised_host: str, port: int) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await bridge.discover()
        yield

    app = FastAPI(title="SkillEarth Earth-Agent MCP bridge", lifespan=lifespan)

    @app.post("/list_models")
    async def list_models() -> dict[str, list[str]]:
        return {"models": sorted(bridge.contracts)}

    @app.post("/get_worker_address")
    async def get_worker_address(request: ModelRequest) -> dict[str, str]:
        address = "" if request.model not in bridge.contracts else f"http://{advertised_host}:{port}/workers/{request.model}"
        return {"address": address}

    @app.post("/workers/{tool_name}/worker_generate")
    async def worker_generate(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            return await bridge.invoke(tool_name, arguments)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown_tool:{tool_name}") from exc
        except Exception as exc:
            return {"error_code": 1, "text": f"{type(exc).__name__}:{exc}", "outputs": {}}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Expose live Earth-Agent MCP tools through SkillEarth's controller protocol.")
    parser.add_argument("--servers", required=True, help="Comma-separated MCP HTTP endpoints ending in /mcp.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=21001)
    parser.add_argument("--advertised-host", default="127.0.0.1")
    parser.add_argument(
        "--work-dir",
        default=str(Path(__file__).resolve().parents[1] / "tool_outputs"),
    )
    args = parser.parse_args()
    servers = [item.strip() for item in args.servers.split(",") if item.strip()]
    if not servers:
        raise ValueError("at_least_one_mcp_server_is_required")
    bridge = MCPBridge(servers, work_dir=args.work_dir)
    uvicorn.run(create_app(bridge, args.advertised_host, args.port), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
