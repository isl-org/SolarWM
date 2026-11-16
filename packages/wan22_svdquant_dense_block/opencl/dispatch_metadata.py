"""Static-dispatch metadata: parameter domains, hardware constraints, legal pool, default and ranking.

Owned by static dispatch. A runtime tuner only consumes the resulting ``(default, pool)``. Dependency-free; the
source-embedded ``AUTOTUNE_METADATA`` block is the declarative form used by OpenCL and Vulkan sources, and
``check_candidate``/``rank_pool`` serve any backend whose pool is a list of integer configs.

Shapes: ``bucket_dim``/``bucket_key`` define the shape regimes used for ranking and for runtime cache keys (dynamic
dimensions bucketed, static ones exact). Constraints may use ``DEVICE_<LIMIT>`` (queried hardware) and
``WORKLOAD_<DIM>`` (the LIVE dimension, never its bucket) tokens, so legality stays exact for every shape in a bucket.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from string import Formatter
from typing import Mapping


def bucket_dim(n: int) -> int:
    """Bucket a dynamic dimension into individual small integers (1..16), then multiples of powers of 2.

    For n <= 16, returns exact individual integers 1..16.
    For n > 16, rounds up to a power-of-2 step (2^(floor(log2(n - 1)) - 1)), yielding
    24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, ... (monotonic and idempotent).
    """
    n = int(n)
    if n <= 16:
        return max(1, n)
    step = 1 << ((n - 1).bit_length() - 2)
    return ((n + step - 1) // step) * step


def bucket_key(dynamic: Mapping[str, int], **static) -> dict:
    """Hybrid workload key: dynamic dimensions bucketed (``M`` -> ``M_bucket``), static dimensions, dtypes, layouts and
    modes exact. Use the same key for static ranking and for the runtime cache; legality, bounds, tails and launch
    grids must still use the live dimensions, never the bucket."""
    return {**{f"{name}_bucket": bucket_dim(value) for name, value in dynamic.items()}, **static}


@dataclass(frozen=True)
class Parameter:
    name: str
    values: tuple[int, ...]
    default: int


@dataclass(frozen=True)
class Constraint:
    operation: str
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class KernelMetadata:
    version: int
    entrypoint: str
    parameters: tuple[Parameter, ...]
    conditional_options: tuple[tuple[str, int, str], ...]
    constraints: tuple[Constraint, ...]

    @property
    def defaults(self) -> dict[str, int]:
        return {parameter.name: parameter.default for parameter in self.parameters}

    def build_options(self, values: Mapping[str, int]) -> list[str]:
        options = [f"-D{parameter.name}={values[parameter.name]}" for parameter in self.parameters]
        for name, expected, option in self.conditional_options:
            if values[name] == expected:
                options.extend(option.format_map({key: str(value) for key, value in values.items()}).split())
        return options

    def candidates(self, device, workload: Mapping[str, int] | None = None) -> list[dict[str, int]]:
        names = [parameter.name for parameter in self.parameters]
        pool = [
            dict(zip(names, combination, strict=True))
            for combination in itertools.product(*(param.values for param in self.parameters))
        ]
        return [
            values
            for values in pool
            if all(constraint_is_valid(rule, values, device, workload) for rule in self.constraints)
        ]


def constraint_is_valid(
    constraint: Constraint, values: Mapping[str, int], device, workload: Mapping[str, int] | None = None
) -> bool:
    def resolve(token):
        if token.startswith("WORKLOAD_"):  # live dimension, e.g. WORKLOAD_N; never a bucket
            if workload is None or token.removeprefix("WORKLOAD_") not in workload:
                raise ValueError(f"Workload dimension {token} is unknown; pass the live dimensions")
            return int(workload[token.removeprefix("WORKLOAD_")])
        if token.startswith("DEVICE_"):  # DEVICE_MAX_WORK_GROUP_SIZE -> device.max_work_group_size, etc.
            value = getattr(device, token.removeprefix("DEVICE_").lower(), None)
            if value is None:
                raise ValueError(f"Device limit {token} is unknown; query the actual GPU limit")
            return int(value)
        return values[token] if token in values else int(token)

    args = [resolve(token) for token in constraint.arguments]
    if constraint.operation == "le":
        return args[0] <= args[1]
    if constraint.operation == "ge":
        return args[0] >= args[1]
    if constraint.operation == "divisible":
        return args[1] != 0 and args[0] % args[1] == 0
    if constraint.operation == "product_le":
        first, divisor, second, other_divisor, limit = args
        return (
            divisor > 0
            and other_divisor > 0
            and first % divisor == 0
            and second % other_divisor == 0
            and (first // divisor) * (second // other_divisor) <= limit
        )
    raise ValueError(f"Unsupported autotune constraint: {constraint.operation}")


def parse_metadata(source: str) -> KernelMetadata:
    if source.count("AUTOTUNE_METADATA_BEGIN") != 1 or source.count("AUTOTUNE_METADATA_END") != 1:
        raise ValueError("Missing AUTOTUNE_METADATA_BEGIN ... AUTOTUNE_METADATA_END block or duplicate blocks")
    match = re.search(r"AUTOTUNE_METADATA_BEGIN(.*?)AUTOTUNE_METADATA_END", source, re.DOTALL)
    if match is None:
        raise ValueError("Malformed AUTOTUNE_METADATA block")
    version = None
    entrypoint = None
    parameters = []
    options = []
    constraints = []
    identifier = r"[A-Za-z_][A-Za-z_0-9]*"
    for raw in match.group(1).splitlines():
        line = raw.strip().removeprefix("*").removeprefix("//").strip()
        if not line:
            continue
        directive, *args = line.split()
        if directive == "version" and len(args) == 1 and version is None:
            version = int(args[0])
        elif directive == "entrypoint" and len(args) == 1 and entrypoint is None:
            entrypoint = args[0]
            if not re.fullmatch(identifier, entrypoint):
                raise ValueError("Invalid metadata entrypoint")
        elif directive == "param" and len(args) == 3:
            name, domain, default_text = args
            if not re.fullmatch(identifier, name) or not default_text.startswith("default="):
                raise ValueError("Invalid parameter name or missing default=<value>")
            values = tuple(int(value) for value in domain.split(","))
            default = int(default_text.removeprefix("default="))
            if default not in values or len(set(values)) != len(values):
                raise ValueError(f"Invalid values/default for parameter {name}")
            parameters.append(Parameter(name, values, default))
        elif directive == "option-if" and len(args) >= 2:
            name, expected = args[0].split("=", 1)
            options.append((name, int(expected), " ".join(args[1:])))
        elif directive == "constraint" and len(args) >= 2:
            constraints.append(Constraint(args[0], tuple(args[1:])))
        else:
            raise ValueError(f"Invalid or duplicate metadata directive: {line}")
    if version != 1 or not entrypoint or not parameters:
        raise ValueError("Metadata requires version 1, entrypoint and parameters")
    names = {param.name for param in parameters}
    if len(names) != len(parameters):
        raise ValueError("Duplicate metadata parameter names")
    for name, expected, option in options:
        if name not in names or expected not in next(param.values for param in parameters if param.name == name):
            raise ValueError(f"Conditional option references unknown parameter/value {name}={expected}")
        for _, field, spec, conversion in Formatter().parse(option):
            if field is not None and (field not in names or spec or conversion):
                raise ValueError(f"Invalid conditional option field: {field}")
    arities = {"le": 2, "ge": 2, "divisible": 2, "product_le": 5}
    for rule in constraints:
        if arities.get(rule.operation) != len(rule.arguments):
            raise ValueError(f"Unsupported or malformed constraint: {rule}")
        for token in rule.arguments:
            if token not in names and not re.fullmatch(r"(?:DEVICE|WORKLOAD)_[A-Za-z0-9_]+|-?\d+", token):
                raise ValueError(f"Constraint references unknown value {token}")
    return KernelMetadata(version, entrypoint, tuple(parameters), tuple(options), tuple(constraints))


def check_candidate(metadata: KernelMetadata, values: Mapping[str, int], device, workload=None) -> None:
    """Raise ValueError unless ``values`` is inside the declared domains and satisfies every hardware constraint."""
    if not isinstance(values, Mapping) or set(values) != set(metadata.defaults):
        raise ValueError(f"Candidate parameter schema mismatch: {values}")
    if any(
        type(values[param.name]) is not int or values[param.name] not in param.values for param in metadata.parameters
    ):
        raise ValueError(f"Candidate outside metadata domain: {values}")
    if not all(constraint_is_valid(rule, values, device, workload) for rule in metadata.constraints):
        raise ValueError(f"Candidate violates metadata constraints: {values}")


def static_pool(
    metadata: KernelMetadata, device, workload: Mapping[str, int] | None = None, order=None
) -> tuple[dict[str, int], list[dict[str, int]]]:
    """Legal pool for ``device`` (and live ``workload`` dims) and a deterministic default.

    Without ``order`` the default is the declared one when legal, else the first legal candidate. With a measured
    ``order`` (best first, see ``order_from_timings``) the pool is ranked by it and the default is its first legal
    entry: the declared default is only the fallback when nothing has been measured.
    """
    pool = metadata.candidates(device, workload)
    if not pool:
        raise ValueError("No valid candidates from kernel metadata on this device")
    if order:
        pool = rank_pool(pool, order)
        return dict(pool[0]), pool
    default = metadata.defaults if metadata.defaults in pool else pool[0]
    return dict(default), pool


def order_from_timings(timings) -> list[dict[str, int]]:
    """Best-first candidate order from a runtime sweep (``runtime_selection.select(..., report)['timings']``).

    Use it offline to write ranking evidence (e.g. JSON per hardware class and shape class); run the tuner with
    ``finalists=len(pool)`` so every candidate has a median.
    """
    inf = float("inf")
    return [
        dict(t["candidate"])
        for t in sorted(timings, key=lambda t: t["median_seconds"] or t["screen_seconds"] or inf)
    ]


def rank_pool(pool, order) -> list[dict[str, int]]:
    """Reorder ``pool`` by measured ``order`` (best first). Unlisted candidates keep their order after the listed
    ones; nothing is dropped, so ranking never shrinks the legal pool."""
    position = {tuple(sorted(dict(c).items())): i for i, c in enumerate(order)}
    key = lambda c: position.get(tuple(sorted(dict(c).items())), len(position))
    return [dict(c) for c in sorted(pool, key=key)]
