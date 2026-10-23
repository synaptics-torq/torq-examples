"""Pure-Python registry for the latest torq-examples demo set.

This module is intentionally independent from the legacy DUT shell workflow and
is designed to model the newest repo structure without changing any files under
`torq-examples/`.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


def _resolve_repo_root() -> Path:
    """Resolve the torq-examples root for both workspace and copied-in-subfolder layouts."""
    here = Path(__file__).resolve().parent

    candidate_roots = []
    for parent in [here, *here.parents]:
        candidate_roots.extend([
            parent,
            parent / "torq-examples",
        ])

    seen: set[Path] = set()
    for candidate in candidate_roots:
        normalized = candidate.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)

        if normalized.exists() and (normalized / "setup_demos.py").exists():
            return normalized

    raise RuntimeError(
        f"Could not locate torq-examples repo from {here}. "
        "Expected a sibling folder or a copied-in-subfolder layout."
    )


REPO_ROOT = _resolve_repo_root()
SETUP_DEMOS_PATH = REPO_ROOT / "setup_demos.py"


@dataclass(frozen=True)
class SetupSource:
    app_dir: str
    setup_file: str
    setup_function: str
    model_repositories: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class DemoSpec:
    name: str
    setup_entry: str
    app_dir: str
    inference_entry: str | None
    profiling_entry: str | None
    requirements_file: str | None = None
    setup_model_repositories: tuple[tuple[str, str], ...] = ()
    model_dir_patterns: tuple[str, ...] = ()
    asset_groups: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class DemoModelCase:
    demo_name: str
    model_name: str
    repo_id: str
    model_path: str

    @property
    def case_id(self) -> str:
        return f"{self.demo_name}_{Path(self.repo_id).name}"


def _static_value(node: ast.AST, values: dict[str, object]) -> object:
    """Evaluate dependency-free constants used by demo setup declarations."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise ValueError(node.id)
        return values[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_static_value(item, values) for item in node.elts]
        return items if isinstance(node, ast.List) else tuple(items)
    if isinstance(node, ast.Dict):
        return {
            _static_value(key, values): _static_value(value, values)
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _static_value(node.left, values) + _static_value(node.right, values)
    if isinstance(node, ast.Subscript):
        container = _static_value(node.value, values)
        key = _static_value(node.slice, values)
        return container[key]
    raise ValueError(type(node).__name__)


def _module_constants(module_ast: ast.Module) -> dict[str, object]:
    values: dict[str, object] = {}
    pending: list[tuple[str, ast.AST]] = []
    for node in module_ast.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    pending.append((target.id, node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                pending.append((node.target.id, node.value))

    while pending:
        unresolved: list[tuple[str, ast.AST]] = []
        progress = False
        for name, value_node in pending:
            try:
                values[name] = _static_value(value_node, values)
                progress = True
            except (KeyError, TypeError, ValueError):
                unresolved.append((name, value_node))
        if not progress:
            break
        pending = unresolved
    return values


def _dispatch_branches(function: ast.FunctionDef) -> list[tuple[str, list[ast.stmt]]]:
    branches: list[tuple[str, list[ast.stmt]]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if (
            isinstance(test.left, ast.Name)
            and test.left.id == "name"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and isinstance(test.comparators[0].value, str)
        ):
            branches.append((test.comparators[0].value, node.body))
    return branches


def _setup_call_from_branch(
    body: list[ast.stmt],
    root_values: dict[str, object],
) -> tuple[Path, str, list[str] | None]:
    imported_functions: dict[str, tuple[str, str]] = {}
    branch_values = dict(root_values)
    loaded_modules: dict[str, Path] = {}
    setup_module_values: dict[Path, dict[str, object]] = {}
    for node in body:
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.endswith("setup_demo"):
                setup_file = REPO_ROOT / (node.module.replace(".", "/") + ".py")
                module_values = setup_module_values.get(setup_file)
                if module_values is None:
                    module_ast = ast.parse(setup_file.read_text(encoding="utf-8"))
                    module_values = _module_constants(module_ast)
                    setup_module_values[setup_file] = module_values
                for imported in node.names:
                    local_name = imported.asname or imported.name
                    imported_functions[local_name] = (
                        str(setup_file),
                        imported.name,
                    )
                    if imported.name in module_values:
                        branch_values[local_name] = module_values[imported.name]
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_load_demo_module"
        ):
            path_parts = [_static_value(arg, branch_values) for arg in node.value.args]
            if all(isinstance(part, str) for part in path_parts):
                loaded_modules[node.targets[0].id] = REPO_ROOT.joinpath(*path_parts)

    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        setup_file: Path | None = None
        setup_function: str | None = None
        if isinstance(node.func, ast.Name) and node.func.id in imported_functions:
            path_text, setup_function = imported_functions[node.func.id]
            setup_file = Path(path_text)
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in loaded_modules
        ):
            setup_file = loaded_modules[node.func.value.id]
            setup_function = node.func.attr
        if setup_file is None or setup_function is None:
            continue

        requested_models: list[str] | None = None
        if node.args:
            value = _static_value(node.args[0], branch_values)
            if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
                requested_models = list(value)
        return setup_file, setup_function, requested_models
    raise RuntimeError("Could not locate the demo setup call in setup_demos.py")


def _repositories_from_setup_file(
    setup_file: Path,
    setup_function: str,
    requested_models: list[str] | None,
) -> tuple[tuple[str, str], ...]:
    module_ast = ast.parse(setup_file.read_text(encoding="utf-8"))
    values = _module_constants(module_ast)
    function = next(
        (
            node
            for node in module_ast.body
            if isinstance(node, ast.FunctionDef) and node.name == setup_function
        ),
        None,
    )
    if function is None:
        raise RuntimeError(f"Could not find {setup_function} in {setup_file}")

    model_setup_aliases = {"setup_demo", "download_models"}
    model_setup_modules: set[str] = set()
    for node in module_ast.body:
        if isinstance(node, ast.ImportFrom) and node.module == "utils.model_setup":
            for imported in node.names:
                if imported.name in {"setup_demo", "download_models"}:
                    model_setup_aliases.add(imported.asname or imported.name)
        elif isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "utils.model_setup":
                    model_setup_modules.add(imported.asname or imported.name)

    def is_model_setup_call(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        if isinstance(node.func, ast.Name):
            return node.func.id in model_setup_aliases
        return (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"setup_demo", "download_models"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in model_setup_modules
        )

    shared_calls = [
        node
        for scope in (function, module_ast)
        for node in ast.walk(scope)
        if is_model_setup_call(node)
    ]
    shared_call = next(
        (
            call
            for call in shared_calls
            if call.args or any(keyword.arg == "repo_map" for keyword in call.keywords)
        ),
        None,
    )
    models_node: ast.AST | None = None
    repo_map: object | None = None
    if shared_call is not None:
        keyword_args = {
            keyword.arg: keyword.value
            for keyword in shared_call.keywords
            if keyword.arg is not None
        }
        repo_map_node = (
            shared_call.args[0]
            if shared_call.args
            else keyword_args.get("repo_map")
        )
        models_node = (
            shared_call.args[1]
            if len(shared_call.args) >= 2
            else keyword_args.get("models")
        )
        if repo_map_node is not None:
            repo_map = _static_value(repo_map_node, values)

    defaults = values.get("_DEFAULT_MODELS")
    if requested_models is None and models_node is not None:
        try:
            resolved_models = _static_value(models_node, values)
            if isinstance(resolved_models, (list, tuple)):
                defaults = resolved_models
        except (KeyError, TypeError, ValueError):
            pass
    if requested_models is None and isinstance(defaults, (list, tuple)) and all(
        isinstance(item, str) for item in defaults
    ):
        requested_models = list(defaults)

    if repo_map is None:
        def looks_like_repo_id(value: str) -> bool:
            parts = value.split("/")
            return (
                len(parts) >= 2
                and all(parts)
                and not any(char.isspace() for char in value)
                and not value.startswith(("/", "./", "../"))
            )

        repo_maps = [
            value
            for value in values.values()
            if isinstance(value, dict)
            and value
            and all(
                isinstance(key, str)
                and isinstance(repo_id, str)
                and looks_like_repo_id(repo_id)
                for key, repo_id in value.items()
            )
        ]
        requested_names = {
            model_spec.rsplit(":", 1)[0] if ":" in model_spec else model_spec
            for model_spec in (requested_models or [])
        }
        if repo_maps:
            repo_map = max(
                repo_maps,
                key=lambda candidate: (
                    len(requested_names.intersection(candidate)),
                    len(candidate),
                ),
            )
        else:
            repo_ids = [
                value
                for value in values.values()
                if isinstance(value, str) and looks_like_repo_id(value)
            ]
            if repo_ids:
                repo_map = {repo_id: repo_id for repo_id in repo_ids}

    if not isinstance(repo_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in repo_map.items()
    ):
        raise RuntimeError(
            f"Could not resolve static Hugging Face repository data in {setup_file}"
        )

    if requested_models is None:
        requested_models = list(repo_map)

    repositories: list[tuple[str, str]] = []
    for model_spec in requested_models:
        model_name = model_spec.rsplit(":", 1)[0] if ":" in model_spec else model_spec
        repositories.append((model_name, repo_map.get(model_name, model_name)))
    return tuple(repositories)


def discover_setup_sources() -> dict[str, SetupSource]:
    """Discover demo names, setup modules, selected models, and repo IDs via AST."""
    root_ast = ast.parse(SETUP_DEMOS_PATH.read_text(encoding="utf-8"))
    root_values = _module_constants(root_ast)
    setup_dispatcher = next(
        (
            node
            for node in root_ast.body
            if isinstance(node, ast.FunctionDef) and node.name == "setup_demo"
        ),
        None,
    )
    if setup_dispatcher is None:
        raise RuntimeError(f"Could not find setup_demo() in {SETUP_DEMOS_PATH}")

    sources: dict[str, SetupSource] = {}
    for demo_name, branch in _dispatch_branches(setup_dispatcher):
        setup_file, setup_function, requested_models = _setup_call_from_branch(
            branch,
            root_values,
        )
        repositories = _repositories_from_setup_file(
            setup_file,
            setup_function,
            requested_models,
        )
        relative_setup_file = setup_file.relative_to(REPO_ROOT).as_posix()
        sources[demo_name] = SetupSource(
            app_dir=setup_file.parent.relative_to(REPO_ROOT).as_posix(),
            setup_file=relative_setup_file,
            setup_function=setup_function,
            model_repositories=repositories,
        )
    return sources


def _app_dir_for_demo(name: str, setup_sources: dict[str, SetupSource]) -> str:
    source = setup_sources.get(name)
    if source is None:
        raise RuntimeError(f"No setup source discovered for demo '{name}'")
    return source.app_dir


def _asset_groups_for_demo(
    name: str,
    setup_sources: dict[str, SetupSource],
) -> tuple[tuple[str, ...], ...]:
    source = setup_sources.get(name)
    if source is None:
        return ()
    # All executable Torq model packages contain at least one VMFB. Detailed
    # runtime inputs are resolved by each command builder, not by the registry.
    return (("*.vmfb*",),)


def _model_dir_patterns_for_demo(
    name: str,
    setup_sources: dict[str, SetupSource],
) -> tuple[str, ...]:
    """Derive model directory patterns from discovered HF repository IDs."""
    source = setup_sources.get(name)
    if source is None:
        return ()
    return tuple(
        dict.fromkeys(f"{Path(repo_id).name}*" for _, repo_id in source.model_repositories)
    )


def _setup_model_repositories_for_demo(
    name: str,
    setup_sources: dict[str, SetupSource],
) -> tuple[tuple[str, str], ...]:
    source = setup_sources.get(name)
    return source.model_repositories if source is not None else ()


def repo_demo_names() -> list[str]:
    """Read the static DEMOS list without executing setup_demos.py.

    This function is called while pytest parameterizes tests during collection,
    so importing setup_demos.py here would also import its dependency tree and
    could block collection before any test or progress output is produced.
    """
    module_ast = ast.parse(SETUP_DEMOS_PATH.read_text(encoding="utf-8"))
    for node in module_ast.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "DEMOS" for target in targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
            raise RuntimeError(f"DEMOS in {SETUP_DEMOS_PATH} must be a list of strings")
        return value
    raise RuntimeError(f"Could not find DEMOS in {SETUP_DEMOS_PATH}")


def _python_entry_for_demo(
    name: str,
    setup_sources: dict[str, SetupSource],
) -> str | None:
    source = setup_sources.get(name)
    if source is None:
        return None
    candidate = REPO_ROOT / source.app_dir / "src" / "infer.py"
    return candidate.relative_to(REPO_ROOT).as_posix() if candidate.exists() else None


def _requirements_file_for_demo(
    name: str,
    setup_sources: dict[str, SetupSource],
) -> str | None:
    source = setup_sources.get(name)
    if source is None:
        return None
    candidate = REPO_ROOT / source.app_dir / "requirements.txt"
    return candidate.relative_to(REPO_ROOT).as_posix() if candidate.exists() else None


def demo_registry() -> dict[str, DemoSpec]:
    registry: dict[str, DemoSpec] = {}
    setup_sources = discover_setup_sources()
    for name in repo_demo_names():
        registry[name] = DemoSpec(
            name=name,
            setup_entry="setup_demos.py",
            app_dir=_app_dir_for_demo(name, setup_sources),
            inference_entry=_python_entry_for_demo(name, setup_sources),
            profiling_entry="profile.py",
            requirements_file=_requirements_file_for_demo(name, setup_sources),
            setup_model_repositories=_setup_model_repositories_for_demo(name, setup_sources),
            model_dir_patterns=_model_dir_patterns_for_demo(name, setup_sources),
            asset_groups=_asset_groups_for_demo(name, setup_sources),
        )
    return registry


def demo_model_cases(
    remote_root: str = "/home/torq-examples",
) -> list[DemoModelCase]:
    """Build one pytest case per demo model repository discovered from setup."""
    models_root = remote_root.rstrip("/") + "/models"
    cases: list[DemoModelCase] = []
    for demo_name, spec in demo_registry().items():
        for model_name, repo_id in spec.setup_model_repositories:
            cases.append(DemoModelCase(
                demo_name=demo_name,
                model_name=model_name,
                repo_id=repo_id,
                model_path=f"{models_root}/{repo_id}",
            ))
    return cases
