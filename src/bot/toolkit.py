"""
A small tool abstraction: an async function plus a pydantic model describing
its arguments, which is all a model needs to call it and all we need to
validate what it sent back.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, create_model


@dataclass
class Runtime:
    """What a tool receives as its `runtime` parameter."""

    context: Any
    run_id: str | None = None


class ToolError(Exception):
    pass


def _inline_refs(node: Any, defs: dict, seen: tuple[str, ...] = ()) -> Any:
    """Copy `node` with every `$ref` replaced by the definition it points at."""
    if isinstance(node, list):
        return [_inline_refs(item, defs, seen) for item in node]
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if isinstance(ref, str):
        key = ref.rsplit("/", 1)[-1]
        if key in defs and key not in seen:
            resolved = _inline_refs(defs[key], defs, seen + (key,))
            extra = {k: v for k, v in node.items() if k != "$ref"}
            return {**resolved, **_inline_refs(extra, defs, seen)}
        raise ValueError(f"cannot inline schema reference {ref!r}")
    out = {}
    for key, value in node.items():
        if key == "properties" and isinstance(value, dict):
            # Property names are arbitrary: one called "title" is not metadata.
            out[key] = {name: _inline_refs(sub, defs, seen) for name, sub in value.items()}
        else:
            out[key] = _inline_refs(value, defs, seen)
    return out


def _strip_titles(node: Any) -> Any:
    if isinstance(node, list):
        return [_strip_titles(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key == "title" and isinstance(value, str):
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {name: _strip_titles(sub) for name, sub in value.items()}
        else:
            out[key] = _strip_titles(value)
    return out


class Tool:
    def __init__(self, coroutine: Callable, args_schema: type[BaseModel]):
        self.name: str = coroutine.__name__
        self.description: str = inspect.cleandoc(coroutine.__doc__ or "")
        self.args_schema: type[BaseModel] = args_schema
        self.coroutine: Callable = coroutine
        params = inspect.signature(coroutine).parameters
        self.wants_runtime: bool = "runtime" in params
        self._accepted = set(params)
        self._takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def __repr__(self) -> str:
        return f"Tool({self.name})"

    def json_schema(self) -> dict:
        raw = self.args_schema.model_json_schema()
        defs = raw.pop("$defs", {})
        schema = _strip_titles(_inline_refs(raw, defs))
        schema["type"] = "object"
        schema.setdefault("properties", {})
        schema["required"] = list(schema.get("required", []))
        return schema

    async def ainvoke(self, args: dict, runtime: Runtime | None = None) -> str:
        model = self.args_schema.model_validate(args)
        kwargs = {
            name: getattr(model, name)
            for name in type(model).model_fields
            if self._takes_kwargs or name in self._accepted
        }
        if self.wants_runtime:
            if runtime is None:
                raise ToolError(f"tool {self.name} needs a runtime")
            kwargs["runtime"] = runtime
        return str(await self.coroutine(**kwargs))


def _schema_from_signature(fn: Callable) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name == "runtime" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = Any if param.annotation is inspect.Parameter.empty else param.annotation
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (annotation, default)
    return create_model(f"{fn.__name__}_args", **fields)


def tool(args_schema: type[BaseModel] | None = None):
    """Decorator: `@tool(args_schema=Model)` over an async function makes a Tool.

    Without `args_schema` the model is built from the function's signature.
    """

    def decorate(fn: Callable) -> Tool:
        return Tool(fn, args_schema or _schema_from_signature(fn))

    return decorate
