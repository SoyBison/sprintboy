import json
from enum import Enum
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError, field_validator

from bot.toolkit import Runtime, Tool, ToolError, tool



class Kind(str, Enum):
    A = "a"
    B = "b"


class Ref(BaseModel):
    artist: str
    title: str


class Query(BaseModel):
    names: list[str]
    kind: Kind
    refs: list[Ref] = []

    @field_validator("names", mode="before")
    @classmethod
    def listify(cls, v):
        return [v] if isinstance(v, str) else v


@tool(args_schema=Query)
async def sample(names: list[str], kind: Kind, refs: list[Ref], runtime) -> str:
    """Do a thing.

    More detail.
    """
    return f"{names}|{kind.value}|{len(refs)}|{runtime.context}"


@tool(args_schema=Query)
async def no_runtime(names: list[str], kind: Kind) -> str:
    return f"{names}-{kind.value}"


def walk(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield k
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


def test_decorator_builds_tool():
    assert isinstance(sample, Tool)
    assert sample.name == "sample"
    assert sample.description == "Do a thing.\n\nMore detail."
    assert sample.args_schema is Query
    assert sample.wants_runtime
    assert not no_runtime.wants_runtime


def test_json_schema_is_self_contained():
    schema = sample.json_schema()
    assert schema["type"] == "object"
    assert schema["required"] == ["names", "kind"]
    assert "$ref" not in set(walk(schema)) and "$defs" not in schema
    assert schema["properties"]["kind"]["enum"] == ["a", "b"]
    item = schema["properties"]["refs"]["items"]
    assert item["properties"]["title"]["type"] == "string"  # a field named "title" survives
    assert "title" not in schema


def test_json_schema_without_fields_still_valid():
    class Empty(BaseModel):
        pass

    @tool(args_schema=Empty)
    async def nothing():
        return "x"

    assert nothing.json_schema() == {"type": "object", "properties": {}, "required": []}


@pytest.mark.asyncio
async def test_ainvoke_coerces_and_passes_runtime():
    out = await sample.ainvoke(
        {"names": "X", "kind": "b", "refs": [{"artist": "a", "title": "t"}]},
        Runtime(context="ctx"),
    )
    assert out == "['X']|b|1|ctx"


@pytest.mark.asyncio
async def test_ainvoke_skips_runtime_when_unwanted():
    assert await no_runtime.ainvoke({"names": ["X"], "kind": "a"}) == "['X']-a"


@pytest.mark.asyncio
async def test_ainvoke_needs_runtime():
    with pytest.raises(ToolError):
        await sample.ainvoke({"names": ["X"], "kind": "a"})


@pytest.mark.asyncio
async def test_ainvoke_validation_error():
    with pytest.raises(ValidationError):
        await no_runtime.ainvoke({"names": ["X"], "kind": "zzz"})


@pytest.mark.asyncio
async def test_coroutine_is_raw_function():
    out = await sample.coroutine(
        names=["X"], kind=Kind.A, refs=[], runtime=SimpleNamespace(context=1)
    )
    assert out == "['X']|a|0|1"


@pytest.mark.asyncio
async def test_schema_from_signature():
    @tool()
    async def greet(who: str, times: int = 2, runtime=None) -> str:
        """Say hi."""
        return f"{who}x{times}:{runtime.context}"

    schema = greet.json_schema()
    assert set(schema["properties"]) == {"who", "times"}
    assert schema["required"] == ["who"]
    assert schema["properties"]["times"]["default"] == 2
    assert await greet.ainvoke({"who": "bo", "times": "3"}, Runtime(context="c")) == "boX3:c".replace("X", "x")
    json.dumps(schema)
