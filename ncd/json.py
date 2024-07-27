
from ncd import *
import dataclasses
from collections.abc import Collection
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from _typeshed import DataclassInstance

def forward_shift(target: str):
    return '\n'.join([f'  {x}' for x in target.split('\n')])

def split_outer(target: str, splitter = ',', l = '{', r = '}'):
    depth = 0
    prev = 0
    for i, t in enumerate(target):
        if t == l:
            depth += 1
        if t == r:
            depth -= 1
        if t == splitter and depth == 0:
            yield(target[prev:i])
            prev = i

def toJson(target: 'DataclassInstance | Collection | Any'):
    cls = type(target)

    # Case: its a dataclass
    if hasattr(target, '__dataclass_fields__'):
        header = f'{cls.__module__}.{cls.__name__}'
        fields = dataclasses.fields(target) #type: ignore
        children = [f'"type": "{header}"'] + [f'"{f.name}": {toJson(getattr(target, f.name))}' for f in fields]
        return '{\n' + forward_shift(',\n'.join(children)) + '\n}'
    if isinstance(target, str):
        return f'"{target}"'
    if isinstance(target, Collection):
        children = [f'{toJson(x)}' for x in target]
        if children == []:
            return '[]'
        return '[\n' + forward_shift('\n,'.join(children)) + ']'
    else:
        return f'{str(target)}'