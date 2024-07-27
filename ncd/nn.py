from __future__ import annotations
from dataclasses import dataclass, field
from itertools import repeat
from typing import Callable
from ncd import Shape, Variant, Cartesian, Homf, Conf, V

@dataclass(repr=False, frozen=True)
class Linear(Variant):
    content: str = ""
    _dom: Shape = field(default_factory=Conf)
    _cod: Shape = field(default_factory=Conf)

    def __str__(self):
        return "L" + self.content


def einsplit(content: str) -> tuple[Shape, Shape]:
    """Generates dom / cod Shape from einsum expression.
    eg. 'i j k, j k -> i k'
        => ([i j k^], [j k^]) -> [i k^]
    """

    keys: set[str] = set([x.strip(", ") for x in content.split(" ")]) - set([
        "->"
    ])
    configs: dict[str, Conf] = {k: Conf(k) for k in keys}  # type: ignore
    dom, cod = [x.split(",") for x in content.split("->")]

    def toHomf(target: str):
        return Homf(
            Cartesian(*[configs[k] for k in target.strip().split(" ")])
        )

    dom = Cartesian(*map(toHomf, dom))
    cod = Cartesian(*map(toHomf, cod))

    return dom, cod


@dataclass(repr=False, frozen=True)
class Einops(Variant):
    @classmethod
    def construct(cls, content: str):
        dom, cod = einsplit(content)
        return cls.corrected(content, dom, cod)


@dataclass(repr=False, frozen=True)
class SoftMax(Variant):
    @classmethod
    def construct(cls, target: Shape | None = None):
        target = target or Homf(Conf())
        return cls.corrected("◁", target, target)


@dataclass(repr=False, frozen=True)
class Addition(Variant):
    content: str = "+"
    _dom: Shape = field(default_factory=lambda: V + V)
    _cod: Shape = V