from __future__ import annotations
import dataclasses
import itertools
from collections.abc import Collection
from dataclasses import asdict, dataclass, field
from functools import cache, cached_property, partial, reduce
from itertools import repeat, starmap
from .display_text import TextTile, bcolors
import random
import inspect
from typing import (
    Any,
    Callable,
    ClassVar,
    Concatenate,
    Generator,
    Generic,
    Iterable,
    Iterator,
    Literal,
    ParamSpec,
    Self,
    Sized,
    Tuple,
    TypeAlias,
    TypeVar,
)

# from ._Conf import (
#     CompositionError,
#     Functor,
#     ConfKey,
#     Conf,
#     Configuration,
#     GetConfig,
#     Delta,
# )

__all__ = [
    "Shape",
    "Cartesian",
    "ConstructionRule",
    "Constructor",
    "Composed",
    'Coproduct',
    'Del',
    'Variant',
    "Homf",
    "GenericField",
    "Reversed",
    "CompositionError",
    "Functor",
    "ConfKey",
    "Conf",
    "Configuration",
    "GetConfig",
    "Duplicate",
    "compose",
    "TextTile",
    "Identity",
]


C = TypeVar("C")
T = TypeVar("T")
P = ParamSpec("P")
import typing


#######
def zip_defaults[T](
    xss: Iterable[Collection[T] | tuple[T, ...]],
    ds: Iterable[T],
):
    xss, ds = list(xss), list(ds)
    max_length = max(map(len, xss))
    xss = [
        tuple(xs) + tuple(repeat(d, max_length - len(xs)))
        for xs, d in zip(xss, ds)
    ]
    transposed: tuple[tuple[T]] = tuple(zip(*xss))
    return transposed


def instanceOf(cls):
    return lambda x: isinstance(x, cls)


import functools


def traceErrors[**P, Q](f: Callable[P, Q]) -> Callable[P, Q]:
    @functools.wraps(f)
    def wrap(*args: P.args, **kwds: P.kwargs):
        try:
            return f(*args, **kwds)
        except Exception as e:
            print("Trace:", f.__name__, args, kwds)
            raise e

    return wrap


#######
# Returns NONE if the rule is not applicable,
# Returns the NEW SHAPE if it is.
def AddRule(cls: Constructor):
    def wrap(f: Callable[..., "None | Shape"]):
        rule = ConstructionRule(f)
        cls.rules.append(rule)
        return rule
    return wrap


class ConstructionRule[**P]:
    def __init__(self, f: Callable[Concatenate[C, P], "None | Shape"]):
        self.f = traceErrors(f)
    @classmethod
    def addRule(cls, f: Callable[Concatenate[C, P], "None | Shape"]):
        target_class = '.'.join(f.__qualname__.split('.')[:-1])
        rule = ConstructionRule(f)
        Constructor.ruleDict[target_class] = Constructor.ruleDict.get(
                target_class, []
            ) + [rule]
    def __call__(self, cls, *args: P.args, **kwds: P.kwargs):
        return self.f(cls, *args, **kwds)


#######
class Constructor[**Init, **Corr](type):
    ruleDict: dict = dict()
    def __init__(cls, *args, **kwargs):
        if '__init__' in args[2]:
            args[2]['construct'] = args[2].pop('__init__')
        if getattr(cls, "rules", None) is None:
            cls.rules: list[ConstructionRule] = []
        cls.rules = cls.rules + Constructor.ruleDict.get(cls.__qualname__, [])
        return super().__init__(*args, **kwargs)
    # We enter externally through the CONSTRUCT method.
    # This rearranges the variables.
    def use_defaults(
        cls: type["Shape"],
        *args: Corr.args,
        **kwds: Corr.kwargs,
    ):
        def remove_missing(x):
            return x if x != dataclasses.MISSING else None

        def get_default(f: dataclasses.Field[T]):
            try:
                return (
                    remove_missing(f.default) or f.default_factory()  # type: ignore
                )  # type: ignore
            except TypeError as e:
                print(cls, f.name, f.default, f.default_factory)
                raise e

        size = len(args)
        fields = dataclasses.fields(cls)[size:]
        return args + tuple(map(get_default, fields))
    def construct(
        cls: type["Shape"],
        *args: Init.args,
        **kwds: Init.kwargs,
    ) -> "Shape":
        return (
            cls.corrected(*args, **kwds)
            if bool(kwds)
            else cls.corrected(*cls.use_defaults(*args, **kwds))
        )
    # This checks whether the corrected parameters are correct.
    def corrected(
        cls: type["Shape"], *args: Corr.args, **kwds: Corr.kwargs
    ) -> "Shape":
        for r in cls.rules:
            outcome = r(cls, *args, **kwds)
            if outcome is not None:
                return outcome
        return super().__call__(*args, **kwds)  # type: ignore
    @typing.override
    def __call__(
        cls: type["Shape"],
        *args: Init.args,
        **kwds: Init.kwargs,
    ) -> "Shape":
        return cls.construct(*args, **kwds)


#######
ShapeConstructable: TypeAlias = "Shape | str | int"


#######
@dataclass(frozen=True)
class Shape(metaclass=Constructor):
    content: Any
    __match_args__ = ("content", "dom", "cod", "identity")
    ### The core properties.
    @cached_property
    def identity(self) -> bool:
        return True
    @cached_property
    def dom(self):
        return self if self.identity else self.domain()
    @cached_property
    def cod(self):
        return self if self.identity else self.codomain()
    ### We skip the identity checks.
    def domain(self):
        return self
    def codomain(self):
        return self
    ### Make sure the shapes become shapes.
    @staticmethod
    def solveMatch(type_name: str | type, target) -> Any:
        try:
            match type_name:
                case "tuple[Shape, ...]" | "tuple[Shape]":
                    return tuple(map(shape, target))
                case "Shape":
                    return shape(target)
            return target
        except Exception as e:
            print(type_name, target, type(target))
            raise e
    ### TODO: Make this work for kwargs.
    @ConstructionRule.addRule
    def toShape(cls: type[Self], *args, **kwargs):  # type: ignore
        if not args:
            return None
        types = [x.type for x in dataclasses.fields(cls)]
        converted = tuple(starmap(Shape.solveMatch, zip(types, args)))
        return None if converted == args else cls.corrected(*converted)
    # rules: ClassVar = [toShape]
    ### For unpacking, ###
    def __iter__(self):
        for key in map(lambda x: x.name, dataclasses.fields(self)):
            yield getattr(self, key)
    def __post_init__(self):
        """Checks whether all rules have been satisfied."""
        for r in type(self).rules:  # type: ignore
            rule = r(type(self), *self)
            assert not rule
    # The naming methods.
    def __repr__(self):
        if self.identity:
            return f"{str(self)}"
        return f"{self.dom}--{self}->{self.cod}"
    def __str__(self):
        return bcolors.apply(
            str(self.content),
            bcolors.fg.CYAN if self.identity else bcolors.fg.GREEN,
            bcolors.UNDERLINE,
        )
    def tile(self, mapping: Callable[[Shape], str] = str):
        return TextTile(mapping(self))
    # The algebraic methods
    def __add__(self, other):
        return Cartesian.construct(self, other)
    def __radd__(self, other):
        return Cartesian.construct(other, self)
    def __rshift__(self, other):
        return Homf.construct(self, other)
    def __lshift__(self, other):
        return Homf.construct(Cartesian(), self, other)
    def __matmul__(self, other):
        return compose(self, shape(other))

Identity = typing.Annotated[Shape, "identity"]

#######
def shape(target: ShapeConstructable) -> Shape:
    match target:
        case Shape():
            return target
        case str():
            target = target.strip()
            if "()" == target or ''==target:
                return Cartesian()
            if "[]" == target or "V" == target:
                return Homf()
            if ":" in target:
                content, dc = target.split(":")
                dc = tuple(map(shape, dc.split("->")))
                assert len(dc) == 2
                return Variant(content, *dc)
            if ',' in target:
                return Cartesian.construct(*map(shape, target.split(",")))
            if '|' in target:
                return Coproduct.corrected(tuple(map(shape,target.split('|'))))
            if "^" in target or " " in target:
                homf = target.split("^")
                match homf:
                    case [left]:
                        targ = 'V'
                        right = ''
                    case [left,targ]:
                        targ = targ or 'V'
                        right = ''
                    case [left,targ,right]:
                        targ = targ or 'V'
                left = Cartesian.construct(*map(shape, left.split(" ")))
                targ = shape(targ)
                right = Cartesian.construct(*map(shape, right.split(" ")))
                return Homf.corrected(left, targ, right)
            if target[0] == "*":
                return Conf(target[1:])
            return Shape.construct(target)
        case int():
            return Shape(target)
    raise ValueError(f"Cannot convert {target} {type(target)} {isinstance(target, Shape)} into shape!")


########
@dataclass(repr=False, frozen=True)
class Variant(Shape):
    _dom: Shape
    _cod: Shape
    @cached_property
    def identity(self) -> bool:
        return False
    def domain(self):
        return self._dom
    def codomain(self):
        return self._cod


#######
def identity(target: Shape):
    return target.identity
def dom(target: Shape):
    return target.dom
def cod(target: Shape):
    return target.cod

#######
@dataclass(repr=False, frozen=True)
class IterShape(Shape):
    content: tuple[Shape, ...] = ()
    name_convention: ClassVar[tuple[str, str, str]] = (
        "(",
        "; ",
        ")",
    )

    ### CLASS SETUP ###
    def __init_subclass__(cls, name_convention=("(", "; ", ")")):
        cls.name_convention = name_convention
        return super().__init_subclass__()

    def __new__(cls, *args):
        assert cls != IterShape
        return super().__new__(cls)

    ### GENERAL METHODS ###
    @classmethod
    def get_content(cls, target: Shape) -> tuple[Shape, ...]:
        match target:
            case cls():  # type: ignore
                return target.content
            case _:
                return (target,)

    ### CONSTRUCTION ###
    # The algebraic rules for simplification.
    @ConstructionRule.addRule
    def flatten(cls: type[Self], content: tuple[Shape, ...]):  # type: ignore
        if not any(map(instanceOf(cls), content)):
            return None
        return cls.construct(*sum(map(cls.get_content, content), ()))
    @ConstructionRule.addRule
    def sizeOne(cls: type[Self], content: tuple[Shape, ...]):  # type: ignore
        return None if not len(content) == 1 else content[0]
    # rules: ClassVar = Shape.rules + [flatten, sizeOne]
    @classmethod
    def construct(cls, *args: ShapeConstructable):
        return cls.corrected(args)

    ### NAMING ###
    def __str__(self):
        l, j, r = self.name_convention
        return l + j.join(map(str, self.content)) + r

    def tile(self, mapping=str):
        l, j, r = self.name_convention
        return TextTile(
            "\n".join([
                str(x.tile(mapping)) + j.strip() +'-' for x in self.content
            ])
            ,
            "",
            "-",
        )


#######
@dataclass(repr=False, frozen=True)
class MonoidalProduct(IterShape):
    @cached_property
    def identity(self):
        return all(map(identity, self.content))

    def domain(self):
        return type(self).construct(*map(dom, self.content))

    def codomain(self):
        return type(self).construct(*map(cod, self.content))

    ### THE RULES ###
    @ConstructionRule.addRule
    def decompose(cls, content: tuple[Shape, ...]):
        if not any(map(instanceOf(Composed), content)):
            return None
        # Transpose the shapes
        composeds = map(Composed.get_content, content)
        codomains = map(cod, content)
        return Composed.corrected(
            starmap(
                cls.construct,
                zip_defaults(composeds, codomains),
            )
        )

    # rules: ClassVar = IterShape.rules + [decompose]


#######
@dataclass(repr=False, frozen=True)
class Cartesian(MonoidalProduct, name_convention=("(", ", ", ")")):
    pass

@dataclass(repr=False, frozen=True)
class Coproduct(MonoidalProduct, name_convention=("(", "|", ")")):
    pass


#######
@dataclass(repr=False, frozen=True)
class Composed(IterShape, name_convention=("", "--", "")):
    ### CORE ###
    @cached_property
    def identity(self):
        return False
    def domain(self):
        return self.content[0].dom
    def codomain(self):
        return self.content[-1].cod

    ### CONSTRUCTION ###
    @ConstructionRule.addRule
    def removeId(cls, content: tuple[Shape, ...]):
        if not any(map(identity, content)):
            return None
        # Remove the identities,
        if all(map(identity, content)):
            return content[0]
        return cls.construct(*filter(lambda x: not identity(x), content))

    ### NAMING ###
    def __str__(self):
        if self.content == ():
            return '~>'
        return '--'.join(map(str, self.content))
    def __repr__(self):
        if self.content == ():
            return '~>'
        return (
            "->".join([f"{x.dom}--{x}" for x in self.content])
            + f"->{self.content[-1].cod}"
        )
    def tile(self, mapping=str):
        if mapping is repr:
            return self.dom.tile(str) + sum(
                [x.tile(str) + x.cod.tile() for x in self.content], TextTile()
            )
        else:
            return sum([x.tile(str) for x in self.content], TextTile())


### HOMF


@dataclass(repr=False, frozen=True)
class GenericField(Shape):
    content: str = "V"


V = GenericField()


@dataclass(repr=False, frozen=True)
class Reversed(Shape):
    content: Shape

    ### CORE ###
    @cached_property
    def identity(self):
        return False

    def domain(self):
        return self.content.cod

    def codomain(self):
        return self.content.dom

    ### CONSTRUCTION ###
    @ConstructionRule.addRule
    def constructReversed(cls, content: Shape):
        match content:
            case Composed(xs):
                return Composed(*map(Reversed, reversed(xs)))
            case Reversed(x):
                return x
            case Shape(identity=True):
                return content


@dataclass(repr=False, frozen=True)
class Homf(Shape):
    content: Shape = field(default_factory=Cartesian)
    target: Shape = V
    right: Shape = field(default_factory=Cartesian)

    ### CORE ###
    @cached_property
    def identity(self):
        return all(map(identity, self))

    def domain(self):
        return Homf.construct(
            self.content.cod, self.target.dom, self.right.cod
        )

    def codomain(self):
        return Homf.construct(
            self.content.dom, self.target.cod, self.right.dom
        )

    ### CONSTRUCTION ###
    @ConstructionRule.addRule
    def homfConstruction(cls, content: Shape, target: Shape, right: Shape):
        match content, target, right:
            # Decompose
            case (Composed(), _, _) | (_, Composed(), _) | (_, _, Composed()):
                zipped = zip_defaults(
                    (
                        tuple(reversed(Composed.get_content(content))),
                        Composed.get_content(target),
                        tuple(reversed(Composed.get_content(right))),
                    ),
                    (dom(content), cod(target), dom(right)),
                )
                return Composed.construct(*starmap(Homf.construct, zipped))
            # Expand Products
            case (
                (Coproduct(), _, _)
                | (_, Cartesian(), _)
                | (_, _, Coproduct())
            ):
                products = itertools.product(
                    Coproduct.get_content(content),
                    Cartesian.get_content(target),
                    Coproduct.get_content(right),
                )
                return Cartesian.construct(*starmap(Homf.construct, products))
            # Reduce One
            case Cartesian([]), _, Cartesian([]):
                return target
            # Reduce Homf
            case [_, Homf(c2, t2, r2), _]:
                return Homf.construct(content + c2, t2, r2 + right)
            # Reduce Identity
            case (
                _,
                Shape(identity=True),
                _,
            ) if right != Cartesian():
                return Homf.construct(content + right, target)
        return None

    # rules: ClassVar[list[ConstructionRule]] = Shape.rules + [homfConstruction]

    def __str__(self):
        left = " ".join(map(str, Cartesian.get_content(self.content)))
        right = " ".join(map(str, Cartesian.get_content(self.right)))
        middle = (
            "→" * bool(left)
            + str(self.target) * (self.target != V)
            + "←" * bool(right)
        )
        return f"[{left}{middle}{right}]"


### CONF


class CompositionError(Exception):
    def __init__(self, *shapes: Shape) -> None:
        super().__init__("Could not compose " + " ".join(map(repr, shapes)))


@dataclass
class Functor:
    """content: Represents the endonats."""

    @traceErrors
    def __call__(self, target: Shape):
        match target:
            case Shape():
                return type(target).corrected(
                    *map(
                        lambda x: tuple(map(self, x))
                        if isinstance(x, tuple)
                        else self(x),
                        target,
                    )
                )
            case _:
                return target

@dataclass
class DictFunctor(Functor):
    mapping: dict[Shape, Shape] = field(default_factory=dict)

    def __call__(self, target: Shape):
        new_value = self.mapping.get(target, None)
        if new_value:
            return new_value
        return super().__call__(target)

def create_key():
    r = random.randint(0, 2**16)
    return r


import sys, gc
import collections


@dataclass(frozen=True, order=True)
class ConfKey:
    name: str = ""
    key: int | str = field(default_factory=create_key)
    registry: ClassVar[collections.Counter] = collections.Counter()

    # Add to registry
    def __post_init__(self):
        ConfKey.registry.update([self.name])

    def __repr__(self) -> str:
        count = ConfKey.registry[self.name]
        if count == 1:
            return bcolors.apply(self.name, bcolors.fg.YELLOW)
        count = 2 + len(f"{ConfKey.registry[self.name]:X}")
        return bcolors.apply(
            self.name + f".{self.key:X}"[:count], bcolors.fg.YELLOW
        )

    def __hash__(self):
        return hash(self.key)

    def __del__(self):
        ConfKey.registry.subtract([self.name])


@dataclass(frozen=True, repr=False)
class Conf(Shape):
    content: str = "*"
    key: ConfKey = field(default_factory=ConfKey)

    @ConstructionRule.addRule
    def constructConf(
        cls: type[Self],  # type: ignore
        content: str,
        key: ConfKey | str,
    ):
        if isinstance(key, str):
            return cls.corrected(content, ConfKey(key))

    def __str__(self):
        return f"{bcolors.apply(self.content, bcolors.fg.BLUE)}={self.key}"

    @classmethod
    def construct(cls, content: str = "*", key: ConfKey | None = None):
        if key is None:
            key = ConfKey(content)
        return cls.corrected(content, key)


@dataclass
class Configuration(Functor):
    # This is a dictionary configured so that,
    # We have a ConfKey pointing to a LARGER ConfKey
    # OR pointing to a shape.
    config: dict[ConfKey, ConfKey | Shape] = field(default_factory=dict)

    def align(self, first: Shape, second: Shape):
        match first, second:
            case Conf(key=key_x), Conf(key=key_y):
                if key_x < key_y:
                    self.config[key_x] = second
                if key_y < key_x:
                    self.config[key_y] = first
            case (Conf(key=key), _ as s) | (_ as s, Conf(key=key)):
                self.config[key] = s
            case _, _:
                if first != second:
                    raise CompositionError(first, second)
        return self

    def __call__(self, target: Shape | Any):
        if not isinstance(target, Conf):
            return super().__call__(target)
        content, key = target.content, target.key
        while True:
            next_key = self.config.get(key)
            match next_key:
                case Conf():
                    key = next_key.key
                case None:
                    return Conf(content, key)
                case Shape():
                    return next_key


@dataclass
class GetConfig(Functor):
    configs: set[ConfKey] = field(default_factory=set)

    def __call__(self, target: Shape | Any):
        if isinstance(target, Conf):
            self.configs.add(target.key)
        return super().__call__(target)


@dataclass(repr=False, frozen=True)
class Duplicate(Shape):
    content: Shape = field(default_factory=Homf)
    multiple: int = 1

    ### CORE
    @cached_property
    def identity(self) -> bool:
        return False

    def domain(self):
        return self.content

    def codomain(self):
        return Cartesian.corrected((self.content,) * self.multiple)

    @ConstructionRule.addRule
    def constructDelta(
        cls: type[Self],  # type: ignore
        content: Shape,
        multiple: int,
    ):
        if multiple == 1:
            return content
        if isinstance(content, Homf):
            ctr: tuple[Shape, Shape, Shape] = tuple(content)  # type: ignore
            left, target, right = ctr
            return (left + right) >> Duplicate.construct(multiple, target)

    @classmethod
    def construct(cls, multiple: int, content: ShapeConstructable = Homf()):
        content = shape(content)
        return cls.corrected(content, multiple)

    def __str__(self):
        return (
            f"Δ{self.multiple}" + (f"{self.content}"
            if self.content != V
            else "")
        )



@dataclass(repr=False, frozen=True)
class Para(Variant):
    @cached_property
    def load(self):
        return self._cod

    @cached_property
    def save(self):
        return self._dom

    def __str__(self):
        return bcolors.apply(
            str(self.content),
            bcolors.fg.MAGENTA,
            bcolors.BOLD,
            bcolors.UNDERLINE,
        )

@dataclass(repr=False, frozen=True)
class Load(Para):
    @classmethod
    def construct(cls, target: ShapeConstructable, name: str | None = None):
        target = shape(target)
        if not name:
            name = str(target)
        return cls.corrected(name, Cartesian(), target)

    def __str__(self):
        return f"{self.content}↳" + super().__str__()


@dataclass(repr=False, frozen=True)
class Save(Para):
    @classmethod
    def construct(cls, target: ShapeConstructable, name: str | None = None):
        target = shape(target)
        if not name:
            name = str(target)
        return cls.corrected(name, target, Cartesian())

    def __str__(self):
        return super().__str__() + f"⮧{self.content}"


@dataclass
class GetTape(Functor):
    load: Shape = Cartesian()
    save: Shape = Cartesian()

    def __call__(self, target: Shape):
        if isinstance(target, Para):
            self.load = self.load + target.load
            self.save = self.save + target.save
        return super().__call__(target)

@dataclass(repr=False, frozen=True)
class Del(Variant):
    @ConstructionRule.addRule
    def removeEmpty(cls, content: str, _dom: Shape, _cod: Shape):
        if _dom == Cartesian():
            return Cartesian()

    @classmethod
    def construct(cls, content: ShapeConstructable):
        content = shape(content)
        return cls.corrected(f"d{content}", content, Cartesian())


@dataclass(frozen=True, repr=False)
class Box(Shape):
    content: Shape
    name: str = "main"

    @cached_property
    def identity(self):
        return self.content.identity

    def domain(self):
        return self.content.dom

    def codomain(self):
        return self.content.cod

    def __str__(self):
        return f"{self.name}: {self.content}"

    def __repr__(self):
        return f"{self.name}: {self.content.__repr__()}"
    
from ncd.composition import compose