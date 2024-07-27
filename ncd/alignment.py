from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Any, TypeVar, Protocol, Annotated, ClassVar, Literal
import functools
from ncd.display_text import bcolors
import random

def create_key():
    r = random.randint(0, 2**32)
    return r

@dataclass(frozen=True, order=True)
class Key:
    name: str = ''
    key:  int = field(default_factory=create_key)
    
    def __repr__(self):
        return bcolors.apply(
            self.name + f'.{self.key:X}'[:3], bcolors.fg.YELLOW
        )

class NoValueT:
    """ Only one instance of NoValue can exist. """
    universal = None

    def __str__(self):
        return 'NV'

    def __new__(cls):
        if NoValueT.universal is not None:
            return NoValueT.universal
        return super().__new__(cls)
    def __init__(self):
        if NoValueT.universal is None:
            NoValueT.universal = self
    def __bool__(self):
        raise ValueError('Taking the bool of NoValueT! Did you mean to check None-ness?')

    
NoValue = NoValueT()

def xor[T, D](x: T | D, y: T | D, default: D = NoValue):
    """ Returns whichever of the two is not the default value,
    or returns the default value. """
    if y != default and x != default:
        raise ValueError(x, y)
    return y if x == default else x

@dataclass(frozen=True)
class KeyPool[T]:
    keys: frozenset[Key]
    value: T | NoValueT = NoValue

    def __str__(self) -> str:
        return '{' + ', '.join(map(str,self.keys)) + '}: ' + f'{self.value}'

    def union(self, other: KeyPool[T] | None = None , *others: KeyPool[T]) -> KeyPool[T]:
        """ Combines a series of pools. """
        if other is None:
            return self
        value = xor(self.value, other.value, NoValue)
        keys = self.keys | other.keys
        new_pool = KeyPool(keys, value)
        return new_pool.union(*others)
    
    def __or__(self, other: KeyPool[T]):
        return self.union(other)

@dataclass(frozen=True)
class KeyContext[T]:
    pools: frozenset[KeyPool[T]] = field(default_factory=frozenset)

    def __str__(self) -> str:
        return '<'+', '.join(map(str, self.pools))+'>'

    def add(self, new: KeyPool[T]):
        disjoint = filter(lambda x: not bool(new.keys & x.keys), self.pools)
        new = new.union(*filter(lambda x: bool(new.keys & x.keys), self.pools))
        return KeyContext(frozenset([new, *disjoint]))

    def apply(self, key: Key) -> Key | T:
        for k in self.pools:
            if key in k.keys:
                return max(k.keys) if isinstance(k.value, NoValueT) else k.value
        raise KeyError(key)