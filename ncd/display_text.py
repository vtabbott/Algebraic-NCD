from __future__ import annotations
from dataclasses import dataclass
import itertools
from typing import Any


def lenc(target: str):
    target = bcolors.clear(target)
    return len(bcolors.clear(target))


###### COLORS
class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    class fg:
        BLACK = "\033[30m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        CYAN = "\033[36m"
        WHITE = "\033[37m"
        RESET = "\033[39m"

    class bg:
        BLACK = "\033[40m"
        RED = "\033[41m"
        GREEN = "\033[42m"
        YELLOW = "\033[43m"
        BLUE = "\033[44m"
        MAGENTA = "\033[45m"
        CYAN = "\033[46m"
        WHITE = "\033[47m"
        RESET = "\033[49m"

    class style:
        BRIGHT = "\033[1m"
        DIM = "\033[2m"
        NORMAL = "\033[22m"
        RESET_ALL = "\033[0m"

    @staticmethod
    def clear(target: Any):
        target = str(target)
        out = ""
        while target:
            where = target.find("\033")
            if where == -1:
                out += target
                return out
            out += target[:where]
            target = target[where:]
            where = target.find("m")
            target = target[1 + where :]
        return out

    @staticmethod
    def apply(target: Any, *mods: str) -> str:
        target = bcolors.clear(str(target))
        return "".join(map(str, mods)) + target + bcolors.ENDC


@dataclass(frozen=True)
class TextTile:
    content: str = ""
    bottom_pad: str = ""
    right_pad: str = " "

    def height(self):
        return len(self.content.split("\n"))

    def width(self):
        return max(map(lenc, self.content.split("\n")))

    def horizontalJoin(self, other: TextTile):
        return TextTile(
            "\n".join([
                u + v
                for u, v in zip(
                    self.lines(self.width(), other.height()),
                    other.lines(0, self.height()),
                )
            ]),
            other.bottom_pad,
            other.right_pad,
        )

    def lines(self, width=0, height=0):
        _lines = self.content.split("\n") + [self.bottom_pad] * (
            height - self.height()
        )

        _lines = [
            x
            + " " * (max(0, width - lenc(x)) % lenc(self.right_pad))
            + self.right_pad
            * (max(0, width - lenc(x)) // lenc(self.right_pad))
            for x in _lines
        ]

        return _lines

    def __str__(self):
        return str(self.content)

    def __add__(self, other: TextTile):
        return self.horizontalJoin(other)

    def __mul__(self, other: TextTile):
        return TextTile(self.content + "\n" + other.content)
