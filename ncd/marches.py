from __future__ import annotations
from dataclasses import dataclass, field
from itertools import repeat
from typing import Callable
from ncd import (
    Shape,
    Variant,
    Cartesian,
    Homf,
    Conf,
    V,
    Composed,
    Duplicate,
    GetConfig,
    display_text
)
from .nn import Linear, Einops, SoftMax, Addition

@dataclass
class March:
    content: list[Shape] = field(default_factory=list)
    source: list[March] = field(default_factory=list)
    target: list[March] = field(default_factory=list)
    input_var: list[str] = field(default_factory=list)
    output_var: list[str] = field(default_factory=list)

    def link(self, other: March):
        self.target.append(other)
        other.source.append(self)

    def refine(self):
        if len(self.source) == 1 and len(self.source[0].target) == 1:
            source = self.source[0]
            for x in source.source:
                for i, x_s in enumerate(x.target):
                    if x_s is source:
                        x.target[i] = self
            self.source = source.source
            self.content = source.content + self.content
        for x in self.source:
            x.refine()

    def __repr__(self) -> str:
        rep = (
            "Content:"
            + str(self.output_var)
            + "@".join(map(str, self.content))
            + str(self.input_var)
        )
        for march in self.source:
            rep += "\n" + "\n".join([
                "\t" + x for x in repr(march).split("\n")
            ])
        return rep

    def __str__(self):
        return "@".join(map(str, self.content)) if self.content else "[]"

    def get_initial(self):
        if len(self.source) > 0:
            return self.source[0].get_initial()
        else:
            return self


def cart_length(target: Shape):
    return len(Cartesian.get_content(target))


# Progresses through a shape, generating a web of connected marches.
def progress_frontier(
    target: Shape, frontier: list[March] | None = None
) -> list[March]:
    if frontier is None:
        # Start with an empty march.
        initial = March()
        size = cart_length(target.dom)
        frontier = list(repeat(initial, size))
    # Ensure sizes match.
    if not len(frontier) == cart_length(target.dom):
        print(len(frontier), cart_length(target.dom))
        assert False
    # Return empty.
    if not frontier:
        return frontier
    match target:
        case Composed([x, *xs]):
            return progress_frontier(
                Composed(*xs), progress_frontier(x, frontier)
            )
        case Cartesian([x, *xs]):
            size = cart_length(x.dom)
            return progress_frontier(x, frontier[:size]) + progress_frontier(
                Cartesian(*xs), frontier[size:]
            )
        case Shape(identity=True):
            return frontier
        case Shape():
            next_march = March([target])
            for x in frontier:
                x.link(next_march)
            return [next_march] * cart_length(target.cod)
        case _:
            raise ValueError("Must march through a shape!")


def cap_frontier(frontier: list[March]):
    final = March()
    for f in frontier:
        f.link(final)
    return final


# Depth-first variable assignment.
def assign_var(final: March):
    # frontier = frontier.copy()
    varlist = iter("abcdefghijklmnopqrstuv")
    # Get a variable
    current_var = next(varlist)
    # Source -> Target
    initial = March()
    sources_added = [initial]
    # final = March(source=frontier)
    links = [(s, final) for s in final.source]
    # links = [(s, final) for s in frontier]
    while len(links) > 0:
        # Get the last added link.
        # Source / Target
        s, t = links.pop(0)
        # Append the variable - this connects them.
        # The order should be correct as we are doing depth-first
        # from the top to the bottom, and we have no cross-overs
        s.output_var.append(current_var)
        t.input_var.append(current_var)
        if not s.source and s is not initial:
            initial.link(s)
        if s not in sources_added and s.source:
            links = [(ss, s) for ss in s.source] + links
            sources_added.append(s)
        else:
            current_var = next(varlist)
    return initial


TripletList = list[tuple[list[str], list[Shape], list[str]]]


# Breadth first code creation - ensures order is maintained.
def code_from_initial(initial: March) -> TripletList:
    marches = initial.target
    covered = [initial]
    output = []
    while len(marches) > 0:
        march = marches.pop(0)
        # march = marches.pop(0)
        output.append((march.output_var, march.content, march.input_var))
        covered.append(march)
        additional = [
            x for x in march.target if all([y in covered for y in x.source])
        ]
        marches += additional
    return output


def code_generate(target: Shape):
    fr = progress_frontier(target)
    [f.refine() for f in fr]
    initial = assign_var(cap_frontier(fr))
    return code_from_initial(initial)


def torch_init(target: Shape, dim=0) -> tuple[str, str] | None:
    # input length
    dom_size = cart_length(target.dom)
    cod_size = cart_length(target.cod)
    # holders for input/outputs
    dom: str = ", ".join(repeat("{}", dom_size))
    cod: str = ", ".join(repeat("{}", cod_size))

    def var_expr(expr: Shape) -> str:
        match expr:
            case Homf():
                return (
                    "("
                    + ", ".join(
                        map(var_expr, Cartesian.get_content(expr.content))
                    )
                    + ")"
                )
            case Conf():
                return expr.key.name
            case _:
                return str(expr)

    match target:
        case Homf(target=content, right=right):
            dim = -cart_length(right)
            return torch_init(content, dim)
        case Linear():
            return (
                f"self.L{target.content}",
                f"Multilinear({var_expr(target.dom)},{var_expr(target.cod)})",
            )
        case _:
            return None


def torch_code(target: Shape, left_dim=0, right_dim=0) -> tuple[str, str] | None:
    # input length
    dom_size = cart_length(target.dom)
    cod_size = cart_length(target.cod)
    # holders for input/outputs
    dom: str = ", ".join(repeat("{}", dom_size))
    cod: str = ", ".join(repeat("{}", cod_size))

    match target:
        case Homf(content=left, target=content, right=right):
            left_dim = cart_length(left)
            right_dim = cart_length(right)
            return torch_code(content, left_dim, right_dim)
        case Addition():
            return cod, "{} + {}"
        case SoftMax():
            return (
                cod,
                f'torch.softmax({dom}, ' + f'dim=-{right_dim+1})',
            )
        case Duplicate(multiple=n):
            return cod, "{0}, " * (n - 1) + "{0}"
        case Einops(content):
            return cod, f'einops.einsum({dom}, "{content}")'
        case Linear():
            return cod, f"L{target.content}({dom})"
        case _:
            return cod, f"{target}({dom})"


def to_code(
    ycx: TripletList,
    mapper: Callable[[Shape], tuple[str, str] | None] = torch_code,
) -> str:
    lines = []
    for y, content, x in ycx:
        for c in content:
            cd = mapper(c)
            if cd is not None:
                cod, dom = cd
                lines.append([cod.format(*y), dom.format(*x)])
    return "\n".join([l + " = " + r for l, r in lines])


def to_torch(target: Shape, name: str = "GeneratedNN"):
    config = GetConfig()
    config(target)

    name = name.replace(" ", "_")

    space = "    "

    # Get the frontier
    fr = progress_frontier(target)
    [f.refine() for f in fr]
    # Assign the variables (depth first)\
    final = cap_frontier(fr)
    initial = assign_var(final)

    # Get the code in order (breadth first)
    code_framework = code_from_initial(initial)

    class_header = f"class {name}(nn.Module):"

    # Get the init block
    init_header = f'{space}def __init__(self, ' \
        + f'{', '.join([x.name for x in config.configs])}):' \
        + f'\n{space}{space}super().__init__()'
    init_content = "\n".join([
        f"{space}{space}{x}"
        for x in to_code(code_framework, torch_init).split("\n")
    ])

    # Get the forward block
    fwd_header = f'{space}def forward(self, ' \
        +f'{', '.join([x for x in initial.output_var])}):'
    fwd_content = (
        "\n".join([
            f"{space}{space}{x}"
            for x in to_code(code_framework, torch_code).split("\n")
        ])
        + f'\n{space}{space}return {', '.join(final.input_var)}'
    )

    # Return the text.
    return display_text.bcolors.clear("\n".join([
        class_header,
        init_header,
        init_content,
        fwd_header,
        fwd_content,
    ]))
