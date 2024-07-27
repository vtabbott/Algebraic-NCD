from ncd import (
    Shape,
    Configuration,
    Cartesian,
    Composed,
    Homf,
    Coproduct,
    Conf,
    Duplicate,
    Del,
    CompositionError,
    Identity
)
import typing
import dataclasses
import itertools



def nat_try_compose(
    first: Identity, second: Identity, conf: Configuration | None = None
) -> tuple[Configuration, Identity, Shape]:
    """
    We compose A @ B by generating functors C, T>>, and a natural transform
    N such that C(A) @ C(N) @ C(T >> B) composes.

    Returns:
        tuple[Configuration, Identity, Shape]:
            - C: A configuration functor aligning configurables.
            - T: An identity shape, which lifts the latter.
            - N: A natural transform shape, which links the two expressions.
    """
    if conf is None:
        conf = Configuration()
    first = conf(first)
    second = conf(second)
    # They might already match.
    if first == second:
        return conf, Cartesian(), first
    # Otherwise, we find C, T, N.
    match first, second:
        case Cartesian([x, *xs]), Cartesian([y, *ys]):
            # They need to be of equal length.
            assert len(xs) == len(ys)
            # We feed the same configuration through.
            conf, tens, natx = nat_try_compose(x, y, conf)
            conf, new_tens, natxs = nat_try_compose(
                Cartesian(*xs), tens >> Cartesian(*ys), conf
            )
            assert new_tens == Cartesian()
            return conf, tens, natx + natxs
        case Shape(), Cartesian(ys):
            nat = Duplicate(len(ys), first)
            conf, tens, nats = nat_try_compose(nat.cod, second, conf)
            return conf, tens, Composed(nat, nats)
        case Cartesian(xs), Shape():
            conf, tens, nats = nat_try_compose(
                first, Cartesian(*itertools.repeat(second, len(xs))), conf
            )
            return conf, Coproduct(*((tens,)*len(xs))), nats
        case (Conf(), _) | (_, Conf()):
            return conf.align(first, second), Cartesian(), first
        case Homf(xs, a), Homf(ys, b):
            xs = Cartesian.get_content(xs)
            ys = Cartesian.get_content(ys)
            conf.align(a, b)
            for x, y in zip(xs[-len(ys):],ys[-len(xs):]):
                conf.align(x, y)
            return conf, Cartesian(*xs[:-len(ys)]), Del(Cartesian(*ys[:-len(xs)])) >> first
        case Homf(xs, a), b:
            conf.align(a, b)
            return conf, xs, first
        case a, Homf(ys, b):
            conf.align(a, b)
            return conf, Cartesian(), Del(ys) >> first
    raise CompositionError(first, second)

def compose(first: Shape, second: Shape):
    try:
        conf, tens, nats = nat_try_compose(first.cod, second.dom)
    except Exception as e:
        print(f"Could not compose {first}, {second}")
        raise e
    return Composed(conf(first), conf(nats), conf(tens >> second))

if __name__ == '__main__':
    pass
    # Run some tests!
    # print(nat_try_compose(shape('*a, *b, a'), shape('a, a, a')))
    # print(nat_try_compose(shape('a'), shape('*x, a, a')))
    # print(nat_try_compose(shape('a, a, a'), shape('a')))
    # # Try configurations
    # print(nat_try_compose(shape('*a'), shape('x b')))
    # print(nat_try_compose(shape('x *a'), shape('x b')))
    # print(nat_try_compose(shape('b'), shape('x^b')))