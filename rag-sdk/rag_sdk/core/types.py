from collections.abc import Awaitable, Callable
from typing import ParamSpec, Protocol, TypeAlias, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

AsyncCallable: TypeAlias = Callable[P, Awaitable[R]]  # type: ignore[valid-type]


class EncodeDecorator(Protocol[P, R]):  # type: ignore[misc]
    def __call__(self, fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]: ...
