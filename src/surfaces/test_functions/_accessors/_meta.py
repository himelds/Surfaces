"""MetaAccessor: read-only proxy to class-level metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .._function_spec import MetaSpec

if TYPE_CHECKING:
    from .._base_test_function import BaseTestFunction


class MetaAccessor:
    """Read-only proxy to class-level metadata attributes.

    Parameters
    ----------
    func : BaseTestFunction
        The test function instance.
    """

    def __init__(self, func: "BaseTestFunction") -> None:
        self._func = func

    @property
    def _meta(self) -> MetaSpec:
        meta = getattr(type(self._func), "_meta", MetaSpec())
        if isinstance(meta, MetaSpec):
            return meta
        return MetaSpec(**meta)

    def as_dict(self) -> dict:
        """Return the resolved metadata as a plain dict."""
        return self._meta.as_dict()

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-compatible get."""
        return self.as_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        data = self.as_dict()
        if key not in data:
            raise KeyError(key)
        return data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.as_dict()

    @property
    def name(self) -> Optional[str]:
        return self._meta.name

    @property
    def slug(self) -> Optional[str]:
        return self._meta.slug

    @property
    def _name_(self) -> Optional[str]:
        return self._meta.slug

    @property
    def latex_formula(self) -> Optional[str]:
        return self._meta.latex_formula

    @property
    def pgfmath_formula(self) -> Optional[str]:
        return self._meta.pgfmath_formula

    @property
    def reference(self) -> Optional[str]:
        return self._meta.reference

    @property
    def reference_url(self) -> Optional[str]:
        return self._meta.reference_url

    @property
    def tagline(self) -> Optional[str]:
        return self._meta.tagline

    @property
    def display_bounds(self) -> Any:
        return self._meta.display_bounds

    @property
    def display_projection(self) -> Optional[dict[str, Any]]:
        return self._meta.display_projection

    @property
    def category(self) -> Optional[str]:
        return self._meta.category

    @property
    def domain(self) -> Optional[str]:
        return self._meta.domain

    @property
    def func_id(self) -> Optional[int]:
        return self._func.spec.func_id

    def __repr__(self) -> str:
        return f"MetaAccessor(name={self.name!r})"
