import numpy as np


class PlanckCosmo:

    def __init__(
        self,
        sigma8: float = 0.80,
        Omega_c: float = 0.25,
        Omega_b: float = 0.045,
        h: float = 0.67,
        ns: float = 0.965,
    ) -> None:

        self._Omega_c = Omega_c
        self._Omega_b = Omega_b
        self._h = h
        self._sigma8 = sigma8
        self._ns = ns

    @property
    def ombh2(self):
        return self._Omega_b * self._h**2

    @property
    def omch2(self):
        return self._Omega_c * self._h**2

    @property
    def Omega_b(self):
        return self._Omega_b

    @property
    def Omega_c(self):
        return self._Omega_c

    @property
    def Omega_m(self):
        return self._Omega_c + self._Omega_b

    @property
    def h(self):
        return self._h

    @property
    def H0(self):
        return self._h * 100

    @property
    def ns(self):
        return self._ns

    @property
    def sigma8(self):
        return self._sigma8
