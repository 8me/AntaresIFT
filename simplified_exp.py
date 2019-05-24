import nifty5 as ift
import numpy as np


class sexp(ift.Operator):
    def __init__(self, domain, y):
        self._domain = domain
        self._target = domain
        self._y = y

    def _sexp(self, x):
        retval = np.zeros(x.shape)
        mask = x > self._y
        # retval[mask] = x[mask] - self._y + 1
        retval[mask] = 0.5*(x[mask] - self._y + 1)**2
        retval[np.logical_not(mask)] = np.exp(x[np.logical_not(mask)] -
                                              self._y)
        return retval

    def _sexp_bar(self, x):
        retval = np.zeros(x.shape)
        mask = x > self._y
        # retval[mask] = 1
        retval[mask] = (x[mask] - self._y + 1)
        retval[np.logical_not(mask)] = np.exp(x[np.logical_not(mask)] -
                                              self._y)
        return retval

    def apply(self, x):
        self._check_input(x)
        lin = isinstance(x, ift.Linearization)
        v = x._val if lin else x

        if not lin:
            sex_field = ift.Field(self._domain, self._sexp(v.val))
            return sex_field
        else:
            sex_field = ift.Field(self._domain, self._sexp(v.val))
            dop = ift.makeOp(ift.Field(self._domain,
                                       self._sexp_bar(v.val)))(x.jac)
            return x.new(sex_field, dop)
