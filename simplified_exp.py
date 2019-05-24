import nifty5 as ift

class _OpProd(ift.Operator):
    def __init__(self, domain):
        self._domain = domain
        self._target = domain

    def apply(self, x):
        self._check_input(x)
        lin = isinstance(x, ift.Linearization)
        v = x._val if lin else x

        if not lin:
            return
        wm = x.want_metric
        lin1 = self._op1(Linearization.make_var(v1, wm))
        lin2 = self._op2(Linearization.make_var(v2, wm))
        op = (makeOp(lin1._val)(lin2._jac))._myadd(
            makeOp(lin2._val)(lin1._jac), False)
        return lin1.new(lin1._val*lin2._val, op(x.jac))
