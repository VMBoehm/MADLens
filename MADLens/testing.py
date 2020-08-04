from vmad import Builder
from numpy.testing import assert_array_equal, assert_allclose

import numpy

class BaseScalarTest:
    """ Basic correctness of gradient against numerical with to_scalar """

    to_scalar = None # operator for norm-2 scalar

    x = numpy.arange(10)  # free variable x
    x_ = numpy.eye(10)    # a list of directions of x_ to check for directional gradients.

    y = sum(x ** 2)       # expected output variable y, scalar
                          # NotImplemented to bypass the value comparison
    epsilon = 1e-3
    rtol = 1e-6
    atol = 1e-10
    def inner(self, x, y):
        return numpy.sum(x * y)

    def model(self, x):
        return x          # override and build the model will be converted to a scalar later.

    def setup(self):
        with Builder() as m:
            x = m.input('x')
            x = self.model(x)
            # avoid using the bound function.
            y = type(self).to_scalar(x)
            m.output(y=y)

        self.m = m

        y_ = []
        for x_ in self.x_:
            # run a step along x_
            xl = self.x - x_ * (self.epsilon * 0.5)
            xr = self.x + x_ * (self.epsilon * 0.5)

            # numerical
            yl = self.m.compute(init=dict(x=xl), vout='y', return_tape=False)
            yr = self.m.compute(init=dict(x=xr), vout='y', return_tape=False)

            y_1 = (yr - yl) / self.epsilon

            y_.append(y_1)

        y, tape = self.m.compute(init=dict(x=self.x), vout='y', return_tape=True)
        self.tape = tape
        self.y_ = y_

    def test_opr(self):
        init = dict(x=self.x)
        y1 = self.m.compute(vout='y', init=init, return_tape=False)

        if self.y is not NotImplemented:
            # correctness
            assert_allclose(y1, self.y, rtol=self.rtol, atol=self.atol)

    def test_powerfulness(self):
        import numpy

        if numpy.allclose(self.y_, 0):
            raise AssertionError("The test case is not powerful enough, since all derivatives at this point are zeros")

    def test_jvp_finite(self):
        jvp = self.tape.get_jvp()

        for x_, y_ in zip(self.x_, self.y_):
            init = dict(x_=x_)
            y_1 = jvp.compute(init=init, vout='y_', return_tape=False)

            assert_allclose(y_1, y_, rtol=self.rtol, atol=self.atol)

    def test_vjp_finite(self):
        vjp = self.tape.get_vjp()

        init = dict(_y=1.0)
        _x = vjp.compute(init=init, vout='_x', return_tape=False)

        for x_, y_ in zip(self.x_, self.y_):
            assert_allclose(self.inner(_x, x_), y_, rtol=self.rtol, atol=self.atol)

class BaseVectorTest:
    """ Basic correctness of gradient against numerical for vector functions """

    x = numpy.arange(10)  # free variable x
    x_bases = None    # a list of directions of x_ to check for directional gradients.
                               # if None, generate based on the shape of x.
    y = x ** 2 # expected output variable y

    y_bases = None    # a list of directions of y_ to check for directional gradients.
                               # if None, generate based on the shape of y.


    epsilon = 1e-7

    rtol = 1e-5
    atol = 1e-10
    def allclose(self, x, y):  # measuring the
        assert numpy.shape(x) == numpy.shape(y)
        return numpy.allclose(x, y, rtol=self.rtol, atol=self.atol)

    def inner(self, a, b):
        return numpy.sum(a * b)

    def model(self, x):
        return x          # override and build the model will be converted to a scalar later.

    def _make_bases(self, x):
        x_ = []
        eye = numpy.eye(x.size)
        for row in eye:
            x_.append(row.reshape(x.shape).copy())
        return x_

    def _make_model(self):
        with Builder() as m:
            x = m.input('x')
            y = self.model(x)
            m.output(y=y)
        return m

    def _make_finite_jacobian(self):
        y_ = []
        for x_ in self.x_bases:
            # run a step along x_
            xl = self.x - x_ * (self.epsilon * 0.5)
            xr = self.x + x_ * (self.epsilon * 0.5)

            # numerical
            yl = self.m.compute(init=dict(x=xl), vout='y', return_tape=False)
            yr = self.m.compute(init=dict(x=xr), vout='y', return_tape=False)

            y_1 = (yr - yl) / self.epsilon

            y_.append(y_1)

        return y_

    def setup(self):

        if hasattr(self.y, '__call__'):
            self.y = self.y(self.x)

        self.m = self._make_model()

        if self.x_bases is None:
            self.x_bases = self._make_bases(self.x)

        if self.y_bases is None:
            self.y_bases = self._make_bases(self.y)

        self.finite_jvp = self._make_finite_jacobian()

        if numpy.allclose(self.finite_jvp, 0):
            raise AssertionError("The test case is not powerful enough, since all derivatives at this point are zeros")

        y, tape = self.m.compute(init=dict(x=self.x), vout='y', return_tape=True)

        self.tape = tape

        self._evaluated_y = y


    def test_opr(self):
        # if this is wrong, apl is wrong
        assert_allclose(self._evaluated_y, self.y, rtol=self.rtol, atol=self.atol)

    def test_jvp_finite(self):
        jvp = self.tape.get_jvp()

        for x_, y_ in zip(self.x_bases, self.finite_jvp):
            init = dict(x_=x_)
            y_1 = jvp.compute(init=init, vout='y_', return_tape=False)

            if not self.allclose(y_1, y_):
                raise AssertionError("jvp comparison failed")

    def test_vjp_finite(self):
        vjp = self.tape.get_vjp()

        for _y in self.y_bases:
            init = dict(_y=_y)
            _x = vjp.compute(init=init, vout='_x', return_tape=False)

            for x_, y_ in zip(self.x_bases, self.finite_jvp):
                v1 = self.inner(_y, y_)
                v2 = self.inner(_x, x_)
                if not self.allclose(v1, v2):
                    raise AssertionError("jvp comparison failed")

