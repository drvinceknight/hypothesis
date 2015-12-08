# coding=utf-8
#
# This file is part of Hypothesis (https://github.com/DRMacIver/hypothesis)
#
# Most of this work is copyright (C) 2013-2015 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others. See
# https://github.com/DRMacIver/hypothesis/blob/master/CONTRIBUTING.rst for a
# full list of people who may hold copyright, and consult the git log if you
# need to determine who owns an individual contribution.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.
#
# END HEADER

from __future__ import division, print_function, absolute_import

import sys
import struct
from collections import namedtuple

import hypothesis.internal.conjecture.utils as d
from hypothesis.internal.floats import sign, float_to_int, int_to_float
from hypothesis.searchstrategy.misc import SampledFromStrategy
from hypothesis.searchstrategy.strategies import SearchStrategy, \
    MappedSearchStrategy


class IntStrategy(SearchStrategy):

    """A generic strategy for integer types that provides the basic methods
    other than produce.

    Subclasses should provide the produce method.

    """


class IntegersFromStrategy(SearchStrategy):

    def __init__(self, lower_bound, average_size=100000.0):
        super(IntegersFromStrategy, self).__init__()
        self.lower_bound = lower_bound
        self.average_size = average_size

    def __repr__(self):
        return u'IntegersFromStrategy(%d)' % (self.lower_bound,)

    def do_draw(self, data):
        return self.lower_bound + d.geometric(data, 1.0 / self.average_size)


class RandomGeometricIntStrategy(IntStrategy):

    """A strategy that produces integers whose magnitudes are a geometric
    distribution and whose sign is randomized with some probability.

    It will tend to be biased towards mostly negative or mostly
    positive, and the size of the integers tends to be biased towards
    the small.

    """

    def __repr__(self):
        return u'RandomGeometricIntStrategy()'

    def do_draw(self, data):
        return d.n_byte_signed(data, d.integer_range(data, 0, 8))


class WideRangeIntStrategy(IntStrategy):
    Parameter = namedtuple(
        u'Parameter',
        (u'center', u'width'),
    )

    def __repr__(self):
        return u'WideRangeIntStrategy()'

    def do_draw(self, data):
        return d.n_byte_signed(data, 20)


class BoundedIntStrategy(SearchStrategy):

    """A strategy for providing integers in some interval with inclusive
    endpoints."""

    def __init__(self, start, end):
        SearchStrategy.__init__(self)
        self.start = start
        self.end = end
        if start > end:
            raise ValueError(u'Invalid range [%d, %d]' % (start, end))

    def __repr__(self):
        return u'BoundedIntStrategy(%d, %d)' % (self.start, self.end)

    def do_draw(self, data):
        return d.integer_range(data, self.start, self.end)


def is_integral(value):
    try:
        return int(value) == value
    except (OverflowError, ValueError):
        return False

NASTY_FLOATS = [
    0.0, 0.5, 1.0 / 3, 10e6, 10e-6, 1.175494351e-38, 2.2250738585072014e-308,
    1.7976931348623157e+308, 3.402823466e+38, 9007199254740992, 1 - 10e-6,
    2 + 10e-6, 1.192092896e-07, 2.2204460492503131e-016,
    float('inf'), float('nan'),
]
NASTY_FLOATS.extend([-x for x in NASTY_FLOATS])
assert len(NASTY_FLOATS) == 32
INFINITY = float('inf')


class FloatStrategy(SearchStrategy):

    """Generic superclass for strategies which produce floats."""

    def __init__(self):
        SearchStrategy.__init__(self)
        self.int_strategy = RandomGeometricIntStrategy()

    def __repr__(self):
        return u'%s()' % (self.__class__.__name__,)

    def do_draw(self, data):
        def draw_float_bytes(random, n):
            assert n == 8
            i = random.randint(1, 10)
            if i == 1:
                f = random.choice(NASTY_FLOATS)
            elif i <= 3:
                f = random.random() * 2 - 1.0
            else:
                return bytes(random.randint(0, 255) for _ in range(8))
            return struct.pack(b'!d', f)
        return struct.unpack(b'!d', data.draw_bytes(8, draw_float_bytes))[0]


def compose_float(sign, exponent, fraction):
    as_long = (sign << 63) | (exponent << 52) | fraction
    return struct.unpack(b'!d', struct.pack(b'!Q', as_long))[0]


class FullRangeFloats(FloatStrategy):

    Parameter = namedtuple(
        u'Parameter',
        (u'negative_probability', u'subnormal_probability')
    )

    def __init__(self, allow_nan=True, allow_infinity=True):
        super(FullRangeFloats, self).__init__()
        self.allow_nan = allow_nan
        self.allow_infinity = allow_infinity


class FixedBoundedFloatStrategy(FloatStrategy):

    """A strategy for floats distributed between two endpoints.

    The conditional distribution tries to produce values clustered
    closer to one of the ends.

    """
    Parameter = namedtuple(
        u'Parameter',
        (u'cut', u'leftwards')
    )

    def __init__(self, lower_bound, upper_bound):
        FloatStrategy.__init__(self)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        assert upper_bound >= lower_bound
        self.intervals = []
        if sign(upper_bound) > 0:
            if sign(lower_bound) > 0:
                self.intervals.append((
                    float_to_int(self.lower_bound),
                    float_to_int(self.upper_bound)))
            else:
                self.intervals.append((
                    float_to_int(0.0),
                    float_to_int(self.upper_bound)))
                self.intervals.append((
                    float_to_int(-0.0),
                    float_to_int(self.lower_bound)))
        else:
            self.intervals.append((
                float_to_int(self.upper_bound),
                float_to_int(self.lower_bound)))

    def __repr__(self):
        return u'FixedBoundedFloatStrategy(%s, %s)' % (
            self.lower_bound, self.upper_bound,
        )

    def do_draw(self, data):
        interval = d.choice(data, self.intervals)
        return int_to_float(
            d.integer_range(data, *interval)
        )


class NastyFloats(SampledFromStrategy):

    def __init__(self, allow_nan=True, allow_infinity=True):
        elements = [
            0.0,
            -0.0,
            sys.float_info.min,
            -sys.float_info.min,
            -sys.float_info.max,
            sys.float_info.max
        ]
        if allow_infinity:
            elements.extend([
                float(u'inf'),
                -float(u'inf')
            ])
        if allow_nan:
            elements.extend([
                float(u'nan')
            ])

        SampledFromStrategy.__init__(self, elements=elements)


class ComplexStrategy(MappedSearchStrategy):

    """A strategy over complex numbers, with real and imaginary values
    distributed according to some provided strategy for floating point
    numbers."""

    def __repr__(self):
        return u'ComplexStrategy()'

    def pack(self, value):
        return complex(*value)
