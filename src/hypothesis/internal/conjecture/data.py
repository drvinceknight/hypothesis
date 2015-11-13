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

from enum import IntEnum

from hypothesis.errors import Frozen


class Status(IntEnum):
    OVERRUN = 0
    INVALID = 1
    VALID = 2
    INTERESTING = 3


class StopTest(BaseException):

    def __init__(self, data):
        super(StopTest, self).__init__()
        self.data = data


class TestData(object):

    def __init__(self, buffer: bytes):
        assert isinstance(buffer, bytes)
        self.buffer = buffer
        self.index = 0
        self.status = Status.VALID
        self.frozen = False
        self.intervals = []
        self.interval_stack = []
        self.cost = 0

    def __assert_not_frozen(self, name):
        if self.frozen:
            raise Frozen(
                'Cannot call %s on frozen TestData' % (
                    name,))

    def draw(self, strategy):
        self.start_example()
        result = strategy.do_draw(self)
        self.stop_example()
        return result

    def start_example(self):
        self.__assert_not_frozen('start_example')
        self.interval_stack.append(self.index)

    def stop_example(self):
        self.__assert_not_frozen('stop_example')
        k = self.interval_stack.pop()
        if k != self.index:
            t = (k, self.index)
            if not self.intervals or self.intervals[-1] != t:
                self.intervals.append(t)

    def incur_cost(self, cost):
        self.__assert_not_frozen('incur_cost')
        assert not self.frozen
        self.cost += cost

    def freeze(self):
        if self.frozen:
            return
        self.frozen = True
        # Intervals are sorted as longest first, then by interval start.
        self.intervals.sort(
            key=lambda se: (se[1] - se[0], se[0])
        )
        if self.status == Status.INTERESTING:
            self.buffer = self.buffer[:self.index]

    def draw_bytes(self, n: int) ->bytes:
        self.__assert_not_frozen('draw_bytes')
        self.index += n
        if self.index > len(self.buffer):
            self.status = Status.OVERRUN
            self.freeze()
            raise StopTest(self)
        self.intervals.append((self.index - n, self.index))
        result = self.buffer[self.index - n:self.index]
        assert len(result) == n
        return result

    def mark_interesting(self):
        self.__assert_not_frozen('mark_interesting')
        if self.status == Status.VALID:
            self.status = Status.INTERESTING
        raise StopTest(self)

    def mark_invalid(self):
        self.__assert_not_frozen('mark_invalid')
        if self.status != Status.OVERRUN:
            self.status = Status.INVALID
        raise StopTest(self)

    @property
    def rejected(self):
        return self.status == Status.INVALID or self.status == Status.OVERRUN
