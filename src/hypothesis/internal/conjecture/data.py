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
from hypothesis.internal.compat import text_type, binary_type
from hypothesis.errors import Frozen

# We want to determine whether given two outputs of the same length one of them
# is "better" than the other for the purpose of simplification. Bytewise order
# is not a good way to do this because it would e.g. put control characters
# first. Once we've done that, we might as well start reordering stuff
# according to my personal whims and prejudices. So this is the "canonical"
# order of ASCII in conjecture land according to Word of God. Fight me.
CHR_ORDER = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g',
    'H', 'h', 'I', 'i', 'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n',
    'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's', 'T', 't', 'U', 'u',
    'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z',
    ' ',
    '_', '-', '=', '~',
    '"', "'",
    ':', ';', ',', '.', '?', '!',
    '(', ')', '{', '}', '[', ']', '<', '>',
    '*', '+', '/', '&', '|', '%',
    '#', '$', '@',  '\\', '^', '`',
    '\t', '\n', '\r',
    '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07', '\x08',
    '\x0b', '\x0c', '\x0e', '\x0f', '\x10', '\x11', '\x12', '\x13', '\x14',
    '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a', '\x1b', '\x1c', '\x1d',
    '\x1e', '\x1f',
]

TEXT_BYTE_ORDER = [0] * len(CHR_ORDER)
for i, c in enumerate(CHR_ORDER):
    TEXT_BYTE_ORDER[ord(c)] = i
TEXT_BYTE_ORDER.extend(range(127, 256))
assert len(TEXT_BYTE_ORDER) == 256
assert sorted(TEXT_BYTE_ORDER) == list(range(256))


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

    def __init__(self, buffer, generate_up_to=-1, random=None):
        assert isinstance(buffer, bytes)
        self.buffer = buffer
        self.output = bytearray()
        self.index = 0
        self.status = Status.VALID
        self.frozen = False
        self.intervals = []
        self.interval_stack = []
        self.costs = [0] * (max(generate_up_to, len(self.buffer)) + 1)
        self.random = random
        self.generate_up_to = generate_up_to
        if self.random is not None:
            self.duplication_rate = random.random()
        else:
            self.duplication_rate = 0.0
        self.words = {}
        self.start_example()

    def __assert_not_frozen(self, name):
        if self.frozen:
            raise Frozen(
                'Cannot call %s on frozen TestData' % (
                    name,))

    def note(self, value):
        self.__assert_not_frozen("note")
        if not isinstance(value, (text_type, binary_type)):
            value = repr(value)
        if isinstance(value, text_type):
            value = value.encode('utf-8')
        assert isinstance(value, binary_type)
        self.output.extend(value)

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
        self.costs[self.index] += cost

    def freeze(self):
        if self.frozen:
            return
        self.stop_example()
        self.frozen = True
        # Intervals are sorted as longest first, then by interval start.
        self.intervals.sort(
            key=lambda se: (se[0] - se[1], se[0])
        )
        if self.status == Status.INTERESTING:
            self.buffer = self.buffer[:self.index]

    def draw_bytes(self, n: int) ->bytes:
        self.__assert_not_frozen('draw_bytes')
        self.index += n
        if self.index > len(self.buffer):
            if self.index <= self.generate_up_to:
                if self.index - n < len(self.buffer):
                    k = self.index - len(self.buffer)
                    self.buffer += self.random.getrandbits(k * 8).to_bytes(
                        k, 'big')
                else:
                    # We might want to fetch a previous one here.
                    if (
                        n in self.words and
                        self.random.random() <= self.duplication_rate
                    ):
                        self.buffer += self.random.choice(self.words[n])
                    else:
                        self.buffer += self.random.getrandbits(n * 8).to_bytes(
                            n, 'big')
            else:
                self.status = Status.OVERRUN
                self.freeze()
                raise StopTest(self)
        self.intervals.append((self.index - n, self.index))
        result = self.buffer[self.index - n:self.index]
        assert len(result) == n
        if self.generate_up_to > 0:
            self.words.setdefault(n, []).append(result)
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

    def better_than(self, other):
        return (
            self.interest_key1() < other.interest_key1() and
            self.interest_key2() <= other.interest_key2()
        )

    def interest_key1(self):
        return (
            len(self.buffer), self.buffer,
        )

    def interest_key2(self):
        return (
            self.costs,
            len(self.output), [TEXT_BYTE_ORDER[c] for c in self.output],
        )
