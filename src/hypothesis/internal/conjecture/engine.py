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

import time
from random import Random

from hypothesis.settings import Settings
from hypothesis.reporting import debug_report
from hypothesis.internal.conjecture.data import Status, StopTest, TestData


class RunIsComplete(Exception):
    pass


class TestRunner(object):

    def __init__(
        self, test_function, settings, random=None
    ):
        self._test_function = test_function
        self.settings = settings or Settings()
        self.last_data = None
        self.changed = 0
        self.shrinks = 0
        self.examples_considered = 0
        self.valid_examples = 0
        self.start_time = time.time()
        self.random = random or Random()

    def new_buffer(self):
        buffer = self.rand_bytes(self.settings.buffer_size)
        self.last_data = TestData(buffer)
        self.test_function(self.last_data)
        self.last_data.freeze()

    def test_function(self, data):
        try:
            self._test_function(data)
        except StopTest as e:
            if e.data is not data:
                raise e

    def consider_new_test_data(self, data):
        # Transition rules:
        #   1. Transition cannot decrease the status
        #   2. Any transition which increases the status is valid
        #   3. If the previous status was interesting, only shrinking
        #      transitions are allowed.
        if self.last_data.status < data.status:
            return True
        if self.last_data.status > data.status:
            return False
        if data.status == Status.INVALID:
            return data.index >= self.last_data.index
        if data.status == Status.OVERRUN:
            return data.index <= self.last_data.index
        if data.status == Status.INTERESTING:
            assert len(data.buffer) <= len(self.last_data.buffer)
            if len(data.buffer) == len(self.last_data.buffer):
                assert data.buffer < self.last_data.buffer
            return data.better_than(self.last_data)
        return True

    def incorporate_new_buffer(self, buffer):
        if (
            self.settings.timeout > 0 and
            time.time() >= self.start_time + self.settings.timeout
        ):
            raise RunIsComplete()
        self.examples_considered += 1
        if (
            buffer[:self.last_data.index] ==
            self.last_data.buffer[:self.last_data.index]
        ):
            return False
        data = TestData(buffer)
        self.test_function(data)
        data.freeze()
        if data.status >= self.last_data.status:
            debug_report('%r -> %r, %s' % (
                data.buffer[:data.index], data.status,
                data.output.decode('utf-8'),
            ))
        if data.status >= Status.VALID:
            self.valid_examples += 1
        if self.consider_new_test_data(data):
            if self.last_data.status == Status.INTERESTING:
                self.shrinks += 1
            self.last_data = data
            self.changed += 1
            if self.shrinks >= self.settings.max_shrinks:
                raise RunIsComplete()
            return True
        return False

    def run(self):
        try:
            self._run()
        except RunIsComplete:
            pass

    def _run(self):
        self.new_buffer()
        mutations = 0
        while self.last_data.status != Status.INTERESTING:
            if (
                self.valid_examples >= self.settings.max_examples or
                self.examples_considered >= self.settings.max_iterations
            ):
                return
            if mutations >= self.settings.max_mutations:
                mutations = 0
                self.new_buffer()
            else:
                self.incorporate_new_buffer(
                    self.mutate_data_to_new_buffer()
                )
            mutations += 1

        for c in range(256):
            if self.incorporate_new_buffer(bytes(
                min(c, b) for b in self.last_data.buffer
            )):
                break

        initial_changes = self.changed
        change_counter = -1
        while (
            initial_changes + self.settings.max_shrinks >=
            self.changed > change_counter
        ):
            assert self.last_data.status == Status.INTERESTING
            change_counter = self.changed
            interval_change_counter = -1
            while self.changed > interval_change_counter:
                interval_change_counter = self.changed
                i = 0
                while i < len(self.last_data.intervals):
                    u, v = self.last_data.intervals[i]
                    if not self.incorporate_new_buffer(
                        self.last_data.buffer[:u] +
                        self.last_data.buffer[v:]
                    ):
                        i += 1
            i = 0
            while i < len(self.last_data.intervals):
                u, v = self.last_data.intervals[i]
                self.incorporate_new_buffer(
                    self.last_data.buffer[:u] +
                    bytes(sorted(self.last_data.buffer[u:v])) +
                    self.last_data.buffer[v:]
                )
                i += 1

            local_changes = -1
            while local_changes < self.changed:
                local_changes = self.changed
                for c in range(1, 256):
                    buf = self.last_data.buffer
                    if buf.count(c) > 1:
                        if self.incorporate_new_buffer(bytes(
                            c - 1 if b == c else b
                            for b in buf
                        )):
                            buf = self.last_data.buffer
                            for d in range(c):
                                if self.incorporate_new_buffer(bytes(
                                    d if b == c - 1 else b
                                    for b in buf
                                )):
                                    break
            k = 8
            for i in range(len(self.last_data.buffer) - k):
                buf = self.last_data.buffer
                if i + k > len(buf):
                    break
                self.incorporate_new_buffer(
                    buf[:i] + bytes(k) + buf[i + k:]
                )
            i = 0
            while i < len(self.last_data.buffer):
                buf = self.last_data.buffer
                if not self.incorporate_new_buffer(
                    buf[:i] + buf[i + 1:]
                ):
                    for c in range(buf[i]):
                        if self.incorporate_new_buffer(
                            buf[:i] + bytes([c]) + buf[i + 1:]
                        ):
                            break
                        elif self.incorporate_new_buffer(
                            buf[:i] + bytes([c]) + self.rand_bytes((
                                len(buf) - i - 1))
                        ):
                            break
                i += 1
            i = 0
            while i + 1 < len(self.last_data.buffer):
                j = i + 1
                buf = self.last_data.buffer
                if buf[i] > buf[j]:
                    self.incorporate_new_buffer(
                        buf[:i] + bytes([buf[j], buf[i]]) + buf[j + 1:]
                    )
                i += 1
            if self.changed > change_counter:
                continue
            i = 0
            while i < len(self.last_data.buffer):
                buf = self.last_data.buffer
                if not self.incorporate_new_buffer(
                    buf[:i] + buf[i + 1:]
                ):
                    if buf[i] == 0:
                        buf = bytearray(buf)
                        j = i
                        while j >= 0:
                            if buf[j] > 0:
                                buf[j] -= 1
                                self.incorporate_new_buffer(bytes(buf))
                                break
                            else:
                                buf[j] = 255
                            j -= 1
                i += 1
            if self.changed > change_counter:
                continue
            buckets = [[] for _ in range(256)]
            for i, c in enumerate(self.last_data.buffer):
                buckets[c].append(i)
            indices = []
            for bucket in buckets:
                if len(bucket) > 1:
                    indices.extend(
                        (j, k)
                        for j in bucket for k in bucket
                        if j < k
                    )
            for j, k in indices:
                buf = self.last_data.buffer
                if k >= len(buf):
                    continue
                if buf[j] == buf[k]:
                    c = buf[j]
                    if c == 0:
                        if j > 0 and buf[j - 1] > 0 and buf[k - 1] > 0:
                            self.incorporate_new_buffer(
                                buf[:j - 1] +
                                bytes([buf[j - 1] - 1, 255]) +
                                buf[j + 1:k - 1] +
                                bytes([buf[k - 1] - 1, 255]) +
                                buf[k + 1:]
                            )
                    c = buf[j]
                    if c > 0:
                        bd = bytes([c - 1])
                        if self.incorporate_new_buffer(
                            buf[:j] + bd + buf[j + 1:k] + bd +
                            buf[k + 1:]
                        ):
                            for d in range(c - 1):
                                buf = self.last_data.buffer
                                bd = bytes([d])
                                if self.incorporate_new_buffer(
                                    buf[:j] + bd + buf[j + 1:k] + bd +
                                    buf[k + 1:]
                                ):
                                    break
            if self.changed > change_counter:
                continue
            buf = self.last_data.buffer
            for j in range(len(buf)):
                buf = self.last_data.buffer
                if j >= len(buf):
                    break
                if buf[j] == 0:
                    continue
                for k in range(j + 1, len(buf)):
                    buf = self.last_data.buffer
                    if k >= len(buf):
                        break
                    if buf[j] > buf[k]:
                        self.incorporate_new_buffer(
                            buf[:j] + bytes([buf[k]]) + buf[j + 1:k] +
                            bytes([buf[j]]) + buf[k + 1:]
                        )
                    buf = self.last_data.buffer
                    if k >= len(buf):
                        break
                    if buf[j] > 0 and buf[k] > 0 and buf[j] != buf[k]:
                        if self.incorporate_new_buffer(
                            buf[:j] + bytes([buf[j] - 1]) + buf[j + 1:k] +
                            bytes([buf[k] - 1]) + buf[k + 1:]
                        ):
                            break
                    if buf[j] == 0:
                        break
                    for t in range(256):
                        if self.incorporate_new_buffer(
                            buf[:j] + bytes([buf[j] - 1]) + buf[j + 1:k] +
                            bytes([t]) + buf[k + 1:]
                        ):
                            break

    def mutate_data_to_new_buffer(self):
        n = min(len(self.last_data.buffer), self.last_data.index)
        if not n:
            return b''
        if n == 1:
            return self.rand_bytes(1)

        if self.last_data.status == Status.OVERRUN:
            result = bytearray(self.last_data.buffer)
            for i, c in enumerate(self.last_data.buffer):
                t = self.random.randint(0, 2)
                if t == 0:
                    result[i] = 0
                elif t == 1:
                    result[i] = self.random.randint(0, c)
                else:
                    result[i] = c
            return bytes(result)

        probe = self.random.randint(0, 255)
        if probe <= 200 or len(self.last_data.intervals) <= 1:
            c = self.random.randint(0, 2)
            i = self.random.randint(0, self.last_data.index - 1)
            result = bytearray(self.last_data.buffer)
            if c == 0:
                result[i] ^= (1 << self.random.randint(0, 7))
            elif c == 1:
                result[i] = 0
            else:
                result[i] = 255
            return bytes(result)
        else:
            int1 = None
            int2 = None
            while int1 == int2:
                i = self.random.randint(0, len(self.last_data.intervals) - 2)
                int1 = self.last_data.intervals[i]
                int2 = self.last_data.intervals[
                    self.random.randint(
                        i + 1, len(self.last_data.intervals) - 1)]
            return self.last_data.buffer[:int1[0]] + \
                self.last_data.buffer[int2[0]:int2[1]] + \
                self.last_data.buffer[int1[1]:]

    def rand_bytes(self, n):
        if n == 0:
            return b''
        return self.random.getrandbits(n * 8).to_bytes(n, 'big')


def find_interesting_buffer(test_function, settings=None):
    runner = TestRunner(test_function, settings)
    runner.run()
    if runner.last_data.status == Status.INTERESTING:
        return runner.last_data.buffer
