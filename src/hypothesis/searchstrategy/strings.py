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
import unicodedata

from hypothesis.errors import InvalidArgument
from hypothesis.internal import charstree
from hypothesis.internal.compat import hunichr, text_type, binary_type, hrange
from hypothesis.searchstrategy.strategies import SearchStrategy, \
    MappedSearchStrategy
from hypothesis.internal.conjecture import utils as d


class OneCharStringStrategy(SearchStrategy):

    """A strategy which generates single character strings of text type."""
    specifier = text_type
    zero_point = ord(u'0')

    def __init__(self,
                 whitelist_categories=None,
                 blacklist_categories=None,
                 blacklist_characters=None,
                 min_codepoint=None,
                 max_codepoint=None):
        if whitelist_categories is not None:
            whitelist_categories = set(whitelist_categories)
        blacklist_categories = set(blacklist_categories or [])
        blacklist_characters = set(blacklist_characters or [])

        min_codepoint = int(min_codepoint or 0)
        max_codepoint = int(max_codepoint or sys.maxunicode)

        intervals = []
        for i in hrange(min_codepoint, max_codepoint + 1):
            c = hunichr(i)
            if c in blacklist_categories:
                continue
            cat = unicodedata.category(c)
            if (
                whitelist_categories is not None and
                cat not in whitelist_categories
            ):
                continue
            if cat in blacklist_categories:
                continue
            if not intervals or i > intervals[-1][-1] + 1:
                intervals.append([i, i])
            else:
                assert i == intervals[-1][-1] + 1
                intervals[-1][-1] += 1
        if not intervals:
            raise InvalidArgument("Empty set of allowed characters")
        self.intervals = intervals

    def do_draw(self, data):
        interval = d.choice(data, self.intervals)
        return hunichr(d.integer_range(data, *interval))

    def is_good(self, char):
        if char in self.blacklist_characters:
            return False

        categories = charstree.categories(self.unicode_tree)
        if unicodedata.category(char) not in categories:
            return False

        codepoint = ord(char)
        return self.min_codepoint <= codepoint <= self.max_codepoint


class StringStrategy(MappedSearchStrategy):

    """A strategy for text strings, defined in terms of a strategy for lists of
    single character text strings."""

    def __init__(self, list_of_one_char_strings_strategy):
        super(StringStrategy, self).__init__(
            strategy=list_of_one_char_strings_strategy
        )

    def __repr__(self):
        return u'StringStrategy()'

    def pack(self, ls):
        return u''.join(ls)


class BinaryStringStrategy(MappedSearchStrategy):

    """A strategy for strings of bytes, defined in terms of a strategy for
    lists of bytes."""

    def __repr__(self):
        return u'BinaryStringStrategy()'

    def pack(self, x):
        assert isinstance(x, list), repr(x)
        ba = bytearray(x)
        return binary_type(ba)
