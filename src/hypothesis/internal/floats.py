import math
import struct


def sign(x):
    return math.copysign(1, x)


def is_negative(x):
    return sign(x) < 0


def count_between_floats(x, y):
    assert x <= y
    if is_negative(x):
        if is_negative(y):
            return float_to_int(x) - float_to_int(y) + 1
        else:
            return count_between_floats(x, -0.0) + count_between_floats(0.0, y)
    else:
        assert not is_negative(y)
        return float_to_int(y) - float_to_int(x) + 1


def float_to_int(value):
    return (
        struct.unpack(b'!Q', struct.pack(b'!d', value))[0]
    )


def int_to_float(value):
    return (
        struct.unpack(b'!d', struct.pack(b'!Q', value))[0]
    )
