from enum import Enum


class NormTypes(Enum):
    NONE = 0
    MINMAX = 1
    ZNORM = 2


def empty_norm():
    def wrapped(el):
        return el
    return wrapped


def minmax_norm(minimum, maximum):
    def wrapped(el, mi=minimum, ma=maximum):
        return (el - mi) / (ma - mi)
    return wrapped


def z_norm(mean, std):
    def wrapped(el, m=mean, s=std):
        return (el - m) / s
    return wrapped
