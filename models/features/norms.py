from enum import Enum


class NormTypes(Enum):
    NONE = 0
    MINMAX = 1
    ZNORM = 2


def empty_norm(el):
    return el


def minmax_norm(el, minimum, maximum):
    return (el - minimum) / (maximum - minimum)


def z_norm(el, mean, std):
    return (el - mean) / std
