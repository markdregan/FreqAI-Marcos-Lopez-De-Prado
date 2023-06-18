import numpy as np


def clip_kelly(b):
    return np.min([np.max([b, 0]), 1])


def kelly_bet_size(p, win, loss, kelly_fraction):

    raw_kelly = (p / abs(loss)) - ((1 - p) / abs(win))
    clipped_kelly = clip_kelly(raw_kelly)

    return clipped_kelly * kelly_fraction
