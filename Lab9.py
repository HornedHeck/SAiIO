import numpy as nmp

from Lab7 import max_flow


def pairs_count(x: nmp.ndarray, y: nmp.ndarray, w: nmp.ndarray):
    # build w_x
    v = nmp.arange(len(x) + len(y))
    v_x = nmp.arange(len(v) + 2)
    w_x = nmp.ones((len(v_x), len(v_x))) * (-nmp.inf)
    for i in v:
        for j in v:
            if w[i, j] != nmp.inf:
                w_x[i, j] = w[i, j]

    s = v_x[-2]
    t = v_x[-1]
    w_x[s, x] = 1
    w_x[y, t] = 1
    print(w_x)
    # find max flow
    return max_flow(v_x, w_x, s, t)


def main():
    x = nmp.array([0, 3, 5, 6, 7])
    y = nmp.array([1, 2, 4, 8])
    e = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (3, 4),
        (4, 5),
        (4, 6),
        (4, 7),
        (4, 8)
    ]
    v_len = len(x) + len(y)
    w = nmp.ones((v_len, v_len)) * nmp.inf
    for i, j in e:
        w[i, j] = 1
        w[j, i] = 1
    print(pairs_count(x, y, w))


if __name__ == '__main__':
    main()
