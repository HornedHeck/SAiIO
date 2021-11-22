import numpy as nmp

from Lab6 import floyd_warshall


def min_price_flow(v: nmp.ndarray, w: nmp.ndarray, p: nmp.ndarray, s: int, t: int, m):
    p_raw = p
    p = p.copy()

    def __is_reversal(i, j) -> bool:
        return p_raw[i, j] == nmp.inf

    flows = nmp.zeros(w.shape)
    flow = 0
    path = floyd_warshall(v, p, s, t)
    while path[0] != nmp.inf:
        delta_f = nmp.array(
            [flows[j, i] if __is_reversal(i, j)
             else w[i, j] - flows[i, j]
             for i, j in zip([s] + path[1][:-1], path[1])] +
            [m - flow]
        )
        delta_f = delta_f.min(initial=nmp.inf)
        for i, j in zip([s] + path[1][:-1], path[1]):
            if __is_reversal(i, j):
                flows[j, i] -= delta_f
                p[j, i] = -p[i, j]
                if flows[j, i] == 0:
                    p[i, j] = nmp.inf
            else:
                flows[i, j] += delta_f
                p[j, i] = -p[i, j]
                if flows[i, j] == w[i, j]:
                    p[i, j] = nmp.inf
        flow += delta_f
        print(f'Flow of {delta_f} by {nmp.array([s] + path[1]) + 1}')

        if flow == m:
            break

        path = floyd_warshall(v, p, s, t)

    total_price = 0
    for i in v:
        for j in v:
            if p_raw[i, j] != nmp.inf:
                total_price += flows[i, j] * p_raw[i, j]
    return flow, total_price


def main():
    v = nmp.arange(4)
    w = nmp.array([
        [nmp.inf, 4, 2, nmp.inf, ],
        [nmp.inf, nmp.inf, nmp.inf, 2, ],
        [nmp.inf, 3, nmp.inf, 3, ],
        [nmp.inf, nmp.inf, nmp.inf, nmp.inf, ]
    ])
    p = nmp.array([
        [nmp.inf, 5, 1, nmp.inf, ],
        [nmp.inf, nmp.inf, nmp.inf, 1, ],
        [nmp.inf, 1, nmp.inf, 6, ],
        [nmp.inf, nmp.inf, nmp.inf, nmp.inf, ]
    ])
    print(min_price_flow(v, w, p, 0, 3, 3))


if __name__ == '__main__':
    main()
