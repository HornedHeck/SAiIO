from collections import deque

import numpy as nmp


def __get_sum_res(s1, s2):
    if s1[0] == -nmp.inf or s2[0] == -nmp.inf:
        return -nmp.inf, []
    else:
        return min(s1[0], s2[0]), s1[1] + s2[1]


def __floyd_warshall_mod(v: nmp.ndarray, w: nmp.ndarray, s: int, t: int):
    def floyd_warshall_int(i: int, j: int, k: int):
        if k == 0:
            return w[i, j], [j]
        else:
            single_res = floyd_warshall_int(i, j, k - 1)
            sum_res_1 = floyd_warshall_int(i, k, k - 1)
            if sum_res_1[0] != -nmp.inf:
                sum_res_2 = floyd_warshall_int(k, j, k - 1)
            else:
                sum_res_2 = -nmp.inf, []
            sum_res = __get_sum_res(sum_res_1, sum_res_2)
        if single_res[0] >= sum_res[0]:
            return single_res
        else:
            return sum_res

    return floyd_warshall_int(s, t, len(v) - 1)


def bws(v: nmp.ndarray, w: nmp.ndarray, s: int, t: int):
    # v_i , (v_i path)
    expanded = [False for _ in v]
    q = deque()
    q.append((s, []))
    while len(q) != 0:
        v_i, path = q.popleft()
        if v_i == t:
            path_sum = nmp.array([w[i, j] for i, j in zip(path, path[1:] + [t])]).min(initial=nmp.inf)
            return path_sum, path[1:] + [t]
        path_x = path + [v_i]
        for v_j in v:
            if w[v_i, v_j] > 0 and not expanded[v_j]:
                q.append((v_j, path_x))
                expanded[v_j] = True

    return -nmp.inf, []


def max_flow(v: nmp.ndarray, w: nmp.ndarray, s: int, t: int):
    w_raw = w
    w = w.copy()
    flow = 0
    path = bws(v, w, s, t)
    while path[0] != -nmp.inf:
        for v_i, v_j in zip([s] + path[1][:-1], path[1]):
            w[v_i, v_j] -= path[0]
            if w[v_i, v_j] == 0:
                w[v_i, v_j] = -nmp.inf
            if w[v_j, v_i] == -nmp.inf:
                w[v_j, v_i] = -path[0]
            else:
                w[v_j, v_i] -= path[0]

        flow += path[0]
        print(f'Flow of {path[0]} by {nmp.array([s] + path[1]) + 1}')
        path = bws(v, w, s, t)

    return flow


def main():
    v = nmp.arange(6)
    edges = [
        (0, 1, 7),
        (0, 2, 4),
        (1, 2, 4),
        (1, 4, 2),
        (2, 3, 4),
        (2, 4, 8),
        (3, 5, 12),
        (4, 3, 4),
        (4, 5, 5),
    ]
    w = nmp.ones((len(v), len(v))) * (-nmp.inf)
    for i, j, w_ij in edges:
        w[i, j] = w_ij

    print(max_flow(v, w, 0, 5))


if __name__ == '__main__':
    main()
