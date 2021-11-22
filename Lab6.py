import numpy as nmp


def floyd_warshall(v: nmp.ndarray, w: nmp.ndarray, start: int, end: int):
    def floyd_warshall_int(i: int, j: int, k: int):
        if k == 0:
            if w[i, j] is None:
                return nmp.inf, []
            return w[i, j], [j]
        else:
            single_res = floyd_warshall_int(i, j, k - 1)
            sum_res_1 = floyd_warshall_int(i, k, k - 1)
            sum_res_2 = floyd_warshall_int(k, j, k - 1)
            if single_res[0] < sum_res_1[0] + sum_res_2[0]:
                return single_res
            else:
                return sum_res_1[0] + sum_res_2[0], sum_res_1[1] + sum_res_2[1]

    return floyd_warshall_int(start, end, len(v) - 1)


def main():
    v = nmp.arange(4)
    e = nmp.array([
        [None, 1, 6, None],
        [None, None, 4, 1],
        [None, None, None, None],
        [None, None, 1, None],
    ])
    for i in range(len(v)):
        for j in range(len(v)):
            fw = floyd_warshall(v, e, i, j)
            if fw[0] != nmp.inf:
                print(f'{i + 1} -> {j + 1}: {fw[0]} {fw[1]}')


if __name__ == '__main__':
    main()
