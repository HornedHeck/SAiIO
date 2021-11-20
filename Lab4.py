import numpy as nmp

from Lab6 import floyd_warshall


def main():
    v = nmp.arange(4)
    e = nmp.array([
        [None, 1, 6, None],
        [None, None, 4, 1],
        [None, None, None, None],
        [None, None, 1, None],
    ])
    e_path = nmp.array([[-1 if e[i, j] is not None else None for j in range(len(v))] for i in range(len(v))])
    e_weight = nmp.array([[-e[i, j] if e[i, j] is not None else None for j in range(len(v))] for i in range(len(v))])
    print('Longest path (by points)')
    for v_i in v:
        fw = floyd_warshall(v, e_path, 0, v_i)
        print(f'1 -> {v_i + 1}: {-fw[0]} {fw[1]}')
    print('Longest path (by weight)')
    for v_i in v:
        fw = floyd_warshall(v, e_weight, 0, v_i)
        print(f'1 -> {v_i + 1}: {-fw[0]} {fw[1]}')


if __name__ == '__main__':
    main()
