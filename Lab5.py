import numpy as nmp


def deikstra(v: nmp.ndarray, e: nmp.ndarray, start: int):
    weights = nmp.ones(v.shape) * nmp.inf
    weights[start] = 0
    paths = nmp.array([None for i in range(len(weights))])
    paths[start] = []
    not_visited = nmp.array([True for _ in v])
    current_map = nmp.arange(len(v))
    for i in range(len(v) - 1):
        not_visited_weights = weights[not_visited]
        current = not_visited_weights.argmin()
        e_line = e[not_visited][current, not_visited]
        for j in range(len(e_line)):
            if e_line[j] is not None:
                new_weight = not_visited_weights[current] + e_line[j]
                if new_weight < not_visited_weights[j]:
                    not_visited_weights[j] = new_weight
                    paths[current_map[j]] = paths[current_map[current]] + [current_map[current]]
        weights[not_visited] = not_visited_weights
        not_visited[current_map[current]] = False
        current_map = nmp.delete(current_map, current)

    return weights, paths


def main():
    v = nmp.arange(6)
    e = nmp.array([
        [None, 7, 9, None, None, 14],
        [7, None, 10, 15, None, None],
        [9, 10, None, 11, None, 2],
        [None, 15, 11, None, 6, None],
        [None, None, None, 6, None, 9],
        [14, None, 2, None, 9, None],
    ])
    print(deikstra(v, e, 0))


if __name__ == '__main__':
    main()
