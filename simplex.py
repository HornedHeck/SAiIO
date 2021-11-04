from typing import Optional

import numpy as nmp


def simplex_min(
        a: nmp.ndarray,
        c: nmp.ndarray,
        x_0: nmp.ndarray,
        j_b_0: nmp.ndarray
) -> Optional[tuple[nmp.ndarray, nmp.ndarray]]:
    a_b_r = nmp.eye(len(j_b_0))
    j_b = j_b_0
    j_n = nmp.setdiff1d(range(len(c)), j_b)
    x = x_0
    while True:
        dn = (c[j_n] - c[j_b].dot(a_b_r).dot(a[:, j_n]))
        in_i = nmp.where(dn < 0)[0]
        in_i = in_i[0] if len(in_i) > 0 else 0

        if dn[in_i] >= 0:
            break

        abr_a = a_b_r.dot(a[:, j_n[in_i]])
        thetas = nmp.array([x[j_b[i]] / abr_a[i] if abr_a[i] != 0 else -1 for i in range(len(j_b))])

        valid_theta_idx = nmp.where(thetas > 0)[0]
        out_i = valid_theta_idx[thetas[valid_theta_idx].argmin()]

        theta = thetas[out_i]

        x_b = x[j_b.astype(int)] - abr_a * theta
        x[j_n[in_i]] += theta
        x = nmp.array([x_b[nmp.where(j_b == i)[0][0]] if i in j_b else x[i] for i in range(len(x))])

        tmp = j_b[out_i]
        j_b[out_i] = j_n[in_i]
        j_n[in_i] = tmp

        a_b_r = nmp.linalg.inv(a[:, j_b])

    return x, j_b


def simplex_max(
        a: nmp.ndarray,
        c: nmp.ndarray,
        x_0: nmp.ndarray,
        j_b_0: nmp.ndarray
) -> Optional[tuple[nmp.ndarray, nmp.ndarray]]:
    a_b_r = nmp.eye(len(j_b_0))
    j_b = j_b_0
    j_n = nmp.setdiff1d(range(len(c)), j_b)
    x = x_0
    while True:
        dn = (c[j_n] - c[j_b].dot(a_b_r).dot(a[:, j_n]))
        in_i = nmp.where(dn > 0)[0]
        in_i = in_i[0] if len(in_i) > 0 else 0

        if dn[in_i] <= 0:
            break

        abr_a = a_b_r.dot(a[:, j_n[in_i]])
        thetas = nmp.array([x[j_b[i]] / abr_a[i] if abr_a[i] != 0 else -1 for i in range(len(j_b))])

        valid_theta_idx = nmp.where(thetas > 0)[0]
        out_i = valid_theta_idx[thetas[valid_theta_idx].argmin()]

        theta = thetas[out_i]

        x_b = x[j_b.astype(int)] - abr_a * theta
        x[j_n[in_i]] += theta
        x = nmp.array([x_b[nmp.where(j_b == i)[0][0]] if i in j_b else x[i] for i in range(len(x))])

        tmp = j_b[out_i]
        j_b[out_i] = j_n[in_i]
        j_n[in_i] = tmp

        a_b_r = nmp.linalg.inv(a[:, j_b])

    return x, j_b
