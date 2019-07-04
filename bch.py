import numpy as np
from gf import P, gen_pow_matrix, linsolve, minpoly, polyprod, sump, polydiv, polyval, euclid


class BCH:
    """Класс БЧХ-кодов"""
    def __init__(self, n, t):
        self.n = n
        self.t = t
        q = P(n + 1).pw
        data = np.loadtxt("primpoly.txt", delimiter=', ', dtype=np.int)
        for i in data:
            if P(i).pw == q:
                self.pm = gen_pow_matrix(i)
                break
        zeros = []
        for i in range(1, 2 * t + 1):
            zeros.append(self.pm[i][1])
        self.R = np.asarray(zeros, dtype=int)
        p, roots = minpoly(self.R, self.pm)
        self.g = p
        self.k = self.n - self.g.size + 1

    def encode(self, U):
        """Кодирование БЧХ"""
        fi = []
        x = np.zeros(self.g.shape[0], dtype=int)
        x[0] = 1
        for word in U:
            first = polyprod(x, word, self.pm)
            second = polydiv(first, self.g, self.pm)
            v = sump(first, second[1])
            if v.size < self.pm.shape[0] - 1:
                v = np.concatenate((np.zeros(self.pm.shape[0] - 1 - v.shape[0], dtype=int), v))
            fi.append(v)
        return np.asarray(fi)

    def decode(self, W, method="euclid"):
        """Декодирование БЧХ"""
        fi = []
        for w in W:
            synd = polyval(w, self.R, self.pm)
            t = 0
            for i in synd:
                if i != 0:
                    t = -1
                    break
            if t == 0:
                fi.append(w)
                continue
            if method == "pgz":
                v = self.t
                while v > 0:
                    synd_m = np.empty([v, v], dtype=int)
                    for i in range(synd_m.shape[0]):
                        for j in range(synd_m.shape[1]):
                            synd_m[i][j] = synd[i + j]
                    synd_b = np.empty([v], dtype=int)
                    for i in range(synd_b.shape[0]):
                        synd_b[i] = synd[v + i]

                    L = linsolve(synd_m, synd_b, self.pm)

                    if np.isnan(L[0]):
                        v -= 1
                        continue

                    L = np.hstack([L, np.asarray(1)])

                    L_roots = set()
                    for i in range(self.pm.shape[0]):
                        x = polyval(L, np.asarray([self.pm[i][1]]), self.pm)
                        if (x[0] == 0):
                            L_roots.add(self.pm[i][1])
                    for i in L_roots:
                        j = self.pm[i - 1][0]
                        w[j - 1] ^= 1
                    synd = polyval(w, self.R, self.pm)
                    t = 0
                    for i in synd:
                        if i != 0:
                            t = -1
                            break
                    if t == 0:
                        fi.append(w)
                        break
                    else:
                        fi.append(np.nan)
                        break
                if v == 0:
                    fi.append(np.nan)
            else:
                synd = np.concatenate((np.flip(synd), np.asarray([1])))
                x = np.concatenate((np.asarray([1]), np.zeros(2 * self.t + 1, dtype=int)))
                a, b, L = euclid(x, synd, self.pm, self.t)
                L_roots = set()
                for i in range(self.pm.shape[0]):
                    x = polyval(L, np.asarray([self.pm[i][1]]), self.pm)
                    if x[0] == 0:
                        L_roots.add(self.pm[i][1])
                power = L.size - 1
                i = 0
                while L[i] == 0:
                    power -= 1
                    i += 1
                if len(L_roots) != power:  # количество корней не совпадает
                    fi.append(np.nan)
                    continue
                for i in L_roots:
                    j = self.pm[i - 1][0]
                    w[j - 1] ^= 1
                synd = polyval(w, self.R, self.pm)
                t = 0
                for i in synd:
                    if i != 0:
                        t = -1
                        break
                if t == 0:
                    fi.append(w)
                    continue
                else:
                    fi.append(np.nan)
                    continue
        return np.asarray(fi)

    def dist(self):
        """Определение расстояния"""
        k = self.k
        words = []
        for i in range(1, 1 << k):
            u = []
            ii = i
            while ii != 0:
                u.append(ii & 1)
                ii >>= 1
            u.reverse()
            u = np.asarray(u, dtype=int)
            if u.size < k:
                u = np.concatenate((np.zeros(k - u.size, dtype=int), u))
            words.append(u)
        v = self.encode(np.asarray(words))
        res = v.shape[0]
        for i in range(v.shape[0]):
            for j in range(v.shape[0]):
                if i != j:
                    x = sump(v[i, :], v[j, :])
                    t = 0
                    for q in range(v.shape[1]):
                        if x[q]:
                            t += 1
                    if t >= self.t * 2 + 1:
                        res = min(res, t)
        return res
