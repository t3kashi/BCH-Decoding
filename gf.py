import numpy as np

class P:
    """Полином над полем F2
       n -- коэффициенты полинома, представленные 10ным числом
       pw -- старшая степень полинома
    """
    def __init__(self, n):
        self.kf = n
        a = 1
        power = 0
        while a <= self.kf:
            a <<= 1
            power += 1
        self.pw = power - 1

def mul(self, other, ma):
    """Умножение двух полиномов по модулю поля
       self -- первый элемент умножения,
       other -- второй элемент умножения,
       ma -- таблица соответствия полиномиального и степенного представления элементов поля"""
    a = P(self).kf
    b = P(other).kf
    power = (ma[a - 1][0] + ma[b - 1][0]) % (ma.shape[0])
    if a != 0 and b != 0:
        fi = ma[power - 1][1]
    else:
        fi = 0
    return fi

def div(self, other, ma):
    """Деление элементов поля. Аналогично с mul"""
    a = P(self).kf
    b = P(other).kf
    power = (ma[a - 1][0] - ma[b - 1][0]) % (ma.shape[0])
    if (a != 0 and b != 0):
        fi = ma[power - 1][1]
    else:
        fi = 0
    return fi

def mod(self, other):
    """Вспомогательная функция для gen_pow_matrix, возвращает
       остаток от деления self на other в поле"""
    fi = P(self.kf)
    while fi.pw >= other.pw:
        b = P(1 << (fi.pw - other.pw)).kf
        a = other.kf
        mul = 0
        while b != 0:
            if b % 2 == 1:
                mul ^= a
            b >>= 1
            a <<= 1
        fi = P(fi.kf ^ P(mul).kf)
    return fi

def gen_pow_matrix(primpoly):
    """Генерация таблицы соответствия степенного и десятичного представления элементов поля,
       используется в реализации умножения и деления полиномов в поле"""
    p = P(primpoly).pw
    a = np.zeros([1 << p, 2], dtype=int)
    a[0][1] = 1
    a[-1][1] = 1
    alpha = P(2)
    for i in range(1, a.shape[0]):
        a[i][1] = alpha.kf
        alpha = mod(P(alpha.kf << 1), (P(primpoly)))
    for i in range(0, a.shape[0] - 1):
        c = a[i][1]
        a[c][0] = i
        if i == 0:
            a[c][0] = a.shape[0] - 1
    a = np.delete(a, 0, axis=0)
    return a

def add(X, Y):
    """Сложение двух таблиц элементов в поле"""
    size = X.shape[0]
    fi = np.zeros([size, 2], dtype=int)
    for i in range(size):
        fi[i][0] = X[i][0] ^ Y[i][0]
        fi[i][1] = X[i][1] ^ Y[i][1]
    return fi

def sum(X, axis = 0):
    """Суммирование элементов матрицы по какой-либо из осей"""
    return np.bitwise_xor.reduce(X, axis)



def sump(p1, p2):
    """Поэлементное суммирование коэффициентов полинома"""
    fi = p1.copy()
    c = p2.copy()
    if fi.shape[0] != c.shape[0]:
        if fi.shape[0] < c.shape[0]:
            fi = np.concatenate((np.zeros(c.shape[0] - fi.shape[0], dtype=int), fi))
        else:
            c = np.concatenate((np.zeros(fi.shape[0] - c.shape[0], dtype=int), c))
    for i in range(0, fi.size):
        fi[i] ^= c[i]
    return fi

def prod(X, Y, pm):
    """Поэлементное умножение двух матрииц одинакового размера"""
    a = np.zeros(X.shape, dtype=int)
    b = np.zeros(Y.shape, dtype=int)
    fi = np.zeros(a.shape, dtype=int)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i][j] = pm[X[i][j] - 1][0]
            b[i][j] = pm[Y[i][j] - 1][0]
    for i in range(fi.shape[0]):
        for j in range(fi.shape[1]):
            if (X[i][j] != 0 and Y[i][j] != 0):
                power = (a[i][j] + b[i][j]) % (pm.shape[0])
                if power != 0:
                    fi[i][j] = pm[power-1][1]
                else:
                    power = pm.shape[0] - 1
                    fi[i][j] = pm[power][1]
            else:
                fi[i][j] = 0
    return fi


def prod_on1(X, Y, pm):
    """Поэлементное умножение матрицы на один элемент"""
    a = np.zeros(X.shape, dtype=int)
    b = pm[Y - 1][0]
    fi = np.zeros(a.shape, dtype=int)
    for i in range(a.shape[0]):
        a[i] = pm[X[i] - 1][0]
    for i in range(fi.shape[0]):
        if X[i] != 0 and Y != 0:
            power = (a[i] + b) % (pm.shape[0])
            if power != 0:
                fi[i] = pm[power-1][1]
            else:
                power = pm.shape[0] - 1
                fi[i] = pm[power][1]
        else:
            fi[i] = 0
    return fi


def divide(X, Y, pm):
    """Поэлементное деление двух матрииц одинакового размера"""
    a = np.zeros(X.shape, dtype=int)
    b = np.zeros(Y.shape, dtype=int)
    fi = np.zeros(a.shape, dtype=int)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i][j] = pm[X[i][j] - 1][0]
            b[i][j] = pm[Y[i][j] - 1][0]
    for i in range(fi.shape[0]):
        for j in range(fi.shape[1]):
            if (X[i][j] != 0):
                power = (a[i][j] - b[i][j]) % (pm.shape[0])
                if power != 0:
                    fi[i][j] = pm[power-1][1]
                else:
                    power = pm.shape[0] - 1
                    fi[i][j] = pm[power][1]
            else:
                fi[i][j] = 0
    return fi


def divide_on1(X, Y, pm):
    """Поэлементное деление матрицы на один элемент"""
    a = np.zeros(X.shape, dtype=int)
    b = pm[Y - 1][0]
    fi = np.zeros(a.shape, dtype=int)
    for i in range(a.shape[0]):
        a[i] = pm[X[i] - 1][0]
    for i in range(fi.shape[0]):
        power = (a[i] - b) % (pm.shape[0])
        if (X[i] != 0):
            if power != 0:
                fi[i] = pm[power - 1][1]
            else:
                power = pm.shape[0] - 1
                fi[i] = pm[power][1]
        else:
            fi[i] = 0
    return fi

def minpoly(x, pm1):
    """Нахождение минимального полинома по заданному вектору"""
    rts = []
    numbers = set()
    p = []
    pm = np.vstack((np.array([0, 1]), pm1))
    for root in x:
        if root in numbers:
            continue
        rp = pm[root][0]
        b = [root]
        rts.append(root)
        numbers.add(root)
        rp = (rp * 2) % (pm.shape[0] - 1)
        if rp == 0:
            rp = pm.shape[0] - 1
        while pm[rp][1] != root:
            rts.append(pm[rp][1])
            numbers.add(pm[rp][1])
            b.append(pm[rp][1])
            rp = (rp * 2) % (pm.shape[0] - 1)
            if rp == 0:
                rp = pm.shape[0] - 1
        for i in b:
            p.append(np.asarray([1, i]))
    pm = np.delete(pm, 0, axis=0)
    fi = np.asarray([1])
    for i in p:
        fi = polyprod(fi, i, pm)
    return fi, np.sort(np.asarray(rts))


def polyval(p, x, pm):
    """Значение полинома при заданных X"""
    fi = np.zeros(x.shape[0], dtype=int)
    k = np.zeros(p.shape[0], dtype=int)
    for i in range(fi.shape[0]):
        arg = 1
        for j in range(k.shape[0] - 1, -1, -1):
            k[j] = arg
            arg = prod(np.asarray([[arg]]), np.asarray([[x[i]]]), pm)[0][0]
        fi[i] = sum(prod(np.asarray([p]), np.asarray([k]), pm)[0])
    return fi

def polyprod(p1, p2, pm):
    """Произведение двух полиномов"""
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    fi = np.zeros(p1.shape[0] + p2.shape[0] - 1, dtype=int)
    for power in range(fi.shape[0]):
        X = []
        Y = []
        for i in range(max(0, power - p2.shape[0] + 1), min(power + 1, p1.shape[0])):
            X.append(p1[p1.shape[0] - i - 1])
            Y.append(p2[p2.shape[0] - (power - i) - 1])
        fi[fi.shape[0] - power - 1] = sum(prod(np.asarray([X]), np.asarray([Y]), pm)[0])
    return fi

def polydiv(p1, p2, pm):
    """Целочисленное деление полиномов с остатком"""
    p1 = p1
    p2 = p2
    p = p1.copy()
    fi = []
    c = divide(np.asarray([[1]]), np.asarray([[p2[0]]]), pm)[0][0]
    for power in range(p1.shape[0] - p2.shape[0], -1, -1):
        if p[0] == 0:
            fi.append(0)
            p = p[1:].copy()
            continue
        coef = prod(np.asarray([[c]]), np.asarray([[p[0]]]), pm)[0][0]
        fi.append(coef)
        d = p2.copy()
        d = prod(np.asarray([d]), np.asarray([np.ones(d.size, dtype=int) * coef]), pm)[0]
        d = np.concatenate((np.asarray(d), np.zeros(power, dtype=int)))
        p = (p ^ d)[1:]
    return fi, p.copy()


def linsolve(A, b1, pm):
    """Нахождение корней СЛАУ"""
    n = A.shape[0]
    qq = A.copy()
    b = b1.copy()
    for k in range(n): # Меняем местами ряды
        swap_row = k
        while swap_row < qq.shape[0] and not qq[swap_row, k]:
            swap_row += 1
        if swap_row == qq.shape[0]:
            return np.asarray([np.nan])
        if swap_row != k:
            qq[k, :], qq[swap_row, :] = qq[swap_row, :], np.copy(qq[k, :])
            b[k], b[swap_row] = b[swap_row], np.copy(b[k]) # Делаем диагональные элементы равными единице
        if qq[k, k] != 1:
            b[k] = div(b[k], qq[k, k], pm)
            qq[k, :] = divide_on1(qq[k, :], qq[k, k], pm) # Зануляем недиагональнве
        for i in range(n):
            if i != k:
                b[i] = b[i] ^ mul(b[k], qq[i, k], pm)
                qq[i, :] = sump(qq[i, :], prod_on1(qq[k, :], qq[i, k], pm))
    return b


def euclid(p1, p2, pm, max_deg=0):
    """Алгоритм Евклиида для деления полимов"""
    p1 = p1.copy()
    p2 = p2.copy()
    if p2[0] == 0:
        return np.nan, np.nan, np.nan
    r0 = p1.copy() # Нулевой шаг
    r1 = p2.copy()
    x0 = np.asarray([0])
    x1 = np.asarray([1])
    y0 = np.asarray([1])
    y1 = np.asarray([0])

    while 1:
        q, r = polydiv(r0, r1, pm)
        x = sump(x1, polyprod(q, x0, pm))
        x1 = x0
        x0 = x
        y = sump(y1, polyprod(q, y0, pm))
        y1 = y0
        y0 = y
        r0 = r1
        r1 = r
        if (max_deg == 0 and r[0] == 0) or (max_deg != 0 and r.size <= max_deg):
            return r, x, y
