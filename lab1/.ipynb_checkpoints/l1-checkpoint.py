# %%
import json
import numpy as np
import numpy.ma as ma
import pprint

def initProgram(file_name):
    # открыть файл с данными, поместить все данные в переменную raw_data
    with open(file_name) as json_file:
        raw_data = json.load(json_file)
    
    # подготовка данных
    suppliers = np.array(raw_data['Suppliers'], dtype=int)
    consumers = np.array(raw_data['Consumers'], dtype=int)
    costs = np.array(raw_data['Costs'], dtype=int)
    capacity = np.zeros(costs.shape, dtype=int)
    
    print(f'Поставщики: {suppliers}')
    print(f'Потребители: {consumers}')
    print(f'Матрица тарифов:\n {costs}')

    print(f'Проверка исходных данных на сбалансированность:')
    if np.sum(suppliers) != np.sum(consumers):
        print(f'задача несбалансирована, ∑a = {np.sum(suppliers)}, ∑b = {np.sum(consumers)}')
        return
    print(f'задача сбалансирована, ∑a=∑b={np.sum(suppliers)}')

    return suppliers, consumers, costs, capacity

# %%


def northWest(suppliers, consumers, costs, capacity):
    consumer_index = 0
    lost_basis = []
    while consumer_index < len(consumers):
        supplier_index = 0
        while supplier_index < len(suppliers):
            consumer = consumers[consumer_index]
            supplier = suppliers[supplier_index]
            if consumer == 0:
                break
            if supplier == 0:
                supplier_index += 1
                continue
            # проверка "потери" базисной клетки
            if supplier == consumer and supplier_index != len(suppliers)-1 and consumer_index != len(consumers)-1:
                lost_basis.insert(0, (supplier_index, consumer_index+1))
            # выбираем наименьшее количество товара, которое возможно переместить
            capacity[supplier_index][consumer_index] = min(supplier, consumer)
            # вычитаем получившееся количество товара у поставщика и потребителя
            consumers[consumer_index] -= capacity[supplier_index][consumer_index]
            suppliers[supplier_index] -= capacity[supplier_index][consumer_index]
            supplier_index += 1
        consumer_index += 1

    # вычисляем индексы ненулевых значений матрицы перевозок
    a = tuple(map(tuple, np.transpose(np.nonzero(capacity))))

    # вычисляем стоимость перевозок
    sum = 0
    for i in a:
        sum += costs[i]*capacity[i]

    # маскируем все нулевые элементы матрицы перевозок
    capacity = ma.array(capacity)
    capacity = ma.masked_equal(capacity, 0)
    # удаляем маску на "пропавших" базисных элементах
    for i in lost_basis:
        capacity[i] = ma.nomask
    return capacity, sum

# %%


def minElem(suppliers, consumers, costs, capacity):
    costs = ma.array(costs)
    # пока все элементы матрицы перевозок не замаскированы
    while costs.mask.all() == False:
        # находим индекс минимального элемента матрицы тарифов
        idx = np.unravel_index(costs.argmin(), costs.shape)
        # выбираем наименьшее количество товара, которое возможно переместить
        if suppliers[idx[0]] < consumers[idx[1]]:
            # записываем его в матрицу перевозок
            capacity[idx] = suppliers[idx[0]]
            # маскируем строку матрицы тарифов
            costs[idx[0]] = ma.masked
        else:
            capacity[idx] = consumers[idx[1]]
            # маскируем столбец матрицы тарифов
            costs[:, idx[1]] = ma.masked
        # вычитаем получившееся количество товара у поставщика и потребителя
        suppliers[idx[0]] -= capacity[idx]
        consumers[idx[1]] -= capacity[idx]

    costs.mask = ma.nomask
    # вычисляем индексы ненулевых значений матрицы перевозок
    a = tuple(map(tuple, np.transpose(np.nonzero(capacity))))
    # вычисляем стоимость перевозок
    sum = 0
    for i in a:
        sum += costs[i]*capacity[i]
    # маскируем все нулевые элементы матрицы перевозок
    capacity = ma.array(capacity)
    capacity = ma.masked_equal(capacity, 0)
    # проверка матрицы перевозок на вырожденность
    if capacity.count() != (len(suppliers)+len(consumers)-1):
        # маскируем элементы матрицы тарифов в соответствии
        # с ненулевыми элементами матрицы перевозок
        for i in a:
            costs[i] = ma.masked
        # добавляем недостающее количество базисных нулей в матрицу перевозок
        for _ in range(len(suppliers)+len(consumers)-1 - capacity.count()):
            # находим индекс минимального элемента матрицы тарифов
            idx = np.unravel_index(costs.argmin(), costs.shape)
            # маскируем этот элемент в матрице тарифов
            costs[idx] = ma.masked
            # удаляем маску этого элемента в матрице перевозок
            capacity[idx] = ma.nomask
    return capacity, sum


# %%
# graph = [[1, 2], [1, 4], [4, 5], [2, 5], [2, 3], [6, 5], [7, 5]]
# graph = [[1,2], [1,4], [2,1], [2,3], [2,5], [3,2], [4,1], [4,5], [5,2], [5,4], [5,6], [5,7], [6,5], [7,5]]
# graph = [[1, 3], [1, 2], [2, 3], [4, 2], [4, 3]]

def cyclesMethod(plan, costs):
    cycles = []
    # for edge in graph:
    #     for node in edge:
    #         if not node == 4:
    #             continue
    #         findNewCycles([node])
    i = 0
    j = 0
    for row in plan:
        for el in row:
            if ma.is_masked(el):
                plan[i][j] = 0
                node = (i,j)
                graph = graphBuilder(plan)
                cycles = findNewCycles(node, graph)
                plan[i][j] = ma.masked
            j+=1
        j=0
        i+=1
    # return cycles
    # for cy in cycles:
    #     path = [str(node) for node in cy]
    #     s = ",".join(path)
    #     print(s)


def findNewCycles(path, graph):
    start_node = path[0]
    next_node = None
    cycles = []
    sub = []

    # visit each edge and each node of each edge
    for edge in graph:
        node1, node2 = edge
        if start_node in edge:
            if node1 == start_node:
                next_node = node2
            else:
                next_node = node1
        if next_node != None and not visited(next_node, path):
            # neighbor node not on path yet
            sub = [next_node]
            sub.extend(path)
            # explore extended path
            findNewCycles(sub, graph)
        elif len(path) > 2 and next_node == path[-1]:
            # cycle found
            p = rotate_to_smallest(path)
            inv = invert(p)
            if isNew(p, cycles) and isNew(inv, cycles):
                cycles.append(p)
    return cycles


def invert(path):
    return rotate_to_smallest(path[::-1])

#  rotate cycle path such that it begins with the smallest node


def rotate_to_smallest(path):
    n = path.index(min(path))
    return path[n:]+path[:n]


def isNew(path, cycles):
    return not path in cycles


def visited(node, path):
    return node in path


# %%


def potential(costs, mask):
    u = ma.masked_all(costs.shape[0], dtype=int)
    v = ma.masked_all(costs.shape[1], dtype=int)
    # предположим, первый элемент матрицы u равен нулю
    u[0] = 0

    # вычисляем потенциалы u и v
    while True:
        i = 0
        j = 0
        # проходим по матрице стоимостей
        for row in mask:
            for el in row:
                # если стоимость замаскирована, невозможно вычислить u или v
                if el == True:
                    j += 1
                    continue
                # вычисляем u или v, если они еще не вычислены
                if ma.is_masked(u[i]) and not ma.is_masked(v[j]):
                    u[i] = costs[i][j] - v[j]
                if ma.is_masked(v[j]) and not ma.is_masked(u[i]):
                    v[j] = costs[i][j] - u[i]
                j += 1
            j = 0
            i += 1
        # проверяем, все ли значения массивов u и v вычислены
        if not ma.is_masked(u) and not ma.is_masked(v):
            break

    # вычисляем косвенные тарифы
    indCosts = costs.copy()
    i = 0
    j = 0
    for row in mask:
        for el in row:
            if el == True:
                # s = c - c'; c' = u + v
                indCosts[i][j] = costs[i][j] - (u[i]+v[j])
            j += 1
        j = 0
        i += 1

    # вычисляем индексы отрицательных значений матрицы косвенных тарифов
    costsIndex = tuple(map(tuple, np.transpose(np.where(indCosts < 0))))

    return u, v, indCosts, costsIndex


# %%
def graphBuilder(capacity):
    i = 0
    j = 0
    arr = []
    for row in capacity:
        for el in row:
            if ma.is_masked(el):
                j += 1
                continue

            k = 1
            while True:
                res, masked = checkNeighbour(capacity, i-k, j)
                if res:
                    arr.append(((i, j), (i-k, j)))
                    break
                if not res and not masked:
                    break
                k += 1

            k = 1
            while True:
                res, masked = checkNeighbour(capacity, i+k, j)
                if res:
                    arr.append(((i, j), (i+k, j)))
                    break
                if not res and not masked:
                    break
                k += 1

            k = 1
            while True:
                res, masked = checkNeighbour(capacity, i, j-k)
                if res:
                    arr.append(((i, j), (i, j-k)))
                    break
                if not res and not masked:
                    break
                k += 1

            k = 1
            while True:
                res, masked = checkNeighbour(capacity, i, j+k)
                if res:
                    arr.append(((i, j), (i, j+k)))
                    break
                if not res and not masked:
                    break
                k += 1
            j += 1
        i += 1
        j = 0

    for row in arr:
        arr.remove((row[1], row[0]))

    return arr


def checkNeighbour(arr, i, j):
    res = True
    masked = False
    if i < 0 or j < 0:
        return False, False
    try:
        res = not ma.is_masked(arr[i][j])
        masked = not res
    except IndexError:
        return False, False
    return res, masked

# %%


def main():
    suppliers, consumers, costs, capacity = initProgram('data/Problem5.json')
    northWestCap, northWestSum = northWest(suppliers, consumers, costs, capacity)
    print(f'Получившийся опорный план:\n{northWestCap}')
    print(f'Суммарные затраты перевозок: {northWestSum}')
    
    suppliers, consumers, costs, capacity = initProgram('data/Problem5.json')
    minElCap, minElSum = minElem(suppliers, consumers, costs, capacity)
    print(f'Получившийся опорный план:\n{minElCap}')
    print(f'Суммарные затраты перевозок: {minElSum}')
    
    u, v, inderectCosts, costsIndex = potential(costs, ma.getmask(northWestCap))
    print(f'Метод северо-западного угла\nПотенциал поставщика u:{u}\nПотенциал потребителя v:{v}\nКосвенные тарифы:\n{inderectCosts}\nПотенциальные клетки для загрузки (с отрицательными косвенными тарифами):{costsIndex}\n')
    u, v, inderectCosts, costsIndex = potential(costs, ma.getmask(minElCap))
    print(f'Метод минимального элемента\nПотенциал поставщика u:{u}\nПотенциал потребителя v:{v}\nКосвенные тарифы:\n{inderectCosts}\nПотенциальные клетки для загрузки (с отрицательными косвенными тарифами):{costsIndex}\n')


# %%
main()

# %%
