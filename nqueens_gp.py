import operator
import statistics
from functools import partial

import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

NQUEENS_N = 8


def progn(*args):
    for arg in args:
        arg()


def prog2(out1, out2):
    return partial(progn, out1, out2)


class NQueens(object):
    def __init__(self, n):
        self.n = n
        self.board = list(range(self.n))
        self.index1 = 0
        self.index2 = 0

    def reset(self):
        self.board = list(range(self.n))
        self.index1 = 0
        self.index2 = 0

    def move_index1_up(self):
        self.index1 += 1 if self.index1 < self.n - 1 else self.index1

    def move_index1_up2(self):
        self.index1 += 2 if self.index1 < self.n - 2 else self.index1

    def move_index1_down(self):
        self.index1 -= 1 if self.index1 > 0 else self.index1

    def move_index1_down2(self):
        self.index1 -= 2 if self.index1 > 1 else self.index1

    def move_index1_start(self):
        self.index1 = 0

    def move_index1_end(self):
        self.index1 = self.n

    def move_index2_up(self):
        self.index2 += 1 if self.index2 < self.n - 1 else self.index2

    def move_index2_up2(self):
        self.index2 += 2 if self.index2 < self.n - 2 else self.index2

    def move_index2_down(self):
        self.index2 -= 1 if self.index2 > 0 else self.index2

    def move_index2_down2(self):
        self.index2 -= 2 if self.index2 > 1 else self.index2

    def move_index2_start(self):
        self.index2 = 0

    def move_index2_end(self):
        self.index2 = self.n

    def swap(self):
        if 0 <= self.index1 < self.n and 0 <= self.index2 < self.n:
            self.board[self.index1], self.board[self.index2] = self.board[self.index2], self.board[self.index1]

    def run(self, routine):
        self.reset()
        routine()

    def eval_score(self):
        score = 0

        board_2d = []
        for index in range(self.n):
            row = [0] * self.n
            row[self.board[index]] = 1
            board_2d.append(row)

        # main diagonal
        for index in range(self.n - 1):
            if index != 0:
                score += max(list(np.diag(board_2d, k=-index)).count(1) - 1, 0)
            score += max(list(np.diag(board_2d, k=index)).count(1) - 1, 0)

        # second diagonal
        for _index in range(self.n - 1):
            if _index != 0:
                score += max(list(np.diag(np.fliplr(board_2d), k=-_index)).count(1) - 1, 0)
            score += max(list(np.diag(np.fliplr(board_2d), k=_index)).count(1) - 1, 0)

        return score

    def eval_score2(self):
        score = 0

        board_2d = []
        for index in range(self.n):
            row = [0] * self.n
            row[self.board[index]] = 1
            board_2d.append(row)

        for row_index in range(self.n):
            for col_index in range(self.n):
                if board_2d[row_index][col_index] == 1:
                    queen_score = 0
                    # right down
                    index = 1
                    while True:
                        if row_index + index < self.n and col_index + index < self.n:
                            if board_2d[row_index + index][col_index + index] == 1:
                                queen_score += 1
                            index += 1
                        else:
                            break
                    # left up
                    index = 1
                    while True:
                        if row_index - index >= 0 and col_index - index >= 0:
                            if board_2d[row_index - index][col_index - index] == 1:
                                queen_score += 1
                            index += 1
                        else:
                            break
                    # left down
                    index = 1
                    while True:
                        if row_index + index < self.n and col_index - index >= 0:
                            if board_2d[row_index + index][col_index - index] == 1:
                                queen_score += 1
                            index += 1
                        else:
                            break
                    # right up
                    index = 1
                    while True:
                        if row_index - index >= 0 and col_index + index < self.n:
                            if board_2d[row_index - index][col_index + index] == 1:
                                queen_score += 1
                            index += 1
                        else:
                            break
                    score += queen_score

        return score


# agent
nqueens = NQueens(n=NQUEENS_N)

pset = gp.PrimitiveSet(name='nqueens', arity=0)

# primitives
pset.addPrimitive(primitive=prog2, arity=2)

# terminals
pset.addTerminal(nqueens.move_index1_up)
pset.addTerminal(nqueens.move_index1_up2)
pset.addTerminal(nqueens.move_index1_down)
pset.addTerminal(nqueens.move_index1_down2)
pset.addTerminal(nqueens.move_index1_start)
pset.addTerminal(nqueens.move_index1_end)

pset.addTerminal(nqueens.move_index2_up)
pset.addTerminal(nqueens.move_index2_up2)
pset.addTerminal(nqueens.move_index2_down)
pset.addTerminal(nqueens.move_index2_down2)
pset.addTerminal(nqueens.move_index2_start)
pset.addTerminal(nqueens.move_index2_end)

pset.addTerminal(nqueens.swap)

# fitness
creator.create(name='FitnessMin', base=base.Fitness, weights=(-1.0,))
creator.create(name='Individual', base=gp.PrimitiveTree, fitness=creator.FitnessMin)

# population
toolbox = base.Toolbox()
toolbox.register('expr_init', gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)


def eval_nqueens(individual):
    routine = gp.compile(expr=individual, pset=pset)
    nqueens.run(routine)
    score = nqueens.eval_score2()
    return score,


# eval and mutate
toolbox.register('evaluate', eval_nqueens)
toolbox.register('select', tools.selTournament, tournsize=7)
toolbox.register('mate', gp.cxOnePoint)
toolbox.register('expr_mut', gp.genFull, min_=0, max_=2)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# bloat control
toolbox.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value=9))
toolbox.decorate('mutate', gp.staticLimit(key=operator.attrgetter('height'), max_value=9))


# mp
pool = multiprocessing.Pool()
toolbox.register('map', pool.map)


def print_graph(edges, labels):
    for edge in edges:
        print(str(edge[0]) + '.' + labels[edge[0]] + '->' + str(edge[1]) + '.' + labels[edge[1]])


def plot(_log1, _log2, _log3, _title):
    data = pd.DataFrame(columns=('Generation', 'Best Fitness', 'Worst Fitness', 'Average Fitness', 'Median Fitness'))

    gen = _log1.select('gen')
    median1, median2, median3 = _log1.select('med'), _log2.select('med'), _log3.select('med')
    avg1, avg2, avg3 = _log1.select('avg'), _log2.select('avg'), _log3.select('avg')
    min1, min2, min3 = _log1.select('min'), _log2.select('min'), _log3.select('min')
    max1, max2, max3 = _log1.select('max'), _log2.select('max'), _log3.select('max')

    median = [statistics.mean(k) for k in zip(median1, median2, median3)]
    avg = [statistics.mean(k) for k in zip(avg1, avg2, avg3)]
    min_ = [statistics.mean(k) for k in zip(min1, min2, min3)]
    max_ = [statistics.mean(k) for k in zip(max1, max2, max3)]

    data['Generation'] = gen
    data['Best Fitness'] = min_
    data['Worst Fitness'] = max_
    data['Average Fitness'] = avg
    data['Median Fitness'] = median

    data.to_csv(_title + '.csv')

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, min_, 'b-', label='Best Fitness')
    ax1.set(xlabel='Generation', ylabel='Fitness',
            title=_title + ' Avg Results')
    ax1.grid()
    line2 = ax1.plot(gen, max_, 'r-', label='Worst Fitness')
    line3 = ax1.plot(gen, avg, 'g-', label='Average Fitness')
    line4 = ax1.plot(gen, median, 'y-', label='Median Fitness')
    lns = line1 + line2 + line3 + line4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')
    plt.show()


def run():
    pop = toolbox.population(n=150)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('med', np.median)
    stats.register('min', np.min)
    stats.register('max', np.max)

    _, log = algorithms.eaSimple(population=pop, toolbox=toolbox, cxpb=0.9, mutpb=0.2, ngen=150, stats=stats,
                                  halloffame=hof)

    return log, hof


if __name__ == '__main__':
    log1, hof1 = run()
    log2, hof2 = run()
    log3, hof3 = run()

    plot(log1, log2, log3, _title='N=8')

    nodes, edges, labels = gp.graph(hof3[0])
    print_graph(edges, labels)
