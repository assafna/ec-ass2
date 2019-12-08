import operator
from functools import partial

import numpy as np

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
                temp = list(np.diag(board_2d, k=-index))
                score += max(list(np.diag(board_2d, k=-index)).count(1) - 1, 0)
            score += max(list(np.diag(board_2d, k=index)).count(1) - 1, 0)

        # second diagonal
        for _index in range(self.n - 1):
            if _index != 0:
                score += max(list(np.diag(np.fliplr(board_2d), k=-_index)).count(1) - 1, 0)
            score += max(list(np.diag(np.fliplr(board_2d), k=_index)).count(1) - 1, 0)

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
    score = nqueens.eval_score()
    if score == 0:
        nodes, edges, labels = gp.graph(individual)
        # print_graph(edges, labels)
    return score,


# eval and mutate
toolbox.register('evaluate', eval_nqueens)
toolbox.register('select', tools.selTournament, tournsize=7)
toolbox.register('mate', gp.cxOnePoint)
toolbox.register('expr_mut', gp.genFull, min_=0, max_=2)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# bloat control
toolbox.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value=20))
toolbox.decorate('mutate', gp.staticLimit(key=operator.attrgetter('height'), max_value=20))


def print_graph(edges, labels):
    for edge in edges:
        print(str(edge[0]) + '.' + labels[edge[0]] + '->' + str(edge[1]) + '.' + labels[edge[1]])


if __name__ == '__main__':
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)

    algorithms.eaSimple(population=pop,
                        toolbox=toolbox,
                        cxpb=0.9,
                        mutpb=0.2,
                        ngen=100,
                        stats=stats,
                        halloffame=hof
                        )
