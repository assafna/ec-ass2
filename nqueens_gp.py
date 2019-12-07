from functools import partial

import numpy as np
import pygraphviz as pgv

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


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

    def move_index1_down(self):
        self.index1 -= 1 if self.index1 > 0 else self.index1

    def move_index2_up(self):
        self.index2 += 1 if self.index2 < self.n - 1 else self.index2

    def move_index2_down(self):
        self.index2 -= 1 if self.index2 > 0 else self.index2

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

        if score == 0:
            print(self.board)

        return score


nqueens = NQueens(n=8)

pset = gp.PrimitiveSet(name='nqueens', arity=0)
pset.addPrimitive(primitive=prog2, arity=2)
pset.addTerminal(nqueens.move_index1_up)
pset.addTerminal(nqueens.move_index1_down)
pset.addTerminal(nqueens.move_index2_up)
pset.addTerminal(nqueens.move_index2_down)
pset.addTerminal(nqueens.swap)

creator.create(name='FitnessMin', base=base.Fitness, weights=(-1.0,))
creator.create(name='Individual', base=gp.PrimitiveTree, fitness=creator.FitnessMin)

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
        draw_graph(nodes, edges, labels)
    return score,


toolbox.register('evaluate', eval_nqueens)
toolbox.register('select', tools.selTournament, tournsize=7)
toolbox.register('mate', gp.cxOnePoint)
toolbox.register('expr_mut', gp.genFull, min_=0, max_=2)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


def draw_graph(nodes, edges, labels):
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("tree.pdf")


if __name__ == '__main__':
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 100, stats, halloffame=hof)
