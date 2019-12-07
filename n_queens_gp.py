import random
import operator

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

N_QUEENS = 8


class NQueens:
    def __init__(self, _n):
        self.n = _n
        self.score = _n * _n
        self.current_num_of_queens = 0
        self.board = list([0] * _n)

    # def reset_board(self):
    #     self.score = 0
    #     self.board = list([0] * self.n)
    #
    # def add_queen(self, _row_index, _col_index):
    #     if self.current_num_of_queens <= self.n:
    #         self.board[_row_index] = _col_index
    #         self.current_num_of_queens += 1
    #
    # def move_queen_up(self, _index):
    #     if self.board[_index] != 0 and _index < self.n:
    #         self.board[_index] += 1
    #
    # def move_queen_down(self, _index):
    #     if self.board[_index] != 0 and _index > 0:
    #         self.board[_index] -= 1

    def build_2d_board(self):
        _2d_board = []
        for _index in range(self.n):
            _row = [0] * self.n
            _row[self.board[_index]] = 1
            _2d_board.append(_row)

        return _2d_board

    # def run(self, _routine):
    #     self.reset_board()
    #     _routine()

    def rows_conflicts(self):
        _2d_board = self.build_2d_board()
        for _row_index in range(self.n):
            _row_content = [_2d_board[_col_index][_row_index] for _col_index in range(self.n)]
            return _row_content.count(1) - 1

    def diagonals_conflicts(self):
        _2d_board = self.build_2d_board()
        _score = 0

        # main diagonal
        for _index in range(self.n - 1):
            if _index != 0:
                _score += list(np.diag(_2d_board, k=-_index)).count(1) - 1
            _score += list(np.diag(_2d_board, k=_index)).count(1) - 1

        # second diagonal
        for _index in range(self.n - 1):
            if _index != 0:
                _score += list(np.diag(np.transpose(_2d_board), k=-_index)).count(1) - 1
            _score += list(np.diag(np.transpose(_2d_board), k=_index)).count(1) - 1

        return _score


def evalNQueens(_individual):
    _routine = gp.compile(_individual, pset)
    nqueens.run(_routine=_routine)
    return nqueens.score


if __name__ == '__main__':
    nqueens = NQueens(_n=N_QUEENS)

    pset = gp.PrimitiveSet(name='main', arity=2)

    # functions
    # pset.addPrimitive(NQueens.add_queen, 2)
    # pset.addPrimitive(NQueens.move_queen_up, 1)
    # pset.addPrimitive(NQueens.move_queen_down, 1)

    pset.addPrimitive(operator.add, 2)
    pset.addTerminal(NQueens.rows_conflicts)
    pset.addTerminal(NQueens.diagonals_conflicts)

    # terminals
    for i in range(N_QUEENS):
        pset.addTerminal(i)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=N_QUEENS)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evalNQueens)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 40, stats, halloffame=hof)
