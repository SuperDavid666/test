import numpy as np
import pandas as pd
import time
import math
from PSO import PSO


class SPSO:

    def __init__(self, func, cbest_x, cbest_y, n_dim=None, pop=40, max_iter=2000, lb=-1e5, ub=1e5, w=0.4, c1=2, c2=2):

        self.func = func
        self.w = w  # inertia
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.pop = pop  # number of particles
        self.n_dim = n_dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.cx = (cbest_x - self.lb) / (self.ub - self.lb)
        self.cbest_x = cbest_x
        self.cbest_y = cbest_y

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))
        self.v_high = (self.ub - self.lb) * 0.2
        self.V = np.random.uniform(low=-self.v_high, high=self.v_high,
                                   size=(self.pop, self.n_dim))  # speed of particles
        self.Y = self.cal_y()  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = self.Y.copy()  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

    def update_V(self):
        r1 = np.random.rand(self.pop, self.n_dim)
        r2 = np.random.rand(self.pop, self.n_dim)
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)

        self.V = np.clip(self.V, -self.v_high, self.v_high)

    def update_X(self):
        self.X = self.X + self.V
        self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        """
        personal best
        :return:
        """
        self.need_update = self.pbest_y > self.Y
        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        """
        global best
        :return:
        """
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min][0]

    def chaos_search(self):
        for i in range(100):
            self.cx = 4 * self.cx * (1 - self.cx)
            x = self.cx * (self.ub - self.lb) + self.lb
            y = self.func(np.array([x]))
            if y < self.cbest_y:
                print(1)
                self.cbest_y = y
                self.cbest_x = x

        if self.cbest_y < self.gbest_y:
            print(2)
            self.gbest_y = self.cbest_y
            self.gbest_x = self.cbest_x
        # else:
        #     self.cbest_x = self.gbest_x
        #     self.cx = (self.cbest_x - self.lb) / (self.ub - self.lb)
        #     self.cbest_y = self.gbest_y

    def run(self, max_iter=None, precision=None, N=20):
        '''
        precision: None or float
            If precision is None, it will run the number of max_iter steps
            If precision is a float, the loop will stop if continuous N difference between pbest less than precision
        N: int
        '''
        self.max_iter = max_iter or self.max_iter
        c = 0
        for iter_num in range(self.max_iter):
            self.chaos_search()
            self.update_V()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()
            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
                    c = c + 1
                    if c > N:
                        break
                else:
                    c = 0

            self.gbest_y_hist.append(self.gbest_y)

        return self.gbest_x, self.gbest_y

    fit = run


class TCOA:
    """
    Implementation of two chaos mechanism optimization algorithm

    Parameters
    ----------------------
    func : function
        The func you want to do optimal
    n_dim : int
        Number of dimension, which is number of parameters of func.
    lb : array_like
        The lower bound of every variables of func
    ub : array_like
        The upper bound of every variables of func
    Attributes
    ----------------------
    max_min : array_like, shape is (n_dim, )
        Upper-bound minus lower-bound
    area_dist : float
        The length of the optimization space, which is
        The 2-norm of (upper-bounds - lower-bounds)
    CX : array_like, shape is (2, n_dim)
        The two chaos vectors
    X : array_like, shape is (2, n_dim)
        The two optimization vectors
    y : float
        the y value for current X
    best_X : array_like, shape is (2, n_dim)
        global best X
    best_y : array_like, shape is (2, 1)
        global best y
    """

    def __init__(self, func, n_dim=None, lb=-1e5, ub=1e5):

        # self.func = func_transformer(func)  # the function you want you optimize
        self.func = func
        self.n_dim = n_dim  # the number of variables of func

        # the lower bound and upper bound of each variable
        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        self.max_min = self.ub - self.lb  # the range of optimization variables
        self.area_dist = np.linalg.norm(self.max_min)  # the length of diagonal of optimization space
        assert self.n_dim == len(self.lb) == len(self.ub), "dim == len(lb) == len(ub) is not True"
        assert np.all(self.ub > self.lb), "upper-bound must be greater than lower-bound"

        # initialize two chaos variable vectors
        # CX1 = [0.632123]  # f3-f9
        # CX2 = [0.837253]
        # CX1 = [0.]    # f10-f11
        # CX2 = [0.]
        # CX1 = [0.31122]    # f12
        # CX2 = [0.01121]
        # CX1 = [0.92]  # f13 - 14
        # CX2 = [0.18]
        # CX1 = [0.04]  # f15 - 17
        # CX2 = [0.71]
        # CX1 = [0.13598]  # f18 - 28
        # CX2 = [0.978325]
        # CX1 = [0.13598]  # f18 - 29
        # CX2 = [0.978325]
        CX1 = [0.135982]  # f30
        CX2 = [0.978325]
        # CX1 = [np.random.rand()]
        # CX2 = [np.random.rand()]
        for i in range(self.n_dim - 1):
            CX1.append(CX1[i] + 0.0001)
            CX2.append(CX2[i] + 0.0001)
        self.CX = np.array([CX1, CX2])
        # self.X = np.zeros((2, self.n_dim))
        self.X = self.CX_to_X()
        self.Y = self.cal_Y()
        print("CX", self.CX)
        # print("X", self.X)
        # print("Y", self.Y)

        self.pbest_X = self.X.copy()  # the best location X for each chaos search
        self.pbest_y = self.Y.copy()  # the best y for each best X

        self.gbest_X = np.zeros(self.n_dim)  # global best X
        self.gbest_y = np.inf  # global best y

    def CX_to_X(self):
        # Transform chaotic variables to optimization variables
        self.X = self.CX * self.max_min + self.lb
        return self.X

    def cal_Y(self):
        # calculate the y value for X
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y

    def logistic_mapping(self, cx):
        # logistic mapping function
        cx = 4 * cx * (1 - cx)
        return cx

    def tent_mapping(self, cx):
        # tent mapping function
        beta = 0.4
        new_cx = []
        for x in cx:
            if 0 <= x < beta:
                x = x / beta
            elif beta <= x < 1:
                x = (1 - x) / (1 - beta)
            new_cx.append(x)
        return np.array(new_cx)

    def chaotic_mapping(self):
        # implement chaotic mapping
        self.CX[0] = self.logistic_mapping(self.CX[0])
        self.CX[1] = self.tent_mapping(self.CX[1])

        return self.CX

    def decrease_search_space(self, gamma, dist):
        # decrease the optimization space
        self.lb = np.array([max(self.lb[i], np.min(self.pbest_X[:, i]) - 1.5 * gamma * dist)
                            for i in range(self.n_dim)])
        self.ub = np.array([min(self.ub[i], np.max(self.pbest_X[:, i]) + 1.5 * gamma * dist)
                            for i in range(self.n_dim)])
        #
        # self.lb = np.array([max(self.lb[i], np.min(self.pbest_X[:, i]) - 1.5 * gamma * abs(self.pbest_X[0, i] - self.pbest_X[1, i]))
        #                     for i in range(self.n_dim)])
        # self.ub = np.array([min(self.ub[i], np.max(self.pbest_X[:, i]) + 1.5 * gamma * abs(self.pbest_X[0, i] - self.pbest_X[1, i]))
        #                     for i in range(self.n_dim)])

        self.max_min = self.ub - self.lb
        self.area_dist = np.linalg.norm(self.max_min)

    def update_pbest(self):
        # update the pbest X and pbest y
        need_update = self.Y < self.pbest_y
        self.pbest_y = np.where(need_update, self.Y, self.pbest_y)
        self.pbest_X = np.where(need_update, self.X, self.pbest_X)

    def run(self, decrease_iter=3000):
        """
        Run two chaos mechanism optimization algorithm
        decrease_iter : int
            decrease the searching space when iteration reaches this value
        """
        gamma = 0.25  # a parameter belong (0, 0.25]
        iter_num = 0
        while True:
            iter_num += 1
            self.CX_to_X()
            self.cal_Y()
            self.update_pbest()
            if iter_num > decrease_iter:

                # cal the distance of two chaos
                dist = np.linalg.norm(self.pbest_X[0] - self.pbest_X[1])
                if dist < gamma * self.area_dist:
                    print("Distance of two chaos: ", dist)
                    print("Distance of seaching space: ", self.area_dist)
                    self.decrease_search_space(gamma, dist)
                    print("=" * 100)
                    print("Searching space: ")
                    print(self.lb)
                    print(self.ub)
                    print("=" * 100)

                    break

            self.chaotic_mapping()

        idx_min = self.pbest_y.argmin()
        self.gbest_x = self.pbest_X[idx_min]
        self.gbest_y = self.pbest_y[idx_min][0]
        print("cbest_x: ", self.gbest_x)
        print("cbest_y: ", self.gbest_y)

        # simple_pso
        # spso = SPSO(func=self.func, cbest_x=self.gbest_x, cbest_y=self.gbest_y, n_dim=self.n_dim, pop=40,
        #             max_iter=1999, lb=self.lb, ub=self.ub)
        # spso.run()
        # # # print(pso.gbest_y_hist)
        # return spso.gbest_x, spso.gbest_y, spso.gbest_y_hist

        # original_pso
        pso = PSO(func=self.func,
                  n_dim=self.n_dim, pop=40, max_iter=1999, lb=self.lb, ub=self.ub)
        pso.run()

        return pso.gbest_x, pso.gbest_y, pso.gbest_y_hist


if __name__ == '__main__':
    from cec2017.functions import f30

    # dims = [10, 30, 50]
    dims = [10]
    fitness = 3000
    for dim in dims:

        result = {"y": [], "et": []}
        for i in range(50):
            print("%d start !!!" % (i + 1))

            s = time.time()
            pso = TCOA(func=f30, n_dim=dim, lb=[-100] * dim, ub=[100] * dim)
            best_x, best_y, hist = pso.run()
            e = time.time()

            # alter ===================================================================================
            error = np.abs(best_y[0] - fitness)
            et = e - s

            result["y"].append(error)
            result["et"].append(et)

            print("cost_time: ", et)
            print("best_x: ", best_x)
            print("error: ", error)
            # print(pso.gbest_y_hist)
            print("=" * 100)

            # if i == 24:
            #     import json
            #
            #     with open("DCS-PSO_30_cec17_f1.txt", 'w', encoding='utf-8') as f:
            #         f.write(json.dumps(hist))
            #     break
            # break

        df = pd.DataFrame(result)
        print(df['y'].mean())
        print(df['y'].std())
        df.loc[len(df)] = [df['y'].mean(), df['y'].std()]
        df.to_excel("DCS-PSO_30_cec17_f30.xlsx", index=False)
