import math
import numpy as np
import GUI as gui


class State:
    def __init__(self, number: int, cost: int, goal: bool, actions: int):
        self.number = number  # label of this state
        self.cost = cost  # static cost taken from following any action
        self.goal = goal  # goal flag
        self.T = [[] for i in range(actions)]  # a list of successors states and probabilities following each action

    def __str__(self):
        return str(self.number)

    def __repr__(self):
        return str(self.number)


class MDP:
    def __init__(self, Nx, Ny, actions,T):
        self.A = actions  # number of actions
        self.Nx = Nx
        self.Ny = Ny
        self.T = T
        self.S = [State(index + 1, 1, False, actions) for index in range(Nx * Ny + 1)]


def dual_criterion_risk_sensitive(mdp, risk_factor=-0.01, minimum_error=0.0001):
    """
    update the values of the state's object and also returns a list with the values of each state
    :param mdp: a MDP object
    :param risk_factor: used for weight the risk value
    :param minimum_error: precision measured by the number of zeros on epsilon
    :return: a list of probability to goal values, a list of risk values, and a list of best actions, both for each state
    """
    # initializations
    delta1 = float('Inf')
    delta2 = 0
    v_lambda = [0] * len(mdp.S)  # risk values
    pg = [0] * len(mdp.S)  # probability to reach the goal
    best_actions = [0] * len(mdp.S)  # best policy that gave us the best values

    for s in mdp.S:  # goal states treatment
        if s.goal:
            v_lambda[s.number - 1] = -1 if risk_factor > 0 else 1  # -sgn(risk_factor)
            pg[s.number - 1] = 1

    def linear_combination(t, vector):
        """
        an auxiliar function in order to clear the code
        :param t: a transaction list obtained by following an action on a state, usually mdp.S[i].T[a]
        :param vector: a vector whose will be linear combined with the states and probabilities of t
        :return: a summation representing the linear combination
        """
        summ = 0
        for s2 in t:
            summ += s2['prob'] * vector[s2['state'] - 1]
        return summ

    while delta1 >= minimum_error or delta2 <= 0:
        v_previous = v_lambda.copy()
        p_previous = pg.copy()

        A = [[] for i in range(len(mdp.S))]  # max prob actions for each state

        for s in mdp.S:

            if s.goal:
                continue

            # probability value section
            max_prob_t = max(s.T, key=lambda t: linear_combination(t, p_previous))
            pg[s.number - 1] = linear_combination(max_prob_t, p_previous)

            # keeping all the actions that tie with the max prob action in the A list of this state
            A[s.number - 1] = []
            for t_index in range(len(s.T)):  # the indexes represent the taken actions
                if linear_combination(s.T[t_index], p_previous) == pg[s.number - 1]:
                    A[s.number - 1].append(t_index)

            # risk value section
            best_action = max(A[s.number - 1], key=lambda a: math.exp(risk_factor * s.cost) *
                                                                        linear_combination(s.T[a], v_previous))
            v_lambda[s.number - 1] = math.exp(risk_factor * s.cost) * linear_combination(s.T[best_action], v_previous)
            best_actions[s.number-1] = best_action


        # updating deltas
        n_delta1 = -float('Inf')
        n_delta2 = float('Inf')
        for s in mdp.S:

            if s.goal:
                continue

            n_delta1 = max(n_delta1, abs(v_lambda[s.number-1] - v_previous[s.number-1]) + abs(
                pg[s.number-1] - p_previous[s.number-1]))

            all_actions = set([i for i in range(mdp.A)])
            max_prob_actions = set(A[s.number-1])
            poor_actions = all_actions - max_prob_actions
            for a in poor_actions:
                n_delta2 = min(n_delta2, pg[s.number-1] - linear_combination(s.T[a], pg))

        delta1 = n_delta1
        ''' if there will be no poor actions, delta2 assumes the best value, Inf'''
        delta2 = n_delta2


    return pg,v_lambda,best_actions

def Problem(nx,ny,na):
    '''
    :param nx: number of states in the x axis
    :param ny: number of states in the y axis
    :param na: number of actions
    '''
    'define a value'
    T = np.zeros((na,nx,ny))

    x=0
    for a in range(na):  # each state
        for y in range(ny -1) :
            x = a+y
            if (x>10): 
                  x=10
            T[a,y,nx-1] = 0.08*x + 0.02*a
            T[a,y,x] = 1- T[a,y,nx-1]
    return T

# Test Script
T = Problem(12, 12,4)

mdp = MDP(12, 12, 4,T)

# # Dual criterion
(prob_to_goal,risk_values,best_actions) = dual_criterion_risk_sensitive(mdp, -0.01, 0.0001)
gui.plot(mdp,[prob_to_goal,risk_values],['PG','V'],best_actions)