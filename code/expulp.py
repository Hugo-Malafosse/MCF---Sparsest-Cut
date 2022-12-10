
from pulp import LpProblem, LpMinimize, LpVariable, LpInteger, PULP_CBC_CMD, LpStatus, LpConstraintLE, LpAffineExpression, value

if __name__ == '__main__':
    # Paramètres

    D = 10

    # Définition des variables

    zsu = LpVariable('zsu', lowBound=0)
    zsv = LpVariable('zsv', lowBound=0)
    zut = LpVariable('zut', lowBound=0)
    zvt = LpVariable('zvt', lowBound=0)
    zuv = LpVariable('zuv', lowBound=0)
    y = LpVariable('y', lowBound=0)

    # Problème

    Lp_prob = LpProblem('Problem', LpMinimize)

    # Create problem Variables
    Lp_prob += 4 * zsu + 2 * zsv + 1 * zut + 6 * zvt + 3 * zuv

    # Constraints:
    Lp_prob += zsu + zut >= y
    Lp_prob += zsv + zvt >= y
    Lp_prob += zsu + zuv + zvt >= y
    Lp_prob += zsv + zuv + zut >= y
    Lp_prob += y >= 1/D
    # Display the problem
    print(Lp_prob)

    status = Lp_prob.solve()  # Solver
    print(LpStatus[status])  # The solution status

    # Printing the final solution
    print(value(zsu), value(zsv), value(zut), value(zvt), value(zuv), value(y), value(Lp_prob.objective))
