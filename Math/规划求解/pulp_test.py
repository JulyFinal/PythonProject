# -*- coding: utf-8 -*-
import pulp as pulp

def solve_ilp(objective , constraints) :
    print(objective)
    print(constraints)
    prob = pulp.LpProblem('LP1' , pulp.LpMaximize)
    prob += objective
    for cons in constraints :
        prob += cons
    print(prob)
    status = prob.solve()
    if status != 1 :
        #print 'status'
        #print status
        return None
    else :
        #return [v.varValue.real for v in prob.variables()]
        return [v.varValue.real for v in prob.variables()]



#解如下整数线性规划
#maximize  z = c*x = 3*x1 + 4*x2 + 5*x3
#subject to :
#x1 2 3 >= 0
#x1 + 2*x2 <  20
#x2 + 3*x3 <= 40

V_NUM = 3
#变量，直接设置下限
variables = [pulp.LpVariable('X%d'%i , lowBound = 0 , cat = pulp.LpInteger) for i in range(0 , V_NUM)]
#目标函数
c = [3 , 4 , 5]
objective = sum([c[i]*variables[i] for i in range(0 , V_NUM)])
#约束条件
constraints = []

a1 = [1 , 2 , 0]
constraints.append(sum([a1[i]*variables[i] for i in range(0 , V_NUM)]) <= 100)
a2 = [0 , 1 , 3]
constraints.append(sum([a2[i]*variables[i] for i in range(0 , V_NUM)]) <= 40)
print(constraints)

res = solve_ilp(objective , constraints)
print(res)