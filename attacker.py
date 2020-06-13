
from gurobipy import *
import numpy as np





#====================  Attack by deleting edges
def attack_del_LP(model, budget_num,node_i,node_j,n,d_n, C, a_ij):
    
    edge_num = int(len(node_i))
    
    
    
    #========= model ==========
    w_n_1 = model[0:d_n]
    w_n_2 = model[d_n:2*d_n]
    w_e_1 = model[2*d_n]
    w_e_2 = model[2*d_n +1]


    #================ Linear Programming =================

    m = Model()
    m.setParam('OutputFlag', 0)

    e = []
    z = []
    for k in range(edge_num):
        e.append(m.addVar(lb = 0.0,ub = 1.0, vtype = GRB.CONTINUOUS))
        z.append(m.addVars(2,lb = 0.0, ub = 1.0, vtype = GRB.CONTINUOUS))

    y_n = []
    for i in range(n):
        y_n.append(m.addVars(2,lb = 0.0,ub = 1.0,vtype = GRB.CONTINUOUS))


    #---- obj

    obj = LinExpr(0.0)

    for i in range(n):
        obj += C[i*2]*y_n[i][0] + C[i*2+1]*y_n[i][1]

    for k in range(edge_num):
        obj += w_e_1*z[k][0] + w_e_2*z[k][1] - (a_ij[k*2] + a_ij[k*2+1])*e[k]


    m.setObjective(obj,GRB.MAXIMIZE)


    for i in range(n):
        m.addConstr(y_n[i][0] + y_n[i][1] - 1.0 == 0.0)


    for k in range(edge_num):
        node_index_i = node_i[k]
        node_index_j = node_j[k]
        m.addConstr(z[k][0] <= y_n[node_index_i][0])
        m.addConstr(z[k][0] <= y_n[node_index_j][0])
        m.addConstr(z[k][0] <= e[k])

        m.addConstr(z[k][1] <= y_n[node_index_i][1])
        m.addConstr(z[k][1] <= y_n[node_index_j][1])
        m.addConstr(z[k][1] <= e[k])

    constr = LinExpr(0.0)
    constr.addTerms(np.ones(edge_num),e)

    m.addConstr(edge_num - constr - budget_num <= 0.0)


    m.optimize()


    y_n_result = []
    for i in range(n):
        y_n_result.append(y_n[i][0].x)
        y_n_result.append(y_n[i][1].x)

    e_result = []

    for k in range(edge_num):
    	e_result.append(e[k].x)



    return [y_n_result, e_result]
    

#-------- rounding procedure, single phase
def rounding_single_phase(n,K,y_n,y_label):
    #y_label is an integer array: -1 not assinged; 
    # randomly draw a label
    label = np.random.randint(K) #0 or 1
    # draw a random number from [0,1]
    val = np.random.uniform()

    for i in range(n):
        if y_label[2*i] == -1 and y_label[2*i+1] == -1: #if node i is not assigned yet
            if label == 0:
                if val < y_n[2*i]:
                    y_label[2*i] = 1
                    y_label[2*i+1] = 0
            else:
                if val < y_n[2*i+1]:
                    y_label[2*i] = 0
                    y_label[2*i+1] = 1
    return y_label


#------ rounding procedure
def rounding(y_n_result):

    n = int(len(y_n_result)/2)

    y_label = - np.ones(2*n,dtype = int) # -1 means not assigned yet
    K = 2
    phase_num = 0
    while np.isin(-1,y_label):
        phase_num += 1
        y_label = rounding_single_phase(n,K,y_n_result,y_label)
    return y_label


# generate adversarial graph by deleting edges
def genAdvGraph_del(model,budget,X, A, y):
    
    n = X.shape[0]
    d_n = X.shape[1]
    #y_label_true = y

    Y_n = np.zeros(2*n,dtype = float)
    for i in range(n):
        if y[i] > 0.0:
            Y_n[i*2] = 1.0
            Y_n[i*2 +1] = 0.0
        else:
            Y_n[i*2] = 0.0
            Y_n[i*2 +1] = 1.0

    node_i = []
    node_j = [] # store the node index (i,j) for each edge
    for i in range(n-1):
        for j in range(i+1,n):
            if A[i,j] > 0.5:
                node_i.append(i)
                node_j.append(j)
    edge_num = int(len(node_i))
    
    
    Y_e = np.zeros(2*edge_num,dtype = float) #[ y_ij_1, y_ij_2 ]
    for k in range(edge_num):
        node_index_i = node_i[k]
        node_index_j = node_j[k]
        Y_e[2*k] = Y_n[2*node_index_i]*Y_n[2*node_index_j]
        Y_e[2*k+1] = Y_n[2*node_index_i+1]*Y_n[2*node_index_j+1]

    #number of deleted edges
    budget_num = int(round(budget*edge_num))
   
    #========= model ==========
    
    w_n_1 = model[0:d_n]
    w_n_2 = model[d_n:2*d_n]
    w_e_1 = model[2*d_n]
    w_e_2 = model[2*d_n +1]

    #---- parameter
    C = np.zeros(2*n, dtype = float)
    for i in range(n):
        C[2*i] = np.inner(w_n_1,X[i,:]) - Y_n[2*i]
        C[2*i + 1] = np.inner(w_n_2,X[i,:]) - Y_n[2*i + 1]

    a_ij = np.zeros(2*edge_num,dtype = float)
    for i in range(edge_num):
        a_ij[2*i] = w_e_1*Y_e[2*i]
        a_ij[2*i+1] = w_e_2*Y_e[2*i+1]
    


    [y_n_result, e_result] = attack_del_LP(model,budget_num,node_i,node_j,n,d_n, C, a_ij)
    #---------- select best from L rounds
    L = 100
    obj_best = -9999.0
    e_result_best = np.ones(edge_num, dtype = float)
    for times in range(L):
        y_label = rounding(y_n_result)

        #calculate the weight for each edge
        w_edge = []
        for k in range(edge_num):
            i = node_i[k]
            j = node_j[k]
            w_edge.append(w_e_1*y_label[i*2]*y_label[j*2] + w_e_2*y_label[2*i+1]*y_label[j*2+1] - (a_ij[2*k] + a_ij[2*k+1]))

        #select the least w_edge to delete
        sorted_edge_index = np.argsort(w_edge)
        top_budget_num_index = sorted_edge_index[0:budget_num]
        selected_edge_index = []
        for index in top_budget_num_index:
            if w_edge[index] < 0.0:
                selected_edge_index.append(index)

        e_result = np.ones(edge_num, dtype = float)
        for index in selected_edge_index:
            e_result[index] = 0.0

        #-- compute the objective value

        obj = 0.0
        obj += np.inner(w_edge,e_result)

        for i in range(n):
            obj += C[i*2]*y_label[i*2] + C[i*2+1]*y_label[i*2+1]

        
        if obj > obj_best:
            obj_best = obj
            print obj_best 
            e_result_best = e_result

    A_m = A.copy()

    for k in range(edge_num):
        if e_result_best[k] < 0.5:
            i = node_i[k]
            j = node_j[k]
            A_m[i,j] = 0.0
            A_m[j,i] = 0.0

    return A_m





#===========================  Attack by deleting and adding edges ===================

def attack_add_del_LP(model, budget_num, del_budget_num,add_budget_num,node_i,node_j, non_node_i,non_node_j, n,d_n, C, a_ij):
    
    edge_num = int(len(node_i))
    non_edge_num = int(len(non_node_i))    
    #========= model ==========
    w_n_1 = model[0:d_n]
    w_n_2 = model[d_n:2*d_n]
    w_e_1 = model[2*d_n]
    w_e_2 = model[2*d_n +1]

    #================ Linear Programming =================

    m = Model()
    m.setParam('OutputFlag', 0)

    e_non = [] # one variable for each non edge
    z = [] #non_edge variable, z_ij = y_i *y_j * e_ij
    for k in range(non_edge_num):
        e_non.append(m.addVar(lb = 0.0, ub = 1.0,vtype = GRB.CONTINUOUS))
        z.append(m.addVars(2,lb = 0.0, vtype = GRB.CONTINUOUS))

    e = [] # one variable for each edge
    s = [] #edge variable s_ij = y_i * y_j * e_ij, one variable for each edge 
    for k in range(edge_num):
        e.append(m.addVar(lb = 0.0, ub = 1.0,vtype = GRB.CONTINUOUS))
        s.append(m.addVars(2,lb = 0.0,vtype = GRB.CONTINUOUS))

    y_n = []
    for i in range(n):
        y_n.append(m.addVars(2,lb = 0.0,vtype = GRB.CONTINUOUS))


    #---- obj

    obj = LinExpr()

    for i in range(n):
        obj += C[i*2]*y_n[i][0] + C[i*2+1]*y_n[i][1]

    for k in range(edge_num):
        obj += w_e_1*s[k][0] + w_e_2*s[k][1] - (a_ij[k*2] + a_ij[k*2+1])*e[k]


    for k in range(non_edge_num):
        obj += w_e_1*z[k][0] + w_e_2*z[k][1]  


    #obj += Const

    m.setObjective(obj,GRB.MAXIMIZE)


    for i in range(n):
        m.addConstr(y_n[i][0] + y_n[i][1] - 1.0 == 0.0)

    
    #for each edge
    for k in range(edge_num):
        node_index_i = node_i[k]
        node_index_j = node_j[k]
        m.addConstr(s[k][0] <= y_n[node_index_i][0])
        m.addConstr(s[k][0] <= y_n[node_index_j][0])
        m.addConstr(s[k][0] <= e[k])
        

        m.addConstr(s[k][1] <= y_n[node_index_i][1])
        m.addConstr(s[k][1] <= y_n[node_index_j][1])
        m.addConstr(s[k][1] <= e[k])
        
    # for each non_edge
    
    for k in range(non_edge_num):
        non_node_index_i = non_node_i[k]
        non_node_index_j = non_node_j[k]
        m.addConstr(z[k][0] <= y_n[non_node_index_i][0])
        m.addConstr(z[k][0] <= y_n[non_node_index_j][0])
        m.addConstr(z[k][0] <= e_non[k])

        m.addConstr(z[k][1] <= y_n[node_index_i][1])
        m.addConstr(z[k][1] <= y_n[node_index_j][1])
        m.addConstr(z[k][1] <= e_non[k])

    
    constr = LinExpr(0.0)
    constr.addTerms(np.ones(non_edge_num, dtype = float),e_non)
    m.addConstr(constr <= add_budget_num)

    constr = LinExpr(0.0)
    constr.addTerms(np.ones(edge_num),e)
    m.addConstr(constr >= edge_num - del_budget_num)

    m.optimize()


    y_n_result = []
    for i in range(n):
        y_n_result.append(y_n[i][0].x)
        y_n_result.append(y_n[i][1].x)

    return y_n_result


#------ generate adversarial graph (delete and add edges)
def genAdvGraph_add_del(model,budget,del_ratio,X, A, y):
    n = X.shape[0]
    d_n = X.shape[1]

    Y_n = np.zeros(2*n,dtype = float)
    for i in range(n):
        if y[i] > 0.0:
            Y_n[i*2] = 1.0
            Y_n[i*2 +1] = 0.0
        else:
            Y_n[i*2] = 0.0
            Y_n[i*2 +1] = 1.0

    node_i = []
    node_j = [] # store the node index (i,j) for each edge
    non_node_i = []
    non_node_j = []
    for i in range(n-1):
        for j in range(i+1,n):
            if A[i,j] > 0.5:
                node_i.append(i)
                node_j.append(j)
            else:
                if y[i]*y[j] < 0.0: 
                    non_node_i.append(i)
                    non_node_j.append(j)

    edge_num = len(node_i)
    budget_num = round(budget*edge_num)
    non_edge_num = len(non_node_i)

    del_budget_num = int(budget_num*del_ratio)
    add_budget_num = int(budget_num - del_budget_num)
    
    
    Y_e = np.zeros(2*edge_num,dtype = float) #[ y_ij_1, y_ij_2 ]
    for k in range(edge_num):
        node_index_i = node_i[k]
        node_index_j = node_j[k]
        Y_e[2*k] = Y_n[2*node_index_i]*Y_n[2*node_index_j]
        Y_e[2*k+1] = Y_n[2*node_index_i+1]*Y_n[2*node_index_j+1]

    
    
    #========= model ==========
    w_n_1 = model[0:d_n]
    w_n_2 = model[d_n:2*d_n]
    w_e_1 = model[2*d_n]
    w_e_2 = model[2*d_n +1]

    #---- parameter
    C = np.zeros(2*n, dtype = float)
    for i in range(n):
        C[2*i] = np.inner(w_n_1,X[i,:]) - Y_n[2*i]
        C[2*i + 1] = np.inner(w_n_2,X[i,:]) - Y_n[2*i + 1]

    a_ij = np.zeros(2*edge_num,dtype = float)
    for i in range(len(a_ij)/2):
        a_ij[2*i] = w_e_1*Y_e[2*i]
        a_ij[2*i+1] = w_e_2*Y_e[2*i+1]

    y_n_result = attack_add_del_LP(model, budget_num, del_budget_num,add_budget_num,node_i,node_j, non_node_i,non_node_j, n,d_n, C, a_ij)
    p#rint "--- budget --- del_num --- add_num ---> ", del_budget_num, add_budget_num

    #---------- select best from L rounds
    L = 100
    obj_best = -9999.99
    e_result_best = np.zeros(edge_num,dtype = float)
    non_e_result_best = np.zeros(non_edge_num,dtype = float)

    for times in range(L):
        y_label = rounding(y_n_result)


        #------calculate the weight for each edge
        edge_weight = []
        for k in range(edge_num):
            i = node_i[k]
            j = node_j[k]
            edge_weight.append(w_e_1*y_label[i*2]*y_label[j*2] + w_e_2*y_label[2*i+1]*y_label[j*2+1] - a_ij[2*k] - a_ij[2*k+1])

        # lexsort: break ties randomly
        random_order = np.random.random(len(edge_weight))

        sorted_edge_index = np.lexsort((random_order,edge_weight))
        top_del_budget_num_index = sorted_edge_index[0: del_budget_num]

        e_result = np.ones(edge_num, dtype = float)
        for index in top_del_budget_num_index:
            e_result[index] = 0.0
        #------calculate the weight for each non_edge
        non_edge_weight = []
        for k in range(non_edge_num):
            i = non_node_i[k]
            j = non_node_j[k]
            non_edge_weight.append(w_e_1*y_label[i*2]*y_label[j*2] + w_e_2*y_label[2*i+1]*y_label[j*2+1])


        # lexsort: break ties randomly
        random_order = np.random.random(len(non_edge_weight))

        sorted_non_edge_index = np.lexsort((random_order,non_edge_weight))
        sorted_non_edge_index = np.flip(sorted_non_edge_index,0)
        top_add_budget_num_index = sorted_non_edge_index[0: add_budget_num]

        non_e_result = np.zeros(non_edge_num, dtype = np.float)
        for index in top_add_budget_num_index:
            non_e_result[index] = 1.0

        #-- compute the objective value

        obj = 0.0

        obj += np.inner(edge_weight,e_result)
        obj += np.inner(non_edge_weight,non_e_result)

        for i in range(n):
            obj += C[i*2]*y_label[i*2] + C[i*2+1]*y_label[i*2+1]

        if obj > obj_best:
            obj_best = obj
            print obj_best 
            e_result_best = e_result
            del_edge_index = top_del_budget_num_index
            non_e_result_best = non_e_result
            add_edge_index = top_add_budget_num_index
    #
    #print "--- real --- del_num --- add_num ---> ", (edge_num - sum(e_result_best)), sum(non_e_result_best), float((edge_num - sum(e_result_best))+ sum(non_e_result_best))/ edge_num
    A_m = A.copy()

    for idx in del_edge_index:
        i = node_i[idx]
        j = node_j[idx]
        A_m[i,j] = 0.0
        A_m[j,i] = 0.0

    for idx in add_edge_index:
        i = non_node_i[idx]
        j = non_node_j[idx]
        A_m[i,j] = 1.0
        A_m[j,i] = 1.0

    return A_m



def genAdvGraph_rand(seed,budget,del_ratio,A, y):
    #randomly delete edges connecting to same nodes
    #randomly add edges connecting to different nodes
    n = A.shape[0]
    node_i = []
    node_j = [] # store the node index (i,j) for each edge
    non_node_i = []
    non_node_j = []
    for i in range(n-1):
        for j in range(i+1,n):
            if A[i,j] > 0.5:
                node_i.append(i)
                node_j.append(j)
            else:
                if y[i]*y[j] < 0.0: 
                    non_node_i.append(i)
                    non_node_j.append(j)

    edge_num = len(node_i)
    budget_num = round(budget*edge_num)
    non_edge_num = len(non_node_i)

    del_budget_num = int(budget_num*del_ratio)
    add_budget_num = int(budget_num - del_budget_num)

    np.random.seed(seed)
    selected_edge_index = np.random.choice(range(edge_num), del_budget_num,replace = False)
    selected_non_edge_index = np.random.choice(range(non_edge_num), add_budget_num,replace = False)

    A_m = A.copy()

    for idx in selected_edge_index:
        node_index_i = node_i[idx]
        node_index_j = node_j[idx]
        A_m[node_index_i,node_index_j] = 0.0
        A_m[node_index_j,node_index_i] = 0.0

    for idx in selected_non_edge_index:
        node_index_i = non_node_i[idx]
        node_index_j = non_node_j[idx]
        A_m[node_index_i,node_index_j] = 1.0
        A_m[node_index_j,node_index_i] = 1.0

    return A_m


def genAdvGraph_rand_del(seed,budget,A, y):
    #randomly delete edges connecting to same nodes
    n = A.shape[0]
    node_i = []
    node_j = [] # store the node index (i,j) for each edge

    for i in range(n-1):
        for j in range(i+1,n):
            if A[i,j] > 0.5:
                node_i.append(i)
                node_j.append(j)


    edge_num = int(len(node_i))
    budget_num = int(round(budget*edge_num))



    np.random.seed(seed)
    selected_edge_index = np.random.choice(range(edge_num), budget_num,replace = False)


    A_m = A.copy()

    for idx in selected_edge_index:
        node_index_i = node_i[idx]
        node_index_j = node_j[idx]
        A_m[node_index_i,node_index_j] = 0.0
        A_m[node_index_j,node_index_i] = 0.0

    return A_m




