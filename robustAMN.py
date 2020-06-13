from gurobipy import *
import numpy as np
from sklearn.metrics import classification_report

from sklearn.svm import LinearSVC


def svm_single_cate(c, X_train,X_test,y_train,y_test):
    clf = LinearSVC(C = c)

    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)

    return float(sum(y_pred == y_test))/len(y_test)

def AMN_train(c, X, A, y):
    #X: training data
    #A: graph matrix
    #y: training label
    #c: parameter
    # w = (w_n_1, w_n_2, w_e_1,w_e_2)
    n = X.shape[0]
    d_n = X.shape[1]
    

    #=============== build qudratic programming model

    m = Model("qp")
    m.setParam('OutputFlag', 0)
    

    #-------- variables
    # node weight vector, dimension = d_n
    w_n_1 = []
    w_n_2 = []
    for i in range(d_n):
        w_n_1.append(m.addVar(vtype = GRB.CONTINUOUS))
        w_n_2.append(m.addVar(vtype = GRB.CONTINUOUS))
    
    # edge weight  
    w_e_1 = m.addVar(lb = 0.0, vtype = GRB.CONTINUOUS)
    w_e_2 = m.addVar(lb = 0.0, vtype = GRB.CONTINUOUS)

    # create variable z for each node
    Z_node = []
    for i in range(n):
        Z_node.append(m.addVar(vtype = GRB.CONTINUOUS))
    

    # create variable z for each edge
    Z_edge = []
    node_i = []
    node_j = [] # store the node index (i,j) for each edge
    edge_index_matrix = np.zeros((n,n),dtype = int) # store the edge index for a pair of nodes (i,j)
    current_edge_index = 0
    for i in range(n-1):
        for j in range(i+1,n):
            if A[i,j] > 0.5:
                node_i.append(i)
                node_j.append(j)
                edge_index_matrix[i,j] = current_edge_index
                edge_index_matrix[j,i] = current_edge_index
                current_edge_index += 1
                Z_edge.append(m.addVars(4,lb = 0.0, vtype = GRB.CONTINUOUS)) # formate z = (z_ij_1,z_ij_2,z_ji_1,z_ji_2)

    edge_num = int(len(Z_edge))


    slack = m.addVar(lb = 0.0, vtype = GRB.CONTINUOUS)

    # create label vector
    #node indicator y_n = (y_i_1,y_i_2)
    #y = np.loadtxt(train_label_file)
    y_n = np.zeros(2*n,dtype = float)
    for i in range(n):
        if y[i] > 0.0: #means first class (y_i_1,y_i_2) = (1,0)
            y_n[i*2] = 1.0
            y_n[i*2+1] = 0.0
        else:
            y_n[i*2] = 0.0
            y_n[i*2+1] = 1.0

    # edge vector (y_ij_1,y_ij_2); y_ij_1 = y_i_1 * y_j_1; length = 2*|E|
    y_e = np.zeros(2*edge_num,dtype = float)
    for k in range(edge_num):
        node_index_i = node_i[k]
        node_index_j = node_j[k]
        y_e[2*k] = y_n[2*node_index_i]*y_n[2*node_index_j]
        y_e[2*k+1] = y_n[2*node_index_i+1]*y_n[2*node_index_j+1]

    #----- construct objective
    obj = QuadExpr(0.0)
    obj += w_e_1*w_e_1 + w_e_2*w_e_2

    # for i in range(d_n):
    #     obj += w_n_1[i]*w_n_1[i] + w_n_2[i]*w_n_2[i]
    obj.addTerms(np.ones(d_n,dtype = float),w_n_1,w_n_1)
    obj.addTerms(np.ones(d_n,dtype = float),w_n_2,w_n_2)
    obj = 0.5*obj 
    obj += c*slack

    
    #----- construct constraint

    # constr: wXy - N + slack - sum(z_i) >= 0

    constr = LinExpr(0.0)

    #for each node
    for i in range(n):      
        constr.addTerms(X[i,:]*y_n[2*i],w_n_1)
        constr.addTerms(X[i,:]*y_n[2*i+1],w_n_2)

        constr -= Z_node[i]
        

    #for each edge
    for k in range(edge_num):
        constr += w_e_1*y_e[2*k] + w_e_2*y_e[2*k+1]

    constr = constr - n + slack 


    m.addConstr(constr >= 0.0)


    # const_2 : add constraint for each node and each class c
    for i in range(n):
        constr_class_1 = LinExpr(0.0)
        constr_class_2 = LinExpr(0.0)

        constr_class_1 += Z_node[i]
        constr_class_2 += Z_node[i]
        #------ add edge variables
        #find the nodes that connects to node i
        adj_nodes = np.argwhere(A[i,:] > 0.5)
        for k in range(len(adj_nodes)):
            j = adj_nodes[k][0]
            edge_index = edge_index_matrix[i,j]
            if i < j:
                constr_class_1 -= Z_edge[edge_index][0]
                constr_class_2 -= Z_edge[edge_index][1]
            else:
                constr_class_1 -= Z_edge[edge_index][2]
                constr_class_2 -= Z_edge[edge_index][3]


        #--- add the rest variables
        constr_class_1.addTerms(-X[i,:],w_n_1)
        constr_class_1 += y_n[2*i]

        constr_class_2.addTerms(-X[i,:],w_n_2)
        constr_class_2 += y_n[2*i+1]

        m.addConstr(constr_class_1 >= 0.0)
        m.addConstr(constr_class_2 >= 0.0)

    #const 3: add const for each edge and class
    for k in range(edge_num):
        constr_class_1 = LinExpr(0.0)
        constr_class_2 = LinExpr(0.0)

        constr_class_1 = Z_edge[k][0] + Z_edge[k][2] - w_e_1
        constr_class_2 = Z_edge[k][1] + Z_edge[k][3] - w_e_2

        m.addConstr(constr_class_1 >= 0.0)
        m.addConstr(constr_class_2 >= 0.0)


    m.setObjective(obj,GRB.MINIMIZE)
    m.optimize()


    

    w_n_1_result = []
    w_n_2_result = []
    for i in range(d_n):
        w_n_1_result.append(w_n_1[i].x)
        w_n_2_result.append(w_n_2[i].x)
    w_e_1_result = w_e_1.x
    w_e_2_result = w_e_2.x
    
    all_parameters = w_n_1_result + w_n_2_result
    all_parameters.append(w_e_1_result)
    all_parameters.append(w_e_2_result)

    return all_parameters


def AMN_predict(model,X, A,y):
    n = X.shape[0]
    d_n = X.shape[1]
    w_n_1 = model[0:d_n]
    w_n_2 = model[d_n:2*d_n]
    w_e_1 = model[2*d_n]
    w_e_2 = model[2*d_n +1]

    #------construct optimization problem, linear programming
    m = Model("lp")
    m.setParam('OutputFlag', 0)
    #create variable y_i_1 and y_i_2 for each node
    Y_node = []
    for i in range(n):
        Y_node.append(m.addVar(lb = 0.0, vtype = GRB.CONTINUOUS)) #y_i_1
        Y_node.append(m.addVar(lb = 0.0, vtype = GRB.CONTINUOUS)) #y_i_2


    # create variable y for each edge, edge (i,j) has two variables y_ij_1 and y_ij_2
    Y_edge = []
    node_i = []
    node_j = [] # store the node index (i,j) for each edge
    for i in range(n-1):
        for j in range(i+1,n):
            if A[i,j] > 0.5:
                node_i.append(i)
                node_j.append(j)
                Y_edge.append(m.addVar(lb = 0.0, vtype = GRB.CONTINUOUS)) #y_ij_1
                Y_edge.append(m.addVar(lb = 0.0, vtype = GRB.CONTINUOUS)) #y_ij_2

    edge_num = int(len(node_i))

    #construct objective
    obj = LinExpr(0.0)

    # for each node
    for i in range(n):      
        obj += np.inner(X[i,:],w_n_1)*Y_node[i*2]
        obj += np.inner(X[i,:],w_n_2)*Y_node[i*2+1]
        

    #for each edge
    for k in range(edge_num):
        obj += w_e_1*Y_edge[k*2]
        obj += w_e_2*Y_edge[k*2+1]

    

    #construct constraints

    # const 1: y_i_1 + y_i_2 = 1 for each node
    for i in range(n):
        m.addConstr(Y_node[i*2] + Y_node[i*2+1] == 1.0)

    #const 2: y_ij_1 <= y_i_1, y_ij_1 <= y_j_1

    for k in range(edge_num):
        node_index_i = node_i[k]
        node_index_j = node_j[k]

        m.addConstr(Y_edge[2*k] <= Y_node[2*node_index_i])
        m.addConstr(Y_edge[2*k] <= Y_node[2*node_index_j])

        m.addConstr(Y_edge[2*k+1] <= Y_node[2*node_index_i+1])
        m.addConstr(Y_edge[2*k+1] <= Y_node[2*node_index_j+1])





    m.setObjective(obj,GRB.MAXIMIZE)
    m.optimize()


    # Y_node_pred = []

    # for i in range(n):
    #     Y_node_pred.append(Y_node[2*i].x)
    #     Y_node_pred.append(Y_node[2*i+1].x)
    y_pred = []  # 1.0 or -1.0

    for i in range(n):
        if Y_node[2*i].x > 0.5:
            y_pred.append(1.0)
        else:
            y_pred.append(-1.0)

    return float(sum(y_pred == y))/len(y)

    


def robustAMN_to_delete_edges(c, budget,X,A,y):
    #e_ij a binary variable for each edge (i,j); if e_ij = 0, indicate that the edge should be deleted
    n = X.shape[0]
    d_n = X.shape[1]
    
    Y_n = np.zeros(2*n,dtype = float) #node class indicators (..., y_i_1,y_i_2,...)
    for i in range(n):
        if y[i] > 0.0: #means first class (y_i_1,y_i_2) = (1,0)
            Y_n[i*2] = 1.0
            Y_n[i*2+1] = 0.0
        else:
            Y_n[i*2] = 0.0
            Y_n[i*2+1] = 1.0


    node_i = []
    node_j = [] # store the node index (i,j) for each edge
    edge_index_matrix = np.zeros((n,n),dtype = int) # store the edge index for a pair of nodes (i,j)
    current_edge_index = 0
    for i in range(n-1):
        for j in range(i+1,n):
            if A[i,j] > 0.5:
                node_i.append(i)
                node_j.append(j)
                edge_index_matrix[i,j] = current_edge_index
                edge_index_matrix[j,i] = current_edge_index
                current_edge_index += 1
    edge_num = int(len(node_i))
    

    Y_e = np.zeros(2*edge_num,dtype = float) #[ y_ij_1, y_ij_2 ]
    for k in range(edge_num):
        node_index_i = node_i[k]
        node_index_j = node_j[k]
        Y_e[2*k] = Y_n[2*node_index_i]*Y_n[2*node_index_j]
        Y_e[2*k+1] = Y_n[2*node_index_i+1]*Y_n[2*node_index_j+1]

    #number of deleted edges
    D = round(budget*edge_num)




    #=================== Quadratic programming ===================
    m = Model()
    m.setParam('OutputFlag', 0)
    
    #---- variables
    w_n_1 = [] # for class 1, it has dimension d_n
    w_n_2 = [] # for class 2
    for i in range(d_n):
        w_n_1.append(m.addVar(vtype = GRB.CONTINUOUS))
        w_n_2.append(m.addVar(vtype = GRB.CONTINUOUS))
    
    
    w_e_1 = m.addVar(lb = 0.0, vtype = GRB.CONTINUOUS) # edge weight for class 1
    w_e_2 = m.addVar(lb = 0.0, vtype = GRB.CONTINUOUS) # edge weight for class 2


    #slack variable
    slack = m.addVar(lb = 0.0, vtype = GRB.CONTINUOUS)

    # node variable t = (t_1 ,..., t_N)
    t = []
    for i in range(n):
        t.append(m.addVar(vtype = GRB.CONTINUOUS)) # it has no constraint >= 0

    T = []
    SE = []
    p = []
    for k in range(edge_num):
        T.append(m.addVars(4,lb = 0.0, vtype = GRB.CONTINUOUS))# formate T[k] = (t_ij_1,t_ij_2,t_ji_1,t_ji_2)
        SE.append(m.addVars(2,lb = 0.0, vtype = GRB.CONTINUOUS)) #SE[k] = [SE_ij_1,SE_ij_2]
        p.append(m.addVar(lb = 0.0, vtype = GRB.CONTINUOUS))

    t_D = m.addVar(lb = 0.0,vtype = GRB.CONTINUOUS)
    


    #-------- construct objective
    obj = QuadExpr(0.0)
    obj += w_e_1*w_e_1 + w_e_2*w_e_2

    obj.addTerms(np.ones(d_n, dtype = float),w_n_1,w_n_1)
    obj.addTerms(np.ones(d_n, dtype = float),w_n_2,w_n_2)
    obj = 0.5*obj
    obj += c*slack


    #------- constraints
    constr = LinExpr(0.0)
    constr += slack - n + (edge_num - D)*t_D
    for i in range(n):
        constr -= t[i]
        constr.addTerms(X[i,:]*Y_n[2*i],w_n_1)
        constr.addTerms(X[i,:]*Y_n[2*i+1],w_n_2)

    for k in range(edge_num):
        constr -= p[k]

    m.addConstr(constr >= 0.0)


    for i in range(n):
        constr_1 = LinExpr(0.0)
        constr_2 = LinExpr(0.0)

        constr_1.addTerms(-X[i,:],w_n_1)
        constr_2.addTerms(-X[i,:],w_n_2)

        constr_1 += Y_n[i*2] + t[i]
        constr_2 += Y_n[i*2+1] + t[i]

        #------ add edge variables
        #find the nodes that connects to node i
        adj_nodes = np.argwhere(A[i,:] > 0.5)
        for k in range(len(adj_nodes)):
            j = adj_nodes[k][0]
            edge_index = edge_index_matrix[i,j]
            
            if i < j:
                constr_1 -= T[edge_index][0]
                constr_2 -= T[edge_index][1]
            else:
                constr_1 -= T[edge_index][2]
                constr_2 -= T[edge_index][3]

        m.addConstr(constr_1 >= 0.0)
        m.addConstr(constr_2 >= 0.0)


    for k in range(edge_num):
        m.addConstr(T[k][0]+ T[k][2] + SE[k][0] >= w_e_1)
        m.addConstr(T[k][1]+ T[k][3] + SE[k][1] >= w_e_2)
        m.addConstr(w_e_1*Y_e[k*2] + w_e_2*Y_e[k*2+1] + p[k] - SE[k][0] - SE[k][1] - t_D >= 0.0)

    m.setObjective(obj,GRB.MINIMIZE)
    m.optimize()

    w_n_1_result = []
    w_n_2_result = []

    for i in range(d_n):
        w_n_1_result.append(w_n_1[i].x)
        w_n_2_result.append(w_n_2[i].x)

    w_e_1_result = w_e_1.x
    w_e_2_result = w_e_2.x
    
    
    all_parameters = w_n_1_result + w_n_2_result
    all_parameters.append(w_e_1_result)
    all_parameters.append(w_e_2_result)
    return all_parameters


def robustAMN_to_del_add_edges( c,budget, X, A,y):


    #------- data ----
    n = X.shape[0]
    d_n = X.shape[1]


    Y_n = np.zeros(2*n,dtype = float)

    for i in range(n):
        if y[i] > 0.0:
            Y_n[i*2] = 1.0
            Y_n[i*2 +1] = 0.0
        else:
            Y_n[i*2] = 0.0
            Y_n[i*2+1] = 1.0

    #edge_index_matrix: store the indecis of edges and non_edges of the node (i,j)
    edge_index_matrix = np.zeros((n,n),dtype = int)
    non_edge_index_matrix = np.zeros((n,n),dtype = int)

    # store the nodes for each edge and non_edge
    node_i = []
    node_j = []
    non_node_i = []
    non_node_j = []

    current_edge_index = 0
    current_non_edge_index = 0

    for i in range(n-1):
        for j in range(i+1,n):
            if A[i,j] > 0.5:
                node_i.append(i)
                node_j.append(j)
                edge_index_matrix[i,j] = current_edge_index
                edge_index_matrix[j,i] = current_edge_index
                current_edge_index += 1
            else:
                if y[i]*y[j] < 0.0:  # if nodes have different labels
                    non_node_i.append(i)
                    non_node_j.append(j)
                    non_edge_index_matrix[i,j] = current_non_edge_index
                    non_edge_index_matrix[j,i] = current_non_edge_index
                    current_non_edge_index += 1


    edge_num = current_edge_index
    non_edge_num = current_non_edge_index

    del_budget_num = edge_num*budget*0.5
    add_budget_num = edge_num*budget*0.5

    Y_e = np.zeros(2*edge_num,dtype = float) #[ y_ij_1, y_ij_2 ]
    for k in range(edge_num):
        node_index_i = node_i[k]
        node_index_j = node_j[k]
        Y_e[2*k] = Y_n[2*node_index_i]*Y_n[2*node_index_j]
        Y_e[2*k+1] = Y_n[2*node_index_i+1]*Y_n[2*node_index_j+1]




    #=================== Quadratic programming ===================
    m = Model()
    m.setParam('OutputFlag', 0)
    
    #---- variables
    w_n_1 = [] # for class 1, it has dimension d_n
    w_n_2 = [] # for class 2
    for i in range(d_n):
        w_n_1.append(m.addVar(lb = 0.0, vtype = GRB.CONTINUOUS))
        w_n_2.append(m.addVar(lb = 0.0, vtype = GRB.CONTINUOUS))
    
    
    w_e_1 = m.addVar(lb = 0.0, vtype = GRB.CONTINUOUS) # edge weight for class 1
    w_e_2 = m.addVar(lb = 0.0, vtype = GRB.CONTINUOUS) # edge weight for class 2

    slack = m.addVar(lb = 0.0, vtype = GRB.CONTINUOUS)

    #node variable
    t = []
    for i in range(n):
        t.append(m.addVar(vtype = GRB.CONTINUOUS))

    
    #for each edge
    T = [] #
    SE = []
    H = []
    for k in range(edge_num):
        T.append(m.addVars(4, lb = 0.0, vtype = GRB.CONTINUOUS)) # [t_ij_1, t_ij_2,t_ji_1,t_ji_2]
        SE.append(m.addVars(2,lb = 0.0, vtype = GRB.CONTINUOUS)) #[SE_ij_1,SE_ij_2]
        H.append(m.addVar(lb = 0.0,vtype = GRB.CONTINUOUS)) #h_ij


    #for each non edge

    Q = []
    ZE = []
    G = []

    for k in range(non_edge_num):
        Q.append(m.addVars(4,lb = 0.0,vtype = GRB.CONTINUOUS)) #[q_ij_1,q_ij_2,q_ji_1,q_ji_2]
        ZE.append(m.addVars(2,lb = 0.0,vtype = GRB.CONTINUOUS)) #[ZE_ij_1,ZE_ij_2]
        G.append(m.addVar(lb = 0.0,vtype = GRB.CONTINUOUS)) #g_ij

    

    TD_1 = m.addVar(lb = 0.0, vtype = GRB.CONTINUOUS)
    TD_2 = m.addVar(lb = 0.0, vtype = GRB.CONTINUOUS)



    #-------- construct objective
    obj = QuadExpr(0.0)
    obj += w_e_1*w_e_1 + w_e_2*w_e_2

    obj.addTerms(np.ones(d_n,dtype = float),w_n_1,w_n_1)
    obj.addTerms(np.ones(d_n,dtype = float),w_n_2,w_n_2)
    obj = 0.5*obj 
    obj += c*slack

    m.setObjective(obj,GRB.MINIMIZE)


    #--- constraints
    const = LinExpr(0.0)
    const += slack - n - add_budget_num*TD_2 + (edge_num - del_budget_num)*TD_1

    for i in range(n):
        const -= t[i]
        const.addTerms(X[i,:]*Y_n[i*2],w_n_1)
        const.addTerms(X[i,:]*Y_n[i*2+1],w_n_2)

    for k in range(edge_num):
        const -= H[k]

    for k in range(non_edge_num):
        const -= G[k]
    m.addConstr(const >= 0.0)


    for i in range(n):
        
        const_1 = LinExpr(0.0)
        const_2 = LinExpr(0.0)

        const_1.addTerms(-X[i,:],w_n_1)
        const_2.addTerms(-X[i,:],w_n_2)

        const_1 += Y_n[i*2] + t[i]
        const_2 += Y_n[i*2+1] + t[i]


        #---- add edge and non_edge variables

        #find the nodes that connects to node i
        adj_nodes = np.argwhere(A[i,:] > 0.5)

        for k in range(len(adj_nodes)):
            j = adj_nodes[k][0]
            edge_index = edge_index_matrix[i,j]
            if i<j:       
                const_1 -= T[edge_index][0]
                const_2 -= T[edge_index][1]
            else:
                const_1 -= T[edge_index][2]
                const_2 -= T[edge_index][3]

        #find the nodes that not connected to i and has opposite label

        non_adj_nodes = np.argwhere(non_edge_index_matrix[i,:] > 0.5)

        for k in range(len(non_adj_nodes)):
            j = non_adj_nodes[k][0]

            non_edge_index = non_edge_index_matrix[i,j]

            if i < j:
                const_1 -= Q[non_edge_index][0]
                const_2 -= Q[non_edge_index][1]
            else:
                const_1 -= Q[non_edge_index][2]
                const_2 -= Q[non_edge_index][3]

        m.addConstr(const_1 >= 0.0)
        m.addConstr(const_2 >= 0.0)

    for k in range(edge_num):
        m.addConstr(T[k][0] + T[k][2] - w_e_1 + SE[k][0] >= 0.0)
        m.addConstr(T[k][1] + T[k][3] - w_e_2 + SE[k][1] >= 0.0)

        m.addConstr(w_e_1*Y_e[k*2] + w_e_2*Y_e[k*2+1] - SE[k][0] - SE[k][1] + H[k] - TD_1 >= 0.0)

    for k in range(non_edge_num):
        m.addConstr(G[k] + TD_2 - ZE[k][0] - ZE[k][1] >= 0.0)



    m.optimize()

    w_n_1_result = []
    w_n_2_result = []

    for i in range(d_n):
        w_n_1_result.append(w_n_1[i].x)
        w_n_2_result.append(w_n_2[i].x)

    w_e_1_result = w_e_1.x
    w_e_2_result = w_e_2.x
    
    
    all_parameters = w_n_1_result + w_n_2_result
    all_parameters.append(w_e_1_result)
    all_parameters.append(w_e_2_result)

    return all_parameters