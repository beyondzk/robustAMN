import numpy as np
import scipy as sy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def extract_data_vs_all(positive_class):
    X = np.loadtxt("../data/cora.content", dtype = str)
    paper_IDs = X[:,0]
    paper_cates = X[:,1434]
    feature_num = 1433

    Cate_name = ['Case_Based', 'Genetic_Algorithms','Neural_Networks','Probabilistic_Methods','Reinforcement_Learning','Rule_Learning','Theory']
 
    case_IDs = []
    gene_IDs = []
    neu_IDs = []
    prob_IDs = []
    rein_IDs = []
    rule_IDs = []
    theo_IDs = []

    for i in range(len(paper_IDs)):
        cate = paper_cates[i]
        if cate == Cate_name[0]:
            case_IDs.append(paper_IDs[i])
        elif cate == Cate_name[1]:
            gene_IDs.append(paper_IDs[i])
        elif cate == Cate_name[2]:
            neu_IDs.append(paper_IDs[i])
        elif cate == Cate_name[3]:
            prob_IDs.append(paper_IDs[i])
        elif cate == Cate_name[4]:
            rein_IDs.append(paper_IDs[i])
        elif cate == Cate_name[5]:
            rule_IDs.append(paper_IDs[i])
        elif cate == Cate_name[6]:
            theo_IDs.append(paper_IDs[i])

    All_IDs = [case_IDs, gene_IDs,neu_IDs,prob_IDs,rein_IDs,rule_IDs,theo_IDs]


    #--- create IDs

    positive_IDs = All_IDs[positive_class-1]
    rest_IDs = []
    
    for ind in range(1,8):
        if ind == positive_class:
            continue
        
        rest_IDs = rest_IDs + All_IDs[ind-1]

    seed = 14
    np.random.seed(seed)    
    selected = np.random.choice(range(len(rest_IDs)),len(positive_IDs),replace = False)

    negative_IDs = []
    for i in selected:
        negative_IDs.append(rest_IDs[i])
    total_IDs = positive_IDs + negative_IDs

    y_label = - np.ones(len(positive_IDs) + len(negative_IDs),dtype = float) # positive label = 1.0, negative lable = -1.0
    for i in range(len(positive_IDs)):
        y_label[i] = 1.0


    #--- create data

    X_data = np.zeros((len(positive_IDs)+len(negative_IDs), feature_num),dtype = float)

    for i in range(len(positive_IDs)):
        pos_id = positive_IDs[i]
        index = np.argwhere(paper_IDs == pos_id)[0][0]
        X_data[i,:] = X[index,1:1434]

    for i in range(len(negative_IDs)):
        neg_id = negative_IDs[i]
        index = np.argwhere(paper_IDs == neg_id)[0][0]
        X_data[i+len(positive_IDs),:] = X[index,1:1434]

    np.savetxt("../data/total_IDs_vs_all.txt",total_IDs,fmt = '%s')
    np.savetxt("../data/data_vs_all.txt", X_data,fmt = '%f')
    np.savetxt("../data/label_vs_all.txt",y_label, fmt = '%f')

def extract_graph(IDs):
    #extract the graph for all nodes in IDs

    IDs = list(IDs)
    cite_matrix = np.loadtxt("../data/cora.cites.txt", dtype = str)
    n = len(IDs)
    A = np.zeros((n,n),dtype = float)

    for i in range(len(cite_matrix)):
        paper_id_1 = cite_matrix[i,0]
        paper_id_2 = cite_matrix[i,1]
        
        if paper_id_1 in IDs and paper_id_2 in IDs:
            paper_index_1 = IDs.index(paper_id_1)
            paper_index_2 = IDs.index(paper_id_2)

            A[paper_index_1,paper_index_2] = 1.0
            A[paper_index_2,paper_index_1] = 1.0
    return A 

def connected_node_index(A):
    #return the index of connected nodes:
    index = []
    for i in range(len(A)):
        if sum(A[i,:]) > 0.5:
            index.append(i)
    
    return index




















