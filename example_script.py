import robustAMN
import genData_Reuters
import attacker

import numpy as np 
import matplotlib.pyplot as plt


# =============== Parameters =============


def att_del(cate_name,SVM_c,AMN_c,AMN_RD_C,AMN_RD_B,selected_feature_num,total_feature_num):
    total_file_ID_file = '../data/%s_total_fileID.txt'% cate_name
    total_label_file = '../data/%s_total_label.txt'% cate_name


    edge_num_per_node = 3
    RUN = range(1,21)
    Budget = [0.0,0.05,0.1,0.15,0.2,0.25,0.3]
    SVM_ACC = 0.0   
    AMN_ACC = np.zeros(len(Budget),dtype = float)   
    AMN_RD_ACC = np.zeros(len(Budget),dtype = float)
    

    # ============ Generate Training and Test Data ==========
    for run in RUN:
        #print "--------------> ", run
        amn_c = AMN_c[run-1]
        AMN_RD_c = AMN_RD_C[run-1]
        AMN_RD_budget = AMN_RD_B[run-1]        
        [X_train,X_test,train_file_ID,test_file_ID,train_label,test_label] = genData_Reuters.split_train_test_single_cate(run, selected_feature_num,total_feature_num,total_file_ID_file,total_label_file)

        [train_graph_matrix,test_graph_matrix] = genData_Reuters.graph_extract_train_test(edge_num_per_node,train_file_ID,test_file_ID)

        #============= Train models ======================

        #--- AMN
        AMN_model = robustAMN.AMN_train(amn_c,X_train,train_graph_matrix,train_label)
        #--- robust AMN
        AMN_RD_model = robustAMN.robustAMN_to_delete_edges(AMN_RD_c, AMN_RD_budget, X_train,train_graph_matrix,train_label)

        #---------  attack by deleting edges
        for i in range(len(Budget)):
            budget = Budget[i]
            
            print "---- run --- budget ------>", run, budget

            #---- attack AMN
            AMN_adv_test_graph_matrix = attacker.genAdvGraph_del(AMN_model,budget,X_test,test_graph_matrix,test_label)
            

            acc = robustAMN.AMN_predict(AMN_model,X_test,AMN_adv_test_graph_matrix,test_label)
            
            
            AMN_ACC[i] += acc
            
            #---- attack robust AMN
            AMN_RD_adv_test_graph_matrix = attacker.genAdvGraph_del(AMN_RD_model,budget,X_test,test_graph_matrix,test_label)

            acc = robustAMN.AMN_predict(AMN_RD_model,X_test,AMN_RD_adv_test_graph_matrix,test_label)
            
            AMN_RD_ACC[i] += acc
            
   
    AMN_ACC = AMN_ACC/len(RUN)
    

    AMN_RD_ACC = AMN_RD_ACC/len(RUN)
    

    SVM_ACC = SVM_ACC/len(RUN)
    
    # --- save figure
    fig, ax = plt.subplots()
    ax.plot(Budget,AMN_ACC,'-rs', label = 'AMN')
    ax.plot(Budget,AMN_RD_ACC,'-bs', label = 'Robust-AMN')    

    
    plt.xlabel("Percentage of edges")
    plt.ylabel("Accuracy")
    ax.legend()

    plt.savefig("./%s_att_del_%s_outOf_%s_AMN_RD.pdf" %(cate_name, str(selected_feature_num),str(total_feature_num)))

    result = []
    result = result + list(AMN_ACC) + list(AMN_RD_ACC)

    np.savetxt("./%s_att_del_%s_outOf_%s_AMN_RD.txt" %( cate_name, str(selected_feature_num),str(total_feature_num)), result,fmt = '%f')




cate_name = 'trade'

selected_feature_num = 200
total_feature_num = 200
SVM_c = 0.05
edge_num_per_node = 3
att_budget = 0.1
AMN_c = np.loadtxt('../result/reuters_%s_outOf_%s_AMN_c.txt'% (str(selected_feature_num), str(total_feature_num)))
robust_C = np.loadtxt('../result/reuters_%s_outOf_%s_AMN_RD_c.txt'% (str(selected_feature_num), str(total_feature_num)))
robust_B = np.loadtxt('../result/reuters_%s_outOf_%s_AMN_RD_b.txt'% (str(selected_feature_num), str(total_feature_num)))


# result saved in figure: 
att_del(cate_name,SVM_c, AMN_c,robust_C,robust_B,selected_feature_num,total_feature_num)
