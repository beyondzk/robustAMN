import numpy as np
import scipy as sy

from nltk.corpus import reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split


from sklearn.metrics import f1_score, precision_score, recall_score

from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
 

#extract the file IDs for four labels [trade, crude,grain,money-fx]
#each file is labled only one of this four lables
#nltk.download('reuters')

def extract_file_ID(cate_name):
    trade_docs_ID = reuters.fileids('trade')
    crude_docs_ID = reuters.fileids('crude')
    grain_docs_ID = reuters.fileids('grain')
    money_docs_ID = reuters.fileids('money-fx')

    labels = ['trade','crude','grain','money-fx']
    #print trade_docs_ID
    seed = 14
    all_docs_ID = trade_docs_ID + crude_docs_ID + grain_docs_ID + money_docs_ID

    single_trade_file_ID = []
    single_crude_file_ID = []
    single_grain_file_ID = []
    single_money_file_ID = []

    for fileID in all_docs_ID:
        categories = reuters.categories(fileID)
        intersect = list(set(categories) & set(labels))
        if len(intersect) == 1:
            category = intersect[0]
            if category == labels[0]:
                single_trade_file_ID.append(fileID)
            elif category == labels[1]:
                single_crude_file_ID.append(fileID)
            elif category == labels[2]:
                single_grain_file_ID.append(fileID)
            else:
                single_money_file_ID.append(fileID)

    np.savetxt('../data/%s_fileID.txt'% labels[0],single_trade_file_ID,fmt = '%s')
    np.savetxt('../data/%s_fileID.txt'% labels[1],single_crude_file_ID,fmt = '%s')
    np.savetxt('../data/%s_fileID.txt'% labels[2],single_grain_file_ID,fmt = '%s')
    np.savetxt('../data/%s_fileID.txt'% labels[3],single_money_file_ID,fmt = '%s')

    if cate_name == 'trade':
        #rest_file_ID = single_crude_file_ID + single_grain_file_ID + single_money_file_ID
        rest_file_ID = np.concatenate((single_crude_file_ID,single_grain_file_ID), axis = None)
        
        rest_file_ID = np.concatenate((rest_file_ID,single_money_file_ID), axis = None)
        
        # randomly select negative data points from other 3 categories
        np.random.seed(seed)
        selected_negative_file_ID = np.random.choice(rest_file_ID,len(single_trade_file_ID), replace = False)

        total_file_ID = np.concatenate((single_trade_file_ID,selected_negative_file_ID), axis = None)

        total_label = np.ones(len(total_file_ID), dtype = np.float)
        
        for i in range(len(single_trade_file_ID),len(total_label)):
            total_label[i] = -1.0




    if cate_name == 'crude':
        #rest_file_ID = single_crude_file_ID + single_grain_file_ID + single_money_file_ID
        rest_file_ID = np.concatenate((single_trade_file_ID,single_grain_file_ID),axis = None)
        
        rest_file_ID = np.concatenate((rest_file_ID,single_money_file_ID),axis = None)
        
        # randomly select negative data points from other 3 categories
        np.random.seed(seed)
        selected_negative_file_ID = np.random.choice(rest_file_ID,len(single_crude_file_ID), replace = False)

        total_file_ID = np.concatenate((single_crude_file_ID,selected_negative_file_ID),axis = None)

        total_label = np.ones(len(total_file_ID), dtype = np.float)
        
        for i in range(len(single_crude_file_ID),len(total_label)):
            total_label[i] = -1.0


    if cate_name == 'grain':
        #rest_file_ID = single_crude_file_ID + single_grain_file_ID + single_money_file_ID
        rest_file_ID = np.concatenate((single_trade_file_ID,single_crude_file_ID),axis = None)
        
        rest_file_ID = np.concatenate((rest_file_ID,single_money_file_ID),axis = None)
        
        # randomly select negative data points from other 3 categories
        np.random.seed(seed)
        selected_negative_file_ID = np.random.choice(rest_file_ID,len(single_grain_file_ID), replace = False)

        total_file_ID = np.concatenate((single_grain_file_ID,selected_negative_file_ID),axis = None)

        total_label = np.ones(len(total_file_ID), dtype = np.float)
        
        for i in range(len(single_grain_file_ID),len(total_label)):
            total_label[i] = -1.0

    if cate_name == 'money-fx':
        #rest_file_ID = single_crude_file_ID + single_grain_file_ID + single_money_file_ID
        rest_file_ID = np.concatenate((single_trade_file_ID,single_crude_file_ID),axis = None)
        
        rest_file_ID = np.concatenate((rest_file_ID,single_grain_file_ID),axis = None)
        
        # randomly select negative data points from other 3 categories
        np.random.seed(seed)
        selected_negative_file_ID = np.random.choice(rest_file_ID,len(single_money_file_ID), replace = False)

        total_file_ID = np.concatenate((single_money_file_ID,selected_negative_file_ID),axis = None)

        total_label = np.ones(len(total_file_ID), dtype = np.float)
        
        for i in range(len(single_money_file_ID),len(total_label)):
            total_label[i] = -1.0

    print labels[0], len(single_trade_file_ID)
    print labels[1], len(single_crude_file_ID)
    print labels[2], len(single_grain_file_ID)
    print labels[3], len(single_money_file_ID)

    np.savetxt('../data/%s_total_fileID.txt'% cate_name,total_file_ID,fmt = '%s')
    np.savetxt('../data/%s_total_label.txt'% cate_name,total_label, fmt = '%f')



def split_train_test_single_cate(seed, selected_feature_num,total_feature_num,total_file_ID_file,total_label_file): # a random process
   
    total_file_ID = np.loadtxt(total_file_ID_file,dtype = 'str')
    total_label = np.loadtxt(total_label_file)

    total_docs = []
    for fileID in total_file_ID:
            total_docs.append(reuters.raw(fileID))

    vectorizer = CountVectorizer(max_features = total_feature_num, stop_words = 'english',binary = True)
    X_data = vectorizer.fit_transform(total_docs)
    X_data = X_data.toarray()
    
    # select feature
    np.random.seed(14)
    selected_feature = np.random.choice(range(total_feature_num),selected_feature_num,replace = False)
    X_data = X_data[:,selected_feature]


    X_train, X_test, train_file_ID, test_file_ID, train_label, test_label = train_test_split(X_data,total_file_ID,total_label,test_size=0.5,random_state = seed)

    return [X_train,X_test,train_file_ID,test_file_ID,train_label,test_label]

def graph_extract_train_test(edge_num_per_node,train_file_ID,test_file_ID):
    # extract the graphs for train and test (or validation) together
    test_docs = []
    train_docs = []

    for fileID in test_file_ID:
        test_docs.append(reuters.raw(fileID))

    for fileID in train_file_ID:
        train_docs.append(reuters.raw(fileID))
    #=================== extract graph structure
    
    graph_feature_num = 5000
    

    #---- for training data
    train_num = len(train_docs)
    vectorizer_train = TfidfVectorizer(min_df=3,max_df=0.90, max_features=graph_feature_num,use_idf=True, stop_words = 'english', sublinear_tf=True,norm='l2')
    X_train = vectorizer_train.fit_transform(train_docs)
    X_train = X_train.toarray()
    

    #adj matrices for edges and non_edges
    train_graph_matrix = np.zeros((train_num,train_num),dtype = float)

    #--- compute consine distance
    train_dist_matrix = np.ones((train_num,train_num),dtype = float) # initilized to 1.0, in order to find the min ones

    for i in range(train_num-1):
        for j in range(i+1,train_num):
            train_dist_matrix[i,j] = sy.spatial.distance.cosine(X_train[i,:],X_train[j,:])
            train_dist_matrix[j,i] = train_dist_matrix[i,j]
    
    

    #--- extract edges
    for i in range(train_num):
        current_row = train_dist_matrix[i,:]
        close_node = []
        while len(close_node) < edge_num_per_node:
            node_index = np.argmin(current_row)
            close_node.append(node_index)
            current_row[node_index] = 1.0
        
        for node in close_node:
            train_graph_matrix[i,node] = 1.0
            train_graph_matrix[node,i] = 1.0

    #---- for validation (test) set
    

    test_num = len(test_docs)
    vectorizer_test = TfidfVectorizer(min_df=3,max_df=0.90, max_features= graph_feature_num,use_idf=True, stop_words = 'english', sublinear_tf=True,norm='l2')
    X_test = vectorizer_test.fit_transform(test_docs)
    X_test = X_test.toarray()
    

    #adj matrices for edges and non_edges
    test_graph_matrix = np.zeros((test_num,test_num),dtype = float)
    #--- compute consine distance
    test_dist_matrix = np.ones((test_num,test_num),dtype = float) # initilized to 1.0, in order to find the min ones

    for i in range(test_num-1):
        for j in range(i+1,test_num):
            test_dist_matrix[i,j] = sy.spatial.distance.cosine(X_test[i,:],X_test[j,:])
            test_dist_matrix[j,i] = test_dist_matrix[i,j]
    
    

    #--- extract edges
    for i in range(test_num):
        current_row = test_dist_matrix[i,:]
        close_node = []
        while len(close_node) < edge_num_per_node:
            node_index = np.argmin(current_row)
            close_node.append(node_index)
            current_row[node_index] = 1.0
        
        for node in close_node:
            test_graph_matrix[i,node] = 1.0
            test_graph_matrix[node,i] = 1.0
    return [train_graph_matrix,test_graph_matrix]



















