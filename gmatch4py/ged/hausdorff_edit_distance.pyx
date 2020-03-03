# coding = utf-8

import numpy as np
cimport numpy as np
from ..base cimport Base

cdef class HED(Base):
    """
    Implementation of Hausdorff Edit Distance described in

    Improved quadratic time approximation of graph edit distance by Hausdorff matching and greedy assignement
    Andreas Fischer, Kaspar Riesen, Horst Bunke
    2016
    """

    cdef int node_del
    cdef int node_ins
    cdef int edge_del
    cdef int edge_ins
    cdef float alpha
    cdef float beta
    cdef float edge_distance

    def __init__(self, int node_del=1, int node_ins=1, int edge_del=1, int edge_ins=1, 
                float alpha=0.5, float beta=0.5, float edge_distance=0.1):
        """Constructor for HED"""
        Base.__init__(self,1,False)
        self.node_del = node_del
        self.node_ins = node_ins
        self.edge_del = edge_del
        self.edge_ins = edge_ins
        self.alpha = alpha
        self.beta = beta
        self.edge_distance = edge_distance


    cpdef np.ndarray compare(self,list listgs, list selected):
        cdef int n = len(listgs)
        cdef np.ndarray comparison_matrix = np.zeros((n, n)).astype(float)
        cdef int i,j
        for i in range(n):
            for j in range(i, n):
                g1,g2=listgs[i],listgs[j]
                f=self.isAccepted(g1,i,selected)
                if f:
                    comparison_matrix[i, j] = self.hed(g1, g2)
                else:
                    comparison_matrix[i, j] = np.inf
                comparison_matrix[j, i] = comparison_matrix[i, j]

        return comparison_matrix
    
    cpdef np.ndarray compare_block(self, list listgs, int s_x, int e_x, int
                                                                s_y, int e_y):
        cdef int x_range = e_x - s_x
        cdef int y_range = e_y - s_y
        cdef np.ndarray comparison_matrix = np.zeros((x_range, y_range)).astype(float)
        cdef int i,j
        for i in range(s_x, e_x):
            for j in range(s_y, e_y):
                g1,g2=listgs[i],listgs[j]
                comparison_matrix[i-s_x, j-s_y] = self.hed(g1, g2) 
        return comparison_matrix
        
    cpdef np.ndarray compare_diagonal(self, list listgs, int s_x, int e_x, int
                                                                s_y, int e_y):
        cdef int x_range = e_x - s_x
        cdef int y_range = e_y - s_y
        cdef np.ndarray comparison_matrix = np.zeros((x_range, y_range)).astype(float)
        cdef int i,j
        for i in range(s_y, e_y):
            for j in range(i, e_x):
                g1,g2=listgs[i],listgs[j]
                comparison_matrix[j-s_y, i-s_y] = self.hed(g1, g2) 
        return comparison_matrix
                
    cpdef np.ndarray compare_test_train(self, list test, list train):
        cdef int ntest = len(test)
        cdef int ntrain = len(train)
        cdef np.ndarray comparison_matrix = np.zeros((ntest, ntrain)).astype(float)
        cdef int i,j
        for i in range(ntest):
            for j in range(ntrain):
                g1,g2=test[i],train[j]
                comparison_matrix[i, j] = self.hed(g1, g2) 
        return comparison_matrix

    cdef float hed(self, g1, g2):
        """
        Compute de Hausdorff Edit Distance
        :param g1: first graph
        :param g2: second graph
        :return:
        """
        # FLW add lower bound to avoid too strong underestimation
        estimate = self.sum_fuv(g1, g2) + self.sum_fuv(g2, g1)
        node_diff = np.abs(len(list(g1.nodes)) - len(list(g2.nodes)))
        edge_diff = np.abs(len(list(g1.edges)) - len(list(g2.edges)))
        lower_bound = node_diff + edge_diff 
        return np.max([estimate, lower_bound])

    cdef float sum_fuv(self, g1, g2):
        """
        Compute Nearest Neighbour Distance between G1 and G2
        :param g1: First Graph
        :param g2: Second Graph
        :return:
        """
        cdef np.ndarray min_sum = np.zeros(len(g1))
        nodes1 = list(g1.nodes)
        nodes2 = list(g2.nodes)
        nodes2.extend([None])
        cdef np.ndarray min_i
        for i in range(len(nodes1)):
            min_i = np.zeros(len(nodes2))
            for j in range(len(nodes2)):
                min_i[j] = self.fuv(g1, g2, nodes1[i], nodes2[j])
            min_sum[i] = np.min(min_i)
        return np.sum(min_sum)

    # FLW
    cdef float substitution_cost(self, g1, n1, n2):
        std = g1.graph['std']
        x = self.beta * std[0] * (n1[0] - n2[0])**2
        y = (1 - self.beta) * std[1] * (n1[1] - n2[1])**2
        return np.sqrt(x + y)

    cdef float fuv(self, g1, g2, n1, n2):
        """
        Compute the Node Distance function
        :param g1: first graph
        :param g2: second graph
        :param n1: node of the first graph
        :param n2: node of the second graph
        :return:
        """
        if n2 == None:  # Del
            return self.alpha * self.node_del + ((1-self.alpha) * (self.edge_del / 2.) * g1.degree(n1))
        if n1 == None:  # Insert
            return self.alpha * self.node_ins + ((1-self.alpha) * (self.edge_ins / 2.) * g2.degree(n2))
        else:
            if n1 == n2:
                return 0
            else:
                return ((self.alpha * self.substitution_cost(g1, n1, n2) +
                        (1-self.alpha) * self.hed_edge(g1, g2, n1, n2)) / 2)

    cdef float hed_edge(self, g1, g2, n1, n2):
        """
        Compute HEDistance between edges of n1 and n2, respectively in g1 and g2
        :param g1: first graph
        :param g2: second graph
        :param n1: node of the first graph
        :param n2: node of the second graph
        :return:
        """
        return self.sum_gpq(g1, n1, g2, n2) + self.sum_gpq(g2, n2, g1, n1)


    cdef float sum_gpq(self, g1, n1, g2, n2):
        """
        Compute Nearest Neighbour Distance between edges around n1 in G1  and edges around n2 in G2
        :param g1: first graph
        :param n1: node in the first graph
        :param g2: second graph
        :param n2: node in the second graph
        :return:
        """

        #if isinstance(g1, nx.MultiDiGraph):
        cdef list edges1 = list(g1.edges(n1)) if n1 else []
        cdef list edges2 =  list(g2.edges(n2)) if n2 else []

        cdef np.ndarray min_sum = np.zeros(len(edges1))
        edges2.extend([None])
        cdef np.ndarray min_i
        for i in range(len(edges1)):
            min_i = np.zeros(len(edges2))
            for j in range(len(edges2)):
                min_i[j] = self.gpq(edges1[i], edges2[j], n1, n2)
            min_sum[i] = np.min(min_i)
        return np.sum(min_sum)

    cdef float gpq(self, tuple e1, tuple e2, n1, n2):
        """
        Compute the edge distance function
        :param e1: edge1
        :param e2: edge2
        :return:
        """
        if e2 == None:  # Del
            return self.edge_del
        if e1 == None:  # Insert
            return self.edge_ins
        else:
            # FLW - edge substitutions need to happen, if we replace n2 with n1 and they are not connected with eachother via a third node n3
            alternate_n1 = [f for f in e1 if f != n1][0]
            alternate_n2 = [f for f in e2 if f != n2][0]
            distance = np.sqrt((alternate_n1[0] - alternate_n2[0])**2 + (alternate_n1[1] - alternate_n2[1])**2)
            # TODO: think about better distance
            if distance < self.edge_distance:
                return 0
            else:
                return (self.edge_del + self.edge_ins) / 2.
           # if e1 == e2:
           #     return 0
           # return (self.edge_del + self.edge_ins) / 2.
