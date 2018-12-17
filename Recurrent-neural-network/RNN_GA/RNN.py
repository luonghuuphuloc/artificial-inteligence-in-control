#Class RNN

import numpy as np

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex

class RNN_GA:

    # Khởi tạo các thông số của class như ma trận U,V,W, số lớp ẩn (hidden_nim), số ngõ vào(word_nim)
    def __init__(self, word_dim, hidden_dim = 4,num_in_pop = 10):

        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.num_in_pop = num_in_pop
        self.U_parent = []
        self.V_parent = []
        self.W_parent = []
       
        # Randomly initialize the network parameters
        for i in range(num_in_pop):
            self.U_parent.append(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim)))
            self.V_parent.append(np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim)))
            self.W_parent.append(np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim)))
       

    def forward_propagation(self, x,i):
        U = self.U_parent[i]
        V = self.V_parent[i]
        W = self.W_parent[i]

        # Số lớp ẩn
        T = len(x)
        
        # Các ma trận lớp ẩn và ngõ ra được lưu lại để sử dụng sau
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        o = np.zeros((T, self.word_dim))
       

        for t in np.arange(T): 
            s[t] = np.tanh(U[:,x[t]] + W.dot(s[t-1]))
            o[t] = softmax(V.dot(s[t]))
        return [o, s]
    

    # Tính toán hàm fitness cho quần thể
    def fitness(self,x,y):
        fitness = []
        for i in range(self.num_in_pop):
            o,s = self.forward_propagation(x,i)
            fitness.append(- np.mean(np.sum(y * np.log(o), axis=1)))
        return fitness 
    

    # Training mạng RNN bằng giải thuật di truyền
    def train(self,x,y,num_of_generations):
        self.graph = []
        for i in range(num_of_generations):
            fitness = self.fitness(x,y)
            i = np.argsort(fitness)
            self.graph.append(fitness[i[0]])
            u,v,w = self.select_mating_pool(fitness)
            u_cross,v_cross,w_cross = self.crossover(u,v,w) 
            u_cross,v_cross,w_cross = self.mutate(u_cross,v_cross,w_cross)

            for i in range(self.num_in_pop): 
                if i < 5:
                    self.U_parent[i] = u[i] 
                    self.V_parent[i] = v[i] 
                    self.W_parent[i] = w[i] 

                else:
                    self.U_parent[i] = u_cross[i-5]                   
                    self.V_parent[i] = v_cross[i-5]                    
                    self.W_parent[i] = w_cross[i-5]
        fitness = self.fitness(x,y)
        index = np.argsort(fitness)
        self.best_index = index[0]
        o, s = self.forward_propagation(x,self.best_index)
        self.yhat = np.argmax(o, axis=1)+1

    def predict(self, x):

        # Tính lan truyền thuận và với giá trị U, V, W sau khi training để dự đoán kết quả 
        o, s = self.forward_propagation(x,self.best_index)
        y_pre = np.argmax(o, axis=1)+1
        print('y_pre:',y_pre)      
        return y_pre[len(y_pre)-1]



#   Genetic Algorithm function
    
    # Chương trình con chọn lọc các cá thể từ quần thể thông qua hàm fitness
    def select_mating_pool(self,fitness):
        index = np.argsort(fitness)
        u = []
        v = []
        w = []
        for i in range(int(self.num_in_pop/2)):
            u.append(self.U_parent[index[i]])
            v.append(self.V_parent[index[i]])
            w.append(self.W_parent[index[i]])
        return [u,v,w]

    # Trao đổi chéo các cá thể vừa được chọn
    def crossover(self,u,v,w):
        u_cross = []
        v_cross = []
        w_cross = []

        for i in range(np.shape(u)[0]):
            u_cross.append(np.vstack((u[0][0:2,:],u[i][2:4,:])))
            v_cross.append(np.vstack((v[0][0:2,:],v[i][2:4,:])))
            w_cross.append(np.vstack((w[0][0:2,:],w[i][2:4,:])))
        return [u_cross,v_cross,w_cross]

    # Đột biến random các cá thể
    def mutate(self,u_cross,v_cross,w_cross):
        idx = np.random.randint(4,size = (3,2))
        k = np.random.randint(5)
        for i in range(3):
            u_cross[k][idx[i][0]][idx[i][1]] = u_cross[k][idx[i][0]][idx[i][1]] + np.random.uniform(-1.0,1.0,1)
            v_cross[k][idx[i][0]][idx[i][1]] = v_cross[k][idx[i][0]][idx[i][1]] + np.random.uniform(-1.0,1.0,1)
            w_cross[k][idx[i][0]][idx[i][1]] = w_cross[k][idx[i][0]][idx[i][1]] + np.random.uniform(-1.0,1.0,1)
        return [u_cross,v_cross,w_cross]