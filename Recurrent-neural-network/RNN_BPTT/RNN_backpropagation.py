
import numpy as np
import sys
import matplotlib.pyplot as plt
from keras.utils import to_categorical

class RNN_func:
	
	def __init__(self, word_dim, hidden_dim=4, bptt_truncate=0):
        # Khởi tạo thông số cơ bản (vocabulary, lớp ẩn)
		self.word_dim = word_dim
		self.hidden_dim = hidden_dim
		self.bptt_truncate = bptt_truncate
        # Khởi tạo thông số mạng ngẫu nhiên
		self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
		self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
		self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

	def forward_propagation(self, x):
    # Tông số time steps
		T = len(x)		
		# Lưu lại các giá trị lớp ra o và lớp ẩn s để sử dụng sau này
		s = np.zeros((T, self.hidden_dim))
		s[-1] = np.zeros(self.hidden_dim)
		
		o = np.zeros((T, self.word_dim))
		# Tính o và s theo công thức
		# s[t] = tanh(U.x[t] + W.s[t-1])
		# o[t] = softmax(V.s[t])
		for t in np.arange(T):
		    
			s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
			o[t] = softmax(self.V.dot(s[t]))
		return [o, s]
 
	
	def predict(self, x):
		# Tính lan truyền thuận và với giá trị U, V, W sau khi training để dự đoán kết quả 
		o, s = self.forward_propagation(x)
		y_pre = np.argmax(o, axis=1) + 1		      
		return y_pre[len(y_pre)-1] 


	def calculate_total_loss(self, x, y):
	#Tính sai số theo cross-entropy				
		o, s = self.forward_propagation(x)
		#Tạo one-hot vector
		y_temp = to_categorical(y, num_classes=5)
		y_temp = np.delete(y_temp,0,1)
		
		E = (- np.mean(np.sum(y_temp * np.log(o), axis=1)))
		
		return E 


	def bptt(self, x, y):
	#Tính backpropagation through time
	    T = len(y)
	    # Tính lan truyền thuận
	    o, s = self.forward_propagation(x)
	    # Đạo hàm sai số theo U, L, W
	    dEdU = np.zeros(self.U.shape)
	    dEdV = np.zeros(self.V.shape)
	    dEdW = np.zeros(self.W.shape)
	    delta_o = o
	    
	    delta_o[np.arange(len(y)), y-1] -= 1	    
	    
	    # Mỗi bước ngược
	    for t in np.arange(T)[::-1]:
	        dEdV += np.outer(delta_o[t], s[t].T)
	        # Khởi tạo delta_t
	        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
	        # Backpropagation through time theo chuỗi ngược liên tiếp 
	        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:	            
	            dEdW += np.outer(delta_t, s[bptt_step-1])              
	            dEdU[:,x[bptt_step]] += delta_t
	            # update cho bước kế tiếp
	            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
	    return [dEdU, dEdV, dEdW]
 	 
	

	def numpy_sgd_step(self, x, y, learning_rate):
		# tính gradient của sai số
		dEdU, dEdV, dEdW = self.bptt(x, y)
		# Cập nhật trọng số theo gradient
		self.U -= learning_rate * dEdU
		self.V -= learning_rate * dEdV
		self.W -= learning_rate * dEdW

	
# SGD Loop
# - model: RNN model 
# - X_train: training data set
# - y_train: training data labels
# - learning_rate: Khởi tạo learning rate cho SGD
# - nepoch: số lượng epoch
# - evaluate_loss_after: Đánh giá sai số sau mỗi k epoch
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=1):
    # Theo dõi sai số
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_total_loss(X_train, y_train)           
            losses.append(loss)
                      
            #Chỉnh lại learning rate nếu sai số tăng lên
            if (len(losses) > 1 and losses[-1]> losses[-2]):
                learning_rate = learning_rate * 0.75 
                print ("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        
            # One SGD step
        model.numpy_sgd_step(X_train, y_train, learning_rate)
        num_examples_seen += 1    
   
    return losses

def softmax(x):
#Hàm softmax
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)