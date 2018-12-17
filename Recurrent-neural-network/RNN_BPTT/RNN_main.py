from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras import Sequential
from keras.models import load_model
from keras.layers import Embedding, LSTM, Dense
import numpy as np
from RNN_backpropagation import *
import matplotlib.pyplot as plt

def generate_seq(model, tokenizer, seed_text, numtext=4):
     result  = seed_text
     seed_text = " ".join(seed_text)
     in_text= seed_text     
     for _ in range(numtext):
          # Mã hóa input từ chuỗi kí tự thành chuỗi số
          encoded = tokenizer.texts_to_sequences([in_text])[0]
          encoded = np.array(encoded)

          # Dự đoán kết quả
          yhat = model.predict(encoded)

          # Kết quả dự đoán được cũng là dạng số, ta tìm kí tự tương ứng với số vừa tìm được
          out_word = ''
          for word, index in tokenizer.word_index.items():
               if index == yhat:
                    out_word = word
                    break

          # Kí tự vừa tìm được ta ghép vào input ban đầu và tiếp tục tìm kiếm
          in_text = in_text + ' ' + out_word
          result = result + out_word

          # Khi mạng dự đoán được kí tự 'o' --> Kết thúc
          if out_word == 'o':
            break
     return result


# source text
data = """ h e l l o """
loss = []
# Mã hóa kí tự
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]


# Xác định số lượng kí tự
vocab_size = len(tokenizer.word_index) +1
print('Vocabulary Size: %d' % vocab_size)


# Tạo ma trận X,y 
sequences = list()
for i in range(1, len(encoded)):
    sequence = encoded[i-1:i+1]
    sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))
sequences = np.array(sequences)
X, y = sequences[:,0],sequences[:,1]
X = np.array([2, 3, 1, 1])
y = y.astype(int)

#Khai báo mạng RNN sử dụng backpropagation through time
model = RNN_func(word_dim=4,hidden_dim=4,bptt_truncate = 0)

#Train RNN với phương pháp Stochastic Gradient Decent (SGD)
loss = train_with_sgd(model, X, y, nepoch=1000, learning_rate=0.005)

print('Input = h --->',generate_seq(model,tokenizer,'h'))
print('Input = he --->',generate_seq(model,tokenizer,'he'))
print('Input = hel --->',generate_seq(model,tokenizer,'hel'))
print('Input = hel --->',generate_seq(model,tokenizer,'hell'))
print(loss)
plt.plot(loss)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
