#Chuong trinh train và test mạng RNN

import numpy as np
from RNN import RNN_GA
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import matplotlib.pyplot as plt


# Chương trình con ghép các kí tự sau khi predict thành từ "hello"
def generate_seq(model, tokenizer, seed_text):
    result  = seed_text
    seed_text = " ".join(seed_text)
    in_text= seed_text
    
    while(1):
        # Mã hóa input từ chuỗi kí tự thành chuỗi số
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = array(encoded)

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

# Mã hóa kí tự
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]


# Xác định số lượng kí tự
vocab_size = len(tokenizer.word_index) 
print('Vocabulary Size: %d' % vocab_size)


# Tạo ma trận X,y 
sequences = list()
for i in range(1, len(encoded)):
    sequence = encoded[i-1:i+1]
    sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))
sequences = array(sequences)
X, y = sequences[:,0],sequences[:,1]
print(X)
print(y)

# Đổi y thành dạng one-hot vector
y = to_categorical(y)
y = np.delete(y,0,1)


# Khởi tạo model RNN
model = RNN_GA(4)
# Training model
model.train(X,y,num_of_generations = 150)

# Dự đoán kết quả
print('Input = h --->',generate_seq(model,tokenizer,'h'))
print('Input = he --->',generate_seq(model,tokenizer,'he'))
print('Input = hel --->',generate_seq(model,tokenizer,'hel'))
plt.plot(model.graph)
plt.ylabel('fitness')
plt.show()


