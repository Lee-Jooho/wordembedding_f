from t_data_from_x_data import *
import time
import numpy as np

def train_try(wi,iters_num,learning_rate,net):
    sentence_words = wi.inform['sentence_words']
    word_vector = wi.inform['word_vector']
    voc = wi.inform['voc']
    #print(sentence_words)
    #print(word_vector)
    train_loss_list = []
    start = time.time()
    for i in range(iters_num):
        for x in range(len(sentence_words)):
            for y in range(len(sentence_words[x])):
                # y = 문장 중에서 x번째 문장에 있는 단어들의 index
        
                #print('x = {}'.format(x)), print('y = {}'.format(y))
                #print(sentence_words[x][y])
                #word_index = list(word_vector[x][y]).index(1) 
                #print("word_vector[x][y]의 index :", word_index  )
                x_data = []
                tdata = getting_tdata(wi, x, y)
                x_data.append(word_vector[x][y])
                x_data_np = np.array(x_data)
            
                for word in tdata:
                    t_data_vector = np.zeros_like(voc, dtype= int) # 실행속도가 조금 더 빠르다고 합니다.
                    index = voc.index(word)
                    t_data_vector[index] = 1
                    t_data_np = np.array(t_data_vector)
                    grad = net.gradient(x_data_np,t_data_np)
            
                #for key in ('W1','W2'):
                for key in ('W1','b1','W2','b2'):
                #for key in ('W1','b1','W2'):
                    net.params[key] -= learning_rate * grad[key]
            
                    loss = net.loss(x_data_np,t_data_np)
                    if i % 100 == 0:
                        train_loss_list.append(loss)
    
    print("time : " ,time.time()-start)
    print(train_loss_list)
    return net