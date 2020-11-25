from t_data_from_x_data import *
from WordInput import *

def Get_T_Dic(wi,Dic_of_words):
    voc = wi.inform['voc']
    sentence_words = wi.inform['sentence_words']
    word_vector = wi.inform['word_vector']
#print(sentence_words)
#print(word_vector)
    
    for x in range(len(sentence_words)):
        for y in range(len(sentence_words[x])):
            # y = 문장 중에서 x번째 문장에 있는 단어들의 index
            #print(len(word_vector[x]))
        
            #print('x = {}'.format(x)), print('y = {}'.format(y))
            #print(word_vector[x][y])
            #print(sentence_words[x][y])
            word_index = int(np.where(word_vector[x][y] ==1 )[0][0])
            #print("word_vector[x][y]의 index :", word_index  )
            
            t_data = getting_tdata(sentence_words, x, y)
            
            #print("t_data : ",t_data)
          
    
            for word in t_data:
                    index_of_tdata_word = voc.index(word)
                                   
                    try:
                        if index_of_tdata_word not in Dic_of_words[word_index]:
                            Dic_of_words[word_index].append(index_of_tdata_word)
                    except KeyError:
                        Dic_of_words[word_index] = []
                        Dic_of_words[word_index].append(index_of_tdata_word)
            
    return Dic_of_words
            
        #여기서 학습을 진행해야됨