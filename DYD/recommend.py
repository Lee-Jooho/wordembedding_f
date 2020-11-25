from cos_sim import *
from twolayernet import *


def get_recommendations_cos(word_vec_total,wi,idx):
    
    word = wi.inform['voc'][idx]
    
    sim_scores = list(enumerate(cos_sim(word_vec_total,idx)))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
   
    sim_scores = sim_scores[1:11] # 본인 단어 빼주기 위함.
    
    word_index = [i[0] for i in sim_scores]
    
    word_sim = [wi.inform['voc'][x] for x in word_index]
    
    
    return word, word_sim

def get_recommendations_dis(word_vec_total,wi,idx):
    
    word = wi.inform['voc'][idx]
    
    sim_scores2 = list(enumerate(dis_sim(word_vec_total,idx)))

    sim_scores2 = sorted(sim_scores2, key=lambda x: x[1])

    sim_scores2 = sim_scores2[1:11] # 본인 단어 빼주기 위함.

    word_index2 = [i[0] for i in sim_scores2]
    
    word_sim2 = [wi.inform['voc'][x] for x in word_index2]
    
    return word, word_sim2


def get_recommendations_predict(net,wi,idx,num): # idx는 학습시킬 단어의 인덱스, num은 보여줄 예상값의 갯수
    predict = net.predict(wi.inform['voc_vectors'][idx])
    
    predict_sorted = sorted(list(enumerate(predict)),key = lambda x: x[1] ,reverse = True) # 인덱스랑 값이랑 같이 찍음
    
    predict_sorted2 = sorted(list(predict),reverse = True) # 값만 불러옴 >> softmax 하기 위함.
    
    # 둘다 sorted를 하기 때문에 predict_sorted에 있는 값의 index와 밑에거의 index는 같음.
    
    predict_softmax = softmax(np.array(predict_sorted2))
    predict_softmax_num = predict_softmax[1:num+1]
    
    
    for i in range(num):
        print(wi.inform['voc'][predict_sorted[i][0]],np.round(predict_softmax_num[i]*100,2),"%","\n")