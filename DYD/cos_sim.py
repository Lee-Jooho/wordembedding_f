import numpy as np
import math


def size(arr):
    total_num = 0
    
    for num in arr:
        total_num += (num**2)
        
    return math.sqrt(total_num)

def cos_sim(word_vec_total, idx):
    
    arr2 = word_vec_total[idx]
    cosine_sim = []
    
    for i in range(len(word_vec_total)):
        cos = np.dot(word_vec_total[i],arr2) / (size(word_vec_total[i]) * size(arr2))
        
        cosine_sim.append(cos)
        # 여기서 arr1, arr2 는 1* n 차 배열이여야 된다.
    
    # 학습하고 나온 가중치 W1들의 값을 가지고 비교할 예정.
    
    
    return cosine_sim

def dis_sim(word_vec_total, idx):
    
    arr3 = word_vec_total[idx]
    distance_sim = []
    
    for i in range(len(word_vec_total)):
        sum = 0
        for j in range(len(word_vec_total[i])):
            sum += (arr3[j]-word_vec_total[i][j])**2
        
        distance = math.sqrt(sum)
        
        distance_sim.append(distance)
        
        
    return distance_sim
        
            