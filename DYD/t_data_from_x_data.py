import numpy as np

def getting_tdata(wi, x ,y):
    sentence_words = wi.inform['sentence_words']
    voc = wi.inform['voc']
    t_data = []
    
    
    if len(sentence_words[x]) == 1:
        t_data.append(sentence_words[x][y])
    
    if len(sentence_words[x]) == 2:
        if y == 0 :
            t_data.append(sentence_words[x][y+1])
        if y == 1:
            t_data.append(sentence_words[x][y-1])

    if len(sentence_words[x]) == 3:
        if y == 0 :
            t_data.append(sentence_words[x][y+1])
            t_data.append(sentence_words[x][y+2])
        if y == 1:
            t_data.append(sentence_words[x][y-1])
            t_data.append(sentence_words[x][y+1])
        if y == 2:
            t_data.append(sentence_words[x][y-2])
            t_data.append(sentence_words[x][y-1])

    if len(sentence_words[x]) == 4:
        if y == 0 :
            t_data.append(sentence_words[x][y+1])
            t_data.append(sentence_words[x][y+2])
        if y == 1:
            t_data.append(sentence_words[x][y-1])
            t_data.append(sentence_words[x][y+1])
            t_data.append(sentence_words[x][y+2])
        if y == 2:
            t_data.append(sentence_words[x][y-2])
            t_data.append(sentence_words[x][y-1])
            t_data.append(sentence_words[x][y+1])
        if y == 3:
            t_data.append(sentence_words[x][y-2])
            t_data.append(sentence_words[x][y-1])

    if len(sentence_words[x]) >= 5:
        if y == 0 :
            t_data.append(sentence_words[x][y+1])
            t_data.append(sentence_words[x][y+2])
        if y == 1:
            t_data.append(sentence_words[x][y-1])
            t_data.append(sentence_words[x][y+1])
            t_data.append(sentence_words[x][y+2])
        if y == len(sentence_words[x])-2:
            t_data.append(sentence_words[x][y-2])
            t_data.append(sentence_words[x][y-1])
            t_data.append(sentence_words[x][y+1])
        if y == len(sentence_words[x])-1:
            t_data.append(sentence_words[x][y-2])
            t_data.append(sentence_words[x][y-1])
        if y< len(sentence_words[x])-2 and y > 1:
            t_data.append(sentence_words[x][y-2])
            t_data.append(sentence_words[x][y-1])
            t_data.append(sentence_words[x][y+1])
            t_data.append(sentence_words[x][y+2])

    
    
    return t_data