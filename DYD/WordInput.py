import numpy as np
import string
import re
import nltk
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
class word_inform():
    
    def __init__(self):
        self.inform = {}
        
    
    def wordinput(self):
        
        WI = input('문장을 입력해주세요 : ') # 문장 받아오기. WI = word input.
        WI = WI.replace('\n',' ') # 문단에 줄 내림이 있다면, 스페이스바로 바꿔주기
        
        
        #be = {'am', 'is', 'are', 'be' , 'was', 'were'} #  be 동사 저장.
    
        WI = WI.lower()
                
        #WI = WI.replace("i'm",'i am') # be동사를 찾아내기 위해, 변환을 해준다.
        #WI = WI.replace("he's",'he is') # be동사를 찾아내기 위해, 변환을 해준다.
        #WI = WI.replace("she's",'she is') # be동사를 찾아내기 위해, 변환을 해준다.
        #WI = WI.replace("that's",'that is') # be동사를 찾아내기 위해, 변환을 해준다
        #WI = WI.replace("what's",'what is') # be동사를 찾아내기 위해, 변환을 해준다.
        #WI = WI.replace("it's",'it is') # be동사를 찾아내기 위해, 변환을 해준다.  (is 줄임말 풀어주기.)
        
        #WI = WI.replace("you're",'you are') # be동사를 찾아내기 위해, 변환을 해준다.
        #WI = WI.replace("they're",'they are') # be동사를 찾아내기 위해, 변환을 해준다.
        #WI = WI.replace("we're",'we are') # be동사를 찾아내기 위해, 변환을 해준다.
    
        #Auxiliary_verb = {'will','would','can','could','shall','should','may','might','must'}
        
        #WI = WI.replace("i'll",'i will') # 조동사를 찾아내기 위해, 변환을 해준다.
        #WI = WI.replace("you'll",'you will') # 조동사를 찾아내기 위해, 변환을 해준다.
        #WI = WI.replace("they'll",'they will') # 조동사를 찾아내기 위해, 변환을 해준다.
        #WI = WI.replace("we'll",'we will') # 조동사를 찾아내기 위해, 변환을 해준다.
        #WI = WI.replace("he'll",'he will') # 조동사를 찾아내기 위해, 변환을 해준다.
        #WI = WI.replace("she'll",'she will') # 조동사를 찾아내기 위해, 변환을 해준다.
        #WI = WI.replace("it'll",'it will') # 조동사를 찾아내기 위해, 변환을 해준다.
        #WI = WI.replace("that'll",'that will') # 조동사를 찾아내기 위해, 변환을 해준다.
    
        #WI = WI.replace("i'd",'i would') # 조동사를 찾아내기 위해, 변환을 해준다.
        #WI = WI.replace("you'd",'you would') # 조동사를 찾아내기 위해, 변환을 해준다.
        #WI = WI.replace("they'd",'they would') # 조동사를 찾아내기 위해, 변환을 해준다.
        #WI = WI.replace("we'd",'we would') # 조동사를 찾아내기 위해, 변환을 해준다.
        #WI = WI.replace("he'd",'he would') # 조동사를 찾아내기 위해, 변환을 해준다.
        #WI = WI.replace("she'd",'she would') # 조동사를 찾아내기 위해, 변환을 해준다.
        
        #WI = re.sub("[.]{2,}",'',WI) # 마침표 두개이상 없애기
        WI = re.sub('[\w]+@[\w.]+',' ',WI)
        WI = re.sub("[^a-zA-Z.]+",' ',WI) # 마침표, 알파벳 제외 제거하기
        WI = ' '.join(WI.split())
        sentence = WI.split('.') # 문단에 마침표가 있다면, 문장을 분리해주기. punctuation 함수는 위에 re.sub 에서 다 제거해주기 때문에 사용안함.
        
        sentence_words = [s.split() for s in sentence] # 각각의 문장속에 있는 단어 분리 해주기.
        self.inform['sentence_words'] = sentence_words
        
        
    def word_voc(self,voc):
        before_voc_length = len(voc)
        sentence_words = self.inform['sentence_words'] # 입력받은 문장 그대로.
        
        for length in range(len(sentence_words)):
            for vocab in sentence_words[length]:
                #if vocab.isdigit() == False:  # 숫자가 계속 학습하는 문장에 들어가서 학습 효율이 떨어지는 듯 하다. ( 따라서 숫자는 제외한다.)
                if vocab not in stop_words:
                    if vocab not in voc:
                        voc.append(vocab)
                        
    
        self.inform['voc'] = voc
        after_voc_length = len(voc)
        
        self.inform['voc_length_diff'] = (after_voc_length - before_voc_length)
        self.inform['voc_length'] = after_voc_length
        
        word_vector = [[] for i in sentence_words]
        word_sentence = [[] for i in sentence_words]
        voc_vectors = []
        
        for word in voc:
            voc_vector = np.zeros_like(voc, dtype = int)# 단어장 크기의 새로운 벡터를 만든다.
            index_of_input_word = voc.index(word)
            voc_vector[index_of_input_word] += 1 # 한단어가 단어장의 몇번 index에 있는지를 확인.
            
            voc_vectors.append(voc_vector)
        
        self.inform['voc_vectors'] = voc_vectors
    
        # word_vector >> 입력받은 문장들을 단어별로 구분해 놓은 리스트.
    
    
        for length in range(len(sentence_words)):
            for word in sentence_words[length]:
                if word.isdigit() == False: # 숫자가 계속 학습하는 문장에 들어가서 학습 효율이 떨어지는 듯 하다. ( 따라서 숫자는 제외한다.)
                    if word not in stop_words:
                        voc_vector = np.zeros_like(voc, dtype = int)# 단어장 크기의 새로운 벡터를 만든다.
                        index_of_input_word = voc.index(word)
                        voc_vector[index_of_input_word] += 1 # 한단어가 단어장의 몇번 index에 있는지를 확인.
            
                        word_vector[length].append(voc_vector)
                        word_sentence[length].append(word)
        
        self.inform['sentence_words'] = word_sentence
        self.inform['word_vector'] = word_vector
        
        
        
        