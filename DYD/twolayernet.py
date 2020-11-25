import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common_layers import *
from training_param_add import * # import numpy as np가 되어있음.
from gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:

    def __init__(self):
        self.check = None
        
    def twolayernet(self, wi, weight_init_std = 0.9):
        # 重みの初期化
        voc_length = wi.inform['voc_length']
        voc_length_diff = wi.inform['voc_length_diff']
        input_size = voc_length
        output_size = voc_length
        hidden_size = 7
        
        if self.check is None:
            self.params = {}
            self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
            self.params['b1'] = np.zeros(hidden_size)
            self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
            self.params['b2'] = np.zeros(output_size)

            # レイヤの生成
            self.layers = OrderedDict()
            self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
            self.layers['Relu1'] = Relu()
            self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
            
            self.check = 1
        else:
            print("checking new words ~~ing~~ ")
            print("추가된 단어 수 " , voc_length_diff) #단어가 몇개가 추가되엇는지 내가 확인하기 위함.
            if voc_length_diff == 0:
                print("network is ready to use")
            else:
                print("adding words in vocabulary ~~ing~~ ")
                add_param = new_word_in_voc(self.params['W1'], self.params['W2'],self.params['b2'],voc_length_diff)
                add_param.make_new_params()
                for key in('W1','W2','b2'):
                    self.params[key] = add_param.params[key]
            
                print("ready to use")
                
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
                       
        
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads