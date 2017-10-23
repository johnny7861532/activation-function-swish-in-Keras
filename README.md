# Activation-function-swish-in-Keras
### Google just release a paper to describe a new activation function: SWISH: A SELF-GATED ACTIVATION FUNCTION ###
According to the paper is new self-gate activation function is more powerful than relu, and can improves neural net's accuarcy by just simple replace relu with swish.

Paper is here to offer: https://arxiv.org/pdf/1710.05941.pdf

here is how to put it into Keras.

```python
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras import optimizers
from sklearn.utils import class_weight
from keras import backend as K

def swish(x):
    return K.sigmoid(x) * x

adam = optimizers.Adam(lr = 0.005)
classifier = Sequential()
classifier.add(Dense(units = 10,activation = swish,kernel_initializer = 'uniform',input_dim = 10))
classifier.add(Dense(units = 50,activation = swish, kernel_initializer = 'uniform'))
classifier.add(Dense(units = 25,activation = swish, kernel_initializer = 'uniform'))
classifier.add(Dense(units = 10,activation = swish, kernel_initializer = 'uniform'))
classifier.add(Dense(units = 5,activation = swish, kernel_initializer = 'uniform'))
classifier.add(Dense(units = 1,activation = 'sigmoid',kernel_initializer = 'uniform'))
classifier.compile(optimizer = adam,loss = 'binary_crossentropy'
                   ,metrics = ['accuracy'])

classifier.fit(x_train,y_train,batch_size = 100,epochs = 100,validation_data  =(x_test,y_test)
,shuffle = True)


```
## If swish really better than relu? ##

### swish confusion matrix ###
![image](https://github.com/johnny7861532/activation-function-swish-in-Keras/blob/master/fraud%20cm.png)

### relu confusion matrix ###
![image](https://github.com/johnny7861532/activation-function-swish-in-Keras/blob/master/relu%20compare.png)


### swish training history ###
![image](https://github.com/johnny7861532/activation-function-swish-in-Keras/blob/master/swish100%20epochs%20history.png)

### relu training history ###
![image](https://github.com/johnny7861532/activation-function-swish-in-Keras/blob/master/relu%20history.png)



it seems that swish is not powerful as I expect, and swish training need about 20% extra time to training.

Also compare with their training history, relu seem got better training curve.

So far I can't feel the power of the swish activation function.
