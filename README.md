# Activation-function-swish-in-Keras
### Google just release a paper to describe a new activation function: SWISH: A SELF-GATED ACTIVATION FUNCTION ###
According to the paper is new self-gate activation function is more powerful than relu, and can improves neural net's accuarcy by just simple replace relu with swish.

Paper is here to offer: https://arxiv.org/pdf/1710.05941.pdf

here is how to put it into realist in Keras.

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
,shuffle = True,class_weight = weight)


```
