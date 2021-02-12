
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Activation,InputLayer,Embedding
from tensorflow.keras.losses import sparse_categorical_crossentropy,categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import IPython
from keras.utils import to_categorical


#hyperparameters
m1_lr=0.001
m2_lr=0.01
m1_lstm_units=128
m2_lstm_units=128
m1_epochs=25
m2_epochs=15
m1_batch_size=128
m2_batch_size=128
m2_embedding_col=96

class Vocab_builder():
  '''
  Builds vocabulary and 
  word to index and index to word dictionaries
  from dataset
  '''
  def __init__(self,lang,series):
    self.lang=lang
    self.data=series
  def tokenize(self,line):
    return line.split(' ')
  def build_vocab(self):
    self.uniq_words=set()
    
    self.maxlen=0
    count=3
    self.num_list=[]
    for index,line in self.data.items():
      self.word_list=self.tokenize(line)
      self.maxlen=max(len(self.word_list),self.maxlen)
      for word in self.word_list:
        if(word not in self.uniq_words and word!='<EOS>' and word!='<SOS>'):
          self.uniq_words.add(word)
          self.num_list.append(count)
          count+=1
      
    self.vocab_list=['<PAD>','<SOS>','<EOS>']+sorted(list(self.uniq_words))
    self.num_list=[0,1,2]+self.num_list
    print("Built vocabulary having {} elements".format(len(self.vocab_list)))
    print("Largest sentence length (with tags):{}".format(self.maxlen))
    return dict(zip(self.vocab_list,self.num_list)),dict(zip(self.num_list,self.vocab_list))


df=pd.read_csv('drive/MyDrive/dataset.csv')

#Objects of Vocab_builder class
eng=Vocab_builder('eng',df['eng'])
ger=Vocab_builder('ger',df['ger'])

eng_w2i,eng_i2w=eng.build_vocab()
ger_w2i,ger_i2w=ger.build_vocab()

def sent_to_ind(sentence,lang):
  '''
  Tokenizes a string and
  converts it to an np array of 
  indices and pads the 
  array according to max sentence length
  '''
  ind_list=[]
  if lang=='eng':
    tokens=eng.tokenize(sentence)
    for token in tokens:
      ind_list.append(eng_w2i[token])
    while len(ind_list)<max(ger.maxlen,eng.maxlen):
      ind_list.append(0)
  else:
    tokens=ger.tokenize(sentence)
    for token in tokens:
      ind_list.append(ger_w2i[token])
    while len(ind_list)<max(ger.maxlen,eng.maxlen):
      ind_list.append(0)
    
  return np.array(ind_list)

def sent_to_np(series,lang,translate_mode):
  '''
  Converts a dataframe column to 
  a unsqueezed np array of indexes
  with padding for feeding into NN
  '''
  ret_list=[]
  if translate_mode==False :
    if lang=='eng':
      for index,val in series.items():
        ret_list.append(sent_to_ind(val,'eng'))
    else:
      for index,val in series.items():
        ret_list.append(sent_to_ind(val,'ger'))
    
    ret_list=np.array(ret_list)
    return np.expand_dims(ret_list,axis=2)
  else:
    ans=sent_to_ind(series,'eng')
    ans=np.expand_dims(ans,axis=0)
    ans=np.expand_dims(ans,axis=2)
    return ans


#loading training data
train_x=np.load("data/training/train_x.npy")
train_y=np.load("data/training/train_y.npy")
t_x=np.load("data/training/train_x_embedding.npy")

#loading testing data
test_x=np.load("data/testing/test_x.npy")
test_y=np.load("data/testing/test_y.npy")
te_x=np.load("data/testing/test_x_embedding.npy")

#loading validation data
val_x=np.load("data/validation/val_x.npy")
val_y=np.load("data/validation/val_y.npy")
v_x=np.load("data/validation/val__x_embedding.npy")

#Making base model using best hyperparameters
def base_LSTM_model(m1_lstm_units,m1_lr):
  '''
  Simple LSTM model
  '''
  lstm=LSTM(m1_lstm_units,return_sequences=True,activation='tanh')  #LSTM layer with output being hiddent state at time t
  layer_at_t=TimeDistributed(Dense(len(ger.vocab_list),activation='softmax')) #Dense layer acting on hidden output at each step to generate predictions
  model=tf.keras.Sequential()
  model.add(InputLayer(train_x.shape[1:]))
  model.add(lstm)
  model.add(layer_at_t)

  model.compile(loss=categorical_crossentropy,optimizer=Adam(m1_lr),metrics=['accuracy','MeanSquaredError',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
  return model

def embedding_LSTM(m2_lstm_units,m2_lr,embedding_col):
  '''
  LSTM model with embedding layer
  '''
  lstm=LSTM(m2_lstm_units,return_sequences=True,activation='tanh')
  print(t_x.shape[1])
  embed=Embedding(len(ger.vocab_list),embedding_col,input_length=t_x.shape[1])
  layer_at_t=TimeDistributed(Dense(len(ger.vocab_list),activation="softmax"))
  model=tf.keras.Sequential()
  model.add(embed)
  model.add(lstm)
  model.add(layer_at_t)

  model.compile(loss=categorical_crossentropy,optimizer=Adam(m2_lr),metrics=['accuracy','MeanSquaredError',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
  return model

base_model=base_LSTM_model(m1_lstm_units,m1_lr)
print(base_model.summary())

base_model.fit(train_x,train_y,m1_batch_size,m1_epochs,validation_data=(val_x,val_y))
l,acc,mse,p,r=base_model.evaluate(test_x,test_y)
print("Base model loss for testing set:{}".format(l))
print("Base model accuracy for testing set:{}".format(acc))
print("Base model precision for testing set:{}".format(p))
print("Base model recall for testing set:{}".format(r))
print("Base model f1_score for testing set:{}".format((2*p*r)/(p+r)))

base_model.save("model/base_model.h5")

embeded_model=embedding_LSTM(m2_lstm_units,m2_lr,m2_embedding_col)
print(embeded_model.summary())

embeded_model.fit(t_x,train_y,batch_size=m2_batch_size,epochs=m2_epochs,validation_data=(v_x,val_y))
l,acc,mse,p,r=embeded_model.evaluate(te_x,test_y)
print("Embedded model loss for testing set:{}".format(l))
print("Embedded model accuracy for testing set:{}".format(acc))
print("Embedded model MSE for testing set:{}".format(mse))
print("Embedded model precision for testing set:{}".format(p))
print("Embedded model recall for testing set:{}".format(r))
print("Embedded model f1_score for testing set:{}".format((2*p*r)/(p+r)))

embeded_model.save("model/embedded_model.h5")