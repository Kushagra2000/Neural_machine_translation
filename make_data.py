import tensorflow as tf
print(tf.__version__)
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
from keras.utils import to_categorical




df=pd.read_csv('dataset.csv')

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

eng=Vocab_builder('eng',df['eng'])
ger=Vocab_builder('ger',df['ger'])

eng_w2i,eng_i2w=eng.build_vocab()

ger_w2i,ger_i2w=ger.build_vocab()

print("Vocabularies made")


#splitting the data into training,testing and validation sets

train_x,test_x,train_y,test_y=train_test_split(df['eng'],df['ger'],test_size=0.1,random_state=42)
train_x,val_x,train_y,val_y=train_test_split(train_x,train_y,test_size=0.23,random_state=42)

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


train_x=sent_to_np(train_x,'eng',False)
train_y=sent_to_np(train_y,'ger',False)
test_x=sent_to_np(test_x,'eng',False)
test_y=sent_to_np(test_y,'ger',False)
val_x=sent_to_np(val_x,'eng',False)
val_y=sent_to_np(val_y,'ger',False)

#One hot encoding class labels
train_y=to_categorical(train_y,num_classes=len(ger.vocab_list))
test_y=to_categorical(test_y,num_classes=len(ger.vocab_list))
val_y=to_categorical(val_y,num_classes=len(ger.vocab_list))


#Removing singleton axis (3rd axis)
#for embedding layer
t_x=np.squeeze(train_x,axis=2)
v_x=np.squeeze(val_x,axis=2)
te_x=np.squeeze(test_x,axis=2)

#Saving training testing and validation arrays
np.save("data/training/train_x.npy",train_x)
np.save("data/training/train_y.npy",train_y)
np.save("data/testing/test_x.npy",test_x)
np.save("data/testing/test_y.npy",test_y)
np.save("data/validation/val_x.npy",val_x)
np.save("data/validation/val_y.npy",val_y)
np.save("data/training/train_x_embedding.npy",t_x)
np.save("data/testing/test_x_embedding.npy",te_x)
np.save("data/validation/val_x_embedding.npy",v_x)

print("Data saved in data/")