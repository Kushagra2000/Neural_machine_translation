#This file cleans the data and generates the dataset used
#for making the vocabulary

import re
import unicodedata
import pandas as pd



def process_text(lines,translate_mode):
  ''' 
  Takes a list of tab seperated
  English and German sentences as input 
  and returns dataframe of processed
  English and German sentence pairs
  '''
  proc_lines=[]
  if translate_mode==False:
    for line in lines:
      line=line.strip() #to remove newlines at the end of a sentence
      line=line.split("\t") #splitting by tabs
      line=line[:-1]  #remove contributing information
      
      line[0]=''.join(c for c in unicodedata.normalize('NFD', line[0]) if unicodedata.category(c) != 'Mn')
      line[0]=line[0].encode('utf8','ignore').decode('utf8')
      line[0]=line[0].replace(u'\u200b',' ')
      line[0]=line[0].lower()
      line[0]=line[0].replace('\xa0', ' ')
      line[0]=re.sub(r"([.,?!;])",r" \1 ",line[0])   #adding spaces before and after punctuation
      line[0]=re.sub(r"[0-9]"," ",line[0])  #removing numbers
      line[0]=re.sub(r'["]'," ",line[0])  #removing quotes
      line[0]=re.sub(r"[']","",line[0])
      line[0]=re.sub(r"[%-,]"," ",line[0])
      line[0]=re.sub(r"[:]"," ",line[0])
      line[0]=re.sub(r'[" "]+'," ",line[0])  #removing excess spaces
      line[0]=line[0].strip() #removing spaces from the end of string
      line[0]="<SOS> "+line[0]+" <EOS>"

      line[1]=''.join(c for c in unicodedata.normalize('NFD', line[1]) if unicodedata.category(c) != 'Mn')      
      line[1]=line[1].replace(u'\u200b',' ')
      line[1]=line[1].replace('\xa0', ' ')
      line[1]=line[1].lower()
      line[1]=re.sub(r"([.,?!;])",r" \1 ",line[1])
      line[1]=re.sub(r"[0-9]",r" ",line[1])
      line[1]=re.sub(r'["]',"",line[1])
      line[1]=re.sub(r'[—]',"",line[1])
      line[1]=re.sub(r'[„]',"",line[1])
      line[1]=re.sub(r'[“]',"",line[1])
      line[1]=re.sub(r'[–]',"",line[1])
      line[1]=re.sub(r'[‘‚]',"",line[1])
      line[1]=re.sub(r"[']","",line[1])
      line[1]=re.sub(r"[%()-,]"," ",line[1])
      line[1]=re.sub(r"[:]"," ",line[1])
      line[1]=re.sub(r'[" "]+'," ",line[1])
      line[1]=line[1].strip()
      line[1]="<SOS> "+line[1]+" <EOS>"

      proc_lines.append(line) 
    return pd.DataFrame.from_records(data=proc_lines,columns=['eng','ger'])
  else:
    lines=lines.strip()
    lines=''.join(c for c in unicodedata.normalize('NFD', lines) if unicodedata.category(c) != 'Mn')
    lines=lines.encode('utf8','ignore').decode('utf8')
    lines=lines.replace(u'\u200b',' ')
    lines=lines.lower()
    lines=lines.replace('\xa0', ' ')
    lines=re.sub(r"([.,?!;])",r" \1 ",lines)   #adding spaces before and after punctuation
    lines=re.sub(r"[0-9]"," ",lines)
    lines=re.sub(r'["]'," ",lines)
    lines=re.sub(r"[']","",lines)
    lines=re.sub(r"[%-,]"," ",lines)
    lines=re.sub(r"[:]"," ",lines)
    lines=re.sub(r'[" "]+'," ",lines)  #removing excess spaces
    lines=lines.strip() #removing spaces from the end of string
    lines="<SOS> "+lines+" <EOS>"
    return lines

num_examples=20000
with open('deu-eng/deu.txt','r',encoding='utf-8') as f:
  lines=(f.readlines())
lines=lines[:num_examples]

df=process_text(lines,False)
df.drop_duplicates(subset="eng",inplace=True)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('dataset.csv')
print("Processed dataset saved")