#!/usr/bin/env python

# coding: utf-8

 

# In[1]:

 

 

import sys

print(sys.executable)

print(sys.version)

print(sys.version_info)

 

 

# In[2]:

 

 

#!pip install tensorflow_datasets

#https://360digitmg.com/bert-variants-and-their-differences

 

 

# In[3]:

 

 

# Data Source

"""

ag_custom_case_number__c,

ROW_NUMBER() OVER(PARTITION BY ca.ag_custom_case_number__c, pic.name ORDER BY contentver.src_createddate DESC) AS notes_rank,

contentver.src_createddate AS notes_created,

casehistory.src_createddate AS intake_completed,

pic.name as issue_type,

length(contentver.versiondata) AS len,

convert_from(decode(contentver.versiondata, 'base64'),'UTF8') AS notes

FROM

bct_schema."CASE" ca inner join bct_schema."CASE__HISTORY" casehistory on ca.id = casehistory.caseid

inner join bct_schema."CONTENTDOCUMENTLINK" contdoclink on ca.parentid = contdoclink.linkedentityid

inner join bct_schema."CONTENTVERSION" contentver on contdoclink.contentdocumentid = contentver.contentdocumentid

inner join bct_schema."AG_PRODUCT__C" prod on ca.ag_product__c = prod.id

inner join bct_schema."AG_CASE_PRODUCT__C" cprod on ca.parentid = cprod.ag_case__c

inner join bct_schema."AG_DOSAGE_FORM__C" dose_form on cprod.ag_dosage_form__c = dose_form.id

inner join bct_schema."AG_PCM_ISSUE__C" pi on ca.ag_custom_case_number__c  = pi.ag_pcm_sub_case_number_apex__c

inner join bct_schema."AG_PCM_ISSUE_CODE__C" pic on pic.id = pi.ag_as_reported_code__c

inner join bct_schema."AG_PCM_ISSUE_CODE_FAMILY__C" picf on pi.ag_cause_code_family__c = picf.id

WHERE

prod.name = 'Repatha' and dose_form.name = 'Automated mini-doser' and ca.ag_intake_channel_type__c IS NOT NULL and

casehistory.field = 'Status' and casehistory.newvalue = 'Intake Complete' and contentver.src_createddate <= casehistory.src_createddate;

"""

 

 

# In[4]:

 

 

get_ipython().system('pip install pandas')

get_ipython().system('pip install sklearn')

get_ipython().system('pip install tensorflow')

get_ipython().system('pip install tensorflow_hub')

get_ipython().system('pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org tensorflow-text')

get_ipython().system('pip install nltk')

get_ipython().system('pip install gensim')

 

 

# In[5]:

 

 

import numpy as np

import pandas as pd

import os

 

from bs4 import BeautifulSoup

 

import tensorflow as tf

import tensorflow_hub as hub

import tensorflow_text as text

 

from tensorflow import keras

 

from keras.utils import np_utils

from pickle import dump

 

import nltk

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import remove_stopwords

 

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder

 

#!pip freeze > requirements.txt

 

 

# In[7]:

 

 

data = pd.read_csv("C:/Users/gmodi/Downloads/data-1665151649746.csv")

#data = data.query("notes_rank == 1 & len >= 100")

data["issue_type"].value_counts().head(15)

 

 

# In[7]:

 

 

#data = pd.read_csv("C:/Users/gmodi/Downloads/data-1665151649746.csv")

#data["issue_type"].value_counts().head(15)

#data = data.query("notes_rank == 1 & len >= 100")

#data = data.query("len >= 10")

data = data[data["issue_type"].isin(['17 - Delivery Issue',

                                     '16.2 - Activation Issue',

                                     '17.1 - Delivery Issue',

                                     '5.1 - Door Arrived Closed',

                                     '11.8 - IFU Performed Out of Sequence',

                                     '13 - Power Supply Issue',

                                     '7.2 - Door Closure Issue',

                                     '21 - Leakage without Error',

                                     '12 - Self-Test Failure',

                                     '5 - Door Arrived Closed',

                                     '7.1 - Door Closure Issue'

                                     ])]

data = data[["notes","issue_type","len"]]

 

 

# In[8]:

 

 

data

 

 

# In[8]:

 

 

def clean_notes(text):   

    import re

    soup = BeautifulSoup(text, 'html.parser')

    list1 = [item.get_text() for item in list(soup.children)]

    list2 = [i for i in list1 if len(i) == max([len(i) for i in list1])]

    list3 = [re.sub('[^a-zA-Z:]+', ' ', _) for _ in list2]

    return list3[0]

   

data['notes']=data['notes'].apply(lambda cw : clean_notes(cw))

data["len"] = data["notes"].apply(len)

data = data.query(" len > 300 ")

 

 

# In[10]:

 

 

#data['notes'] = [','.join(map(str, l)) for l in data["notes_clean"]]

 

 

# In[9]:

 

 

data["notes"] = data["notes"].apply(lambda x: ' '.join(simple_preprocess(x, min_len=4, max_len=15)))

data["notes"] = data["notes"].apply(lambda x: remove_stopwords(''.join(x)))

data["issue_type"] = data["issue_type"].str.replace(' ', '')

 

 

# In[14]:

 

 

data.head(5)

 

 

# In[15]:

 

 

from sklearn.model_selection import train_test_split

 

 

# In[16]:

 

 

x_train,x_test,y_train,y_test = train_test_split(data[["notes","issue_type"]],data["issue_type"],test_size=0.3)

#x_train,x_test,y_train,y_test = train_test_split(data["notes"],data["issue_type"],test_size=0.3, stratify=data["issue_type"])

 

 

# In[17]:

 

 

x_train.iloc[100]

 

 

# In[18]:

 

 

y_train.iloc[6]

 

 

# In[19]:

 

 

bert_preprocess = hub.KerasLayer(https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3)

bert_encoder = hub.KerasLayer(https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4)

 

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

preprocessed_text = bert_preprocess(text_input)

outputs = bert_encoder(preprocessed_text)

 

l = tf.keras.layers.Dropout(0.2, name="dropout")(outputs['pooled_output'])

l = tf.keras.layers.Dense(11, activation='softmax', name="output")(l)

 

model = tf.keras.Model(inputs=[text_input], outputs = [l])

model.summary()

 

 

# In[20]:

 

 

METRICS = [

      tf.keras.metrics.BinaryAccuracy(name='accuracy'),

      tf.keras.metrics.Precision(name='precision'),

      tf.keras.metrics.Recall(name='recall'),

      tf.keras.metrics.AUC(name='auc')

    ]

 

 

# In[21]:

 

 

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics= METRICS)

 

 

# In[22]:

 

 

# encode class values as integers

encoder = LabelEncoder()

encoder.fit(y_train)

encoded_Y = encoder.transform(y_train)

# convert integers to dummy variables (i.e. one hot encoded)

dummy_y = np_utils.to_categorical(encoded_Y)

encoder_name_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

print(encoder_name_mapping)

 

 

# In[23]:

 

 

dump(encoder, open('encoder.pkl', 'wb'))

 

 

# In[24]:

 

 

model.fit(x_train, dummy_y, epochs=1)

 

 

# In[25]:

 

 

model.save("tf_bert_model_10_21")

reconstructed_model = keras.models.load_model("tf_bert_model_10_21")

 

 

# In[26]:

 

 

reconstructed_model.save("tf_bert_model_h5.h5")

 

 

# In[22]:

 

 

txt = [" Per IBC, patient reports the status light started flashing red about one minute after pressing the start button. Injection site was punctured and pumping noises were heard.  Medication window is clear with a bubble.  Number of Replacements : Please, capture the values below or state “Unknown/Unavailable at time of report”<br>• Dosage: 420mg once a month<br>• Indication: Hyperlipidemia <br>• Device: Pushtronex<br>• Product Complaint Event Description: Per IBC, patient reports the status light started flashing red about one minute after pressing the start button. Injection site was punctured and pumping noises were heard.  Medication window is clear with a bubble.  Number of Replacements: 1<br> : 420mg once a month <br>"]

 

 

# In[49]:

 

 

i = 17

y_test.iloc[i] + "_" + x_test.iloc[i]

 

 

# In[48]:

 

 

temp = encoder.inverse_transform([np.argsort(model.predict([x_test.iloc[i]]))[0][10]]) + "_" + encoder.inverse_transform([np.argsort(model.predict([x_test.iloc[i]]))[0][9]]) + "_" + encoder.inverse_transform([np.argsort(model.predict([x_test.iloc[i]]))[0][8]])

temp[0]

 

 

# In[87]:

 

 

y_test.iloc[i] in temp[0]

 

 

# In[172]:

 

 

y_test = y_test.to_frame()

y_test["predicted"] = " "

 

 

# In[19]:

 

 

y_test.head()

 

 

# In[18]:

 

 

for i in range(len(y_test)):

    temp = encoder.inverse_transform([np.argsort(model.predict([x_test.iloc[i]]))[0][10]]) + "_" + encoder.inverse_transform([np.argsort(model.predict([x_test.iloc[i]]))[0][9]]) + "_" + encoder.inverse_transform([np.argsort(model.predict([x_test.iloc[i]]))[0][8]])

    if y_test["issue_type"].iloc[i] in temp[0]:

        y_test["predicted"].iloc[i] = 1

    else:

        y_test["predicted"].iloc[i] = 0

 

 

# In[175]:

 

 

y_test["predicted"].value_counts()       

 

 

# In[ ]:

 

 

if x_test.issue_type.iloc[i] in x_test.prediction.iloc[i]:

        x_test.predicted.iloc[i] = 1

    else:

        x_test.predicted.iloc[i] = 0

 

 

# In[117]:

 

 

y_test.iloc[1]

 

 

# In[127]:

 

 

y_test = y_test.to_frame()

 

 

# In[128]:

 

 

y_test["predicted"] =x_test.apply(lambda x : encoder.inverse_transform([np.argsort(reconstructed_model.predict([x]))[0][10]])[0])

 

 

# In[131]:

 

 

y_test.head(50)

 

 

# In[129]:

 

 

confusion_matrix(y_test["issue_type"], y_test["predicted"])

 

print(classification_report(y_test["issue_type"], y_test["predicted"]))

print(confusion_matrix(y_test["issue_type"], y_test["predicted"]))

 

 

# In[124]:

 

 

type(y_test)

 

 

# In[70]:

 

 

reconstructed_model.predict(txt)

 

 

# In[71]:

 

 

l1 = [np.sort(reconstructed_model.predict(txt))[0][8],

      np.sort(reconstructed_model.predict(txt))[0][7],

      np.sort(reconstructed_model.predict(txt))[0][6]]

 

 

# In[72]:

 

 

l2 = encoder.inverse_transform([np.argsort(reconstructed_model.predict(txt))[0][8],

                           np.argsort(reconstructed_model.predict(txt))[0][7],

                           np.argsort(reconstructed_model.predict(txt))[0][6]]).tolist()

 

 

# In[75]:

 

 

l2

 

 

# In[73]:

 

 

df = pd.DataFrame(list(zip(l2, l1)), columns =['key', 'value'])

 

 

# In[74]:

 

 

df

 

 

# In[ ]:

 

 

df.iloc[1].key, df.iloc[1].value

 

 

# In[6]:

 

 

#!pip install pandas

#!pip install sklearn

#!pip install tensorflow

#!pip install tensorflow_hub

#!pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org tensorflow-text

 

 

import numpy as np

import pandas as pd

import os

 

import tensorflow as tf

import tensorflow_hub as hub

import tensorflow_text as text

 

from sklearn.model_selection import train_test_split

 

from tensorflow import keras

 

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils

from pickle import load

import re

 

 

txt = [" Per IBC, patient reports the status light started flashing red about one minute after pressing the start button. Injection site was punctured and pumping noises were heard.  Medication window is clear with a bubble.  Number of Replacements : Please, capture the values below or state “Unknown/Unavailable at time of report”<br>• Dosage: 420mg once a month<br>• Indication: Hyperlipidemia <br>• Device: Pushtronex<br>• Product Complaint Event Description: Per IBC, patient reports the status light started flashing red about one minute after pressing the start button. Injection site was punctured and pumping noises were heard.  Medication window is clear with a bubble.  Number of Replacements: 1<br> : 420mg once a month <br>"]

clean = re.compile('<.*?>')

text = re.sub(clean, '', str(txt[0]))

text = re.sub(r"[^a-zA-Z]"," ",str(text))

text = " ".join(text.split())

text

 

encoder = load(open('encoder.pkl', 'rb'))

amd_model = keras.models.load_model("tf_bert_model")

predicted = amd_model.predict([text])

 

l1 = [np.sort(predicted)[0][8],

      np.sort(predicted)[0][7],

      np.sort(predicted)[0][6]]

 

l2 = encoder.inverse_transform([np.argsort(predicted)[0][8],

                           np.argsort(predicted)[0][7],

                           np.argsort(predicted)[0][6]]).tolist()

 

df = pd.DataFrame(list(zip(l2, l1)), columns =['key', 'value'])

 

out = {

    "AutomationId": "data.AutomationId",

    "DosageForm": "data.DosageForm",

    "Product": "data.Product",

    "ProductID": "data.ProductID",

    "MasterCase": "data.MasterCase",

    "PCM_Subcase": "data.PCM_Subcase",

    "OccurCountry": "data.OccurCountry",

    "PCM_ISSUES": [{

        "verbatim": "Needle broken",

        "report_codes": [

            {"reported_code": df["key"].iloc[0], "item_type": "output01", "confidence": df["value"].iloc[0].tolist()},

            {"reported_code": df["key"].iloc[1], "item_type": "output11", "confidence": df["value"].iloc[0].tolist()},

            {"reported_code": df["key"].iloc[2], "item_type": "output21", "confidence": df["value"].iloc[0].tolist()},

        ],

    },]

}

 

 

# In[7]:

 

 

out

 

 

# In[2]:

 

 

predicted

 

 

# In[23]:

 

 

 

 

# In[8]:

 

 

amd_model.predict([text])

 

 

# In[9]:

 

 

notesItems = [text,text,text,text,text]

 

 

# In[13]:

 

 

results = [amd_model.predict([item]) for item in notesItems]

 

 

# In[14]:

 

 

results

 

 

# In[15]:

 

 

type(results)

 

 

# In[16]:

 

 

df

 

 

# In[17]:

 

 

[np.sort(item) for item in results]

 

 

# In[43]:

 

 

np.argsort(predicted)[0][8]

 

 

# In[47]:

 

 

encoder.inverse_transform([np.argsort(item)[0][7] for item in results])

 

 

# In[ ]:

 

 

encoder.inverse_transform([[np.argsort(item) for item in results]])

 

 

# In[ ]:

 

 

encoder.inverse_transform([np.argsort(item) for item in results])

 

 

# In[48]:

 

 

def clean_notes(text):   

    import re

    """remove html"""

    clean = re.compile('<.*?>')

    text = re.sub(clean, '', text)

    """remove special char"""

    text = re.sub(r"[^a-zA-Z0-9]"," ",text)

    text = " ".join(text.split())

    return text

 

 

# In[51]:

 

 

cleanNotesItems = [clean_notes(item) for item in notesItems]

 

 

# In[ ]:

 

 

cleanNotesItems

 

 

# In[54]:

 

 

def predicIssueType(text):

    predicted = amd_model.predict([text])

    return predicted

 

 

# In[59]:

 

 

issuePredicted = [predicIssueType(notes) for notes in cleanNotesItems]

 

 

# In[69]:

 

 

issuePredicted

 

 

# In[63]:

 

 

temp = predicIssueType(cleanNotesItems[0])

 

 

# In[66]:

 

 

temp

 

 

# In[67]:

 

 

amd_model.predict([cleanNotesItems[0]])

 

 

# In[68]:

 

 

issuePredicted = [amd_model.predict([notes]) for notes in cleanNotesItems]

 

 

# In[70]:

 

 

issuePredicted[0]

 

 

# In[86]:

 

 

type(np.sort(issuePredicted))

 

 

# In[87]:

 

 

type(np.argsort(issuePredicted))

 

 

# In[96]:

 

 

temp = np.argsort(issuePredicted)

 

 

# In[102]:

 

 

temp[3][0]

 

 

# In[95]:

 

 

encoder.inverse_transform(temp[0][0])

 

 

# In[85]:

 

 

encoder.inverse_transform(np.argsort(issuePredicted)[0][0])

 

 

# In[103]:

 

 

np.array([encoder.inverse_transform(num[0]) for num in temp])

 

 

# In[119]:

 

 

a1 = np.sort(issuePredicted)