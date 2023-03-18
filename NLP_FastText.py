


#!/usr/bin/env python

# coding: utf-8

 

# ### Other Dosage Forms

 

# In[1]:

 

 

"""

SELECT

ag_custom_case_number__c,

prod.name,

dose_form.name,

ROW_NUMBER() OVER(PARTITION BY ca.ag_custom_case_number__c, pic.name ORDER BY contentver.src_createddate DESC) AS notes_rank,

pic.name as issue_type,

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

dose_form.name IN ('Solution for injection in pre-filled pen','Vial - liquid','Vial - lyophilized','Software based device','Tablet') and

ca.ag_intake_channel_type__c IS NOT NULL and

casehistory.field = 'Status' and

casehistory.newvalue = 'Intake Complete' and

contentver.src_createddate <= casehistory.src_createddate;

"""

 

 

# In[2]:

 

 

import sys

print(sys.executable)

print(sys.version)

print(sys.version_info)

 

 

# In[3]:

 

 

import numpy as np

import pandas as pd

import texthero as hero

from texthero import stopwords

from texthero import preprocessing

from texthero import visualization

from texthero import representation

 

from bs4 import BeautifulSoup

 

import fasttext

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import remove_stopwords

 

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

from sklearn import preprocessing

 

 

# In[10]:

 

 

masterData = pd.read_csv("C:/Users/gmodi/Downloads/data-1666644445166.csv")

#masterData = pd.read_csv("C:/Users/gmodi/Downloads/data-1671237323737.csv")

#masterData["issue_type"] = masterData["issue_type"].str.replace(' ', '_').replace('/', '_').replace('-', '')

masterData["issue_type"] = masterData["issue_type"].str.replace(r'[^0-9a-zA-Z:,]+', '_')

masterData["len"] = masterData["notes"].apply(len)

 

 

# In[11]:

 

 

masterData.head(5)

 

 

# In[12]:

 

 

# Functions

 

 

# In[13]:

 

 

def clean_notes(text):   

    import re

    soup = BeautifulSoup(text, 'html.parser')

    list1 = [item.get_text() for item in list(soup.children)]

    list2 = [i for i in list1 if len(i) == max([len(i) for i in list1])]

    list3 = [re.sub('[^a-zA-Z:]+', ' ', _) for _ in list2]

    return list3[0]

 

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):

    lb = preprocessing.LabelBinarizer()

    lb.fit(y_test)

    y_test = lb.transform(y_test)

    y_pred = lb.transform(y_pred)

    return roc_auc_score(y_test, y_pred, average=average)

 

 

# In[14]:

 

 

def normalize(s):

    """

    Given a text, cleans and normalizes it. Feel free to add your own stuff.

    """

    s = s.lower()

    # Replace ips

    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)

    # Isolate punctuation

    s = re.sub(r'([.\(\)\!\?\-\\\/\,])', r' \1 ', s)

    # Remove some special characters

    s = re.sub(r'([\;\:\|•«\n])', ' ', s)

    # Replace numbers and symbols with language

    s = s.replace('&', ' and ')

    s = s.replace('@', ' at ')

    s = s.replace('0', ' zero ')

    s = s.replace('1', ' one ')

    s = s.replace('2', ' two ')

    s = s.replace('3', ' three ')

    s = s.replace('4', ' four ')

    s = s.replace('5', ' five ')

    s = s.replace('6', ' six ')

    s = s.replace('7', ' seven ')

    s = s.replace('8', ' eight ')

    s = s.replace('9', ' nine ')

    return s

 

 

# # Solution for injection in pre-filled pen

 

# In[15]:

 

 

data = masterData.query(" name == 'Solution for injection in pre-filled pen' ").copy()

valueCount = data["issue_type"].value_counts(normalize=True).to_frame().cumsum()*100

data = data[data["issue_type"].isin(valueCount.index.tolist()[0:9])]

data = data[["notes","issue_type","len"]]

 

data['notes']=data['notes'].apply(lambda cw : clean_notes(cw))

data = data.query(" len > 500 ")

 

data["notes"] = data["notes"].apply(lambda x: ' '.join(simple_preprocess(x, min_len=4, max_len=15)))

data["notes"] = data["notes"].apply(lambda x: remove_stopwords(''.join(x)))

 

 

# In[16]:

 

 

data["pca"] = (data["notes"].pipe(representation.tfidf, max_features=100).pipe(representation.pca))

hero.scatterplot(data, col="pca", color="issue_type", title="PCA issue_type")

 

 

# In[17]:

 

 

data["labeled_notes"] = data["issue_type"].apply(lambda x: '__label__' + x + " " ) + data["notes"]

 

x_train,x_test,y_train,y_test = train_test_split(data[["labeled_notes","issue_type"]],data["issue_type"],test_size=0.30)

x_train.to_csv("C:/Users/gmodi/Downloads/x_train.csv",index=False,header=False)

x_test.to_csv("C:/Users/gmodi/Downloads/x_test.csv",index=False,header=False)

 

model = fasttext.train_supervised(input="C:/Users/gmodi/Downloads/x_train.csv", wordNgrams=4, epoch = 100, lr = 1)

model.test("C:/Users/gmodi/Downloads/x_test.csv",k=3)

 

 

# In[18]:

 

 

# predict the data

x_test["predicted"] = x_test["labeled_notes"].apply(lambda x: model.predict(x)[0][0]).str.replace('__label__','')

 

#Create the confusion matrix

print(classification_report(x_test["issue_type"], x_test["predicted"]))

print(confusion_matrix(x_test["issue_type"], x_test["predicted"]))

multiclass_roc_auc_score(x_test["issue_type"], x_test["predicted"])

 

 

# In[19]:

 

 

x_test["prediction"] = x_test["labeled_notes"].apply(lambda x: model.predict(x,3)).astype(str).replace('__label__','')

#x_test["prediction"] = x_test["prediction"].astype(str)

#x_test["prediction"] = x_test["prediction"].str.replace('__label__','')

 

for i in range(len(x_test)):

    if x_test.issue_type.iloc[i] in x_test.prediction.iloc[i]: x_test.predicted.iloc[i] = 1

    else: x_test.predicted.iloc[i] = 0

 

#x_test.to_csv("C:/Users/gmodi/Downloads/x_test_results.csv")       

x_test["predicted"].value_counts(normalize=True)*100      

 

 

# In[20]:

 

 

model.save_model("C:/Users/gmodi/MyProjects/OtherDosageForms/FastText_SIPFP.bin")

model.save_model("C:/Users/gmodi/MyProjects/OtherDosageForms/FastText_SIPFP.ftz")

 

 

# # Vial - liquid

 

# In[21]:

 

 

#data = masterData.query(" name == 'Vial - liquid' ").copy()

data = masterData

valueCount = data["issue_type"].value_counts(normalize=True).to_frame().cumsum()*100

data = data[data["issue_type"].isin(valueCount.index.tolist()[0:9])]

data = data[["notes","issue_type","len"]]

 

data['notes']=data['notes'].apply(lambda cw : clean_notes(cw))

data = data.query(" len > 500 ")

 

data["notes"] = data["notes"].apply(lambda x: ' '.join(simple_preprocess(x, min_len=4, max_len=15)))

data["notes"] = data["notes"].apply(lambda x: remove_stopwords(''.join(x)))

 

 

# In[22]:

 

 

data["pca"] = (data["notes"].pipe(representation.tfidf, max_features=100).pipe(representation.pca))

hero.scatterplot(data, col="pca", color="issue_type", title="PCA issue_type")

 

 

# In[23]:

 

 

data["labeled_notes"] = data["issue_type"].apply(lambda x: '__label__' + x + " " ) + data["notes"]

 

x_train,x_test,y_train,y_test = train_test_split(data[["labeled_notes","issue_type"]],data["issue_type"],test_size=0.20)

x_train.to_csv("C:/Users/gmodi/Downloads/x_train.csv",index=False,header=False)

x_test.to_csv("C:/Users/gmodi/Downloads/x_test.csv",index=False,header=False)

 

model = fasttext.train_supervised(input="C:/Users/gmodi/Downloads/x_train.csv", wordNgrams=4, epoch = 100, lr = 1)

model.test("C:/Users/gmodi/Downloads/x_test.csv",k=3)

 

 

# In[24]:

 

 

# predict the data

x_test["predicted"] = x_test["labeled_notes"].apply(lambda x: model.predict(x)[0][0]).str.replace('__label__','')

 

#Create the confusion matrix

confusion_matrix(x_test["issue_type"], x_test["predicted"])

 

print(classification_report(x_test["issue_type"], x_test["predicted"]))

print(confusion_matrix(x_test["issue_type"], x_test["predicted"]))

multiclass_roc_auc_score(x_test["issue_type"], x_test["predicted"])

 

 

# In[25]:

 

 

x_test["prediction"] = x_test["labeled_notes"].apply(lambda x: model.predict(x,3)).astype(str).replace('__label__','')

#x_test["prediction"] = x_test["prediction"].astype(str)

#x_test["prediction"] = x_test["prediction"].str.replace('__label__','')

 

x_test["predicted"] = ""

for i in range(len(x_test)):

    if x_test.issue_type.iloc[i] in x_test.prediction.iloc[i]: x_test.predicted.iloc[i] = 1

    else: x_test.predicted.iloc[i] = 0

 

#x_test.to_csv("C:/Users/gmodi/Downloads/x_test_results.csv")       

x_test["predicted"].value_counts(normalize=True)*100     

 

 

# In[26]:

 

 

model.save_model("C:/Users/gmodi/MyProjects/OtherDosageForms/FastText_Vial_liquid.bin")

model.save_model("C:/Users/gmodi/MyProjects/OtherDosageForms/FastText_Vial_liquid.ftz")

 

 

# # Vial - lyophilized

 

# In[27]:

 

 

#data = masterData.query(" name == 'Vial - lyophilized' ").copy()

data = masterData

valueCount = data["issue_type"].value_counts(normalize=True).to_frame().cumsum()*100

data = data[data["issue_type"].isin(valueCount.index.tolist()[0:14])]

data = data[["notes","issue_type","len"]]

 

data['notes']=data['notes'].apply(lambda cw : clean_notes(cw))

data = data.query(" len > 200 ")

 

data["notes"] = data["notes"].apply(lambda x: ' '.join(simple_preprocess(x, min_len=4, max_len=15)))

data["notes"] = data["notes"].apply(lambda x: remove_stopwords(''.join(x)))

 

 

# In[28]:

 

 

valueCount.head(20)

 

 

# In[29]:

 

 

data = data.query(" issue_type not in ['customer_feedback','other','To_be_determined']").copy()

 

 

# In[30]:

 

 

data["pca"] = (data["notes"].pipe(representation.tfidf, max_features=100).pipe(representation.pca))

hero.scatterplot(data, col="pca", color="issue_type", title="PCA issue_type")

 

 

# In[31]:

 

 

data["labeled_notes"] = data["issue_type"].apply(lambda x: '__label__' + x + " " ) + data["notes"]

 

x_train,x_test,y_train,y_test = train_test_split(data[["labeled_notes","issue_type"]],data["issue_type"],test_size=0.30)

x_train.to_csv("C:/Users/gmodi/Downloads/x_train.csv",index=False,header=False)

x_test.to_csv("C:/Users/gmodi/Downloads/x_test.csv",index=False,header=False)

 

model = fasttext.train_supervised(input="C:/Users/gmodi/Downloads/x_train.csv", wordNgrams=4, epoch = 100, lr = 1)

model.test("C:/Users/gmodi/Downloads/x_test.csv",k=3)

 

 

# In[32]:

 

 

# predict the data

x_test["predicted"] = x_test["labeled_notes"].apply(lambda x: model.predict(x)[0][0]).str.replace('__label__','')

 

print(classification_report(x_test["issue_type"], x_test["predicted"]))

print(confusion_matrix(x_test["issue_type"], x_test["predicted"]))

multiclass_roc_auc_score(x_test["issue_type"], x_test["predicted"])

 

 

# In[33]:

 

 

x_test["prediction"] = x_test["labeled_notes"].apply(lambda x: model.predict(x,3)).astype(str).replace('__label__','')

#x_test["prediction"] = x_test["prediction"].astype(str)

#x_test["prediction"] = x_test["prediction"].str.replace('__label__','')

 

x_test["predicted"] = ""

for i in range(len(x_test)):

    if x_test.issue_type.iloc[i] in x_test.prediction.iloc[i]: x_test.predicted.iloc[i] = 1

    else: x_test.predicted.iloc[i] = 0

 

#x_test.to_csv("C:/Users/gmodi/Downloads/x_test_results.csv")       

x_test["predicted"].value_counts(normalize=True)*100     

 

 

# In[34]:

 

 

model.words  

 

 

# In[35]:

 

 

model.labels

 

 

# In[36]:

 

 

model.wordNgrams

 

 

# In[37]:

 

 

model.get_word_vector('drug_particles').shape

 

 

# In[38]:

 

 

model.get_nearest_neighbors('overdose')

 

 

# In[39]:

 

 

model.get_nearest_neighbors('interface_needle')

 

 

# In[40]:

 

 

model.get_nearest_neighbors('syringe')

 

 

# In[41]:

 

 

model.save_model("C:/Users/gmodi/MyProjects/OtherDosageForms/FastText_Vial_lyophilized.bin")

model.save_model("C:/Users/gmodi/MyProjects/OtherDosageForms/FastText_Vial_lyophilized.ftz")

 

 

# # Software based device

 

# In[42]:

 

 

data = masterData.query(" name == 'Software based device' ").copy()

valueCount = data["issue_type"].value_counts(normalize=True).to_frame().cumsum()*100

data = data[data["issue_type"].isin(valueCount.index.tolist()[0:9])]

data = data[["notes","issue_type","len"]]

 

data['notes']=data['notes'].apply(lambda cw : clean_notes(cw))

data = data.query(" len > 500 ")

 

data["notes"] = data["notes"].apply(lambda x: ' '.join(simple_preprocess(x, min_len=4, max_len=15)))

data["notes"] = data["notes"].apply(lambda x: remove_stopwords(''.join(x)))

 

 

# In[43]:

 

 

data["pca"] = (data["notes"].pipe(representation.tfidf, max_features=100).pipe(representation.pca))

hero.scatterplot(data, col="pca", color="issue_type", title="PCA issue_type")

 

 

# In[44]:

 

 

data["labeled_notes"] = data["issue_type"].apply(lambda x: '__label__' + x + " " ) + data["notes"]

 

x_train,x_test,y_train,y_test = train_test_split(data[["labeled_notes","issue_type"]],data["issue_type"],test_size=0.30)

x_train.to_csv("C:/Users/gmodi/Downloads/x_train.csv",index=False,header=False)

x_test.to_csv("C:/Users/gmodi/Downloads/x_test.csv",index=False,header=False)

 

model = fasttext.train_supervised(input="C:/Users/gmodi/Downloads/x_train.csv", wordNgrams=4, epoch = 100, lr = 1)

model.test("C:/Users/gmodi/Downloads/x_test.csv",k=3)

 

 

# In[45]:

 

 

# predict the data

x_test["predicted"] = x_test["labeled_notes"].apply(lambda x: model.predict(x)[0][0]).str.replace('__label__','')

 

print(classification_report(x_test["issue_type"], x_test["predicted"]))

print(confusion_matrix(x_test["issue_type"], x_test["predicted"]))

multiclass_roc_auc_score(x_test["issue_type"], x_test["predicted"])

 

 

# In[46]:

 

 

x_test["prediction"] = x_test["labeled_notes"].apply(lambda x: model.predict(x,3)).astype(str).replace('__label__','')

#x_test["prediction"] = x_test["prediction"].astype(str)

#x_test["prediction"] = x_test["prediction"].str.replace('__label__','')

 

x_test["predicted"] = ""

for i in range(len(x_test)):

    if x_test.issue_type.iloc[i] in x_test.prediction.iloc[i]: x_test.predicted.iloc[i] = 1

    else: x_test.predicted.iloc[i] = 0

 

#x_test.to_csv("C:/Users/gmodi/Downloads/x_test_results.csv")       

x_test["predicted"].value_counts(normalize=True)*100 

 

 

# In[47]:

 

 

model.save_model("C:/Users/gmodi/MyProjects/OtherDosageForms/FastText_SBD.bin")

model.save_model("C:/Users/gmodi/MyProjects/OtherDosageForms/FastText_SBD.ftz")

 

 

# # Tablet

 

# In[48]:

 

 

data = masterData.query(" name == 'Tablet' ").copy()

valueCount = data["issue_type"].value_counts(normalize=True).to_frame().cumsum()*100

data = data[data["issue_type"].isin(valueCount.index.tolist()[0:9])]

data = data[["notes","issue_type","len"]]

 

data['notes']=data['notes'].apply(lambda cw : clean_notes(cw))

data = data.query(" len > 500 ")

 

data["notes"] = data["notes"].apply(lambda x: ' '.join(simple_preprocess(x, min_len=4, max_len=15)))

data["notes"] = data["notes"].apply(lambda x: remove_stopwords(''.join(x)))

 

 

# In[49]:

 

 

data["pca"] = (data["notes"].pipe(representation.tfidf, max_features=100).pipe(representation.pca))

hero.scatterplot(data, col="pca", color="issue_type", title="PCA issue_type")

 

 

# In[50]:

 

 

data["labeled_notes"] = data["issue_type"].apply(lambda x: '__label__' + x + " " ) + data["notes"]

 

x_train,x_test,y_train,y_test = train_test_split(data[["labeled_notes","issue_type"]],data["issue_type"],test_size=0.30)

x_train.to_csv("C:/Users/gmodi/Downloads/x_train.csv",index=False,header=False)

x_test.to_csv("C:/Users/gmodi/Downloads/x_test.csv",index=False,header=False)

 

model = fasttext.train_supervised(input="C:/Users/gmodi/Downloads/x_train.csv", wordNgrams=3, epoch = 100, lr = 1)

print(model.test("C:/Users/gmodi/Downloads/x_test.csv",k=3))

 

# predict the data

x_test["predicted"] = x_test["labeled_notes"].apply(lambda x: model.predict(x)[0][0]).str.replace('__label__','')

 

print(classification_report(x_test["issue_type"], x_test["predicted"]))

print(confusion_matrix(x_test["issue_type"], x_test["predicted"]))

multiclass_roc_auc_score(x_test["issue_type"], x_test["predicted"])

 

 

# In[51]:

 

 

x_test["prediction"] = x_test["labeled_notes"].apply(lambda x: model.predict(x,3)).astype(str).replace('__label__','')

#x_test["prediction"] = x_test["prediction"].astype(str)

#x_test["prediction"] = x_test["prediction"].str.replace('__label__','')

 

x_test["predicted"] = ""

for i in range(len(x_test)):

    if x_test.issue_type.iloc[i] in x_test.prediction.iloc[i]: x_test.predicted.iloc[i] = 1

    else: x_test.predicted.iloc[i] = 0

 

#x_test.to_csv("C:/Users/gmodi/Downloads/x_test_results.csv")       

x_test["predicted"].value_counts(normalize=True)*100 

 

 

# In[52]:

 

 

model.save_model("C:/Users/gmodi/MyProjects/OtherDosageForms/FastText_Tablet.bin")

model.save_model("C:/Users/gmodi/MyProjects/OtherDosageForms/FastText_Tablet.ftz")

 

 

# In[53]:

 

 

PCM_ISSUES, report_codes = [],[]

 

 

# In[54]:

 

 

sampleRequest = """ <p><span style="font-size: 10pt;">Wellbean nurse </span><span style="font-size: 13.3333px;">Francis</span><span style="font-size: 10pt;"> (colleague of Richard Norris) called back on 07-Aug-2019 at 15:16 provided information in addition to 19-0000093-PC-01</span></p><p><span style="font-size: 10pt;">Â </span></p><p><span style="font-size: 10pt;">Caller stated they did not have further information to provide except injection site was on patientâ€™s leg. The activation button was pressed but the Sureclick pen did not work. There were no click sound, no needle penetration and no partial dose received from the complained unit. No replacement is required.</span></p><p><span style="font-size: 10pt;">Â </span></p><p><b style="font-size: 10pt;">ACTIVATION / INJECTION ISSUES </b></p><p><span style="font-size: 10pt;">1. Was the inspection window clear prior to injection? If not, what color was it? â€“ unknown by Wellbean nurse</span></p><p><span style="font-size: 10pt;">2. Did you remove the needle cap immediately prior to injection? â€“ unknown by Wellbean nurse</span></p><p><span style="font-size: 10pt;">3. Did the needle safety cover retract into the device when pushed against the skin? â€“ unknown by Wellbean nurse</span></p><p><span style="font-size: 10pt;">4. Were you able to press the activation button? â€“ unknown by Wellbean nurse, but the patient stated â€œthe button was pressedâ€</span></p><p><span style="font-size: 10pt;">5. Did the needle pierce the skin? â€“ No</span></p><p><span style="font-size: 10pt;">6. Did the inspection window change color prior to lifting the device from the skin? â€“ unknown by Wellbean nurse</span></p><p><span style="font-size: 10pt;">If yes, did the color change completely or partially? â€“ N/A</span></p><p><span style="font-size: 10pt;">If yes, did the inspection window take more than 15 seconds to change color? â€“ N/A</span></p><p><span style="font-size: 10pt;">7. Currently, how is the inspection window? (Clear, Fully Yellow, Partially Yellow) â€“ unknown by Wellbean nurse</span></p><p><span style="font-size: 10pt;">8. After administration or attempted administration, is the needle protruding beyond the needle safety cover? â€“ unknown by Wellbean nurse</span></p>"""

clean_notes(sampleRequest)

 

 

# In[ ]:

 

 

##### Fast API Main Python File.

 

 

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel

import uvicorn

import numpy as np

import pandas as pd

import re

import nltk

from bs4 import BeautifulSoup

import fasttext

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import remove_stopwords

 

def clean_notes(text):   

    soup = BeautifulSoup(text, 'html.parser')

    list1 = [item.get_text() for item in list(soup.children)]

    list2 = [i for i in list1 if len(i) == max([len(i) for i in list1])]

    list3 = [re.sub('[^a-zA-Z:]+', ' ', _) for _ in list2]

    return list3[0]   

 

# Declaring our FastAPI instance

app = FastAPI()

 

# Defining path operation for root endpoint

@app.get("/")

def main():

    return {

        "message": "Welcome to Amgen AI!"

    }

class request_body(BaseModel):

    AutomationId: str

    DosageForm: str

    Product: str

    ProductID: str

    MasterCase: str

    PCM_Subcase: str

    OccurCountry: str

    PPQ: str

    Notes: str

 

@app.post("/AMD")

def AMD(data: request_body):

    amd_model = fasttext.load_model("FastText_AMD.ftz")

    issuePredicted = amd_model.predict(clean_notes(data.Notes),k=3)

    PCM_ISSUES, report_codes = [],[]

    for j in range(3):

        report_codes.append({'reported_code': issuePredicted[0][j].replace('__label__',''), 'item_type': 'AMD','confidence': round(issuePredicted[1][j]*100,2)})

    PCM_ISSUES.append({'verbatim': list3[0], 'report_codes': report_codes})   

    return {

        "AutomationId": data.AutomationId,

        "DosageForm": data.DosageForm,

        "Product": data.Product,

        "ProductID": data.ProductID,

        "MasterCase": data.MasterCase,

        "PCM_Subcase": data.PCM_Subcase,

        "OccurCountry": data.OccurCountry,

        "PPQ": data.PPQ,

        "PCM_ISSUES": PCM_ISSUES

     }

