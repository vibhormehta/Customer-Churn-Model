# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:46:38 2017

@author: vibhor
"""

import pandas as pd
data = pd.read_csv('D:\Datasets\Datasets\Chapter 2/titanic3.csv')


import pandas as pd
import os
path = 'D:/Datasets/Datasets/Chapter 2/'
filename = 'Customer Churn Model.txt'
fullpath = os.path.join(path,filename)
data2 = pd.read_csv(fullpath)

fullpath2 = os.path.join(path,'Customer Churn Model.txt')
data3= pd.read_csv(fullpath2)
data3.columns.value


import pandas as pd
data = pd.read_csv('D:/Datasets/Datasets/Chapter 2/Customer Churn Model.txt')
data.columns.values


data_columns = pd.read_csv(path + 'Customer Churn Columns.csv')
data_column_list = data_columns['Column_Names'].tolist()
data=pd.read_csv(path, + 'Customer Churn Model.txt',header=None,names=data_column_list)
data.columns.values


counter=0
main_dict = {}
for col in cols:
    main_dict[col]=[]


    data=pd.read_csv('D:/Datasets/Datasets/Chapter 2/Customer Churn Model.txt')
    cols=data.next().split(',')
    no_cols=len(cols)


for line in data:
    values = line.strip().split(',')
for i in range(len(cols)):
    main_dict[cols[i]].append(values[i])
    counter += 1
print "The dataset has %d rows and %d columns" % (counter,no_cols)

import pandas as pd
data=pd.read_csv(path + 'titanic3.csv')
data.head()

data.describe()
data.dtypes
data.head(2)

missing1 = pd.isnull(data['body']).values.ravel().sum()
print( 'No. of missing values is %d') %missing1

     
     data.plot(kind='scatter',x='Day Mins',y='Day Charge')
     
     import matplotlib.pyplot as plt
figure,axs = plt.subplots(2, 2,sharey=True,sharex=True)
data.plot(kind='scatter',x='Day Mins',y='Day Charge',ax=axs[0][0])
data.plot(kind='scatter',x='Night Mins',y='Night Charge',ax=axs[0][1])
data.plot(kind='scatter',x='Day Calls',y='Day Charge',ax=axs[1][0])
data.plot(kind='scatter',x='Night Calls',y='Night Charge',ax=axs[1][1])


data.head()




import matplotlib.pyplot as plt
plt.hist(data['Day Calls'],bins=8)
plt.xlabel('Day Calls Value')
plt.ylabel('Frequency')
plt.title('Frequency of Day Calls')


import matplotlib.pyplot as plt
plt.boxplot(data['Day Calls'])
plt.ylabel('Day Calls')
plt.title('Box Plot of Day Calls')


import matplotlib.pyplot as plt
plt.boxplot(data['Day Calls'])
plt.ylabel('Day Calls')
plt.title('Box Plot of Day Calls')

data1=data[(data['Total Mins']>500) & (data['State']=='VA')]
data1.shape

account_length=data['Account Length']
account_length.head()
type(account_length)

wanted=['Account Length','VMail Message','Day Calls']
column_list=data.columns.values.tolist()
sublist=[x for x in column_list if x not in wanted]
subdata=data[sublist]
subdata.head(1)


data1=data[data['State']=='VA']
data1.shape

subdata_first_50=data[['Account Length','VMail Message','Day Calls']][1:50]
subdata_first_50

import numpy as np
Vibhor = np.random.random()

ht = np.random.randint(0,1)

def randint_range(n,a,b):
    x=[]
    for i in range(n):
        x.append(np.random.randint(a,b))
        return x

randint_range(10,2,1000)

import random
for i in range(3):
    print random.randrange(0,100,5)
    
column_list = data.columns.values.tolist()
random.choice(column_list)

np.random.seed(565)
for i in range(5):
    print np.random.random()
    
    
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
a=np.random.uniform(1,100,10000)
b=range(1,101)
plt.hist(a)


import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
a=np.random.randn(3)
b=range(1,101)
plt.hist(a)


x=np.random.random(0,1,1000)
y=np.random.random(0,1,1000).tolist()

pi_avg=0
pi_value_list=[]
for i in range(100):
    value=0
    x=np.random.uniform(0,1,1000).tolist()
    y=np.random.uniform(0,1,1000).tolist()
    for j in range(1000):
        z=np.sqrt(x[j]*x[j]+y[j]*y[j])
        if z<=1:
            value+=1
    float_value=float(value)
    pi_value=float_value*4/1000
    pi_value_list.append(pi_value)
    pi_avg+=pi_value
pi=pi_avg/100
print pi
ind=range(1,101)
fig=plt.plot(ind,pi_value_list)
fig


def pi_run(nums,loops):
    pi_avg=0
    pi_value_list=[]
    for i in range(loops):
        value=0
        x=np.random.uniform(0,1,nums).tolist()
        y=np.random.uniform(0,1,nums).tolist()
        for j in range(nums):
            z=np.sqrt(x[j]*x[j]+y[j]*y[j])
            if z<=1:
                value+=1
        float_value=float(value)
        pi_value=float_value*4/nums
        pi_value_list.append(pi_value)
        pi_avg+=pi_value
    pi=pi_avg/loops
    ind=range(1,loops+1)
    fig=plt.plot(ind,pi_value_list)
    return (pi,fig)


pi_run(1000,1000)


import numpy as np
import pandas as pd
a=['Male','Female']
b=['Rich','Poor','Middle Class']
gender=[]
seb=[]
for i in range(1,101):
    gender.append(np.random.choice(a))
    seb.append(np.random.choice(b))
height=30*np.random.randn(100)+155
weight=20*np.random.randn(100)+60
age=10*np.random.randn(100)+35
income=1500*np.random.randn(100)+15000

                           
df=pd.DataFrame({'Gender':gender,'Height':height,'Weight':weight,'Age':age,
'Income':income,'Socio-Eco':seb})
df.head()

path = 'c:/users/vibhor/documents/'
ab_file = 'abandoned_data_seed.csv'
re_file = 'reservation_data_seed.csv'
import os
ab_path = os.path.join(path,ab_file) 
ab_path

import numpy as np
import pandas as pd
import sklearn as sk
abandoned = pd.read_csv(ab_path)
reservation= pd.read_csv(os.path.join(path,re_file))

len(abandoned)

len(abandoned['test_control'] = 'test')