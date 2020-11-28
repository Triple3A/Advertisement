#!/usr/bin/env python
# coding: utf-8

# In[115]:


import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats


# پروژه صفر: آشنایی با هوش مصنوعی
# هدف از این پروژه پرکردن نواقص یک دیتاست با پیشبینی آنها به کمک تحلیل آماری است

# جدولی از ویژگی ها داده شده است که باتوجه به هر ویژگی برای بعضی از سطرها مشخص شده است که کاربر روی تبلیغ کلیک کرده است یا خیر. برای بعضی از سطرها مشخص نشده است که باید با توجه به سطرهای پر شده مقدار سطرهای خالی را تخمین زد

# In[116]:


#1
df = pd.read_csv('advertising_dataset.csv')
df #'df' is dataFrame. It contains all of the informations


# In[117]:


df.tail(n=5) #return the last n-rows


# In[118]:


df.head(n=5) #return the first n-rows


# In[119]:


df.describe(include='all') #describe specifications of columns
#count = Number of non-NaNs
#unique = Number of unique objects
#top = The object with the highest frequency
#freq = The frequency of the top
#mean = Mean of columns
#std = Std of columns
#min = minimum of columns
#25% = 25% of maximum
#50% = 50% of maximum
#75% = 75% of maximum
#max = maximum of maximum


# In[120]:


#2
print(df.info())
#'info' method show the information of each column such as name of column, number of non-NaN elements, Dtype of column, number of each Dtype and memory usage
df.replace({'Gender': {'Female': 0, 'Male': 1}}, inplace=True)
df
#Label Encoding
#Replace the Female with 0 and Male with 1


# In[121]:


#3
print(df.isna().sum()) #print number of NaN for each column
for col in df.select_dtypes(include = ['int64', 'float64']).columns:
#select_dtypes method returns the columns with specific types
    if(col != 'Clicked on Ad'):
        df[col].fillna(df[col].mean(), inplace=True) #fill the NaN elements with mean of the column
print(df.isna().sum()) #to prove that nan members were filled
df #printing the DataFrame


# In[123]:


#4
print('Male: ', (df['Gender'] == 1).sum()) #number of Male
print('Female: ', (df['Gender'] == 0).sum()) #number of Female
print('Clicked on Ad: ', (df['Clicked on Ad'] == 1.0).sum()) #number of members who clicked on Ad
print('Didn\'t click on Ad: ', (df['Clicked on Ad'] == 0.0).sum()) #number of members who didn't click on Ad


# In[124]:


#5
print('Number of members who are Male and have atleast 21 years: ', ((df['Gender'] == 1) & (df['Age'] > 20)).sum())


# In[125]:


#6
#Vectorization solution
ticVec = time.time()
print(df.groupby('Clicked on Ad')['Age'].mean())
#Show the mean of 'Age' column according to the 'Clicked on Ad' column
tocVec = time.time()
durVec = (tocVec - ticVec) * 1000
#Calculate the time of vectorization


# In[129]:


#7
#For loop solution
ticFor = time.time()
size = df.shape[0]

clicked = 0.0
notClicked = 0.0
numClicked = 0
numNotClicked = 0

#Doing for loop in each elements of 'Age' column
for i in range(size):
    if(df['Clicked on Ad'][i] == 1.0):
        clicked += df['Age'][i]
        numClicked += 1
    elif(df['Clicked on Ad'][i] == 0.0):
        notClicked += df['Age'][i]
        numNotClicked += 1
meanClicked = clicked / numClicked
meanNotClicked = notClicked / numNotClicked
#Calculate the mean of 'Age' column according to the 'Clicked on Ad' column
print('Mean of not clicked: ', meanNotClicked)
print('Mean of clicked: ', meanClicked)
tocFor = time.time()
durFor = (tocFor - ticFor) * 1000
#Calculate the time of for loop
print('Vectorization: %f ms' %(durVec))
print('For loop %f ms' %(durFor))
#Vectorization is 10 times faster


# In[130]:


#8
df.hist(bins = 50)


# تابع بالا نمودار توزیع هر ستون را نمایش می دهد و ورودی
# 
# bins = 50
# 
# باعث می شود که در نمودار ۵۰ ستون وجود داشته باشد
# این نمودار نشان می دهد که فراوانی هر عضو هر ستون در مقایسه با بقیه اعضای آن ستون چقدر است
# برای مثال جنسیت خانم ها بیشتر از جنسیت مردهاست

# In[131]:


#9
dfNumbers = df.select_dtypes(include = ['float64']).copy(deep=True)
for col in dfNumbers.columns:
    if(col != 'Clicked on Ad'):
        dfNumbers[col] = (dfNumbers[col] - dfNumbers[col].mean()) / dfNumbers[col].std()
#The operation of normalization for each column except 'Clicked on Ad' to improve the information
dfNumbers #printing the dataFrame


# In[132]:


#10
means = dfNumbers.groupby('Clicked on Ad').mean()
#Calculate the mean of each column according to 'Clicked on Ad' column
stds = dfNumbers.groupby('Clicked on Ad').std()
#Calculate the STD(Standard Deviation) of each column according to 'Clicked on Ad' column
x = {}
yClicked = {}
yNotClicked = {}
for col in dfNumbers.columns:
    if(col != 'Clicked on Ad'):
        x[col] = dfNumbers[col].sort_values(ascending=True)
        #Sorting the elements of 'col' column
        yClicked[col] = scipy.stats.norm.pdf(x[col], means[col][1], stds[col][1])
        #Calculate the pdf of elements of 'col' column for who clicked on ad
        yNotClicked[col] = scipy.stats.norm.pdf(x[col], means[col][0], stds[col][0])        
        #Calculate the pdf of elements of 'col' column for who didn't click on ad
        plt.plot(x[col], yClicked[col], color = 'blue')
        plt.plot(x[col], yNotClicked[col], color = 'red')
        plt.xlabel('x')
        plt.ylabel('Normal Distribution')
        plt.title(col)
        plt.show()
        plt.figure()


# نمودار آبی برای کلیک شده ها و نمودار قرمز برای کلیک نشده هاست
# هرچه انحراف معیار کلیک شده ها کمتر باشد آن ویژگی بهتری است زیرا بازه ای که دیتاها درآن قرار دارند کوچکتر است بنابراین دیتاهایی که نزدیک به میانگین هستند به احتمال بهتری می توان آنها را تخمین زد
# 
# هرچه فاصله میانگین های دو نمودار بیشتر باشد آن ویژگی بهتری است زیرا برخورد نمودار ها در قسمت های پایینی نمودارها صورت می گیرد و لذا تصمیم گیری آسان تر می شود
# 
# نمودارهای کلیک نشده بالا تیزی یکسانی دارند(یعنی انحراف معیار تقریبا یکسانی دارند) بنابراین باتوجه به آن نمی توان تصمیم گیری کرد که کدام ویژگی بهتر است
# از بین نمودارهای کلیک شده نمودار 'استفاده روزانه از اینترنت' تیزتر است(یعنی انحراف معیار کمتری دارد) و همینطور فاصله میانگین دو نمودار برای این ویژگی از بقیه نمودارها بیشتر است
# 
# بنابراین بهترین ویژگی 'استفاده روزانه از اینترنت' است. ولی راهکار بهتر این است که یک ترکیبی از ویژگی ها باشد و فقط از یک ویژگی برای تخمین استفاده نشود

# In[133]:


#11
attribute = 'Daily Internet Usage'
conditions = scipy.stats.norm.pdf(dfNumbers[attribute], means[attribute][1], stds[attribute][1]) > scipy.stats.norm.pdf(dfNumbers[attribute], means[attribute][0], stds[attribute][0])
#check the condition for each row
dfOut = dfNumbers[dfNumbers['Clicked on Ad'].isnull()].copy(deep=True)
conditions = conditions[dfNumbers['Clicked on Ad'].isnull()]

dfOut[conditions] = dfOut.fillna(1.0)
#fill NaN with 1.0 for elements that have True condition
dfOut = dfOut.fillna(0.0)
#fill NaN with 0.0 for rest of NaN elements(that have False condition)
dfOut.to_csv('Predictions.csv', columns = ['Clicked on Ad'])
#csv file has index column and 'Clicked on Ad' column
dfOut


# برای تخمین باید مقدار تابع چگالی احتمال را برای هر عضو ستون استفاده روزانه از اینترنت برای دو حالت کلیک شده و کلیک نشده بدست آورد و نمودار هرکدام که بالاتر بود در آن نقطه(یعنی مقدار چگالی احتمال بیشتری داشت) نشان دهنده کلیک شدن یا نشدن روی آن تبلیغ است زیرا به میانگین آن نمودار نزدیک تر است و بنابراین احتمال اینکه این داده متعلق به آن نمودار باشد بیشتر است
