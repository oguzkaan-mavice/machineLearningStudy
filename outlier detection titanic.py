# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 14:13:19 2021

@author: kaanm
"""

# kütüphaneler yüklüyoruz 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# veri setini yüklüyoruz

train=pd.read_csv(r"C:\Users\kaanm\OneDrive\Desktop\Git Hub Project\machineLearningStudy\train.csv")

train.head()

train.info()

#%% Değişkenleri anlamak

# numeric değişkenler:
    # Age,SibSp,Parch,Fare,Pclass,PassengerId,Survived,

# categoric değişkenler:
    # Name,Sex,Ticket,Cabin,Embarked

# PassengerId --> yolcunun yol kimliği : numeric olarak değerlendirmemeliyiz
# Survived --> yolcunun kaza sonucu kurtulup kurtulmadığını gösterir. Binary : numeric değil, target variable
# Pclass --> yolcunun seyahat ettiği bilet tipine göre sosyo-ekonomik durumunu gösterir. Numeric olarak değerlendirmemeliyiz. Ordinal
# Name --> yolcunun ismi
# Sex --> yolcunun cinsiyeti
# Age --> Yolcunun yaşı
# SibSp --> Yolcunun sahip olduğu kardeş/eş sayısı
# Parch --> yolcunun sahip olduğu anne-baba-çocuk sayısı
# Ticket --> biletin numarası
# Fare --> Yolcunun bilete ödediği para
# Cabin --> Yolcunun cabin numarası
# Embarked --> Yolcunun yola çıktığı liman

# Numeric : Age, SibSp, Parch, Fare
# Categoric : Sex, Embarked, Cabin, Name, Ticket, PassengerId
# Ordinal : Pclass
# Target : Survived

#%% Aykırı değerlere ilk bakış

fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(20,8))

ax1.boxplot(train["Age"])
ax1.set_title("Age")

ax2.boxplot(train["SibSp"])
ax2.set_title("SibSp")

ax3.boxplot(train["Parch"])
ax3.set_title("Parch")

ax4.boxplot(train['Fare'])
ax4.set_title("Fare")


# aykırı değerlerin dağılımları

fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(20,8))

ax1.hist(train["Age"])
ax1.set_title("Age")

ax2.hist(train["SibSp"])
ax2.set_title("SibSp")

ax3.hist(train["Parch"])
ax3.set_title("Parch")

ax4.hist(train['Fare'])
ax4.set_title("Fare")

# 4 ü de sola yatık, normal dağılıma sahip değil. Sadece Age normal dağılımı yakın
# Normal dağılım olsaydı Z score kullanırdık. Z ckoru ortalama ve standart sapma kullanarak
# hesaplanır, ondan dolayı aykırı değerlerden etkilenir.
# Onun yerine aykırı değerlere dayanıklı olan IQR ve MAD kullanacağız.

#%% IQR - MAD


def iqr(df,x):
    q1 = np.quantile(df[x],0.25)
    q3 = np.quantile(df[x],0.75)
    diff = q3 - q1
    lower_t = q1 - (1.5 * diff)
    upper_t = q3 + (1.5 * diff)
    return df[(df[x]<lower_t) | df[x]>upper_t]

age_output = iqr(train,"Age")
SibSp_output = iqr(train,"SibSp")
Parch_output = iqr(train,"Parch")
Fare_output = iqr(train,"Fare")

print(len(age_output),len(SibSp_output),len(Parch_output),len(Fare_output))

def double_mad(df,x):
    c = 1.4826
    q2 = np.median(df[x])
    bot_val = df.loc[df[x] <= df[x].median(),x]
    bot_mad = (abs(bot_val - q2).median()) * c
    up_val = df.loc[df[x] >= df[x].median(),x]
    up_mad = (abs(up_val-q2).median()) * c
    lower_t = q2 - (3 * bot_mad)
    upper_t = q2 + (3 * up_val)
    return df[(df[x] < lower_t) | ( df[x] > upper_t)]
    
age_mad_output = iqr(train,"Age")
SibSp_mad_output = iqr(train,"SibSp")
Parch_mad_output = iqr(train,"Parch")
Fare_mad_output = iqr(train,"Fare")  

print(len(age_mad_output),len(SibSp_mad_output),len(Parch_mad_output),len(Fare_mad_output))  

# age, fare, sibsp değişkenlerinde aykırı değere rastlamadık
# Parch değişkeninde datanın yaklaşıl 25% aykırı olarak geliyor, bunun sebebi dağılımdan dolayı

#%% Aykırı değerlere istatistiksel bakış

train[["Age","SibSp","Parch","Fare"]].agg(['skew','kurtosis']).transpose()

# Çarpıklık değeri:

# -0.5 ile 0.5 arasında ise; neredeyse simetrik dağılıma,
# -0.5 ile -1.0 veya 0.5 ile 1.0 arasında ise; orta derecede çarpık dağılıma,
# -1.0’den veya 1.0’ten büyük ise; değişken çok çarpık dağılıma sahiptir.

# age simetrik dağılıma sahipken, diğerleri yüksek çarpıklığa sahip.

# Log ve Sqrt işlemleri ile dönüşüm

train["SibSp_Log"]=np.log(train["SibSp"])
train["SibSp_Sqrt"]=np.sqrt(train["SibSp"])

train[["SibSp_Log","SibSp_Sqrt"]].agg(["skew","kurtosis"]).transpose()

# çarpıklığını sqrt methodu ile azalttık, log uygulayamıyoruz çünkü değişkende 0 değerleri mevcut.


train["Parch_Log"]=np.log(train["Parch"])
train["Parch_Sqrt"]=np.sqrt(train["Parch"])

train[["Parch_Log","Parch_Sqrt"]].agg(["skew","kurtosis"]).transpose()

# çarpıklığını sqrt methodu ile azalttık, log uygulayamıyoruz çünkü değişkende 0 değerleri mevcut.

train["Fare_Log"]=np.log(train["Fare"])
train["Fare_Sqrt"]=np.sqrt(train["Fare"])

train[["Fare_Log","Fare_Sqrt"]].agg(["skew","kurtosis"]).transpose()

# çarpıklığını sqrt methodu ile azalttık, log uygulayamıyoruz çünkü değişkende 0 değerleri mevcut.

#%%
# son bakış
    
SibSp_Sqrt_mad_output = iqr(train,"SibSp_Sqrt")
Parch_Sqrt_mad_output = iqr(train,"Parch_Sqrt")
Fare_Sqrt_mad_output = iqr(train,"Fare_Sqrt")  

print(len(SibSp_Sqrt_mad_output),len(Parch_Sqrt_mad_output),len(Fare_Sqrt_mad_output))



fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(20,8))

ax1.hist(train["Age"])
ax1.set_title("Age")

ax2.hist(train["SibSp_Sqrt"])
ax2.set_title("SibSp_Sqrt")

ax3.hist(train["Parch_Sqrt"])
ax3.set_title("Parch_Sqrt")

ax4.hist(train['Fare_Sqrt'])
ax4.set_title("Fare_Sqrt")

