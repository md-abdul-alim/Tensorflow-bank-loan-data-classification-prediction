#!/usr/bin/env python
# coding: utf-8

# In[155]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[156]:


df = pd.read_csv('lending_club_loan_two.csv')
data_info=pd.read_csv('lending_club_info.csv',index_col='LoanStatNew')


# In[157]:


print(data_info.loc['revol_util']['Description'])


# In[158]:


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


# In[159]:


feat_info('mort_acc')


# In[160]:


df.info()


# #info Note: This method prints information about a DataFrame including the index dtype and columns,non-null values and memory usage.

# In[161]:


df.describe()


# describe Note:Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
# 
# DataFrame.count
# 
#     Count number of non-NA/null observations.
# DataFrame.max
# 
#     Maximum of the values in the object.
# DataFrame.min
# 
#     Minimum of the values in the object.
# DataFrame.mean
# 
#     Mean of the values.
# DataFrame.std
# 
#     Standard deviation of the observations.
# DataFrame.select_dtypes
# 
#     Subset of a DataFrame including/excluding columns based on their dtype.

# In[162]:


sns.countplot(x='loan_status',data=df)


# In[163]:


feat_info('loan_status')


# Loan_status Note: blue means who already paid loan. yellow means who did not pay loan yet

# In[164]:


plt.figure(figsize=(15,7))
sns.distplot(df['loan_amnt'],kde=False, bins=100)


# In[165]:


feat_info('loan_amnt')


# loan_amnt Note:
# (x_label= 500tk teke maximum 40000tk range show korano hoise)
# (y_label= total loan taker 0-396030 member)
# 

# In[166]:


df.corr()


# corr Note:
#     Pandas dataframe.corr() is used to find the pairwise correlation of all columns in the dataframe. Any na values are automatically excluded. For any non-numeric data type columns in the dataframe it is ignored.

# In[167]:


#showing correlation by heatmap
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True)#,cmap='viridis' #for coloring


# Explore "installment" feature and print out their descriptions and perform a scatterplot between them

# In[168]:


feat_info('installment')


# In[169]:


feat_info('loan_amnt')


# In[170]:


plt.figure(figsize=(15,8))
sns.scatterplot(x='installment',y='loan_amnt', data=df)


# # Calculate the summary statistics for the loan amount, grouped by the loan_status

# In[171]:


df.groupby('loan_status')['loan_amnt'].describe()


# groupby Note: loan porisod er vitti te koto jon loan porisob koreche r koto jon koreni ta dekhano hoise

# # Let's explore the Grade and subGrade columns that LendingClub attributes to the loans what are the unique possible grades and subgrades?

# In[172]:


df['grade'].unique()


# Create a countplot per grade set the hue to the loan_status label

# In[173]:


sns.countplot(x='grade', data=df, hue='loan_status')
#grade wise loan_status show korano hoise. kon grade e koto jon loan pay korse r koreni


# In[174]:


df['sub_grade'].unique()


# In[175]:


plt.figure(figsize=(15,4))
sns.countplot(x='sub_grade',data=df,hue='loan_status')
#elomelo vabe subgrade ploting


# In[176]:


#we will plot the graph in order again
plt.figure(figsize=(15,4))
subgrade_order=sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order=subgrade_order,palette='coolwarm')#palette just for coloring
#result: we are getting a ordering graph.


# In[177]:


plt.figure(figsize=(15,4))
subgrade_order=sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,hue='loan_status',order=subgrade_order)#palette just for coloring
#result: we are getting a ordering graph.


# In[178]:


#loan_status er sapekke grading(with respect of loan_status we are showing grading)
#sub grading er kisu grade er modde tulona .sob gula na
f_and_g=df[(df['grade']=='G') | (df['grade']=='F')]

plt.figure(figsize=(15,4))
subgrade_order=sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order=subgrade_order,palette='coolwarm',hue='loan_status')


# # Create a new column called 'loan_repaid' which will contain a 1 if the loan status was "fully paid" and a 0 if it was "charged off"

# catagorical teke 0/1 binary te convert kore aro loan pay er bpr ta aro clear vabe dekbo onno column er sate

# In[179]:


df['loan_repaid']=df['loan_status'].map({'Fully Paid':1, 'Charged Off':0})


# In[180]:


df[['loan_repaid','loan_status']]


# Create a bar plot showing the correlation of the numeric features to the new loan_repaid column

# In[181]:


df.corr()['loan_repaid']


# In[182]:


df.corr()['loan_repaid'].plot(kind='bar')


# In[183]:


#we are trying to show how the correation of loan_repaid to others coloumn
df.corr()['loan_repaid'].sort_values().plot(kind='bar')


# In[184]:


#we can solve that problem within one line
#now we drop the loan_repaid column after showing the statistic
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')


# # Dealing with missing data

# In[185]:


#length of the dataset
len(df)


# Now we will show which coloum have how many missing value

# In[186]:


df.isnull().sum()


# Convert this Series to be in term of percentage of the total DataFrame

# In[187]:


#Full dataframe ke 100% dore kon data koto percent missing ache ta check korbo
100*df.isnull().sum()/len(df)


# # Je column er  data missing ache oi column gula analysis kore dekbo koto ta importent. important na hole drop kore dibo

# emp_title column er Data Analysis korbo first

# Let's examine emp_title and emp_length to see whether it will be okay to drop them. Print out their feature information using the feat_info() function from the top of this notebook
# 

# In[188]:


feat_info('emp_title')


# In[189]:


feat_info('emp_length')


# How many unique employment job titles are there?

# In[190]:


df['emp_title'].unique()


# In[191]:


df['emp_title'].nunique()
#this will show how much time all emp_title have


# Pandas dataframe.nunique() function return Series with number of distinct observations over requested axis. If we set the value of axis to be 0, then it finds the total number of unique observations over the index axis. If we set the value of axis to be 1, then it find the total number of unique observations over the column axis. It also provides the feature to exclude the NaN values from the count of unique numbers.

# In[192]:


df['emp_title'].value_counts()
#which title have how many time this will show us


# Realistically there are too many unique job titles to try to convert this to a dummy variable feature. Let's remove that emp_title column

# In[193]:


df=df.drop('emp_title',axis=1)


# In[194]:


df.info()


# In[195]:


df


# # Next emp_length column er Data Analysis korbo

# In[196]:


#df['emp_length'].unique()
##or (Same output)
df['emp_length'].dropna().unique()


# In[197]:


#sort kore choto teke boro sajabo
#dropna() use korbo kono nan type value takle. <1 year er age kono value nai
emp_length_order=sorted(df['emp_length'].dropna().unique())
emp_length_order


# In[198]:


#kono employee koto bosor dore ache seta ber korbo
plt.figure(figsize=(15,4))
sns.countplot(x='emp_length', data= df, order=emp_length_order)


# In[199]:


#loan_status ser sapekke fully paid & charged off kon employee koto bosor dore ache
plt.figure(figsize=(15,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')


# plot out the countplot with a hue separating Fully Paid vs Charged Off

# In[200]:


#ekon amra Fully paid & charged off er radio ber korbo per year
#Step 1: loan_status er sapekke Charged off & Fully Paid ke select korbo


# In[201]:


#jader loan status Charged off tader data sudu
df[df['loan_status']== 'Charged Off']


# In[202]:


#jader loan status Fully Paid tader data sudu
df[df['loan_status']== 'Fully Paid']


# In[203]:


#Step 2: Selected data ke emp_length er sapekke grouping korbo


# In[204]:


df[df['loan_status']== 'Fully Paid'].groupby('emp_length')


# In[205]:


df[df['loan_status']== 'Charged Off'].groupby('emp_length')


# In[206]:


#Step 3: year wise Kon group e koto data ache Ta count korbo


# In[207]:


df[df['loan_status']== 'Fully Paid'].groupby('emp_length').count()


# In[208]:


df[df['loan_status']== 'Charged Off'].groupby('emp_length').count()


# In[209]:


#age emp_length wise all fully pain & charged off data show korse.but amder sudu loan_status er data dorkar.
#tai loan_status er data ber korbo sudu


# In[210]:


df[df['loan_status']== 'Fully Paid'].groupby('emp_length').count()['loan_status']


# In[211]:


df[df['loan_status']== 'Charged Off'].groupby('emp_length').count()['loan_status']


# In[212]:


#Step 5: Ratio ber korar jonno data gulake alada variable e hold korbo


# In[213]:


emp_Fully_paid=df[df['loan_status']== 'Fully Paid'].groupby('emp_length').count()['loan_status']


# In[214]:


emp_Charged_Off=df[df['loan_status']== 'Charged Off'].groupby('emp_length').count()['loan_status']


# In[215]:


#charged off / fully paid je kono ekta dorlei hobe.karon amra just % ratio ta dia check korbo sob year er % similar kina
#emp_Charged_Off/(emp_Charged_Off+emp_Fully_paid)
emp_Fully_paid/(emp_Charged_Off+emp_Fully_paid)


# In[216]:


emp_len_ratio=emp_Fully_paid/(emp_Charged_Off+emp_Fully_paid)
emp_len_ratio.plot(kind='bar')


# In[217]:


#Charge off rates are extremely similar across all employment lengths. Go ahead and drop the emp_length column
# Oporer bar plot teke dekte emp_length sob gulai similar.tai drop kore dibo.final calculation e temon effect hobe na similar datar karone


# In[218]:


df=df.drop('emp_length',axis=1)
df


# # Revisit the DataFrame to see what feature columns still have missing data

# In[219]:


df.isnull().sum()


# In[220]:


# Again title column analysis


# In[221]:


df['purpose'].head(30)
#most of the data are debt_consolidation


# In[222]:


df['title'].head(30)
#most of are also Debt consolidation


# In[223]:


feat_info('purpose')


# In[224]:


feat_info('title')


# Result of title
# 1. title and purpose both are provided by the borrower
# 2. title just string except subcategory/description of the purpose column
# 3. title e missing data ache but purpose e full data ache.jehutu title r purpose same type er data carry kortese tai title column ke drop kore dibo

# In[225]:


df=df.drop('title',axis=1)
df


# In[226]:


df.isnull().sum()


# In[227]:


# Next mort_acc Data Analysis


# In[228]:


df['mort_acc'].head(30)


# Basic analysis of mort_acc column
# 1. mort_acc column have most data missing. around 10% of data missing with respect of total data.so we can't drop this column.
# 2. missing data ke fill up korar jonno mort_acc er sate most highly correlation ache emn column khuje ber korte hobe first.

# In[229]:


#mort_acc data jehutu numeric r on data missing tai drop na kore er sate similar ache emn data dia fill up korar try korbo


# In[230]:


#so find correlation with the mort_acc column
df.corr()['mort_acc'].sort_values()
#Result: mort_acc er sate sob che correlation ache total_acc column er.


# In[231]:


#Let's try this fillna() approach
#Step 1: Group the DataFrame by the total_acc
#Step 2: Calculate the mean value for the mort_acc per total_acc entry.


# In[232]:


#full dataframe er jonno mean() ber korbo with respect total_acc
df.groupby('total_acc').mean()


# In[233]:


#sudu mort_acc er jonn mean() ber korbo
df.groupby('total_acc').mean()['mort_acc']


# In[234]:


total_acc_avg=df.groupby('total_acc').mean()['mort_acc']


# Let's fill in the missing mort_acc values based on their total_acc value. if the mort_acc is missing, then we will fill in that missing value with the mean value corresponding to its total_acc value from the Series we created above. This involves using an .apply() method with two columns

# In[235]:


def fill_mort_acc(total_acc,mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]#mort_acc datar jonno total_acc datar je mean() data ta ber hoisilo oi te return korbe
    else:#missing value na takle mort_acc return hobe else part e
        return mort_acc
        
#np.isnan check korbe missing value ache kina


# In[236]:


df['mort_acc']=df.apply(lambda x: fill_mort_acc(x['total_acc'],x['mort_acc']),axis=1)


# # Result of mort_acc

# In[237]:


#we just fill data with respect of total_acc data
df.isnull().sum()


# In[238]:


#ki type data ache seta check korbo
df['revol_util'].head(30)
#jehutu numeric data r kub besi value missing na tai drop korbona


# In[239]:


df.corr()['revol_util'].sort_values()


# In[240]:


df.groupby('int_rate').mean()['revol_util']


# In[241]:


int_rate_avg=df.groupby('int_rate').mean()['revol_util']


# In[242]:


def fill_revol_util(int_rate,revol_util):
    if np.isnan(revol_util):
        return int_rate_avg[int_rate]#mort_acc datar jonno total_acc datar je mean() data ta ber hoisilo oi te return korbe
    else:#missing value na takle mort_acc return hobe else part e
        return revol_util


# In[243]:


df['revol_util']=df.apply(lambda x: fill_revol_util(x['int_rate'],x['revol_util']),axis=1)


# In[244]:


# Result of revol_util
df.isnull().sum()


# In[245]:


df['pub_rec_bankruptcies'].head(30)


# In[246]:


#most of the data are 0 .that's why will drop it
df=df.drop('pub_rec_bankruptcies',axis=1)


# In[247]:


df.isnull().sum()


# # Categorical Variables and Dummy Variables

# In[248]:


#Now we just need to deal with the string values due to the categorical columns
#List all the columns that are currently non-numeric
df.select_dtypes(['object']).columns


# # term feature

# In[249]:


feat_info('term')


# In[250]:


df['term'].value_counts()


# Convert the term feature into either a 36 or 60 integer numeric data type using .apply() or .map()

# In[251]:


df['term']=df['term'].apply(lambda term: int(term[:3]))


# In[252]:


df['term'].value_counts()
#result: month word ta soria fellam


# # grade feature

# In[253]:


#we already know grade is part of sub_grade, so just drop the grade featuer
df=df.drop('grade',axis=1)


# # subgrade feature

# Convert the subgrade into dummy variables. Then concatenate these new columns to the original darafarame. Remember to drop the original subgrade column and to add drop_first=True to our get_dummies call

# In[254]:


#Step 1: get the dummy variable
#Here we will take a dummy copy of sub_grade first
dummies=pd.get_dummies(df['sub_grade'],drop_first=True)
#use of drop_first=True
#---->drop_first=True is important to use, as it helps in reducing the extra column created during dummy variable creation. Hence it reduces the correlations created among dummy variables
#Step 2: secondly we will drop the sub_grade column & also concat the dummies at the same time.but first sub_grade will remove first
df=pd.concat([df.drop('sub_grade',axis=1),dummies],axis=1)
#Male/Female: we will just take Male 0/1.Don't need to take both.like Male 0 & Female 1.
#for A/B/C :we will take 0 for A & 1 for B. if it is not A or B.Then it will automatically C


# In[255]:


df.columns
#a2,a3,a4.... esob er data gula ke 0/1/2/3/5 esb data te convert kore felbo


# In[256]:


#Check Again the non-numeric
df.select_dtypes(['object']).columns


# In[257]:


#check all catagoricall value to see type
#df['verification_status'].head(30)
#df['purpose'].head(30)
#df['initial_list_status'].head(30)
#df['application_type'].head(30)


# Convert these columns:['verification_status','application_type','initial_list_status','purpose'] into dummy variables and concatenate them with the original dataframe. Remember to set drop_first=True and to drop the original columns

# In[258]:


dummies=pd.get_dummies(df[['verification_status','application_type','initial_list_status','purpose']],drop_first=True)

df=pd.concat([df.drop(['verification_status','application_type','initial_list_status','purpose'],axis=1),dummies],axis=1)


# In[259]:


#Check Again the non-numeric
df.select_dtypes(['object']).columns


# # home_ownership

# In[260]:


df['home_ownership'].head(20)


# In[261]:


df['home_ownership'].value_counts()


# Convert these to dummy variables, but replace NONE and ANY with OTHER, so that we end up with just 4 categories, MORTGAGE, RENT,OWN,OTHER. then concatenate them with the original dataframe. Remember to set drop_first=True and to drop the original columns

# In[262]:


df['home_ownership']=df['home_ownership'].replace(['NONE','ANY'],'OTHER')
#result: Replace NONE & ANY with OTHER


# In[263]:


df['home_ownership'].value_counts()


# In[264]:


#creating dummies
dummies=pd.get_dummies(df['home_ownership'],drop_first=True)

df=pd.concat([df.drop('home_ownership',axis=1),dummies],axis=1)


# In[265]:


#Check Again the non-numeric
df.select_dtypes(['object']).columns


# # address feature

# Let's feature engineer a zip code column from the address in the data set. Create a column called 'zip_code' that extracts the zip code from the address column

# In[266]:


df['address']


# In[267]:


#we will extract zip code from the address column
df['address'].apply(lambda address:address[-5:])


# In[268]:


df['zip_code']=df['address'].apply(lambda address:address[-5:])


# In[269]:


#checking how much type of zip code is there
df['zip_code'].value_counts()


# In[270]:


#creating zip code dummies
dummies=pd.get_dummies(df['zip_code'],drop_first=True)

df=pd.concat([df.drop('zip_code',axis=1),dummies],axis=1)


# In[271]:


#we will drop the original address column
df=df.drop('address',axis=1)
df


# In[272]:


#Check Again the non-numeric
df.select_dtypes(['object']).columns


# # issue_d

# This would be data leakage, we wouldn't know beforehand whether or not a loan would be issued when using our model, so in theory we wouldn't have an issue_date, drop this feature
# 

# In[273]:


feat_info('issue_d')
#ke loan dibe r dibena ber korte hobe.kintu amra janina ke kokon loan korbe.tai ei data amder kono kaje asbe na.drop kore dibo


# In[274]:


df['issue_d'].head(10)


# In[275]:


#kokon ke loan nibe seta amra janina.tai ei coloum er dorkar nia amder model
df=df.drop('issue_d',axis=1)


# In[276]:


df.select_dtypes(['object']).columns


# # earliest_cr_line

# This appears to be a historical time stamp feature. Extract the year from this feature using a .apply function, then convert into a numeric feature. Set this new data to a feature column called 'earliest_cr_year'.Then drop the earliest_cr_line feature

# In[277]:


feat_info('earliest_cr_line')


# In[278]:


df['earliest_cr_line'].head(30)


# In[279]:


df['earliest_cr_line'].value_counts()


# In[280]:


#catagorical month gula delete kore dia sudu numeric year gula rakbo
df['earliest_cr_line']=df['earliest_cr_line'].apply(lambda date: int(date[-4:]))
df['earliest_cr_line']


# In[281]:


#kon year koto bar ache ta check korbo
df['earliest_cr_line'].value_counts()


# In[282]:


df.select_dtypes(['object']).columns


# # loan_status

# drop the loan_status column we created earlier, since its a duplicate of the loan_repaid column. We'll use the loan_repaid column since its already in 0s and 1s

# In[283]:


df['loan_status'].value_counts()


# In[284]:


df['loan_repaid'].value_counts()


# In[285]:


df=df.drop('loan_status',axis=1)


# # Data Preprocessing

# In[286]:


#Train Test Split


# In[287]:


from sklearn.model_selection import train_test_split


# Set X and y variables to the .values of the features and label

# In[288]:


X=df.drop('loan_repaid',axis=1).values


# In[289]:


y=df['loan_repaid'].values


# In[290]:


print(len(df))


# In[291]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# # Normalizing the Data

# Use a MinMaxScaler to normalize the feature data X_train and X_test. Recall we don't want data leakge from the test set so we only fit on the X_train data

# In[292]:


from sklearn.preprocessing import MinMaxScaler


# MinMaxScaler rescales the data set such that all feature values are in the range [0, 1] as shown in the right panel below. However, this scaling compress all inliers in the narrow range [0, 0.005] for the transformed number of households.
# 
# Many machine learning algorithms perform better when numerical input variables are scaled to a standard range. Scaling the data means it helps to Normalize the data within a particular range.
# 
# When MinMaxScaler is used the it is also known as Normalization and it transform all the values in range between (0 to 1) formula is x = [(value - min)/(Max- Min)]

# In[293]:


scaler=MinMaxScaler()


# In[294]:


X_train=scaler.fit_transform(X_train)


# fit() just calculates the parameters (e.g. μ and σ
# 
# in case of StandardScaler) and saves them as an internal object's state. Afterwards, you can call its transform() method to apply the transformation to any particular set of examples.
# 
# fit_transform() joins these two steps and is used for the initial fitting of parameters on the training set x
# , while also returning the transformed x′. Internally, the transformer object just calls first fit() and then transform() on the same data.
# 
# fit the imputer calculates the means of columns from some data, and by transform it applies those means to some data (which is just replacing missing values with the means). If both these data are the same (i.e. the data for calculating the means and the data that means are applied to) you can use fit_transform which is basically a fit followed by a transform.

# In[295]:


X_test=scaler.transform(X_test)


# transform replaces the missing values with a number. By default this number is the means of columns of some data that you choose.

# In[296]:


X_train.shape


# In[297]:


X_test.shape


# # Creating the Model

# In[298]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


# In[299]:


model=Sequential()
#Dense layer: ager layer porer layer er sate kivabe connected takbe
#nuron ki: 
#activatation: je layer a value gula pass hobe oi valure gula jeno ekta fixed range er vitore ase
#https://www.tensorflow.org/api_docs/python/tf/keras/activations
model.add(Dense(78,activation='relu'))#relu=rectified linear unit. #ekane node =78
#dropout:https://towardsdatascience.com/understanding-and-implementing-dropout-in-tensorflow-and-keras-a8a3a02c1bfa?gi=46319f112fcc
model.add(Dropout(0.2))
#dropout note:The primary purpose of dropout is to minimize the effect of overfitting within a trained network.Dropout technique works by randomly reducing the number of interconnecting neurons within a neural network. At every training step, each neuron has a chance of being left out, or rather, dropped out of the collated contribution from connected neurons.

model.add(Dense(39,activation='relu'))
model.add(Dropout(0.2))
#dropout: 
model.add(Dense(19,activation='relu'))
model.add(Dropout(0.2))
#model.add.Softmax() #last layer e je koyta dense takbe tader sob gular maje 100% prediction distribution kore dibe
#softmax: probability distribution: 
model.add(Dense(units=1,activation='sigmoid'))#we use sigmoid when output is 0/1 binary formate
#compile:
#https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
#optimizer er kaj holo kivabe model ta updrage hobe based on the loss 
#https://www.tensorflow.org/api_docs/python/tf/keras/losses
#loss: kono prediction e model ekta data/item er jonno koto vul bolse/result disse setai loss

model.compile(loss='binary_crossentropy',optimizer='adam')


# Fit the model to the training data for at least 25 epochs. Also add in the validation data for later plotting. Optional: add in a batch_size of 256

# In[300]:


model.fit(x=X_train,y=y_train, epochs=25,batch_size=256,validation_data=(X_test,y_test))
#fit method is for traning.
#epochs: koto bar training korbe
#batch_size: 


# # Save my model

# In[301]:


from tensorflow.keras.models import load_model


# In[302]:


model.save('loanmodel.h5')


# # Evaluating Model Performance

# In[305]:


model.history.history


# In[306]:


losses=pd.DataFrame(model.history.history)


# val_loss is the value of cost function for your cross-validation data and loss is the value of cost function for your training data.

# In[312]:


losses
#loss= traing loss & val_loss=test loss


# In[311]:


losses.plot()


# # Classification report and Confusion matrix for the X_test set

# In[313]:


from sklearn.metrics import classification_report,confusion_matrix


# In[314]:


predictions=model.predict_classes(X_test)
print(classification_report(y_test,predictions))


# In[318]:


confusion_matrix(y_test,predictions)


# # Finished

# In[319]:


df['loan_repaid'].value_counts()


# In[321]:


len(df)


# In[320]:


318357/len(df)
#result: data set is normaly giving its own prediction up to 80%.our model prediction 89%.not pretty good


# # Give the customer below, would i offer this person a loan?

# In[336]:


import random
random.seed(101)
random_ind=random.randint(0,len(df))

new_customer=df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer


# In[337]:


new_customer.values


# In[338]:


#we will give shape like our model.our model have 76 features.so we will add this
new_customer.values.reshape(1,77)


# In[339]:


new_customer=scaler.transform(new_customer.values.reshape(1,77))
new_customer


# In[340]:


model.predict_classes(new_customer)
#loan dia jabe kina check korlam.
#result 1.tai loan dia jabe


# Now check, did this person actually end up paying back their loan?

# In[341]:


df.iloc[random_ind]


# In[342]:


df.iloc[random_ind]['loan_repaid']
#result 1: tar mane loan pay back korbe.


# In[ ]:





# In[ ]:




