#!/usr/bin/env python
# coding: utf-8

# # RANDOM FORESTS

# In[3]:


from fastai.imports import *
np.set_printoptions(linewidth=130)


# In[ ]:


df = pd.read_csv('train.csv')
tst_df = pd.read_csv('test.csv')
modes = df.mode().iloc[0]


# In[ ]:


df.mode().iloc[0]


# In[ ]:


def proc_data(df):#processing the data
    df['Fare'] = df.Fare.fillna(0)
    df.fillna(modes, inplace=True)
    df['LogFare'] = np.log1p(df['Fare'])
    df['Embarked'] = pd.Categorical(df['Embarked'])
    df['Sex'] = pd.Categorical(df['Sex'])

proc_data(df)
proc_data(tst_df)


# In[6]:


df.Sex.head()


# In[6]:


cats=["Sex","Embarked"]
conts=['Age', 'SibSp', 'Parch', 'LogFare',"Pclass"]
dep="Survived"


# In[7]:


df


# In[7]:


df.Sex.cat.codes.head()


# ## Binary splits - A random forest is a ensemble of trees and a tree is a ensemble of binary splits

# In[8]:


import seaborn as sns


# In[9]:


fig,axs = plt.subplots(1,2, figsize=(11,5))
sns.barplot(data=df, y=dep, x="Sex", ax=axs[0]).set(title="Survival rate")
sns.countplot(data=df, x="Sex", ax=axs[1]).set(title="Histogram");#WE CAN SEE THAT SURVIVAL RATES OF MALE AND FEMALES ARE VERY DIFFERENT 


# In[10]:


from numpy import random
from sklearn.model_selection import train_test_split

random.seed(42)
trn_df,val_df = train_test_split(df, test_size=0.25)#SPLTTING INTO TRAINING AND TEST SETS USING SCIKIT LEARN 
trn_df[cats] = trn_df[cats].apply(lambda x: x.cat.codes)#CONVERTING ALL CATEGORICAL VARIABLES INTO THERE CODES
val_df[cats] = val_df[cats].apply(lambda x: x.cat.codes)


# In[11]:


def xs_y(df):#RETURNS THE INDEPENDENT AND DEPENDENT VARIABLES
    xs = df[cats+conts].copy() #CATEGORICAL PLUS CONSTANT VARIABLES GIVE INDEPENDENT VARIABLES OR X 
    return xs,df[dep] if dep in df else None

trn_xs,trn_y = xs_y(trn_df)#train
val_xs,val_y = xs_y(val_df)#validation 


# In[11]:


get_ipython().run_line_magic('pinfo', 'copy')


# In[12]:


preds = val_xs.Sex==0 #they survived if they r female (true if sex is female)


# In[13]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(val_y,preds)


# In[14]:


df_fare = trn_df[trn_df.LogFare>0]
fig,axs = plt.subplots(1,2, figsize=(11,5))
sns.boxenplot(data=df_fare, x=dep, y="LogFare", ax=axs[0])
sns.kdeplot(data=df_fare, x="LogFare", ax=axs[1]);


# In[15]:


preds = val_xs.LogFare>2.7 #trying to predict using fares distribution


# In[16]:


mean_absolute_error(val_y, preds) #worse error


# In[17]:


def _side_score(side, y): #to find a score  how good of a binary split it is whether it is categorical or continous or whatever data 
    tot = side.sum() # a good split is in which all of the dependent variables on one side are all pretty much the same, if all the males have the same out come that is did not survive and all the females have the same outcome that is did survive then that is a good split
    if tot<=1: return 0
    return y[side].std()*tot#multiplied by the number of count in that group


# In[18]:


def score(col, y, split): 
    lhs = col<=split
    return (_side_score(lhs,y) + _side_score(~lhs,y))/len(y)#lhs+rhs


# In[19]:


score(trn_xs["Sex"], trn_y, 0.5)


# In[20]:


score(trn_xs["LogFare"], trn_y, 0.5) #lower score is better


# In[21]:


def iscore(nm, split):
    col = trn_xs[nm]
    return score(col, trn_y, split)

from ipywidgets import interact
interact(nm=conts, split=15.5)(iscore);


# In[22]:


interact(nm=cats, split=2)(iscore);


# In[23]:


#to automate the above 2 cells


# In[24]:


nm = "Age"
col = trn_xs[nm]
unq = col.unique()#take all the unq values of age and try each one
unq.sort()


# In[25]:


unq


# In[26]:


scores = np.array([score(col, trn_y, o) for o in unq if not np.isnan(o)])
unq[scores.argmin()]#find the minimum one(gives index)


# In[27]:


def min_col(df, nm):
    col,y = df[nm],df[dep]
    unq = col.dropna().unique()
    scores = np.array([score(col, y, o) for o in unq if not np.isnan(o)])
    idx = scores.argmin()
    return unq[idx],scores[idx]

min_col(trn_df, "Age")


# In[28]:


cols = cats+conts
{o:min_col(trn_df, o) for o in cols}


# In[29]:


# this is called one-r model ,one of the best classifiers , dont assume you have to go complicated 


# # DESCISION TREE

# In[30]:


cols.remove("Sex")
ismale = trn_df.Sex==1
males,females = trn_df[ismale],trn_df[~ismale]


# In[31]:


# Now let's find the single best binary split for males
{o:min_col(males, o) for o in cols}


# In[32]:


{o:min_col(females, o) for o in cols} #SAME FOR FEMALES # the least in this 0.3335 is the biggest factor of these that will predict whether a female will survive or not


# In[33]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz

m = DecisionTreeClassifier(max_leaf_nodes=4).fit(trn_xs, trn_y);


# In[34]:


import graphviz

def draw_tree(t, df, size=10, ratio=0.6, precision=2, **kwargs):
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True, rounded=True,
                      special_characters=True, rotate=False, precision=precision, **kwargs)
    return graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s))


# In[35]:


draw_tree(m, trn_xs, size=10)


# In[36]:


#bad idea to be a adult male in titanic
#in females pclass less than 2.5 , 116 of them survived
#gini is a way to see how good the split is , lesser the gini better it is, 2nd leaf node is 5050 so it is bad
def gini(cond):
    act = df.loc[cond, dep]
    return 1 - act.mean()**2 - (1-act).mean()**2


# In[37]:


gini(df.Sex=='female'), gini(df.Sex=='male')


# In[38]:


# for one-r loss was 0.215
mean_absolute_error(val_y, m.predict(val_xs)) #loss for the descision tree with 4 leaf nodes


# In[39]:


m = DecisionTreeClassifier(min_samples_leaf=50) #minimum of 50 values in each leaf node
m.fit(trn_xs, trn_y)
draw_tree(m, trn_xs, size=12)


# In[40]:


mean_absolute_error(val_y, m.predict(val_xs)) #better


# In[41]:


tst_df[cats] = tst_df[cats].apply(lambda x: x.cat.codes)
tst_xs,_ = xs_y(tst_df)

def subm(preds, suff):
    tst_df['Survived'] = preds
    sub_df = tst_df[['PassengerId','Survived']]
    sub_df.to_csv(f'sub-{suff}.csv', index=False)

subm(m.predict(tst_xs), 'tree') #SCORE IS LIL WORSE BUT WE DID NOT HAVE TO DO MORE PREPROCESSING HARD WORK IN DECISION TREES COMPARED TO OUR LINEAR AND NEURAL NETWORK MODELS


# In[42]:


#FOR TABULAR DATA DECISION TREES ARE VERY USEFUL


# In[43]:


df.Embarked.head()


# In[44]:


df.Embarked.cat.codes.head()


# ### HOW TO MAKE THIS MORE ACCURATE

# In[45]:


#THERE ARE LIMITS TO WHAT A DECISION TREE COULD DO
#WE CAN USE BAGGING
#LETS SAY WE ARE USING NOT SO GOOD MODEL , WE CAN USE MAKE LOTS AND LOTS OF SLIGHTLY DIFFERENT DECISION TREES THEN IF WE AVERAGE THOSE ERRORS 0 WILL COME 
#MINDBLOWING INSIGHT


# ### RANDOM FOREST
# WE ARE CREATING BUNCH OF DECISION TREES USING DIFFERENT SUBSETS OF DATA 

# In[46]:


def get_tree(prop=0.75):
    n = len(trn_y)
    idxs = random.choice(n, int(n*prop))
    return DecisionTreeClassifier(min_samples_leaf=5).fit(trn_xs.iloc[idxs], trn_y.iloc[idxs])


# In[47]:


trees = [get_tree() for t in range(100)]


# In[48]:


all_probs = [t.predict(val_xs) for t in trees]
avg_probs = np.stack(all_probs).mean(0)

mean_absolute_error(val_y, avg_probs)


# In[49]:


#THIS WAS IDENTICAL TO RANDOMFOREST CLASSIFIER OF SKLEARN
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(100, min_samples_leaf=5)
rf.fit(trn_xs, trn_y);
mean_absolute_error(val_y, rf.predict(val_xs))


# In[50]:


subm(rf.predict(tst_xs), 'randforest') #slightly more worse but in ideal real world datasets, randomforest does a better job than single tree


# In[51]:


# One particularly nice feature of random forests is they can tell us which independent variables were the most important in the model, using feature_importances_:


# In[52]:


pd.DataFrame(dict(cols=trn_xs.columns, imp=m.feature_importances_)).plot('cols', 'imp', 'barh');


# In[54]:


# it shows which improved the gini most, for any dataset we can find the variable which is most important


# In[ ]:


#having more trees always improves accuracy tiny bits not more than 100 trees 
#always start  a tabular model with random forest

