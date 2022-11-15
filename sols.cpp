// ========================================================================================
// ASSIGNMENT 1

// q1a
import numpy as np

arr = np.array([1,2,3,6,4,5])
newArray = np.flipud(arr)

print("Reversed Array : ",newArray)

// q1b
import numpy as np

array1 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]])

# Method - 1 (using flatten method)
# Row Major Flattening
array2 = array1.flatten('C')
print(array2)

# Column Major Flattening
array3 = array1.flatten('F')
print(array3)


# Method - 2 (using reshape method)
array4 = array1.reshape(-1)
print(array4)


// q1c
import numpy as np

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[1, 2], [3, 4]])

print(f"Array 1 = {arr1}")
print(f"Array 2 = {arr2}")

# is greater comparison
print("arr1 > arr2")
print(np.greater(arr1,arr2))

# is greater than equal to comparison
print("arr1 >= arr2")
print(np.greater_equal(arr1,arr2))

# is less than comparison
print("arr1 < arr2")
print(np.less(arr1,arr2))

# is less than equal to comparison
print("arr1 <= arr2")
print(np.less_equal(arr1,arr2))

# is equal to comparison
print("arr1 = arr2")
print(np.equal(arr1,arr2))


# another method to check equality
if (np.array_equal(arr1,arr2)):
    print("arr1 == arr2")

else:
    print("They are not equal")


// q1d
import numpy as np

# i
x = np.array([1,2,3,4,5,1,2,1,1,1])
print("Most frequently occuring element is = ", np.bincount(x).argmax());

# ii
y = np.array([1,1,1,2,3,4,2,4,3,3])
print("Most frequently occuring element is = ", np.bincount(y).argmax());


// q1e
import numpy as np

gfg = np.matrix('[4,1,9; 12,3,1; 4,5,6]')

# (i) Sum Of all Elements
print("Sum of all elements of matrix = ", gfg.sum())

# (ii) Sum of all Elements Row-wise
print("Sum of all Elements Row-wise = ", gfg.sum(axis=1))

# (iii) Sum of all Elements Column-wise
print("Sum of all Elements Columns-wise = ", gfg.sum(axis=0))

// q1f
import numpy as np

n_array = np.array([[55,25,15],[30,44,2],[11,45,77]])

# (i) Sum of diagonal elements
print("Sum of diagonal elements of the Matrix = ", np.trace(n_array))

a, b = np.linalg.eig(n_array)
# (ii) Eigen values of matrix
print("Eigen Values of the Matrix are = ", a)

# (iii) Eigen Vectors of the Matrix
print("Eigen Vectors of the Matrix are = ", b)

# (iv) Inverse of the matrix
print("Inverse of Matrix = ", np.linalg.inv(n_array))

# (v) Determinant of the matrix
print("Determinant of matrix = ", np.linalg.det(n_array))

// q1g
import numpy as np

# (i)

p = [[1,2],[2,3]]
q = [[4,5],[6,7]]

print("Product of the two matrices = ", np.matmul(p,q))

print("Covariance between the two matrices = ", np.cov(p,q))


# (ii)

p1 = [[1,2],[2,3],[4,5]]
q1 = [[4,5,1],[6,7,2]]

print("Product of the two matrices = ", np.matmul(p1,q1))
# the line below produces error
# because covariance can be calculate for matrices of same dimensions only
# print("Covariance between the two matrices = ", np.cov(p1,q1))


// q1h
import numpy as np

x = np.array([[2,3,4],[3,2,9]])
y = np.array([[1,5,0],[5,10,3]])

# Inner Product
print("Inner Product = ", np.inner(x,y))

# Outer Product
print("Outer Product = ", np.outer(x,y))

# Cartesian Product
print("Cartesian Product = ", np.cross(x,y))


// q2a
import numpy as np

array1 = np.array([[1,-2,3],[-4,5,-6]])

# (i) element wise absolute value
print("Element wise absolute value = ", np.absolute(array1))

# (ii) Percentile
np.ndarray.flatten(array1)


print("25th percentile along every row : ", np.percentile(array1, 25, axis=1))
print("25th percentile along every column : ", np.percentile(array1, 25, axis=0))


print("50th percentile along every row : ", np.percentile(array1, 50, axis=1))
print("50th percentile along every column : ", np.percentile(array1, 50, axis=0))


print("75th percentile along every row : ", np.percentile(array1, 75, axis=1))
print("75th percentile along every column : ", np.percentile(array1, 75, axis=0))


# (iii) Mean Median Mode

print("Mean of each Column : ", np.mean(array1,axis=0))
print("Mean of each Row : ", np.mean(array1,axis=0))

print("Median of each Column : ", np.median(array1,axis=0))
print("Median of each Row : ", np.median(array1,axis=0))

print("Standard Deviation of each Column : ", np.std(array1,axis=0))
print("Standard Deviation of each Row : ", np.std(array1,axis=0))


// q2b
import numpy as np

a = np.array([-1.8,-1.6,-0.5,0.5,1.6,1.8,3.0])

# floor
print ("Floor : ", np.floor(a))

# ceiling
print ("Ceiling : ", np.ceil(a))

# truncated value
print ("Truncated Values : ", np.trunc(a))

# Rounded values
print ("Rounded Values : ", np.round(a))

// q3a
import numpy as np

a = np.array([10,52,62,16,16,54,453])

# (i) sorted array
print ("Sorted array : ", np.sort(a))

# (ii) indices of sorted array
print ("Indices of sorted array : ", np.argsort(a))

# (iii) 4 smallest elements
a = np.sort(a)
print ("4 smallest elements : ", a[:4])

# (iv) 5 largest elements
print ("5 largest elements : ", a[-5:])


// q3b
from unicodedata import decimal
import numpy as np

a = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])

# (i) integer elements only 
result1 = a[a == a.astype(int)]
print ("Integer Elements only : ", result1)

# (ii) float elements only
print ("Float Elements only : ", a[a != a.astype(int)])


// q4a
from distutils import text_file
import numpy as np
from PIL import Image

def write_pixel(t, handle):
    handle.write("%d, "%t)

def write_pixel_col(t, handle):
    handle.write("%d %d, "%(t[0],t[1]))


def img_to_array(path,chroma):
    text_file_handle = open("out.txt","w")

    im = Image.open(path)

    width, height = im.size

    for column in range(0,width):
        for row in range(0,height):
            if (chroma == 1):
                write_pixel(im.getpixel((column,row)), text_file_handle)
            else:
                write_pixel_col(im.getpixel((column,row)), text_file_handle)
            text_file_handle.write("\n")
        
    print ("Data Successfully updated")
    text_file_handle.close()


path = "gray.jpg"
chroma = int(input("Enter 1 if image is grayscale. Otherwise Enter 2 : "))

img_to_array(path,chroma)


// ===========================================================================


// ASSIGNMENT 2 

// IMPORTING LIBRARIES 
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

// IMPORTING DATA 
df = pd.read_csv('./AWCustomers.csv')
df1 = pd.read_csv('./AWSales.csv')

// MERGING DF AND DF1 ON THE BASIS OF 'CUSTOMERID'
df = pd.merge(df, df1, on='CustomerID')

// TO PRINT ALL THE COLUMNS NAMES OF THE DATAFRAME 
df.columns

// TO SEE COUNT OF NON-NULL VALUES AND DATATYPE OF VALUES IN COLUMNS FOR THE DATAFRAME 
df.info()

// TO SEE COUNT OF NULL VALUES IN COLUMNS FOR THE DATAFRAME 
df.isnull().sum()


// NOW AS WE CAN SEE FROM ABOVE INFORMATION THAT WE HAVE SO MUCH
// NULL VALUES IN TITLE MIDDLENAME ADDRESSLINE2 ETC 
// SO ITS GOOD TO REMOVE THEM FROM DF
// SECONDLY WE'LL ALSO REMOVE CUSTOMERID, FIRSTNAME, LASTNAME, PHONENUMBER ETC 
// FIELD AS BUYER'S PURCHASE BEHAVIOUR HAVE NOTHING TO DO WITH NAME
df.drop(['CustomerID', "Title", 'FirstName', 'MiddleName', 'LastName', 'AddressLine2', "Suffix"], axis=1, inplace=True)


// TO GET COUNT OF UNIQUE VALUES IN COLUMNS IN THE DATAFRAME 
df.unique()


// NOW DROP ADDRESSLINE1, PHONENUMBER, BIRTHDATE ETC. BECAUSE THEY LARGE AMOUNT OF UNIQUE DATA
df.drop(['AddressLine1', 'PhoneNumber', 'BirthDate'], axis=1, inplace=True)


// NOW CONVERT ALL THE CATEGORICAL DATA INTO NUMERICAL FORM, 
df['CountryRegionName'] = pd.factorize(df['CountryRegionName'])[0]
df['City'] = pd.factorize(df['City'])[0]
df['Education'] = pd.factorize(df['Education'])[0]
df['Occupation'] = pd.factorize(df['Occupation'])[0]
df['Gender'] = pd.factorize(df['Gender'])[0]
df['MaritalStatus'] = pd.factorize(df['MaritalStatus'])[0]


// NOW WE WILL USE THE BIRTHDATE COLUMN AND MAKE A AGE COLUMN USING THAT. 
df['BirthDate']= pd.to_datetime(df['BirthDate'])

// import datetime
currentTime = datetime.datetime.now()
def get_age(birth_date,today=currentTime):
    y=today-birth_date
    return y.days    //365

df['Age']=df['BirthDate'].apply(lambda x: get_age(x))

// DROP THE BIRTHDATE COLUMN NOW 
df.drop(['BirthDate'],axis=1,inplace=True)

// NOW LETS FIND REALTIONS BETWEEN THE DATA 
df.corr()

// THEN WE CAN MAKE A HEATMAP FOR THE DATAFRAME,  
sb.heatmap(df.corr(),cmap="Blues")


// NOW WE WILL APPLY NORMALIZATION ON THESE THREE COLUMNS, 
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled=scaler.fit_transform(df[['YearlyIncome','Age','AvgMonthSpend']])
df['YearlyIncome_scaled']=scaled[:,0]
df['Age_scaled']=scaled[:,1]
df['AvgMonthSpend_scaled']=scaled[:,2]
df.drop(['YearlyIncome','Age','AvgMonthSpend'],axis=1,inplace=True)



// TO APPLY ONE HOT ENCODING TO THE DATA, TO DO BINARISATION 
newdata = pd.get_dummies(data, columns = ['Remarks', 'Gender'])
print(one_hot_encoded_data)


// TO FIND COSINE DISTANCE, JACCARD DISTANCE BTW TWO COLUMNS
from scipy.spatial import distance
distance.cosine(df['Education'].values,df['AvgMonthSpend_scaled'].values)

distance.jaccard(df['Education'].values,df['AvgMonthSpend_scaled'].values)


// =========================================================================================




// ASSIGNMENT 3 

// importing libraries 
import pandas as pd
import numpy as np
df=pd.read_csv("https://raw.githubusercontent.com/girikgarg8/ML_and_Data_Science_Datasets/master/USA_Housing.csv")

// to see information of the dataset
df.info()

// to print first 5 values 
df.head()

// #Step-1 Removing noise
import missingno as ms
ms.bar(df) // #every entry is 5000, so no missing data here


// #Step-2 checking for redundancy among input features
import seaborn as sns
sns.heatmap(df.iloc[:,0:5].corr(),annot=True)
// #as no two features have correlation greater than equal to 0.7/0.8, feature elimination is not required


// #Step-3 Split input and output features
X=df.iloc[:,0:5]
Y=df.iloc[:,5]
Y=np.array(Y)
Y=Y.reshape(-1,1) // # -1 means we want pandas to figure out number of rows by itself
print(Y)


// #Scaling the values of input features
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_ScaledOriginal=X_scaled
X_scaled=np.insert(X_scaled,0,values=1,axis=1) // #inserting a column of 1's in the X matrix
print(X_scaled)


// #Train-test split
from sklearn.model_selection import KFold
kf=KFold(n_splits=5)


// #using k fold(or kfold) 5 times ,calculating regression coefficients 
// #also predicting values on test set and performance evaluation
for train_index,test_index in kf.split(X_scaled):
  x_train,x_test,y_train,y_test=X_scaled[train_index],X_scaled[test_index],Y[train_index],Y[test_index]
  A=x_train.T.dot(x_train)
  B=np.linalg.inv(A)
  C=B.dot(x_train.T)
  beta=C.dot(y_train)
  print ('Beta matrix is ')
  print( beta,end='\n \n')
  print ('First five predicted values are ')
  predicted=x_test.dot(beta)
  print (predicted[:5],end='\n \n')
  error=y_test-predicted
  square_error=np.power(error,2)
  sum_square_error=np.sum(square_error)
  y_mean=np.mean(y_test)
  total_variance=np.sum((y_test-y_mean)**2)
  print ("R-square value is ")
  print (1-sum_square_error/total_variance,end='\n \n ')

// #for understanding R square error, refer this link 
// https://www.youtube.com/watch?v=YE7E27-FJ90
// # R square error is 1-(RSS)/(TSS) where RSS is summation of (y-y hat) square and TSS is summation of (y- y mean) square
// # The estimated or predicted values in a regression or other predictive model are termed the y-hat values. 
// #Q1 E PART DOUBT PENDING

// #Q2
// #as dataset is same as previous question, we don't need to load it again or do the prprocessing step
// #splitting data into training,validation and testing dataset
from sklearn.model_selection import train_test_split
x_temp,x_test,y_temp,y_test=train_test_split(X_ScaledOriginal,Y,test_size=0.3)
x_train,x_validation,y_train,y_validation=train_test_split(x_temp,y_temp,test_size=0.2)


// #taking different learning rates and computing the regression coefficients
def gradient_descent(lrate): 
  beta=np.zeros(6) // #initialising parametrs as 1, features are from 0 to 4 but as there is one constant beta naught in the multi variable regression, so number of gradients will be 6
  number_of_iterations=100
  n=x_train.shape[0]
  learning_rate=lrate
  for i in range(number_of_iterations):
    x0_gradient=0
    x1_gradient=0
    x2_gradient=0
    x3_gradient=0
    x4_gradient=0
    x5_gradient=0
    for j in range(len(x_train)):
      a=x_train[j,0]
      b=x_train[j,1]
      c=x_train[j,2]
      d=x_train[j,3]
      e=x_train[j,4]
      f=y_train[j]
      # this is the summation step
      x0_gradient+=(beta[0]+beta[1]*a+beta[2]*b+beta[3]*c+beta[4]*d+beta[5]*e-f) 
      x1_gradient+=((beta[0]+beta[1]*a+beta[2]*b+beta[3]*c+beta[4]*d+beta[5]*e-f)*a)
      x2_gradient+=((beta[0]+beta[1]*a+beta[2]*b+beta[3]*c+beta[4]*d+beta[5]*e-f)*b)
      x3_gradient+=((beta[0]+beta[1]*a+beta[2]*b+beta[3]*c+beta[4]*d+beta[5]*e-f)*c)
      x4_gradient+=((beta[0]+beta[1]*a+beta[2]*b+beta[3]*c+beta[4]*d+beta[5]*e-f)*d)
      x5_gradient+=((beta[0]+beta[1]*a+beta[2]*b+beta[3]*c+beta[4]*d+beta[5]*e-f)*e)
      #here beta values are getting updated
      beta[0]=beta[0]-lrate/n*x0_gradient
      beta[1]=beta[1]-lrate/n*x1_gradient
      beta[2]=beta[2]-lrate/n*x2_gradient
      beta[3]=beta[3]-lrate/n*x3_gradient
      beta[4]=beta[4]-lrate/n*x4_gradient
      beta[5]=beta[5]-lrate/n*x5_gradient
  print(beta)

gradient_descent(0.001)
gradient_descent(0.01)
gradient_descent(0.1)
gradient_descent(1)

// # For each set of regression coefficients, compute R2_score for validation and test set and find the best value of regression coefficients. 


// # Q3
import numpy as np
import pandas as pd
colnames=['symboling','normalized_losses','make','fuel_type','aspiration','num_doors','body_style','drive_wheels','engine_location','wheel_base','length','width','height','curb_weight','engine_type','num_cylinders','engine_size','fuel_system','bore','stroke','compression_ratio','horsepower','peak_rpm','city_mpg','highway_mpg','price']
df=pd.read_csv('https://raw.githubusercontent.com/girikgarg8/ML_and_Data_Science_Datasets/master/CarPricePrediction.csv',names=colnames)

df=df.replace('?',np.NaN)
df.head()

from sklearn.impute import SimpleImputer
df.dropna(subset=['price'],inplace=True) 
imputer = SimpleImputer(missing_values = np.nan, strategy ='most_frequent') 
// replace NAN with mode
imputer1=imputer.fit(df)
df=pd.DataFrame(imputer1.transform(df))     // #pd.DataFrame is required becaude by default imputer returns an ndarray
df.columns=colnames       // #specifying column names
print(df)


dict1={"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"zero":0,"twelve":12} #mapping the values
df['num_doors']=df['num_doors'].map(dict1)
df['num_cylinders']=df['num_cylinders'].map(dict1)


// #dummy encoding scheme
dummy_encoding_body_style=pd.get_dummies(df['body_style'],prefix="body_style",drop_first=True)
df=pd.concat([df,dummy_encoding_body_style],axis=1)
df.drop(['body_style'],axis=1,inplace=True)
dummy_encoding_drive_wheels=pd.get_dummies(df['drive_wheels'],prefix="drive_wheels",drop_first=True)
df.drop(['drive_wheels'],axis=1,inplace=True)
df=pd.concat([df,dummy_encoding_drive_wheels],axis=1)
print(df)

// #label encoding
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['make']=encoder.fit_transform(df['make'])
df['aspiration']=encoder.fit_transform(df['aspiration'])
df['engine_location']=encoder.fit_transform(df['engine_location'])
df['fuel_type']=encoder.fit_transform(df['fuel_type'])

df.head()

// #(iv) For fuel_system: replace values containing string pfi to 1 else all values to 0.
cond=(df['fuel_system'].str.contains('pfi')==True)
df.loc[cond,'fuel_system']=1
cond2=(df['fuel_system']).str.contains('pfi')==False
df.loc[cond2,'fuel_system']=0

df.head()

// #For engine_type: replace values containing string ohc to 1 else all values to 0.
cond3=(df['engine_type'].str.contains('ohc')==True)
cond4=((df['engine_type']).str.contains('ohc')==False)
df.loc[cond3,'engine_type']=1
df.loc[cond4,'engine_type']=0

df.head()

df.info()

// # splitting given dataframe into x and y data
x_data=df.loc[:,df.columns!='price']
y_data=df.loc[:,df.columns=='price']


x_data.info()
y_data.info()


//   #https://stackoverflow.com/questions/54426845/how-to-check-if-a-pandas-dataframe-contains-only-numeric-column-wise
for i in df.columns: // #converting data type to number if possible
  df[i]=pd.to_numeric(df[i], errors='ignore')
for i in x_data.columns:
  x_data[i]=pd.to_numeric(x_data[i],errors='ignore')
for i in y_data:
  y_data[i]=pd.to_numeric(y_data[i],errors='ignore')


//   #Scaling the input data 
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
features_to_scale=x_data.select_dtypes(np.number).columns 
# print(df.columns)
x_scaled=scaler.fit_transform(x_data[features_to_scale]) #specifying the columns to normalise
x_Scaled=pd.DataFrame(x_scaled,columns=features_to_scale)
x_data=x_Scaled #assigning the scaled values to x_data
print(x_data)


print(x_data)
print(y_data)

// #Train a linear regressor on 70% of data (using inbuilt linear regression function of Python) and test its performance on remaining 30% of data.
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.3,random_state=42)
lr=LinearRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test))

// #Dimensional Reduction using PCA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
pca=PCA(n_components=11)
x_pca=pca.fit_transform(x_data)
x_train_pca,x_test_pca,y_train_pca,y_test_pca=train_test_split(x_pca,y_data,test_size=0.3,random_state=42)
lr.fit(x_train_pca,y_train_pca)
print(lr.score(x_test_pca,y_test_pca))

// ========================================================================================


// ASSIGNMENT 4 

// question 1 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X=np.array([i*np.pi/180 for i in range (60,300,4)])
np.random.seed(10)
y=np.sin(X)+np.random.normal(0, 0.15, len(X))
df= pd.DataFrame(np.column_stack([X,y]),columns=['X','y'])
for i in range (2,16):
  colname='X_%d'%i
  df[colname]=df['X']**i


  X=df.drop(['y'],axis=1).values
Y=df.iloc[:,1].values
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_scaled=np.insert(X_scaled,0,values=1,axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
betas=[]
l_rate = [0.0001, 0.001, 0.01, 0.1, 1, 10]
r_para = [pow(10,-15), pow(10,-10), pow(10,-5), pow(10,-3), 0, 1, 10, 20]
for learning_rate in l_rate:
  for r_p in r_para:
    beta = np.zeros(X_train.shape[1])
    for j in range(X_train.shape[1]):
      parsum=0
      for i in range(X_train.shape[0]):
        sum=0
        sum+=beta[0]
        for k in range(X_train.shape[1]):
          if k==0:
            continue
          sum+=beta[k]*X_train[i][k]
        sum-=Y_train[i]
        sum*=X_train[i][j]
      parsum=sum
      one=(parsum*learning_rate)/X_train.shape[0]
      two=1-((learning_rate*r_p)/X_train.shape[0])
      beta[j]=(beta[j]*two)-one
    betas.append(beta)


r2 = []
from sklearn.metrics import r2_score
for beta in betas:
Y_pred_val = X_val.dot(beta)
r2.append(r2_score(Y_val, Y_pred_val))


betas=[]
l_rate=[0.0001,0.001,0.01,0.1,1,10]
r_para=[pow(10,-15),pow(10,-10),pow(10,-5),pow(10,-3),0,1,10,20]
for learning_rate in l_rate:
  for r_p in r_para:
    beta=np.zeros(X_train.shape[1])
    for j in range(X_train.shape[1]): 
      parsum=0
      for i in range(X_train.shape[0]):
        sum=0
        for k in range(X_train.shape[1]):
          sum+=beta[k]*X_train[i][k]
        sum-=Y_train[i]
        sum*=X_train[i][j]
      parsum=sum
      one=(parsum*learning_rate)/X_train.shape[0]
      two=1-((learning_rate*r_p)/X_train.shape[0])
      beta[j]=(beta[j]*two)-one
    betas.append(beta)


    max_index = r2.index(max(r2))
optimal_beta = betas[max_index]
Y_pred_final = X_test.dot(optimal_beta)
r2_final = r2_score(Y_test, Y_pred_final)
r2_final



// question 2 

// part a 
df=pd.read_csv('Hitters.csv')
df.info()

df.describe().T

df[df.isnull().any(axis=1)].head(3)

df.isnull().sum().sum()

df=df.copy()
df.corr()

df['Year_lab'] = pd.cut(x=df['Years'], bins=[0, 3, 6, 10, 15, 19, 24])
df.groupby(['League','Division', 'Year_lab']).agg({'Salary':'mean'})


df['Salary'] = df.groupby(['League', 'Division', 'Year_lab'])['Salary'].transform(lambda x: x.fillna(x.mean()))
df.head()

df.isnull().sum()

df.shape

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['League'] = le.fit_transform(df['League'])
df['Division'] = le.fit_transform(df['Division'])
df['NewLeague'] = le.fit_transform(df['NewLeague'])
df['Year_lab'] = le.fit_transform(df['Year_lab'])

df.info()

from sklearn import preprocessing
df_X= df.drop(["Salary","League","Division","NewLeague"], axis=1)
scaled_cols5=preprocessing.normalize(df_X)
scaled_cols=pd.DataFrame(scaled_cols5, columns=df_X.columns)
scaled_cols.head()

cat_df=pd.concat([df.loc[:,"League":"Division"],df.loc[:,"NewLeague":"Year_lab"]], axis=1)
cat_df.head()

df= pd.concat([scaled_cols,cat_df,df["Salary"]], axis=1)
print(df)
print(df.shape)

// part b 

from sklearn.model_selection import train_test_split
X=df.drop("Salary", axis=1)
y=df["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

// part c 
from sklearn import metrics
from sklearn.metrics import mean_squared_error

#Linear Regression
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()

model = linreg.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(metrics.r2_score(y_test, y_pred))

#Ridge Regression
from sklearn.linear_model import Ridge
ridreg = Ridge(alpha=0.5748, normalize=True)
model = ridreg.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(metrics.r2_score(y_test, y_pred))

#Lasso Regression
from sklearn.linear_model import Lasso
lasreg = Lasso(alpha=0.5748, normalize=True)
model = lasreg.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(metrics.r2_score(y_test, y_pred))


// question 3 
from sklearn.datasets import load_boston
boston_dataset = load_boston()
dataset = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)

dataset.head()

dataset['MEDV'] = boston_dataset.target
dataset.head()

X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 25)
from sklearn.model_selection import cross_val_score

#Ridge Regression Model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha=3.8, fit_intercept=True))
]

ridge_pipe = Pipeline(steps)
ridge_pipe.fit(X_train, y_train)

cv_ridge = cross_val_score(estimator = ridge_pipe, X = X_train, y = y_train.ravel(), cv = 10)
print('RidgeCV: ', cv_ridge.mean())

#Lasso Regression Model
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Lasso(alpha=0.012, fit_intercept=True, max_iter=3000))
]

lasso_pipe = Pipeline(steps)
lasso_pipe.fit(X_train, y_train)

cv_lasso = cross_val_score(estimator = lasso_pipe, X = X_train, y = y_train, cv = 10)
print('LassoCV: ', cv_lasso.mean())



// ASSIGNMENT 6 

// question 1 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

column = ['label','text']
df = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv', sep = '\t', names = column)

df.head()

df.info()

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13,
                                                    shuffle=True , stratify=y)
print(y.value_counts())

// The multinomial Naive Bayes classifier is used to classify discrete features like word counts.
// Here we use CountVectorizer to turn the collection of text SMS into numerical features. Default values are kept as they give good results.

pipe = Pipeline(steps=[('vectorize', CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b')),
                       ('classifier', MultinomialNB())])

pipe.fit(X_train, y_train)


y_predict = pipe.predict(X_test)

print(accuracy_score(y_test, y_predict))
print(classification_report(y_test, y_predict))

mat = confusion_matrix(y_test, y_predict)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=True, cmap='coolwarm', linewidths=5)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()


SMS = [["You just won 50000 dollars worth cash prizes"],
       ["You can redeem 5000 dollars in cash"],
       ["I'll come within 5 minutes to meet you"],
       ["You just won 50 dollars to play games"],
       ["How are you doing my friend?"],
       ["You just won 50 dollars to have sex"],
       ["Greg, can you call me back once you get this?"],
       ["You just won 50 dollars to buy food"],
       ["Winner! To claim your gift call 0908878877"],
       ["Attend this free COVID webinar today: Book your session now"],
       ["Your online account has been locked. Please verify payment information"]]

for sms in SMS:
  print(pipe.predict(sms), sms)





// question 2 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import itertools
from scipy.stats import norm
import scipy.stats
from sklearn.naive_bayes import GaussianNB

%matplotlib inline
sns.set()

#Load the data set
iris = sns.load_dataset("iris")
iris = iris.rename(index = str, columns = {'sepal_length':'1_sepal_length','sepal_width':'2_sepal_width', 'petal_length':'3_petal_length', 'petal_width':'4_petal_width'})

#Plot the scatter of sepal length vs sepal width
sns.FacetGrid(iris, hue="species", size=7) .map(plt.scatter,"1_sepal_length", "2_sepal_width", )  .add_legend()
plt.title('Scatter plot')
df1 = iris[["1_sepal_length", "2_sepal_width",'species']]

def predict_NB_gaussian_class(X,mu_list,std_list,pi_list): 
    #Returns the class for which the Gaussian Naive Bayes objective function has greatest value
    scores_list = []
    classes = len(mu_list)
    
    for p in range(classes):
        score = (norm.pdf(x = X[0], loc = mu_list[p][0][0], scale = std_list[p][0][0] )  
                * norm.pdf(x = X[1], loc = mu_list[p][0][1], scale = std_list[p][0][1] ) 
                * pi_list[p])
        scores_list.append(score)
             
    return np.argmax(scores_list)

def predict_Bayes_class(X,mu_list,sigma_list): 
    #Returns the predicted class from an optimal bayes classifier - distributions must be known
    scores_list = []
    classes = len(mu_list)
    
    for p in range(classes):
        score = scipy.stats.multivariate_normal.pdf(X, mean=mu_list[p], cov=sigma_list[p])
        scores_list.append(score)
             
    return np.argmax(scores_list)

    # Plotting decision boundaries
#Estimating the parameters
mu_list = np.split(df1.groupby('species').mean().values,[1,2])
std_list = np.split(df1.groupby('species').std().values,[1,2], axis = 0)
pi_list = df1.iloc[:,2].value_counts().values / len(df1)

# Our 2-dimensional distribution will be over variables X and Y
N = 100
X = np.linspace(4, 8, N)
Y = np.linspace(1.5, 5, N)
X, Y = np.meshgrid(X, Y)

#fig = plt.figure(figsize = (10,10))
#ax = fig.gca()
color_list = ['Blues','Greens','Reds']
my_norm = colors.Normalize(vmin=-1.,vmax=1.)

g = sns.FacetGrid(iris, hue="species", size=10, palette = 'colorblind') .map(plt.scatter, "1_sepal_length", "2_sepal_width",)  .add_legend()
my_ax = g.ax


#Computing the predicted class function for each value on the grid
zz = np.array(  [predict_NB_gaussian_class( np.array([xx,yy]).reshape(-1,1), mu_list, std_list, pi_list) 
                     for xx, yy in zip(np.ravel(X), np.ravel(Y)) ] )


#Reshaping the predicted class into the meshgrid shape
Z = zz.reshape(X.shape)


#Plot the filled and boundary contours
my_ax.contourf( X, Y, Z, 2, alpha = .1, colors = ('blue','green','red'))
my_ax.contour( X, Y, Z, 2, alpha = 1, colors = ('blue','green','red'))

# Addd axis and title
my_ax.set_xlabel('Sepal length')
my_ax.set_ylabel('Sepal width')
my_ax.set_title('Gaussian Naive Bayes decision boundaries')

plt.show()

from sklearn.naive_bayes import GaussianNB

#Setup X and y data
X_data = df1.iloc[:,0:2]
y_labels = df1.iloc[:,2].replace({'setosa':0,'versicolor':1,'virginica':2}).copy()

#Fit model
model_sk = GaussianNB(priors = None)
model_sk.fit(X_data,y_labels)


# Our 2-dimensional classifier will be over variables X and Y
N = 100
X = np.linspace(4, 8, N)
Y = np.linspace(1.5, 5, N)
X, Y = np.meshgrid(X, Y)

#fig = plt.figure(figsize = (10,10))
#ax = fig.gca()
color_list = ['Blues','Greens','Reds']
my_norm = colors.Normalize(vmin=-1.,vmax=1.)

g = sns.FacetGrid(iris, hue="species", size=10, palette = 'colorblind') .map(plt.scatter, "1_sepal_length", "2_sepal_width",)  .add_legend()
my_ax = g.ax


#Computing the predicted class function for each value on the grid
zz = np.array(  [model_sk.predict( [[xx,yy]])[0] for xx, yy in zip(np.ravel(X), np.ravel(Y)) ] )


#Reshaping the predicted class into the meshgrid shape
Z = zz.reshape(X.shape)


#Plot the filled and boundary contours
my_ax.contourf( X, Y, Z, 2, alpha = .1, colors = ('blue','green','red'))
my_ax.contour( X, Y, Z, 2, alpha = 1, colors = ('blue','green','red'))

# Addd axis and title
my_ax.set_xlabel('Sepal length')
my_ax.set_ylabel('Sepal width')
my_ax.set_title('Gaussian Naive Bayes decision boundaries')

plt.show()

// Accuracy for both the implementations
// #Numpy accuracy
y_pred = np.array(  [predict_NB_gaussian_class( np.array([xx,yy]).reshape(-1,1), mu_list, std_list, pi_list) 
                     for xx, yy in zip(np.ravel(X_data.values[:,0]), np.ravel(X_data.values[:,1])) ] )
display(np.mean(y_pred == y_labels))

// #Sklearn accuracy
display(model_sk.score(X_data,y_labels))



// question 3 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,plot_confusion_matrix

df=pd.read_csv('/content/gender_classification_v7.csv')

df.head()

df.info()

df['gender'].value_counts()

df['Gender']=df['gender'].map({'Male':1,'Female':0})

df=df.drop('gender', axis = 1)
df.head()

x = df.iloc[:,0:7]
y = df.iloc[:,7]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=100)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix

knn = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV
k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
  
# defining parameter range
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)
  
# fitting the model for grid search
grid_search=grid.fit(x_train, y_train)

print(grid_search.best_params_)

accuracy = grid_search.best_score_ *100
print("Accuracy of training with tuning is : {:.2f}%".format(accuracy) )

knn = KNeighborsClassifier(n_neighbors=19)

knn.fit(x, y)

y_test_hat=knn.predict(x_test) 
print(y_test_hat)

test_accuracy=accuracy_score(y_test,y_test_hat)*100

print("Accuracy of testing dataset with tuning is : {:.2f}%".format(test_accuracy) )





// ASSIGNMENT 7 

from sklearn import svm, datasets
import pandas as pd

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['flower'] = iris.target
#print(df)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(iris.data,iris.target, test_size=0.3)

model = svm.SVC(kernel='rbf',C=100, gamma='auto')
model.fit(x_train,y_train)
model.score(x_test,y_test)

from sklearn.model_selection import cross_val_score

cross_val_score(svm.SVC(kernel='rbf',C=10, gamma='auto'), iris.data,iris.target, cv=5)

cross_val_score(svm.SVC(kernel='linear',C=100, gamma='auto'), iris.data,iris.target, cv=5)

from sklearn.model_selection import GridSearchCV
obj = GridSearchCV(svm.SVC(gamma='auto'),
                   {
                       'C':[1,10,20],
                       'kernel':['linear','rbf']
                   }, cv = 5, return_train_score=False            
                  )
obj.fit(iris.data,iris.target)

obj.cv_results_

df1 = pd.DataFrame(obj.cv_results_)
df1[['param_C','param_kernel','mean_test_score']]


