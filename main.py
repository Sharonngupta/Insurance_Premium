import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


# instantiate model and Standard Scaler
LinearReg = LinearRegression()
scale = StandardScaler()

# set figure parameters
sns.set(rc={'figure.figsize':(15,15)})
sns.set(font_scale=2)



insurance_df = pd.read_csv('/content/insurance.csv') #loading the csv file as a dataframe
print(insurance_df) # take a look at the dataframe
print(insurance_df.dtypes) # there are floats and also integers and objects
print(insurance_df.isna().sum()) #there are no missing data

# Dependent variables is charges (continuous)
sns.histplot(x='charges', data= insurance_df) #trying to see the distribution of the dependent variable
plt.title('Distribution of charges')

#Exploratory Data Analysis
sns.boxplot(x='age', y='charges',  data=insurance_df) #Age does increase charges but its not the only factor given the existence of outliers across different ages
plt.xticks(rotation=90)
plt.title('Charges vs Age')


insurance_cat = pd.get_dummies(columns=['sex','smoker','region'], drop_first=True, data= insurance_df) #creating dummy variables for the discrete X variables. Drop first to ensure no multi collinearity
print(insurance_cat.head(5)) #print out dataframe to see if the changes is what we want

insurance_cat.corr() #create a large correlation table to see if there is any correlation between variables

insurance_cat.value_counts()

sns.heatmap(insurance_cat.corr()[['charges']], annot=True, cmap='coolwarm') #drill down to the correlation to charges, the dependent variable and use seaborn for visualization
plt.title('Correlation of Independent Variables to Hospital Charges')



#Base line 

Y_baseline = insurance_cat['charges'] #baseline y variable
X_col = [x for x in insurance_cat if x != 'charges'] #using list comprehension to get a list of x variables
X_baseline = insurance_cat[X_col] #creating the dataframe for x variables
X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(X_baseline,Y_baseline, test_size=0.33, random_state=42) #Train test split the baseline dataset

X_train_base_sc = scale.fit_transform(X_train_base) #scaling the train data
X_test_base_sc = scale.transform(X_test_base) #scaling the test data
LinearReg.fit(X_train_base_sc,y_train_base) #fitting the train data to the linear regressor


# Scatterplot of Predicted vs actual values'

y_pred_base = LinearReg.predict(X_test_base_sc) #predicting y using X_test and model parameters
sns.scatterplot(y= y_pred_base, x = y_test_base) #scatterplot for prediction vs actual
plt.title('Prediction vs Actual') #set title
plt.show() #show plot


#creating a table to see results
model_result = pd.DataFrame({'Model':'Baseline', 'Train Score': [LinearReg.score(X_train_base_sc,y_train_base)],'Test Score': [LinearReg.score(X_test_base_sc,y_test_base)], 'CV Score': [cross_val_score(LinearReg, X_train_base_sc,y_train_base).mean()]})
model_result.set_index('Model').T #transpose the dataframe and set index name to 'model'
model_result = np.round(model_result, decimals=2) #set to 2 decimal for better viewing

print(model_result.to_markdown()) #create border around the table for easy viewing


#Generating the feature importance plot

coefs = pd.DataFrame(zip(X_baseline.columns, LinearReg.coef_), columns=['Features','Co-effs']) #putting coefs in a dataframe
coefs_sorted =coefs.sort_values(by='Co-effs',ascending=True) #sort the coefficients
print(coefs_sorted) #print table
sns.barplot(data=coefs_sorted, x="Features", y="Co-effs") #plot the sorted dataframe
plt.xticks(rotation=90) #rotate the ticks
plt.title('Baseline Model: Feature Importance') #set a title




#ML model with feature engineering

poly = PolynomialFeatures(include_bias=False) #instantiate the polynomial function
X_poly = poly.fit_transform(X_baseline) #fit and transform the original X baseline dataframe
X_poly = pd.DataFrame(X_poly,columns=poly.get_feature_names(X_baseline.columns)) #create a new dataframe with feature names from the original X_baseline
print(X_poly.head(5)) #print to see if the dataframe is what we expected.

corr_df = pd.concat([Y_baseline, X_poly], axis=1) #reattached the y_baseline so that we can do a correlation plot
sns.heatmap(corr_df.corr()[['charges']][corr_df.corr()[['charges']]>0.7], annot=True, cmap='coolwarm') #extended correlation plot using all the engineered features

corr_df['age_quantile'] = pd.qcut(corr_df['age'], 4, #cutting the bmi into 4 quantiles for visualization purposes
                               labels = [1,2,3,4])
sns.boxplot(x='age_quantile', y='charges', data= corr_df).set_title('Effect of Age on Charges') #creating a boxplot

print(corr_df['age'].describe()) #using describe to know where the quantile cut offs will be. 

corr_df['age_quantile'] = pd.qcut(corr_df['age'], 4, #cutting the bmi into 4 quantiles for visualization purposes
                               labels = [1,2,3,4])
sns.boxplot(x='age_quantile', y='charges', hue='smoker_yes', data= corr_df).set_title('Effect of Age and Smoker on Charges') #creating a boxplot

print(corr_df['age'].describe()) #using describe to know where the quantile cut offs will be. 

# Same as above
print(corr_df['bmi'].describe())
corr_df['bmi_quantile'] = pd.qcut(corr_df['bmi'], 4,
                               labels = ['1','2','3','4'])
sns.boxplot(x='bmi_quantile', y='charges', hue='smoker_yes', data= corr_df)
plt.title('Effects of Smoking and BMI on Charges')

corr_df['bmi_quantile'] = pd.qcut(corr_df['bmi'], 4, 
                               labels = ['1','2','3','4']) 
corr_df['age_quantile'] = pd.qcut(corr_df['age'], 4,  
                               labels = ['1','2','3','4']) 
sns.boxplot(x='age_quantile', y='charges', hue='bmi_quantile', data= corr_df) 
plt.title('Effects of Age and BMI on Charges')

X_train, X_test, y_train, y_test = train_test_split(X_poly, Y_baseline, test_size = 0.33, random_state=42) #train test split for ML model with full features

X_train_sc = scale.fit_transform(X_train)
X_test_sc = scale.transform(X_test)

LinearReg.fit(X_train_sc, y_train) #fitting model to train data
r2_train = LinearReg.score(X_train_sc, y_train) # score for train data
r2_test = LinearReg.score(X_test_sc, y_test) #score for test data. 

y_pred = LinearReg.predict(X_test_sc) #predicted values

sns.scatterplot(y= y_pred, x = y_test) #scatter of predicted vs actual
plt.title('Prediction vs Actual')
plt.show()

#Creating table with all the relevant scores

model_result = pd.DataFrame({'Model':'ML Model', 'Train Score': [LinearReg.score(X_train_sc,y_train)],'Test Score': [LinearReg.score(X_test_sc,y_test)], 'CV Score': [cross_val_score(LinearReg, X_train_sc,y_train).mean()]})
model_result.set_index('Model').T
model_result = np.round(model_result, decimals=2)
print(model_result.columns)
print(model_result.to_markdown())

coefs = pd.DataFrame(zip(X_poly.columns, LinearReg.coef_), columns=['Features','Co-effs'])
coefs_sorted =coefs.sort_values(by='Co-effs',ascending=True)
print(coefs_sorted.shape)
print(coefs_sorted.to_markdown())

#creating a barplot with sorted feature coefficients.
sns.barplot(data=coefs_sorted, x="Features", y="Co-effs")
plt.xticks(rotation=90)
plt.title('ML Model: Feature Importance')


l_alphas = np.logspace(-3,1,100)  #creating a set of alphas to test
 
# Cross-validate over our list of Lasso alphas.

lasso_cv = LassoCV (alphas=l_alphas, cv = 5, max_iter=4000) #testing for the best alpha
 
# Fit model using best Lasso alpha! 
X_train_sc = scale.fit_transform(X_train) #scale the train data
X_test_sc = scale.transform(X_test) #scale the test data
lasso_cv.fit(X_train_sc,y_train) #fitting the model
lasso_cv.alpha_

print(lasso_cv.score(X_train_sc, y_train)) #train score
print(lasso_cv.score(X_test_sc, y_test)) #test score


coefs = pd.DataFrame(zip(X_poly.columns, lasso_cv.coef_), columns=['Features','Co-effs'])
coefs_sorted =coefs.sort_values(by='Co-effs',ascending=True)
print(coefs_sorted)
sns.barplot(data=coefs_sorted, x="Features", y="Co-effs") #getting feature importance plot
plt.xticks(rotation=90)
plt.title('ML Model: Feature Importance')
plt.show()

lasso_label = coefs[coefs['Co-effs'] != 0] #getting a dataframe of all the features which are non zeros
print(lasso_label)
print(lasso_label.shape) #finding out how many factors are left

y_pred = lasso_cv.predict(X_test_sc)
sns.scatterplot(y= y_pred, x = y_test) #creating a predicted vs actual plot
sns.scatterplot(y= y_pred_base, x=y_test_base)
plt.title('Predicted Vs Actual')
plt.show()
plt.tight_layout()

# Consolidating all results in a table for viewing
model_result = pd.DataFrame({'Model':'ML Model (Regularised)', 'Train Score': [lasso_cv.score(X_train_sc,y_train)],'Test Score': [lasso_cv.score(X_test_sc,y_test)]})
model_result.set_index('Model').T
model_result = np.round(model_result, decimals=2)
print(model_result.columns)
print(model_result.to_markdown())


# getting model predictions for a set of hypothetical profiles. 

age = [30,50,30,30]
bmi = [20, 20,20,35]
children = [0,0,0,0]
sex_male = [1,1,1,1]
smoker_yes = [1,1,0,1]
region_northwest = [1,1,1,1]
region_southeast = [0,0,0,0]
region_southwest = [0,0,0,0]

test = pd.DataFrame(zip(age,bmi,children, sex_male, smoker_yes, region_northwest, region_southeast, region_southwest), columns = X_baseline.columns)
test_poly = poly.transform(test) #using the same poly features to transform the hypothetical data
test_sc = scale.transform(test_poly) #using the same standard scaler to transform the hypothetical data
y_pred = lasso_cv.predict(test_sc) #using lasso parameters to predict charges
print(y_pred)

def profit_bootstrap(x,y,z): #create a function to do bootstrapping 
  Test_Data = pd.concat([X_test, y_test], axis = 1) #combine the y test and x test data back
  Test_Quantile = Test_Data.quantile([0, .25, .5, .75,1], axis = 0) #cut the data into quartiles
  print(Test_Quantile[x]) #print out the values for the different quartiles
  profit = pd.DataFrame() #create empty dataframe
  for i in range(0,4):#iterate through the different quartiles
    test_old = Test_Data.loc[(Test_Data[x]>=Test_Quantile[x].iloc[i]) & (Test_Data[x]<=Test_Quantile[x].iloc[i+1])& (Test_Data[y]==z)] #filtering rules
    print(str(i+1)+' Quartile', test_old.shape) # check if the n observations in the different quartiles are roughly equal
    X_test_old = test_old.iloc[:,:-1] # resplit dataset into X_test and y_test
    y_test_old = test_old.iloc[:,-1] # resplit dataset into X_test and y_test
    y_pred_old = lasso_cv.predict(scale.transform(X_test_old)) #use model parameters to scale and predict expected premiums
    error_old = y_pred_old - y_test_old #error can also be seen as the profit if actual is the charge paid by the insurer and premiums is paid to the insurance
    bootstrap = [] #empty list
    for a in range(1000): #iterate through 1000 times
      a= (resample(error_old, n_samples=500, replace=True)).mean() #sample 500 times with replacement
      bootstrap.append(a) #append results to the empty list
    Boot_df = pd.DataFrame(bootstrap, columns = [i]) #convert into dataframe
    profit = pd.concat([profit,Boot_df], axis=1) #concat all the bootstraps data into the dataframe
  print(profit.describe()) #statistics for the different bootstraps
  sns.displot(profit,fill=True, palette=sns.color_palette('bright')[:4], height=5, aspect=5, bins=50) #distribution plot
  plt.legend(title= x +' Quartile', loc='upper left', labels=['1st Quartile', '2nd Quartile', '3rd Quartile', '4th Quartile']) #customise legend and title
  plt.title('Bootstrap sampling distribution of profits by ' + x +' Quartile')#customise legend and title
  plt.xlabel('Profits')

profit_bootstrap('age','smoker_yes',0)
profit_bootstrap('bmi','smoker_yes',0)



# need to simply this into a function. 
Test_Data = pd.concat([X_test, y_test], axis = 1)
cost_cols = [col for col in Test_Data.columns if 'smoker' in col]
Test_Smoker = Test_Data[Test_Data['smoker_yes']==1]
Hypothetical_data = Test_Data[Test_Data['smoker_yes']==1]
Hypothetical_data[cost_cols] = 0
print(type(Hypothetical_data))
X_test_original = Test_Smoker.iloc[:,:-1]

Hypothetical_x = Hypothetical_data.iloc[:,:-1]

y_test_original = Test_Smoker.iloc[:,-1]
y_pred_original = lasso_cv.predict(scale.transform(X_test_original))
y_pred_hypothetical = lasso_cv.predict(scale.transform(Hypothetical_x))
error_original = y_pred_original - y_test_original
error_hypothetical = y_pred_original - y_pred_hypothetical
original = []
hypothetical =[]
cost = pd.DataFrame()
for a in range(1000):
    a= (resample(error_original, n_samples=500, replace=True)).mean()
    original.append(a)
    original_df = pd.DataFrame(original,columns=['Original'])
for b in range(1000):
    b= (resample(error_hypothetical, n_samples=500, replace=True)).mean()
    hypothetical.append(b)
    hypothetical_df = pd.DataFrame(hypothetical,columns=['Hypothetical'])
cost = pd.concat([cost,original_df,hypothetical_df], axis=1)
print(cost.describe())

sns.displot(cost,fill=True, palette=sns.color_palette('bright')[:2], height=5, aspect=5, bins=100)
plt.title('Bootstrap sampling distribution of profits between Smokers and Hypothetical Population')
plt.legend(labels=['Hypothetical','Original'])
plt.xlabel('Profits')


Test_BMI = pd.concat([X_test, y_test], axis = 1)
BMI_cols = [col for col in Test_Data.columns if 'bmi' in col]

Hypothetical_BMI = Test_BMI.copy()
Hypothetical_BMI[BMI_cols] = Hypothetical_BMI[BMI_cols] * 0.95

X_test_original = Test_BMI.iloc[:,:-1]

Hypothetical_x = Hypothetical_BMI.iloc[:,:-1]

y_test_original = Test_BMI.iloc[:,-1]
y_pred_original = lasso_cv.predict(scale.transform(X_test_original))
y_pred_hypothetical = lasso_cv.predict(scale.transform(Hypothetical_x))
error_original = y_pred_original - y_test_original
error_hypothetical = y_pred_original - y_pred_hypothetical

original = []
hypothetical =[]
cost2 = pd.DataFrame()
for a in range(1000):
    a= (resample(error_original, n_samples=500, replace=True)).mean()
    original.append(a)
    original_df = pd.DataFrame(original,columns=['Original'])
for b in range(1000):
    b= (resample(error_hypothetical, n_samples=500, replace=True)).mean()
    hypothetical.append(b)
    hypothetical_df = pd.DataFrame(hypothetical,columns=['Hypothetical'])
cost2 = pd.concat([cost2,original_df,hypothetical_df], axis=1)
print(cost2.describe())

sns.displot(cost2,fill=True, palette=sns.color_palette('bright')[:2], height=10, aspect=2, bins=50)
plt.title('Bootstrap sampling distribution of profits with slight variation in BMI')
plt.legend(labels=['Hypothetical','Original'])
plt.xlabel('Profits')

# Appendix

def creating_plot(X,y,z,df): #function to create box plot
  X_extended = X + '_quantile'
  df[X_extended] = pd.qcut(df[X],4,labels=False)
  sns.boxplot(x=X_extended, y= y, hue=  z, data = df ).set_title('Effect of '+ X + ' And ' + z + ' on ' +y)
  print(df[X].describe())

creating_plot('age', 'charges', 'smoker_yes', corr_df)



#Creating a function to reduce the code needed for two fits. 
def fitting_model(model, X, y):
  Standard = StandardScaler()
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)
  X_train_sc = Standard.fit_transform(X_train)
  X_test_sc = Standard.transform(X_test)
  model.fit(X_train, y_train)
  r2_train = model.score(X_train, y_train)
  r2_test = model.score(X_test, y_test)
  print(r2_train, r2_test)


fitting_model(LinearRegression(), X_original, Y_original) # original model
fitting_model(LinearRegression(), X_poly, Y) #poly fit model
