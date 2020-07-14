#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing the dataset
df_exist=pd.read_excel('Project_dataset.xlsx',sheetname='Existing employees')
df_left=pd.read_excel('Project_dataset.xlsx',sheetname='Employees who have left')

#creating a new attribute, 'Attrition' for each dataframe which tells whether or not an employee has left company X
df_exist['Attrition']='No'
df_left['Attrition']='Yes'

#merging the two datafame into a single dataframe
df_all=pd.concat([df,df2])

#checking for missing data
df_all.isnull()

#converting the object types to category codes
df_all['Attrition']=df_all['Attrition'].astype('category').cat.codes
df_all['dept']=df_all['dept'].astype('category').cat.codes
df_all['salary']=df_all['salary'].astype('category').cat.codes

#cleaning out the unecessary attributes, which in this case is Emp ID
df_allc= df_all[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','dept','salary','Attrition']]

#visualizing the categorical values using pie charts
df_salary= df_allc.groupby('salary', axis=0).sum()
df_salary['salary']=df_allc['salary'].value_counts().plot.pie(figsize=(5,5), autopct='%1.1f%%',startangle=90,shadow=True)
plt.title('EMPLOYEE ATTRITION  BASED ON SALARY')
plt.axis('equal')
plt.show()
plt.savefig('salary.png')

df_time= df_allc.groupby('time_spend_company', axis=0).sum()
df_time['time']=df_allc['time_spend_company'].value_counts().plot.pie(figsize=(5,5), autopct='%1.1f%%',startangle=90,shadow=True)
plt.title('EMPLOYEE ATTRITION BASED ON TIME SPENT IN THE COMPANY')
plt.axis('equal')
plt.show()
plt.savefig('time_spend_company.png')

df_prom= df_allc.groupby('promotion_last_5years', axis=0).sum()
df_prom['promotion_last_5years']=df_allc['promotion_last_5years'].value_counts().plot.pie(figsize=(5,6), autopct='%1.1f%%',startangle=90,shadow=True)
plt.title('EMPLOYEE ATTRITION BASED ON PROMOTION')
plt.axis('equal')
plt.show()
plt.savefig('promotion_last_5years.png')

df_accident= df_allc.groupby('Work_accident', axis=0).sum()
df_accident['salary']=df_allc['Work_accident'].value_counts().plot.pie( figsize=(5,6), autopct='%1.1f%%',startangle=90,shadow=True)
plt.title('EMPLOYEE ATTRITION BASED ON WORK ACCIDENT')
plt.axis('equal')
plt.show()
plt.savefig('work_accident.png')

df_project= df_allc.groupby('number_project', axis=0).sum()
df_project['number_project']=df_allc['number_project'].value_counts().plot.pie(figsize=(5,6), autopct='%1.1f%%',startangle=90,shadow=True)
plt.title('EMPLOYEE ATTRITION BASED ON GIVEN NUMBER OF PROJECT')
plt.axis('equal')
plt.show()
plt.savefig('number_project.png')

df_department= df_allc.groupby('dept', axis=0).sum()
df_department['dept']=df_allc['dept'].value_counts().plot.pie(figsize=(5,6), autopct='%1.1f%%',startangle=90,shadow=True)
plt.title('EMPLOYEE ATTRITION BASED ON DEPARTMENT')
plt.axis('equal')
plt.show()
plt.savefig('dept.png')

#visualizing the non categorical variables using violin plot
#taking [11000:] as Attrition=Yes for employees that had left
#taking [:11000] as Attrition=No for existing employees
viol_satisfaction1=df_allc[['satisfaction_level']][11000:]
sns.violinplot(viol_satisfaction1, orient='h')
plt.title('EMPLOYEE ATTRITION BASED ON SATISFACTION LEVEL OF EMPLOYEES THAT HAD LEFT')
plt.show()
plt.savefig('satisfaction_level1.png')
viol_satisfaction2=df_allc[['satisfaction_level']][:11000]
sns.violinplot(viol_satisfaction2, orient='h')
plt.title('EMPLOYEE ATTRITION BASED ON SATISFACTION LEVEL OF EXISTING EMPLOYEES')
plt.show()
plt.savefig('satisfaction_level2.png')

viol_evaluation1=df_allc[['last_evaluation']][11000:]
sns.violinplot(viol_evaluation1, orient='h')
plt.title('EMPLOYEE ATTRITION BASED ON LAST EVALUATION OF EMPLOYEES THAT HAD LEFT')
plt.show()
plt.savefig('last_evaluation1.png')
viol_evaluation2=df_allc[['last_evaluation']][:11000]
sns.violinplot(viol_evaluation2, orient='h')
plt.title('EMPLOYEE ATTRITION BASED ON LAST EVALUATION OF EXISTING EMPLOYEES')
plt.show()
plt.savefig('last_evaluation2.png')


viol_monthly_hours1=df_allc[['average_montly_hours']][11000:]
sns.violinplot(viol_monthly_hours1, orient='h')
plt.title('EMPLOYEE ATTRITION BASED ON AVERAGE MONTHLY HOURS OF EMPLOYEES THAT HAD LEFT')
plt.show()
plt.savefig('monthly_hours1.png')
viol_monthly_hours2=df_allc[['average_montly_hours']][:11000]
sns.violinplot(viol_monthly_hours2, orient='h')
plt.title('EMPLOYEE ATTRITION BASED ON AVERAGE MONTHLY HOURS OF EXISTING EMPLOYEES')
plt.show()
plt.savefig('monthly_hours2.png')

#Building the model
#Separating features and target
X = df_allc.iloc[:,0:9].values
Y = df_allc.iloc[:,9].values

#Splitting the data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30,random_state=0)

#Fitting Random Forest Classification to Training set
from sklearn.ensemble import RandomForestClassifier 
RF = RandomForestClassifier()
RF.fit(X_train,Y_train)

#predicting the attrition value for the test data
prediction = RF.predict(X_test)
prediction

#evaluating the result
from sklearn.metrics import classification_report
print(classification_report(Y_test,prediction))
