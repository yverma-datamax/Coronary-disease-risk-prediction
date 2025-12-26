# Coronary-disease-risk-prediction
Predictive analytics project using the UCI Heart Disease dataset to identify key lifestyle and demographic risk factors for cardiovascular disease. Implements data preprocessing, feature selection, and binary classification models to support data-driven health insights.

Business
Business Definition: Understanding Heart Disease Risk Factors
HeartPal is a health application that leverages wearable technology and predictive analytics to help users monitor cardiovascular health. The primary business question is: Which lifestyle and demographic factors most strongly influence the likelihood of heart disease? By analyzing subsets of the dataset, HeartPal aims to identify key predictors beyond basic medical indicators, such as the impact of exercise frequency, cholesterol levels, and age-related trends.
To address this, HeartPal will filter the dataset to focus on specific user demographics, such as age groups or individuals with pre-existing conditions, and assess how different variables contribute to heart disease risk. This insight will enable the app to provide targeted health recommendations, refine its risk assessment algorithm, and support partnerships with healthcare providers. Ultimately, this project will help HeartPal enhance its predictive capabilities, offering users more personalized and actionable health insights. 

Dataset- Heart disease 
The dataset is a subset of the Cleveland heart disease database, commonly used for, machine learning experiments. It contains attributes collected from patients and aims to predict the presence of coronary artery heart disease.
The dataset contains 14 Key attributes of patient demographics, clinical test results, and symptoms. The target variable, “num”, indicates the presence of heart disease on a scale of 0 to 4. However, our analysis will simplify this to binary classification: 0 = no heart disease, 1,2,3,4 = Presence of heart disease. 
Key attributes used in the model
The following features will be used in building the predictive model:
age: Age of the patient in years
sex: Biological sex (0 = female, 1 = male)
cp (chest pain type): Categorized as 1–4, indicating different types of chest pain
treetops (resting blood pressure): Blood pressure measured in mm Hg
chol (serum cholesterol): Cholesterol level in mg/dL
fbs (fasting blood sugar): Whether fasting blood sugar > 120 mg/dL (1 = true, 0 = false)
restecg (resting electrocardiographic results): Categorized as 0–2, representing ECG results
thalach (maximum heart rate achieved): Measured in beats per minute
exang (exercise-induced angina): 1 = yes, 0 = no
oldpeak: ST depression induced by exercise relative to rest
slope: Slope of the peak exercise ST segment (values 0–2)
ca (number of major vessels colored by fluoroscopy): Ranges from 0 to 3
thal: A categorical variable related to the thalassemia test (values 1–3)
Target Variable
num: The presence of heart disease  0 = no heart disease, 1,2,3,4 = Presence of heart disease

Importing and Understanding The Data and Dependencies 
Before conducting any data analysis, it is crucial to understand the dataset and ensure that the necessary dependencies are installed and accessible. This study utilizes the Heart Disease Dataset from the UCI Machine Learning Repository, which contains 303 instances and 13 primary features related to patient demographics, medical history, and diagnostic indicators. The dataset is primarily used for classification tasks, aiming to predict the presence of heart disease based on medical attributes.
To facilitate data analysis, we confirm that the required libraries, including ucimlrepo, pandas, and numpy, are installed in the environment. The installation logs indicate that all dependencies are satisfied, ensuring a smooth workflow for further processing.
The 1989 dataset includes numerical and categorical variables, with some missing values denoted as "NaN." It has been widely used in medical and machine-learning research.
In the next step, we will load the data and conduct a preliminary exploration to assess its structure, identify missing values, and prepare it for further analysis.
This step includes importing data to our notebook, making data frames, defining the name, naming the features and target, and downloading all the required libraries for our next steps. (fig-1)
(figure-1 importing required libraries and dataset)

Next, we printed the headers of the columns to understand their structure, size, and the types of data we would be dealing with. We also checked for missing values and found that two of our columns (ca and thal) have some missing values. Depending on the number of missing values, we will decide how to tackle the problem. 

(figure-2  printing headers)
Checking for Missing Values and Data Cleaning 
After loading the data and getting an idea of the dataset, we initially assessed missing values. (fig- 3) The dataset contained missing values in two columns:

ca (number of major vessels colored by fluoroscopy): 4 missing values (1.32% of the dataset)
thal (thalassemia test result): 2 missing values (0.66% of the dataset)

(figure-3  checking for missing values)

To tackle the missing values, we adopted the following approach 
Given the small percentage of missing values and the numerical nature of ca and thal, mean imputation was chosen to fill in the missing values. The rationale behind this decision was to maintain the data distribution without introducing potential biases that could occur with methods such as mode or median imputation.
ca column: Missing values were replaced using the mean of the non-missing ca values. (fig- 4)

(figure-4  filling missing ‘ca’ values with mean)

thal column: The same imputation approach was applied to thal using its mean value. (fig- 5)

(figure-5  filling missing ‘thal’ values with mean)

After this step, the percentage of missing values in ca and thal dropped to 0%.(fig- 6)

(figure-6  checking the percentage of missing values)
Data Exploration and Visualization for understanding 
After addressing missing values, the next crucial step was visualizing the dataset to gain a deeper understanding of its distribution and relationships between variables. This process involved generating histograms (fig- 7), a correlation heatmap, and summary statistics to explore the dataset more effectively.
Histogram Analysis 
To understand the distribution of each feature, we created a series of histograms, as shown in the figure. These histograms provided insight into each variable's skewness, central tendency, and presence of outliners. 

(figure-7  histogram for distribution of each column)
Correlation Heatmap Analysis
A correlation matrix heatmap (fig- 8) was generated to assess relationships between different variables. The key insights from the heatmap included:
Strong Correlations: Features such as ca (number of major vessels) and thal showed a relatively high correlation with num (the target variable), suggesting their importance in predictive modeling.
Negative Correlations: thalach (maximum heart rate achieved) exhibited a negative correlation with num, implying that higher heart rates might be associated with lower chances of heart disease.
Low Correlation Features: Some variables, such as fasting blood sugar (fbs), displayed a very low correlation with num, indicating that they might have limited predictive value.


(figure-8 heatmap for understanding correlation)
Circling back to the question
Now that we have the analytical knowledge about our clean data, we will break down the question to understand what is being asked and how to extract those things from our data. 
Question-  Which lifestyle and demographic factors most strongly influence the likelihood of heart disease?
The dataset includes various predictors such as age, sex, blood pressure, cholesterol, chest pain type, and indicators of heart stress such as exercise-induced angina and ST depression (oldpeak). On-hot encoding was performed to convert categorical variables into dummy variables, ensuring compatibility with logistic regression. 
Logistic regression was chosen because it is interpretable and effective in binary classification problems. The dataset was split into training (80%) and testing (20%) sets and a logistic regression model was trained using num (presence of heart disease) as the target variable.
Understanding Model Using the Confusion Matrix
To assess the logistic regression model's predictive performance, I generated a confusion matrix (fig- 9) that evaluates how well the model distinguishes between individuals with and without heart disease.
Interpretation of Results
The model correctly identified 24 cases of heart disease but misclassified 8 cases, meaning these individuals were falsely reassured of being disease-free.
The model correctly identified 22 healthy individuals but falsely flagged 7 cases as having heart disease.
The false negatives (8) are particularly concerning in a medical setting, as undiagnosed heart disease could lead to serious health risks.
The relatively low false-positive rate (7 cases) suggests that unnecessary alarms are minimized, which is important for reducing unnecessary medical interventions.

(figure-9 confusion matrix)

Interpretation of the results 
The bar chart visualizes the importance of the feature (Fig- 10) derived from the logistic regression model in predicting heart disease. The coefficients represent the strength and direction of each factor’s influence on heart disease likelihood.
Key Insights:
Chest Pain Type (cp_4, cp_2, cp_3):
The strongest predictor is cp_4 (atypical angina), followed by cp_2 (non-anginal pain). A higher coefficient indicates a stronger positive association with heart disease.
Conversely, cp_3 (asymptomatic pain) has a negative coefficient, suggesting a lower likelihood of heart disease. This is because non-anginal pain typically originates from musculoskeletal issues, acid reflux, or anxiety, rather than coronary artery disease.


Sex:
A high coefficient for sex suggests that being male is a significant risk factor for heart disease, aligning with medical studies that show men generally have a higher risk.


Exercise-Induced Angina (exang):
A positive coefficient for exang suggests that experiencing chest pain during exercise is strongly linked to heart disease.


Oldpeak (ST Depression Induced by Exercise):
This feature measures changes in heart activity during stress. A higher oldpeak value corresponds to a greater likelihood of heart disease.


Age and Other Factors:
Age shows a mild positive impact, meaning that older individuals are slightly more prone to heart disease.
Cholesterol (chol), Fasting Blood Sugar (fbs), and Resting Blood Pressure (trestbps) have very small coefficients, implying a weaker influence compared to other factors. This is because cholesterol, blood sugar, and blood pressure directly do not lead to heart disease, they lead to ‘ca’ which is one of the most strongest factors of having heart disease. 

(figure-10 features importance and direction)



Business Interpretation:
The results indicate that chest pain type, sex, and exercise-induced angina are the most influential lifestyle and demographic factors in predicting heart disease. From a healthcare perspective, here are some actionable recommendations- 
Targeted Screening Programs for Asymptomatic Patients (cp_4)
Since asymptomatic individuals have the highest risk, healthcare providers should implement proactive heart disease screening for patients with risk factors, even if they report no chest pain.
Employers and insurance companies can encourage annual cardiovascular health screenings for individuals over a certain age threshold.


Early Intervention for Patients with Atypical Angina (cp_2)
Patients experiencing atypical chest pain should receive priority referrals for diagnostic testing (e.g., stress tests, echocardiograms).
Primary care doctors should be trained to recognize atypical symptoms as potential early indicators of heart disease.


Reassessing Non-Anginal Chest Pain Cases (cp_3)
Patients with chest pain that is not caused by heart issues may not need extensive heart tests. Instead, they might benefit from checking for stomach or muscle problems. 
By reducing unnecessary heart tests for these patients, healthcare costs can go down while also making diagnosis more efficient.


Gender-Specific Heart Disease Prevention Strategies
There is a strong link between being male and having a higher risk of heart disease. Because of this, healthcare campaigns should teach men about early signs of heart problems and ways to live healthier. 
Workplace wellness programs should be designed to promote heart-healthy habits, especially for those at higher risk.


Exercise-Based Cardiac Risk Monitoring
Since exercise-induced angina (exang) is a key predictor, individuals who report chest discomfort during physical activity should be directed to stress testing and lifestyle modification programs.
Gyms, fitness programs, and personal trainers should be educated on recognizing exercise-related heart risk indicators.


Cholesterol, blood sugar level, and blood pressure
Although according to the model chol, fbs, and trestbps have very little effect on heart disease, they lead to several major blood vessels that are identified as having a blockage or narrowing (ca) which further leads to heart disease. 
