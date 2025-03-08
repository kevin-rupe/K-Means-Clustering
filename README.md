# K-Means-Clustering
Project completed in pursuit of Master's of Science in Data Analytics.

## PART I: RESEARCH QUESTION

### PROPOSAL OF QUESTION

How effective can a K-Means model assist the hospital with patient readmissions by grouping patients into specific clusters for analysis?

### DEFINED GOAL

The primary goal of this analysis is to develop a model that accurately groups patients into clusters that helps the hospital identify patients who are at more risk for readmission.

## PART II: TECHNIQUE JUSTIFICATION

### EXPLANATION OF THE CLUSTERING TECHNIQUE

For this analysis, I chose to use the K-Means clustering method. K-Means clustering is an unsupervised learning algorithm that groups data points that are close to one another. (Banoula, 2024) Before using the K-Means clustering algorithm, the data set values should be scaled in order to provide the most accurate model. Once the data has been scaled, then I will choose a k-value based upon visual inspection of the plot. 

The algorithm will randomly assign “k” points from the dataset, known as centroids. The goal of the algorithm is to locate the center of each cluster. In order to do this, each point in the dataset is assigned to the closest centroid using Euclidean distance (Holbrook, n.d.). Once all of the data points are assigned to each “k” cluster, then the centroids are recalculated using the mean of all data points inside the cluster. The algorithm repeats this process until there is no significant changes in the centroid’s location (aka, “convergence”) (PulkitS, 2024).

Based upon initial visual inspection of the dataset, I expect the outcome will be that there are 2 groups, or clusters. One cluster is where patients had shorter initial hospital stays, and the other is where patients had longer initial hospital stays. 

### SUMMARY OF THE TECHNIQUE ASSUMPTION

There are several limitations based on certain assumptions that must be presumed to be true for the K-Means Clustering method to be accurate. One of these assumptions is that all of the clusters are the same size, and are roughly the same shape. In real world data, this may not always be true that each of your different clusters will be the same size. Therefore, given the data, this may not be the best method to choose to analyze the data. (Patel, 2023)

### PACKAGES OR LIBRARIES LIST

![IMG_1627](https://github.com/user-attachments/assets/c6a671b4-6421-43e4-b6ea-1fa2ba6c0657)

## PART III: DATA PREPARATION


### DATA PREPROCESSING

One important preprocessing goal is to scale the numeric values. The K-Means Clustering Technique only works with numerical variables, but these values must be scaled prior to use to gain the most effective results. Using Scikit-learn’s Standard Scaler on the chosen variables, this technique removes the mean of each variable which centers the data around zero. (Scikit-learn, 2007-2024). 

### DATA SET VARIABLES

![IMG_1628](https://github.com/user-attachments/assets/9c632d47-0789-44e5-a102-4fdcb7ff8873)

### STEPS FOR ANALYSIS

First, I need to import the Pandas library and load the dataset into Jupyter Notebook (Pandas, 2023).
```python
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

#import CSV file
df = pd.read_csv("C:/Users/e0145653/Documents/WGU/D212 - Data Mining II/medical_clean.csv")
```

Next, I will view the data to get a sense of what variables we have in our dataset by viewing the data frame’s first few rows. There appear to be some variables here that I will not need for this analysis; that are useless based on my research question. 
```python
#view dataset
df.head(3)
```
![IMG_1629](https://github.com/user-attachments/assets/f4582ff4-3945-485f-bdbf-590128d88f1e)

After evaluating all of the variables, I decide on dropping the variables that I feel are not significant to my research question. 
```python
#Removing columns not needed for this exercise
df.drop(['Customer_id', 'Interaction', 'UID', 'TimeZone', 'County', 'Zip', 'Lat', 'Lng', 
        'Area', 'Job', 'Item1', 'Item2', 'Item3', 'Item4','Item5','Item6', 'Item7', 'Item8'], axis=1, inplace=True)
```

Next, I want to visualize a few numeric variables that I want to use in my analysis. After I’ve looked over some of the graphs, I decide to use the Initial_Days and Age variables in my analysis. 
```python
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()

ax = sns.scatterplot(data = df,
                    x = 'Initial_days',
                    y = 'Age',
                    s = 35)
```
![IMG_1630](https://github.com/user-attachments/assets/b8aecac6-118b-4582-87d5-d0b54779c4b5)

Before running the K-Means Clustering algorithm, I scale the data using Scikit-learn’s Standard Scaler, fitting and transforming the data into a new DataFrame. 
```python
#Normalize data using z-scores with StandardScaler from sklearn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#fit & transform the data using the scaler
scaled_df = scaler.fit_transform(df[['Initial_days', 'Age']])

#create the scaled dataframe
scaled_df = pd.DataFrame(scaled_df, columns = ['Initial_days', 'Age'])

#Select variables of interest
clusterscaled = scaled_df[['Initial_days', 'Age']].describe().round(2)

#print the new scaled_df
print(clusterscaled)
```
Displaying the scaled DataFrame, you can see that all the values are now centered around zero. 
![IMG_1631](https://github.com/user-attachments/assets/92639c7e-fb6b-48a1-88b2-f2d58f2e52d7)

## PART IV: ANALYSIS

### OUTPUT AND INTERMEDIATE CALCULATIONS

Before running the final algorithm, I need to find the appropriate value of k. I create a formula that will use k values 2 through 11 in the KMeans algorithm and assign the resulting inertia values to a list.

```python
#Choose k-optimal values
wcss = []

for k in range (2,11):
    model = KMeans(n_clusters = k,
                  n_init = 50,
                  random_state = 122)
    model.fit(scaled_df)
    wcss.append(model.inertia_)

wcss_s = pd.Series(wcss, index = range(2,11))
```

I can then visualize these on a scatter plot and finding the correct k-value. Where the graph makes a steep elbow before the line plot begins to flatten out will be the most appropriate k-value to use in my analysis.

![IMG_1632](https://github.com/user-attachments/assets/f9829237-4493-4355-9405-c9c97d94b70f)



### CODE EXECUTION

Upon my visual inspection, I would have chosen 2 for the number of clusters, but after seeing the graph above, I now know the best number of clusters if 4. I now run the K-Means Clustering technique, and evaluate the results of the model. 
```python
#Apply the correct K-number to the model
fin_model = KMeans(n_clusters = 4,
                  n_init = 25,
                  random_state = 122)

fin_model.fit(scaled_df)
```
![IMG_1633](https://github.com/user-attachments/assets/274125d5-412b-4616-a6b3-cfddc40083a7)

Next, I locate the centroids of each cluster, and then display the scaled DataFrame scatter plot along with the centroids scatter plot overlaid. 
```python
#Create the Centroid DataFrame
centroid = pd.DataFrame(fin_model.cluster_centers_,
                        columns = ['Initial_days', 'Age'])

print(centroid)
```
![IMG_1636](https://github.com/user-attachments/assets/27ba145f-d9ed-4c72-bd2a-af3da87c05c4)

```python
#Visualize the Centroid of the clusters
plt.figure(figsize=(7,5))

ax = sns.scatterplot(data = scaled_df, x = 'Initial_days', y = 'Age',
                    hue = fin_model.labels_, palette = 'colorblind',
                    alpha = 0.9, s = 150, legend = True)

ax = sns.scatterplot(data = centroid, x = 'Initial_days', y = 'Age',
                    hue = centroid.index, palette = 'colorblind',
                    s = 450, marker = 'D', ec = 'black', legend = False)

for i in range(len(centroid)):
    plt.text(x = centroid.Initial_days[i], y = centroid.Age[i], s = i,
            horizontalalignment = 'center', verticalalignment = 'center',
            size = 15, weight = 'bold', color = 'white')
```
![IMG_1634](https://github.com/user-attachments/assets/9063b76f-aa8d-4073-894a-ed7de4be606f)


## PART V: DATA SUMMARY AND IMPLICATIONS


### QUALITY OF THE CLUSTERING TECHNIQUE

Evaluating the quality of the clusters, I can see that since there are 4 clusters that are almost perfectly distributed equally. 
```python
#Evaluate the model
labels = pd.Series(fin_model.labels_).value_counts()
print(labels)
```
![IMG_1637](https://github.com/user-attachments/assets/0681293c-6a08-4cb8-94a2-9273bdc848f4)

To be able to determine the quality of these clusters, we can look more closely at the inertia values. Inertia measures the sum of squared distances between each data point and the assigned centroid (Afrimi, 2023). As the value of k increases, inertia decreases. We can visualize the optimal inertia value and the corresponding k-value by plotting this in Seaborn. The elbow method is then used by visually inspecting the graph to determine the appropriate k-value where the graph takes a less sharp decline (i.e., it’s the point that looks like the elbow). 
![IMG_1635](https://github.com/user-attachments/assets/52bc9829-f746-4848-888c-bd769d64e8ae)

Therefore, we can see that measured with respect to the inertia values, the most appropriate k-value is indeed 4. 

### RESULTS AND IMPLICATIONS

The result of this analysis shows four distinct clusters. Cluster 0 contains 2,482 observations (24.8%) which represent older patients with lower initial hospital stays. Cluster 1 contains 2,558 observations (25.6%) which represents young patients with higher initial hospital stays. Cluster 2 contains 2,516 observations (25.1%) which represents young patients with lower initial hospital stays. Cluster 3 contains 2,444 observations (24.4%) older patients with higher initial hospital stays.  

This implies that there are then four unique groups of patients that the hospital can use to research further to gain more insight. For example, the hospital could compare and contrast related data in Cluster 0 and Cluster 3 to gain insight into older patients with lower hospital stays vs. higher hospital stays. They could evaluate the readmission rates in each cluster in conjunction with their own analysis which could help define certain areas of interest that could help with decreasing their readmissions. 

Similarly, they could compare and contrast Cluster 0 and Cluster 2 to gain insights into lower hospital stays of older patients vs. younger patients. Cluster 0 and Cluster 1 are polar opposites, Cluster 2 and Cluster 3 are also. There are actually distinct ways in which each cluster can be analyzed against each of the 3 opposing clusters. Useful insights can be gained by the hospital willing to research these clusters. 

### LIMITATION

One limitation in K-Means Clustering is that it can be tricky to choose the correct number of ‘k’ clusters (Dortmund, n.d.). Choosing different values of ‘k’ can lead to different sized clusters and have different results. While a little challenging, figuring this out in Python is not too difficult. Visualizing the inertia measurements on a scatterplot with different k values, you can easily deduce the most appropriate ‘k’ value by choosing the value where the graph begins to “elbow” less steeply. 

### COURSE OF ACTION

My recommendation for the hospital would be to take the patient data from each combination of clusters, and research the consistent medical factors present, and those not present, that could then be used in marketing ads to educate the communities on how to lower their risk of being readmitted the hospital. I would also recommend they look at various demographic and socioeconomic statistics of the patients in each cluster to locate any patterns that might explain the risk of readmission. 

Educating the hospital staff on certain high-risk situations might be able to mitigate readmission rates during the patient’s initial stay. Also, educating patients holistically might encourage healthier living in key high-risk areas that could have a meaningful impact on lower hospital readmissions.


## PART VI: SUPPORTING DOCUMENTATION

#### WEB SOURCES 

Pandas (2023, June 28). Retrieved September 27, 2023, from https://pandas.pydata.org/docs/reference/index.html.

Waskom, M. (2012-2022). Seaborn Statistical Data Visualization. Retrieved September 27, 2023, from https://seaborn.pydata.org/index.html.

Scikit Learn (2007-2024). StandardScaler. Retrieved October 3, 2024, from https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html. 

Banoula, M. (2024). All About K-means Clustering Algorithm. Retrieved October 3, 2024, from https://www.simplilearn.com/tutorials/machine-learning-tutorial/k-means-clustering-algorithm#what_is_kmeans_clustering.

Patel, K. (2023). Understanding the Limitations of K-Means Clustering. Retrieved October 3, 2024, from https://medium.com/@kadambaripatel79/understanding-the-limitations-of-k-means-clustering-1fb5335f7859.  

Dortmund University. Limitations of k-Means Clustering. Retrieved on October 3, 2024, from https://dm.cs.tu-dortmund.de/en/mlbits/cluster-kmeans-limitations/. 

Holbrook, R. (n.d.). Clustering With K-Means. Untangle complex spatial relationships with cluster labels. Retrieved on October 7, 2024, from https://www.kaggle.com/code/ryanholbrook/clustering-with-k-means.

Afrimi, D. (May 1, 2023). Text Clustering using NLP techniques. Retrieved on October 7, 2024, from https://medium.com/@danielafrimi/text-clustering-using-nlp-techniques-c2e6b08b6e95.

PulkitS. (October 9, 2024). K-Means Clustering – Introduction. Retrieved on October 10, 2024, from https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/.

#### SOURCES

Bruce, P.A. (2020). Practical statistics for data scientists. 50+ essential concepts using r and python. O’Reilly Media, Incorporated. WGU Library.

Larose, C.D., Larose, D.T. (2019) Data science using Python and R. Chichester, NJ: Wiley Blackwell.


