
#  Importing Libraries 
#Note MinMaxScaller is not used in this exercise, but usually used to scale the large values to smaller or smaller values to larger

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
%matplotlib inline


# Importing  dataset
df=pd.read_csv("C:\\Users\\ARWINDD\\OneDrive\\Desktop\\PandasDemo\\Iris_with_color.csv")

# Top five rows
df.head()


# ScatterPlot to view the datapoints behaviour
plt.scatter(df['PetalLengthCm'],df['PetalWidthCm'])
plt.show()

# Model of K-MEANS
km=KMeans(n_clusters=3)


#Clustering the data points
y_predicted=km.fit_predict(df[['PetalLengthCm','PetalWidthCm']])
y_predicted

# Appending a column with clustered groups
df['cluster']=y_predicted
df.head()

## Plotting without Centroids of the clusters
df1=df[df['cluster']==0]
df2=df[df['cluster']==1]
df3=df[df['cluster']==2]
plt.scatter(df1.PetalLengthCm,df1.PetalWidthCm,color='red')
plt.scatter(df2.PetalLengthCm,df2.PetalWidthCm,color='purple')
plt.scatter(df3.PetalLengthCm,df3.PetalWidthCm,color='green')
plt.legend()
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.show()

# Finding centroids 
km.cluster_centers_

# Plotting along with centroids
df1=df[df['cluster']==0]
df2=df[df['cluster']==1]
df3=df[df['cluster']==2]
plt.scatter(df1.PetalLengthCm,df1.PetalWidthCm,color='red')
plt.scatter(df2.PetalLengthCm,df2.PetalWidthCm,color='purple')
plt.scatter(df3.PetalLengthCm,df3.PetalWidthCm,color='green')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='blue',marker='*',label='Centroid')
plt.legend()
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.show()




