from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
import numpy as np  # linear alg
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import random 
import keras
import pickle
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda

from keras import backend as K
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, silhouette_score, silhouette_samples, homogeneity_completeness_v_measure
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, MinMaxScaler
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, SpectralClustering, AgglomerativeClustering
#from s_dbw import S_Dbw, SD

random.seed(25)
run = 0
metric_names = ['Silhouette Coefficients ', 'Average Silhouette Metric ', 'Scaled Multiplied Silhouette Metric ']
name9 = 'test case {}/{} Correlation.xlsx'.format(run, metric_names[0])
writer1 = pd.ExcelWriter(name9)
name10 = 'test case {}/{} Correlation.xlsx'.format(run, metric_names[1])
writer2 = pd.ExcelWriter(name10)
name11 = 'test case {}/{} Correlation.xlsx'.format(run, metric_names[2])
writer3 = pd.ExcelWriter(name11)
writer_list = [writer1, writer2, writer3]

##### User defined functions #####
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
      uses y_true as Y and y_pred as the Euclidean distance between dissimilar points
    '''
    margin = 1
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true) 

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def cluster_centroid(embeddings, y_sc, numclusters):
    centroids = [] #center indice values
    for i in range(numclusters): #for each class...
        inds = np.where(y_sc==i)[0] #list of all images within the same proposed class
        inds_embed = embeddings[inds] #feature vectors of all these images
        inds_embed1 = np.array(inds_embed) #numpy array of this matrix
        column_mean = inds_embed1.mean(axis=0) #mean per dimension (x, y, z, etc.)
        centroids.append(column_mean) #append to the list
    return centroids

def wss(centroids, embeddings, y_sc, numclusters, n): #in 2d - total wss (not average)
    wss_percluster = [] # list of total distance per class
    closest_inds = []
    farthest_inds = []
    for i in range(numclusters):
        inds = np.where(y_sc==i)[0] #list of all images within the same proposed class
        inds_embed = embeddings[inds] #feature vectors of all these images
        inds_embed1 = np.array(inds_embed) #numpy array of this matrix
        after_subtraction = inds_embed1 - centroids[i] #subtracting the centroid indices from the image indices
        squared_matrix = np.square(after_subtraction) #square the distance values
        sum_rows = np.sum(squared_matrix, axis = 1) #add all fo the distances of the same image 
        sum_cols = np.sum(sum_rows) #add all of image distances together of the same class 
        wss_percluster.append(sum_cols) #include this in the total distances per class
        ## find the images farthest/closest to centroid
        indclosest = np.argpartition(sum_rows, n)[:n]
        indfarthest = np.argpartition(sum_rows, -n)[-n:]
        indclosestnum = inds[indclosest]
        indfarthestnum = inds[indfarthest]
        closest_inds.append(indclosestnum)
        farthest_inds.append(indfarthestnum) #final image number
    tot_wss = np.sum(wss_percluster) #sum all of the distances from all of the clusters into 1 value 
    return tot_wss, closest_inds, farthest_inds


# Silhouette plot 
def silhouette_plt(y_sc, silhouette_vals, run, numclusters, indicator):
    cluster_labels = np.unique(y_sc)
    #n_clusters = cluster_labels.shape[0]
    ax_lower, ax_upper = 0, 0
    cticks = []
    sil_plot = plt.figure()
    for i, k in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_sc == k]
        c_silhouette_vals.sort()
        ax_upper += len(c_silhouette_vals)
        color = plt.jet()
        plt.barh(range(ax_lower, ax_upper), c_silhouette_vals, height=1.0, 
                         edgecolor='none', color=color)
        cticks.append((ax_lower + ax_upper) / 2)
        ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--") 
    plt.yticks(cticks, cluster_labels)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    if indicator == 0:
        str2 = 'Unscaled Silhouette Plot For {} Clusters'.format(numclusters)
        name = 'test case {}/{} cluster/og silhouette plot'.format(run, numclusters)
    else:
        str2 = 'Min-Max Silhouette Plot For {} Clusters'.format(numclusters)
        name = 'test case {}/{} cluster/minmax silhouette plot'.format(run, numclusters)
    plt.title(str2)
    plt.tight_layout()
    plt.show()
    sil_plot.savefig(name)
    return silhouette_avg

# Find the average silhouette number per cluster
def silhouette_cluster_average(y_sc, silhouette_values, numclusters):
    silhouette_clusters = [] 
    for i in range(numclusters):
        inds = np.where(y_sc==i)[0]
        avg_ind = sum(silhouette_values[inds])/len(inds) 
        silhouette_clusters.append(avg_ind)
    return silhouette_clusters

# Make the silhouette cluster an array corresponding to the image numbers
def silhouette_cluster_array(y_sc, silhouette_clusters):
    silhouette_cluster_arr = []
    for im in range(len(y_sc)):
        cat = y_sc[im]
        c = silhouette_clusters[cat]
        silhouette_cluster_arr.append(c)
    silhouette_cluster_arr = np.array(silhouette_cluster_arr)
    return silhouette_cluster_arr

# Save histograms for the metrics
def save_hist(data, filename, run, numclusters):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(data, color="black")
    ax.set_title(filename + 'Histogram')
    ax.set_xlabel(filename + 'Values')
    ax.set_ylabel('Frequency')
    #ax.set_xticks(minor=False)
    fig.tight_layout()
    fig.savefig("test case {}/{} cluster/Histogram - {}.png".format(run, numclusters, filename), format='png')
    plt.show(fig)


##### Import Trained Model #####
model = keras.models.load_model("/content/drive/MyDrive/MedIx REU/horseVhumans/my model" , compile = True, custom_objects={"contrastive_loss": contrastive_loss})

# pre set parameters
colors = ['#a52a2a', '#ff7f0e', '#1f77b4', '#2ca02c',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf', '#2cb49b', '#1ae700', '#711d7a', '#ff4d82','#2700ea']
radio_colors = ['#a52a2a', '#ffb732', '#6db6ac', '#6ba10c']

###### Retrieve the embeddings, run KNN analysis, and print tsne for TRAINING DATA #####
# import training data

##horse first
train0 = pd.read_csv("/content/drive/MyDrive/MedIx REU/horseVhumans/TrainingPixel0.csv")
train1 = pd.read_csv("/content/drive/MyDrive/MedIx REU/horseVhumans/TrainingPixel2.csv")
train0 = np.array(train0).reshape(-1,300,300,3) 
train1 = np.array(train1).reshape(-1,300,300,3) 
train_data = np.concatenate((train0, train1), axis=0)
train_label = pd.read_csv("/content/drive/MyDrive/MedIx REU/horseVhumans/trainingPixelLabel.csv")
train_label = np.array(train_label)
#train_data = np.array(train).reshape(-1,300,300,3) 

# get the embeddings and run KNN analysis
embeddings_raw1 = model.predict(train_data)
embeddings1 = (embeddings_raw1 - embeddings_raw1.min()) / (embeddings_raw1.max() - embeddings_raw1.min())
neigh = KNeighborsClassifier(n_neighbors=3)  
print(len(embeddings1))
print(len(train_label))
#print(len(train_data[0]))
neigh.fit(embeddings1, train_label)
train_pred_class = neigh.predict(embeddings1)
# calculate the training data accuracy
accuracy_test = accuracy_score(y_true=train_label, y_pred=train_pred_class)
print("KNN Training Accuracy: ", accuracy_test)


###### Retrieve the embeddings, run KNN analysis, and print tsne for VALIDATION DATA#####
# import validation data
validation0 = pd.read_csv("/content/drive/MyDrive/MedIx REU/horseVhumans/ValidationPixel.csv")
validation1 = pd.read_csv("/content/drive/MyDrive/MedIx REU/horseVhumans/ValidationPixel2.csv")
validation0 = np.array(validation0).reshape(-1,300,300,3) 
validation1 = np.array(validation1).reshape(-1,300,300,3) 
validation_data = np.concatenate((validation0, validation1), axis=0)
validation_label = pd.read_csv("/content/drive/MyDrive/MedIx REU/horseVhumans/validationPixelLabel.csv")
validation_label = np.array(validation_label)
 
# get the embeddings and run KNN analysis
embeddings_raw2 = model.predict(validation_data)
embeddings2 = (embeddings_raw2 - embeddings_raw2.min()) / (embeddings_raw2.max() - embeddings_raw2.min())
neigh.fit(embeddings2, validation_label)
validation_pred_class = neigh.predict(embeddings2)
# calculate the validation data accuracy
accuracy_test = accuracy_score(y_true=validation_label, y_pred=validation_pred_class)
print("KNN Validation Accuracy: ", accuracy_test)


# print the TSNE graph for validation data, along with it's ground truth label color
X_embedded2 = TSNE(n_components=2,random_state=2).fit_transform(embeddings2)
og_tsne1 = plt.figure(figsize=(10,10))
for i in train_legend:
    inds = np.where(validation_label==i)[0]
    plt.scatter(X_embedded2[inds,0],X_embedded2[inds,1],alpha=0.5, color=colors[i])
name1 = 'test case {}/tsne/validation tsne'.format(run)
plt.legend(train_legend)
plt.title('TSNE of Horses vs. Humans Validation Data')
og_tsne1.savefig(name1)

# print the TSNE graphs for the training data, along with it's ground truth label color
train_legend = ['Horse', 'Human']                                                               ######### make sure
X_embedded1 = TSNE(n_components=2,random_state=2).fit_transform(embeddings1)
og_tsne = plt.figure(figsize=(10,10))
for i in train_legend:
    inds = np.where(train_label==i)[0]
    plt.scatter(X_embedded1[inds,0],X_embedded1[inds,1],alpha=0.5, color=colors[i])
name0 = 'test case {}/tsne/train tsne'.format(run)
plt.title('TSNE of Horses vs. Humans Training Data')
plt.legend(train_legend)
og_tsne.savefig(name0)

"""
###### Retrieve the embeddings, run KNN analysis, and print tsne for TESTING DATA#####
# Reading in the testing data
test = pd.read_csv("/content/drive/MyDrive/MedIx REU/Datasets/TestingValsALL.csv")    #50 agree + no agree (all testing)
#test = pd.read_csv("/content/drive/MyDrive/MedIx REU/Datasets/TestingValsNOAgree.csv") #--no agree
#test = pd.read_csv("/content/drive/MyDrive/MedIx REU/Datasets/TestingVals50Agree.csv") #--50 agree
test_ids = test['InstanceID']
test_images = test.drop(['InstanceID'],  axis=1)
test_data = np.array(test_images).reshape(-1,71,71,1) 
test_ids = np.array(test_ids) # length = 591
# get the embeddings and run KNN analysis
y_pred_test_before = model.predict(test_data)
y_pred_test = (y_pred_test_before - y_pred_test_before.min()) / (y_pred_test_before.max() - y_pred_test_before.min())
knn_pred = neigh.predict(y_pred_test)
knn_prob = neigh.predict_proba(y_pred_test)
#print(knn_prob)
#print(knn_pred)
# print the TSNE graph for testing data, but without ground truth color, because there is none 
X_embedded3 = TSNE(n_components=2,random_state=2).fit_transform(y_pred_test)
og_tsne = plt.figure(figsize=(10,10))
for i in range(len(X_embedded3)):
    plt.scatter(X_embedded3[i,0],X_embedded3[i,1],alpha=0.5, color='black')
name2 = 'test case {}/tsne/test tsne'.format(run)
og_tsne.savefig(name2)

"""
##### Run all data (testing, training, and validation) to retrieve embeddings and plot tsne #####  
# read in file with all of the data



allPixel = np.concatenate((train_data, validation_data), axis=0)
allLabels = np.concatenate((train_label, validation_label, ), axis=0)
#trainLabel = pd.read_csv("/content/drive/MyDrive/MedIx REU/horseVhumans/datasetLabeltrain.csv")
#validationLabel = pd.read_csv("/content/drive/MyDrive/MedIx REU/horseVhumans/datasetLabelvalidation.csv")
#total = pd.concat(validationLabel, trainLabel)
dataset_label = allLabels # 1 = Training, 2 = Validation, 
# get the embeddings
total_embeddings_before = model.predict(allPixel)
total_embeddings2 = (total_embeddings_before - total_embeddings_before.min()) / (total_embeddings_before.max() - total_embeddings_before.min())


##### TSNE for each type #####

## basic tsne - no colors
tot_embedded2 = TSNE(n_components=2,random_state=2).fit_transform(total_embeddings2)
tot_tsne2 = plt.figure(figsize=(10,10))
for i in range(len(tot_embedded2)):
    plt.scatter(tot_embedded2[i,0],tot_embedded2[i,1],alpha=0.5, color='black')
name3 = 'test case {}/tsne/total tsne option 2'.format(run)
tot_tsne2.savefig(name3)
hvh_classes = ["Horses", "Humans"]
## plotting the tsne with the colors according to the data (training vs testing) 
    # blue = training and validation, orange = testing
hvh_classes = ['Training Data', 'Validation Data']
tot_tsne2_col = plt.figure(figsize=(10,10))
for i in range(2): #bc 2 is the number of classes (training vs testing)
  inds = np.where(dataset_label==(i+1))[0]
  plt.scatter(tot_embedded2[inds, 0], tot_embedded2[inds,1], alpha=0.5, color=colors[i])
plt.title('TSNE Plot of Training and Testing Data')
plt.legend(hvh_classes)
name4 = 'test case {}/tsne/total tsne option 2 color'.format(run)
tot_tsne2_col.savefig(name4)

"""
## plotting the tsne with varying levels of radiologist disagreement
radiologist = pd.read_csv("/content/drive/MyDrive/AgreementFiles/Everything_Agreements - integers.csv")
radio_ids = radiologist['InstanceID']
agreements = radiologist['Agreement']
tsne_radio = plt.figure(figsize=(10,10))
radio_classes = [3, 2, 1, 0]
radio_names = ['All Radiologists Agree', 'High Radiologist Agreement', 'Low Radiologist Agreement', 'No Radiologist Agreement']
agreement_inds = []
for i in radio_classes:
  inds = np.where(agreements==i)
  plt.scatter(tot_embedded2[inds, 0], tot_embedded2[inds,1], alpha=0.5, color=radio_colors[i])
  agreement_inds.append(inds)  # should be in the order of high agreement -> low agreement 
plt.title('TSNE Plot with Radiologist Agreement Levels')
plt.legend(radio_names)
name5 = 'test case {}/tsne/radiologist tsne'.format(run)
tsne_radio.savefig(name5)
"""
##### Run spectral clustering on the embeddings #####
numbers = [2]
silhouette_total = []
wss_total = []
for numclusters in numbers:
  embeddings = total_embeddings2 #change depending on if we wanna run test cases vs all cases
  X_embedded = tot_embedded2 #change depending on if we wanna run test cases vs all cases
  sc = SpectralClustering(n_clusters=numclusters, random_state=0) 
  y_sc = sc.fit_predict(embeddings)

  # plot the tsne with spectral clusters
  og_spectral = plt.figure(figsize=(10,10))
  for i in range(numclusters):
      inds = np.where(y_sc==i)[0]
      plt.scatter(X_embedded[inds,0],X_embedded[inds,1],alpha=0.5, color=colors[i])
  name3 = 'test case {}/{} cluster/spectral tsne'.format(run, numclusters)
  og_spectral.savefig(name3)


  ##### Data Analysis: WSS & Silhouette #####
  # Calculate centroids and WSS
  centroids = cluster_centroid(embeddings, y_sc, numclusters)
  wss_clusterlist, closest_ind_number, largest_ind_number = wss(centroids, embeddings, y_sc, numclusters, 0) # only want the 0 highest and lowest points -- because the clusters are so small this is necessary for now
  silhouette = silhouette_score(embeddings, y_sc, metric = 'euclidean', random_state=9)
  wss_total.append(wss_clusterlist) # for the wss elbow plot 
  silhouette_total.append(silhouette) 

  # First, do all silhouette work on the raw embeddings
  silhouette_valsb4 = silhouette_samples(embeddings, y_sc, metric='euclidean') # get the silhouette value per sample
  silhouette_avgb4 = silhouette_plt(y_sc, silhouette_valsb4, run, numclusters, 0) # create and save the silhouette plot, and find the average of all of the silhouette scores
  silhouette_clustersb4 = silhouette_cluster_average(y_sc, silhouette_valsb4, numclusters)
  sil_clusterb4_avg = np.mean(silhouette_clustersb4) # average all of the averaged silhouette cluster scores
  stdclustersb4 = np.std(silhouette_clustersb4) # find the standard deviation of the averaged silhouette cluster scores
  sil_clus_arrayb4 = silhouette_cluster_array(y_sc, silhouette_clustersb4) # make the silhouette cluster an array corresponding to the image numbers
  print(len(sil_clus_arrayb4))
  # Min Max all of the silhouette values, then repeat 
  scaler = MinMaxScaler()
  silhouette_vals = scaler.fit_transform(silhouette_valsb4.reshape(-1,1))
  silhouette_vals = silhouette_vals.flatten()
  sil_avg_minmax = silhouette_plt(y_sc, silhouette_vals, run, numclusters, 1)
  silhouette_clusters = silhouette_cluster_average(y_sc, silhouette_vals, numclusters)
  sil_cluster_avg = np.mean(silhouette_clusters)
  stdclusters = np.std(silhouette_clusters)  
  sil_clus_array = silhouette_cluster_array(y_sc, silhouette_clusters)

  # Calculate other clustering metrics, trying S_dbw and SD --- google colab does not have module, may have to pip install?
  # s_dbw_score = S_Dbw(embeddings, y_sc)
  # sd_score = SD(embeddings, y_sc)

  # Writing to a file
  filename = 'test case {}/{} cluster/data analysis'.format(run, numclusters)
  f = open(filename, "w")
  f.write('Silhouette Values Before Min Max Scaling:\n')
  f.write('Silhouette Average of All Values Before Min Max Scaling (Red Line On Plot): {}\n'.format(silhouette_avgb4))
  f.write('Silhouette Average of All Cluster Averages Before Min Max Scaling: {}\n'.format(sil_clusterb4_avg))
  f.write('Silhouette Clusters Standard Deviation Before Min Max Scaling: {}\n\n'.format(stdclustersb4))
  f.write('Silhouette Values After Min Max Scaling:\n')
  f.write('Silhouette Average of All Values After Min Max Scaling (Red Line On Plot): {}\n'.format(sil_avg_minmax))
  f.write('Silhouette Average of All Cluster Averages After Min Max Scaling: {}\n'.format(sil_cluster_avg))
  f.write('Silhouette Clusters Standard Deviation After Min Max Scaling: {}\n'.format(stdclusters))
  #f.write('\nOther Clustering Metrics:')
  #f.write('\nS-DBW Validitiy Index Score: {}\n'.format(s_dbw_score))
  #f.write('\nSD Validity Index Score: {}\n'.format(sd_score))
  f.close()

  # Take the average of the UNSCALED silhouette values 
  scaler = MinMaxScaler()
  ci_avgU = 1 - np.array([np.mean(k) for k in zip(silhouette_valsb4, sil_clus_arrayb4)])  ##  < to switch the metric backwards
  ci_avgU2 = np.array([np.mean(k) for k in zip(silhouette_valsb4, sil_clus_arrayb4)])  ## < for original metric (comparable to KNN)

  # Take the average of the scaled silhouette values 
  ci_avg = 1 - np.array([np.mean(k) for k in zip(silhouette_vals, sil_clus_array)])  ##  < to switch the metric backwards
  ci_avg2 = np.array([np.mean(k) for k in zip(silhouette_vals, sil_clus_array)])  ## < for original metric (comparable to KNN)

  # Multiply the scaled silhouette values
  ci = 1 - (np.multiply(silhouette_vals, sil_clus_array))  ##  < to switch the metric backwards
  ci_norm = scaler.fit_transform(ci.reshape(-1,1)) #< min max the confidence interval values
  ci_norm = ci_norm.reshape(len(y_sc))
  ci2 = np.multiply(silhouette_vals, sil_clus_array)  ##  < for original metric (comparable to KNN)
  ci2_norm = scaler.fit_transform(ci2.reshape(-1,1)) #< min max the confidence interval values
  ci2_norm = ci2_norm.reshape(len(y_sc)) 
  #'Image Number': total_ids,  
  # Output to a file
  dictionary = {
                'Multiplied Sil Metric Scaled': ci_norm, 'Rev Multiplied Sil Metric Scaled': ci2_norm,
                'Avg Sil Metric': ci_avg, 'Rev Avg Sil Metric': ci_avg2,
                'Avg Raw Sil Metric': ci_avgU,  'Rev Avg Raw Sil Metric': ci_avgU2, 
                'Sil Value': silhouette_valsb4, 'Sil Value - S': silhouette_vals, 'Sil Cluster Value': sil_clus_arrayb4, 
                'Sil Cluster Value - S': sil_clus_array, 'Clustered Class': y_sc}
  data = pd.DataFrame(dictionary)
  filename2 = 'test case {}/{} cluster/output file.xlsx'.format(run, numclusters)
  data.to_excel(filename2)

  ## Plot histograms for the metrics 
  list_of_metrics = [ci_norm, ci_avg, ci_avgU] 
  metric_names= ['Multiplied Silhouette Metric Scaled ', 'Average Silhouette Metric ', 'Average Raw Silhouette Metric ']
  for q in range(len(list_of_metrics)):
      save_hist(list_of_metrics[q], metric_names[q], run, numclusters)

  # ## Plot all points above 50% for all of the evaluation metrics 
  # for m in range(len(list_of_metrics)):
  #     leastinds = np.where(list_of_metrics[m] > 0.5)[0]
  #     newleast1 = plt.figure(figsize=(10,10))
  #     for i in range(numclusters):
  #         inds = np.where(y_sc==i)[0]
  #         plt.scatter(X_embedded[inds,0], X_embedded[inds,1], alpha=0.5, color=colors[i])
  #     for l in leastinds:      
  #         plt.scatter(X_embedded[l,0], X_embedded[l,1], color = 'black')
  #     title1 = 'TSNE Plot With {} Clusters and >0.5 Degree Uncertainty With {}'.format(numclusters, metric_names[m])
  #     plt.title(title1)
  #     plt.show()
  #     name1 = 'test case {}/{} cluster/above  0.5 {} - {} clusters.png'.format(run, numclusters, metric_names[m], numclusters)
  #     newleast1.savefig(name1)
      
  # ## Plot the highest 10% and 20% uncertainty of images - only 20% for now 
  # listnum = [282]
  # for m in range(len(list_of_metrics)):
  #     for r in listnum:
  #         indlargest = np.argpartition(list_of_metrics[m], -r)[-r:]
  #         most = plt.figure(figsize=(10,10))
  #         for i in range(numclusters):
  #             inds = np.where(y_sc==i)[0]
  #             plt.scatter(X_embedded[inds,0], X_embedded[inds,1], alpha=0.5, color=colors[i])
  #         for l in indlargest:
  #             plt.scatter(X_embedded[l,0], X_embedded[l,1], color = 'black')
  #         title1 = 'TSNE Plot With {} Clusters and {}% Highest Uncertainty Points With {}'.format(numclusters, str(round(r/(len(y_sc))*100)), metric_names[m])
  #         plt.title(title1)
  #         plt.show()
  #         name1 = 'test case {}/{} cluster/{} highest uncertainty - {} clusters - {}'.format(run, numclusters, r, numclusters, metric_names[m])
  #         most.savefig(name1)
"""
  for m in range(len(list_of_metrics)):
    newlegend = ['All Radiologists Agree', 'High Radiologist Agreement', 'Low Radiologist Agreement', 'No Radiologist Agreement', 'Metric Highest Uncertainty']
    newlistnum = [353, 706]
    for r in newlistnum:
      indlargest = np.argpartition(list_of_metrics[m], -r)[-r:]
      new1 = plt.figure(figsize=(11,11))
      for i in radio_classes:
        inds = np.where(agreements==i)
        plt.scatter(tot_embedded2[inds, 0], tot_embedded2[inds,1], alpha=0.5, color=radio_colors[i]) 
      for k in indlargest:
        plt.scatter(X_embedded[k,0], X_embedded[k,1], alpha=0.5, color = 'black')
      title2 = 'Radiologist Disagreement TSNE Plot With {} Clusters and Comparison of {}Highest {}%'.format(numclusters, metric_names[m], str(round(r/(len(y_sc))*100)))
      plt.title(title2)
      plt.legend(newlegend)
      plt.show()
      name8 = 'test case {}/{} cluster/RADIO - {}% highest uncertainty - {} clusters - {}'.format(run, numclusters, str(round(r/(len(y_sc))*100)), numclusters, metric_names[m])
      new1.savefig(name8)
"""
##### Plot the silhouette and WSS elbow plots #####
# WSS Plot
wss_plot = plt.figure()
plt.plot(numbers, wss_total, color='blue', linewidth=2)
plt.title('WSS Score on Varying Spectral Cluster Sizes')
plt.xticks(np.linspace(2, 6, num=5))
plt.xlabel('Number of Clusters')
plt.ylabel('WSS Value')
plt.show()
name6 = 'test case {}/WSS and Sil/wss error plot'.format(run)
wss_plot.savefig(name6)
# Silhouette Plot
sil_plot = plt.figure()
plt.plot(numbers, silhouette_total, color='blue', linewidth=2)
plt.title('Silhouette Score on Varying Spectral Cluster Sizes')
plt.xticks(np.linspace(2, 6, num=5))
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.show() 
name7 = 'test case {}/WSS and Sil/silhouette plot'.format(run)
sil_plot.savefig(name7)
print(wss_total)
print(silhouette_total)

'''
Extra Notes

# # OPTION 1:
# total_embeddings = np.vstack((embeddings1, embeddings2, y_pred_test)) # length = 1412
# print(len(total_embeddings))
# tot_embedded = TSNE(n_components=2,random_state=2).fit_transform(total_embeddings)
# tot_tsne = plt.figure(figsize=(10,10))
# for i in range(len(tot_embedded)):
#     plt.scatter(tot_embedded[i,0],tot_embedded[i,1],alpha=0.5, color='black')
# name2 = 'test case {}/tsne/total tsne'.format(run)
# tot_tsne.savefig(name2)
'''

print("DONE")

