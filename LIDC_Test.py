from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
import numpy as np  # linear alg
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import random 
import keras
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

##### Parameters #####
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

# Comparison of highest uncertainty and radiologist disagreement points - do this with instance ID, not index number!!
# radio_classes = [3 - All Agree , 2 - High Agreement, 1 - Low Agreement, 0 - No Agreement] 
def radio_comparison(agreement_id, metric, amount, radio_class):
  accuracies = []
  for r in amount:
    indslargest_metric = np.argpartition(metric, -r)[-r:]
    metric_id = total_ids[indslargest_metric] 
    rad_ids = agreement_id[radio_class]
    intersection = np.intersect1d(rad_ids, metric_id) 
    accuracy = len(intersection)  #number of overlap points 
    accuracies.append(accuracy)
  return accuracies

# Comparison of highest uncertainty and radiologist disagreement points - divided by number of images in each category
def radio_comparison2(agreement_id, metric, amount, radio_class):
  accuracies = []
  for r in amount:
    indslargest_metric = np.argpartition(metric, -r)[-r:]
    metric_id = total_ids[indslargest_metric] 
    rad_ids = agreement_id[radio_class]
    intersection = np.intersect1d(rad_ids, metric_id) 
    accuracy = ((len(intersection))/(len(rad_ids)))  #number of overlap points / total number of metric points
    accuracies.append(accuracy)
  return accuracies


##### Import Trained Model #####
model = keras.models.load_model("/content/drive/MyDrive/MedIx REU/Datasets/my model" , compile = True, custom_objects={"contrastive_loss": contrastive_loss})

# pre set parameters
colors = ['#a52a2a', '#ff7f0e', '#1f77b4', '#2ca02c',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf', '#2cb49b', '#1ae700', '#711d7a', '#ff4d82','#2700ea']
radio_colors = ['#a52a2a', '#ffb732', '#6db6ac', '#6ba10c']

###### Retrieve the embeddings, run KNN analysis, and print tsne for TRAINING DATA #####
# import training data
train = pd.read_csv("/content/drive/MyDrive/MedIx REU/Datasets/TrainPixel.csv")
train_ids = (train['InstanceID']).values.tolist() #to list 
train_label = train['Binary Rating']
train_label = pd.Series(train_label).array
train_images = train.drop(['InstanceID', "Binary Rating"], axis=1)
train_data = np.array(train_images).reshape(-1,71,71,1) 
# get the embeddings and run KNN analysis
embeddings_raw1 = model.predict(train_data)
embeddings1 = (embeddings_raw1 - embeddings_raw1.min()) / (embeddings_raw1.max() - embeddings_raw1.min())
neigh = KNeighborsClassifier(n_neighbors=3, metric = 'cosine')   
neigh.fit(embeddings1, train_label)
train_pred_class = neigh.predict(embeddings1)
# calculate the training data accuracy
accuracy_test = accuracy_score(y_true=train_label, y_pred=train_pred_class)
print("Train: ")
print(accuracy_test)
# print the TSNE graphs for the training data, along with it's ground truth label color
train_legend = ['Spiculated','Not Spiculated'] # 0 = Spiculated, 1 = Not Spiculated
X_embedded1 = TSNE(n_components=2,random_state=2).fit_transform(embeddings1)
og_tsne = plt.figure(figsize=(10,10))
for i in range(2):
    inds = np.where(train_label==i)[0]
    plt.scatter(X_embedded1[inds,0],X_embedded1[inds,1],alpha=0.5, color=colors[i])
name0 = 'test case {}/tsne/train tsne'.format(run)
plt.title('TSNE of Training Data With Spiculation Rating')
plt.legend(train_legend)
og_tsne.savefig(name0)


###### Retrieve the embeddings, run KNN analysis, and print tsne for VALIDATION DATA#####
# import validation data
validation = pd.read_csv("/content/drive/MyDrive/MedIx REU/Datasets/TestPixel.csv")
validation_ids = (validation['InstanceID']).values.tolist() #to list 
validation_label = validation['Binary Rating']
validation_label = pd.Series(validation_label).array
validation_images = validation.drop(['InstanceID', "Binary Rating"],  axis=1)
validation_data = np.array(validation_images).reshape(-1,71,71,1) 
# get the embeddings and run KNN analysis
embeddings_raw2 = model.predict(validation_data)
embeddings2 = (embeddings_raw2 - embeddings_raw2.min()) / (embeddings_raw2.max() - embeddings_raw2.min())
neigh.fit(embeddings2, validation_label)
validation_pred_class = neigh.predict(embeddings2)
# calculate the validation data accuracy
accuracy_test = accuracy_score(y_true=validation_label, y_pred=validation_pred_class)
print("Validation: ")
print(accuracy_test)
# print the TSNE graph for validation data, along with it's ground truth label color
X_embedded2 = TSNE(n_components=2,random_state=2).fit_transform(embeddings2)
og_tsne1 = plt.figure(figsize=(10,10))
for i in range(2):
    inds = np.where(validation_label==i)[0]
    plt.scatter(X_embedded2[inds,0],X_embedded2[inds,1],alpha=0.5, color=colors[i])
name1 = 'test case {}/tsne/validation tsne'.format(run)
plt.title('TSNE of Validation Data With Spiculation Rating')
plt.legend(train_legend)
og_tsne1.savefig(name1)


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


##### Run all data (testing, training, and validation) to retrieve embeddings and plot tsne #####  
# read in file with all of the data
total = pd.read_csv("/content/drive/MyDrive/MedIx REU/Datasets/AllPixels - 2.csv")
total_ids = total['InstanceID']
dataset_label = total['Dataset'] # 1 = Training, 2 = Validation, 3 = Testing
total_images = total.drop(['InstanceID','Dataset'],  axis=1)
total_data = np.array(total_images).reshape(-1,71,71,1) 
total_ids = np.array(total_ids) # length = 1413
# get the embeddings
total_embeddings_before = model.predict(total_data)
total_embeddings2 = (total_embeddings_before - total_embeddings_before.min()) / (total_embeddings_before.max() - total_embeddings_before.min())


##### TSNE for each type #####

## basic tsne - no colors
tot_embedded2 = TSNE(n_components=2,random_state=2).fit_transform(total_embeddings2)
tot_tsne2 = plt.figure(figsize=(10,10))
for i in range(len(tot_embedded2)):
    plt.scatter(tot_embedded2[i,0],tot_embedded2[i,1],alpha=0.5, color='black')
name3 = 'test case {}/tsne/total tsne option 2'.format(run)
tot_tsne2.savefig(name3)

## plotting the tsne with the colors according to the data (training vs testing) 
    # blue = training and validation, orange = testing
lidc_classes = ['Training Data', 'Testing Data']
tot_tsne2_col = plt.figure(figsize=(10,10))
for i in range(2): #bc 2 is the number of classes (training vs testing)
  inds = np.where(dataset_label==(i+1))[0]
  plt.scatter(tot_embedded2[inds, 0], tot_embedded2[inds,1], alpha=0.5, color=colors[i])
plt.title('TSNE Plot of Training and Testing Data')
plt.legend(lidc_classes)
name4 = 'test case {}/tsne/total tsne option 2 color'.format(run)
tot_tsne2_col.savefig(name4)

## plotting the tsne with varying levels of radiologist disagreement
radiologist = pd.read_csv("/content/drive/MyDrive/AgreementFiles/Everything_Agreements - integers.csv")
radio_ids = radiologist['InstanceID'] #pandas dataframe
radio_ids2 = radio_ids.values #list
agreements = radiologist['Agreement']
tsne_radio = plt.figure(figsize=(11,11))
radio_classes = [0, 1, 2, 3]
radio_names = ['No Radiologist Agreement', 'Low Radiologist Agreement', 'High Radiologist Agreement', 'All Radiologists Agree']
agreement_inds = []
agreement_id = []
for i in radio_classes:
  inds = np.where(agreements==i)
  plt.scatter(tot_embedded2[inds, 0], tot_embedded2[inds,1], alpha=0.5, color=radio_colors[i])
  ids = radio_ids2[inds]
  agreement_inds.append(inds)  # should be in the order of high agreement -> low agreement  # also, these are the index numbers
  agreement_id.append(ids)  # these are the instance id's
#plt.title('TSNE Plot with Radiologist Agreement Levels')                                                                       # commented out title
plt.title('(b)', fontweight="bold", fontsize=20, y=-0.1)
plt.legend(radio_names, fontsize=13, loc=3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
name5 = 'test case {}/tsne/radiologist tsne'.format(run)
tsne_radio.savefig(name5)

## Print tsne with class labels (spic/non-spic) for training and validation data 
spic_tsne = plt.figure(figsize=(11,11))
tot_train_data = np.concatenate((agreement_id[2], agreement_id[3]))
tot_train_inds = np.hstack((agreement_inds[2], agreement_inds[3]))[0]
spic_colors = ['#ff0a0a', '#2d00c2'] # RED, BLUE   # 0 = Spiculated, 1 = Not Spiculated
spic_legend = ['Spiculated', 'Not Spiculated', 'Unknown']
for i in range(len(tot_embedded2)): 
  plt.scatter(tot_embedded2[i, 0], tot_embedded2[i, 1], alpha=0.5, color='#949899')
for k in range(len(tot_train_data)):
  val = tot_train_data[k]
  emb_ind = tot_train_inds[k]
  if val in validation_ids:
    indexk = validation_ids.index(val)
    binary = validation_label[indexk]
    plt.scatter(tot_embedded2[emb_ind, 0], tot_embedded2[emb_ind, 1], alpha=0.5, color=spic_colors[binary])
  if val in train_ids:
    indexk = train_ids.index(val)
    binary = train_label[indexk]
    plt.scatter(tot_embedded2[emb_ind, 0], tot_embedded2[emb_ind, 1], alpha=0.5, color=spic_colors[binary])
#plt.title('TSNE Plot With Gold Standard Spiculation of Training Images')                                                         # commented out title
plt.title('(a)', fontweight="bold", fontsize=20, y=-0.1)
name9 = 'test case {}/tsne/spiculation tsne'.format(run)
plt.legend(spic_legend, fontsize=17, loc=3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
spic_tsne.savefig(name9)


##### Run spectral clustering on the embeddings #####
numbers = [2]# 3, 4, 5, 6, 7, 8, 9, 10]
silhouette_total = []
wss_total = []
for numclusters in numbers:
  embeddings = total_embeddings2 #change depending on if we wanna run test cases vs all cases
  X_embedded = tot_embedded2 #change depending on if we wanna run test cases vs all cases
  sc = SpectralClustering(n_clusters=numclusters, random_state=0) 
  y_sc = sc.fit_predict(embeddings)
  print(len(embeddings))

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

  # Apply the reverse metric to the silhouette coefficients 
  rev_silhouette_vals = silhouette_vals  
  actual_silhouette_vals = 1 - silhouette_vals

  # Take the average of the scaled silhouette values
  scaler = MinMaxScaler() 
  ci_avg = 1 - np.array([np.mean(k) for k in zip(silhouette_vals, sil_clus_array)])  ##  < to switch the metric backwards
  ci_avg2 = np.array([np.mean(k) for k in zip(silhouette_vals, sil_clus_array)])  ## < for original metric (comparable to KNN)

  # Multiply the scaled silhouette values
  ci = 1 - (np.multiply(silhouette_vals, sil_clus_array))  ##  < to switch the metric backwards
  ci_norm = scaler.fit_transform(ci.reshape(-1,1)) #< min max the confidence interval values
  ci_norm = ci_norm.reshape(len(y_sc))
  ci2 = np.multiply(silhouette_vals, sil_clus_array)  ##  < for original metric (comparable to KNN)
  ci2_norm = scaler.fit_transform(ci2.reshape(-1,1)) #< min max the confidence interval values
  ci2_norm = ci2_norm.reshape(len(y_sc)) 

  # Output to a file
  dictionary = {'Image Number': total_ids, 'Sil Vals Scaled': actual_silhouette_vals, 'Rev Sil Vals Scaled': rev_silhouette_vals,
                'Avg Sil Metric': ci_avg, 'Rev Avg Sil Metric': ci_avg2, 
                'Multiplied Sil Metric Scaled': ci_norm, 'Rev Multiplied Sil Metric Scaled': ci2_norm,
                'Sil Cluster Value Scaled': sil_clus_array, 'Clustered Class': y_sc}
  data = pd.DataFrame(dictionary)
  filename2 = 'test case {}/{} cluster/output file.xlsx'.format(run, numclusters)
  data.to_excel(filename2)

  ## Plot histograms for the metrics 
  list_of_metrics = [actual_silhouette_vals, ci_avg, ci_norm]        
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

  for m in range(len(list_of_metrics)):
    newlegend = ['All Radiologists Agree', 'High Radiologist Agreement', 'Low Radiologist Agreement', 'No Radiologist Agreement', 'Metric Highest Uncertainty']
    newlistnum = [466] #33% #[353, 706] #25%, 50%
    for r in newlistnum:
      indlargest = np.argpartition(list_of_metrics[m], -r)[-r:]
      new1 = plt.figure(figsize=(11,11))
      for i in radio_classes:
        inds = np.where(agreements==i)
        plt.scatter(tot_embedded2[inds, 0], tot_embedded2[inds,1], alpha=0.5, color=radio_colors[i]) 
      for k in indlargest:
        plt.scatter(X_embedded[k,0], X_embedded[k,1], alpha=0.5, color = 'black')
      title2 = 'TSNE Plot With Highest {}% Uncertainty Using {}and {} Clusters'.format(str(round(r/(len(y_sc))*100)), metric_names[m], numclusters)
      #plt.title(title2)                                                                                                                              # commented out title
      plt.title('(c)', fontweight="bold", fontsize=20, y=-0.1)    
      plt.xticks(fontsize=12)
      plt.yticks(fontsize=12)
      plt.legend(newlegend, fontsize=13, loc=3)
      plt.show()
      name8 = 'test case {}/{} cluster/RADIO - {}% highest uncertainty - {} clusters - {}'.format(run, numclusters, str(round(r/(len(y_sc))*100)), numclusters, metric_names[m])
      new1.savefig(name8)

  ## Number of Images in Overlap
  amounts = [141, 466, 706] #%~10%, 33%, 50%
  percentages = ['10%', '33%', '50%']
  #amounts = list(range(70, 1400, 5)) #5% - ~50%
  #percentages = []
  #for i in amounts:
  #  a = str(round((i/1412), 3)) + '%'
  #  percentages.append(a)
  for j in range(len(list_of_metrics)):
    acc_df = {'Metric Highest Uncertainty %': percentages}
    for l in radio_classes:  #[0, 1, 2, 3] = Lowest ---> Highest Agreement
      acc = radio_comparison(agreement_id, list_of_metrics[j], amounts, l)
      acc_df[radio_names[l]] = acc
    combination = np.add((acc_df['No Radiologist Agreement']), (acc_df['Low Radiologist Agreement']))
    acc_df['No & Low Combination'] = combination
    combination2 = np.add((acc_df['High Radiologist Agreement']), (acc_df['All Radiologists Agree']))
    acc_df['All & High Combination'] = combination2
    acc_df = pd.DataFrame(acc_df)
    sheet_names = '{} - Total Number'.format(numclusters)
    acc_df.to_excel(writer_list[j], sheet_name = sheet_names, index = True)

  ## Accuracy (divided by number of images per category (low, high, none, etc.))
  for j in range(len(list_of_metrics)):
    acc_df1 = {'Metric Highest Uncertainty %': percentages}
    for l in radio_classes:  #[0, 1, 2, 3] = Lowest ---> Highest Agreement
      acc = radio_comparison2(agreement_id, list_of_metrics[j], amounts, l)
      acc_df1[radio_names[l]]= acc
    acc_combination = combination/591 #total number of low and no cases
    acc_df1['No & Low Combination'] = acc_combination
    acc_combination2 = combination2/822 #total number of all and high cases
    acc_df1['All & High Combination'] = acc_combination2
    acc_df1 = pd.DataFrame(acc_df1)
    sheet_names = '{} - Over Radio Total'.format(numclusters)
    acc_df1.to_excel(writer_list[j], sheet_name = sheet_names, index = True)

    # Find the most uncertain images each run -- using multiplication
    mostuncertain = np.argpartition(ci_norm, -10)[-10:]
    uncertain_images = total_ids[mostuncertain]
    plt.imshow(uncertain_images[0], cmap = 'gray')
    plt.imshow(uncertain_images[1], cmap = 'gray')
    plt.imshow(uncertain_images[2], cmap = 'gray')
    plt.imshow(uncertain_images[3], cmap = 'gray')
    plt.imshow(uncertain_images[4], cmap = 'gray')
    plt.imshow(uncertain_images[5], cmap = 'gray')
    plt.imshow(uncertain_images[6], cmap = 'gray')
    plt.imshow(uncertain_images[7], cmap = 'gray')
    plt.imshow(uncertain_images[8], cmap = 'gray')
    plt.imshow(uncertain_images[9], cmap = 'gray')
    

 

writer1.save()
writer2.save()
writer3.save()

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