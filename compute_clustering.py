import concurrent.futures
import datetime
import logging
import math
import os
import time

from joblib import Memory
import numpy as np
import numpy.ma as ma
import pandas as pd
from hdbscan import HDBSCAN
import sklearn
from sklearn.cluster import DBSCAN  # DBSCAN and other features
"""
if sklearn.__version__ >= '0.21.0':
    from sklearn.cluster import optics
"""
from pyclustering.cluster.optics import optics

from sklearn.metrics.cluster import unsupervised #silhouette_samples
from sklearn.utils import check_X_y
from sklearn.preprocessing import LabelEncoder
#import pdb

from . import canf

"""TODO: 
        - control functions' variables
"""
# Instantiate logs
lib_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(lib_path)
logging.basicConfig(filename=os.path.join(lib_path, "logs", "clustering.log"),
                    level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())



def new_silhouette_samples(X, labels, metric='precomputed', **kwds):
    """ I will remove 'distances' object since in this case X is already a distance 
        matrix. I will also exclude '-1' label from the computation""" 
    X = X.astype(np.float32)
    X, labels = check_X_y(X, labels, accept_sparse=['csc', 'csr'])
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    unsupervised.check_number_of_labels(len(le.classes_), X.shape[0])
    
    #distances = pairwise_distances(X, metric=metric, **kwds)
    unique_labels = le.classes_
    n_samples_per_label = np.bincount(labels, minlength=len(unique_labels))

    # For sample i, store the mean distance of the cluster to which
    # it belongs in intra_clust_dists[i]
    intra_clust_dists = np.zeros(X.shape[0], dtype=X.dtype)

    # For sample i, store the mean distance of the second closest
    # cluster in inter_clust_dists[i]
    inter_clust_dists = np.inf + intra_clust_dists

    for curr_label in range(len(unique_labels)):
        # Do not consider noise label (that is 0)        
        if curr_label != 0:
            # Find inter_clust_dist for all samples belonging to the same
            # label.
            mask = labels == curr_label
            current_distances = X[mask]

            # Leave out current sample.
            n_samples_curr_lab = n_samples_per_label[curr_label] - 1
            if n_samples_curr_lab != 0:
                intra_clust_dists[mask] = np.sum(
                    current_distances[:, mask], axis=1) / n_samples_curr_lab

            # Now iterate over all other labels, finding the mean
            # cluster distance that is closest to every sample.
            for other_label in range(len(unique_labels)):
                if other_label != curr_label and other_label != 0:
                    other_mask = labels == other_label
                    other_distances = np.mean(
                        current_distances[:, other_mask], axis=1)
                    inter_clust_dists[mask] = np.minimum(
                        inter_clust_dists[mask], other_distances)

    sil_samples = inter_clust_dists - intra_clust_dists
    sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    # score 0 for clusters of size 1, according to the paper
    sil_samples[n_samples_per_label.take(labels) == 1] = 0
    return sil_samples

unsupervised.silhouette_samples = new_silhouette_samples

###############################################################################
# Class definition
###############################################################################

class ComputeClustering:
    def __init__(self, distance_matrix, elements, output_folder,
                epsilon=0.5, min_points=10, percentage_elements_to_cluster=65,
                smin=0.3):

        # Output destination for the clustering results
        self.output_folder = os.path.abspath(output_folder)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if not os.path.exists(os.path.join(self.output_folder, "results")):
            os.makedirs(os.path.join(self.output_folder, "results"))
        logging.debug(output_folder)
        
        # Input clustering objects 
        self.D = distance_matrix
        self.elements = elements

        # LENTA DBSCAN specific attribute
        self.percentage_elements_to_cluster = percentage_elements_to_cluster
        self.smin = smin

        # DBSCAN specific attributes
        self.min_points = min_points
        self.epsilon = epsilon

        # Generic clustering output attributes
        self.labels = []
        self.core_samples_mask = []
        self.mean_silhouette_labels = {}
        self.clustering_stats = []

    ###########################################################################
    # Auxiliary methods
    ###########################################################################

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
        return

    # TODO: check function
    def get_kth(self, row, index):
        kth_el = np.sort(row)[index]
        return kth_el
    """
    def compute_silhouette(self):
        # remove noise points and extract a new distance matrix without them
        index_mask = self.labels != -1
        labels_indexes = np.array(range(len(self.labels)))[index_mask]
        no_noise_labels = np.array(self.labels)[index_mask]
        no_noise_dist_matrix = self.extract_sub_matrix(labels_indexes, index_mask,
                                                      self.D, mmap_name = 'silh_matrix')
        try:
            # Compute silhouette for each point != noise
            result = silhouette_samples(X=no_noise_dist_matrix, labels=no_noise_labels,
                                        metric='precomputed')
            # Extract a mean value of silhouette for each cluster
            mean_silhouette_labels = dict()
            for l in set(no_noise_labels):
                label_mask = np.asarray(no_noise_labels) != l  # masking (hiding) all elements different from label l
                masked_result = ma.masked_array(result, mask=label_mask)
                mean_silhouette_labels[l] = masked_result.mean()
            self.mean_silhouette_labels = mean_silhouette_labels
        finally:
            no_noise_dist_matrix_filename = no_noise_dist_matrix.filename
            del no_noise_dist_matrix
            os.remove(no_noise_dist_matrix_filename)
        return self.mean_silhouette_labels, result"""
    
    def compute_silhouette(self):
        result = unsupervised.silhouette_samples(self.D, self.labels, metric='precomputed')
        mean_silhouette_labels = dict()        
        for l in set(self.labels):
            if l != -1:
                label_mask = np.asarray(self.labels) != l
                masked_result = ma.masked_array(result, mask=label_mask)
                mean_silhouette_labels[l] = masked_result.mean()
        self.mean_silhouette_labels = mean_silhouette_labels
        return self.mean_silhouette_labels, result

    def compute_silhouette_overall(self):
        self.compute_silhouette()
        values = np.array(list(self.mean_silhouette_labels.values()))
        n_clusters = len([l for l in set(self.labels) if l != -1])
        return [values.mean(), np.median(values), values.min(), values.max(), values.std(), n_clusters]

    def clustered_elements_stats(self):
        print(len(np.where(self.labels != -1)[0]), len(self.labels))
        perc = float(len(np.where(np.array(self.labels) != -1)[0])) / len(self.labels) * 100
        n_clusters = len([l for l in set(self.labels) if l != -1])
        return [round(perc, 2), n_clusters] 

    def extract_stats(self):
        self.compute_silhouette()
        #logging.debug(self.mean_silhouette_labels)
        #logging.debug(self.labels)
        for cl_label in self.mean_silhouette_labels.keys():
            cluster_stats = dict(cluster_label=cl_label)
            cluster_stats['number_of_elements'] = np.count_nonzero(self.labels == cl_label)
            cluster_stats['silhouette'] = self.mean_silhouette_labels[cl_label]
            self.clustering_stats.append(cluster_stats)
        df_stats = pd.DataFrame(self.clustering_stats)
        df_stats.to_csv(os.path.join(self.output_folder, 'results', 'stats.csv'))
        return self.clustering_stats

    def extract_neighbors(self, my_matrix, min_pts):
        k_dist = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_sort = {executor.submit(self.get_kth, row, min_pts): row for row in my_matrix}
            for future in concurrent.futures.as_completed(future_to_sort):
                try:
                    k_dist.append(future.result())
                except Exception as exc:
                    print('generated an exception: %s' % exc)
                    # sys.exit()
        k_dist = np.sort(np.asarray(k_dist, dtype='float16'))
        return k_dist

    def extract_sub_matrix(self, cl_indexes, index_mask, distance_matrix, mmap_name = 'cluster_matrix'):
        """ Extract a reduced version of the complete distance matrix, looking at the
        elements belonging to the actual cluster"""
        # cluster_matrix = np.zeros( (len(cl_indexes), len(cl_indexes) ), dtype='float16')
        cluster_map = np.memmap(self.output_folder + "/results/" + mmap_name, 
                                dtype='float16', mode='w+',
                                shape=(len(cl_indexes), len(cl_indexes)))

        #cluster_map = self.D[index_mask]
        cluster_map[:] = self.D[:,index_mask][index_mask]
        print(self.D[index_mask])
        """
        matrix_upper_bound = 5000 # use it to periodically store results on disk
        cluster_map_section = cluster_map[:matrix_upper_bound]
        to_compute_section = list()
        for i, index in enumerate(cl_indexes):
            if i % 5000 == 0 and i != 0:
                cluster_map_section  = np.array(to_compute_section)
                cluster_map.flush()
                del cluster_map
                matrix_upper_bound = i + 5000 if (i + 5000) < len(cl_indexes) else len(cl_indexes)
                cluster_map = np.memmap(self.output_folder + "/results/" + mmap_name, 
                        dtype='float16', mode='r+',
                        shape=(len(cl_indexes), len(cl_indexes)))
                cluster_map_section = cluster_map[i:matrix_upper_bound]
            to_compute_section.append(np.asarray(distance_matrix[index, :])[index_mask])
        cluster_map_section = np.array(to_compute_section)
        cluster_map.flush()
        """
        cluster_map.flush()
        print('cluster_map', cluster_map)
        del cluster_map
        cluster_matrix = np.memmap(self.output_folder + "/results/" + mmap_name,
                                   dtype='float16', mode='r+',
                                   shape=(len(cl_indexes), len(cl_indexes)))
        return cluster_matrix

    def final_element_mean_intracluster_distance(self):
        list_of_mean_dist = list()
        for i, row in enumerate(self.D):
            array_of_ind =  np.where(np.array(self.labels) == self.labels[i])[0]
            array_of_ind = np.delete(array_of_ind, np.where(array_of_ind == i))
            array_of_dist = np.array(row)[array_of_ind]
            list_of_mean_dist.append(array_of_dist.mean())
        return list_of_mean_dist

    def save_clustering_result(self):
        df_clusters = pd.DataFrame(data={'elements': self.elements,\
                                         'cluster_label': self.labels,\
                                         'mean_intracluster_distance': self.final_element_mean_intracluster_distance()})
        df_clusters.to_csv(os.path.join(self.output_folder, 'results', 'clusters.csv'))
        #df_stats = pd.DataFrame(self.clustering_stats)
        #df_stats.to_csv(os.path.join(self.output_folder, 'results', 'stats.csv'))

    def compute_kdist_graph(self, distance_matrix, min_points):
        # process to order matrix rows by increasing distance
        logging.debug("trying to extract the best epsilon, be patient...")
        threshold = int((distance_matrix.shape[0] / 100) *  # threshold to choose epsilon (def. 75%)
                        self.percentage_elements_to_cluster)
        kth_values = self.extract_neighbors(distance_matrix, min_points)
        # choose epsilon
        elected_epsilon = math.ceil(kth_values[threshold] * 100) / 100
        # plot k-dist graph
        # n_elements = list(range(1, len(kth_values) + 1))
        # plot_fig(min_points, n_elements, kth_values, elected_epsilon)
        logging.debug("Elected epsilon: %f" % elected_epsilon)
        return elected_epsilon

    def select_automatic_epsilon(self):
        self.epsilon = self.compute_kdist_graph(self.D, self.min_points)
        return self.epsilon

    def load_clustering_results(self, results_dataframe):
        self.labels = list(results_dataframe['cluster_label'])

    ###########################################################################
    # Clustering Methods
    ###########################################################################

    # OK!
    def compute_dbscan(self, algorithm='auto'):
        """ Return labels list for elements which distance matrix was created for"""

        logging.debug("Starting computing DBSCAN," 
                      + "at {0}".format(datetime.datetime.
                                       fromtimestamp(time.time()).
                                       strftime('%Y-%m-%d %H:%M:%S')))


        # compute dbscan
        db = DBSCAN(metric='precomputed',  # Use a pre-computed distance matrix
                    eps=self.epsilon,
                    min_samples=self.min_points,
                    algorithm=algorithm).fit(self.D)


        ## extract infos on core points
        # self.core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # self.core_samples_mask[db.core_sample_indices_] = True

        self.labels = db.labels_.astype(int)

        logging.debug("Finished computing DBSCAN," 
                      + "at {0}".format(datetime.datetime.
                                       fromtimestamp(time.time()).
                                       strftime('%Y-%m-%d %H:%M:%S')))
        logging.debug(str(db))

        # self.extract_stats()

        return self.labels #, self.clustering_stats


    def compute_optics(self):
        logging.debug("Starting computing OPTICS," 
                      + "at {0}".format(datetime.datetime.
                                       fromtimestamp(time.time()).
                                       strftime('%Y-%m-%d %H:%M:%S'))) 

        #cores, labels = optics(self.D, min_samples=self.min_points, \
        #                       metric='precomputed', algorithm='brute')
        
        opt = optics(self.D, self.epsilon, self.min_points, \
                    data_type='distance_matrix', ccore=False)

        opt.process()

        labels = [-1]*len(self.D)
        for i, cl in enumerate(opt.get_clusters()):
            for j in cl:
                labels[j] = i

        self.labels = np.asarray(labels).astype(int)

        logging.debug("Finished computing OPTICS," 
                      + "at {0}".format(datetime.datetime.
                                       fromtimestamp(time.time()).
                                       strftime('%Y-%m-%d %H:%M:%S')))       
        return self.labels
    # OK!
    def compute_hdbscan(self):
        logging.debug("Starting computing HDBSCAN," 
                      + "at {0}".format(datetime.datetime.
                                       fromtimestamp(time.time()).
                                       strftime('%Y-%m-%d %H:%M:%S')))
        cachedir = None
        if self.D.shape[0] > 50000:
            cachedir = "/tmp/joblib"
            logging.debug("using cachedir /tmp/joblib")        
        hdb = HDBSCAN(min_cluster_size=self.min_points,
                      metric='precomputed', memory=Memory(cachedir=cachedir)).fit(self.D)

        self.labels = hdb.labels_.astype(int)
        logging.debug("Finished computing HDBSCAN," 
                      + "at {0}".format(datetime.datetime.
                                       fromtimestamp(time.time()).
                                       strftime('%Y-%m-%d %H:%M:%S')))
        logging.debug(str(hdb))

        # self.extract_stats()

        return self.labels #, self.clustering_stats

    # TODO: check function
    def compute_lenta_dbscan(self, algorithm='auto', allowed_hierarchic_iterations=10):
        """ Extract the silhouette value for every cluster, and iterate the
        clustering algorithms over those cluster with negative silhouette values"""


        logging.debug("Starting computing ITERATIVE-DBSCAN," 
                      + "at {0}".format(datetime.datetime.
                                       fromtimestamp(time.time()).
                                       strftime('%Y-%m-%d %H:%M:%S')))

        self.select_automatic_epsilon()

        # compute classic dbscan in first instance
        first_db = DBSCAN(metric='precomputed',  # Use a pre-computed distance matrix
                          eps=self.epsilon,
                          min_samples=self.min_points,
                          algorithm=algorithm).fit(self.D)

        # extract infos on core points
        self.core_samples_mask = np.zeros_like(first_db.labels_, dtype=bool)
        self.core_samples_mask[first_db.core_sample_indices_] = True

        self.labels = np.asarray(first_db.labels_.astype(int))

        initial_max_label = max(self.labels)

        logging.debug("Initially extracted {0} clusters".format(sum(x is not -1 for x in set(self.labels))))
        # greatest_label = max(self.labels)

        self.compute_silhouette()

        if max(self.labels) > 0:
            flag = True
            label = 0
            # allowed_hierarchic_iterations = 10
            while flag:
                cluster_matrix = np.array([])
                file_mmap = None
                # create a mask for the elements belonging to that label
                index_mask = self.labels == label
                cl_indexes = np.array(range(len(self.labels)))[index_mask]

                # compute silhouette formula
                logging.debug("CLUSTER {0}".format(label))
                #if label > initial_max_label:
                #    self.compute_silhouette()

                # stats.append(silhouette_labels[label])

                logging.debug("Number of elements in the cluster:" 
                               + "%i" % np.count_nonzero(self.labels == label))

                # Try to group differently points in clusters with bad silhouette
                if self.mean_silhouette_labels[label] < self.smin:
                    if allowed_hierarchic_iterations > 0 and np.count_nonzero(self.labels == label) > self.min_points:
                        
                        # Re-apply DBSCAN only over the elements of the cluster
                        cluster_matrix = self.extract_sub_matrix(cl_indexes, index_mask,
                                                                        self.D)
                        print(cluster_matrix)
                        logging.debug(os.listdir(self.output_folder + '/results'))
                        elected_epsilon = self.compute_kdist_graph(cluster_matrix,
                                                               self.min_points)
                        try:
                            db_hier = DBSCAN(metric='precomputed',  # Use a pre-computed distance matrix
                                             eps=elected_epsilon,
                                             min_samples=self.min_points,
                                             algorithm=algorithm).fit(cluster_matrix)

                            db_hier_labels = db_hier.labels_
                            # core = db_hier.core_sample_indices_

                            logging.debug("From cluster {0} ".format(label) +
                                          "extracted {0} clusters".format(max(db_hier_labels) + 1))

                            # if we are in a no-end loop
                            if max(db_hier_labels) + 1 == 1 and label == max(self.labels):
                                for cl_index in cl_indexes:
                                    self.labels[cl_index] = -1
                                allowed_hierarchic_iterations = 0
                                logging.debug("From cluster {0} ".format(label) 
                                              + "is not possible to extract more clusters".format(max(db_hier_labels) + 1))
                                #del self.mean_silhouette_values[label]
                                self.compute_silhouette()
                            else:
                                allowed_hierarchic_iterations -= 1
                                logging.debug("Extracting new clusters")
                                logging.debug(os.listdir(self.output_folder + '/results'))                      
                                db_hier_labels = np.array(
                                    [lab + (max(self.labels) + 1) if lab != -1 else lab for lab in db_hier_labels])
                                logging.debug("Updating the system")
                                # Update the labels' list with the new labels
                                for i, cl_index in enumerate(cl_indexes):
                                    self.labels[cl_index] = db_hier_labels[i]
                                logging.debug("re-compute silhouette")
                                logging.debug(os.listdir(self.output_folder + '/results'))
                                self.compute_silhouette()
                            # file_mmap = cluster_matrix.filename
                        finally:
                            logging.debug("Finally " + str(os.listdir(self.output_folder + '/results')))
                            cluster_matrix_filename = cluster_matrix.filename
                            del cluster_matrix
                            os.remove(cluster_matrix_filename)
                            # if cluster_matrix.shape != (0,) and file_mmap != None:
                            #     del cluster_matrix  # free some memory...
                            #     os.remove(file_mmap)

                    else:
                        for cl_index in cl_indexes:
                            self.labels[cl_index] = -1

                label += 1
                if label > max(self.labels):
                    flag = False

        logging.debug("Finished computing ITERATIVE-DBSCAN," 
                      + "at {0}".format(datetime.datetime.
                                       fromtimestamp(time.time()).
                                       strftime('%Y-%m-%d %H:%M:%S')))
        # self.extract_stats()
            
        return self.labels #, self.clustering_stats

    def compute_canf(self):
        logging.debug("Starting computing CANF," 
                      + "at {0}".format(datetime.datetime.
                                       fromtimestamp(time.time()).
                                       strftime('%Y-%m-%d %H:%M:%S')))
        self.labels = canf.CANF(self.D, self.elements)
        logging.debug("Finished computing CANF," 
                      + "at {0}".format(datetime.datetime.
                                       fromtimestamp(time.time()).
                                       strftime('%Y-%m-%d %H:%M:%S')))
        return self.labels


