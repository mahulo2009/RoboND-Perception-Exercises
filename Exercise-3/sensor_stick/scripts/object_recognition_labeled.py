#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

import pickle

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    ## Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)

    ## Voxel Grid Downsampling
    voxel_grid_filter = pcl_data.make_voxel_grid_filter()
    voxel_grid_filter.set_leaf_size(0.01, 0.01, 0.01)
    pcl_data = voxel_grid_filter.filter()

    # PassThrough Filter Z
    passthrough_filter = pcl_data.make_passthrough_filter()
    filter_axis = 'z'
    passthrough_filter.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough_filter.set_filter_limits(axis_min, axis_max)
    pcl_data = passthrough_filter.filter()

    # PassThrough Filter X
    passthrough_filter = pcl_data.make_passthrough_filter()
    filter_axis = 'y'
    passthrough_filter.set_filter_field_name(filter_axis)
    axis_min = -5.0
    axis_max = -1.4
    passthrough_filter.set_filter_limits(axis_min, axis_max)
    pcl_data = passthrough_filter.filter()

    # RANSAC Plane Segmentation
    segmenter = pcl_data.make_segmenter()    
    segmenter.set_model_type(pcl.SACMODEL_PLANE)
    segmenter.set_method_type(pcl.SAC_RANSAC)    
    segmenter.set_distance_threshold(0.01)
    inliers, coefficients = segmenter.segment()
    # Extract inliers and outliers
    pcl_data_objects = pcl_data.extract(inliers, negative=True)
    pcl_data_table = pcl_data.extract(inliers, negative=False)

    # Euclidean Clustering
    pcl_data_objects_xyz = XYZRGB_to_XYZ(pcl_data_objects)
    tree = pcl_data_objects_xyz.make_kdtree()
    EuclideanClusterExtraction = pcl_data_objects_xyz.make_EuclideanClusterExtraction()
    EuclideanClusterExtraction.set_ClusterTolerance(0.05)
    EuclideanClusterExtraction.set_MinClusterSize(100)
    EuclideanClusterExtraction.set_MaxClusterSize(2000)
    # Search the k-d tree for clusters
    EuclideanClusterExtraction.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = EuclideanClusterExtraction.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([pcl_data_objects_xyz[indice][0],
                                             pcl_data_objects_xyz[indice][1],
                                             pcl_data_objects_xyz[indice][2],
                                             rgb_to_float(cluster_color[j])])

    pcl_data_objects_clustered = pcl.PointCloud_PointXYZRGB()
    pcl_data_objects_clustered.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages
    ros_objects_msg = pcl_to_ros(pcl_data_objects_clustered)
    ros_table_msg = pcl_to_ros(pcl_data_table)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_objects_msg)
    pcl_table_pub.publish(ros_table_msg)

    detected_objects = []
    detected_objects_labels = []
    # Classify the clusters! (loop through each detected cluster one at a time)
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_data_single_object_clustered = pcl_data_objects.extract(pts_list)
        ros_data_single_object_clustered = pcl_to_ros(pcl_data_single_object_clustered)
        # Compute the associated feature vector
        chists = compute_color_histograms(ros_data_single_object_clustered,using_hsv=True)
        normals = get_normals(ros_data_single_object_clustered)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)
        # Publish a label into RViz
        label_pos = list(pcl_data_objects_xyz[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))
        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_data_single_object_clustered
        detected_objects.append(do)

    # Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    detected_objects_pub.publish(detected_objects)

if __name__ == '__main__':
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']
    # ROS node initialization
    rospy.init_node('clustering',anonymous=True)
    # Create Subscribers
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud",pc2.PointCloud2,pcl_callback,queue_size=1)
    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects",pc2.PointCloud2,queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table",pc2.PointCloud2,queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers",Marker,queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects",DetectedObjectsArray,queue_size=1)
    # Initialize color_list
    get_color_list.color_list = []
    # Spin while node is not shutdown
    while not rospy.is_shutdown():
     rospy.spin()
