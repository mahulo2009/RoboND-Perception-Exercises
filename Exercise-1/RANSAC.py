# Import PCL module
import pcl

# Load Point Cloud file
cloud = pcl.load_XYZRGB('tabletop.pcd')

###### Voxel Grid filter
# Create a VoxelGrid filter object for our input point cloud
voxel_grid_filter = cloud.make_voxel_grid_filter()
# Set the voxel (or leaf) size  
voxel_grid_filter.set_leaf_size(0.01, 0.01, 0.01)
# Call the filter function to obtain the resultant downsampled point cloud
cloud_filtered = voxel_grid_filter.filter()
# Save pcd for voxel grid filter
filename = 'voxel_downsampled.pcd'
pcl.save(cloud_filtered, filename)

###### PassThrough filter
# Create a PassThrough filter object.
passthrough_filter = cloud_filtered.make_passthrough_filter()
# Assign axis and range to the passthrough filter object.
filter_axis = 'z'
passthrough_filter.set_filter_field_name(filter_axis)
axis_min = 0.6
axis_max = 1.1
passthrough_filter.set_filter_limits(axis_min, axis_max)
# Finally use the filter function to obtain the resultant point cloud. 
cloud_filtered = passthrough_filter.filter()
# Save pcd for passthrough filter
filename = 'pass_through_filtered.pcd'
pcl.save(cloud_filtered, filename)

###### RANSAC plane segmentation
# Create the segmentation object
segmenter = cloud_filtered.make_segmenter()
# Set the model you wish to fit 
segmenter.set_model_type(pcl.SACMODEL_PLANE)
segmenter.set_method_type(pcl.SAC_RANSAC)
# Max distance for a point to be considered fitting the model
max_distance = 0.01
segmenter.set_distance_threshold(0.01)
# Call the segment function to obtain set of inlier indices and model coefficients
inliers, coefficients = segmenter.segment()
# Extract inliers
extracted_inliers = cloud_filtered.extract(inliers, negative=False)
# Save pcd for table
filename = 'extracted_inliers.pcd'
pcl.save(extracted_inliers, filename)
# Extract outliers
extracted_outliers = cloud_filtered.extract(inliers, negative=True)
# Save pcd for tabletop objects
filename = 'extracted_outliers.pcd'
pcl.save(extracted_outliers, filename)
