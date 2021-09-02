import sys
import os
import numpy as np
import open3d as o3d
import glob
from tqdm import tqdm

def vector_angle(u, v):
    return np.arccos(np.dot(u,v) / (np.linalg.norm(u)* np.linalg.norm(v)))

def get_alignment_params(my_mesh):
    pcd = my_mesh.sample_points_uniformly(number_of_points=len(my_mesh.vertices))

    # Use method of rejection
    # If the plane extracted by RANSAC is not floor, then delete inliers to this extracted
    # plane and re-run RANSAC.
    while True:
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)

        a, b, c, d = plane_model
        plane_normal = np.array([a, b, c])

        # Approximately perpendicular to z-axis
        if plane_normal[2] < 0.8:
            pcd.colors = o3d.utility.Vector3dVector(np.delete(np.array(pcd.colors), inliers, axis=0))
            pcd.points = o3d.utility.Vector3dVector(np.delete(np.array(pcd.points), inliers, axis=0))
            continue
        else:
            break

    assert np.isclose(np.linalg.norm(plane_normal), 1)
    assert plane_normal[2] > 0.8, "Got normal: {}".format(plane_normal) # near positive z-axis
    
    # Plane equation: Ax + By + Cz + D = 0
    z_axis = np.array([0, 0, 1])
    rotation_angle = vector_angle(z_axis, plane_normal)
    rotation_axis = np.cross(plane_normal, z_axis)

    # Axis-angle rotation is encoded in a single (3, 1) array
    # The direction of the vector is rotation axis; the norm is degree.
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotation_axis = rotation_angle * rotation_axis
    
    R = pcd.get_rotation_matrix_from_axis_angle(rotation_axis)
    
    # simple math suggests that (0, 0, d/c) is on the plane
    # Note that this does not ensure the scene to be aligned to (0, 0, 0)
    return np.array((0, 0, d/c)), R
    
def apply_alignment(my_mesh, translate_vec, rotation_matrix):
    my_mesh = my_mesh.translate(translate_vec)
    my_mesh = my_mesh.rotate(rotation_matrix, center=(0,0,0))
    return my_mesh

def main(base_dir):
    gt_mesh_glob = os.path.join(base_dir, "gt_scene*.ply")
    estimated_mesh_glob = os.path.join(base_dir, "scene*.ply")

    gt_mesh_list = glob.glob(gt_mesh_glob)
    estimated_mesh_list = glob.glob(estimated_mesh_glob)

    assert len(gt_mesh_list) == len(estimated_mesh_list)

    # First pass: check file match
    for gt_mesh_path in gt_mesh_list:
        fn = gt_mesh_path.split('/')[-1]
        dir_name = os.path.dirname(gt_mesh_path)
        estimated_fn = fn[3:]
        estimated_mesh_path = os.path.join(dir_name, estimated_fn)
        assert estimated_mesh_path in estimated_mesh_list

    # Second pass: apply transformation
    for gt_mesh_path in tqdm(gt_mesh_list):
        fn = gt_mesh_path.split('/')[-1]
        dir_name = os.path.dirname(gt_mesh_path)
        estimated_fn = fn[3:]
        estimated_mesh_path = os.path.join(dir_name, estimated_fn)
        assert estimated_mesh_path in estimated_mesh_list
        
        gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
        estimated_mesh = o3d.io.read_triangle_mesh(estimated_mesh_path)
        
        T, R = get_alignment_params(gt_mesh)
        gt_mesh = apply_alignment(gt_mesh, T, R)
        estimated_mesh = apply_alignment(estimated_mesh, T, R)
        
        o3d.io.write_triangle_mesh(gt_mesh_path, gt_mesh, write_ascii=False)
        o3d.io.write_triangle_mesh(estimated_mesh_path, estimated_mesh, write_ascii=False)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 {} /path/to/mesh/folder".format(sys.argv[0]))
        exit()

    main(sys.argv[1])
