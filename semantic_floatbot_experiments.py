import pandas as pd
from tqdm import tqdm
from klampt.model.collide import WorldCollider
from klampt import WorldModel,Geometry3D
from klampt.plan.cspace import CSpace,MotionPlan


from planning.disinfection3d import DisinfectionProblem
from planning.robot_cspaces import Robot3DCSpace,CSpaceObstacleSolver 

import pdb
from klampt import vis
import time
import numpy as np
import pickle
from glob import glob
from floatbot_experiments import FloatbotDisinfectionProblem,Floatbot3DCSpace


problem = FloatbotDisinfectionProblem(
    total_dofs = 3,
    linear_dofs = [0,1,2],
    angular_dofs = [],
    frozen_dofs = [],
    base_height_link = 2,
    robot_height = 1.5,
    lamp_linknum = 2,
    lamp_local_coords = [0,0,0],
    active_dofs = [0,1,2],
    robot_cspace_generator = Floatbot3DCSpace,
    robot_cspace_solver = CSpaceObstacleSolver,
    float_height = 0.08)



# all_meshes = glob('./data/aug_10_entire_val_ade20kmodel_vanilla_prob_weighting/estimated_scannet_val/*.ply')

# meshes_series = pd.Series(all_meshes)

# estimated_mask = ~meshes_series.str.split('/',expand = True).iloc[:,-1].str.startswith('gt')

# estimated_meshes = sorted(meshes_series[estimated_mask].tolist())
a = pd.read_csv('meshes_by_floorplan_area.csv', sep = '|')

b = a.loc[~a.mesh_names.str.startswith('gt'),:]
estimated_meshes = b.loc[b.floorplan_area > 36,'mesh_file'].tolist()
done_series = pd.Series(glob('./3D_results/Semantic/*/*/floatbot/floatbot_roadmap_330_divs.p')).str.split('/',expand = True)
# experiments = ['surface_agnostic','hard_cutoff_50']
experiments = ['surface_agnostic_{}_minutes','soft_thresholding_{}_minutes','hard_cutoff_50_{}_minutes','hard_cutoff_25_{}_minutes']
# experiments = ['hard_cutoff_50_{}_minutes','hard_cutoff_25_{}_minutes']

for experiment in experiments: 
    if(experiment == 'surface_agnostic_{}_minutes'):
        hard_cutoff = False
        cutoff_threshold = 0
    elif(experiment == 'soft_thresholding_{}_minutes'):
        hard_cutoff = False
        cutoff_threshold = 0.5
    else:
        hard_cutoff = True
        cutoff_threshold = float(experiment.split('_')[2])/100
    for time_limit in [None,1,2,4,6,8,10,15,30]:
        if(time_limit is not None):
            tmax = time_limit/60
        else:
            tmax = None
        this_experiment = experiment.format(time_limit)
        for mesh_file in estimated_meshes:
            mesh_name = mesh_file.split('/')[-1].split('.')[0]
            if(mesh_name in ['scene0187_00','scene0423_00','scene0208_00']):
                print('skipping hopeless scene {}'.format(mesh_name))
                continue
            filtered_done = done_series[done_series.iloc[:,6] == mesh_name]
            done_experiments = filtered_done.iloc[:,7]
            if(this_experiment not in done_experiments.values):
                try:
                    res = 330
                    # if(mesh_name not in done_meshes):
                    print('\n\n\n\n performing experiment {} on mesh file = {} \n\n\n'.format(this_experiment, mesh_file))
                    total_distance, coverage, resolution, res_dir = problem.perform_experiment(
                        results_dir = './3D_results/Semantic',
                        mesh_file = mesh_file,
                        min_distance = 0.05,
                        from_scratch = True,
                        irradiance_from_scratch = False,
                        float_height = 0.15,
                        power = 10,
                        resolution = res,
                        experiment = this_experiment,
                        tmax = tmax,
                        robot_name = 'floatbot',
                        hard_cutoff = hard_cutoff,
                        cutoff_threshold = cutoff_threshold,
                        area_penalty = 10,
                        semantic_penalty = 10,
                        show_vis = False
                    )
                except Exception as e:
                    print('initial planning failed for mesh {}!'.format(mesh_name))
                    with open('./failed_meshes_floatbot.txt','a') as f:
                        f.write(mesh_name+'\r\n')
                    pass
            else:
                print('skipping experiment {} in mesh {}'.format(this_experiment,mesh_name))
