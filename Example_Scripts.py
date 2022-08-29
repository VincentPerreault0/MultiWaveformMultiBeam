import copy
import time
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from MWMB_Problem import ProbInstance, ProbParameters
from MWMB_Solution import ImpTree
import MWMB_Plot
from MWMB_Algorithm import search, get_best_simplex_tuple_lists_from_BESTS_log_file


"""
Below are a few useful script demos for the problem.

Each can be activated by changing the "if False:..." to "if True:..." and running this script.
"""




"""
Creates new synthetic instances and saves them in the folder "./Problem_Instances/" (which should exist before running the script).
"""
if False:
  overwrite_instances = False

  nb_nodes = 10
  inst_idx_start = 0
  nb_new_instances = 10
  avg_dist_km = 10

  for i in range(nb_new_instances):
    inst_name = "synt_"+str(nb_nodes)+"_node_instance_"+str(i+inst_idx_start)

    if overwrite_instances or not Path("./Problem_Instances/"+inst_name+".pkl").is_file():
      inst = ProbInstance.create_random(nb_nodes, avg_dist_km=avg_dist_km, verbose=False)

      # inst.show(title=inst_name) # Plots the created instance
      inst.save(inst_name)

"""
Creates new synthetic instances at multiple scales and saves them in the folder "./Problem_Instances/" (which should exist before running the script).
"""
if False:
  overwrite_instances = False

  nb_nodes = 10
  inst_idx_start = 0
  nb_new_instances = 10
  avg_dist_km = 10
  coeffs = [0.25, 0.5, 1, 2]
  coeff_labels = ["2-5","5","10","20"]

  for i in range(nb_new_instances):
    if overwrite_instances or not min([Path("./Problem_Instances/synt_"+str(nb_nodes)+"_node_instance_"+str(i+inst_idx_start)+"_"+coeff_label+".pkl").is_file() for coeff_label in coeff_labels]):
      insts = ProbInstance.create_mult_random(nb_nodes, avg_dist_km=avg_dist_km, coeffs=coeffs, verbose=False)
      for coeff_idx in range(len(coeffs)):
        inst_name = "synt_"+str(nb_nodes)+"_node_instance_"+str(i+inst_idx_start)+"_"+coeff_labels[coeff_idx]

        inst = insts[coeff_idx]

        # inst.show(title=inst_name) # Plots the created instance
        inst.save(inst_name)

"""
Converts Ghassan's realistic instances from matlab file format to ProbInstance objects and saves them in the folder "./Problem_Instances/" (which should exist before running the script).
"""
if False:
  overwrite_instances = False

  matlab_files_directory = "./EnvData_17June2022/"
  NosOfSites = [10,15,20,30,40,50]
  InverseNodeDensities = [20,50]

  for NoOfSites in NosOfSites:
    for InverseNodeDensity in InverseNodeDensities:
      for iCase in range(1,201):
        if iCase < 10:
          iCaseTxT = "000"+str(iCase)
        elif iCase < 100:
          iCaseTxT = "00"+str(iCase)
        else:
          iCaseTxT = "0"+str(iCase)
        
        new_inst_name = "real_"+str(NoOfSites)+"_node_instance_"+str(InverseNodeDensity)+"_"+str(iCase-1)

        if overwrite_instances or not Path("./Problem_Instances/"+new_inst_name+".pkl").is_file():
          dictmat = {}
          loadmat(matlab_files_directory+"Results_"+str(NoOfSites)+"Nodes_Env"+str(InverseNodeDensity)+"/InverseNodeDensity"+str(InverseNodeDensity)+"_iCase"+iCaseTxT+".mat", mdict=dictmat)

          coordinates_km = np.zeros((NoOfSites,2))
          for i in range(NoOfSites):
            coordinates_km[i,0] = dictmat["EnvOUT"][0,0][0][0,i][0,0][5][0,0]/1000
            coordinates_km[i,1] = dictmat["EnvOUT"][0,0][0][0,i][0,0][6][0,0]/1000
          coordinates_km[:,0] -= np.min(coordinates_km[:,0])
          coordinates_km[:,1] -= np.min(coordinates_km[:,1])

          path_losses_dB = np.asarray(dictmat["EnvOUT"][0,0][3])
          path_losses_dB = np.stack([path_losses_dB[:,:,0],path_losses_dB[:,:,0],path_losses_dB[:,:,1],path_losses_dB[:,:,1]],axis=2)

          fade_margins_dB = np.asarray(dictmat["EnvOUT"][0,0][4])
          fade_margins_dB = np.stack([fade_margins_dB[:,:,0],fade_margins_dB[:,:,0],fade_margins_dB[:,:,1],fade_margins_dB[:,:,1]],axis=2)

          new_inst = ProbInstance(coordinates_km,path_losses_dB,fade_margins_dB)

          # new_inst.show(title=new_inst_name) # Plots the converted instance
          new_inst.save(new_inst_name)



"""
Computes the "greedy" pseudo-objective of a full mesh cluster for some saved instances and saves the results in a specified directory (which should exist before running the script).
"""
if False:
  results_directory = "./Results/"
  overwrite_results = False

  nbs_nodes = [10,20,50]
  inst_idx_start = 0
  nb_insts = 10
  is_inst_synt = False
  if is_inst_synt:
    is_multi_scale = True
    if is_multi_scale:
      scales = ["2-5","5","10","20"]
    else:
      scales = [1]
  else:
    scales = [20,50]
  
  mesh_high_traffic_routes_prioritizations = [0]+20*[1]

  params = ProbParameters(Mesh_coeff=1, mesh_collisions=True, multibeam=False)

  verbose = False

  for nb_nodes in nbs_nodes:
    for scale in scales:
      for inst_idx in range(inst_idx_start, inst_idx_start+nb_insts):
        if is_inst_synt:
          if is_multi_scale:
            inst_label = "synt_"+str(nb_nodes)+"_node_instance_"+str(inst_idx)+"_"+scale
          else:
            inst_label = "synt_"+str(nb_nodes)+"_node_instance_"+str(inst_idx)
        else:
          inst_label = "real_"+str(nb_nodes)+"_node_instance_"+str(scale)+"_"+str(inst_idx)
        if overwrite_results or not Path(results_directory+inst_label+"_MESH_RESULTS.txt").is_file():
          res_file = open(results_directory+inst_label+"_MESH_RESULTS.txt","w")
          if verbose:
            print(inst_label)
          inst = ProbInstance.load(inst_label)
          full_mesh_imp_tree = ImpTree.create_full_mesh(inst, params)
          if full_mesh_imp_tree.get_obj_value(obj_func="chan_max", get_network=True)[1].A_obj_min > 0:
            tic = time.time()
            best_obj_value = -2
            for curr_mesh_high_traffic_routes_prioritization in mesh_high_traffic_routes_prioritizations:
              full_mesh_imp_tree.prob_params.mesh_high_traffic_routes_prioritization = curr_mesh_high_traffic_routes_prioritization
              obj_value, full_mesh_network = full_mesh_imp_tree.get_obj_value(obj_func="greedy", get_network=True)
              if verbose:
                print(str(curr_mesh_high_traffic_routes_prioritization)+"\t"+str(obj_value))
              if obj_value > best_obj_value:
                best_obj_value = obj_value
                best_full_mesh_network = copy.deepcopy(full_mesh_network)
            eval_time = time.time() - tic
            
            MWMB_Plot.print_log((best_full_mesh_network.A_obj_min,best_full_mesh_network.B_obj_min,best_full_mesh_network.C_obj_min,best_obj_value,eval_time), True, files=[res_file])
            plt.ion()
            best_full_mesh_network.show()
            plt.pause(1e-8)
            plt.ioff()
            plt.savefig(results_directory+inst_label+"_FULL_MESH_NETWORK.png", dpi=300)
            plt.close()
          else:
            MWMB_Plot.print_log((0,0,0,0,0), True, files=[res_file])
          res_file.close()

"""
Searches optimal topology and network in terms of the "greedy" pseudo-objective for some saved instances and saves the results in a specified directory (which should exist before
running the script).
"""
if False:
  results_directory = "./Results/"
  results_filename_postfix = ""
  overwrite_results = False

  nbs_nodes = [10,20,50]
  inst_idx_start = 0
  nb_insts = 10
  is_inst_synt = False
  if is_inst_synt:
    is_multi_scale = True
    if is_multi_scale:
      scales = ["2-5","5","10","20"]
    else:
      scales = [1]
  else:
    scales = [20,50]

  params = ProbParameters(multibeam = True,
                          nb_freqs_per_channel = 2,
                          obj_scenarios = [1/13, 4/13, 8/13],
                          SINR_threshold_throughput_table = [(2,6.5),(5,13),(9,19.5),(11,26),(15,39),(18,52),(20,58.5),(25,65),(29,78)], # [] for Shannon-Hartley or [(2,6.5),(5,13),(9,19.5),(11,26),(15,39),(18,52),(20,58.5),(25,65),(29,78)] for lookup table
                          PTP_coeff = 2,
                          Mesh_coeff = 0.1,
                          mesh_collisions = False
                         )

  search_all_scenarios = True
  nb_best_solutions = 5

  max_time = 600

  neighborhoods = "tree" # "tree" or "mesh" or "hybrid"

  init_simplex_tuple_lists = [] # if specific starting point is required

  verbose = True # print log in console
  show_progression = True # show progression and current solutions in real-time (figure cannot be clicked/moved and slightly slows down the algorithm)

  for nb_nodes in nbs_nodes:
    for scale in scales:
      for inst_idx in range(inst_idx_start, inst_idx_start+nb_insts):
        if is_inst_synt:
          if is_multi_scale:
            inst_label = "synt_"+str(nb_nodes)+"_node_instance_"+str(inst_idx)+"_"+scale
          else:
            inst_label = "synt_"+str(nb_nodes)+"_node_instance_"+str(inst_idx)
        else:
          inst_label = "real_"+str(nb_nodes)+"_node_instance_"+str(scale)+"_"+str(inst_idx)
        new_full_inst_label = results_directory+inst_label+results_filename_postfix
        if overwrite_results or not Path(new_full_inst_label+"_LOG.txt").is_file():
          if verbose and (len(nbs_nodes) > 1 or len(scales) > 1 or nb_insts > 1):
            print("\n\n\n")
          inst = ProbInstance.load(inst_label)
          best_tree, best_network = search(inst, params, neighborhoods, 100000000000, max_time=max_time, get_network=True, full_obj_func="greedy", saved_log_and_plots_filepath_no_extension=new_full_inst_label, init_simplex_tuple_lists=init_simplex_tuple_lists, search_all_scenarios=search_all_scenarios, nb_best_solutions=nb_best_solutions, verbose=verbose, fig_number=0, show_progression=show_progression)
          plt.close('all')
          plt.ion()
          fig, ax = plt.subplots(1,1)
          best_tree.show(ax=ax)
          plt.pause(1e-8)
          plt.ioff()
          plt.savefig(new_full_inst_label+"_BEST_TREE.png", dpi=300)
          plt.ion()
          fig, ax = plt.subplots(1,1)
          best_network.show(ax=ax)
          plt.pause(1e-8)
          plt.ioff()
          plt.savefig(new_full_inst_label+"_BEST_NETWORK.png", dpi=300)

"""
Continues the search for a saved instance from a "_BESTS.txt" file (for instance, if the previous search ended in an error).
"""
if False:
  results_directory = "./Results/"
  old_results_filename_postfix = ""
  new_results_filename_postfix = "_cont"

  nb_nodes = 10
  inst_idx = 0
  is_inst_synt = False
  if is_inst_synt:
    is_multi_scale = True
    if is_multi_scale:
      scale = "10" # "2-5" or "5" or "10" or "20"
      inst_label = "synt_"+str(nb_nodes)+"_node_instance_"+str(inst_idx)+"_"+scale
    else:
      inst_label = "synt_"+str(nb_nodes)+"_node_instance_"+str(inst_idx)
  else:
    scale = 20 # 20 or 50
    inst_label = "real_"+str(nb_nodes)+"_node_instance_"+str(scale)+"_"+str(inst_idx)

  params = ProbParameters(multibeam = True,
                          nb_freqs_per_channel = 2,
                          obj_scenarios = [1/13, 4/13, 8/13],
                          SINR_threshold_throughput_table = [(2,6.5),(5,13),(9,19.5),(11,26),(15,39),(18,52),(20,58.5),(25,65),(29,78)], # [] for Shannon-Hartley or [(2,6.5),(5,13),(9,19.5),(11,26),(15,39),(18,52),(20,58.5),(25,65),(29,78)] for lookup table
                          PTP_coeff = 2,
                          Mesh_coeff = 0.1,
                          mesh_collisions = False
                         )

  search_all_scenarios = True
  nb_best_solutions = 5

  max_time = 600

  neighborhoods = "tree" # "tree" or "mesh" or "hybrid"

  verbose = True # print log in console
  show_progression = True # show progression and current solutions in real-time (figure cannot be clicked/moved and slightly slows down the algorithm)

  old_full_inst_label = results_directory+inst_label+old_results_filename_postfix
  new_full_inst_label = results_directory+inst_label+new_results_filename_postfix
  if Path(old_full_inst_label+"_BESTS.txt").is_file():
    init_simplex_tuple_lists = get_best_simplex_tuple_lists_from_BESTS_log_file(old_full_inst_label+"_BESTS.txt", nb_best_solutions=nb_best_solutions)
    inst = ProbInstance.load(inst_label)
    best_tree, best_network = search(inst, params, neighborhoods, 100000000000, max_time=max_time, get_network=True, full_obj_func="greedy", saved_log_and_plots_filepath_no_extension=new_full_inst_label, init_simplex_tuple_lists=init_simplex_tuple_lists, search_all_scenarios=search_all_scenarios, nb_best_solutions=nb_best_solutions, verbose=verbose, fig_number=0, show_progression=show_progression)
    plt.close('all')
    plt.ion()
    fig, ax = plt.subplots(1,1)
    best_tree.show(ax=ax)
    plt.pause(1e-8)
    plt.ioff()
    plt.savefig(new_full_inst_label+"_BEST_TREE.png", dpi=300)
    plt.ion()
    fig, ax = plt.subplots(1,1)
    best_network.show(ax=ax)
    plt.pause(1e-8)
    plt.ioff()
    plt.savefig(new_full_inst_label+"_BEST_NETWORK.png", dpi=300)

"""
Plots the best topologies/networks from a "_BESTS.txt" file.
"""
if False:
  results_directory = "./Results/"
  results_filename_postfix = ""

  nb_nodes = 10
  inst_idx = 0
  is_inst_synt = False
  if is_inst_synt:
    is_multi_scale = True
    if is_multi_scale:
      scale = "10" # "2-5" or "5" or "10" or "20"
      inst_label = "synt_"+str(nb_nodes)+"_node_instance_"+str(inst_idx)+"_"+scale
    else:
      inst_label = "synt_"+str(nb_nodes)+"_node_instance_"+str(inst_idx)
  else:
    scale = 20 # 20 or 50
    inst_label = "real_"+str(nb_nodes)+"_node_instance_"+str(scale)+"_"+str(inst_idx)
  full_inst_label = results_directory+inst_label+results_filename_postfix
  inst = ProbInstance.load(inst_label)
  
  params = ProbParameters(multibeam = True,
                          nb_freqs_per_channel = 2,
                          obj_scenarios = [1/13, 4/13, 8/13],
                          SINR_threshold_throughput_table = [(2,6.5),(5,13),(9,19.5),(11,26),(15,39),(18,52),(20,58.5),(25,65),(29,78)], # [] for Shannon-Hartley or [(2,6.5),(5,13),(9,19.5),(11,26),(15,39),(18,52),(20,58.5),(25,65),(29,78)] for lookup table
                          PTP_coeff = 2,
                          Mesh_coeff = 0.1,
                          mesh_collisions = False
                         )

  search_all_scenarios = False
  nb_best_solutions = 5

  show_network = True # True: Network or False: ImpTree
  
  if Path(full_inst_label+"_BESTS.txt").is_file() and (not search_all_scenarios or (Path(full_inst_label+"_BESTS_A.txt").is_file() and Path(full_inst_label+"_BESTS_B.txt").is_file() and Path(full_inst_label+"_BESTS_C.txt").is_file())):
    trees_or_nets = []
    best_simplex_tuple_lists = get_best_simplex_tuple_lists_from_BESTS_log_file(full_inst_label+"_BESTS.txt", nb_best_solutions=nb_best_solutions)
    for simplex_tuple_list in best_simplex_tuple_lists:
      imp_tree = ImpTree.create_imp_tree_from_simplex_tuple_list(inst, params, simplex_tuple_list)
      obj_value, network = imp_tree.get_obj_value(obj_func="greedy", get_network=True)
      if not show_network:
        trees_or_nets.append(imp_tree)
      else:
        trees_or_nets.append(network)
    if not search_all_scenarios:
      nb_y, nb_x = MWMB_Plot.get_subplot_nb_y_and_nb_x(len(trees_or_nets))
    else:
      nb_y = 4
      nb_x = nb_best_solutions
      for scenario in ["", "A", "B", "C"]:
        if scenario != "":
          best_simplex_tuple_lists = get_best_simplex_tuple_lists_from_BESTS_log_file(full_inst_label+"_BESTS_"+scenario+".txt", nb_best_solutions=nb_best_solutions)
          for simplex_tuple_list in best_simplex_tuple_lists:
            imp_tree = ImpTree.create_imp_tree_from_simplex_tuple_list(inst, params, simplex_tuple_list)
            obj_value, network = imp_tree.get_obj_value(obj_func="greedy", get_network=True)
            if not show_network:
              trees_or_nets.append(imp_tree)
            else:
              trees_or_nets.append(network)
        while len(trees_or_nets) % nb_best_solutions != 0:
          trees_or_nets.append(ImpTree(inst,params,np.zeros((nb_nodes,nb_nodes))))
    fig, axs = plt.subplots(nb_y, nb_x, num="Best Solutions", gridspec_kw={"left":0.075, "right":0.925, "bottom":0.075, "top":0.925})
    fig.canvas.manager.resize(800,700)
    MWMB_Plot.subplot(trees_or_nets, nb_y, nb_x, fig_axs=(fig, axs))
    plt.show()



