import copy
from pathlib import Path
import random
import time
from math import ceil, floor
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

from MWMB_Solution import ImpTree
import MWMB_Plot

"""
Searches the space of solutions to design a good sub-optimal implicit tree and network given an instance of the problem and parameters where
  'prob_instance' is the associated problem instance;
  'prob_params' is the associated problem parameters;
  'nb_streams' is the number of parallel Taboo Searches (kappa in the thesis);
  'neighborhoods' is which neighborhoods are used in the search, such that
    "tree" --> only the tree neighborhood is used (only full tree solutions are found, i.e. with no mesh clusters);
    "mesh" --> only the mesh neighborhood is used (mesh clusters are possible);
    "hybrid" --> both neighborhoods are used in the search (recommended setting for the general case with mesh clusters);
  'max_iter' is the maximum number of iterations before the search stops;
  'max_no_better_iter' (optional) is the maximum number of consecutive iterations without finding a new best before the search stops; 
  'max_time' (optional) is the maximum time limit at which the search stops; 
  'get_network' (optional) is whether the best Network found (for the best ImpTree found) should be returned as well (True) or not (False);
  'neighborhood_subset_ratio' (optional) is a float in (0,1] that determines the ratio of the neighborhood that should be searched in every neighborhood search;
  'saved_log_and_plots_filepath_no_extension' (optional) is the filepath without extension (folder + filename) with which should be saved the outputted images and text log files;
  'init_neighborhood' (optional) is the initial neighborhood such that
    only possible value is "tree" if 'neighborhoods' is "tree";
    only possible value is "mesh" if 'neighborhoods' is "mesh";
    only possible values are "tree" or "mesh" if 'neighborhoods' is "hybrid";
  'neighb_max_iter' (optional) is the maximum number of iterations before the neighborhood changes ('neighborhoods' is "hybrid") / before the current solutions are reset ('reset'
    and 'neighborhoods' is "tree" or "mesh");
  'neighb_max_no_better_iter' (optional) is the maximum number of consecutive iterations without finding a new best before the neighborhood changes ('neighborhoods' is "hybrid") /
    before the current solutions are reset ('reset' and 'neighborhoods' is "tree" or "mesh");
  'reset' (optional) is, if 'neighborhoods' is "tree" or "mesh", whether the current solutions should be reset once in a while (True) or never (False);
  'init_simplex_tuple_lists' (optional) is a list of initial solutions each in the form of a simplex tuple list;
  'search_all_scenarios' (optional) is whether to also consider all traffic scenarios individually and remember the best implicit trees of each (True) or not (False);
  'fixed_master_hub' (optional) is whether to proceed by considering one possible master hub at a time (True) or all simultaneously (False);
  'full_obj_func' (optional) is the identifier of the desired real objective to be optimized (if different than substitute, only used at the end if nb_nodes <= 10);
  'nb_best_solutions' (optional) is the number of desired best solutions to be outputted by the search.
  'substitute_obj_func' (optional) is the identifier of the desired objective to be optimized or an accurate approximation;
  'nb_best_substitutes' (optional) is the number of desired best solutions to be re-evaluated by the full objective at the end if nb_nodes <= 10;
  'neighborhood_obj_func' (optional) is the identifier of the pseudo-objective used in the neighborhood searches;
  'first_better' (optional) is whether to do the neighborhood searches by first improvement (True, compatible with 'neighborhood_subset_ratio' < 1) or best improvement (False);
  'skip_poor_bound' (optional) is whether to evaluate an upper bound and skip the evaluation of the substitute if the upper bound isn't promising;
  'bound_obj_func' (optional) is the identifier of the pseudo-objective used as an upper bound to decied whether to evaluate the substitute or not;
  'empty_taboo_new_best' (optional) is whether to empty the taboo list of a solution that is a new best (True) or not (False);
  'check_times' (optional) is whether to record the timing of the algorithm and its components;
  'check_for_duplicates' (optional) is whether to keep in memory previous solutions and only evaluate the solutions that have never been visited before;
  'verbose' (optional) is whether to print the intermediary results in the console (True) or not (False);
  'fig_number' (optional) is a number or identifier for the figure object to be created, if none is given;
  'show_best' (optional) is whether to show/save a figure of the 'nb_best_solutions' best final solutions at the end of the algorithm;
  'show_progression' (optional) is whether to show the progress of the search in real time;
  'show_all' (optional) is whether to show all the visited solutions.

Examples:
  best_tree = search(prob_instance, prob_params, nb_streams, neighborhoods, max_iter)
  best_tree, best_network = search(prob_instance, prob_params, nb_streams, neighborhoods, max_iter, get_network = True)
"""
def search(prob_instance, prob_params, neighborhoods, max_iter, max_no_better_iter=-1, max_time=-1, get_network=False, nb_streams=2, neighborhood_subset_ratio=0.5, saved_log_and_plots_filepath_no_extension="", init_neighborhood="tree", neighb_max_iter=25, neighb_max_no_better_iter=-1, reset=True, init_simplex_tuple_lists=[], search_all_scenarios=False, fixed_master_hub=False, full_obj_func="full", nb_best_solutions=1, substitute_obj_func="greedy", nb_best_substitutes=-1, neighborhood_obj_func="chan_avg", first_better=True, skip_poor_bound=True, bound_obj_func="chan_max", empty_taboo_new_best=True, check_times=True, check_for_duplicates=True, verbose=True, fig_number=-1, show_best=True, show_progression=False, show_all=False):
  # META-PARAMETERS INITIALIZATION
  if True:
    nb_nodes = prob_instance.nb_nodes

    if not prob_params.mixed:
      neighborhoods = "tree"

    if nb_nodes > 10 and full_obj_func == "full":
      full_obj_func = "greedy"
    
    if max_no_better_iter == -1:
      max_no_better_iter = round(max_iter/2)

    if neighb_max_no_better_iter == -1:
      if neighborhoods != "tree" and neighborhoods != "mesh":
        neighb_max_no_better_iter = round(neighb_max_iter/2)
      elif reset:
        neighb_max_no_better_iter = neighb_max_iter

    if neighborhood_subset_ratio < 1:
      first_better = True

    if full_obj_func == substitute_obj_func or nb_best_substitutes < nb_best_solutions:
      nb_best_substitutes = nb_best_solutions
    
    if max_time > 0:
      check_times = True

  
  # TIMER START
  if check_times:
    total_tic = time.time()
      

  # INPUT
  if True:
    if saved_log_and_plots_filepath_no_extension != "":
      full_log_file = open(saved_log_and_plots_filepath_no_extension+"_LOG.txt","w")
      full_log_files = [full_log_file]
      bests_log_file = open(saved_log_and_plots_filepath_no_extension+"_BESTS.txt","w")
      bests_log_files = [bests_log_file]
      if search_all_scenarios:
        A_bests_log_file = open(saved_log_and_plots_filepath_no_extension+"_BESTS_A.txt","w")
        A_bests_log_files = [A_bests_log_file]
        B_bests_log_file = open(saved_log_and_plots_filepath_no_extension+"_BESTS_B.txt","w")
        B_bests_log_files = [B_bests_log_file]
        C_bests_log_file = open(saved_log_and_plots_filepath_no_extension+"_BESTS_C.txt","w")
        C_bests_log_files = [C_bests_log_file]
        all_log_files = [full_log_file, bests_log_file, A_bests_log_file, B_bests_log_file, C_bests_log_file]
      else:
        A_bests_log_files = []
        B_bests_log_files = []
        C_bests_log_files = []
        all_log_files = [full_log_file, bests_log_file]
      MWMB_Plot.print_log("Instance \""+(saved_log_and_plots_filepath_no_extension.split("/"))[-1]+"\"\n", True, files=all_log_files)
    else:
      full_log_files = []
      bests_log_files = []
      A_bests_log_files = []
      B_bests_log_files = []
      C_bests_log_files = []
      all_log_files = []
    if search_all_scenarios:
      MWMB_Plot.print_log("Number of Streams : "+str(nb_streams)+"       \tNb. of Best Solutions Memorized : "+str(nb_best_solutions)+"   (Searching All 3 Scenarios)", True, files=all_log_files)
    else:
      MWMB_Plot.print_log("Number of Streams : "+str(nb_streams)+"       \tNb. of Best Solutions Memorized : "+str(nb_best_solutions), True, files=all_log_files)
    if neighborhoods == "tree" or neighborhoods == "mesh":
      MWMB_Plot.print_log("Neighborhood : "+neighborhoods, True, files=all_log_files)
    else:
      MWMB_Plot.print_log("Neighborhoods :     "+str(neighborhoods)+"    \tInitial Neighborhood :            "+init_neighborhood, True, files=all_log_files)
    if len(init_simplex_tuple_lists) > 0:
      MWMB_Plot.print_log("Initial Trees : "+str([init_simplex_tuple_list for init_simplex_tuple_list in init_simplex_tuple_lists]), True, files=all_log_files)
    if max_time > 0:
      if max_time < 60:
        MWMB_Plot.print_log("Maximum Time : "+str(max_time)+" s", True, files=all_log_files)
      elif max_time < 3600:
        max_mins = floor(max_time/60)
        max_secs = max_time - 60 * max_mins
        MWMB_Plot.print_log("Maximum Time : "+str(max_mins)+" m, "+str(max_secs)+" s", True, files=all_log_files)
      else:
        max_hours = floor(max_time/3600)
        max_mins = floor((max_time - 3600*max_hours)/60)
        max_secs = max_time - 3600 * max_hours - 60 * max_mins
        MWMB_Plot.print_log("Maximum Time : "+str(max_hours)+" h, "+str(max_mins)+" m, "+str(max_secs)+" s", True, files=all_log_files)
    MWMB_Plot.print_log("Maximum Number of Iterations :     "+str(max_iter)+"   \tMax. Iterations With no New Best Solution : "+str(max_no_better_iter), True, files=all_log_files)
    if neighborhoods != "tree" and neighborhoods != "mesh":
      MWMB_Plot.print_log("Max. Iterations per Neighborhood : "+str(neighb_max_iter)+"    \tMax. New Best Solution per Neighborhood :   "+str(neighb_max_no_better_iter), True, files=all_log_files)
    elif reset:
      MWMB_Plot.print_log("Max. Iterations Since New Best Solution for Reset :   "+str(neighb_max_no_better_iter), True, files=all_log_files)
    if full_obj_func != substitute_obj_func:
      MWMB_Plot.print_log("Full Objective Function :    "+str(full_obj_func)+"       \tSubstitute Obj. Function :  "+str(substitute_obj_func)+"   (Evaluating the Full Function Only on the Best "+str(nb_best_substitutes)+" Solutions)", True, files=all_log_files)
    else:
      MWMB_Plot.print_log("Full Objective Function :    "+str(full_obj_func), True, files=all_log_files)
    if skip_poor_bound:
      MWMB_Plot.print_log("Neighborhood Obj. Function : "+str(neighborhood_obj_func)+"   \tUpper Bound Obj. Function : "+str(bound_obj_func), True, files=all_log_files)
    else:
      MWMB_Plot.print_log("Neighborhood Obj. Function : "+str(neighborhood_obj_func), True, files=all_log_files)


  # TOPOLOGICAL DATA ANALYSIS
  if True:
    distances_dB = prob_instance.path_losses_dB[:,:,-1] + prob_instance.fade_margins_dB[:,:,-1]

    ## Sorting Edges by Distance
    edge_dist_pairs = []
    for v1 in range(nb_nodes):
      for v2 in range(v1+1,nb_nodes):
        edge_dist_pairs.append(((v1,v2), distances_dB[v1,v2]))
    edge_dist_pairs.sort(key=lambda edge_dist_pair: edge_dist_pair[1])

    ## Sorting Mesh Edges by Signal and Finding First Tree Edges with 0 Signal
    omni_edge_dict = {omni_signal_dB:[] for omni_signal_dB in [prob_params.Mesh_coeff * signal_dB for signal_dB in [6.5, 13, 19.5, 26, 39, 52, 58.5, 65, 78]]}
    for first_zero_edge_idx in range(len(edge_dist_pairs)):      
      omni_signal_dB = prob_params.Mesh_coeff * prob_params.get_perfect_signal_TP(edge_dist_pairs[first_zero_edge_idx][1], nb_beams=0)
      if omni_signal_dB > 0:
        omni_edge_dict[omni_signal_dB].append({edge_dist_pairs[first_zero_edge_idx][0][0],edge_dist_pairs[first_zero_edge_idx][0][1]})
      signal_dB = prob_params.get_perfect_signal_TP(edge_dist_pairs[first_zero_edge_idx][1])
      if signal_dB == 0:
        break
    
    ## Creating Ordered Non-0 Edges
    ordered_edges = [edge_dist_pairs[edge_idx][0] for edge_idx in range(first_zero_edge_idx)]

    ## Checking if Instance is Connectable
    connectable_components = ImpTree.get_adj_matrix_conn_components(ImpTree.create_adj_matrix_from_simplex_tuple_list(ordered_edges, nb_nodes=nb_nodes), nb_nodes=nb_nodes)
    if len(connectable_components) > 1:
      MWMB_Plot.print_log("\nThis instance cannot be fully connected. Search ABORTED.", True, files=all_log_files)
      return
    
    ## Creating Non-0 Edge Filter Matrix
    elite_edge_filter_matrix = np.zeros((nb_nodes, nb_nodes))
    for v1,v2 in ordered_edges:
      elite_edge_filter_matrix[v1,v2] = 1
      elite_edge_filter_matrix[v2,v1] = 1
    elite_edge_filter_matrix = elite_edge_filter_matrix > 0

    if neighborhoods == "tree":
      ## Creating Min Spanning Tree
      if len(init_simplex_tuple_lists) == 0:
        init_min_spanning_tree_simplex_tuple_lists = [ImpTree.create_min_spanning_tree_simplex_tuple_list(nb_nodes=nb_nodes, edge_dist_pairs=edge_dist_pairs)]

      ordered_cluster_lists = []
      ordered_cluster_adj_matrices = []
      ordered_cluster_min_omni_signals = []
      mesh_compatibility_matrix = -1
    else:
      ## Finding Maximal Clusters and Creating Min Spanning Trees
      ordered_cluster_lists = []
      ordered_cluster_adj_matrices = []
      ordered_cluster_min_omni_signals = []
      init_min_spanning_tree_simplex_tuple_lists = []
      omni_adj_matrix = np.zeros((nb_nodes,nb_nodes))
      for omni_signal_idx in reversed(range(10)):
        min_omni_signal_dB = [prob_params.Mesh_coeff * signal_dB for signal_dB in [6.5, 13, 19.5, 26, 39, 52, 58.5, 65, 78, float('inf')]][omni_signal_idx]
        #print(min_omni_signal_dB)

        if omni_signal_idx == 9:
          actual_clusters = []
        elif len(omni_edge_dict[min_omni_signal_dB]) == 0 or (len(ordered_cluster_lists) > 0 and len(ordered_cluster_lists[-1][0]) == nb_nodes):
          continue
        else:
          for edge in omni_edge_dict[min_omni_signal_dB]:
            v1, v2 = tuple(edge)
            omni_adj_matrix[v1, v2] = omni_adj_matrix[v2, v1] = 1
        
          omni_conn_comps = ImpTree.get_adj_matrix_conn_components(omni_adj_matrix)
          pot_clusters = [conn_comp for conn_comp in omni_conn_comps if len(conn_comp) >= 3]

          actual_clusters = []

          ### Constructing Maximal Clusters for Min Omni Signal Value
          while len(pot_clusters) > 0:
            #print("  "+str(pot_clusters))
            pot_cluster = pot_clusters.pop(0)
            pot_cluster_adj_matrix = omni_adj_matrix[np.ix_(pot_cluster,pot_cluster)]

            #print("    "+str(pot_cluster))
            #print("      Node Degree >= 2")
            #### Every Node Should Have a Degree >= 2 in its Cluster
            nodes_to_check = copy.copy(pot_cluster)
            while len(nodes_to_check) > 0:
              node_idx = nodes_to_check.pop(0)
              if np.sum(pot_cluster_adj_matrix[pot_cluster.index(node_idx),:]) < 2:
                pot_cluster.remove(node_idx)
                if len(pot_cluster) == 0:
                  break
                pot_cluster_adj_matrix = omni_adj_matrix[np.ix_(pot_cluster,pot_cluster)]
                nodes_to_check = copy.copy(pot_cluster)
            
            #### No Node Should Separate its Cluster in 2 if Removed
            if len(pot_cluster) > 0:
              #print("    "+str(pot_cluster))
              #print("      Node Shouldn't Be Single Bridge")
              nodes_to_check = copy.copy(pot_cluster)
              while len(nodes_to_check) > 0:
                node_idx = nodes_to_check.pop(0)
                pot_cluster_without_node = [cluster_node for cluster_node in pot_cluster if cluster_node != node_idx]
                without_node_conn_comps = ImpTree.get_adj_matrix_conn_components(omni_adj_matrix[np.ix_(pot_cluster_without_node,pot_cluster_without_node)])
                if len(without_node_conn_comps) > 1:
                  if len(without_node_conn_comps[0]) + 1 > 2:
                    pot_clusters.append([pot_cluster_without_node[pot_cluster_without_node_idx] for pot_cluster_without_node_idx in without_node_conn_comps[0]]+[node_idx])
                  if len(without_node_conn_comps[1]) + 1 > 2:
                    pot_clusters.append([pot_cluster_without_node[pot_cluster_without_node_idx] for pot_cluster_without_node_idx in without_node_conn_comps[1]]+[node_idx])
                  pot_cluster = []
                  break
            
            #### Every Cut of the Cluster in 2 Should Have a Degree >= 2
            if len(pot_cluster) > 0:
              #print("    "+str(pot_cluster))
              #print("      Cut Degree >= 2")
              tmp_adj_lists = [[] for idx in range(len(pot_cluster))]
              w1s,w2s = np.where(omni_adj_matrix[np.ix_(pot_cluster,pot_cluster)]==1)
              for i in range(w1s.size):
                w1 = w1s[i]
                w2 = w2s[i]
                if w2 > w1:
                  tmp_adj_lists[w1].append(w2)
                  tmp_adj_lists[w2].append(w1)
              for i in range(w1s.size):
                w1 = w1s[i]
                w2 = w2s[i]
                if w2 > w1:
                  tmp_adj_lists[w1].remove(w2)
                  tmp_adj_lists[w2].remove(w1)
                  partition_idcs = [-1 for idx in range(len(pot_cluster))]
                  next_idcs_to_visit = [[w1],[w2]]
                  is_not_1_cut = False
                  while len(next_idcs_to_visit[0]) > 0 or len(next_idcs_to_visit[1]) > 0:
                    for part_idx in range(2):
                      idcs_to_visit = next_idcs_to_visit[part_idx]
                      next_idcs_to_visit[part_idx] = []
                      for idx in idcs_to_visit:
                        partition_idcs[idx] = part_idx
                        for adj_idx in tmp_adj_lists[idx]:
                          if partition_idcs[adj_idx] < 0:
                            next_idcs_to_visit[part_idx].append(adj_idx)
                          elif partition_idcs[adj_idx] != part_idx:
                            is_not_1_cut = True
                            break
                        if is_not_1_cut:
                          break
                      if is_not_1_cut:
                        break
                    if is_not_1_cut:
                      break
                  if not is_not_1_cut:
                    for part_idx in range(2):
                      part = [pot_cluster[idx] for idx in range(len(pot_cluster)) if partition_idcs[idx] == part_idx]
                      if len(part) > 2:
                        pot_clusters.append(part)
                    pot_cluster = []
                    break
                  tmp_adj_lists[w1].append(w2)
                  tmp_adj_lists[w2].append(w1)

            if len(pot_cluster) > 0:
              #print("    "+str(pot_cluster))
              actual_clusters.append(pot_cluster)
        
        #print(actual_clusters)
        if (omni_signal_idx < 9 and len(actual_clusters) == 0) or (len(ordered_cluster_lists) > 0 and actual_clusters == ordered_cluster_lists[-1]):
          continue
        elif len(actual_clusters) > 0:
          ordered_cluster_lists.append(actual_clusters)
          ordered_cluster_adj_matrices.append(copy.copy(omni_adj_matrix))
          ordered_cluster_min_omni_signals.append(min_omni_signal_dB)

        ### Creating the Associated Min Spanning Tree
        if len(init_simplex_tuple_lists) == 0:
          init_min_spanning_tree_simplex_tuple_lists.append(ImpTree.create_min_spanning_tree_simplex_tuple_list(nb_nodes=nb_nodes, edge_dist_pairs=edge_dist_pairs, clusters=actual_clusters))

      if len(ordered_cluster_lists) == 0:
        MWMB_Plot.print_log("\nMesh clusters are not possible for this instance... Search will only consider trees.", True, files=all_log_files)
        neighborhoods = "tree"
        if reset:
          neighb_max_no_better_iter = neighb_max_iter
        mesh_compatibility_matrix = -1
      else:
        mesh_compatibility_matrix = omni_adj_matrix == 1


  # GLOBAL VARIABLES INITIALIZATION
  if True:
    ## Master Hubs if Fixed_Master_Hub
    if not fixed_master_hub:
      MHs = [-1]
    else:
      MHs = [exp_node for exp_node in range(nb_nodes)]
      glob_best_neighb_obj_values = [(-1,-1,-1,-1,-1,[],[])]
      glob_best_subst_obj_values = [(-1,-1,-1,-1,-1,[],[])]
      if search_all_scenarios:
        glob_best_subst_A_obj_mins = [(-1,-1,-1,[],[])]
        glob_best_subst_B_obj_mins = [(-1,-1,-1,[],[])]
        glob_best_subst_C_obj_mins = [(-1,-1,-1,[],[])]

    ## Timing Statistics
    if check_times:
      neighb_obj_times = []
      if skip_poor_bound:
        bound_obj_times = []
      subst_obj_times = []
      tree_neighb_search_times = []
      mesh_neighb_search_times = []
      full_obj_times = []

    ## First Better Statistics
    if first_better:
      nb_tree_visited_all_neighbors = 0
      nb_tree_neighbor_search = 0
      nb_mesh_visited_all_neighbors = 0
      nb_mesh_neighbor_search = 0
    
    ## Duplicates
    if check_for_duplicates:
      tree_adj_matrices_dup = []
      iters_streams_dup = []
      nb_duplicates = 0


  # FIXED MASTER HUB LOOP
  for MH in MHs:
    if (verbose or saved_log_and_plots_filepath_no_extension != "") and fixed_master_hub:
      MWMB_Plot.print_log("\n\nMaster Hub = "+str(MH)+"\n", verbose, files=full_log_files)


    # BEAM SEARCH VARIABLES INITIALIZATION
    if True:
      ## Neighborhood
      if neighborhoods == "tree" or neighborhoods == "mesh":
        neighborhood = neighborhoods
        first_tree_switch = False
        first_mesh_switch = False
      else:
        neighborhood = init_neighborhood
        if neighborhood == "tree":
          first_tree_switch = False
          first_mesh_switch = True
        else:# neighborhood == "mesh"
          first_tree_switch = True
          first_mesh_switch = False
      iter_in_curr_neighborhood = 0
      iter_in_curr_neighb_since_new_best = 0

      ## Current Trees
      trees = []
      if len(init_simplex_tuple_lists) == 0:
        if neighborhood == "tree":
          for init_min_spanning_tree_simplex_tuple_list in init_min_spanning_tree_simplex_tuple_lists:
            if len(init_min_spanning_tree_simplex_tuple_list[0]) < nb_nodes:
              trees.append(ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, init_min_spanning_tree_simplex_tuple_list))
        else:
          if len(init_min_spanning_tree_simplex_tuple_lists) == 1:
            trees.append(ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, init_min_spanning_tree_simplex_tuple_lists[0]))
          else:
            for init_min_spanning_tree_idx in range(1,len(init_min_spanning_tree_simplex_tuple_lists)):
              trees.append(ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, init_min_spanning_tree_simplex_tuple_lists[init_min_spanning_tree_idx]))
      else:
        for init_simplex_tuple_list in init_simplex_tuple_lists:
          trees.append(ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, init_simplex_tuple_list))
      
      ## Stream Lineage
      if show_progression or show_best or check_for_duplicates:
        stream_lineages = []
        all_neighb_obj_values = [[]]
        all_subst_obj_values = [[]]

      ## Best Obj. Values
      best_neighb_obj_values = [(-1,-1,-1,-1,-1,[],[])]
      best_subst_obj_values = [(-1,-1,-1,-1,-1,[],[])]
      if search_all_scenarios:
        best_subst_A_obj_mins = [(-1,-1,-1,[],[])]
        best_subst_B_obj_mins = [(-1,-1,-1,[],[])]
        best_subst_C_obj_mins = [(-1,-1,-1,[],[])]
      iter_since_new_best = 0

      ## Taboo Lists
      if len(trees) > 0:
        init_nb_trees = len(trees)
      else:
        if neighborhood == "tree":
          if not fixed_master_hub:
            init_nb_trees = nb_nodes
          else:
            init_nb_trees = 1
        else:# neighborhood == "mesh"
          init_nb_trees = 1
      taboo_add_lists = [[] for i in range(init_nb_trees)]
      taboo_drop_lists = [[] for i in range(init_nb_trees)]

      ## Times
      iter_times = []

      ## Iterations
      i = 0
        

    # BEAM SEARCH INITIALIZATION
    if True:

      ## Initial Trees Initialization (if none provided)
      if verbose or saved_log_and_plots_filepath_no_extension != "":
        MWMB_Plot.print_log("\n0 - init trees", verbose, files=full_log_files)
        best_subst_obj_value = 0

      ## Initial Trees Neighborhood Obj. Values & Ordering
      best_init_tree_quins = []
      for tree_idx in range(len(trees)):
        if check_times:
          tic = time.time()
        neighb_obj_value, neighb_net = trees[tree_idx].get_obj_value(get_network = True, obj_func=neighborhood_obj_func, MH=MH)
        if check_times:
          neighb_obj_times.append(time.time() - tic)
        best_init_tree_quins.append((tree_idx, neighb_net.A_obj_min, neighb_net.B_obj_min, neighb_net.C_obj_min, neighb_obj_value))

      if len(best_init_tree_quins) > 1:
        best_init_tree_quins.sort(key = lambda best_init_tree_quin: best_init_tree_quin[4])
        new_trees = [trees[tree_idx] for tree_idx,_,_,_,_ in best_init_tree_quins]
        trees = new_trees

      if show_progression:
        tree_MHs = []

      ## Initial Trees Bound/Substitute Obj. Values
      for tree_idx in range(len(trees)):
        tree = trees[tree_idx]
        _, neighb_A_obj_min, neighb_B_obj_min, neighb_C_obj_min, neighb_obj_value = best_init_tree_quins[tree_idx]
        simplex_tuple_list = tree.get_simplex_tuple_list()

        if check_for_duplicates:
          tree_adj_matrices_dup.append(copy.deepcopy(tree.adj_matrix))
          iters_streams_dup.append((0,tree_idx))

        if skip_poor_bound:
          if check_times:
            tic = time.time()
          bound_obj_value = tree.get_obj_value(obj_func=bound_obj_func, MH=MH)
          if check_times:
            bound_obj_times.append(time.time() - tic)
        
        if check_times:
          tic = time.time()
        subst_obj_value, subst_net = tree.get_obj_value(obj_func = substitute_obj_func, get_network = True, MH=MH)
        if check_times:
          subst_obj_times.append(time.time() - tic)
        mesh_sizes = subst_net.get_mesh_sizes()

        best_neighb_obj_values.append((neighb_A_obj_min, neighb_B_obj_min, neighb_C_obj_min, neighb_obj_value, 0, mesh_sizes, simplex_tuple_list))

        best_subst_obj_values.append((subst_net.A_obj_min, subst_net.B_obj_min, subst_net.C_obj_min, subst_obj_value, 0, mesh_sizes, simplex_tuple_list))
        if check_times:
          MWMB_Plot.print_log(str((time.time() - total_tic,(subst_net.A_obj_min, subst_net.B_obj_min, subst_net.C_obj_min, subst_obj_value, 0, mesh_sizes, simplex_tuple_list))), False, files=bests_log_files)
        else:
          MWMB_Plot.print_log(str((0,(subst_net.A_obj_min, subst_net.B_obj_min, subst_net.C_obj_min, subst_obj_value, 0, mesh_sizes, simplex_tuple_list))), False, files=bests_log_files)
        if search_all_scenarios:
          best_subst_A_obj_mins.append((subst_net.A_obj_min, subst_net.A_obj_value, 0, mesh_sizes, simplex_tuple_list))
          if check_times:
            MWMB_Plot.print_log(str((time.time() - total_tic,(subst_net.A_obj_min, subst_net.A_obj_value, 0, mesh_sizes, simplex_tuple_list))), False, files=A_bests_log_files)
          else:
            MWMB_Plot.print_log(str((0,(subst_net.A_obj_min, subst_net.A_obj_value, 0, mesh_sizes, simplex_tuple_list))), False, files=A_bests_log_files)
          best_subst_B_obj_mins.append((subst_net.B_obj_min, subst_net.B_obj_value, 0, mesh_sizes, simplex_tuple_list))
          if check_times:
            MWMB_Plot.print_log(str((time.time() - total_tic,(subst_net.B_obj_min, subst_net.B_obj_value, 0, mesh_sizes, simplex_tuple_list))), False, files=B_bests_log_files)
          else:
            MWMB_Plot.print_log(str((0,(subst_net.B_obj_min, subst_net.B_obj_value, 0, mesh_sizes, simplex_tuple_list))), False, files=B_bests_log_files)
          best_subst_C_obj_mins.append((subst_net.C_obj_min, subst_net.C_obj_value, 0, mesh_sizes, simplex_tuple_list))
          if check_times:
            MWMB_Plot.print_log(str((time.time() - total_tic,(subst_net.C_obj_min, subst_net.C_obj_value, 0, mesh_sizes, simplex_tuple_list))), False, files=C_bests_log_files)
          else:
            MWMB_Plot.print_log(str((0,(subst_net.C_obj_min, subst_net.C_obj_value, 0, mesh_sizes, simplex_tuple_list))), False, files=C_bests_log_files)
        
        if show_progression or show_best or check_for_duplicates:
          all_neighb_obj_values[-1].append(neighb_obj_value)
          all_subst_obj_values[-1].append(subst_obj_value)
        if show_progression:
          tree_MHs.append(subst_net.imp_arb.MH)

        if verbose or saved_log_and_plots_filepath_no_extension != "":
          if len(best_neighb_obj_values) == 1 or neighb_obj_value > best_neighb_obj_values[-2][3]:
            new_best_neighb_string = "!!! "
          else:
            new_best_neighb_string = "    "
          if subst_obj_value > best_subst_obj_value:
            new_best_subst_string = "!!! "
          else:
            new_best_subst_string = "    "
          if skip_poor_bound:
            MWMB_Plot.print_log("\t"+new_best_neighb_string+str(round(100*neighb_obj_value)/100)+" \t"+str(round(100*bound_obj_value)/100)+" \t"+new_best_subst_string+str(round(100*subst_obj_value)/100)+" \t   "+str(simplex_tuple_list)+"      mesh sizes = "+str(mesh_sizes), verbose, files=full_log_files)
          else:
            MWMB_Plot.print_log("\t"+new_best_neighb_string+str(round(100*neighb_obj_value)/100)+" \t"+new_best_subst_string+str(round(100*subst_obj_value)/100)+" \t   "+str(simplex_tuple_list)+"      mesh sizes = "+str(mesh_sizes), verbose, files=full_log_files)

      ## Best Obj. Values Ordering
      best_neighb_obj_values = best_neighb_obj_values[-nb_best_solutions:]

      best_subst_obj_values.sort(key = lambda Amin_Bmin_Cmin_value_iter_mesh_simplex_hept: Amin_Bmin_Cmin_value_iter_mesh_simplex_hept[3])
      best_subst_obj_values = best_subst_obj_values[-max(nb_best_substitutes,nb_streams):]

      if search_all_scenarios:
        best_subst_A_obj_mins.sort(key = lambda min_value_iter_mesh_simplex_quin: min_value_iter_mesh_simplex_quin[1])
        best_subst_A_obj_mins = best_subst_A_obj_mins[-nb_best_substitutes:]

        best_subst_B_obj_mins.sort(key = lambda min_value_iter_mesh_simplex_quin: min_value_iter_mesh_simplex_quin[1])
        best_subst_A_obj_mins = best_subst_A_obj_mins[-nb_best_substitutes:]

        best_subst_C_obj_mins.sort(key = lambda min_value_iter_mesh_simplex_quin: min_value_iter_mesh_simplex_quin[1])
        best_subst_A_obj_mins = best_subst_A_obj_mins[-nb_best_substitutes:]
      
      if check_times:
        iter_times.append(time.time() - total_tic)
      else:
        iter_times.append(0)
      if show_progression:
        plt.ion()
        best_subst_value = max(all_subst_obj_values[-1])
        if nb_streams == 1:
          nb_y, nb_x = (5,5)
          #neighb_beam_size = 3
          #subst_beam_size = 1
          fig, axs = plt.subplots(nb_y, nb_x, num="Taboo Search Progression - Iteration 0", gridspec_kw={"left":0.075, "right":0.925, "bottom":0.075, "top":0.925})
          fig.canvas.manager.resize(800,600)
          fig_gridspec = axs[0, 0].get_gridspec()
          for idx_y in range(nb_y):
            for idx_x in range(nb_x):
              axs[idx_y,idx_x].remove()
          neighb_beam_ax = fig.add_subplot(fig_gridspec[-1, :])#fig.add_subplot(fig_gridspec[-1, :neighb_beam_size])
          subst_beam_ax = fig.add_subplot(fig_gridspec[-2, :])#fig.add_subplot(fig_gridspec[-1, neighb_beam_size:neighb_beam_size+subst_beam_size])
          subst_beam_ax.xaxis.set_ticklabels([])
          #subst_beam_ax.yaxis.tick_right()
          fig_ax = fig.add_subplot(fig_gridspec[:-2, :])
          fig_axs = (fig, fig_ax)
          MWMB_Plot.subplot(trees, 1, 1, titles=["0) !!"+str(round(100*all_neighb_obj_values[-1][0])/100)+"   "+str(round(100*all_subst_obj_values[-1][0])/100)+"!!"], fig_axs=fig_axs, MHs=tree_MHs)
        else:
          nb_y, nb_x = (4,ceil(nb_streams/2))#MWMB_Plot.get_subplot_nb_y_and_nb_x(4+nb_streams)
          neighb_beam_size = nb_x#max(ceil(min(nb_y*nb_x-nb_streams,nb_x)/2),2)
          subst_beam_size = nb_x#max(floor(min(nb_y*nb_x-nb_streams,nb_x)/2),2)
          fig, axs = plt.subplots(nb_y, nb_x, num="Beam Search Progression - Iteration 0", gridspec_kw={"left":0.075, "right":0.925, "bottom":0.075, "top":0.925})
          fig.canvas.manager.resize(800,700)
          if nb_x > 1:
            fig_gridspec = axs[0, 0].get_gridspec()
            for idx_x in range(nb_x):
              for idx_y in range(2,4):
                axs[idx_y,idx_x].remove()
          else:
            fig_gridspec = axs[0].get_gridspec()
            for idx_y in range(2,4):
              axs[idx_y].remove()
          neighb_beam_ax = fig.add_subplot(fig_gridspec[-1, :])
          subst_beam_ax = fig.add_subplot(fig_gridspec[-2, :])
          subst_beam_ax.xaxis.set_ticklabels([])
          if nb_x > 1:
            fig_axs = (fig, [axs[idx_y,idx_x] for idx_y in range(0,2) for idx_x in range(nb_x) if nb_x*idx_y + idx_x < nb_streams])
            if 2*nb_x > nb_best_solutions:
              axs[1,-1].axis('off')
          else:
            fig_axs = (fig, [axs[idx_y] for idx_y in range(0,2) if idx_y < nb_streams])
            if 2*nb_x > nb_best_solutions:
              axs[1].axis('off')
          tree_obj_titles = []
          for idx in range(len(trees)):
            neighb_obj_value = all_neighb_obj_values[-1][idx]
            if neighb_obj_value == best_neighb_obj_values[-1][3]:
              neighb_obj_string = "!!"+str(round(100*neighb_obj_value)/100)
            elif neighb_obj_value >= best_neighb_obj_values[0][3]:
              neighb_obj_string = " !"+str(round(100*neighb_obj_value)/100)
            else:
              neighb_obj_string = "  "+str(round(100*neighb_obj_value)/100)
            subst_obj_value = all_subst_obj_values[-1][idx]
            if subst_obj_value == best_subst_obj_values[-1][3]:
              subst_obj_string = str(round(100*subst_obj_value)/100)+"!!"
            elif subst_obj_value >= best_subst_obj_values[0][3]:
              subst_obj_string = str(round(100*subst_obj_value)/100)+"! "
            else:
              subst_obj_string = str(round(100*subst_obj_value)/100)+"  "
            tree_obj_titles.append(neighb_obj_string+"   "+subst_obj_string)
          MWMB_Plot.subplot(trees, 1, nb_streams, titles=tree_obj_titles, fig_axs=fig_axs, MHs=tree_MHs)
        if check_times:
          neighb_beam_ax.set_xlabel("Time (s)")
        else:
          neighb_beam_ax.set_xlabel("Iterations")
        neighb_beam_ax.set_ylabel(neighborhood_obj_func)
        subst_beam_ax.set_ylabel(substitute_obj_func)
        neighb_beam_ax.scatter([iter_times[0]]*len(trees),[all_neighb_obj_values[0][curr_stream_idx] for curr_stream_idx in range(len(trees))], s=10, c='lightgray')
        valid_subst_obj_values = [subst_obj_value for subst_obj_value in all_subst_obj_values[0] if subst_obj_value > -1]
        subst_beam_ax.scatter([iter_times[0]]*len(valid_subst_obj_values), valid_subst_obj_values, s=2.5, c='k')
        tree_MHs = []
        plt.pause(1e-8)

    
    # BEAM SEARCH LOOP
    for i in range(1,max_iter+1):
      ## Variable Neighborhood Change Criteria / Reset Criteria
      if (neighborhoods != "tree" and neighborhoods != "mesh" and (iter_in_curr_neighborhood == neighb_max_iter or iter_in_curr_neighb_since_new_best == neighb_max_no_better_iter)) or (reset and (neighborhoods == "tree" or neighborhoods == "mesh") and iter_in_curr_neighb_since_new_best == neighb_max_no_better_iter):
        trees = []
        if show_progression or show_best or check_for_duplicates:
          old_subst_obj_values = []
        if neighborhoods != "tree" and neighborhoods != "mesh":
          if neighborhood == "tree":
            neighborhood = "mesh"
          else:# neighborhood == "mesh"
            neighborhood = "tree"
          
          if verbose or saved_log_and_plots_filepath_no_extension != "":
            if check_times:
              MWMB_Plot.print_log(str(i)+" - "+str(neighborhood)+"  (t = "+str(round(1000*(time.time() - total_tic))/1000)+")", verbose, files=full_log_files)
            else:
              MWMB_Plot.print_log(str(i)+" - "+str(neighborhood), verbose, files=full_log_files)

          if (neighborhood == "mesh" and first_mesh_switch) or (neighborhood == "tree" and first_tree_switch):
            if neighborhood == "tree":
              for init_min_spanning_tree_simplex_tuple_list in init_min_spanning_tree_simplex_tuple_lists:
                if len(init_min_spanning_tree_simplex_tuple_list[0]) < nb_nodes:
                  trees.append(ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, init_min_spanning_tree_simplex_tuple_list))
            else:
              if len(init_min_spanning_tree_simplex_tuple_lists) == 1:
                trees.append(ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, init_min_spanning_tree_simplex_tuple_lists[0]))
              else:
                for init_min_spanning_tree_idx in range(1,len(init_min_spanning_tree_simplex_tuple_lists)):
                  trees.append(ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, init_min_spanning_tree_simplex_tuple_lists[init_min_spanning_tree_idx]))
            
            old_iter_last_sol_idx = len(all_subst_obj_values[-1])
            for tree_idx in range(len(trees)):
              tree = trees[tree_idx]
              ### Check for duplicates
              if check_for_duplicates:
                is_duplicate = False
                for idx in range(len(tree_adj_matrices_dup)):
                  old_adj_matrix = tree_adj_matrices_dup[idx]
                  old_iter_stream = iters_streams_dup[idx]
                  if (tree.adj_matrix == old_adj_matrix).all():
                    is_duplicate = True
                    nb_duplicates += 1
                    new_duplicates.append(best_move_idx)

                    neighb_obj_value = all_neighb_obj_values[old_iter_stream[0]][old_iter_stream[1]]
                    if skip_poor_bound:
                      bound_obj_value = -1
                    subst_obj_value = all_subst_obj_values[old_iter_stream[0]][old_iter_stream[1]]
                    # if show_progression:
                    #   tree_MHs.append(-1)
                    if verbose or saved_log_and_plots_filepath_no_extension != "":
                      mesh_sizes = -1
                      simplex_tuple_list = []
                      is_new_best_neighb = False
                      is_new_best_subst = False
                    break
                if not is_duplicate:
                  tree_adj_matrices_dup.append(copy.deepcopy(tree.adj_matrix))
                  iters_streams_dup.append((i-1,old_iter_last_sol_idx + tree_idx))

              if not check_for_duplicates or not is_duplicate:
                if check_times:
                  tic = time.time()
                neighb_obj_value, neighb_net = tree.get_obj_value(get_network = True, obj_func=neighborhood_obj_func, MH=MH)
                if check_times:
                  neighb_obj_times.append(time.time() - tic)
                simplex_tuple_list = tree.get_simplex_tuple_list()
                mesh_sizes = neighb_net.get_mesh_sizes()

                is_new_best_neighb = 0
                if neighb_obj_value > best_neighb_obj_values[0][3]:
                  if neighb_obj_value > best_neighb_obj_values[-1][3]:
                    is_new_best_neighb = 2
                  is_best_neighb_duplicate = False
                  for _,_,_,_,_,_,old_simplex_tuple_list in best_neighb_obj_values:
                    if simplex_tuple_list == old_simplex_tuple_list:
                      is_best_neighb_duplicate = True
                      break
                  if not is_best_neighb_duplicate:
                    if is_new_best_neighb == 0:
                      is_new_best_neighb = 1
                    best_neighb_obj_values.append((neighb_net.A_obj_min, neighb_net.B_obj_min, neighb_net.C_obj_min, neighb_obj_value, i, mesh_sizes, simplex_tuple_list))
                    best_neighb_obj_values.sort(key = lambda Amin_Bmin_Cmin_value_iter_mesh_simplex_hept: Amin_Bmin_Cmin_value_iter_mesh_simplex_hept[3])
                    best_neighb_obj_values = best_neighb_obj_values[-nb_best_solutions:]
                neighb_obj_times.append(time.time() - tic)
                
                if check_times:
                  tic = time.time()
                subst_obj_value, subst_net = tree.get_obj_value(obj_func = substitute_obj_func, get_network = True, MH=MH)
                if check_times:
                  subst_obj_times.append(time.time() - tic)

                is_new_best_subst = 0
                if subst_obj_value > best_subst_obj_values[0][3]:
                  if subst_obj_value > best_subst_obj_values[-1][3]:
                    is_new_best_subst = 2
                    if empty_taboo_new_best:
                      new_taboo_add_lists[-1] = new_taboo_add_lists[-1][-len(dropped):]
                      new_taboo_drop_lists[-1] = new_taboo_drop_lists[-1][-len(added):]
                  is_best_subst_duplicate = False
                  for _,_,_,_,_,_,old_simplex_tuple_list in best_subst_obj_values:
                    if simplex_tuple_list == old_simplex_tuple_list:
                      is_best_subst_duplicate = True
                      break
                  if not is_best_subst_duplicate:
                    if is_new_best_subst == 0:
                      is_new_best_subst = 1
                    iter_since_new_best = 0
                    iter_in_curr_neighb_since_new_best = 0
                    best_subst_obj_values.append((subst_net.A_obj_min, subst_net.B_obj_min, subst_net.C_obj_min, subst_obj_value, i, mesh_sizes, simplex_tuple_list))
                    if check_times:
                      MWMB_Plot.print_log(str((time.time() - total_tic,(subst_net.A_obj_min, subst_net.B_obj_min, subst_net.C_obj_min, subst_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=bests_log_files)
                    else:
                      MWMB_Plot.print_log(str((i-1,(subst_net.A_obj_min, subst_net.B_obj_min, subst_net.C_obj_min, subst_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=bests_log_files)
                    best_subst_obj_values.sort(key = lambda Amin_Bmin_Cmin_value_iter_mesh_simplex_hept: Amin_Bmin_Cmin_value_iter_mesh_simplex_hept[3])
                    best_subst_obj_values = best_subst_obj_values[-max(nb_best_substitutes,nb_streams):]
                
                if search_all_scenarios:
                  if subst_net.A_obj_value > best_subst_A_obj_mins[0][1]:
                    is_best_subst_duplicate = False
                    for _,_,_,_,old_simplex_tuple_list in best_subst_A_obj_mins:
                      if simplex_tuple_list == old_simplex_tuple_list:
                        is_best_subst_duplicate = True
                        break
                    if not is_best_subst_duplicate:
                      best_subst_A_obj_mins.append((subst_net.A_obj_min, subst_net.A_obj_value, i, mesh_sizes, simplex_tuple_list))
                      if check_times:
                        MWMB_Plot.print_log(str((time.time() - total_tic,(subst_net.A_obj_min, subst_net.A_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=A_bests_log_files)
                      else:
                        MWMB_Plot.print_log(str((i-1,(subst_net.A_obj_min, subst_net.A_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=A_bests_log_files)
                      best_subst_A_obj_mins.sort(key = lambda min_value_iter_mesh_simplex_quin: min_value_iter_mesh_simplex_quin[1])
                      best_subst_A_obj_mins = best_subst_A_obj_mins[-nb_best_substitutes:]
                  if subst_net.B_obj_value > best_subst_B_obj_mins[0][1]:
                    is_best_subst_duplicate = False
                    for _,_,_,_,old_simplex_tuple_list in best_subst_B_obj_mins:
                      if simplex_tuple_list == old_simplex_tuple_list:
                        is_best_subst_duplicate = True
                        break
                    if not is_best_subst_duplicate:
                      best_subst_B_obj_mins.append((subst_net.B_obj_min, subst_net.B_obj_value, i, mesh_sizes, simplex_tuple_list))
                      if check_times:
                        MWMB_Plot.print_log(str((time.time() - total_tic,(subst_net.B_obj_min, subst_net.B_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=B_bests_log_files)
                      else:
                        MWMB_Plot.print_log(str((i-1,(subst_net.B_obj_min, subst_net.B_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=B_bests_log_files)
                      best_subst_B_obj_mins.sort(key = lambda min_value_iter_mesh_simplex_quin: min_value_iter_mesh_simplex_quin[1])
                      best_subst_B_obj_mins = best_subst_B_obj_mins[-nb_best_substitutes:]
                  if subst_net.C_obj_value > best_subst_C_obj_mins[0][1]:
                    is_best_subst_duplicate = False
                    for _,_,_,_,old_simplex_tuple_list in best_subst_C_obj_mins:
                      if simplex_tuple_list == old_simplex_tuple_list:
                        is_best_subst_duplicate = True
                        break
                    if not is_best_subst_duplicate:
                      best_subst_C_obj_mins.append((subst_net.C_obj_min, subst_net.C_obj_value, i, mesh_sizes, simplex_tuple_list))
                      if check_times:
                        MWMB_Plot.print_log(str((time.time() - total_tic,(subst_net.C_obj_min, subst_net.C_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=C_bests_log_files)
                      else:
                        MWMB_Plot.print_log(str((i-1,(subst_net.C_obj_min, subst_net.C_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=C_bests_log_files)
                      best_subst_C_obj_mins.sort(key = lambda min_value_iter_mesh_simplex_quin: min_value_iter_mesh_simplex_quin[1])
                      best_subst_C_obj_mins = best_subst_C_obj_mins[-nb_best_substitutes:]

              if show_progression or show_best or check_for_duplicates:
                all_neighb_obj_values[-1].append(neighb_obj_value)
                old_subst_obj_values.append(subst_obj_value)
                if check_for_duplicates:
                  if not is_duplicate:
                    all_subst_obj_values[-1].append(subst_obj_value)
                  else:
                    all_subst_obj_values[-1].append(-1)

              if verbose or saved_log_and_plots_filepath_no_extension != "":
                if is_new_best_neighb == 2:
                  new_best_neighb_string = "!!! "
                elif is_new_best_neighb == 1:
                  new_best_neighb_string = "  ! "
                else:
                  new_best_neighb_string = "    "
                if is_new_best_subst == 2:
                  new_best_subst_string = "!!! "
                elif is_new_best_subst == 1:
                  new_best_subst_string = "  ! "
                else:
                  new_best_subst_string = "    "

                if skip_poor_bound:
                  if check_times:
                    tic = time.time()
                  bound_obj_value = tree.get_obj_value(obj_func=bound_obj_func)
                  if check_times:
                    bound_obj_times.append(time.time() - tic)
                  MWMB_Plot.print_log("  "+str(i)+"\t"+new_best_neighb_string+str(round(100*neighb_obj_value)/100)+" \t"+str(round(100*bound_obj_value)/100)+" \t"+new_best_subst_string+str(round(100*subst_obj_value)/100)+" \t   "+str(simplex_tuple_list)+"      mesh sizes = "+str(mesh_sizes), verbose, files=full_log_files)
                else:
                  MWMB_Plot.print_log("  "+str(i)+"\t"+new_best_neighb_string+str(round(100*neighb_obj_value)/100)+" \t"+new_best_subst_string+str(round(100*subst_obj_value)/100)+" \t   "+str(simplex_tuple_list)+"      mesh sizes = "+str(mesh_sizes), verbose, files=full_log_files)
        else:
          if verbose or saved_log_and_plots_filepath_no_extension != "":
            if check_times:
              MWMB_Plot.print_log(str(i)+"  (t = "+str(round(1000*(time.time() - total_tic))/1000)+")", verbose, files=full_log_files)
            else:
              MWMB_Plot.print_log(i, verbose, files=full_log_files)
          
        if neighborhoods == "tree" or neighborhoods == "mesh" or ((neighborhood != "mesh" or not first_mesh_switch) and (neighborhood != "tree" or not first_tree_switch)):
          for stream_idx in range(len(best_subst_obj_values)-nb_streams,len(best_subst_obj_values)):
            if stream_idx >= 0:
              _, _, _, subst_obj_value, it, mesh_sizes, simplex_tuple_list = best_subst_obj_values[stream_idx]
              trees.append(ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, simplex_tuple_list))
              if check_times:
                tic = time.time()
              neighb_obj_value = trees[-1].get_obj_value(obj_func=neighborhood_obj_func)
              if check_times:
                neighb_obj_times.append(time.time() - tic)
              
              if show_progression or show_best or check_for_duplicates:
                all_neighb_obj_values[-1].append(neighb_obj_value)
                old_subst_obj_values.append(subst_obj_value)
                if check_for_duplicates:
                  all_subst_obj_values[-1].append(-1)

              if verbose or saved_log_and_plots_filepath_no_extension != "":
                if skip_poor_bound:
                  if check_times:
                    tic = time.time()
                  bound_obj_value = trees[-1].get_obj_value(obj_func=bound_obj_func)
                  if check_times:
                    bound_obj_times.append(time.time() - tic)
                  MWMB_Plot.print_log("  "+str(it)+"\t    "+str(round(100*neighb_obj_value)/100)+" \t"+str(round(100*bound_obj_value)/100)+" \t    "+str(round(100*subst_obj_value)/100)+" \t   "+str(simplex_tuple_list)+"      mesh sizes = "+str(mesh_sizes), verbose, files=full_log_files)
                else:
                  MWMB_Plot.print_log("  "+str(it)+"\t    "+str(round(100*neighb_obj_value)/100)+" \t    "+str(round(100*subst_obj_value)/100)+" \t   "+str(simplex_tuple_list)+"      mesh sizes = "+str(mesh_sizes), verbose, files=full_log_files)
        taboo_add_lists = [[] for i in range(len(trees))]
        taboo_drop_lists = [[] for i in range(len(trees))]
        iter_in_curr_neighborhood = 0
        iter_in_curr_neighb_since_new_best = 0
        if first_tree_switch:
          first_tree_switch = False
        if first_mesh_switch:
          first_mesh_switch = False

        if show_progression:
          if nb_streams == 1:
            MWMB_Plot.subplot(trees, 1, 1, titles=[str(i)+")   "+str(round(100*all_neighb_obj_values[-1][len(all_neighb_obj_values[-1])-len(trees)])/100)+"   "+str(round(100*old_subst_obj_values[0])/100)+"  "], fig_axs=fig_axs, MHs=tree_MHs)
          else:
            tree_obj_titles = []
            for idx in range(len(trees)):
              tree_obj_titles.append("  "+str(round(100*all_neighb_obj_values[-1][len(all_neighb_obj_values[-1])-len(trees)+idx])/100)+"   "+str(round(100*old_subst_obj_values[idx])/100)+"  ")
            MWMB_Plot.subplot(trees, 1, nb_streams, titles=tree_obj_titles, fig_axs=fig_axs, MHs=tree_MHs)
          if check_times:
            neighb_beam_ax.scatter([iter_times[i-1]]*len(trees),[all_neighb_obj_values[-1][curr_stream_idx] for curr_stream_idx in range(len(all_neighb_obj_values[-1])-len(trees),len(all_neighb_obj_values[-1]))], s=10, c='lightgray')
          else:
            neighb_beam_ax.scatter([i-1]*len(trees),[all_neighb_obj_values[-1][curr_stream_idx] for curr_stream_idx in range(len(all_neighb_obj_values[-1])-len(trees),len(all_neighb_obj_values[-1]))], s=10, c='lightgray')
          if check_for_duplicates:
            non_dup_evaluated_subst_obj_values = [all_subst_obj_values[-1][idx] for idx in range(len(all_neighb_obj_values[-1])-len(trees),len(all_neighb_obj_values[-1])) if all_subst_obj_values[-1][idx] > -1]
            if check_times:
              subst_beam_ax.scatter([iter_times[i-1]]*len(non_dup_evaluated_subst_obj_values), non_dup_evaluated_subst_obj_values, s=2.5, c='k')
            else:
              subst_beam_ax.scatter([i-1]*len(non_dup_evaluated_subst_obj_values), non_dup_evaluated_subst_obj_values, s=2.5, c='k')
          best_subst_value = max(best_subst_value, max(all_subst_obj_values[-1]))
          tree_MHs = []
          plt.pause(1e-8)

      elif verbose or saved_log_and_plots_filepath_no_extension != "":
        if neighborhoods != "tree" and neighborhoods != "mesh":
          if check_times:
            MWMB_Plot.print_log(str(i)+" - "+str(neighborhood)+"  (t = "+str(round(1000*(time.time() - total_tic))/1000)+")", verbose, files=full_log_files)
          else:
            MWMB_Plot.print_log(str(i)+" - "+str(neighborhood), verbose, files=full_log_files)
        else:
          if check_times:
            MWMB_Plot.print_log(str(i)+"  (t = "+str(round(1000*(time.time() - total_tic))/1000)+")", verbose, files=full_log_files)
          else:
            MWMB_Plot.print_log(i, verbose, files=full_log_files)
      iter_in_curr_neighborhood += 1
      iter_in_curr_neighb_since_new_best += 1

      ## Variable Neighborhood Change Criteria / Reset Criteria
      """if neighborhoods != "tree" and neighborhoods != "mesh":
        if iter_in_curr_neighborhood == neighb_max_iter or iter_in_curr_neighb_since_new_best == neighb_max_no_better_iter:
          if neighborhood == "tree":
            neighborhood = "mesh"
          else:# neighborhood == "mesh"
            neighborhood = "tree"
          
          if verbose or saved_log_and_plots_filepath_no_extension != "":
            if check_times:
              MWMB_Plot.print_log(str(i)+" - "+str(neighborhood)+"  (t = "+str(round(1000*(time.time() - total_tic))/1000)+")", verbose, files=full_log_files)
            else:
              MWMB_Plot.print_log(str(i)+" - "+str(neighborhood), verbose, files=full_log_files)

          trees = []
          if show_progression or show_best or check_for_duplicates:
            old_subst_obj_values = []

          if (neighborhood == "mesh" and first_mesh_switch) or (neighborhood == "tree" and first_tree_switch):
            if neighborhood == "tree":
              for init_min_spanning_tree_simplex_tuple_list in init_min_spanning_tree_simplex_tuple_lists:
                if len(init_min_spanning_tree_simplex_tuple_list[0]) < nb_nodes:
                  trees.append(ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, init_min_spanning_tree_simplex_tuple_list))
              first_tree_switch = False
            else:
              if len(init_min_spanning_tree_simplex_tuple_lists) == 1:
                trees.append(ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, init_min_spanning_tree_simplex_tuple_lists[0]))
              else:
                for init_min_spanning_tree_idx in range(1,len(init_min_spanning_tree_simplex_tuple_lists)):
                  trees.append(ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, init_min_spanning_tree_simplex_tuple_lists[init_min_spanning_tree_idx]))
              first_mesh_switch = False
            
            old_iter_last_sol_idx = len(all_subst_obj_values[-1])
            for tree_idx in range(len(trees)):
              tree = trees[tree_idx]
              ### Check for duplicates
              if check_for_duplicates:
                is_duplicate = False
                for idx in range(len(tree_adj_matrices_dup)):
                  old_adj_matrix = tree_adj_matrices_dup[idx]
                  old_iter_stream = iters_streams_dup[idx]
                  if (tree.adj_matrix == old_adj_matrix).all():
                    is_duplicate = True
                    nb_duplicates += 1
                    new_duplicates.append(best_move_idx)

                    neighb_obj_value = all_neighb_obj_values[old_iter_stream[0]][old_iter_stream[1]]
                    if skip_poor_bound:
                      bound_obj_value = -1
                    subst_obj_value = all_subst_obj_values[old_iter_stream[0]][old_iter_stream[1]]
                    # if show_progression:
                    #   tree_MHs.append(-1)
                    if verbose or saved_log_and_plots_filepath_no_extension != "":
                      mesh_sizes = -1
                      simplex_tuple_list = []
                      is_new_best_neighb = False
                      is_new_best_subst = False
                    break
                if not is_duplicate:
                  tree_adj_matrices_dup.append(copy.deepcopy(tree.adj_matrix))
                  iters_streams_dup.append((i-1,old_iter_last_sol_idx + tree_idx))

              if not check_for_duplicates or not is_duplicate:
                if check_times:
                  tic = time.time()
                neighb_obj_value, neighb_net = tree.get_obj_value(get_network = True, obj_func=neighborhood_obj_func, MH=MH)
                if check_times:
                  neighb_obj_times.append(time.time() - tic)
                simplex_tuple_list = tree.get_simplex_tuple_list()
                mesh_sizes = neighb_net.get_mesh_sizes()

                is_new_best_neighb = 0
                if neighb_obj_value > best_neighb_obj_values[0][3]:
                  if neighb_obj_value > best_neighb_obj_values[-1][3]:
                    is_new_best_neighb = 2
                  is_best_neighb_duplicate = False
                  for _,_,_,_,_,_,old_simplex_tuple_list in best_neighb_obj_values:
                    if simplex_tuple_list == old_simplex_tuple_list:
                      is_best_neighb_duplicate = True
                      break
                  if not is_best_neighb_duplicate:
                    if is_new_best_neighb == 0:
                      is_new_best_neighb = 1
                    best_neighb_obj_values.append((neighb_net.A_obj_min, neighb_net.B_obj_min, neighb_net.C_obj_min, neighb_obj_value, i, mesh_sizes, simplex_tuple_list))
                    best_neighb_obj_values.sort(key = lambda Amin_Bmin_Cmin_value_iter_mesh_simplex_hept: Amin_Bmin_Cmin_value_iter_mesh_simplex_hept[3])
                    best_neighb_obj_values = best_neighb_obj_values[-nb_best_solutions:]
                neighb_obj_times.append(time.time() - tic)
                
                if check_times:
                  tic = time.time()
                subst_obj_value, subst_net = tree.get_obj_value(obj_func = substitute_obj_func, get_network = True, MH=MH)
                if check_times:
                  subst_obj_times.append(time.time() - tic)

                is_new_best_subst = 0
                if subst_obj_value > best_subst_obj_values[0][3]:
                  if subst_obj_value > best_subst_obj_values[-1][3]:
                    is_new_best_subst = 2
                    if empty_taboo_new_best:
                      new_taboo_add_lists[-1] = new_taboo_add_lists[-1][-len(dropped):]
                      new_taboo_drop_lists[-1] = new_taboo_drop_lists[-1][-len(added):]
                  is_best_subst_duplicate = False
                  for _,_,_,_,_,_,old_simplex_tuple_list in best_subst_obj_values:
                    if simplex_tuple_list == old_simplex_tuple_list:
                      is_best_subst_duplicate = True
                      break
                  if not is_best_subst_duplicate:
                    if is_new_best_subst == 0:
                      is_new_best_subst = 1
                    iter_since_new_best = 0
                    iter_in_curr_neighb_since_new_best = 0
                    best_subst_obj_values.append((subst_net.A_obj_min, subst_net.B_obj_min, subst_net.C_obj_min, subst_obj_value, i, mesh_sizes, simplex_tuple_list))
                    MWMB_Plot.print_log(str((time.time() - total_tic,(subst_net.A_obj_min, subst_net.B_obj_min, subst_net.C_obj_min, subst_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=bests_log_files)
                    best_subst_obj_values.sort(key = lambda Amin_Bmin_Cmin_value_iter_mesh_simplex_hept: Amin_Bmin_Cmin_value_iter_mesh_simplex_hept[3])
                    best_subst_obj_values = best_subst_obj_values[-max(nb_best_substitutes,nb_streams):]
                
                if search_all_scenarios:
                  if subst_net.A_obj_value > best_subst_A_obj_mins[0][1]:
                    is_best_subst_duplicate = False
                    for _,_,_,_,old_simplex_tuple_list in best_subst_A_obj_mins:
                      if simplex_tuple_list == old_simplex_tuple_list:
                        is_best_subst_duplicate = True
                        break
                    if not is_best_subst_duplicate:
                      best_subst_A_obj_mins.append((subst_net.A_obj_min, subst_net.A_obj_value, i, mesh_sizes, simplex_tuple_list))
                      MWMB_Plot.print_log(str((time.time() - total_tic,(subst_net.A_obj_min, subst_net.A_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=A_bests_log_files)
                      best_subst_A_obj_mins.sort(key = lambda min_value_iter_mesh_simplex_quin: min_value_iter_mesh_simplex_quin[1])
                      best_subst_A_obj_mins = best_subst_A_obj_mins[-nb_best_substitutes:]
                  if subst_net.B_obj_value > best_subst_B_obj_mins[0][1]:
                    is_best_subst_duplicate = False
                    for _,_,_,_,old_simplex_tuple_list in best_subst_B_obj_mins:
                      if simplex_tuple_list == old_simplex_tuple_list:
                        is_best_subst_duplicate = True
                        break
                    if not is_best_subst_duplicate:
                      best_subst_B_obj_mins.append((subst_net.B_obj_min, subst_net.B_obj_value, i, mesh_sizes, simplex_tuple_list))
                      MWMB_Plot.print_log(str((time.time() - total_tic,(subst_net.B_obj_min, subst_net.B_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=B_bests_log_files)
                      best_subst_B_obj_mins.sort(key = lambda min_value_iter_mesh_simplex_quin: min_value_iter_mesh_simplex_quin[1])
                      best_subst_B_obj_mins = best_subst_B_obj_mins[-nb_best_substitutes:]
                  if subst_net.C_obj_value > best_subst_C_obj_mins[0][1]:
                    is_best_subst_duplicate = False
                    for _,_,_,_,old_simplex_tuple_list in best_subst_C_obj_mins:
                      if simplex_tuple_list == old_simplex_tuple_list:
                        is_best_subst_duplicate = True
                        break
                    if not is_best_subst_duplicate:
                      best_subst_C_obj_mins.append((subst_net.C_obj_min, subst_net.C_obj_value, i, mesh_sizes, simplex_tuple_list))
                      MWMB_Plot.print_log(str((time.time() - total_tic,(subst_net.C_obj_min, subst_net.C_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=C_bests_log_files)
                      best_subst_C_obj_mins.sort(key = lambda min_value_iter_mesh_simplex_quin: min_value_iter_mesh_simplex_quin[1])
                      best_subst_C_obj_mins = best_subst_C_obj_mins[-nb_best_substitutes:]

              if show_progression or show_best or check_for_duplicates:
                all_neighb_obj_values[-1].append(neighb_obj_value)
                old_subst_obj_values.append(subst_obj_value)
                if check_for_duplicates:
                  if not is_duplicate:
                    all_subst_obj_values[-1].append(subst_obj_value)
                  else:
                    all_subst_obj_values[-1].append(-1)

              if verbose or saved_log_and_plots_filepath_no_extension != "":
                if is_new_best_neighb == 2:
                  new_best_neighb_string = "!!! "
                elif is_new_best_neighb == 1:
                  new_best_neighb_string = "  ! "
                else:
                  new_best_neighb_string = "    "
                if is_new_best_subst == 2:
                  new_best_subst_string = "!!! "
                elif is_new_best_subst == 1:
                  new_best_subst_string = "  ! "
                else:
                  new_best_subst_string = "    "

                if skip_poor_bound:
                  if check_times:
                    tic = time.time()
                  bound_obj_value = tree.get_obj_value(obj_func=bound_obj_func)
                  if check_times:
                    bound_obj_times.append(time.time() - tic)
                  MWMB_Plot.print_log("  "+str(i)+"\t"+new_best_neighb_string+str(round(100*neighb_obj_value)/100)+" \t"+str(round(100*bound_obj_value)/100)+" \t"+new_best_subst_string+str(round(100*subst_obj_value)/100)+" \t   "+str(simplex_tuple_list)+"      mesh sizes = "+str(mesh_sizes), verbose, files=full_log_files)
                else:
                  MWMB_Plot.print_log("  "+str(i)+"\t"+new_best_neighb_string+str(round(100*neighb_obj_value)/100)+" \t"+new_best_subst_string+str(round(100*subst_obj_value)/100)+" \t   "+str(simplex_tuple_list)+"      mesh sizes = "+str(mesh_sizes), verbose, files=full_log_files)
          else:
            for stream_idx in range(len(best_subst_obj_values)-nb_streams,len(best_subst_obj_values)):
              if stream_idx >= 0:
                _, _, _, subst_obj_value, it, mesh_sizes, simplex_tuple_list = best_subst_obj_values[stream_idx]
                trees.append(ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, simplex_tuple_list))
                if check_times:
                  tic = time.time()
                neighb_obj_value = trees[-1].get_obj_value(obj_func=neighborhood_obj_func)
                if check_times:
                  neighb_obj_times.append(time.time() - tic)
                
                if show_progression or show_best or check_for_duplicates:
                  all_neighb_obj_values[-1].append(neighb_obj_value)
                  old_subst_obj_values.append(subst_obj_value)
                  if check_for_duplicates:
                    all_subst_obj_values[-1].append(-1)

                if verbose or saved_log_and_plots_filepath_no_extension != "":
                  if skip_poor_bound:
                    if check_times:
                      tic = time.time()
                    bound_obj_value = trees[-1].get_obj_value(obj_func=bound_obj_func)
                    if check_times:
                      bound_obj_times.append(time.time() - tic)
                    MWMB_Plot.print_log("  "+str(it)+"\t    "+str(round(100*neighb_obj_value)/100)+" \t"+str(round(100*bound_obj_value)/100)+" \t    "+str(round(100*subst_obj_value)/100)+" \t   "+str(simplex_tuple_list)+"      mesh sizes = "+str(mesh_sizes), verbose, files=full_log_files)
                  else:
                    MWMB_Plot.print_log("  "+str(it)+"\t    "+str(round(100*neighb_obj_value)/100)+" \t    "+str(round(100*subst_obj_value)/100)+" \t   "+str(simplex_tuple_list)+"      mesh sizes = "+str(mesh_sizes), verbose, files=full_log_files)
          taboo_add_lists = [[] for i in range(len(trees))]
          taboo_drop_lists = [[] for i in range(len(trees))]
          iter_in_curr_neighborhood = 0
          iter_in_curr_neighb_since_new_best = 0

          if show_progression:
            if nb_streams == 1:
              MWMB_Plot.subplot(trees, 1, 1, titles=[str(i)+")   "+str(round(100*all_neighb_obj_values[-1][len(all_neighb_obj_values[-1])-len(trees)])/100)+"   "+str(round(100*old_subst_obj_values[0])/100)+"  "], fig_axs=fig_axs, MHs=tree_MHs)
            else:
              tree_obj_titles = []
              for idx in range(len(trees)):
                tree_obj_titles.append("  "+str(round(100*all_neighb_obj_values[-1][len(all_neighb_obj_values[-1])-len(trees)+idx])/100)+"   "+str(round(100*old_subst_obj_values[idx])/100)+"  ")
              MWMB_Plot.subplot(trees, 1, nb_streams, titles=tree_obj_titles, fig_axs=fig_axs, MHs=tree_MHs)
            if check_times:
              neighb_beam_ax.scatter([iter_times[i-1]]*len(trees),[all_neighb_obj_values[-1][curr_stream_idx] for curr_stream_idx in range(len(all_neighb_obj_values[-1])-len(trees),len(all_neighb_obj_values[-1]))], s=10, c='lightgray')
            else:
              neighb_beam_ax.scatter([i-1]*len(trees),[all_neighb_obj_values[-1][curr_stream_idx] for curr_stream_idx in range(len(all_neighb_obj_values[-1])-len(trees),len(all_neighb_obj_values[-1]))], s=10, c='lightgray')
            if check_for_duplicates:
              non_dup_evaluated_subst_obj_values = [all_subst_obj_values[-1][idx] for idx in range(len(all_neighb_obj_values[-1])-len(trees),len(all_neighb_obj_values[-1])) if all_subst_obj_values[-1][idx] > -1]
              if check_times:
                subst_beam_ax.scatter([iter_times[i-1]]*len(non_dup_evaluated_subst_obj_values), non_dup_evaluated_subst_obj_values, s=2.5, c='k')
              else:
                subst_beam_ax.scatter([i-1]*len(non_dup_evaluated_subst_obj_values), non_dup_evaluated_subst_obj_values, s=2.5, c='k')
            best_subst_value = max(best_subst_value, max(all_subst_obj_values[-1]))
            tree_MHs = []
            plt.pause(1e-8)

        elif verbose or saved_log_and_plots_filepath_no_extension != "":
          if check_times:
            MWMB_Plot.print_log(str(i)+" - "+str(neighborhood)+"  (t = "+str(round(1000*(time.time() - total_tic))/1000)+")", verbose, files=full_log_files)
          else:
            MWMB_Plot.print_log(str(i)+" - "+str(neighborhood), verbose, files=full_log_files)
        iter_in_curr_neighborhood += 1
        iter_in_curr_neighb_since_new_best += 1
      elif verbose or saved_log_and_plots_filepath_no_extension != "":
        if check_times:
          MWMB_Plot.print_log(str(i)+"  (t = "+str(round(1000*(time.time() - total_tic))/1000)+")", verbose, files=full_log_files)
        else:
          MWMB_Plot.print_log(i, verbose, files=full_log_files)"""

      if show_progression or show_best or check_for_duplicates:
        all_neighb_obj_values.append([])
        all_subst_obj_values.append([])

      ## Neighborhood Exploration (for all current trees)
      best_tree_move_value_triples = []
      for tree_idx in range(len(trees)):
        if verbose or saved_log_and_plots_filepath_no_extension != "":
          if len(trees) == 1:
            MWMB_Plot.print_log("  taboo_add = "+str(taboo_add_lists[tree_idx]), verbose, files=full_log_files)
            MWMB_Plot.print_log("  taboo_drop = "+str(taboo_drop_lists[tree_idx]), verbose, files=full_log_files)
          else:
            MWMB_Plot.print_log("  "+str(tree_idx)+"       taboo_add = "+str(taboo_add_lists[tree_idx])+"   taboo_drop = "+str(taboo_drop_lists[tree_idx]), verbose, files=full_log_files)
        if check_times:
          tic = time.time()
        curr_tree_best_move_value_taboo_triples, visited_all_neighbors = trees[tree_idx].get_best_moves(nb_best_moves=nb_streams, neighborhood=neighborhood, taboo_lists=[taboo_drop_lists[tree_idx],taboo_add_lists[tree_idx]], obj_func=neighborhood_obj_func, best_neighb_obj_value=best_neighb_obj_values[-1][3], MH=MH, first_better=first_better, neighborhood_subset_ratio=neighborhood_subset_ratio, ordered_edges=ordered_edges, use_elite_edge_filter=True, elite_edge_filter_matrix=elite_edge_filter_matrix, mesh_compatibility_matrix=mesh_compatibility_matrix, ordered_cluster_lists=ordered_cluster_lists, ordered_cluster_adj_matrices=ordered_cluster_adj_matrices, verbose=verbose, log_files=full_log_files)
        # ..., use_elite_edge_filter=elite_edge_filter_has_stabilized, ...
        if check_times:
          if neighborhood == "tree":
            tree_neighb_search_times.append(time.time() - tic)
          else:
            mesh_neighb_search_times.append(time.time() - tic)
        if first_better:
          if neighborhood == "tree":
            nb_tree_neighbor_search += 1
            if visited_all_neighbors:
              nb_tree_visited_all_neighbors += 1
          else:
            nb_mesh_neighbor_search += 1
            if visited_all_neighbors:
              nb_mesh_visited_all_neighbors += 1
        best_tree_move_value_triples.extend([(tree_idx, tree_best_move_value_taboo_triple[0], tree_best_move_value_taboo_triple[1]) for tree_best_move_value_taboo_triple in curr_tree_best_move_value_taboo_triples])

      ## Best Moves Ordering & Duplicates Detection
      if len(best_tree_move_value_triples) > 1:
        best_tree_move_value_triples.sort(key = lambda best_tree_move_value_triple: best_tree_move_value_triple[2])
        new_solution_matrices = []
        new_best_move_triples = []
        new_stream_lineages = []
        for inv_best_move_idx in range(len(best_tree_move_value_triples)):
          best_move_triple = best_tree_move_value_triples[len(best_tree_move_value_triples) - 1 - inv_best_move_idx]
          tree_idx, best_move, best_move_value = best_move_triple
          new_solution_matrix = copy.deepcopy(trees[tree_idx].adj_matrix)
          ImpTree.make_adj_matrix_move(new_solution_matrix, best_move, neighborhood=neighborhood)
          is_duplicate = False
          for new_solution_matrix_idx in range(len(new_solution_matrices)):
            if (new_solution_matrix == new_solution_matrices[new_solution_matrix_idx]).all():
              is_duplicate = True
              new_stream_lineages[new_solution_matrix_idx].append(tree_idx+(len(all_neighb_obj_values[i-1])-len(trees)))
              break
          if not is_duplicate:
            if len(new_solution_matrices) == nb_streams:
              break
            new_solution_matrices.insert(0, new_solution_matrix)
            new_best_move_triples.insert(0, best_move_triple)
            new_stream_lineages.insert(0, [tree_idx+(len(all_neighb_obj_values[i-1])-len(trees))])
        best_tree_move_value_triples = new_best_move_triples
        if show_progression or show_best:
          stream_lineages.append(new_stream_lineages)
      elif show_progression or show_best:
          stream_lineages.append([[0]])
      
      ## Number of Used Moves (for all current trees)
      if len(best_tree_move_value_triples) > 1:
        nbs_desc_old_trees = [0 for tree_idx in range(len(trees))]
        for stream_ancestors in new_stream_lineages:
          for ancestor in stream_ancestors:
            nbs_desc_old_trees[ancestor-(len(all_neighb_obj_values[i-1])-len(trees))] += 1
        if verbose or saved_log_and_plots_filepath_no_extension != "":
          MWMB_Plot.print_log("Number of descendants per old tree : ", verbose, files=full_log_files)
          MWMB_Plot.print_log(nbs_desc_old_trees, verbose, files=full_log_files)

      ## Creation of the New Trees & Taboo Lists
      new_trees = []
      new_taboo_add_lists = []
      new_taboo_drop_lists = []
      if check_for_duplicates:
        new_duplicates = []
      for best_move_idx in range(len(best_tree_move_value_triples)):

        ### Creation of the New Tree & Taboo Lists
        tree_idx, best_move, best_move_value = best_tree_move_value_triples[best_move_idx]
        if len(best_tree_move_value_triples) > 1 and nbs_desc_old_trees[tree_idx] > 1:
          new_tree = copy.deepcopy(trees[tree_idx])
          new_taboo_add_list = copy.deepcopy(taboo_add_lists[tree_idx])
          new_taboo_drop_list = copy.deepcopy(taboo_drop_lists[tree_idx])
          nbs_desc_old_trees[tree_idx] -= 1
        else:
          new_tree = trees[tree_idx]
          new_taboo_add_list = taboo_add_lists[tree_idx]
          new_taboo_drop_list = taboo_drop_lists[tree_idx]
        dropped, added = new_tree.make_move(neighborhood=neighborhood, move=best_move)
        new_trees.append(new_tree)

        new_taboo_add_list += dropped
        new_taboo_drop_list += added
        mesh_sizes = new_tree.get_mesh_sizes()
        taboo_add_list_length = ImpTree.get_taboo_add_list_length(neighborhood, nb_nodes, mesh_sizes, len(ordered_edges))
        taboo_drop_list_length = ImpTree.get_taboo_drop_list_length(neighborhood, nb_nodes, mesh_sizes)
        if taboo_add_list_length > taboo_drop_list_length:
          curr_taboo_add_list_length =  taboo_add_list_length + random.randint(-1,1)
          curr_taboo_drop_list_length = taboo_drop_list_length
        else:
          curr_taboo_add_list_length = taboo_add_list_length
          curr_taboo_drop_list_length = taboo_drop_list_length + random.randint(-1,1)
        new_taboo_add_list = new_taboo_add_list[-curr_taboo_add_list_length:]
        new_taboo_drop_list = new_taboo_drop_list[-curr_taboo_drop_list_length:]
        new_taboo_add_lists.append(new_taboo_add_list)
        new_taboo_drop_lists.append(new_taboo_drop_list)


        ### Check for duplicates
        if check_for_duplicates:
          is_duplicate = False
          for idx in range(len(tree_adj_matrices_dup)):
            old_adj_matrix = tree_adj_matrices_dup[idx]
            old_iter_stream = iters_streams_dup[idx]
            if (new_tree.adj_matrix == old_adj_matrix).all():
              is_duplicate = True
              nb_duplicates += 1
              new_duplicates.append(best_move_idx)

              neighb_obj_value = all_neighb_obj_values[old_iter_stream[0]][old_iter_stream[1]]
              if skip_poor_bound:
                bound_obj_value = -1
              subst_obj_value = all_subst_obj_values[old_iter_stream[0]][old_iter_stream[1]]
              if show_progression:
                tree_MHs.append(-1)
              if verbose or saved_log_and_plots_filepath_no_extension != "":
                mesh_sizes = -1
                simplex_tuple_list = []
                is_new_best_neighb = False
                is_new_best_subst = False
              break
          if not is_duplicate:
            tree_adj_matrices_dup.append(copy.deepcopy(new_tree.adj_matrix))
            iters_streams_dup.append((i,best_move_idx))


        if not check_for_duplicates or not is_duplicate:

          ### New Tree Neighborhood Obj. Value
          if check_times:
            tic = time.time()
          neighb_obj_value, neighb_net = new_tree.get_obj_value(get_network = True, obj_func=neighborhood_obj_func, MH=MH)
          if check_times:
            neighb_obj_times.append(time.time() - tic)
          simplex_tuple_list = new_tree.get_simplex_tuple_list()
          mesh_sizes = neighb_net.get_mesh_sizes()

          is_new_best_neighb = 0
          if neighb_obj_value > best_neighb_obj_values[0][3]:
            if neighb_obj_value > best_neighb_obj_values[-1][3]:
              is_new_best_neighb = 2
            is_best_neighb_duplicate = False
            for _,_,_,_,_,_,old_simplex_tuple_list in best_neighb_obj_values:
              if simplex_tuple_list == old_simplex_tuple_list:
                is_best_neighb_duplicate = True
                break
            if not is_best_neighb_duplicate:
              if is_new_best_neighb == 0:
                is_new_best_neighb = 1
              best_neighb_obj_values.append((neighb_net.A_obj_min, neighb_net.B_obj_min, neighb_net.C_obj_min, neighb_obj_value, i, mesh_sizes, simplex_tuple_list))
              best_neighb_obj_values.sort(key = lambda Amin_Bmin_Cmin_value_iter_mesh_simplex_hept: Amin_Bmin_Cmin_value_iter_mesh_simplex_hept[3])
              best_neighb_obj_values = best_neighb_obj_values[-nb_best_solutions:]

          ### New Tree Bound Obj. Value
          if skip_poor_bound:
            if check_times:
              tic = time.time()
            bound_obj_value, bound_net = new_tree.get_obj_value(get_network = True, obj_func=bound_obj_func, MH=MH)
            if check_times:
              bound_obj_times.append(time.time() - tic)
            adjusted_bound_obj_value = max(neighb_obj_value, bound_obj_value)
          
          ### New Tree Substitute Obj. Value
          if not skip_poor_bound or adjusted_bound_obj_value > best_subst_obj_values[0][3] or (search_all_scenarios and (bound_net.A_obj_value > best_subst_A_obj_mins[0][1] or bound_net.B_obj_value > best_subst_B_obj_mins[0][1] or bound_net.C_obj_value > best_subst_C_obj_mins[0][1])):# or is_new_best_neighb:
            if check_times:
              tic = time.time()
            subst_obj_value, subst_net = new_tree.get_obj_value(obj_func = substitute_obj_func, get_network = True, MH=MH)
            if check_times:
              subst_obj_times.append(time.time() - tic)
            
            is_new_best_subst = 0
            if subst_obj_value > best_subst_obj_values[0][3]:
              if subst_obj_value > best_subst_obj_values[-1][3]:
                is_new_best_subst = 2
                if empty_taboo_new_best:
                  new_taboo_add_lists[-1] = new_taboo_add_lists[-1][-len(dropped):]
                  new_taboo_drop_lists[-1] = new_taboo_drop_lists[-1][-len(added):]
              is_best_subst_duplicate = False
              for _,_,_,_,_,_,old_simplex_tuple_list in best_subst_obj_values:
                if simplex_tuple_list == old_simplex_tuple_list:
                  is_best_subst_duplicate = True
                  break
              if not is_best_subst_duplicate:
                if is_new_best_subst == 0:
                  is_new_best_subst = 1
                iter_since_new_best = 0
                iter_in_curr_neighb_since_new_best = 0
                best_subst_obj_values.append((subst_net.A_obj_min, subst_net.B_obj_min, subst_net.C_obj_min, subst_obj_value, i, mesh_sizes, simplex_tuple_list))
                if check_times:
                  MWMB_Plot.print_log(str((time.time() - total_tic,(subst_net.A_obj_min, subst_net.B_obj_min, subst_net.C_obj_min, subst_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=bests_log_files)
                else:
                  MWMB_Plot.print_log(str((i,(subst_net.A_obj_min, subst_net.B_obj_min, subst_net.C_obj_min, subst_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=bests_log_files)
                best_subst_obj_values.sort(key = lambda Amin_Bmin_Cmin_value_iter_mesh_simplex_hept: Amin_Bmin_Cmin_value_iter_mesh_simplex_hept[3])
                best_subst_obj_values = best_subst_obj_values[-max(nb_best_substitutes,nb_streams):]
            
            if search_all_scenarios:
              if subst_net.A_obj_value > best_subst_A_obj_mins[0][1]:
                is_best_subst_duplicate = False
                for _,_,_,_,old_simplex_tuple_list in best_subst_A_obj_mins:
                  if simplex_tuple_list == old_simplex_tuple_list:
                    is_best_subst_duplicate = True
                    break
                if not is_best_subst_duplicate:
                  best_subst_A_obj_mins.append((subst_net.A_obj_min, subst_net.A_obj_value, i, mesh_sizes, simplex_tuple_list))
                  if check_times:
                    MWMB_Plot.print_log(str((time.time() - total_tic,(subst_net.A_obj_min, subst_net.A_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=A_bests_log_files)
                  else:
                    MWMB_Plot.print_log(str((i,(subst_net.A_obj_min, subst_net.A_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=A_bests_log_files)
                  best_subst_A_obj_mins.sort(key = lambda min_value_iter_mesh_simplex_quin: min_value_iter_mesh_simplex_quin[1])
                  best_subst_A_obj_mins = best_subst_A_obj_mins[-nb_best_substitutes:]
              if subst_net.B_obj_value > best_subst_B_obj_mins[0][1]:
                is_best_subst_duplicate = False
                for _,_,_,_,old_simplex_tuple_list in best_subst_B_obj_mins:
                  if simplex_tuple_list == old_simplex_tuple_list:
                    is_best_subst_duplicate = True
                    break
                if not is_best_subst_duplicate:
                  best_subst_B_obj_mins.append((subst_net.B_obj_min, subst_net.B_obj_value, i, mesh_sizes, simplex_tuple_list))
                  if check_times:
                    MWMB_Plot.print_log(str((time.time() - total_tic,(subst_net.B_obj_min, subst_net.B_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=B_bests_log_files)
                  else:
                    MWMB_Plot.print_log(str((i,(subst_net.B_obj_min, subst_net.B_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=B_bests_log_files)
                  best_subst_B_obj_mins.sort(key = lambda min_value_iter_mesh_simplex_quin: min_value_iter_mesh_simplex_quin[1])
                  best_subst_B_obj_mins = best_subst_B_obj_mins[-nb_best_substitutes:]
              if subst_net.C_obj_value > best_subst_C_obj_mins[0][1]:
                is_best_subst_duplicate = False
                for _,_,_,_,old_simplex_tuple_list in best_subst_C_obj_mins:
                  if simplex_tuple_list == old_simplex_tuple_list:
                    is_best_subst_duplicate = True
                    break
                if not is_best_subst_duplicate:
                  best_subst_C_obj_mins.append((subst_net.C_obj_min, subst_net.C_obj_value, i, mesh_sizes, simplex_tuple_list))
                  if check_times:
                    MWMB_Plot.print_log(str((time.time() - total_tic,(subst_net.C_obj_min, subst_net.C_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=C_bests_log_files)
                  else:
                    MWMB_Plot.print_log(str((i,(subst_net.C_obj_min, subst_net.C_obj_value, i, mesh_sizes, simplex_tuple_list))), False, files=C_bests_log_files)
                  best_subst_C_obj_mins.sort(key = lambda min_value_iter_mesh_simplex_quin: min_value_iter_mesh_simplex_quin[1])
                  best_subst_C_obj_mins = best_subst_C_obj_mins[-nb_best_substitutes:]
          else:
            subst_obj_value = -1
            is_new_best_subst = False
          
          if show_progression:
            if subst_obj_value > -1:
              tree_MHs.append(subst_net.imp_arb.MH)
            else:
              tree_MHs.append(neighb_net.imp_arb.MH)
        
        
        if show_progression or show_best or check_for_duplicates:
          all_neighb_obj_values[-1].append(neighb_obj_value)
          all_subst_obj_values[-1].append(subst_obj_value)


        if verbose or saved_log_and_plots_filepath_no_extension != "":
          if is_new_best_neighb == 2:
            new_best_neighb_string = "!!! "
          elif is_new_best_neighb == 1:
            new_best_neighb_string = "  ! "
          else:
            new_best_neighb_string = "    "
          if is_new_best_subst == 2:
            new_best_subst_string = "!!! "
          elif is_new_best_subst == 1:
            new_best_subst_string = "  ! "
          else:
            new_best_subst_string = "    "
          if len(best_tree_move_value_triples) == 1:
            MWMB_Plot.print_log("      "+str(simplex_tuple_list)+"      mesh sizes = "+str(mesh_sizes), verbose, files=full_log_files)
            MWMB_Plot.print_log("         "+new_best_neighb_string+str(neighb_obj_value), verbose, files=full_log_files)
            if skip_poor_bound:
              MWMB_Plot.print_log("           "+str(bound_obj_value), verbose, files=full_log_files)
            if subst_obj_value > -1:
              MWMB_Plot.print_log("    "+new_best_subst_string+str(subst_obj_value), verbose, files=full_log_files)
          else:
            if skip_poor_bound:
              MWMB_Plot.print_log("  "+str(tree_idx)+"\t"+new_best_neighb_string+str(round(100*neighb_obj_value)/100)+" \t"+str(round(100*bound_obj_value)/100)+" \t"+new_best_subst_string+str(round(100*subst_obj_value)/100)+" \t   "+str(simplex_tuple_list)+"      mesh sizes = "+str(mesh_sizes), verbose, files=full_log_files)
            else:
              MWMB_Plot.print_log("  "+str(tree_idx)+"\t"+new_best_neighb_string+str(round(100*neighb_obj_value)/100)+" \t"+new_best_subst_string+str(round(100*subst_obj_value)/100)+" \t   "+str(simplex_tuple_list)+"      mesh sizes = "+str(mesh_sizes), verbose, files=full_log_files)

      if show_all:
        MWMB_Plot.subplot(trees+new_trees, 4, 5, [i-1 for idx in range(len(trees))]+[i for idx in range(len(new_trees))])
      
      trees = new_trees
      taboo_add_lists = new_taboo_add_lists
      taboo_drop_lists = new_taboo_drop_lists

      if check_times:
        iter_times.append(time.time() - total_tic)
      else:
        iter_times.append(i)
      if show_progression and len(all_neighb_obj_values[-1]) > 0 and len(all_subst_obj_values[-1]) > 0:
        old_best_subst_value = best_subst_value
        best_subst_value = max(best_subst_value, max(all_subst_obj_values[-1]))
        if nb_streams == 1:
          fig.canvas.manager.set_window_title("Taboo Search Progression - Iteration "+str(i))
          neighb_obj_value = all_neighb_obj_values[-1][0]
          if neighb_obj_value == best_neighb_obj_values[-1][3]:
            neighb_obj_string = "!!"+str(round(100*neighb_obj_value)/100)
          elif neighb_obj_value >= best_neighb_obj_values[0][3]:
            neighb_obj_string = " !"+str(round(100*neighb_obj_value)/100)
          else:
            neighb_obj_string = "  "+str(round(100*neighb_obj_value)/100)
          subst_obj_value = all_subst_obj_values[-1][0]
          if subst_obj_value == best_subst_obj_values[-1][3]:
            subst_obj_string = str(round(100*subst_obj_value)/100)+"!!"
          elif subst_obj_value >= best_subst_obj_values[0][3]:
            subst_obj_string = str(round(100*subst_obj_value)/100)+"! "
          else:
            subst_obj_string = str(round(100*subst_obj_value)/100)+"  "
          MWMB_Plot.subplot(trees, 1, 1, titles=[str(i)+") "+neighb_obj_string+"   "+subst_obj_string], fig_axs=fig_axs, MHs=tree_MHs)
        else:
          fig.canvas.manager.set_window_title("Beam Search Progression - Iteration "+str(i))
          tree_obj_titles = []
          for idx in range(len(trees)):
            neighb_obj_value = all_neighb_obj_values[-1][idx]
            if neighb_obj_value == best_neighb_obj_values[-1][3]:
              neighb_obj_string = "!!"+str(round(100*neighb_obj_value)/100)
            elif neighb_obj_value >= best_neighb_obj_values[0][3]:
              neighb_obj_string = " !"+str(round(100*neighb_obj_value)/100)
            else:
              neighb_obj_string = "  "+str(round(100*neighb_obj_value)/100)
            subst_obj_value = all_subst_obj_values[-1][idx]
            if subst_obj_value == best_subst_obj_values[-1][3]:
              subst_obj_string = str(round(100*subst_obj_value)/100)+"!!"
            elif subst_obj_value >= best_subst_obj_values[0][3]:
              subst_obj_string = str(round(100*subst_obj_value)/100)+"! "
            else:
              subst_obj_string = str(round(100*subst_obj_value)/100)+"  "
            tree_obj_titles.append(neighb_obj_string+"   "+subst_obj_string)
          MWMB_Plot.subplot(trees, 1, nb_streams, titles=tree_obj_titles, fig_axs=fig_axs, MHs=tree_MHs)
        if len(all_neighb_obj_values[i-1]) > 0:
          for curr_stream_idx in range(len(stream_lineages[-1])):
            curr_stream_lineage = stream_lineages[-1][curr_stream_idx]
            for curr_stream_ancestor in curr_stream_lineage:
              neighb_beam_ax.plot([iter_times[i-1],iter_times[i]],[all_neighb_obj_values[i-1][curr_stream_ancestor],all_neighb_obj_values[i][curr_stream_idx]], ls='-', lw=0.5, c='k')
        neighb_beam_ax.scatter([iter_times[i]]*len(stream_lineages[i-1]),[all_neighb_obj_values[i][curr_stream_idx] for curr_stream_idx in range(len(stream_lineages[i-1]))], s=10, c='lightgray')
        if check_for_duplicates:
          evaluated_subst_idcs = [idx for idx, subst_obj_value in enumerate(all_subst_obj_values[i]) if subst_obj_value > -1]
          non_dup_evaluated_subst_obj_values = [all_subst_obj_values[i][idx] for idx in (set(evaluated_subst_idcs) - set(new_duplicates))]
          dup_evaluated_subst_obj_values = [all_subst_obj_values[i][idx] for idx in (set(evaluated_subst_idcs) & set(new_duplicates))]
          subst_beam_ax.scatter([iter_times[i]]*len(dup_evaluated_subst_obj_values), dup_evaluated_subst_obj_values, s=2.5, c='gray')
          subst_beam_ax.scatter([iter_times[i]]*len(non_dup_evaluated_subst_obj_values), non_dup_evaluated_subst_obj_values, s=2.5, c='k')
        else:
          evaluated_subst_obj_values = [subst_obj_value for subst_obj_value in all_subst_obj_values[i] if subst_obj_value > -1]
          subst_beam_ax.scatter([iter_times[i]]*len(evaluated_subst_obj_values), evaluated_subst_obj_values, s=2.5, c='k')
        subst_beam_ax.plot([iter_times[i-1],iter_times[i]],[old_best_subst_value,best_subst_value], ls='-', lw=0.5, c='r')
        ax_xlim = neighb_beam_ax.get_xlim()
        subst_beam_ax.set_xlim([ax_xlim[0], ax_xlim[1]])
        tree_MHs = []
        plt.pause(1e-8)
           
      ## Stopping Criteria
      iter_since_new_best += 1
      if iter_since_new_best == max_no_better_iter:
        break
      if max_time > 0 and iter_times[-1] >= max_time:
        break


    # PROGRESSION PLOT SAVING
    if True:
      if show_progression:
        if nb_streams == 1:
          MWMB_Plot.subplot([ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, best_subst_obj_values[-1][6])], 1, 1, titles=[str(best_subst_obj_values[-1][4])+")   "+str(round(100*best_subst_obj_values[-1][3])/100)+"  "], fig_axs=fig_axs, MHs=tree_MHs)
        else:
          trees = []
          tree_obj_titles = []
          for stream_idx in range(len(best_subst_obj_values)-nb_streams,len(best_subst_obj_values)):
            if stream_idx >= 0:
              _, _, _, subst_obj_value, it, mesh_sizes, simplex_tuple_list = best_subst_obj_values[stream_idx]
              trees.append(ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, simplex_tuple_list))
              tree_obj_titles.append(str(it)+")   "+str(round(100*subst_obj_value)/100)+"  ")
          MWMB_Plot.subplot(trees, 1, nb_streams, titles=tree_obj_titles, fig_axs=fig_axs, MHs=tree_MHs)
        tree_MHs = []
        plt.pause(1e-8)
        plt.ioff()
        if saved_log_and_plots_filepath_no_extension != "":
          if fixed_master_hub:
            #plt.savefig(inst_label+"_"+str(MH)+".eps", dpi=300)
            plt.savefig(saved_log_and_plots_filepath_no_extension+"_"+str(MH)+".png", dpi=300)
          else:
            #plt.savefig(inst_label+".eps", dpi=300)
            plt.savefig(saved_log_and_plots_filepath_no_extension+".png", dpi=300)
      elif show_best:
        plt.ion()
        if nb_streams == 1:
          nb_y, nb_x = (5,5)
        else:
          nb_y, nb_x = (4,max(2,ceil(nb_best_solutions/2)))
        fig, axs = plt.subplots(nb_y, nb_x, num="Beam Search Best Solutions", gridspec_kw={"left":0.075, "right":0.925, "bottom":0.075, "top":0.925})
        if nb_streams == 1:
          fig.canvas.manager.resize(800,600)
        else:
          fig.canvas.manager.resize(800,700)
        if nb_streams == 1:
          fig_gridspec = axs[0, 0].get_gridspec()
          for idx_y in range(nb_y):
            for idx_x in range(nb_x):
              axs[idx_y,idx_x].remove()
        else:
          if nb_x > 1:
            fig_gridspec = axs[0, 0].get_gridspec()
            for idx_x in range(nb_x):
              for idx_y in range(2,4):
                axs[idx_y,idx_x].remove()
          else:
            fig_gridspec = axs[0].get_gridspec()
            for idx_y in range(2,4):
              axs[idx_y].remove()
        neighb_beam_ax = fig.add_subplot(fig_gridspec[-1, :])
        subst_beam_ax = fig.add_subplot(fig_gridspec[-2, :])
        if check_times:
          neighb_beam_ax.set_xlabel("Time (s)")
        else:
          neighb_beam_ax.set_xlabel("Iterations")
        neighb_beam_ax.set_ylabel(neighborhood_obj_func)
        subst_beam_ax.set_ylabel(substitute_obj_func)
        subst_beam_ax.xaxis.set_ticklabels([])
        if nb_streams == 1:
          fig_ax = fig.add_subplot(fig_gridspec[:-2, :])
          fig_axs = (fig, fig_ax)
          _, _, _, subst_obj_value, it, mesh_sizes, simplex_tuple_list = best_subst_obj_values[-1]
          trees = [ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, simplex_tuple_list)]
          tree_obj_titles = [str(it)+")   "+str(round(100*subst_obj_value)/100)+"  "]
          MWMB_Plot.subplot(trees, 1, 1, titles=tree_obj_titles, fig_axs=fig_axs, MHs=[])
        else:
          if nb_x > 1:
            fig_axs = (fig, [axs[idx_y,idx_x] for idx_y in range(0,2) for idx_x in range(nb_x) if nb_x*idx_y + idx_x < nb_best_solutions])
            if 2*nb_x > nb_best_solutions:
              axs[1,-1].axis('off')
          else:
            fig_axs = (fig, [axs[idx_y] for idx_y in range(0,2) if idx_y < nb_best_solutions])
            if 2*nb_x > nb_best_solutions:
              axs[1].axis('off')
          trees = []
          tree_obj_titles = []
          for stream_idx in range(len(best_subst_obj_values)-nb_best_solutions,len(best_subst_obj_values)):
            if stream_idx >= 0:
              _, _, _, subst_obj_value, it, mesh_sizes, simplex_tuple_list = best_subst_obj_values[stream_idx]
              trees.append(ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, simplex_tuple_list))
              tree_obj_titles.append(str(it)+")   "+str(round(100*subst_obj_value)/100)+"  ")
          MWMB_Plot.subplot(trees, 1, len(fig_axs[1]), titles=tree_obj_titles, fig_axs=fig_axs, MHs=[])
        best_subst_value = 0
        for j in range(i):
          old_best_subst_value = best_subst_value
          if len(all_neighb_obj_values[j]) > 0 and len(all_subst_obj_values[j]) > 0:
            best_subst_value = max(best_subst_value, max(all_subst_obj_values[j]))
            if j == 0:
              if check_times:
                neighb_beam_ax.scatter([iter_times[0]]*len(all_neighb_obj_values[0]),all_neighb_obj_values[0], s=10, c='lightgray')
                subst_beam_ax.scatter([iter_times[0]]*len(all_subst_obj_values[0]), all_subst_obj_values[0], s=2.5, c='k')
              else:
                neighb_beam_ax.scatter([0]*len(all_neighb_obj_values[0]),all_neighb_obj_values[0], s=10, c='lightgray')
                subst_beam_ax.scatter([0]*len(all_subst_obj_values[0]), all_subst_obj_values[0], s=2.5, c='k')
            else:
              if len(all_neighb_obj_values[j-1]) > 0:
                for curr_stream_idx in range(len(stream_lineages[j-1])):
                  curr_stream_lineage = stream_lineages[j-1][curr_stream_idx]
                  for curr_stream_ancestor in curr_stream_lineage:
                    if check_times:
                      neighb_beam_ax.plot([iter_times[j-1],iter_times[j]],[all_neighb_obj_values[j-1][curr_stream_ancestor],all_neighb_obj_values[j][curr_stream_idx]], ls='-', lw=0.5, c='k')
                    else:
                      neighb_beam_ax.plot([j-1,j],[all_neighb_obj_values[j-1][curr_stream_ancestor],all_neighb_obj_values[j][curr_stream_idx]], ls='-', lw=0.5, c='k')
              if check_times:
                neighb_beam_ax.scatter([iter_times[j]]*len(all_neighb_obj_values[j]),all_neighb_obj_values[j], s=10, c='lightgray')
              else:
                neighb_beam_ax.scatter([j]*len(all_neighb_obj_values[j]),all_neighb_obj_values[j], s=10, c='lightgray')
              if check_for_duplicates:
                evaluated_subst_idcs = [idx for idx, subst_obj_value in enumerate(all_subst_obj_values[j]) if subst_obj_value > -1]
                non_dup_evaluated_subst_obj_values = []
                dup_evaluated_subst_obj_values = []
                for idx in evaluated_subst_idcs:
                  if (j,idx) in iters_streams_dup:
                    non_dup_evaluated_subst_obj_values.append(all_subst_obj_values[j][idx])
                  else:
                    dup_evaluated_subst_obj_values.append(all_subst_obj_values[j][idx])
                if check_times:
                  subst_beam_ax.scatter([iter_times[j]]*len(dup_evaluated_subst_obj_values), dup_evaluated_subst_obj_values, s=2.5, c='gray')
                  subst_beam_ax.scatter([iter_times[j]]*len(non_dup_evaluated_subst_obj_values), non_dup_evaluated_subst_obj_values, s=2.5, c='k')
                else:
                  subst_beam_ax.scatter([j]*len(dup_evaluated_subst_obj_values), dup_evaluated_subst_obj_values, s=2.5, c='gray')
                  subst_beam_ax.scatter([j]*len(non_dup_evaluated_subst_obj_values), non_dup_evaluated_subst_obj_values, s=2.5, c='k')
              else:
                evaluated_subst_obj_values = [subst_obj_value for subst_obj_value in all_subst_obj_values[j] if subst_obj_value > -1]
                if check_times:
                  subst_beam_ax.scatter([iter_times[j]]*len(evaluated_subst_obj_values), evaluated_subst_obj_values, s=2.5, c='k')
                else:
                  subst_beam_ax.scatter([j]*len(evaluated_subst_obj_values), evaluated_subst_obj_values, s=2.5, c='k')
              if check_times:
                subst_beam_ax.plot([iter_times[j-1],iter_times[j]],[old_best_subst_value,best_subst_value], ls='-', lw=0.5, c='r')
              else:
                subst_beam_ax.plot([j-1,j],[old_best_subst_value,best_subst_value], ls='-', lw=0.5, c='r')
        ax_xlim = neighb_beam_ax.get_xlim()
        subst_beam_ax.set_xlim([ax_xlim[0], ax_xlim[1]])
        plt.pause(1e-8)
        plt.ioff()
        if saved_log_and_plots_filepath_no_extension != "":
          if fixed_master_hub:
            #plt.savefig(inst_label+"_"+str(MH)+".eps", dpi=300)
            plt.savefig(saved_log_and_plots_filepath_no_extension+"_"+str(MH)+".png", dpi=300)
          else:
            #plt.savefig(inst_label+".eps", dpi=300)
            plt.savefig(saved_log_and_plots_filepath_no_extension+".png", dpi=300)


    # BEST OBJ. VALUES TRIMMING AND COMPILING
    if True:
      ## Best Obj. Values Trimming
      if best_neighb_obj_values[0][0] == -1:
        best_neighb_obj_values = best_neighb_obj_values[1:]
      if best_subst_obj_values[0][0] == -1:
        best_subst_obj_values = best_subst_obj_values[1:]
      if search_all_scenarios:
        if best_subst_A_obj_mins[0][0] == -1:
          best_subst_A_obj_mins = best_subst_A_obj_mins[1:]
        if best_subst_B_obj_mins[0][0] == -1:
          best_subst_B_obj_mins = best_subst_B_obj_mins[1:]
        if best_subst_C_obj_mins[0][0] == -1:
          best_subst_C_obj_mins = best_subst_C_obj_mins[1:]

      ## Best Obj. Values Compiling
      if fixed_master_hub:
        glob_best_neighb_obj_values = glob_best_neighb_obj_values + best_neighb_obj_values
        glob_best_neighb_obj_values.sort(key = lambda Amin_Bmin_Cmin_value_iter_mesh_simplex_hept: Amin_Bmin_Cmin_value_iter_mesh_simplex_hept[3])
        glob_best_neighb_obj_values = glob_best_neighb_obj_values[-nb_best_solutions:]
        glob_best_subst_obj_values = glob_best_subst_obj_values + best_subst_obj_values
        glob_best_subst_obj_values.sort(key = lambda Amin_Bmin_Cmin_value_iter_mesh_simplex_hept: Amin_Bmin_Cmin_value_iter_mesh_simplex_hept[3])
        glob_best_subst_obj_values = glob_best_subst_obj_values[-nb_best_substitutes:]
        if search_all_scenarios:
          glob_best_subst_A_obj_mins = glob_best_subst_A_obj_mins + best_subst_A_obj_mins
          glob_best_subst_A_obj_mins.sort(key = lambda min_value_iter_mesh_simplex_quin: min_value_iter_mesh_simplex_quin[1])
          glob_best_subst_A_obj_mins = glob_best_subst_A_obj_mins[-nb_best_substitutes:]
          glob_best_subst_B_obj_mins = glob_best_subst_B_obj_mins + best_subst_B_obj_mins
          glob_best_subst_B_obj_mins.sort(key = lambda min_value_iter_mesh_simplex_quin: min_value_iter_mesh_simplex_quin[1])
          glob_best_subst_B_obj_mins = glob_best_subst_B_obj_mins[-nb_best_substitutes:]
          glob_best_subst_C_obj_mins = glob_best_subst_C_obj_mins + best_subst_C_obj_mins
          glob_best_subst_C_obj_mins.sort(key = lambda min_value_iter_mesh_simplex_quin: min_value_iter_mesh_simplex_quin[1])
          glob_best_subst_C_obj_mins = glob_best_subst_C_obj_mins[-nb_best_substitutes:]
  

  # BEST OBJ. VALUES RENAMING (if Fixed_Master_Hub)
  if fixed_master_hub:
    best_neighb_obj_values = glob_best_neighb_obj_values
    best_subst_obj_values = glob_best_subst_obj_values
    if search_all_scenarios:
      best_subst_A_obj_mins = glob_best_subst_A_obj_mins
      best_subst_B_obj_mins = glob_best_subst_B_obj_mins
      best_subst_C_obj_mins = glob_best_subst_C_obj_mins


  # FULL OBJ. FUNCTION EVALUATIONS
  if True:
    if substitute_obj_func != full_obj_func:
      best_full_obj_values = []
      if verbose or saved_log_and_plots_filepath_no_extension != "":
        best_full_obj_value = -1
        MWMB_Plot.print_log("", verbose, files=full_log_files+bests_log_files)
      for _,_,_,subst_obj_value,iter,mesh_sizes,simplex_tuple_list in best_subst_obj_values:
        tree = ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, simplex_tuple_list)
        if check_times:
          tic = time.time()
        full_obj_value, full_net = tree.get_obj_value(obj_func = full_obj_func, get_network = True)
        if check_times:
          full_obj_times.append(time.time() - tic)
        best_full_obj_values.append((full_net.A_obj_min, full_net.B_obj_min, full_net.C_obj_min, full_obj_value, iter, mesh_sizes, simplex_tuple_list))
        if verbose or saved_log_and_plots_filepath_no_extension != "":
          if full_obj_value > best_full_obj_value:
            best_full_obj_value = full_obj_value
            new_best_full_string = "!!! "
          else:
            new_best_full_string = "    "
          MWMB_Plot.print_log("  "+str(iter)+"\t    "+str(round(100*subst_obj_value)/100)+" \t"+new_best_full_string+str(round(100*full_obj_value)/100)+" \t   "+str(simplex_tuple_list)+"      mesh sizes = "+str(mesh_sizes), verbose, files=full_log_files+bests_log_files)
      best_full_obj_values.sort(key = lambda Amin_Bmin_Cmin_value_iter_mesh_simplex_hept: Amin_Bmin_Cmin_value_iter_mesh_simplex_hept[3])
      best_full_obj_values = best_full_obj_values[-nb_best_solutions:]
      if search_all_scenarios:
        best_subst_X_obj_mins = [best_subst_A_obj_mins,best_subst_B_obj_mins,best_subst_C_obj_mins]
        best_full_A_obj_mins = []
        best_full_B_obj_mins = []
        best_full_C_obj_mins = []
        best_full_X_obj_mins = [best_full_A_obj_mins,best_full_B_obj_mins,best_full_C_obj_mins]
        for obj_min_idx in range(3):
          if verbose or saved_log_and_plots_filepath_no_extension != "":
            best_full_X_obj_value = -1
            MWMB_Plot.print_log("", verbose, files=full_log_files+[A_bests_log_files,B_bests_log_files,C_bests_log_files][obj_min_idx])
          for subst_X_obj_min,subst_X_obj_value,iter,mesh_sizes,simplex_tuple_list in best_subst_X_obj_mins[obj_min_idx]:
            tree = ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, simplex_tuple_list)
            if check_times:
              tic = time.time()
            _, full_net = tree.get_obj_value(obj_func = full_obj_func, get_network = True)
            if check_times:
              full_obj_times.append(time.time() - tic)
            if obj_min_idx == 0:
              full_X_obj_min = full_net.A_obj_min
              full_X_obj_value = full_net.A_obj_value
            elif obj_min_idx == 1:
              full_X_obj_min = full_net.B_obj_min
              full_X_obj_value = full_net.B_obj_value
            else:
              full_X_obj_min = full_net.C_obj_min
              full_X_obj_value = full_net.C_obj_value
            best_full_X_obj_mins[obj_min_idx].append((full_X_obj_min, full_X_obj_value, iter, mesh_sizes, simplex_tuple_list))
            if verbose or saved_log_and_plots_filepath_no_extension != "":
              if full_X_obj_value > best_full_X_obj_value:
                best_full_X_obj_value = full_X_obj_value
                new_best_full_string = "!!! "
              else:
                new_best_full_string = "    "
              MWMB_Plot.print_log("  "+str(iter)+"\t    "+str((round(100*subst_X_obj_min)/100,round(100*subst_X_obj_value)/100))+" \t"+new_best_full_string+str((round(100*full_X_obj_min)/100,round(100*full_X_obj_value)/100))+" \t   "+str(simplex_tuple_list)+"      mesh sizes = "+str(mesh_sizes), verbose, files=full_log_files+[A_bests_log_files,B_bests_log_files,C_bests_log_files][obj_min_idx])
          best_full_X_obj_mins[obj_min_idx].sort(key = lambda Xmin_Xvalue_iter_mesh_simplex_quin: Xmin_Xvalue_iter_mesh_simplex_quin[1])
          best_full_X_obj_mins[obj_min_idx] = best_full_X_obj_mins[obj_min_idx][-nb_best_solutions:]
    else:
      best_full_obj_values = best_subst_obj_values[-nb_best_solutions:]
      if search_all_scenarios:
        best_full_A_obj_mins = best_subst_A_obj_mins[-nb_best_solutions:]
        best_full_B_obj_mins = best_subst_B_obj_mins[-nb_best_solutions:]
        best_full_C_obj_mins = best_subst_C_obj_mins[-nb_best_solutions:]


  # OUTPUT
  if True:
    MWMB_Plot.print_log("\nThe best topology found had F_obj = "+str(best_full_obj_values[-1][3])+" at iteration "+str(best_full_obj_values[-1][4])+" (out of "+str(i)+" iterations).      "+str(best_full_obj_values[-1][6])+"      mesh sizes = "+str(best_full_obj_values[-1][5]), True, files=full_log_files+bests_log_files)
    MWMB_Plot.print_log("The 'simplified' best topology had f_obj = "+str(best_neighb_obj_values[-1][3])+" at iteration "+str(best_neighb_obj_values[-1][4])+".      "+str(best_neighb_obj_values[-1][6])+"      mesh sizes = "+str(best_neighb_obj_values[-1][5]), True, files=full_log_files)
    MWMB_Plot.print_log("Best "+str(nb_best_solutions)+" Solutions ("+str(neighborhood_obj_func)+") :", True, files=full_log_files)
    for best_neighb_obj_value in best_neighb_obj_values:
      MWMB_Plot.print_log(best_neighb_obj_value, True, files=full_log_files)
    MWMB_Plot.print_log("Best "+str(nb_best_solutions)+" Solutions ("+str(full_obj_func)+") :", True, files=full_log_files+bests_log_files)
    for best_full_obj_value in best_full_obj_values:
      MWMB_Plot.print_log(best_full_obj_value, True, files=full_log_files+bests_log_files)
    if search_all_scenarios:
      MWMB_Plot.print_log("Best "+str(nb_best_solutions)+" Solutions ("+str(full_obj_func)+", Any --> Any) :", True, files=full_log_files+A_bests_log_files)
      for best_full_A_obj_min in best_full_A_obj_mins:
        MWMB_Plot.print_log(best_full_A_obj_min, True, files=full_log_files+A_bests_log_files)
      MWMB_Plot.print_log("Best "+str(nb_best_solutions)+" Solutions ("+str(full_obj_func)+", All --> MH) :", True, files=full_log_files+B_bests_log_files)
      for best_full_B_obj_min in best_full_B_obj_mins:
        MWMB_Plot.print_log(best_full_B_obj_min, True, files=full_log_files+B_bests_log_files)
      MWMB_Plot.print_log("Best "+str(nb_best_solutions)+" Solutions ("+str(full_obj_func)+", All <-> All) :", True, files=full_log_files+C_bests_log_files)
      for best_full_C_obj_min in best_full_C_obj_mins:
        MWMB_Plot.print_log(best_full_C_obj_min, True, files=full_log_files+C_bests_log_files)
    if check_times:
      MWMB_Plot.print_log("Total Time = "+str(time.time() - total_tic), True, files=full_log_files)
      if len(neighb_obj_times) > 0:
        MWMB_Plot.print_log("Neighborhood obj func with "+str(neighborhood_obj_func)+" Timing -----> Min="+str(min(neighb_obj_times))+"; Max="+str(max(neighb_obj_times))+"; Avg="+str(mean(neighb_obj_times))+";", True, files=full_log_files)
      if len(bound_obj_times) > 0:
        MWMB_Plot.print_log("Upper bound obj func with "+str(bound_obj_func)+" Timing -----> Min="+str(min(bound_obj_times))+"; Max="+str(max(bound_obj_times))+"; Avg="+str(mean(bound_obj_times))+";", True, files=full_log_files)
      if substitute_obj_func == full_obj_func and len(subst_obj_times) > 0:
        MWMB_Plot.print_log("Full obj func Timing -----> Min="+str(min(subst_obj_times))+"; Max="+str(max(subst_obj_times))+"; Avg="+str(mean(subst_obj_times))+";", True, files=full_log_files)
      else:
        if len(subst_obj_times) > 0:
          MWMB_Plot.print_log("Substitute obj func with "+str(substitute_obj_func)+" Timing -----> Min="+str(min(subst_obj_times))+"; Max="+str(max(subst_obj_times))+"; Avg="+str(mean(subst_obj_times))+";", True, files=full_log_files)
        if len(full_obj_times) > 0:
          MWMB_Plot.print_log("Full obj func Timing -----> Min="+str(min(full_obj_times))+"; Max="+str(max(full_obj_times))+"; Avg="+str(mean(full_obj_times))+";", True, files=full_log_files)
      if len(tree_neighb_search_times) > 0:
        MWMB_Plot.print_log("Tree neighborhood search Timing ----> Min="+str(min(tree_neighb_search_times))+"; Max="+str(max(tree_neighb_search_times))+"; Avg="+str(mean(tree_neighb_search_times))+";", True, files=full_log_files)
      if len(mesh_neighb_search_times) > 0:
        MWMB_Plot.print_log("Mesh neighborhood search Timing ----> Min="+str(min(mesh_neighb_search_times))+"; Max="+str(max(mesh_neighb_search_times))+"; Avg="+str(mean(mesh_neighb_search_times))+";", True, files=full_log_files)
    if first_better:
      if neighborhoods != "mesh" and nb_tree_neighbor_search > 0:
        if neighborhood_subset_ratio < 1:
          MWMB_Plot.print_log("Tree neighborhood search visited the full neighborhood subset "+str(round(10000*nb_tree_visited_all_neighbors/nb_tree_neighbor_search)/100)+"% of the time.", True, files=full_log_files)
        else:
          MWMB_Plot.print_log("Tree neighborhood search visited the full neighborhood "+str(round(10000*nb_tree_visited_all_neighbors/nb_tree_neighbor_search)/100)+"% of the time.", True, files=full_log_files)
      if neighborhoods != "tree" and nb_mesh_neighbor_search > 0:
        if neighborhood_subset_ratio < 1:
          MWMB_Plot.print_log("Mesh neighborhood search visited the full neighborhood subset "+str(round(10000*nb_mesh_visited_all_neighbors/nb_mesh_neighbor_search)/100)+"% of the time.", True, files=full_log_files)
        else:
          MWMB_Plot.print_log("Mesh neighborhood search visited the full neighborhood "+str(round(10000*nb_mesh_visited_all_neighbors/nb_mesh_neighbor_search)/100)+"% of the time.", True, files=full_log_files)
    
    if saved_log_and_plots_filepath_no_extension != "":
      for log_file in all_log_files:
        log_file.close()
  
  
  best_tree = ImpTree.create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, best_full_obj_values[-1][6])
  if not get_network:
    return best_tree
  else:
    _, best_network = best_tree.get_obj_value(obj_func=full_obj_func, get_network=True)
    return best_tree, best_network

"""
Returns a list of the best topologies encountered by a search via a BESTS text file (...BESTS.txt, ...BESTS_A.txt, ...BESTS_B.txt, or ...BESTS_C.txt) where
  'filepath' is the filepath to the BESTS text file;
  'nb_best_solutions' (optional) is the number of best solutions to return;
  'value_simplex_pairs' (optional) is whether to return a list of tuples (value, simplex_tuple_list) with the value (True) or only a list of simplex_tuple_lists (False);
  'min_value' (optional) is, if a ...BESTS_X.txt file, whether to consider only the minimum values of scenario X (True) or the aggregate values that also take into account the
    means (False).
"""
def get_best_simplex_tuple_lists_from_BESTS_log_file(filepath, nb_best_solutions=1, value_simplex_pairs=False, min_value=False):
  best_simplex_tuple_lists = []
  is_before_final_bests = True
  if filepath[-9:] == "BESTS.txt":
    bests_type = ""
  elif filepath[-11:] == "BESTS_A.txt":
    bests_type = "A"
  elif filepath[-11:] == "BESTS_B.txt":
    bests_type = "B"
  elif filepath[-11:] == "BESTS_C.txt":
    bests_type = "C"
  else:
    bests_type = -1
  if bests_type != -1 and Path(filepath).is_file():
    with open(filepath) as f:
      lines = f.readlines()
    for line_idx in range(len(lines)):
      line = lines[line_idx].strip()
      if len(line) > 0 and line[0] == "(":
        #print(line)
        if is_before_final_bests:
          new_tuple = tuple(tuple(eval(line))[1])
        else:
          new_tuple = tuple(eval(line))
        if bests_type == "":
          value_simplex_pair = (new_tuple[3],new_tuple[6])
        else:
          if min_value:
            value_simplex_pair = (new_tuple[0],new_tuple[4])
          else:
            value_simplex_pair = (new_tuple[1],new_tuple[4])
        is_duplicate = False
        for old_value_simplex_pair_idx in range(len(best_simplex_tuple_lists)):
          old_value,old_simplex_tuple_list = best_simplex_tuple_lists[old_value_simplex_pair_idx]
          if old_simplex_tuple_list == value_simplex_pair[1]:
            is_duplicate = True
            if value_simplex_pair[0] > old_value:
              best_simplex_tuple_lists[old_value_simplex_pair_idx] = value_simplex_pair
            break
        if not is_duplicate:
          best_simplex_tuple_lists.append(value_simplex_pair)
          best_simplex_tuple_lists.sort(key=lambda value_simplex_pair: value_simplex_pair[0])
          best_simplex_tuple_lists = best_simplex_tuple_lists[-nb_best_solutions:]
      elif len(line) > 0 and line[:4] == "Best":
        is_before_final_bests = False
  if value_simplex_pairs:
    return best_simplex_tuple_lists
  else:
    return [simplex_tuple_list for _,simplex_tuple_list in best_simplex_tuple_lists]

"""
Returns the progression of the objective value of the search in time (or iterations) via a BESTS text file (...BESTS.txt, ...BESTS_A.txt, ...BESTS_B.txt, or ...BESTS_C.txt) where
  'filepath' is the filepath to the BESTS text file.
"""
def get_time_obj_value_progression_from_BESTS_log_file(filepath):
  time_value_pairs = []
  best_obj_value = -2
  if filepath[-9:] == "BESTS.txt":
    bests_type = ""
  elif filepath[-11:] == "BESTS_A.txt":
    bests_type = "A"
  elif filepath[-11:] == "BESTS_B.txt":
    bests_type = "B"
  elif filepath[-11:] == "BESTS_C.txt":
    bests_type = "C"
  else:
    bests_type = -1
  if bests_type != -1 and Path(filepath).is_file():
    with open(filepath) as f:
      lines = f.readlines()
    for line_idx in range(len(lines)):
      line = lines[line_idx].strip()
      if len(line) > 0 and line[0] == "(":
        new_tuple = tuple(eval(line))
        if bests_type == "":
          time_value_pair = (new_tuple[0],new_tuple[1][3])
        else:
          time_value_pair = (new_tuple[0],new_tuple[1][1])
        if time_value_pair[1] > best_obj_value:
          best_obj_value = time_value_pair[1]
          time_value_pairs.append(time_value_pair)
      elif len(line) > 0 and line[:4] == "Best":
        break
  return time_value_pairs

"""
Returns the progression of the iterations of the search in time via a LOG text file (...LOG.txt) where
  'filepath' is the filepath to the LOG text file.
"""
def get_time_iteration_progression_from_LOG_log_file(filepath):
  time_iteration_pairs = []
  if Path(filepath).is_file():
    with open(filepath) as f:
      lines = f.readlines()
    for line_idx in range(len(lines)):
      line = lines[line_idx].strip()
      if len(line) > 0 and line[0].isnumeric() and line[-1] == ")":
        line = line.split(" ")
        time_value_pair = (float(line[-1][:-1]),int(line[0])-1)
        time_iteration_pairs.append(time_value_pair)
  return time_iteration_pairs

"""
Returns the values of the x axis (time) where
  'max_time' is the maximum time of the plot;
  'tic_time' is the time resolution of the plot.
"""
def get_curve_x_tics(max_time, tic_time):
  return [tic_time*tic_idx for tic_idx in range(1+ceil(max_time/tic_time))]

"""
Returns the values of the y axis (iterations) of a plot of the number of iterations with respect to time (the x axis given by get_curve_x_tics(...)) where
  'time_iteration_pairs' is the progression of the iterations given by get_time_iteration_progression_from_LOG_log_file(...);
  'max_time' is the maximum time of the plot;
  'tic_time' is the time resolution of the plot.
"""
def get_iteration_curve_coordinates_from_time_iteration_progression(time_iteration_pairs, max_time, tic_time):
  y_coordinates = [0 for _ in range(1+ceil(max_time/tic_time))]
  curve_tic_idx = 0
  iteration = 0
  for alg_time, iteration in time_iteration_pairs:
    if alg_time <= max_time:
      while alg_time/tic_time > curve_tic_idx+1:
        curve_tic_idx += 1
        y_coordinates[curve_tic_idx] = iteration - 1
  while max_time/tic_time >= curve_tic_idx+1:
    curve_tic_idx += 1
    y_coordinates[curve_tic_idx] = iteration
  return y_coordinates

"""
Returns the values of the y axis (renormalized objective with 1=initial and 0=best / primal integral) of a plot of the primal value/primal integral with respect to time (the x axis
given by get_curve_x_tics(...)) where
  'time_value_pairs' is the progression of the objective value given by get_time_obj_value_progression_from_BESTS_log_file(...);
  'best_obj_value' is the best objective value (y=0);
  'max_time' is the maximum time of the plot;
  'tic_time' is the time resolution of the plot.
"""
def get_obj_value_and_primal_integral_curve_coordinates_from_time_obj_value_progression(time_value_pairs, best_obj_value, max_time, tic_time):
  obj_value_y_coordinates = [1 for _ in range(1+ceil(max_time/tic_time))]
  prim_int_y_coordinates = [0 for _ in range(1+ceil(max_time/tic_time))]

  time_primal_pairs = [(time,1-(value-time_value_pairs[0][1])/(best_obj_value-time_value_pairs[0][1])) for time,value in time_value_pairs]

  idx = 0
  primal_integral = 0
  for tic_idx in range(len(obj_value_y_coordinates)):
    curr_time = tic_idx * tic_time

    if curr_time >= time_value_pairs[0][0]:
      prev_no_classifier_idx = idx
      while idx < len(time_primal_pairs) and curr_time >= time_primal_pairs[idx][0]:
        idx += 1
      obj_value_y_coordinates[tic_idx] = time_primal_pairs[idx-1][1]

      if tic_idx > 0:
        for int_idx in range(prev_no_classifier_idx-1, idx):
          init_time = max(curr_time - tic_time, time_primal_pairs[int_idx][0])
          if int_idx == len(time_primal_pairs)-1:
            end_time = curr_time
          else:
            end_time = min(curr_time, time_primal_pairs[int_idx+1][0])
          primal_integral += time_primal_pairs[int_idx][1] * (end_time - init_time)/max_time

        prim_int_y_coordinates[tic_idx] = primal_integral
    elif tic_idx > 0:
      primal_integral += time_primal_pairs[idx][1] * tic_time/max_time

      prim_int_y_coordinates[tic_idx] = primal_integral
  return obj_value_y_coordinates, prim_int_y_coordinates

"""
Averages the values of the y axis of multiple comparable searches where
  'multiple_curve_coordinates' is a list of lists of y axis values.
"""
def get_average_curve_coordinates_from_multiple_curve_coordinates(multiple_curve_coordinates):
  return [sum([multiple_curve_coordinates[curve_idx][tic_idx] for curve_idx in range(len(multiple_curve_coordinates))])/len(multiple_curve_coordinates) for tic_idx in range(len(multiple_curve_coordinates[0]))]



