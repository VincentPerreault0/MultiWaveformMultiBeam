import copy
from math import ceil, floor, sqrt
from statistics import mean
import time
from MWMB_Problem import FreqMHz
import MWMB_Plot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import warnings

"""
EXPLICIT NODES ARE IDENTIFIED BY AN INDEX BETWEEN 0 AND 'nb_nodes'-1 INCLUSIVELY.
"""



"""
ImpTree
-------
Object representing an implicit tree on a specific ProbInstance with specific ProbParameters.

  .prob_instance {ProbInstance} --> associated problem instance
  .prob_params {ProbParameters} --> associated problem parameters
  .adj_matrix {numpy array of floats of shape ('nb_nodes', 'nb_nodes')} --> adjacency matrix of the implicit tree such that
    a_uv = 1 --> there is a tree edge between nodes u and v
    a_uv = n, with n >= 2 --> nodes u and v are in the same mesh cluster with mesh_index = -n
  .has_obj_value = False {boolean} --> whether its true objective value or greedy pseudo-objective value has been evaluated and stored (True) or not (False)
  .has_old_avg = False {boolean} --> whether it stores the intermediary data for the "avg" pseudo-objective evaluation of a previous neighbor (True) or not (False)
  .has_old_chan_avg = False {boolean} --> whether it stores the intermediary data for the "chan_avg" pseudo-objective evaluation of a previous neighbor (True) or not (False)
  .has_old_chan_max = False {boolean} --> whether it stores the intermediary data for the "chan_max" pseudo-objective evaluation of a previous neighbor (True) or not (False)

  (defined if .has_obj_value)
    .obj_value {float}  --> its true objective value ("full") or greedy pseudo-objective value ("greedy") depending on which has been last evaluated
  
  (defined if .has_old_avg or .has_old_chan_avg or .has_old_chan_max)
    .old_arb_adj_matrix {numpy array of floats of shape ('nb_nodes', 'nb_nodes')} --> adjacency matrix of the best ImpArborescence of a previous ImpTree neighbor (see ImpArborescence
      for details)
    .old_MH_succs_2_partition {list of 2 lists of implicit successors to the master hub} --> partition of the master hub's successors in 2 sets of the best pseudo-network of a
      previous neighbor (see Network for details)
    .old_direct_simp_TPs {numpy array of floats of shape ('nb_nodes', 'nb_nodes')} --> direct throughputs of the best pseudo-network of a previous neighbor (see Network for details)
    .old_mesh_members {dictionary from mesh index to associated list of explicit nodes} --> describes the members of each mesh cluster of a previous neighbor (see
      ImpArborescence for details)
    .old_preds {dictionary from implicit node (except master hub) to associated implicit node} --> describes the predecessors of the best arborescence of a previous neighbor
      (see ImpArborescence for details)
    .old_mesh_routes {dictionary from mesh index to associated dictionary from explicit node to associated dictionary from explicit node to associated list of tuples of explicit nodes}
      --> describes the routes used between every pair of members for each cluster of the best pseudo-network of a previous neighbor (see Network for details)

  (defined if .has_old_chan_avg or .has_old_chan_max)
    .old_chan_assignment {dictionary from implicit node (with successors) to associated frequency index} --> describes the channels of the downlink connections of the predecessors
      of the best pseudo-network of a previous neighbor (see Network for details)
"""
class ImpTree:
  """
  Initializes an ImpTree object where
    'prob_instance' is the associated problem instance;
    'prob_params' is the associated problem parameters;
    'adj_matrix' is its adjacency matrix such that
      a_uv = 1 --> there is a tree edge between nodes u and v;
      a_uv = n, with n >= 2 --> nodes u and v are in the same mesh cluster with mesh_index = -n.
  """
  def __init__(self, prob_instance, prob_params, adj_matrix):
    self.prob_instance = prob_instance # ProbInstance
    self.prob_params = prob_params # ProbParameters
    self.adj_matrix = adj_matrix # (nb_exp_nodes, nb_exp_nodes)
    self.has_obj_value = False # boolean
    self.has_old_avg = False # boolean
    self.has_old_chan_avg = False # boolean
    self.has_old_chan_max = False # boolean

  """
  Defines equality of ImpTree objects.
  """
  def __eq__(self, other):
    if isinstance(other, ImpTree):
      return (self.adj_matrix == other.adj_matrix).all() and self.prob_params == other.prob_params and self.prob_instance == other.prob_instance
    else:
      return False
  
  """
  Prints an ImpTree object.
  """
  def __repr__(self):
    string = "Implicit Tree :\n\t"
    #implicit nodes
    string += "Adjacency Matrix : "+str(self.adj_matrix)
    return string

  """
  Plots an ImpTree as a set of nodes connected by edges (solid lines) or by clusters (added white nodes with dotted lines) where
    'ax' (optional) is the matplotlib axes object in which to plot;
    'pretitle' (optional) is the title string of the plot (ending in " F=obj_value" if the ImpTree has an obj_value);
    'MH' (optional) is the master hub, drawn in green;
    'added_edges' (optional) is a list of edges, drawn in blue;
    'dropped_edges' (optional) is a list of edges, drawn in red.
  """
  def show(self, ax=-1, pretitle="", MH=-1, added_edges=[], dropped_edges=[]):
    extra_border = 1
    line_width = 1
    has_axis = ax != -1
    if not has_axis:
      fig = plt.figure(1)
      ax = fig.add_axes([0.075,0.075,0.85,0.85])
    ax.set_aspect('equal')
    has_MH = MH != -1
    has_added_edge = len(added_edges) > 0
    has_dropped_edge = len(dropped_edges) > 0
    nb_nodes = self.prob_instance.nb_nodes
    coordinates_km = self.prob_instance.coordinates_km
    adj_matrix = self.adj_matrix
    xlim0 = -extra_border
    xlim1 = np.max(coordinates_km[:,0])+extra_border
    ylim0 = -extra_border
    ylim1 = np.max(coordinates_km[:,1])+extra_border
    ax.set_xlim(xlim0,xlim1)
    ax.set_ylim(ylim0,ylim1)
    max_size = max(xlim1-xlim0,ylim1-ylim0)
    if self.has_obj_value:
      ax.set_title(pretitle+"F="+str(round(100*self.obj_value)/100))
    else:
      ax.set_title(" "+pretitle)
    max_mesh_idx = np.max(adj_matrix)
    is_mixed = max_mesh_idx >= 2
    if is_mixed:
      mesh_coordinates_km = {}
      meshes = {v : [] for v in range(nb_nodes)}
      for mesh_abs_idx in range(2, int(max_mesh_idx)+1):
        mesh_idx = -1*mesh_abs_idx
        curr_mesh_members = [int(npint) for npint in set(np.where(adj_matrix == mesh_abs_idx)[0])]
        if len(curr_mesh_members) > 0:
          mesh_coordinates_km[mesh_idx] = (np.mean(coordinates_km[curr_mesh_members,0]),np.mean(coordinates_km[curr_mesh_members,1]))
          for mesh_member in curr_mesh_members:
            meshes[mesh_member].append(mesh_idx)
    # EDGES
    for w1 in range(nb_nodes):
      if is_mixed:
        for mesh_idx in meshes[w1]:
          dx = mesh_coordinates_km[mesh_idx][0] - coordinates_km[w1,0]
          dy = mesh_coordinates_km[mesh_idx][1] - coordinates_km[w1,1]
          radius = sqrt(pow(dx,2) + pow(dy,2))
          ratio_beg = (7/400*max_size)/radius
          ratio_end = (radius - 7/400*max_size)/radius
          ax.plot([coordinates_km[w1,0]+ratio_beg*dx,coordinates_km[w1,0]+ratio_end*dx],[coordinates_km[w1,1]+ratio_beg*dy,coordinates_km[w1,1]+ratio_end*dy],ls=":",lw=1.4*line_width,c='k')
      for w2 in range(w1+1,nb_nodes):
        if adj_matrix[w1,w2] == 1:
          dx = coordinates_km[w2,0] - coordinates_km[w1,0]
          dy = coordinates_km[w2,1] - coordinates_km[w1,1]
          radius = sqrt(pow(dx,2) + pow(dy,2))
          ratio_beg = (7/400*max_size)/radius
          ratio_end = (radius - 7/400*max_size)/radius
          if not has_added_edge and not has_dropped_edge:
            clr = 'k'
          else:
            if has_added_edge and {w1,w2} in added_edges:
              clr = 'blue'
            elif has_dropped_edge and {w1,w2} in dropped_edges:
              clr = 'darkred'
            else:
              clr = 'k'
          ax.plot([coordinates_km[w1,0]+ratio_beg*dx,coordinates_km[w1,0]+ratio_end*dx],[coordinates_km[w1,1]+ratio_beg*dy,coordinates_km[w1,1]+ratio_end*dy],ls="-",lw=line_width,c=clr)
    # NODES
    if not has_MH:
      #if not is_mixed:
      ax.scatter(coordinates_km[:,0].tolist(),coordinates_km[:,1].tolist(), s=30, c='k')
      #else:
      #  mesh_nodes = []
      #  not_mesh_nodes = []
      #  for v in range(nb_nodes):
      #    if np.max(adj_matrix[v,:]) > 1:
      #      mesh_nodes.append(v)
      #    else:
      #      not_mesh_nodes.append(v)
      #  ax.scatter(coordinates_km[not_mesh_nodes,0].tolist(),coordinates_km[not_mesh_nodes,1].tolist(), s=30, c='k')
      #  ax.scatter(coordinates_km[mesh_nodes,0].tolist(),coordinates_km[mesh_nodes,1].tolist(), s=30, c='dimgray')
    else:
      ax.scatter([coordinates_km[MH,0]],[coordinates_km[MH,1]], s=30, c='green')
      non_MH_coordinates = np.concatenate((coordinates_km[:MH,:],coordinates_km[MH+1:,:]))
      ax.scatter(non_MH_coordinates[:,0].tolist(),non_MH_coordinates[:,1].tolist(), s=30, c='k')
    if is_mixed:
      mesh_coordinates_x_km = []
      mesh_coordinates_y_km = []
      for mesh_idx in mesh_coordinates_km.keys():
        mesh_coordinates_x_km.append(mesh_coordinates_km[mesh_idx][0])
        mesh_coordinates_y_km.append(mesh_coordinates_km[mesh_idx][1])
      ax.scatter(mesh_coordinates_x_km,mesh_coordinates_y_km, s=50, c='w', edgecolors='k')
    # NODE LABELS
    for exp_node in range(nb_nodes):
      ax.text(coordinates_km[exp_node,0],coordinates_km[exp_node,1] - 5/100*max_size,str(exp_node),fontsize=8,color='gray',ha='center')
    if not has_axis:
      fig.canvas.draw()
      plt.show()

  """
  Creates an ImpTree corresponding to a random tree where
    'prob_instance' is the ProbInstance;
    'prob_params' is the ProbParameters.
  """
  @staticmethod
  def create_random_exp_tree(prob_instance, prob_params):
    nb_nodes = prob_instance.nb_nodes
    adj_matrix = np.zeros((nb_nodes,nb_nodes))
    nodes = [*range(nb_nodes)]
    first_node = random.choice(nodes)
    nodes.remove(first_node)
    nodes_in_tree = [first_node]
    for i in range(nb_nodes-1):
      w1 = random.choice(nodes)
      nodes.remove(w1)
      w2 = random.choice(nodes_in_tree)
      adj_matrix[w1,w2] = 1
      adj_matrix[w2,w1] = 1
      nodes_in_tree.append(w1)
    return ImpTree(prob_instance, prob_params, adj_matrix)
  
  """
  Creates an ImpTree corresponding to a star topology (one central node to which all other nodes are linked by an edge) where
    'prob_instance' is the ProbInstance;
    'prob_params' is the ProbParameters;
    'MH' is the central node.
  """
  @staticmethod
  def create_star_tree_from_MH(prob_instance, prob_params, MH):
    nb_nodes = prob_instance.nb_nodes
    adj_matrix = np.zeros((nb_nodes,nb_nodes))
    for v in range(nb_nodes):
      if v != MH:
        adj_matrix[v,MH] = 1
        adj_matrix[MH,v] = 1
    return ImpTree(prob_instance, prob_params, adj_matrix)
  
  """
  Creates an ImpTree corresponding to a list of simplex tuples (of size 2 or more) where
    'prob_instance' is the ProbInstance;
    'prob_params' is the ProbParameters;
    'simplex_tuple_list' is a list of tuples representing simplices in the ImpTree;
      size = 2 --> tuple corresponds to a tree edge;
      size > 2 --> tuple corresponds to a mesh cluster;
  """
  @staticmethod
  def create_imp_tree_from_simplex_tuple_list(prob_instance, prob_params, simplex_tuple_list):
    return ImpTree(prob_instance, prob_params, ImpTree.create_adj_matrix_from_simplex_tuple_list(simplex_tuple_list, nb_nodes=prob_instance.nb_nodes, mixed=prob_params.mixed))
  
  """
  Creates an adjacency matrix corresponding to a list of simplex tuples (of size 2 or more) where
    'simplex_tuple_list' is a list of tuples representing simplices in the ImpTree;
      size = 2 --> tuple corresponds to a tree edge;
      size > 2 --> tuple corresponds to a mesh cluster;
    'nb_nodes' (optional) is the number of nodes associated with the instance;
    'mixed' (optional) is whether mesh clusters are allowed (True) or not (False).
  """
  @staticmethod
  def create_adj_matrix_from_simplex_tuple_list(simplex_tuple_list, nb_nodes=-1, mixed=True):
    if nb_nodes == -1:
      nb_nodes = max([max(simplex_tuple) for simplex_tuple in simplex_tuple_list])+1
    adj_matrix = np.zeros((nb_nodes,nb_nodes))
    mesh_idx_counter = 2
    for simplex_tuple in simplex_tuple_list:
      if len(simplex_tuple) == 2:
        adj_matrix[simplex_tuple[0],simplex_tuple[1]] = 1
        adj_matrix[simplex_tuple[1],simplex_tuple[0]] = 1
      else:
        if not mixed:
          print("Illegal Mesh creation : ImpTree.create_imp_tree_from_simplex_tuple_list(...)")
        for v1 in simplex_tuple:
          for v2 in simplex_tuple:
            if v2 != v1:
              adj_matrix[v1,v2] = mesh_idx_counter
        mesh_idx_counter += 1
    return adj_matrix
  
  """
  Creates a list of simplex tuples corresponding to the minimum spanning tree where
    either
      'prob_instance' is the associated ProbInstance to retrieve the distances;
    or
      'edge_dist_pairs' is the ordered list of the tuples ((u,v), dist) in ascending dist;
    'clusters' (optional) is a list of tuples representing mesh clusters;
    'nb_nodes' (optional) is the number of nodes associated with the instance;
  """
  @staticmethod
  def create_min_spanning_tree_simplex_tuple_list(nb_nodes=-1, prob_instance=-1, edge_dist_pairs=[], clusters=[]):
    if nb_nodes < 0:
      nb_nodes = prob_instance.nb_nodes
    if len(edge_dist_pairs) == 0:
      distances_dB = prob_instance.path_losses_dB[:,:,-1] + prob_instance.fade_margins_dB[:,:,-1]
      edge_dist_pairs = []
      for v1 in range(nb_nodes):
        for v2 in range(v1+1,nb_nodes):
          edge_dist_pairs.append(((v1,v2), distances_dB[v1,v2]))
      edge_dist_pairs.sort(key=lambda edge_dist_pair: edge_dist_pair[1])
    simplex_tuple_list = []
    if len(clusters) == 0:
      conn_comps = [{node} for node in range(nb_nodes)]
    else:
      conn_comps = []
      for cluster in clusters:
        cluster_list = list(cluster)
        cluster_list.sort()
        simplex_tuple_list.append(tuple(cluster_list))
        cluster_set = set(cluster)
        does_intersect = False
        for conn_comp in conn_comps:
          if len(conn_comp.intersection(cluster_set)) > 0:
            does_intersect = True
            conn_comp |= cluster_set
            break
        if not does_intersect:
          conn_comps.append(cluster_set)
      for node_idx in range(nb_nodes):
        is_in_a_cluster = False
        for cluster in clusters:
          if node_idx in cluster:
            is_in_a_cluster = True
            break
        if not is_in_a_cluster:
          conn_comps.append({node_idx})
    min_spanning_tree_edges = []
    for edge, _ in edge_dist_pairs:
      v1,v2 = edge
      for conn_comp_idx in range(len(conn_comps)):
        if v1 in conn_comps[conn_comp_idx]:
          break
      v1_conn_comp = conn_comps[conn_comp_idx]
      if not v2 in v1_conn_comp:
        min_spanning_tree_edges.append(edge)
        for conn_comp_idx in range(len(conn_comps)):
          if v2 in conn_comps[conn_comp_idx]:
            break
        v1_conn_comp |= conn_comps.pop(conn_comp_idx)
    for edge in min_spanning_tree_edges:
      edge_list = list(edge)
      edge_list.sort()
      simplex_tuple_list.append(tuple(edge_list))
    simplex_tuple_list.sort(key=lambda list:list[0]*nb_nodes + list[1])
    simplex_tuple_list.sort(key=lambda list:len(list))
    return simplex_tuple_list

  """
  Creates an ImpTree corresponding to a full mesh cluster (a single mesh cluster containing all explicit nodes) where
    'prob_instance' is the ProbInstance;
    'prob_params' is the ProbParameters.
  """
  @staticmethod
  def create_full_mesh(prob_instance, prob_params):
    if not prob_params.mixed:
      print("Illegal Mesh creation : ImpTree.create_full_mesh(...)")
    nb_nodes = prob_instance.nb_nodes
    adj_matrix = 2*np.ones((nb_nodes,nb_nodes))
    for v in range(nb_nodes):
      adj_matrix[v,v] = 0
    return ImpTree(prob_instance, prob_params, adj_matrix)

  """
  Returns the connected components of the ImpTree as a list of lists of explicit nodes.
  """
  def get_conn_components(self):
    return ImpTree.get_adj_matrix_conn_components(self.adj_matrix, nb_nodes=self.prob_instance.nb_nodes)

  """
  Returns the connected components of an adjacency matrix as a list of lists of explicit nodes where
    'adj_matrix' is the adjacency matrix;
    'nb_nodes' (optional) is the number of nodes of the instance.
  """
  @staticmethod
  def get_adj_matrix_conn_components(adj_matrix, nb_nodes=-1):
    if nb_nodes < 0:
      nb_nodes = adj_matrix.shape[0]
    conn_components = []
    unexplored = list(range(nb_nodes))
    while len(unexplored)>0:
      conn_component = []
      nodes_to_explore = [unexplored[0]]
      while len(nodes_to_explore)>0:
        new_node = nodes_to_explore.pop(0)
        if new_node in unexplored:
          unexplored.remove(new_node)
          conn_component.append(new_node)
          neighbors = np.where(adj_matrix[new_node,:]>0)[0].tolist()
          nodes_to_explore += neighbors
      conn_components.append(conn_component)
    return conn_components

  """
  Returns the definition of the ImpTree as a list of simplex tuples with
    size = 2 --> tuple corresponds to a tree edge;
    size > 2 --> tuple corresponds to a mesh cluster.
  """
  def get_simplex_tuple_list(self):
    return ImpTree.get_adj_matrix_simplex_tuple_list(self.adj_matrix)

  """
  Returns the definition of an adjacency matrix as a list of simplex tuples with
    size = 2 --> tuple corresponds to a tree edge;
    size > 2 --> tuple corresponds to a mesh cluster;
  where
    'adj_matrix' is the adjacency matrix.
  """
  @staticmethod
  def get_adj_matrix_simplex_tuple_list(adj_matrix):
    edge_where = np.where(adj_matrix == 1)
    simplices = [(edge_where[0][idx], edge_where[1][idx]) for idx in range(len(edge_where[0])) if edge_where[0][idx] < edge_where[1][idx]]
    max_mesh_idx = np.max(adj_matrix)
    if max_mesh_idx >= 2:
      for mesh_idx in range(2, int(max_mesh_idx)+1):
        simplex = tuple(set(np.where(adj_matrix == mesh_idx)[0]))
        if len(simplex) > 0:
          simplices.append(tuple(set(np.where(adj_matrix == mesh_idx)[0])))
    return simplices
  
  """
  Returns the sizes of the ImpTree's mesh clusters as a list of integers. If there are no mesh clusters, the returned list is empty.
  """
  def get_mesh_sizes(self):
    max_mesh_idx = int(np.max(self.adj_matrix))
    mesh_sizes = []
    for mesh_abs_idx in range(2, max_mesh_idx+1):
      len_curr_mesh_members = len(set(np.where(self.adj_matrix == mesh_abs_idx)[0]))
      if len_curr_mesh_members > 0:
        mesh_sizes.append(len_curr_mesh_members)
    mesh_sizes.sort(reverse=True)
    return mesh_sizes

  """
  Returns the objective/pseudo-objective value of the ImpTree where
    'obj_func' is the identifier of the desired objective or pseudo-objective
      "full" --> real objective (very slow), used only once at the end of the search for small instances (nb_nodes <= 10);
      "greedy" --> accurate but slow pseudo-objective, used for every new current solution in the topology search main loop;
      "chan_avg" --> inaccurate but fast pseudo-objective, used in the neighborhood searches in the topology search main loop;
      "chan_max" --> fast upper bound pseudo-objective (actual upper bound if no mesh cluster), used to decide if "greedy" is worth evaluating;
      "avg" --> very inaccurate but very fast pseudo-objective, never used;
    'get_arb' (optional) is whether the best ImpArborescence found (associated with the returned value) should be returned as well (True) or not (False);
    'get_network' (optional) is whether the best Network found (associated with the returned value) should be returned as well (True) or not (False);
    'MH' (optional) is a fixed master hub (associated with a fixed ImpArborescence) to evaluate the ImpTree, by default there are none;
    'master_hub_preprocessing' (optional) is whether to reduce the number of potential master hubs which is faster but can miss the optimal, by default there is preprocessing.

  Examples:
    best_obj_value = imp_tree.get_obj_value(obj_func = "greedy")
    best_obj_value, best_arb = imp_tree.get_obj_value(obj_func = "greedy", get_arb = True)
    best_obj_value, best_network = imp_tree.get_obj_value(obj_func = "greedy", get_network = True)
    best_obj_value, best_arb, best_network = imp_tree.get_obj_value(obj_func = "greedy", get_arb = True, get_network = True)
  """
  def get_obj_value(self, obj_func = "full", get_arb = False, get_network = False, MH=-1, master_hub_preprocessing=True):
    best_obj_value = -2
    if True: # mesh clusters and memberships
      mesh_members = {}
      meshes = {v : [] for v in range(self.prob_instance.nb_nodes)}
      if self.prob_params.mixed:
        max_mesh_idx = np.max(self.adj_matrix)
        if max_mesh_idx >= 2:
          for mesh_abs_idx in range(2, int(max_mesh_idx)+1):
            mesh_idx = -1*mesh_abs_idx
            curr_mesh_members = [int(npint) for npint in set(np.where(self.adj_matrix == mesh_abs_idx)[0])]
            if len(curr_mesh_members) > 0:
              mesh_members[mesh_idx] = curr_mesh_members
              for mesh_member in curr_mesh_members:
                meshes[mesh_member].append(mesh_idx)
    if MH == -1:
      potential_MHs = []
      for v in range(self.prob_instance.nb_nodes):
        nb_adj = np.sum(self.adj_matrix[v,:]==1)
        if np.max(self.adj_matrix[v,:]) >= 2:
          mesh_idcs_set = set((self.adj_matrix[v,:][self.adj_matrix[v,:]>=2]).tolist())
          nb_adj += len(mesh_idcs_set)
        if nb_adj > 1 or self.prob_instance.nb_nodes <= 2:
          potential_MHs.append(v)
      if len(potential_MHs) == 0: # network is a single mesh
        potential_MHs = [int(npint) for npint in set(np.where(self.adj_matrix > 0)[0])]
    else:
      potential_MHs = [MH]

    if master_hub_preprocessing: # additional pre-processing (saves between 0 and 40% of the time for same obj_val (tested for generalized min_spanning_trees of instances of 10-50 nodes))
      # Floyd-Warshall
      dists = 10000 * np.ones((self.prob_instance.nb_nodes,self.prob_instance.nb_nodes))
      dists[self.adj_matrix==1] = 1
      for mesh_idx in mesh_members.keys():
        members = mesh_members[mesh_idx]
        for member_idx1 in range(len(members)):
          member1 = members[member_idx1]
          for member_idx2 in range(member_idx1+1,len(members)):
            member2 = members[member_idx2]
            dists[member2,member1] = dists[member1,member2] = 1
      for node1 in range(self.prob_instance.nb_nodes):
        dists[node1,node1] = 0
      for node3 in range(self.prob_instance.nb_nodes):
        for node1 in range(self.prob_instance.nb_nodes):
          for node2 in range(node1+1,self.prob_instance.nb_nodes):
            loc_dist = dists[node1,node3] + dists[node3,node2]
            if loc_dist < dists[node1,node2]:
              dists[node2,node1] = dists[node1,node2] = loc_dist
      dists = np.max(dists, axis=1).tolist()
      ALL_DATA = []

      for potential_MH in potential_MHs:
        if potential_MH in meshes:
          nb_meshes = len(meshes[potential_MH])
        else:
          nb_meshes = 0
        ALL_DATA.append([potential_MH, np.sum(self.adj_matrix[potential_MH,:]==1) + nb_meshes, nb_meshes, round(dists[potential_MH])])
        
      data_ordered_vals1 = list(set([data[1] for data in ALL_DATA]))
      data_ordered_vals1.sort()
      data_ordered_vals2 = list(set([data[2] for data in ALL_DATA]))
      data_ordered_vals2.sort()
      data_ordered_vals3 = list(set([data[3] for data in ALL_DATA]))
      data_ordered_vals3.sort(reverse=True)

      if len(mesh_members) == 0:
        mesh_coeff = 0
      else:
        mesh_coeff = mean([len(mesh_members[mesh_idx])-1 for mesh_idx in mesh_members.keys()])
      for data in ALL_DATA:
        data.append(data_ordered_vals1.index(data[1]) + mesh_coeff*data_ordered_vals2.index(data[2]) + data_ordered_vals3.index(data[3]))

      data_ordered_vals4 = list(set([data[4] for data in ALL_DATA]))
      data_ordered_vals4.sort()
      data_ordered_nb_vals4 = [sum([1 for data in ALL_DATA if data[4] == ordered_val4]) for ordered_val4 in data_ordered_vals4]
      total_nb_vals4 = sum(data_ordered_nb_vals4)
      data_ordered_cumul_quantile4 = [sum(data_ordered_nb_vals4[:ordered_nb_val_idx+1])/total_nb_vals4 for ordered_nb_val_idx in range(len(data_ordered_nb_vals4))]

      potential_MHs = [potential_MH for potential_MH,_,_,_,data4 in ALL_DATA if data_ordered_cumul_quantile4[data_ordered_vals4.index(data4)] >= 0.5]
    
    is_full_mesh = len(mesh_members) == 1 and len(list(mesh_members.values())[0]) == self.prob_instance.nb_nodes
    if not is_full_mesh:
      for potential_MH in potential_MHs:
        imp_arb = ImpArborescence(self, potential_MH, mesh_members=mesh_members, meshes=meshes)
        if imp_arb.max_nb_nodes_per_PTM_connection <= self.prob_params.max_nb_nodes_per_PTM_connection and imp_arb.nb_required_channels <= 2:
          if get_network or obj_func == "avg" or obj_func == "chan_avg" or obj_func == "chan_max":
            if obj_func == "avg" and self.has_old_avg:
              obj_value,network = imp_arb.get_obj_value(obj_func="avg", get_network=True, has_old_avg=True, old_imp_arb_adj_matrix=self.old_arb_adj_matrix, old_MH_succs_2_partition=self.old_MH_succs_2_partition, old_direct_simp_TPs=self.old_direct_simp_TPs, old_mesh_members=self.old_mesh_members, old_preds=self.old_preds, old_mesh_routes=self.old_mesh_routes)
            elif obj_func == "chan_avg" and self.has_old_chan_avg:
              obj_value,network = imp_arb.get_obj_value(obj_func="chan_avg", get_network=True, has_old_chan_avg=True, old_imp_arb_adj_matrix=self.old_arb_adj_matrix, old_MH_succs_2_partition=self.old_MH_succs_2_partition, old_chan_assignment=self.old_chan_assignment, old_direct_simp_TPs=self.old_direct_simp_TPs, old_mesh_members=self.old_mesh_members, old_preds=self.old_preds, old_mesh_routes=self.old_mesh_routes)
            elif obj_func == "chan_max" and self.has_old_chan_max:
              obj_value,network = imp_arb.get_obj_value(obj_func="chan_max", get_network=True, has_old_chan_max=True, old_imp_arb_adj_matrix=self.old_arb_adj_matrix, old_MH_succs_2_partition=self.old_MH_succs_2_partition, old_chan_assignment=self.old_chan_assignment, old_direct_simp_TPs=self.old_direct_simp_TPs, old_mesh_members=self.old_mesh_members, old_preds=self.old_preds, old_mesh_routes=self.old_mesh_routes)
            else:
              obj_value,network = imp_arb.get_obj_value(obj_func=obj_func, get_network=True)
          else:
            obj_value = imp_arb.get_obj_value(obj_func=obj_func)
          if obj_value > best_obj_value:
            best_obj_value = obj_value
            if get_arb:
              best_arb = imp_arb
            if get_network or obj_func == "avg" or obj_func == "chan_avg" or obj_func == "chan_max":
              best_network = network
        elif best_obj_value == -2:
          best_obj_value = -1
          if get_arb:
            best_arb = imp_arb
          if get_network:
            best_network = Network(imp_arb, [imp_arb.succ_lists[imp_arb.MH],[]])
            best_network.direct_TPs = np.zeros((self.prob_instance.nb_nodes,self.prob_instance.nb_nodes))
            if best_network.is_mixed:
              best_network.A_mesh_routes = dict()
              best_network.B_mesh_TP_denoms = best_network.C_mesh_TP_denoms = np.zeros((self.prob_instance.nb_nodes,self.prob_instance.nb_nodes))
            best_network.A_obj_min = best_network.A_obj_value = best_network.B_obj_min = best_network.B_obj_value = best_network.C_obj_min = best_network.C_obj_value = best_network.obj_value = -1
            best_network.has_obj_value = True # boolean
            best_network.A_lim_links = []
            best_network.B_lim_links = []
            best_network.C_lim_links = []
    else:
      if obj_func == "greedy":
        network_obj_func = "full"
      else:
        network_obj_func = obj_func
      mesh_idx = list(mesh_members.keys())[0]
      old_MH = potential_MHs[-1]
      imp_arb = ImpArborescence(self, old_MH, mesh_members=mesh_members, meshes=meshes)
      MH_succs_2_partitions = imp_arb.get_all_2_partitions()
      MH_succs_2_partition = MH_succs_2_partitions[0]
      chan_assignments = imp_arb.get_all_chan_assignments(MH_succs_2_partition)
      if network_obj_func != "full" or self.prob_params.nb_freqs_per_channel == 1:
        freq_assignments = chan_assignments
      else:
        freq_assignments = []
        for chan_assignment in chan_assignments:
          loc_chan = chan_assignment[old_MH][0]
          for loc_freq_idx in range(2*(-2 - loc_chan), 2*(-2 - loc_chan)+2):
            freq_assignments.append({old_MH:[loc_freq_idx,-1], mesh_idx:loc_freq_idx})
      for freq_assignment in freq_assignments:
        old_MH = potential_MHs[-1]
        imp_arb = ImpArborescence(self, old_MH, mesh_members=mesh_members, meshes=meshes)
        network = Network(imp_arb, MH_succs_2_partition, freq_assignment)
        SINRs = network.get_obj_value(obj_func=network_obj_func, return_SINRs_after_direct_TPs=True)
        for potential_MH in potential_MHs:
          imp_arb = network.imp_arb
          imp_arb.MH = potential_MH
          imp_arb.preds[old_MH] = mesh_idx
          imp_arb.preds.pop(potential_MH)
          imp_arb.preds[mesh_idx] = potential_MH
          imp_arb.succ_lists[old_MH] = []
          imp_arb.succ_lists[potential_MH] = [mesh_idx]
          imp_arb.succ_lists[mesh_idx] = [mesh_member for mesh_member in mesh_members[mesh_idx] if mesh_member != potential_MH]
          if network.has_freq_assignment:
            network.freq_assignment[potential_MH] = network.freq_assignment.pop(old_MH)
          network.uplink_ant_alignments_rad[old_MH] = network.uplink_ant_alignments_rad.pop(potential_MH)
          network.uplink_ant_nb_beams[old_MH] = network.uplink_ant_nb_beams.pop(potential_MH)
          network.downlink_ant_alignments_rad[potential_MH] = network.downlink_ant_alignments_rad.pop(old_MH)
          network.downlink_ant_beams[potential_MH] = network.downlink_ant_beams.pop(old_MH)
          old_MH = potential_MH
          network.get_effective_TPs(network_obj_func=="full", [], -1, SINRs, keep_previous_routes=True)
          obj_value = network.obj_value
          if obj_value > best_obj_value:
            best_obj_value = obj_value
            if get_network or obj_func == "avg" or obj_func == "chan_avg" or obj_func == "chan_max":
              best_network = copy.deepcopy(network)
              if get_arb:
                best_arb = best_network.imp_arb
            else:
              if get_arb:
                best_arb = copy.deepcopy(imp_arb)

    if best_obj_value > -1:
      if obj_func == "avg":
        self.has_old_avg = True
        self.has_old_chan_avg = False
        self.has_old_chan_max = False
        self.old_arb_adj_matrix = best_network.imp_arb.adj_matrix # (nb_exp_nodes, nb_exp_nodes)
        self.old_MH_succs_2_partition = best_network.MH_succs_2_partition # [[],[]]
        self.old_direct_simp_TPs = best_network.direct_TPs # (nb_exp_nodes, nb_exp_nodes)
        self.old_mesh_members = best_network.imp_arb.mesh_members # dict : mesh -> list of mesh members
        if best_network.is_mixed:
          self.old_preds = best_network.imp_arb.preds # dict : imp_node -> predecessor
          self.old_mesh_routes = best_network.A_mesh_routes # dict : mesh -> dict : mesh_member1 -> dict : mesh_member2 -> list of mesh edges used between member1 and member2
        else:
          self.old_preds = -1
          self.old_mesh_routes = -1
      elif obj_func == "chan_avg" or obj_func == "chan_max":
        self.has_old_avg = False
        if obj_func == "chan_avg":
          self.has_old_chan_avg = True
          self.has_old_chan_max = False
        else:
          self.has_old_chan_avg = False
          self.has_old_chan_max = True
        self.old_arb_adj_matrix = best_network.imp_arb.adj_matrix # (nb_exp_nodes, nb_exp_nodes)
        self.old_MH_succs_2_partition = best_network.MH_succs_2_partition # [[],[]]
        self.old_chan_assignment = best_network.freq_assignment # dict : imp_node -> freq of successor links (MH -> [freq_part1,freq_part2])
        self.old_direct_simp_TPs = best_network.direct_TPs # (nb_exp_nodes, nb_exp_nodes)
        self.old_mesh_members = best_network.imp_arb.mesh_members # dict : mesh -> list of mesh members
        if best_network.is_mixed:
          self.old_preds = best_network.imp_arb.preds # dict : imp_node -> predecessor
          self.old_mesh_routes = best_network.A_mesh_routes # dict : mesh -> dict : mesh_member1 -> dict : mesh_member2 -> list of mesh edges used between member1 and member2
        else:
          self.old_preds = -1
          self.old_mesh_routes = -1
      elif obj_func == "greedy" or obj_func == "full":
        self.obj_value = best_obj_value # float
        self.has_obj_value = True # boolean
    if get_arb:
      if get_network:
        return best_obj_value, best_arb, best_network
      else:
        return best_obj_value, best_arb
    else:
      if get_network:
        return best_obj_value, best_network
      else:
        return best_obj_value
  
  """
  Moves to the best neighbor ImpTree and returns the dropped and added components as well as whether the full neighborhood subset was visited where
    'neighborhood' is the identifier of the desired neighborhood
      "tree" is the tree neighborhood (tree edge swap);
      "mesh" is the mesh neighborhood (creation, destruction, inclusion, exclusion, fusion);
    'taboo_lists' (optional) is a list of 2 lists, the first is the list of taboo drop components and the second is the list of taboo add components;
    'obj_func' (optional) is the identifier of the desired objective or pseudo-objective (see get_obj_value(...) above for details);
    'best_neighb_obj_value' (optional) is the best value seen so far of the 'obj_func' used in these neighborhood searches;
    'first_better' (optional) is whether the neighborhood search should be done as a first improvement search (True) or a best improvement search (False);
    'neighborhood_subset_ratio' (optional) is a float in (0,1] that describes the ratio of the neighborhood that should be searched;
    'MH' (optional) is a fixed master hub to evaluate the neighbors, by default there are none;
    'ordered_edges' (optional) is a list of ordered tuples (representing edges) describing in what order they should be tried;
    'use_elite_edge_filter' (optional) is whether to use a given 'elite_edge_filter_matrix' to filter the possible tree edges;
    'elite_edge_filter_matrix' (optional) is a numpy matrix of boolean of shape (nb_nodes, nb_nodes) that describes which tree edges are possible of which only the top triangle is used;
    'mesh_compatibility_matrix' (optional) is a numpy matrix of boolean of shape (nb_nodes, nb_nodes) that describes which mesh edges are possible;
    'ordered_cluster_lists' is a list of lists of lists of explicit nodes representing the potential mesh clusters for descending levels of minimum signal in the mesh edges (see
      MWMB_Algorithm/search(...)/TOPOLOGICAL DATA ANALYSIS for details);
    'ordered_cluster_adj_matrices' (optional) is a list of numpy matrices of boolean of shape (nb_nodes, nb_nodes) that describe which mesh edges are possible for descending levels of
      minimum signal in the mesh edges (see MWMB_Algorithm/search(...)/TOPOLOGICAL DATA ANALYSIS for details);
    'verbose' (optional) is whether the intermediary details should be printed in the terminal;
    'log_files' (optional) is a list of open .txt files in which to print the intermediary details.
  """
  def get_best_neighbor(self, neighborhood="tree", taboo_lists=[], obj_func = "chan_avg", best_neighb_obj_value=float('inf'), first_better=False, neighborhood_subset_ratio=1, ordered_edges=[], MH=-1, use_elite_edge_filter=False, elite_edge_filter_matrix=-1, mesh_compatibility_matrix=-1, ordered_cluster_lists=-1, ordered_cluster_adj_matrices=-1, verbose=False, log_files=[]):
    best_move_value_taboo_triples, visited_all_neighbors = self.get_best_moves(nb_best_moves=1, neighborhood=neighborhood, taboo_lists=taboo_lists, obj_func=obj_func, best_neighb_obj_value=best_neighb_obj_value, MH=MH, first_better=first_better, neighborhood_subset_ratio=neighborhood_subset_ratio, ordered_edges=ordered_edges, use_elite_edge_filter=use_elite_edge_filter, elite_edge_filter_matrix=elite_edge_filter_matrix, mesh_compatibility_matrix=mesh_compatibility_matrix, ordered_cluster_lists=ordered_cluster_lists, ordered_cluster_adj_matrices=ordered_cluster_adj_matrices, verbose=verbose, log_files=log_files)
    best_move = best_move_value_taboo_triples[0][0]
    if neighborhood == "tree":
      dropped, added = ImpTree.make_edge_swap_move(self.adj_matrix, best_move)
      MWMB_Plot.print_log("  "+str([dropped[0][0],dropped[0][1]])+" X --> "+str([added[0][0],added[0][1]]), verbose, files=log_files)
    else: # neighborhood == "mesh":
      dropped, added = ImpTree.make_mesh_move(self.adj_matrix, best_move)
      MWMB_Plot.print_log("  "+str(best_move), verbose, files=log_files)
    if best_move_value_taboo_triples[0][2]:
      MWMB_Plot.print_log("  is taboo !!", verbose, files=log_files)
    return dropped, added, visited_all_neighbors
  
  """
  Returns the best moves found in the neighborhood as a list of tuples (containing the move, its value and its taboo status) as well as whether the full neighborhood subset was visited
  where
    'nb_best_moves' is the number of desired best moves in the neighborhood;
    'neighborhood' is the identifier of the desired neighborhood
      "tree" is the tree neighborhood (tree edge swap);
      "mesh" is the mesh neighborhood (creation, destruction, inclusion, exclusion, fusion);
    'taboo_lists' (optional) is a list of 2 lists, the first is the list of taboo drop components and the second is the list of taboo add components;
    'obj_func' (optional) is the identifier of the desired objective or pseudo-objective (see get_obj_value(...) above for details);
    'best_neighb_obj_value' (optional) is the best value seen so far of the 'obj_func' used in these neighborhood searches;
    'first_better' (optional) is whether the neighborhood search should be done as a first improvement search (True) or a best improvement search (False);
    'neighborhood_subset_ratio' (optional) is a float in (0,1] that describes the ratio of the neighborhood that should be searched;
    'MH' (optional) is a fixed master hub to evaluate the neighbors, by default there are none;
    'ordered_edges' (optional) is a list of ordered tuples (representing edges) describing in what order they should be tried;
    'use_elite_edge_filter' (optional) is whether to use a given 'elite_edge_filter_matrix' to filter the possible tree edges;
    'elite_edge_filter_matrix' (optional) is a numpy matrix of boolean of shape (nb_nodes, nb_nodes) that describes which tree edges are possible of which only the top triangle is used;
    'mesh_compatibility_matrix' (optional) is a numpy matrix of boolean of shape (nb_nodes, nb_nodes) that describes which mesh edges are possible;
    'ordered_cluster_lists' is a list of lists of lists of explicit nodes representing the potential mesh clusters for descending levels of minimum signal in the mesh edges (see
      MWMB_Algorithm/search(...)/TOPOLOGICAL DATA ANALYSIS for details);
    'ordered_cluster_adj_matrices' (optional) is a list of numpy matrices of boolean of shape (nb_nodes, nb_nodes) that describe which mesh edges are possible for descending levels of
      minimum signal in the mesh edges (see MWMB_Algorithm/search(...)/TOPOLOGICAL DATA ANALYSIS for details);
    'verbose' (optional) is whether the intermediary details should be printed in the terminal;
    'log_files' (optional) is a list of open .txt files in which to print the intermediary details.
  """
  def get_best_moves(self, nb_best_moves=1, neighborhood="tree", taboo_lists=[], obj_func = "chan_avg", best_neighb_obj_value=float('inf'), MH=-1, first_better=False, neighborhood_subset_ratio=1, ordered_edges=[], use_elite_edge_filter=False, elite_edge_filter_matrix=-1, mesh_compatibility_matrix=-1, ordered_cluster_lists=-1, ordered_cluster_adj_matrices=-1, verbose=False, log_files=[]):
    if len(taboo_lists) == 2:
      taboo_drop_list = taboo_lists[0]
      taboo_add_list = taboo_lists[1]
    else:
      taboo_add_list = []
      taboo_drop_list = []
    if neighborhood == "tree":
      if not first_better:
        return self.get_best_edge_swap_moves(nb_best_moves=nb_best_moves, taboo_drop_list=taboo_drop_list, taboo_add_list=taboo_add_list, obj_func=obj_func, best_neighb_obj_value=best_neighb_obj_value, MH=MH, use_elite_edge_filter=use_elite_edge_filter, elite_edge_filter_matrix=elite_edge_filter_matrix, verbose=verbose, log_files=log_files)
      else:
        return self.get_first_better_edge_swap_moves(nb_best_moves=nb_best_moves, neighborhood_subset_ratio=neighborhood_subset_ratio, taboo_drop_list=taboo_drop_list, taboo_add_list=taboo_add_list, obj_func=obj_func, best_neighb_obj_value=best_neighb_obj_value, MH=MH, ordered_edges=ordered_edges, use_elite_edge_filter=use_elite_edge_filter, elite_edge_filter_matrix=elite_edge_filter_matrix, verbose=verbose, log_files=log_files)
    else:# neighborhood == "mesh"
      if not first_better:
       return self.get_best_mesh_moves(nb_best_moves=nb_best_moves, taboo_drop_list=taboo_drop_list, taboo_add_list=taboo_add_list, obj_func=obj_func, best_neighb_obj_value=best_neighb_obj_value, MH=MH, use_elite_edge_filter=use_elite_edge_filter, elite_edge_filter_matrix=elite_edge_filter_matrix, mesh_compatibility_matrix=mesh_compatibility_matrix, verbose=verbose, log_files=log_files)
      else:
       return self.get_first_better_mesh_moves(nb_best_moves=nb_best_moves, neighborhood_subset_ratio=neighborhood_subset_ratio, taboo_drop_list=taboo_drop_list, taboo_add_list=taboo_add_list, obj_func=obj_func, best_neighb_obj_value=best_neighb_obj_value, MH=MH, ordered_edges=ordered_edges, use_elite_edge_filter=use_elite_edge_filter, elite_edge_filter_matrix=elite_edge_filter_matrix, ordered_cluster_lists=ordered_cluster_lists, ordered_cluster_adj_matrices=ordered_cluster_adj_matrices, verbose=verbose, log_files=log_files)
  
  """
  Returns the best moves found in the best improvement search of the tree neighborhood as a list of tuples (containing the move, its value and its taboo status) as well as whether the
  full neighborhood subset was visited where
    'nb_best_moves' is the number of desired best moves in the neighborhood;
    'taboo_drop_list' (optional) is a list of taboo drop components;
    'taboo_add_list' (optional) is a list of taboo add components;
    'obj_func' (optional) is the identifier of the desired objective or pseudo-objective (see get_obj_value(...) above for details);
    'best_neighb_obj_value' (optional) is the best value seen so far of the 'obj_func' used in these neighborhood searches;
    'MH' (optional) is a fixed master hub to evaluate the neighbors, by default there are none;
    'use_elite_edge_filter' (optional) is whether to use a given 'elite_edge_filter_matrix' to filter the possible tree edges;
    'elite_edge_filter_matrix' (optional) is a numpy matrix of boolean of shape (nb_nodes, nb_nodes) that describes which tree edges are possible of which only the top triangle is used;
    'verbose' (optional) is whether the intermediary details should be printed in the terminal;
    'log_files' (optional) is a list of open .txt files in which to print the intermediary details.
  """
  def get_best_edge_swap_moves(self, nb_best_moves=1, taboo_drop_list=[], taboo_add_list=[], obj_func = "chan_avg", best_neighb_obj_value=float('inf'), MH=-1, use_elite_edge_filter=False, elite_edge_filter_matrix=-1, verbose=False, log_files=[]):
    adj_matrix = self.adj_matrix
    w1s,w2s = np.where(self.adj_matrix==1)
    best_move_value_taboo_triples = [([],-1,False)]
    best_taboo_move_value_double = ([],-1)
    nb_neighbors = 0
    nb_taboo_neighbors = 0
    for i in range(w1s.size):
      w1 = w1s[i]
      w2 = w2s[i]
      if w2 > w1:
        is_taboo_drop = (w1,w2) in taboo_drop_list
        adj_matrix[w1,w2] = 0
        adj_matrix[w2,w1] = 0
        conn_components = self.get_conn_components()
        conn_component_1 = conn_components[0]
        conn_component_2 = conn_components[1]
        for w3p in conn_component_1:
          for w4p in conn_component_2:
            if w4p > w3p:
              w3 = w3p
              w4 = w4p
            else:
              w3 = w4p
              w4 = w3p
            if not (w3 == w1 and w4 == w2) and (not use_elite_edge_filter or elite_edge_filter_matrix[w3,w4]):
              nb_neighbors += 1
              is_taboo_add = (w3,w4) in taboo_add_list
              if is_taboo_add or is_taboo_drop:
                nb_taboo_neighbors += 1
              adj_matrix[w3,w4] = 1
              adj_matrix[w4,w3] = 1
              obj_value = self.get_obj_value(obj_func=obj_func, MH=MH)
              if (not is_taboo_drop and not is_taboo_add) or obj_value > best_neighb_obj_value:
                if obj_value > best_move_value_taboo_triples[0][1]:
                  move_value_taboo_triple = ([(w1,w2),(w3,w4)], obj_value, is_taboo_drop or is_taboo_add)
                  best_move_value_taboo_triples.append(move_value_taboo_triple)
                  best_move_value_taboo_triples.sort(key = lambda best_move_value_taboo_triple: best_move_value_taboo_triple[1])
                  best_move_value_taboo_triples = best_move_value_taboo_triples[-nb_best_moves:]
              elif best_move_value_taboo_triples == [([],-1,False)] and obj_value > best_taboo_move_value_double[1]:
                best_taboo_move_value_double = ([(w1,w2),(w3,w4)], obj_value)
              adj_matrix[w3,w4] = 0
              adj_matrix[w4,w3] = 0
        adj_matrix[w1,w2] = 1
        adj_matrix[w2,w1] = 1
    MWMB_Plot.print_log("  Number of neighbors = "+str(nb_neighbors)+"   (|taboo| = "+str(nb_taboo_neighbors)+"   |non-taboo| = "+str(nb_neighbors-nb_taboo_neighbors)+")", verbose, files=log_files)
    if best_move_value_taboo_triples[0] == ([],-1,False):
      best_move_value_taboo_triples = best_move_value_taboo_triples[1:]
    return best_move_value_taboo_triples,True
  
  """
  Returns the best moves found in the first improvement search of the tree neighborhood as a list of tuples (containing the move, its value and its taboo status) as well as whether the
  full neighborhood subset was visited where
    'nb_best_moves' is the number of desired best moves in the neighborhood;
    'taboo_drop_list' (optional) is a list of taboo drop components;
    'taboo_add_list' (optional) is a list of taboo add components;
    'obj_func' (optional) is the identifier of the desired objective or pseudo-objective (see get_obj_value(...) above for details);
    'best_neighb_obj_value' (optional) is the best value seen so far of the 'obj_func' used in these neighborhood searches;
    'neighborhood_subset_ratio' (optional) is a float in (0,1] that describes the ratio of the neighborhood that should be searched;
    'MH' (optional) is a fixed master hub to evaluate the neighbors, by default there are none;
    'ordered_edges' (optional) is a list of ordered tuples (representing edges) describing in what order they should be tried;
    'use_elite_edge_filter' (optional) is whether to use a given 'elite_edge_filter_matrix' to filter the possible tree edges;
    'elite_edge_filter_matrix' (optional) is a numpy matrix of boolean of shape (nb_nodes, nb_nodes) that describes which tree edges are possible of which only the top triangle is used;
    'verbose' (optional) is whether the intermediary details should be printed in the terminal;
    'log_files' (optional) is a list of open .txt files in which to print the intermediary details.
  """
  def get_first_better_edge_swap_moves(self, nb_best_moves=1, neighborhood_subset_ratio=1, taboo_drop_list=[], taboo_add_list=[], obj_func = "chan_avg", best_neighb_obj_value=float('inf'), MH=-1, ordered_edges=[], use_elite_edge_filter=False, elite_edge_filter_matrix=-1, verbose=False, log_files=[]):
    adj_matrix = self.adj_matrix
    best_move_value_taboo_triples = [([],-1,False)]
    init_obj_value = self.get_obj_value(obj_func=obj_func, MH=MH)
    w1s,w2s = np.where(self.adj_matrix==1)
    tree_edges = [(w1s[i],w2s[i]) for i in range(w1s.size) if w2s[i] > w1s[i]]
    curr_tree_edges_to_visit = copy.copy(tree_edges)
    tree_edges_add_edges_to_visit = {}
    found_better = False
    visited_all_neighbors = True
    nb_known_edges = 0
    nb_known_neighbors = 0
    nb_visited_neighbors = 0
    nb_taboo_visited_neighbors = 0
    while len(curr_tree_edges_to_visit) > 0:
      w1,w2 = random.choice(curr_tree_edges_to_visit)
      if not (w1,w2) in tree_edges_add_edges_to_visit:
        adj_matrix[w1,w2] = 0
        adj_matrix[w2,w1] = 0
        conn_components = self.get_conn_components()
        conn_component_1 = conn_components[0]
        conn_component_2 = conn_components[1]
        if len(ordered_edges) > 0:
          is_valid_edge_matrix = np.zeros(adj_matrix.shape)
          is_valid_edge_matrix[np.ix_(conn_component_1,conn_component_2)] = 1
          is_valid_edge_matrix += is_valid_edge_matrix.T
          tree_edges_add_edges_to_visit[(w1,w2)] = [(w3,w4) for w3,w4 in ordered_edges if is_valid_edge_matrix[w3,w4] and {w3,w4} != {w1,w2}]
        else:
          if not use_elite_edge_filter:
            tree_edges_add_edges_to_visit[(w1,w2)] = [(w3,w4) if w4 > w3 else (w4,w3) for w3 in conn_component_1 for w4 in conn_component_2 if {w3,w4} != {w1,w2}]
          else:
            tree_edges_add_edges_to_visit[(w1,w2)] = [(w3,w4) if w4 > w3 else (w4,w3) for w3 in conn_component_1 for w4 in conn_component_2 if {w3,w4} != {w1,w2} and (elite_edge_filter_matrix[w3,w4] or elite_edge_filter_matrix[w4,w3])]
        nb_known_edges += 1
        nb_known_neighbors += len(tree_edges_add_edges_to_visit[(w1,w2)])
        if len(tree_edges_add_edges_to_visit[(w1,w2)]) > 0 and neighborhood_subset_ratio < 1:
          tree_edges_add_edges_to_visit[(w1,w2)] = ImpTree.sample_ordered_list_with_more_weight_in_beginning(tree_edges_add_edges_to_visit[(w1,w2)], neighborhood_subset_ratio)
        if len(tree_edges_add_edges_to_visit[(w1,w2)]) == 0:
          del tree_edges_add_edges_to_visit[(w1,w2)]
          curr_tree_edges_to_visit.remove((w1,w2))
          adj_matrix[w1,w2] = 1
          adj_matrix[w2,w1] = 1
          continue
      else:
        adj_matrix[w1,w2] = 0
        adj_matrix[w2,w1] = 0
      is_taboo_drop = (w1,w2) in taboo_drop_list
      len_edges_to_visit = len(tree_edges_add_edges_to_visit[(w1,w2)])
      if len_edges_to_visit > 1:
        if len(ordered_edges) > 0:
          w3,w4 = tree_edges_add_edges_to_visit[(w1,w2)].pop(0)
        else:
          w3,w4 = tree_edges_add_edges_to_visit[(w1,w2)].pop(random.randint(0,len_edges_to_visit-1))
      else:
        w3,w4 = tree_edges_add_edges_to_visit[(w1,w2)][0]
        del tree_edges_add_edges_to_visit[(w1,w2)]
        curr_tree_edges_to_visit.remove((w1,w2))
      adj_matrix[w3,w4] = 1
      adj_matrix[w4,w3] = 1
      is_taboo_add = (w3,w4) in taboo_add_list
      obj_value = self.get_obj_value(obj_func=obj_func, MH=MH)
      nb_visited_neighbors += 1
      if is_taboo_drop or is_taboo_add:
        nb_taboo_visited_neighbors += 1
      adj_matrix[w3,w4] = 0
      adj_matrix[w4,w3] = 0
      adj_matrix[w1,w2] = 1
      adj_matrix[w2,w1] = 1
      if (not is_taboo_drop and not is_taboo_add) or obj_value > best_neighb_obj_value:
        if obj_value > best_move_value_taboo_triples[0][1]:
          move_value_taboo_triple = ([(w1,w2),(w3,w4)], obj_value, is_taboo_drop or is_taboo_add)
          best_move_value_taboo_triples.append(move_value_taboo_triple)
          best_move_value_taboo_triples.sort(key = lambda best_move_value_taboo_triple: best_move_value_taboo_triple[1])
          best_move_value_taboo_triples = best_move_value_taboo_triples[-nb_best_moves:]
          if obj_value > init_obj_value:
            found_better = True
            found_better_counter = nb_best_moves
      if found_better:
        found_better_counter -= 1
        if found_better_counter == 0 and len(curr_tree_edges_to_visit) > 0:
          visited_all_neighbors = False
          break
    if best_move_value_taboo_triples[0] == ([],-1,False):
      best_move_value_taboo_triples = best_move_value_taboo_triples[1:]
    if nb_known_edges > 0:
      nb_approx_neighbors = round(nb_known_neighbors*(len(tree_edges)/nb_known_edges))
      MWMB_Plot.print_log("  Approx number of neighbors = "+str(nb_approx_neighbors)+"   (|known| = "+str(nb_known_neighbors)+");   |visited| = "+str(nb_visited_neighbors)+"   (|taboo| = "+str(nb_taboo_visited_neighbors)+"   |non-taboo| = "+str(nb_visited_neighbors-nb_taboo_visited_neighbors)+")      \tVisited "+str(round(100*nb_visited_neighbors/nb_approx_neighbors))+"%", verbose, files=log_files)
    else:
      MWMB_Plot.print_log("  Approx number of neighbors = 0   (|known| = 0);   |visited| = 0   (|taboo| = 0   |non-taboo| = 0)      \tVisited ---%", verbose, files=log_files)
    return best_move_value_taboo_triples,visited_all_neighbors

  """
  Samples an ordered list with linearly more probability the earlier the elements are in the list such that the first has about twice the probability of the last where
    'list' is the ordered list to be sampled;
    'ratio' is a float (0,1] that describes the expectation of the ratio of the size of the sampled list relative to the size of the original list.
  """
  @staticmethod
  def sample_ordered_list_with_more_weight_in_beginning(list, ratio):
    idcs = np.random.choice(range(len(list)), size=ImpTree.random_int_from_float(ratio*len(list)), replace=False, p=[(len(list)+idx)/(len(list)*len(list)+0.5*(len(list)-1)*len(list)) for idx in reversed(range(len(list)))])
    idcs.sort()
    return [list[idx] for idx in idcs]

  """
  For a float f between two integers (n <= f <= n+1), returns either n or n+1 such that f-n is the probability in [0,1] of returning n+1 where
    'nb_float' is the float f.

  Examples:
    nb_int = ImpTree.random_int_from_float(2.75)
    # => 25 % chance of returning 2 and 75 % chance of returning 3
    nb_int = ImpTree.random_int_from_float(2)
    # => 100 % chance of returning 2
  """
  @staticmethod
  def random_int_from_float(nb_float):
    nb_int = floor(nb_float)
    nb_dec = nb_float - nb_int
    return nb_int + random.choices([0,1],weights=[1-nb_dec,nb_dec])[0]

  """
  Returns the best moves found in the best improvement search of the mesh neighborhood as a list of tuples (containing the move, its value and its taboo status) as well as whether the
  full neighborhood subset was visited where
    'nb_best_moves' is the number of desired best moves in the neighborhood;
    'taboo_drop_list' (optional) is a list of taboo drop components;
    'taboo_add_list' (optional) is a list of taboo add components;
    'obj_func' (optional) is the identifier of the desired objective or pseudo-objective (see get_obj_value(...) above for details);
    'best_neighb_obj_value' (optional) is the best value seen so far of the 'obj_func' used in these neighborhood searches;
    'MH' (optional) is a fixed master hub to evaluate the neighbors, by default there are none;
    'use_elite_edge_filter' (optional) is whether to use a given 'elite_edge_filter_matrix' to filter the possible tree edges;
    'elite_edge_filter_matrix' (optional) is a numpy matrix of boolean of shape (nb_nodes, nb_nodes) that describes which tree edges are possible of which only the top triangle is used;
    'mesh_compatibility_matrix' (optional) is a numpy matrix of boolean of shape (nb_nodes, nb_nodes) that describes which mesh edges are possible;
    'verbose' (optional) is whether the intermediary details should be printed in the terminal;
    'log_files' (optional) is a list of open .txt files in which to print the intermediary details.
  """
  def get_best_mesh_moves(self, nb_best_moves=1, taboo_drop_list=[], taboo_add_list=[], obj_func = "chan_avg", best_neighb_obj_value=float('inf'), MH=-1, use_elite_edge_filter=False, elite_edge_filter_matrix=-1, mesh_compatibility_matrix=-1, verbose=False, log_files=[]):
    adj_matrix = self.adj_matrix
    has_mesh_compatibility_matrix = isinstance(mesh_compatibility_matrix, (np.ndarray, np.generic)) and mesh_compatibility_matrix.shape == (self.prob_instance.nb_nodes,self.prob_instance.nb_nodes)
    best_move_value_taboo_triples = [([],-1,False)]
    best_taboo_move_index_value_taboo_quad = ([],float('inf'),-1,False)
    nb_neighbors = 0
    nb_taboo_neighbors = 0

    # Initialization
    max_mesh_idx = int(np.max(adj_matrix))
    mesh_members = {}
    found_next_mesh_idx = False
    for mesh_abs_idx in range(2, max_mesh_idx+1):
      mesh_idx = -1*mesh_abs_idx
      curr_mesh_members = [int(npint) for npint in set(np.where(adj_matrix == mesh_abs_idx)[0])]
      if len(curr_mesh_members) > 0:
        mesh_members[mesh_idx] = curr_mesh_members
      elif not found_next_mesh_idx:
        not_in_taboo_lists = True
        for taboo_drop in taboo_drop_list:
          if isinstance(taboo_drop, tuple) and taboo_drop[0] == mesh_idx:
            not_in_taboo_lists = False
            break
        for taboo_add in taboo_add_list:
          if isinstance(taboo_add, tuple) and taboo_add[0] == mesh_idx:
            not_in_taboo_lists = False
            break
        if not_in_taboo_lists:
          next_mesh_abs_idx = mesh_abs_idx
          found_next_mesh_idx = True
    if not found_next_mesh_idx:
      next_mesh_abs_idx = max_mesh_idx + 1

    # All mesh creations
    for v in range(self.prob_instance.nb_nodes):
      if np.max(adj_matrix[v,:]) < 2:
        if np.sum(adj_matrix[v,:]) > 1:
          where_res = np.where(adj_matrix[v,:]==1)
          new_mesh_members = where_res[0].tolist()
          new_mesh_members.insert(0, v)
          if has_mesh_compatibility_matrix:
            new_mesh_compatibility_matrix = mesh_compatibility_matrix[np.ix_(new_mesh_members,new_mesh_members)]
            if len(ImpTree.get_adj_matrix_conn_components(new_mesh_compatibility_matrix)) > 1:
              continue
          new_mesh_members_set = set(new_mesh_members)
          taboo_add_list_index = -1
          for taboo_add_index in range(len(taboo_add_list)):
            taboo_add = taboo_add_list[taboo_add_index]
            if isinstance(taboo_add, set) and taboo_add.issubset(new_mesh_members_set):
              taboo_add_list_index = taboo_add_index
              break
          for new_mesh_member_idx1 in range(len(new_mesh_members)):
            for new_mesh_member_idx2 in range(new_mesh_member_idx1+1,len(new_mesh_members)):
              adj_matrix[new_mesh_members[new_mesh_member_idx1],new_mesh_members[new_mesh_member_idx2]] = next_mesh_abs_idx
              adj_matrix[new_mesh_members[new_mesh_member_idx2],new_mesh_members[new_mesh_member_idx1]] = next_mesh_abs_idx
          obj_value = self.get_obj_value(obj_func=obj_func, MH=MH)
          if obj_value > -1:
            nb_neighbors += 1
            if taboo_add_list_index >= 0:
              nb_taboo_neighbors += 1
          move_value_taboo_triple = ([(),(next_mesh_abs_idx, new_mesh_members)], obj_value, taboo_add_list_index >= 0)
          if taboo_add_list_index < 0 or obj_value > best_neighb_obj_value:
            if obj_value > best_move_value_taboo_triples[0][1]:
              best_move_value_taboo_triples.append(move_value_taboo_triple)
              best_move_value_taboo_triples.sort(key = lambda best_move_value_taboo_triple: best_move_value_taboo_triple[1])
              best_move_value_taboo_triples = best_move_value_taboo_triples[-nb_best_moves:]
          elif best_move_value_taboo_triples == [([],-1,False)] and (taboo_add_list_index < best_taboo_move_index_value_taboo_quad[1] or (taboo_add_list_index == best_taboo_move_index_value_taboo_quad[1] and obj_value > best_taboo_move_index_value_taboo_quad[2])):
            best_taboo_move_index_value_taboo_quad = (move_value_taboo_triple[0], taboo_add_list_index, move_value_taboo_triple[1], move_value_taboo_triple[2])
          for new_mesh_member_idx1 in range(len(new_mesh_members)):
            if new_mesh_member_idx1 == 0:
              for new_mesh_member_idx2 in range(1,len(new_mesh_members)):
                adj_matrix[new_mesh_members[new_mesh_member_idx1],new_mesh_members[new_mesh_member_idx2]] = 1
                adj_matrix[new_mesh_members[new_mesh_member_idx2],new_mesh_members[new_mesh_member_idx1]] = 1
            else:
              for new_mesh_member_idx2 in range(new_mesh_member_idx1+1,len(new_mesh_members)):
                adj_matrix[new_mesh_members[new_mesh_member_idx1],new_mesh_members[new_mesh_member_idx2]] = 0
                adj_matrix[new_mesh_members[new_mesh_member_idx2],new_mesh_members[new_mesh_member_idx1]] = 0

    # All mesh exclusions (including destruction when |Mesh|<=3)
    for mesh_idx in mesh_members.keys():
      curr_mesh_members = mesh_members[mesh_idx]
      if len(curr_mesh_members) > 3:
        for mesh_member in curr_mesh_members:
          taboo_drop_list_index = -1
          for taboo_drop_index in range(len(taboo_drop_list)):
            taboo_drop = taboo_drop_list[taboo_drop_index]
            if isinstance(taboo_drop, tuple) and taboo_drop[0] == mesh_idx and (mesh_member == taboo_drop[1] or (isinstance(taboo_drop[1], list) and mesh_member in taboo_drop[1])):
              taboo_drop_list_index = taboo_drop_index
              break
          adj_matrix[mesh_member, curr_mesh_members] = 0
          adj_matrix[curr_mesh_members, mesh_member] = 0
          conn_components = self.get_conn_components()
          conn_component_1 = conn_components[0]
          conn_component_2 = conn_components[1]
          for w3p in conn_component_1:
            for w4p in conn_component_2:
              if w4p > w3p:
                w3 = w3p
                w4 = w4p
              else:
                w3 = w4p
                w4 = w3p
              if not use_elite_edge_filter or elite_edge_filter_matrix[w3,w4]:
                adj_matrix[w3,w4] = 1
                adj_matrix[w4,w3] = 1
                obj_value = self.get_obj_value(obj_func=obj_func, MH=MH)
                if obj_value > -1:
                  nb_neighbors += 1
                  if taboo_drop_list_index >= 0:
                    nb_taboo_neighbors += 1
                move_value_taboo_triple = ([(mesh_idx,mesh_member,w3,w4),()], obj_value, taboo_drop_list_index >= 0)
                if taboo_drop_list_index < 0 or obj_value > best_neighb_obj_value:
                  if obj_value > best_move_value_taboo_triples[0][1]:
                    best_move_value_taboo_triples.append(move_value_taboo_triple)
                    best_move_value_taboo_triples.sort(key = lambda best_move_value_taboo_triple: best_move_value_taboo_triple[1])
                    best_move_value_taboo_triples = best_move_value_taboo_triples[-nb_best_moves:]
                elif best_move_value_taboo_triples == [([],-1,False)] and (taboo_drop_list_index < best_taboo_move_index_value_taboo_quad[1] or (taboo_drop_list_index == best_taboo_move_index_value_taboo_quad[1] and obj_value > best_taboo_move_index_value_taboo_quad[2])):
                  best_taboo_move_index_value_taboo_quad = (move_value_taboo_triple[0], taboo_drop_list_index, move_value_taboo_triple[1], move_value_taboo_triple[2])
                adj_matrix[w3,w4] = 0
                adj_matrix[w4,w3] = 0
          adj_matrix[mesh_member, curr_mesh_members] = abs(mesh_idx)
          adj_matrix[curr_mesh_members, mesh_member] = abs(mesh_idx)
          adj_matrix[mesh_member, mesh_member] = 0
      else:# destruction of mesh
        taboo_drop_list_index = -1
        for taboo_drop_index in range(len(taboo_drop_list)):
          taboo_drop = taboo_drop_list[taboo_drop_index]
          if isinstance(taboo_drop, tuple) and taboo_drop[0] == mesh_idx:
            for mesh_member in curr_mesh_members:
              if mesh_member == taboo_drop[1] or (isinstance(taboo_drop[1], list) and mesh_member in taboo_drop[1]):
                taboo_drop_list_index = taboo_drop_index
                break
            if taboo_drop_list_index >= 0:
              break
        for mesh_member_idx1 in range(len(curr_mesh_members)):
          for mesh_member_idx2 in range(mesh_member_idx1+1,len(curr_mesh_members)):
            adj_matrix[curr_mesh_members[mesh_member_idx1],curr_mesh_members[mesh_member_idx2]] = 0
            adj_matrix[curr_mesh_members[mesh_member_idx2],curr_mesh_members[mesh_member_idx1]] = 0
        for mesh_member in curr_mesh_members:
          for other_mesh_member in set(curr_mesh_members) - {mesh_member}:
            adj_matrix[mesh_member,other_mesh_member] = 1
            adj_matrix[other_mesh_member,mesh_member] = 1
          obj_value = self.get_obj_value(obj_func=obj_func, MH=MH)
          if obj_value > -1:
            nb_neighbors += 1
            if taboo_drop_list_index >= 0:
              nb_taboo_neighbors += 1
          move_value_taboo_triple = ([(mesh_idx,mesh_member),()], obj_value, taboo_drop_list_index >= 0)
          if taboo_drop_list_index < 0 or obj_value > best_neighb_obj_value:
            if obj_value > best_move_value_taboo_triples[0][1]:
              best_move_value_taboo_triples.append(move_value_taboo_triple)
              best_move_value_taboo_triples.sort(key = lambda best_move_value_taboo_triple: best_move_value_taboo_triple[1])
              best_move_value_taboo_triples = best_move_value_taboo_triples[-nb_best_moves:]
          elif best_move_value_taboo_triples == [([],-1,False)] and (taboo_drop_list_index < best_taboo_move_index_value_taboo_quad[1] or (taboo_drop_list_index == best_taboo_move_index_value_taboo_quad[1] and obj_value > best_taboo_move_index_value_taboo_quad[2])):
            best_taboo_move_index_value_taboo_quad = (move_value_taboo_triple[0], taboo_drop_list_index, move_value_taboo_triple[1], move_value_taboo_triple[2])
          for other_mesh_member in set(curr_mesh_members) - {mesh_member}:
            adj_matrix[mesh_member,other_mesh_member] = 0
            adj_matrix[other_mesh_member,mesh_member] = 0
        for mesh_member_idx1 in range(len(curr_mesh_members)):
          for mesh_member_idx2 in range(mesh_member_idx1+1,len(curr_mesh_members)):
            adj_matrix[curr_mesh_members[mesh_member_idx1],curr_mesh_members[mesh_member_idx2]] = abs(mesh_idx)
            adj_matrix[curr_mesh_members[mesh_member_idx2],curr_mesh_members[mesh_member_idx1]] = abs(mesh_idx)

    # All mesh inclusions (including exclusion from neighbor meshes)
    for mesh_idx in mesh_members.keys():
      curr_mesh_members = mesh_members[mesh_idx]
      for mesh_member in curr_mesh_members:
        # inclusion of adjacent node
        adj_nodes = np.where(adj_matrix[mesh_member,:]==1)[0].tolist()
        for adj_node in adj_nodes:
          if has_mesh_compatibility_matrix:
            if np.max(mesh_compatibility_matrix[adj_node, curr_mesh_members]) == False:
              continue
          taboo_add_list_index = -1
          for taboo_add_index in range(len(taboo_add_list)):
            taboo_add = taboo_add_list[taboo_add_index]
            if isinstance(taboo_add, tuple) and taboo_add[0] == mesh_idx and taboo_add[1] == adj_node:
              taboo_add_list_index = taboo_add_index
              break
          adj_matrix[adj_node, curr_mesh_members] = abs(mesh_idx)
          adj_matrix[curr_mesh_members, adj_node] = abs(mesh_idx)
          obj_value = self.get_obj_value(obj_func=obj_func, MH=MH)
          if obj_value > -1:
            nb_neighbors += 1
            if taboo_add_list_index >= 0:
              nb_taboo_neighbors += 1
          move_value_taboo_triple = ([(),(mesh_idx,adj_node)], obj_value, taboo_add_list_index >= 0)
          if taboo_add_list_index < 0 or obj_value > best_neighb_obj_value:
            if obj_value > best_move_value_taboo_triples[0][1]:
              best_move_value_taboo_triples.append(move_value_taboo_triple)
              best_move_value_taboo_triples.sort(key = lambda best_move_value_taboo_triple: best_move_value_taboo_triple[1])
              best_move_value_taboo_triples = best_move_value_taboo_triples[-nb_best_moves:]
          elif best_move_value_taboo_triples == [([],-1,False)] and (taboo_add_list_index < best_taboo_move_index_value_taboo_quad[1] or (taboo_add_list_index == best_taboo_move_index_value_taboo_quad[1] and obj_value > best_taboo_move_index_value_taboo_quad[2])):
            best_taboo_move_index_value_taboo_quad = (move_value_taboo_triple[0], taboo_add_list_index, move_value_taboo_triple[1], move_value_taboo_triple[2])
          adj_matrix[adj_node, curr_mesh_members] = 0
          adj_matrix[curr_mesh_members, adj_node] = 0
          adj_matrix[mesh_member, adj_node] = 1
          adj_matrix[adj_node, mesh_member] = 1
        """# inclusion/exclusion from adjacent mesh
        other_meshes = [int(npint) for npint in (set(adj_matrix[mesh_member,:][adj_matrix[mesh_member,:] > 1]) - {abs(mesh_idx)})]
        for other_mesh_abs_idx in other_meshes:
          other_mesh_idx = -other_mesh_abs_idx
          other_mesh_members = list(set(mesh_members[other_mesh_idx]) - {mesh_member})
          adj_matrix[mesh_member, other_mesh_members] = 0
          adj_matrix[other_mesh_members, mesh_member] = 0
          if len(other_mesh_members) > 2:
            for other_mesh_member in other_mesh_members:
              if has_mesh_compatibility_matrix:
                if np.max(mesh_compatibility_matrix[other_mesh_member, curr_mesh_members]) == False:
                  continue
              taboo_list_index = -1
              for taboo_add_index in range(len(taboo_add_list)):
                taboo_add = taboo_add_list[taboo_add_index]
                if isinstance(taboo_add, tuple) and taboo_add[0] == mesh_idx and taboo_add[1] == other_mesh_member:
                  taboo_list_index = taboo_add_index
                  break
              if taboo_list_index < 0:
                for taboo_drop_index in range(len(taboo_drop_list)):
                  taboo_drop = taboo_drop_list[taboo_drop_index]
                  if isinstance(taboo_drop, tuple) and taboo_drop[0] == other_mesh_idx and (mesh_member == taboo_drop[1] or (isinstance(taboo_drop[1], list) and mesh_member in taboo_drop[1])):
                    taboo_list_index = taboo_drop_index
                    break
              adj_matrix[other_mesh_member, curr_mesh_members] = abs(mesh_idx)
              adj_matrix[curr_mesh_members, other_mesh_member] = abs(mesh_idx)
              obj_value = self.get_obj_value(obj_func=obj_func, MH=MH)
              if obj_value > -1:
                nb_neighbors += 1
                if taboo_list_index >= 0:
                  nb_taboo_neighbors += 1
              move_value_taboo_triple = ([(other_mesh_idx,mesh_member),(mesh_idx,other_mesh_member)], obj_value, taboo_list_index >= 0)
              if taboo_list_index < 0 or obj_value > best_neighb_obj_value:
                if obj_value > best_move_value_taboo_triples[0][1]:
                  best_move_value_taboo_triples.append(move_value_taboo_triple)
                  best_move_value_taboo_triples.sort(key = lambda best_move_value_taboo_triple: best_move_value_taboo_triple[1])
                  best_move_value_taboo_triples = best_move_value_taboo_triples[-nb_best_moves:]
              elif best_move_value_taboo_triples == [([],-1,False)] and (taboo_list_index < best_taboo_move_index_value_taboo_quad[1] or (taboo_list_index == best_taboo_move_index_value_taboo_quad[1] and obj_value > best_taboo_move_index_value_taboo_quad[2])):
                best_taboo_move_index_value_taboo_quad = (move_value_taboo_triple[0], taboo_list_index, move_value_taboo_triple[1], move_value_taboo_triple[2])
              adj_matrix[other_mesh_member, curr_mesh_members] = 0
              adj_matrix[curr_mesh_members, other_mesh_member] = 0
          else:# destruction of other mesh
            taboo_drop_list_index = -1
            for taboo_drop_index in range(len(taboo_drop_list)):
              taboo_drop = taboo_drop_list[taboo_drop_index]
              if isinstance(taboo_drop, tuple) and taboo_drop[0] == other_mesh_idx:
                for other_mesh_member in mesh_members[other_mesh_idx]:
                  if other_mesh_member == taboo_drop[1] or (isinstance(taboo_drop[1], list) and other_mesh_member in taboo_drop[1]):
                    taboo_drop_list_index = taboo_drop_index
                    break
                if taboo_drop_list_index >= 0:
                  break
            adj_matrix[other_mesh_members[0],other_mesh_members[1]] = 0
            adj_matrix[other_mesh_members[1],other_mesh_members[0]] = 0
            for other_mesh_member in other_mesh_members:
              if has_mesh_compatibility_matrix:
                if np.max(mesh_compatibility_matrix[other_mesh_member, curr_mesh_members]) == False:
                  continue
              if taboo_drop_list_index < 0:
                taboo_list_index = -1
                for taboo_add_index in range(len(taboo_add_list)):
                  taboo_add = taboo_add_list[taboo_add_index]
                  if isinstance(taboo_add, tuple) and taboo_add[0] == mesh_idx and taboo_add[1] == other_mesh_member:
                    taboo_list_index = taboo_add_index
                    break
              else:
                taboo_list_index = taboo_drop_list_index
              adj_matrix[other_mesh_member, curr_mesh_members] = abs(mesh_idx)
              adj_matrix[curr_mesh_members, other_mesh_member] = abs(mesh_idx)
              other_other_mesh_member = list(set(other_mesh_members) - {other_mesh_member})[0]
              for connecting_mesh_member in [mesh_member, other_mesh_member]:
                adj_matrix[connecting_mesh_member,other_other_mesh_member] = 1
                adj_matrix[other_other_mesh_member,connecting_mesh_member] = 1
                obj_value = self.get_obj_value(obj_func=obj_func, MH=MH)
                if obj_value > -1:
                  nb_neighbors += 1
                  if taboo_list_index >= 0:
                    nb_taboo_neighbors += 1
                move_value_taboo_triple = ([(other_mesh_idx,connecting_mesh_member),(mesh_idx,other_mesh_member)], obj_value, taboo_list_index >= 0)
                if taboo_list_index < 0 or obj_value > best_neighb_obj_value:
                  if obj_value > best_move_value_taboo_triples[0][1]:
                    best_move_value_taboo_triples.append(move_value_taboo_triple)
                    best_move_value_taboo_triples.sort(key = lambda best_move_value_taboo_triple: best_move_value_taboo_triple[1])
                    best_move_value_taboo_triples = best_move_value_taboo_triples[-nb_best_moves:]
                elif best_move_value_taboo_triples == [([],-1,False)] and (taboo_list_index < best_taboo_move_index_value_taboo_quad[1] or (taboo_list_index == best_taboo_move_index_value_taboo_quad[1] and obj_value > best_taboo_move_index_value_taboo_quad[2])):
                  best_taboo_move_index_value_taboo_quad = (move_value_taboo_triple[0], taboo_list_index, move_value_taboo_triple[1], move_value_taboo_triple[2])
                adj_matrix[connecting_mesh_member,other_other_mesh_member] = 0
                adj_matrix[other_other_mesh_member,connecting_mesh_member] = 0
              adj_matrix[other_mesh_member, curr_mesh_members] = 0
              adj_matrix[curr_mesh_members, other_mesh_member] = 0
            adj_matrix[other_mesh_members[0],other_mesh_members[1]] = other_mesh_abs_idx
            adj_matrix[other_mesh_members[1],other_mesh_members[0]] = other_mesh_abs_idx
          adj_matrix[mesh_member, other_mesh_members] = other_mesh_abs_idx
          adj_matrix[other_mesh_members, mesh_member] = other_mesh_abs_idx"""

    # All mesh fusions
    for mesh_idx in mesh_members.keys():
      curr_mesh_members = mesh_members[mesh_idx]
      for mesh_member in curr_mesh_members:
        other_meshes = [int(npint) for npint in (set(adj_matrix[mesh_member,:][adj_matrix[mesh_member,:] > 1]) - {abs(mesh_idx)})]
        for other_mesh_abs_idx in other_meshes:
          other_mesh_idx = -other_mesh_abs_idx
          if mesh_idx < other_mesh_idx:
            other_mesh_members = list(set(mesh_members[other_mesh_idx]) - {mesh_member})
            fused_mesh_members = curr_mesh_members + other_mesh_members
            if has_mesh_compatibility_matrix:
              new_mesh_compatibility_matrix = mesh_compatibility_matrix[np.ix_(fused_mesh_members,fused_mesh_members)]
              if len(ImpTree.get_adj_matrix_conn_components(new_mesh_compatibility_matrix)) > 1:
                continue
            for mesh_member_idx1 in range(len(fused_mesh_members)):
              for mesh_member_idx2 in range(mesh_member_idx1+1,len(fused_mesh_members)):
                adj_matrix[fused_mesh_members[mesh_member_idx1],fused_mesh_members[mesh_member_idx2]] = abs(mesh_idx)
                adj_matrix[fused_mesh_members[mesh_member_idx2],fused_mesh_members[mesh_member_idx1]] = abs(mesh_idx)
            obj_value = self.get_obj_value(obj_func=obj_func, MH=MH)
            if obj_value > -1:
              nb_neighbors += 1
            move_value_taboo_triple = ([(),(mesh_idx,other_mesh_idx)], obj_value, False)
            if obj_value > best_move_value_taboo_triples[0][1]:
              best_move_value_taboo_triples.append(move_value_taboo_triple)
              best_move_value_taboo_triples.sort(key = lambda best_move_value_taboo_triple: best_move_value_taboo_triple[1])
              best_move_value_taboo_triples = best_move_value_taboo_triples[-nb_best_moves:]
            for mesh_member_idx1 in range(len(curr_mesh_members)):
              for mesh_member_idx2 in range(len(other_mesh_members)):
                adj_matrix[curr_mesh_members[mesh_member_idx1],other_mesh_members[mesh_member_idx2]] = 0
                adj_matrix[other_mesh_members[mesh_member_idx2],curr_mesh_members[mesh_member_idx1]] = 0
            other_mesh_members = mesh_members[other_mesh_idx]
            for mesh_member_idx1 in range(len(other_mesh_members)):
              for mesh_member_idx2 in range(mesh_member_idx1+1,len(other_mesh_members)):
                adj_matrix[other_mesh_members[mesh_member_idx1],other_mesh_members[mesh_member_idx2]] = abs(other_mesh_idx)
                adj_matrix[other_mesh_members[mesh_member_idx2],other_mesh_members[mesh_member_idx1]] = abs(other_mesh_idx)
    MWMB_Plot.print_log("  Number of neighbors = "+str(nb_neighbors)+"   (|taboo| = "+str(nb_taboo_neighbors)+"   |non-taboo| = "+str(nb_neighbors-nb_taboo_neighbors)+")", verbose, files=log_files)
    if best_move_value_taboo_triples[0] == ([],-1,False):
      best_move_value_taboo_triples = best_move_value_taboo_triples[1:]
    return best_move_value_taboo_triples,True
  
  """
  Returns the best moves found in the first improvement search of the mesh neighborhood as a list of tuples (containing the move, its value and its taboo status) as well as whether the
  full neighborhood subset was visited where
    'nb_best_moves' is the number of desired best moves in the neighborhood;
    'taboo_drop_list' (optional) is a list of taboo drop components;
    'taboo_add_list' (optional) is a list of taboo add components;
    'obj_func' (optional) is the identifier of the desired objective or pseudo-objective (see get_obj_value(...) above for details);
    'best_neighb_obj_value' (optional) is the best value seen so far of the 'obj_func' used in these neighborhood searches;
    'neighborhood_subset_ratio' (optional) is a float in (0,1] that describes the ratio of the neighborhood that should be searched;
    'MH' (optional) is a fixed master hub to evaluate the neighbors, by default there are none;
    'ordered_edges' (optional) is a list of ordered tuples (representing edges) describing in what order they should be tried;
    'use_elite_edge_filter' (optional) is whether to use a given 'elite_edge_filter_matrix' to filter the possible tree edges;
    'elite_edge_filter_matrix' (optional) is a numpy matrix of boolean of shape (nb_nodes, nb_nodes) that describes which tree edges are possible of which only the top triangle is used;
    'mesh_compatibility_matrix' (optional) is a numpy matrix of boolean of shape (nb_nodes, nb_nodes) that describes which mesh edges are possible;
    'ordered_cluster_lists' is a list of lists of lists of explicit nodes representing the potential mesh clusters for descending levels of minimum signal in the mesh edges (see
      MWMB_Algorithm/search(...)/TOPOLOGICAL DATA ANALYSIS for details);
    'ordered_cluster_adj_matrices' (optional) is a list of numpy matrices of boolean of shape (nb_nodes, nb_nodes) that describe which mesh edges are possible for descending levels of
      minimum signal in the mesh edges (see MWMB_Algorithm/search(...)/TOPOLOGICAL DATA ANALYSIS for details);
    'verbose' (optional) is whether the intermediary details should be printed in the terminal;
    'log_files' (optional) is a list of open .txt files in which to print the intermediary details.
  """
  def get_first_better_mesh_moves(self, nb_best_moves=1, neighborhood_subset_ratio=1, taboo_drop_list=[], taboo_add_list=[], obj_func = "chan_avg", best_neighb_obj_value=float('inf'), MH=-1, ordered_edges=[], use_elite_edge_filter=False, elite_edge_filter_matrix=-1, ordered_cluster_lists=-1, ordered_cluster_adj_matrices=-1, verbose=False, log_files=[]):
    adj_matrix = self.adj_matrix
    init_obj_value = self.get_obj_value(obj_func=obj_func, MH=MH)
    has_ordered_clusters = ordered_cluster_lists!=-1 and ordered_cluster_adj_matrices!=-1
    best_move_value_taboo_triples = [([],-1,False)]
    best_taboo_move_index_value_taboo_quad = ([],float('inf'),-1,False)

    found_better = False
    nb_known_incl_mesh = 0
    nb_known_excl_mesh = 0
    nb_known_crea_fus_neighbors = 0
    nb_known_incl_neighbors = 0
    nb_known_excl_neighbors = 0
    nb_known_excl_node = 0
    nb_seen_excl_node = 0
    nb_visited_neighbors = 0
    nb_taboo_visited_neighbors = 0

    # Initialization
    max_mesh_idx = int(np.max(adj_matrix))
    mesh_members = {}
    found_next_mesh_idx = False
    for mesh_abs_idx in range(2, max_mesh_idx+1):
      mesh_idx = -1*mesh_abs_idx
      curr_mesh_members = [int(npint) for npint in set(np.where(adj_matrix == mesh_abs_idx)[0])]
      if len(curr_mesh_members) > 0:
        mesh_members[mesh_idx] = curr_mesh_members
      elif not found_next_mesh_idx:
        not_in_taboo_lists = True
        for taboo_drop in taboo_drop_list:
          if isinstance(taboo_drop, tuple) and taboo_drop[0] == mesh_idx:
            not_in_taboo_lists = False
            break
        for taboo_add in taboo_add_list:
          if isinstance(taboo_add, tuple) and taboo_add[0] == mesh_idx:
            not_in_taboo_lists = False
            break
        if not_in_taboo_lists:
          next_mesh_abs_idx = mesh_abs_idx
          found_next_mesh_idx = True
    if not found_next_mesh_idx:
      next_mesh_abs_idx = max_mesh_idx + 1

    types_to_visit = []
    potential_crea_nodes_to_visit = [v for v in range(self.prob_instance.nb_nodes) if np.max(adj_matrix[v,:]) < 2 and np.sum(adj_matrix[v,:]) > 1]
    if len(potential_crea_nodes_to_visit) > 0:
      types_to_visit.append("creation")      
      crea_mesh_members = []
    if len(mesh_members) > 0:
      types_to_visit += ["inclusion","exclusion"]
      incl_mesh_nodes_to_visit = {mesh_idx:[] for mesh_idx in mesh_members.keys()}
      excl_mesh_nodes_to_visit = {mesh_idx:[] for mesh_idx in mesh_members.keys()}
      fusion_pairs_to_visit = [(mesh_idx1, mesh_idx2) for mesh_idx1 in mesh_members.keys() for mesh_idx2 in mesh_members.keys() if mesh_idx1 < mesh_idx2 and len(set(mesh_members[mesh_idx1]) & set(mesh_members[mesh_idx2])) == 1]
      if len(fusion_pairs_to_visit) > 0:
        fusion_pairs_to_visit = random.sample(fusion_pairs_to_visit, k=ImpTree.random_int_from_float(neighborhood_subset_ratio*len(fusion_pairs_to_visit)))
        if len(fusion_pairs_to_visit) > 0:
          types_to_visit.append("fusion")
          nb_known_crea_fus_neighbors += len(fusion_pairs_to_visit)
    
    while len(types_to_visit) > 0:
      neighborhood_type = random.choice(types_to_visit)
      if neighborhood_type == "creation":
        #nb_seen_crea+=1
        if len(crea_mesh_members) == 0:
          for potential_crea_node in potential_crea_nodes_to_visit:
            where_res = np.where(adj_matrix[potential_crea_node,:]==1)
            potential_new_mesh_members = where_res[0].tolist()
            potential_new_mesh_members.insert(0, potential_crea_node)
            potential_new_mesh_members_set = set(potential_new_mesh_members)
            if not has_ordered_clusters:
              new_mesh_members = potential_new_mesh_members
              new_mesh_members_set = potential_new_mesh_members_set
              new_min_degree = 0
              new_order_idx = 0
            else:
              new_mesh_members_set = {}
              for cluster_idx in range(len(ordered_cluster_lists[-1])):
                cluster = ordered_cluster_lists[-1][cluster_idx]
                if potential_crea_node in cluster:
                  tmp_set = potential_new_mesh_members_set & set(cluster)
                  if len(tmp_set) >= 3 and potential_crea_node in tmp_set:
                    tmp_list = list(tmp_set)
                    potential_crea_node_idx = tmp_list.index(potential_crea_node)
                    tmp_compatibility_matrix = ordered_cluster_adj_matrices[-1][np.ix_(tmp_list,tmp_list)]
                    tmp_conn_components = ImpTree.get_adj_matrix_conn_components(tmp_compatibility_matrix)
                    for tmp_conn_component in tmp_conn_components:
                      if potential_crea_node_idx in tmp_conn_component and len(tmp_conn_component) >= 3 and len(tmp_conn_component) > len(new_mesh_members_set):
                        new_mesh_members_set = {tmp_list[comp_node_idx] for comp_node_idx in tmp_conn_component}
                        new_min_degree = np.min(np.sum(tmp_compatibility_matrix, axis=1))
              if len(new_mesh_members_set) > 0:
                new_mesh_members = [potential_crea_node]+list(new_mesh_members_set-{potential_crea_node})
                new_order_idx = len(ordered_cluster_lists)-1
                for order_idx in reversed(range(len(ordered_cluster_lists)-1)):
                  for cluster_idx in range(len(ordered_cluster_lists[order_idx])):
                    cluster = ordered_cluster_lists[order_idx][cluster_idx]
                    if new_mesh_members_set.issubset(set(cluster)):
                      tmp_compatibility_matrix = ordered_cluster_adj_matrices[order_idx][np.ix_(new_mesh_members,new_mesh_members)]
                      if len(ImpTree.get_adj_matrix_conn_components(tmp_compatibility_matrix)) == 1:
                        new_order_idx = order_idx
                        new_min_degree = np.min(np.sum(tmp_compatibility_matrix, axis=1))
                      break
                  if new_order_idx != order_idx:
                    break
            if len(new_mesh_members_set) > 0:
              taboo_list_index = -1
              for taboo_add_index in range(len(taboo_add_list)):
                taboo_add = taboo_add_list[taboo_add_index]
                if isinstance(taboo_add, set) and taboo_add.issubset(new_mesh_members_set):
                  taboo_list_index = taboo_add_index
                  break
              crea_mesh_members.append((new_mesh_members,new_min_degree,new_order_idx,taboo_list_index))
          nb_known_crea_fus_neighbors += len(crea_mesh_members)
          if len(crea_mesh_members) > 0:
            crea_mesh_members.sort(key=lambda quad: quad[1], reverse=True)
            crea_mesh_members.sort(key=lambda quad: quad[2])
            crea_mesh_members.sort(key=lambda quad: quad[3])
            if neighborhood_subset_ratio < 1:
              crea_mesh_members = ImpTree.sample_ordered_list_with_more_weight_in_beginning(crea_mesh_members, neighborhood_subset_ratio)
          if len(crea_mesh_members) == 0:
            types_to_visit.remove("creation")
            continue
        new_mesh_members,_,_,taboo_list_index = crea_mesh_members.pop(0)
        if len(crea_mesh_members) == 0:
          types_to_visit.remove("creation")
        for new_mesh_member_idx1 in range(len(new_mesh_members)):
          for new_mesh_member_idx2 in range(new_mesh_member_idx1+1,len(new_mesh_members)):
            adj_matrix[new_mesh_members[new_mesh_member_idx1],new_mesh_members[new_mesh_member_idx2]] = next_mesh_abs_idx
            adj_matrix[new_mesh_members[new_mesh_member_idx2],new_mesh_members[new_mesh_member_idx1]] = next_mesh_abs_idx
        obj_value = self.get_obj_value(obj_func=obj_func, MH=MH)
        nb_visited_neighbors += 1
        if taboo_list_index >= 0:
          nb_taboo_visited_neighbors += 1
        for new_mesh_member_idx1 in range(len(new_mesh_members)):
          if new_mesh_member_idx1 == 0:
            for new_mesh_member_idx2 in range(1,len(new_mesh_members)):
              adj_matrix[new_mesh_members[new_mesh_member_idx1],new_mesh_members[new_mesh_member_idx2]] = 1
              adj_matrix[new_mesh_members[new_mesh_member_idx2],new_mesh_members[new_mesh_member_idx1]] = 1
          else:
            for new_mesh_member_idx2 in range(new_mesh_member_idx1+1,len(new_mesh_members)):
              adj_matrix[new_mesh_members[new_mesh_member_idx1],new_mesh_members[new_mesh_member_idx2]] = 0
              adj_matrix[new_mesh_members[new_mesh_member_idx2],new_mesh_members[new_mesh_member_idx1]] = 0
        move_value_taboo_triple = ([(),(next_mesh_abs_idx, new_mesh_members)], obj_value, taboo_list_index >= 0)
      elif neighborhood_type == "inclusion":
        #nb_seen_incl+=1
        mesh_idx = random.choice(list(incl_mesh_nodes_to_visit))
        curr_mesh_members = mesh_members[mesh_idx]
        if len(incl_mesh_nodes_to_visit[mesh_idx]) == 0:
          incl_pairs = []
          for mesh_member in curr_mesh_members:
            # inclusion of adjacent node
            adj_nodes = np.where(adj_matrix[mesh_member,:]==1)[0].tolist()
            for adj_node in adj_nodes:
              new_order_idx = -1
              if not has_ordered_clusters:
                new_min_degree = new_order_idx = 0
              else:
                new_mesh_members = curr_mesh_members + [adj_node]
                new_mesh_members_set = set(new_mesh_members)
                for order_idx in reversed(range(len(ordered_cluster_lists))):
                  for cluster in ordered_cluster_lists[order_idx]:
                    if new_mesh_members_set.issubset(set(cluster)):
                      tmp_compatibility_matrix = ordered_cluster_adj_matrices[order_idx][np.ix_(new_mesh_members,new_mesh_members)]
                      if len(ImpTree.get_adj_matrix_conn_components(tmp_compatibility_matrix)) == 1:
                        new_order_idx = order_idx
                        new_min_degree = np.min(np.sum(tmp_compatibility_matrix, axis=1))
                      break
                  if new_order_idx != order_idx:
                    break
              if new_order_idx >= 0:
                taboo_list_index = -1
                for taboo_add_index in range(len(taboo_add_list)):
                  taboo_add = taboo_add_list[taboo_add_index]
                  if isinstance(taboo_add, tuple) and taboo_add[0] == mesh_idx and taboo_add[1] == adj_node:
                    taboo_list_index = taboo_add_index
                    break
                incl_pairs.append(((mesh_member, adj_node), new_min_degree,new_order_idx,taboo_list_index))
          nb_known_incl_mesh += 1
          nb_known_incl_neighbors += len(incl_pairs)
          if len(incl_pairs) > 0:
            incl_pairs.sort(key=lambda quad: quad[1], reverse=True)
            incl_pairs.sort(key=lambda quad: quad[2])
            incl_pairs.sort(key=lambda quad: quad[3])
            if neighborhood_subset_ratio < 1:
              incl_pairs = ImpTree.sample_ordered_list_with_more_weight_in_beginning(incl_pairs, neighborhood_subset_ratio)
          if len(incl_pairs) == 0:
            incl_mesh_nodes_to_visit.pop(mesh_idx)
            if len(incl_mesh_nodes_to_visit) == 0:
              types_to_visit.remove("inclusion")
            continue
          else:
            incl_mesh_nodes_to_visit[mesh_idx] = incl_pairs
        incl_mesh_node_tuple,_,_,taboo_list_index = incl_mesh_nodes_to_visit[mesh_idx].pop(0)
        if len(incl_mesh_nodes_to_visit[mesh_idx]) == 0:
          incl_mesh_nodes_to_visit.pop(mesh_idx)
          if len(incl_mesh_nodes_to_visit) == 0:
            types_to_visit.remove("inclusion")
        mesh_member, adj_node = incl_mesh_node_tuple
        adj_matrix[adj_node, curr_mesh_members] = abs(mesh_idx)
        adj_matrix[curr_mesh_members, adj_node] = abs(mesh_idx)
        obj_value = self.get_obj_value(obj_func=obj_func, MH=MH)
        nb_visited_neighbors += 1
        if taboo_list_index >= 0:
          nb_taboo_visited_neighbors += 1
        move_value_taboo_triple = ([(),(mesh_idx,adj_node)], obj_value, taboo_list_index >= 0)
        adj_matrix[adj_node, curr_mesh_members] = 0
        adj_matrix[curr_mesh_members, adj_node] = 0
        adj_matrix[mesh_member, adj_node] = 1
        adj_matrix[adj_node, mesh_member] = 1
      elif neighborhood_type == "exclusion":
        #nb_seen_excl+=1
        mesh_idx = random.choice(list(excl_mesh_nodes_to_visit))
        curr_mesh_members = mesh_members[mesh_idx]
        if len(excl_mesh_nodes_to_visit[mesh_idx]) == 0:
          if len(curr_mesh_members) > 3:# exclusion
            if has_ordered_clusters:
              curr_mesh_members_set = set(curr_mesh_members)
              new_order_idx = -1
              for order_idx in reversed(range(len(ordered_cluster_lists))):
                for cluster_idx in range(len(ordered_cluster_lists[order_idx])):
                  cluster = ordered_cluster_lists[order_idx][cluster_idx]
                  if curr_mesh_members_set.issubset(set(cluster)):
                    tmp_compatibility_matrix = ordered_cluster_adj_matrices[order_idx][np.ix_(curr_mesh_members,curr_mesh_members)]
                    if len(ImpTree.get_adj_matrix_conn_components(tmp_compatibility_matrix)) == 1:
                      new_order_idx = order_idx
                    break
                if new_order_idx != order_idx:
                  break
              if new_order_idx > -1:
                curr_mesh_compatibility_matrix = ordered_cluster_adj_matrices[new_order_idx][np.ix_(curr_mesh_members,curr_mesh_members)]
            excl_nodes = []
            for mesh_member_idx in range(len(curr_mesh_members)):
              mesh_member = curr_mesh_members[mesh_member_idx]
              if not has_ordered_clusters or new_order_idx == -1:
                degree = -1
              else:
                degree = np.sum(curr_mesh_compatibility_matrix[mesh_member_idx,:])
              taboo_list_index = -1
              for taboo_drop_index in range(len(taboo_drop_list)):
                taboo_drop = taboo_drop_list[taboo_drop_index]
                if isinstance(taboo_drop, tuple) and taboo_drop[0] == mesh_idx and (mesh_member == taboo_drop[1] or (isinstance(taboo_drop[1], list) and mesh_member in taboo_drop[1])):
                  taboo_list_index = taboo_drop_index
                  break
              excl_nodes.append((mesh_member, degree, taboo_list_index))
            nb_known_excl_node += len(excl_nodes)
            excl_nodes.sort(key=lambda triple: triple[1])
            excl_nodes.sort(key=lambda triple: triple[2])
            excl_mesh_nodes_to_visit[mesh_idx] = [excl_nodes, dict()]
          else:# destruction
            taboo_list_index = -1
            for taboo_drop_index in range(len(taboo_drop_list)):
              taboo_drop = taboo_drop_list[taboo_drop_index]
              if isinstance(taboo_drop, tuple) and taboo_drop[0] == mesh_idx:
                for mesh_member in curr_mesh_members:
                  if mesh_member == taboo_drop[1] or (isinstance(taboo_drop[1], list) and mesh_member in taboo_drop[1]):
                    taboo_list_index = taboo_drop_index
                    break
                if taboo_list_index >= 0:
                  break
            excl_mesh_nodes_to_visit[mesh_idx] = [(mesh_member,taboo_list_index) for mesh_member in mesh_members[mesh_idx]]
            nb_known_excl_neighbors += 3
            nb_known_excl_node += 3
            nb_seen_excl_node += 3
            if neighborhood_subset_ratio < 1:
              excl_mesh_nodes_to_visit[mesh_idx] = random.sample(excl_mesh_nodes_to_visit[mesh_idx], k=ImpTree.random_int_from_float(neighborhood_subset_ratio*len(excl_mesh_nodes_to_visit[mesh_idx])))
              if len(excl_mesh_nodes_to_visit[mesh_idx]) == 0:
                excl_mesh_nodes_to_visit.pop(mesh_idx)
                if len(excl_mesh_nodes_to_visit) == 0:
                  types_to_visit.remove("exclusion")
                continue
          nb_known_excl_mesh += 1
        if len(curr_mesh_members) > 3:# exclusion
          mesh_member,_,taboo_list_index = excl_mesh_nodes_to_visit[mesh_idx][0][0]
          if not mesh_member in excl_mesh_nodes_to_visit[mesh_idx][1].keys():
            #create dict list
            adj_matrix[mesh_member, curr_mesh_members] = 0
            adj_matrix[curr_mesh_members, mesh_member] = 0
            conn_components = self.get_conn_components()
            conn_component_1 = conn_components[0]
            conn_component_2 = conn_components[1]
            adj_matrix[mesh_member, curr_mesh_members] = abs(mesh_idx)
            adj_matrix[curr_mesh_members, mesh_member] = abs(mesh_idx)
            adj_matrix[mesh_member, mesh_member] = 0
            if len(ordered_edges) > 0:
              is_valid_edge_matrix = np.zeros(adj_matrix.shape)
              is_valid_edge_matrix[np.ix_(conn_component_1,conn_component_2)] = 1
              is_valid_edge_matrix += is_valid_edge_matrix.T
              excl_mesh_nodes_to_visit[mesh_idx][1][mesh_member] = [(w3,w4) for w3,w4 in ordered_edges if is_valid_edge_matrix[w3,w4]]
            elif use_elite_edge_filter:
              excl_mesh_nodes_to_visit[mesh_idx][1][mesh_member] = [(w3,w4) if w4 > w3 else (w4,w3) for w3 in conn_component_1 for w4 in conn_component_2 and (elite_edge_filter_matrix[w3,w4] or elite_edge_filter_matrix[w4,w3])]
            else:
              excl_mesh_nodes_to_visit[mesh_idx][1][mesh_member] = [(w3,w4) if w4 > w3 else (w4,w3) for w3 in conn_component_1 for w4 in conn_component_2]
            nb_seen_excl_node += 1
            nb_known_excl_neighbors += len(excl_mesh_nodes_to_visit[mesh_idx][1][mesh_member])
            if len(excl_mesh_nodes_to_visit[mesh_idx][1][mesh_member]) > 0 and neighborhood_subset_ratio < 1:
              excl_mesh_nodes_to_visit[mesh_idx][1][mesh_member] = ImpTree.sample_ordered_list_with_more_weight_in_beginning(excl_mesh_nodes_to_visit[mesh_idx][1][mesh_member], neighborhood_subset_ratio)
            if len(excl_mesh_nodes_to_visit[mesh_idx][1][mesh_member]) == 0:
              excl_mesh_nodes_to_visit[mesh_idx][1].pop(mesh_member)
              excl_mesh_nodes_to_visit[mesh_idx][0].pop(0)
              if len(excl_mesh_nodes_to_visit[mesh_idx][0]) == 0:
                excl_mesh_nodes_to_visit.pop(mesh_idx)
                if len(excl_mesh_nodes_to_visit) == 0:
                  types_to_visit.remove("exclusion")
              continue
          w3,w4 = excl_mesh_nodes_to_visit[mesh_idx][1][mesh_member].pop(0)
          if len(excl_mesh_nodes_to_visit[mesh_idx][1][mesh_member]) == 0:
            excl_mesh_nodes_to_visit[mesh_idx][1].pop(mesh_member)
            excl_mesh_nodes_to_visit[mesh_idx][0].pop(0)
            if len(excl_mesh_nodes_to_visit[mesh_idx][0]) == 0:
              excl_mesh_nodes_to_visit.pop(mesh_idx)
              if len(excl_mesh_nodes_to_visit) == 0:
                types_to_visit.remove("exclusion")
          else:
            excl_mesh_nodes_to_visit[mesh_idx][0].append(excl_mesh_nodes_to_visit[mesh_idx][0].pop(0))
          adj_matrix[mesh_member, curr_mesh_members] = 0
          adj_matrix[curr_mesh_members, mesh_member] = 0
          adj_matrix[w3,w4] = 1
          adj_matrix[w4,w3] = 1
          obj_value = self.get_obj_value(obj_func=obj_func, MH=MH)
          nb_visited_neighbors += 1
          if taboo_list_index >= 0:
            nb_taboo_visited_neighbors += 1
          move_value_taboo_triple = ([(mesh_idx,mesh_member,w3,w4),()], obj_value, taboo_list_index >= 0)
          adj_matrix[w3,w4] = 0
          adj_matrix[w4,w3] = 0
          adj_matrix[mesh_member, curr_mesh_members] = abs(mesh_idx)
          adj_matrix[curr_mesh_members, mesh_member] = abs(mesh_idx)
          adj_matrix[mesh_member, mesh_member] = 0
        else:# destruction
          mesh_member,taboo_list_index = excl_mesh_nodes_to_visit[mesh_idx].pop(0)
          if len(excl_mesh_nodes_to_visit[mesh_idx]) == 0:
            excl_mesh_nodes_to_visit.pop(mesh_idx)
            if len(excl_mesh_nodes_to_visit) == 0:
              types_to_visit.remove("exclusion")
          for mesh_member_idx1 in range(len(curr_mesh_members)):
            for mesh_member_idx2 in range(mesh_member_idx1+1,len(curr_mesh_members)):
              adj_matrix[curr_mesh_members[mesh_member_idx1],curr_mesh_members[mesh_member_idx2]] = 0
              adj_matrix[curr_mesh_members[mesh_member_idx2],curr_mesh_members[mesh_member_idx1]] = 0
          for other_mesh_member in set(curr_mesh_members) - {mesh_member}:
            adj_matrix[mesh_member,other_mesh_member] = 1
            adj_matrix[other_mesh_member,mesh_member] = 1
          obj_value = self.get_obj_value(obj_func=obj_func, MH=MH)
          nb_visited_neighbors += 1
          if taboo_list_index >= 0:
            nb_taboo_visited_neighbors += 1
          move_value_taboo_triple = ([(mesh_idx,mesh_member),()], obj_value, taboo_list_index >= 0)
          for other_mesh_member in set(curr_mesh_members) - {mesh_member}:
            adj_matrix[mesh_member,other_mesh_member] = 0
            adj_matrix[other_mesh_member,mesh_member] = 0
          for mesh_member_idx1 in range(len(curr_mesh_members)):
            for mesh_member_idx2 in range(mesh_member_idx1+1,len(curr_mesh_members)):
              adj_matrix[curr_mesh_members[mesh_member_idx1],curr_mesh_members[mesh_member_idx2]] = abs(mesh_idx)
              adj_matrix[curr_mesh_members[mesh_member_idx2],curr_mesh_members[mesh_member_idx1]] = abs(mesh_idx)
      else:# neighborhood_type == "fusion"
        mesh_idx, other_mesh_idx = fusion_pairs_to_visit.pop(random.randint(0,len(fusion_pairs_to_visit)-1))
        if len(fusion_pairs_to_visit) == 0:
          types_to_visit.remove("fusion")
        taboo_list_index = -1
        curr_mesh_members = mesh_members[mesh_idx]
        mesh_member = list(set(curr_mesh_members) & set(mesh_members[other_mesh_idx]))[0]
        other_mesh_members = list(set(mesh_members[other_mesh_idx]) - {mesh_member})
        fused_mesh_members = curr_mesh_members + other_mesh_members
        for mesh_member_idx1 in range(len(fused_mesh_members)):
          for mesh_member_idx2 in range(mesh_member_idx1+1,len(fused_mesh_members)):
            adj_matrix[fused_mesh_members[mesh_member_idx1],fused_mesh_members[mesh_member_idx2]] = abs(mesh_idx)
            adj_matrix[fused_mesh_members[mesh_member_idx2],fused_mesh_members[mesh_member_idx1]] = abs(mesh_idx)
        obj_value = self.get_obj_value(obj_func=obj_func, MH=MH)
        nb_visited_neighbors += 1
        move_value_taboo_triple = ([(),(mesh_idx,other_mesh_idx)], obj_value, False)
        for mesh_member_idx1 in range(len(curr_mesh_members)):
          for mesh_member_idx2 in range(len(other_mesh_members)):
            adj_matrix[curr_mesh_members[mesh_member_idx1],other_mesh_members[mesh_member_idx2]] = 0
            adj_matrix[other_mesh_members[mesh_member_idx2],curr_mesh_members[mesh_member_idx1]] = 0
        other_mesh_members = mesh_members[other_mesh_idx]
        for mesh_member_idx1 in range(len(other_mesh_members)):
          for mesh_member_idx2 in range(mesh_member_idx1+1,len(other_mesh_members)):
            adj_matrix[other_mesh_members[mesh_member_idx1],other_mesh_members[mesh_member_idx2]] = abs(other_mesh_idx)
            adj_matrix[other_mesh_members[mesh_member_idx2],other_mesh_members[mesh_member_idx1]] = abs(other_mesh_idx)
      
      if taboo_list_index < 0 or obj_value > best_neighb_obj_value:
        if obj_value > best_move_value_taboo_triples[0][1]:
          best_move_value_taboo_triples.append(move_value_taboo_triple)
          best_move_value_taboo_triples.sort(key = lambda best_move_value_taboo_triple: best_move_value_taboo_triple[1])
          best_move_value_taboo_triples = best_move_value_taboo_triples[-nb_best_moves:]
          if obj_value > init_obj_value:
            found_better = True
            found_better_counter = nb_best_moves
      elif best_move_value_taboo_triples == [([],-1,False)] and (taboo_list_index < best_taboo_move_index_value_taboo_quad[1] or (taboo_list_index == best_taboo_move_index_value_taboo_quad[1] and obj_value > best_taboo_move_index_value_taboo_quad[2])):
        best_taboo_move_index_value_taboo_quad = (move_value_taboo_triple[0], taboo_list_index, move_value_taboo_triple[1], move_value_taboo_triple[2])
      if found_better:
        found_better_counter -= 1
        if found_better_counter == 0:
          break

    if best_move_value_taboo_triples[0] == ([],-1,False):
      best_move_value_taboo_triples = best_move_value_taboo_triples[1:]
    nb_approx_neighbors = nb_known_crea_fus_neighbors
    if nb_known_incl_mesh > 0:
      nb_approx_neighbors += round(nb_known_incl_neighbors*(len(mesh_members)/nb_known_incl_mesh))
    if nb_known_excl_mesh > 0 and nb_seen_excl_node > 0:
      nb_approx_neighbors += round(nb_known_excl_neighbors*(nb_known_excl_node/nb_seen_excl_node)*(len(mesh_members)/nb_known_excl_mesh))
    if nb_approx_neighbors > 0:
      MWMB_Plot.print_log("  Approx number of neighbors = "+str(nb_approx_neighbors)+"   (|known| = "+str(nb_known_crea_fus_neighbors+nb_known_incl_neighbors+nb_known_excl_neighbors)+");   |visited| = "+str(nb_visited_neighbors)+"   (|taboo| = "+str(nb_taboo_visited_neighbors)+"   |non-taboo| = "+str(nb_visited_neighbors-nb_taboo_visited_neighbors)+")      \tVisited "+str(round(100*nb_visited_neighbors/nb_approx_neighbors))+"%", verbose, files=log_files)
    else:
      MWMB_Plot.print_log("  Approx number of neighbors = "+str(nb_approx_neighbors)+"   (|known| = "+str(nb_known_crea_fus_neighbors+nb_known_incl_neighbors+nb_known_excl_neighbors)+");   |visited| = "+str(nb_visited_neighbors)+"   (|taboo| = "+str(nb_taboo_visited_neighbors)+"   |non-taboo| = "+str(nb_visited_neighbors-nb_taboo_visited_neighbors)+")      \tVisited 0%", verbose, files=log_files)
    return best_move_value_taboo_triples, len(types_to_visit) == 0

  """
  Makes a given move to the current ImpTree where
    'move' is the move;
    'neighborhood' is the identifier of the desired neighborhood
      "tree" is the tree neighborhood (tree edge swap);
      "mesh" is the mesh neighborhood (creation, destruction, inclusion, exclusion, fusion).
  """
  def make_move(self, move, neighborhood="tree"):
    self.has_obj_value = False
    return ImpTree.make_adj_matrix_move(self.adj_matrix, move, neighborhood)
  
  """
  Makes a given move to an adjacency matrix where
    'adj_matrix' is the adjacency matrix;
    'move' is the move;
    'neighborhood' is the identifier of the desired neighborhood
      "tree" is the tree neighborhood (tree edge swap);
      "mesh" is the mesh neighborhood (creation, destruction, inclusion, exclusion, fusion).
  """
  @staticmethod
  def make_adj_matrix_move(adj_matrix, move, neighborhood="tree"):
    if neighborhood == "tree":
      return ImpTree.make_edge_swap_move(adj_matrix, move)
    else:# neighborhood == "mesh":
      return ImpTree.make_mesh_move(adj_matrix, move)

  """
  Makes a given tree move to an adjacency matrix where
    'adj_matrix' is the adjacency matrix;
    'move' is the move.
  """
  @staticmethod
  def make_edge_swap_move(adj_matrix, move):
    dropped_edge = move[0]
    added_edge = move[1]
    adj_matrix[dropped_edge[0],dropped_edge[1]] = 0
    adj_matrix[dropped_edge[1],dropped_edge[0]] = 0
    adj_matrix[added_edge[0],added_edge[1]] = 1
    adj_matrix[added_edge[1],added_edge[0]] = 1
    return [dropped_edge], [added_edge]
  
  """
  Makes a given mesh move to an adjacency matrix where
    'adj_matrix' is the adjacency matrix;
    'move' is the move.
  """
  @staticmethod
  def make_mesh_move(adj_matrix, move):
    if len(move[0]) == 0:
      if isinstance(move[1][1], list): # Mesh creation ( move = [(),(mesh_abs_idx,[nodes])] )
        mesh_abs_idx, nodes = move[1]
        for node_idx1 in range(len(nodes)):
          for node_idx2 in range(node_idx1+1,len(nodes)):
            adj_matrix[nodes[node_idx1],nodes[node_idx2]] = mesh_abs_idx
            adj_matrix[nodes[node_idx2],nodes[node_idx1]] = mesh_abs_idx
        return [[]], [(-mesh_abs_idx, nodes)]
      elif move[1][1] < 0: #             Mesh fusion ( move = [(),(mesh_idx,other_mesh_idx)] )
        mesh_idx, other_mesh_idx = move[1]
        curr_mesh_members = [int(npint) for npint in set(np.where(adj_matrix == abs(mesh_idx))[0])]
        other_mesh_members = [int(npint) for npint in set(np.where(adj_matrix == abs(other_mesh_idx))[0])]
        fused_mesh_members = curr_mesh_members + other_mesh_members
        for mesh_member_idx1 in range(len(fused_mesh_members)):
          for mesh_member_idx2 in range(mesh_member_idx1+1,len(fused_mesh_members)):
            adj_matrix[fused_mesh_members[mesh_member_idx1],fused_mesh_members[mesh_member_idx2]] = abs(mesh_idx)
            adj_matrix[fused_mesh_members[mesh_member_idx2],fused_mesh_members[mesh_member_idx1]] = abs(mesh_idx)
        return [[]], [[]]
      else: #                            Mesh inclusion ( move = [(),(mesh_idx,node)] )
        mesh_idx, node = move[1]
        curr_mesh_members = [int(npint) for npint in set(np.where(adj_matrix == abs(mesh_idx))[0])]
        adj_matrix[node, curr_mesh_members] = abs(mesh_idx)
        adj_matrix[curr_mesh_members, node] = abs(mesh_idx)
        return [[]], [(mesh_idx, node)]
    else:
      if len(move[1]) > 0: #             Mesh inclusion/exclusion
        mesh_idx, new_shared_node = move[1]
        other_mesh_idx, _ = move[0]
        curr_mesh_members = [int(npint) for npint in set(np.where(adj_matrix == abs(mesh_idx))[0])]
        other_mesh_members = [int(npint) for npint in set(np.where(adj_matrix == abs(other_mesh_idx))[0])]
        if len(other_mesh_members) > 3: #                         ( move = [(other_mesh_idx,prev_shared_node),(mesh_idx,new_shared_node)] )
          _, prev_shared_node = move[0]
          adj_matrix[prev_shared_node, other_mesh_members] = 0
          adj_matrix[other_mesh_members, prev_shared_node] = 0
          adj_matrix[new_shared_node, curr_mesh_members] = abs(mesh_idx)
          adj_matrix[curr_mesh_members, new_shared_node] = abs(mesh_idx)
          return [(other_mesh_idx, prev_shared_node)], [(mesh_idx, new_shared_node)]
        else: # destruction of other mesh                         ( move = [(other_mesh_idx,connecting_node),(mesh_idx,new_shared_node)] )
          _, connecting_node = move[0]
          for other_mesh_member_idx1 in range(len(other_mesh_members)):
            for other_mesh_member_idx2 in range(other_mesh_member_idx1+1,len(other_mesh_members)):
              adj_matrix[other_mesh_members[other_mesh_member_idx1],other_mesh_members[other_mesh_member_idx2]] = 0
              adj_matrix[other_mesh_members[other_mesh_member_idx2],other_mesh_members[other_mesh_member_idx1]] = 0
          adj_matrix[new_shared_node, curr_mesh_members] = abs(mesh_idx)
          adj_matrix[curr_mesh_members, new_shared_node] = abs(mesh_idx)
          other_other_mesh_node = list(set(other_mesh_members) - (set(curr_mesh_members) & set(other_mesh_members)) - {new_shared_node})[0]
          adj_matrix[connecting_node, other_other_mesh_node] = 1
          adj_matrix[other_other_mesh_node, connecting_node] = 1
          return [set(other_mesh_members)], [(mesh_idx, new_shared_node)]
      else:
        if len(move[0]) == 4: #          Mesh exclusion ( move = [(mesh_idx,node,w3,w4),()] )
          mesh_idx, node, w3, w4 = move[0]
          curr_mesh_members = [int(npint) for npint in set(np.where(adj_matrix == abs(mesh_idx))[0])]
          adj_matrix[node, curr_mesh_members] = 0
          adj_matrix[curr_mesh_members, node] = 0
          adj_matrix[w3,w4] = 1
          adj_matrix[w4,w3] = 1
          return [(mesh_idx, node)], [[]]
          #                              Mesh split !?!?! (  )
        else: #                          Mesh destruction ( move = [(mesh_idx,star_node),()] )
          mesh_idx, star_node = move[0]
          curr_mesh_members = [int(npint) for npint in set(np.where(adj_matrix == abs(mesh_idx))[0])]
          for mesh_member_idx1 in range(len(curr_mesh_members)):
            for mesh_member_idx2 in range(mesh_member_idx1+1,len(curr_mesh_members)):
              adj_matrix[curr_mesh_members[mesh_member_idx1],curr_mesh_members[mesh_member_idx2]] = 0
              adj_matrix[curr_mesh_members[mesh_member_idx2],curr_mesh_members[mesh_member_idx1]] = 0
          for other_mesh_member in set(curr_mesh_members) - {star_node}:
            adj_matrix[star_node,other_mesh_member] = 1
            adj_matrix[other_mesh_member,star_node] = 1
          return [set(curr_mesh_members)], [[]]

  """
  Returns the appropriate length of the taboo add list where
    'neighborhood' is the identifier of the current neighborhood;
    'nb_nodes' is the number of nodes of the instance;
    'mesh_sizes' (optional) is a list of the sizes of the current ImpTree's mesh clusters as returned by .get_mesh_sizes();
    'len_ordered_edges' (optional) is the number of possible tree edges to consider in the neighborhood searches.
  """
  @staticmethod
  def get_taboo_add_list_length(neighborhood, nb_nodes, mesh_sizes=[], len_ordered_edges=-1):
    if neighborhood == "tree":
      #return round(sqrt(nb_nodes*(nb_nodes-1)/2)) 
      nb_addable_edges = nb_nodes*(nb_nodes-1)/2
      if len_ordered_edges >= 0:
        elite_edge_ratio = len_ordered_edges/nb_addable_edges
      else:
        elite_edge_ratio = 1
      for mesh_size in mesh_sizes:
        nb_addable_edges -= mesh_size*(mesh_size-1)/2
      return round(sqrt(elite_edge_ratio*nb_addable_edges)) 
    else:# neighborhood == "mesh"
      #return round(sqrt(nb_nodes-2)/2)
      max_nb_non_leaves = nb_nodes-2
      avg_nb_includable_nodes = 1 * sum([mesh_size for mesh_size in mesh_sizes])
      return round(sqrt( max_nb_non_leaves + avg_nb_includable_nodes )/2)
  
  """
  Returns the appropriate length of the taboo drop list where
    'neighborhood' is the identifier of the current neighborhood;
    'nb_nodes' is the number of nodes of the instance;
    'mesh_sizes' (optional) is a list of the sizes of the current ImpTree's mesh clusters as returned by .get_mesh_sizes().
  """
  @staticmethod
  def get_taboo_drop_list_length(neighborhood, nb_nodes, mesh_sizes=[]):
    if neighborhood == "tree":
      #return round(sqrt(nb_nodes-1)/2)
      nb_droppable_edges = nb_nodes-1
      for mesh_size in mesh_sizes:
        nb_droppable_edges -= mesh_size - 1
      return round(sqrt(nb_droppable_edges)/2)
    else:# neighborhood == "mesh"
      #return round(sqrt( sqrt(nb_nodes) + nb_nodes  ))
      nb_destroyable_cluster = sum([1 for mesh_size in mesh_sizes if mesh_size == 3])
      nb_droppable_nodes = sum([mesh_size for mesh_size in mesh_sizes])
      return round(sqrt( nb_destroyable_cluster + nb_droppable_nodes  ))


"""
ImpArborescence
-------
Object representing an implicit arborescence on a specific ProbInstance with specific ProbParameters.

  .prob_instance {ProbInstance} --> associated problem instance
  .prob_params {ProbParameters} --> associated problem parameters
  .MH {int} --> explicit node which is the master hub
  .is_mixed {boolean} --> whether the implicit arborescence contains mesh clusters (True) or not (False)
  .mesh_members {dictionary from mesh index to associated list of explicit nodes} --> describes the members of each mesh cluster
  .meshes {dictionary from explicit node to associated list of mesh indices} --> describes, for each explicit node, the mesh clusters of which it is a member
  .adj_matrix {numpy array of floats of shape ('nb_nodes', 'nb_nodes')} --> adjacency matrix of the implicit tree such that
    a_uv = 1 --> there is a tree edge between nodes u and v and u is the predecessor of v (v is a successor of u)
    a_uv = -1 --> there is a tree edge between nodes u and v and v is the predecessor of u (u is a successor of v)
    a_uv = n, with n >= 2 --> nodes u and v are in the same mesh cluster with mesh_index = -n
  .preds {dictionary from implicit node (except master hub) to associated implicit node} --> describes the predecessor of each implicit node that has a predecessor (i.e. every node
    except the master hub)
  .succ_lists {dictionary from implicit node to associated list of implicit nodes} --> describes the successors of each implicit node
  .nb_required_channels {int} --> the minimum number of channels required to satisfy this implicit arborescence, if it is > 2, this arborescence is not valid
  .has_obj_value = False {boolean} --> whether its true objective value or greedy pseudo-objective value has been evaluated and stored (True) or not (False)

  (defined if .has_obj_value)
    .obj_value {float}  --> its true objective value ("full") or greedy pseudo-objective value ("greedy") depending on which has been last evaluated
"""
class ImpArborescence(ImpTree):
  """
  Initializes an ImpArborescence object where
    imp_tree' is the associated implicit tree object;
    'MH' is the master hub;
    'mesh_members' (optional) is the dictionary that describes the members of each mesh cluster;
    'meshes' (optional) is the dictionary that describes, for each explicit node, the mesh clusters of which it is a member.
  """
  def __init__(self, imp_tree, MH, mesh_members = -1, meshes=-1):
    self.prob_instance = imp_tree.prob_instance # ProbInstance
    self.prob_params = imp_tree.prob_params # ProbParameters
    self.MH = MH # int
    if mesh_members == -1 and meshes == -1: # mesh clusters and memberships
      is_mixed = self.prob_params.mixed
      mesh_members = {}
      meshes = {v : [] for v in range(self.prob_instance.nb_nodes)}
      if is_mixed:
        max_mesh_idx = np.max(imp_tree.adj_matrix)
        is_mixed = max_mesh_idx >= 2
        if is_mixed:
          for mesh_abs_idx in range(2, int(max_mesh_idx)+1):
            mesh_idx = -1*mesh_abs_idx
            curr_mesh_members = [int(npint) for npint in set(np.where(imp_tree.adj_matrix == mesh_abs_idx)[0])]
            if len(curr_mesh_members) > 0:
              mesh_members[mesh_idx] = curr_mesh_members
              for mesh_member in curr_mesh_members:
                meshes[mesh_member].append(mesh_idx)
    else:
      is_mixed = len(mesh_members) > 0
    self.is_mixed = is_mixed # boolean
    self.mesh_members = mesh_members # dict : mesh -> list of mesh members
    self.meshes = meshes # dict : exp_node -> list of containing meshes
    if True: # adj_matrix, preds and succs
      adj_matrix = 1 * imp_tree.adj_matrix
      preds = {}
      succ_lists = {}
      preds_to_visit = [MH]
      while len(preds_to_visit)>0:
        pred = preds_to_visit.pop(0)
        if not is_mixed:
          succs = np.where(adj_matrix[pred,:] == 1)[0].tolist()
        else:
          if pred >= 0:
            succs = np.where(adj_matrix[pred,:] == 1)[0].tolist()
            if len(meshes[pred]) > 0:
              mesh_memberships = copy.copy(meshes[pred])
              if pred in preds and preds[pred] < 0:
                mesh_memberships.remove(preds[pred])
              succs = mesh_memberships + succs
          else:
            mesh_nodes = copy.copy(mesh_members[pred])
            if pred in preds:
              mesh_nodes.remove(preds[pred])
            succs = mesh_nodes
        succ_lists[pred] = succs
        for succ in succs:
          preds[succ] = pred
          if pred >= 0 and succ >= 0:
            adj_matrix[succ,pred] = -1
        preds_to_visit = preds_to_visit + succs
    self.adj_matrix = adj_matrix # (nb_exp_nodes, nb_exp_nodes)
    self.preds = preds # dict : imp_node -> predecessor
    self.succ_lists = succ_lists # dict : imp_node -> list of successors
    if True: # max number of nodes per PTM connection:
      if not is_mixed or np.max(adj_matrix[MH, :]) <= 1:
        max_nb_nodes_per_PTM_connection = ceil(np.sum(adj_matrix[MH, :] == 1)/2)
      else:
        max_nb_nodes_per_PTM_connection = np.sum(adj_matrix[MH, :] == 1)
      no_MH_adj_matrix = np.vstack((adj_matrix[:MH, :], adj_matrix[MH+1:,]))
      no_mesh_nodes_adj_matrix = no_MH_adj_matrix[np.max(no_MH_adj_matrix, axis=1)<=1,:]
      if no_mesh_nodes_adj_matrix.shape[0] > 0:
        max_nb_nodes_per_PTM_connection = max(max_nb_nodes_per_PTM_connection, np.max(np.sum(no_mesh_nodes_adj_matrix == 1, axis=1)))
      if is_mixed:
        mesh_nodes_adj_matrix = no_MH_adj_matrix[np.max(no_MH_adj_matrix, axis=1)>=2,:]
        if mesh_nodes_adj_matrix.shape[0] > 0:
          max_nb_nodes_per_PTM_connection = max(max_nb_nodes_per_PTM_connection, np.max(np.sum(mesh_nodes_adj_matrix == 1, axis=1)))
    self.max_nb_nodes_per_PTM_connection = max_nb_nodes_per_PTM_connection # int
    if not is_mixed: # number of required channels
      nb_required_channels = 2
    else:
      nb_required_channels = 0
      for v in range(self.prob_instance.nb_nodes):
        v_nb_required_channels = len(set(list(adj_matrix[v,:][adj_matrix[v,:]!=0])))
        if v_nb_required_channels > nb_required_channels:
          nb_required_channels = v_nb_required_channels
    self.nb_required_channels = nb_required_channels # int
    self.has_obj_value = False # boolean

  """
  Defines equality of ImpArborescence objects.
  """
  def __eq__(self, other):
    if isinstance(other, ImpArborescence):
      return self.MH == other.MH and (self.adj_matrix == other.adj_matrix).all() and self.prob_params == other.prob_params and self.prob_instance == other.prob_instance
    else:
      return False

  """
  Prints an ImpArborescence object.
  """
  def __repr__(self):
    string = "Implicit Arborescence :\n\t"
    #implicit nodes
    string += "Master Hub  : "+str(self.MH)+"\n\t"
    string += "Mesh Clusters : "+str(self.mesh_members)+"\n\t"
    #string += "Mesh Memberships : "+str(self.meshes)+"\n\t"
    #string += "Predecessors : "+str(self.preds)+"\n\t"
    string += "Successors : "+str(self.succ_lists)
    return string

  """
  Plots an ImpArborescence as a set of nodes (master hub in green) connected by arrows (solid lines: tree edges, dotted lines: cluster membership) where
    'ax' (optional) is the matplotlib axes object in which to plot;
    'pretitle' (optional) is the title string of the plot (ending in " F=obj_value" if the ImpArborescence has an obj_value).
  """
  def show(self, ax=-1, pretitle=""):
    extra_border = 1
    line_width = 0.8
    has_axis = ax != -1
    if not has_axis:
      fig = plt.figure(1)
      ax = fig.add_axes([0.075,0.075,0.85,0.85])
    ax.set_aspect('equal')
    nb_nodes = self.prob_instance.nb_nodes
    coordinates_km = self.prob_instance.coordinates_km
    adj_matrix = self.adj_matrix
    MH = self.MH
    xlim0 = -extra_border
    xlim1 = np.max(coordinates_km[:,0])+extra_border
    ylim0 = -extra_border
    ylim1 = np.max(coordinates_km[:,1])+extra_border
    ax.set_xlim(xlim0,xlim1)
    ax.set_ylim(ylim0,ylim1)
    max_size = max(xlim1-xlim0,ylim1-ylim0)
    if self.has_obj_value:
      ax.set_title(pretitle+"F_obj = "+str(round(100*self.obj_value)/100))
    is_mixed = self.is_mixed
    if is_mixed:
      mesh_coordinates_km = {mesh_idx:(np.mean(coordinates_km[self.mesh_members[mesh_idx],0]),np.mean(coordinates_km[self.mesh_members[mesh_idx],1])) for mesh_idx in self.mesh_members.keys()}
      meshes = self.meshes
    # ARROWS
    for w1 in range(nb_nodes):
      if is_mixed:
        for mesh_idx in meshes[w1]:
          if w1 in self.preds and self.preds[w1] == mesh_idx:
            dx = coordinates_km[w1,0] - mesh_coordinates_km[mesh_idx][0]
            dy = coordinates_km[w1,1] - mesh_coordinates_km[mesh_idx][1]
            radius = sqrt(pow(dx,2) + pow(dy,2))
            ratio_beg = (7/400*max_size)/radius
            ratio_end = (radius - 7/200*max_size)/radius
            ax.arrow(mesh_coordinates_km[mesh_idx][0]+ratio_beg*dx, mesh_coordinates_km[mesh_idx][1]+ratio_beg*dy, ratio_end*dx, ratio_end*dy,length_includes_head=True,head_width=0.25,ls=':',lw=1.4*line_width,color='k')
          else:
            dx = mesh_coordinates_km[mesh_idx][0] - coordinates_km[w1,0]
            dy = mesh_coordinates_km[mesh_idx][1] - coordinates_km[w1,1]
            radius = sqrt(pow(dx,2) + pow(dy,2))
            ratio_beg = (7/400*max_size)/radius
            ratio_end = (radius - 7/200*max_size)/radius
            ax.arrow(coordinates_km[w1,0]+ratio_beg*dx, coordinates_km[w1,1]+ratio_beg*dy, ratio_end*dx, ratio_end*dy,length_includes_head=True,head_width=0.25,ls=':',lw=1.4*line_width,color='k')
      for w2 in range(nb_nodes):
        if adj_matrix[w1,w2] == 1:
          dx = coordinates_km[w2,0] - coordinates_km[w1,0]
          dy = coordinates_km[w2,1] - coordinates_km[w1,1]
          radius = sqrt(pow(dx,2) + pow(dy,2))
          ratio_beg = (7/400*max_size)/radius
          ratio_end = (radius - 7/200*max_size)/radius
          ax.arrow(coordinates_km[w1,0]+ratio_beg*dx, coordinates_km[w1,1]+ratio_beg*dy, ratio_end*dx, ratio_end*dy,length_includes_head=True,head_width=0.25,lw=line_width,color='k')
    # NODES
    ax.scatter([coordinates_km[MH,0]],[coordinates_km[MH,1]], s=30, c='green')
    non_MH_coordinates = np.concatenate((coordinates_km[:MH,:],coordinates_km[MH+1:,:]))
    ax.scatter(non_MH_coordinates[:,0].tolist(),non_MH_coordinates[:,1].tolist(), s=30, c='k')
    if is_mixed:
      mesh_coordinates_x_km = []
      mesh_coordinates_y_km = []
      for mesh_idx in mesh_coordinates_km.keys():
        mesh_coordinates_x_km.append(mesh_coordinates_km[mesh_idx][0])
        mesh_coordinates_y_km.append(mesh_coordinates_km[mesh_idx][1])
      ax.scatter(mesh_coordinates_x_km,mesh_coordinates_y_km, s=50, c='w', edgecolors='k')
    # NODE LABELS
    for exp_node in range(nb_nodes):
      ax.text(coordinates_km[exp_node,0],coordinates_km[exp_node,1] - 5/100*max_size,str(exp_node),fontsize=8,color='gray',ha='center')
    if not has_axis:
      fig.canvas.draw()
      plt.show()

  """
  Returns the sizes of the ImpArborescence's mesh clusters as a list of integers. If there are no mesh clusters, the returned list is empty.
  """
  def get_mesh_sizes(self):
    mesh_sizes = [len(self.mesh_members[mesh_idx]) for mesh_idx in self.mesh_members.keys()]
    mesh_sizes.sort(reverse=True)
    return mesh_sizes

  """
  Returns the objective/pseudo-objective value of the ImpArborescence where
    'obj_func' is the identifier of the desired objective or pseudo-objective
      "full" --> real objective (very slow), used only once at the end of the search for small instances (nb_nodes <= 10);
      "greedy" --> accurate but slow pseudo-objective, used for every new current solution in the topology search main loop;
      "chan_avg" --> inaccurate but fast pseudo-objective, used in the neighborhood searches in the topology search main loop;
      "chan_max" --> fast upper bound pseudo-objective (actual upper bound if no mesh cluster), used to decide if "greedy" is worth evaluating;
      "avg" --> very inaccurate but very fast pseudo-objective, never used;
    'get_network' (optional) is whether the best Network found (associated with the returned value) should be returned as well (True) or not (False);
    'get_best_partitions' (optional) is whether to only return the partitions of the successors of the master hub in 2 sets in descending order of pseudo-objective value for
      'obj_func' == "chan_avg" (True) or to proceed as usual and return the value (False);
    'has_old_avg' (optional) is whether it is given the intermediary data for the "avg" pseudo-objective evaluation of a previous ImpTree neighbor (True) or not (False);
    'has_old_chan_avg' (optional) is whether it is given the intermediary data for the "chan_avg" pseudo-objective evaluation of a previous ImpTree neighbor (True) or not (False);
    'has_old_chan_max' (optional) is whether it is given the intermediary data for the "chan_max" pseudo-objective evaluation of a previous ImpTree neighbor (True) or not (False);
    'old_imp_arb_adj_matrix' (optional) is the adjacency matrix of the best ImpArborescence of a previous ImpTree neighbor;
    'old_MH_succs_2_partition' (optional) is the partition of the master hub's successors in 2 sets of the best pseudo-network of a previous ImpTree neighbor (see Network for details);
    'old_chan_assignment' (optional) is a dictionary that describes the channels of the downlink connections of the predecessors of the best pseudo-network of a previous neighbor (see
      Network for details);
    'old_direct_simp_TPs' (optional) is the direct throughputs of the best pseudo-network of a previous ImpTree neighbor (see Network for details);
    'old_mesh_members' (optional) is the 'mesh_members' of the best ImpArborescence of a previous ImpTree neighbor;
    'old_preds' (optional) is the 'preds' of the best ImpArborescence of a previous ImpTree neighbor;
    'old_mesh_routes' (optional) is a dictionary that describes the routes used between every pair of members for each cluster of the best pseudo-network of a previous neighbor (see
      Network for details).

  Examples:
    best_obj_value = imp_arb.get_obj_value(obj_func = "greedy")
    best_obj_value, best_network = imp_arb.get_obj_value(obj_func = "greedy", get_network = True)
  """
  def get_obj_value(self, obj_func = "full", get_network = False, get_best_partitions=False, has_old_avg=False, has_old_chan_avg=False, has_old_chan_max=False, old_imp_arb_adj_matrix=-1, old_MH_succs_2_partition=-1, old_chan_assignment=-1, old_direct_simp_TPs=-1, old_mesh_members=-1, old_preds=-1, old_mesh_routes=-1):
    best_obj_value = -1
    MH_succs_2_partitions = self.get_all_2_partitions()
    if obj_func == "avg":
      alignments_and_beams = []
      has_tmp_avg = has_old_avg
      tmp_imp_arb_adj_matrix = old_imp_arb_adj_matrix
      tmp_MH_succs_2_partition = old_MH_succs_2_partition
      tmp_direct_simp_TPs = old_direct_simp_TPs
      tmp_mesh_members = old_mesh_members
      tmp_preds = old_preds
      tmp_mesh_routes = old_mesh_routes
      for MH_succs_2_partition_idx in range(len(MH_succs_2_partitions)):
        MH_succs_2_partition = MH_succs_2_partitions[MH_succs_2_partition_idx]
        network = Network(self, MH_succs_2_partition, alignments_and_beams=alignments_and_beams)
        obj_value = network.get_obj_value(obj_func="avg", has_old_avg=has_tmp_avg, old_imp_arb_adj_matrix=tmp_imp_arb_adj_matrix, old_MH_succs_2_partition=tmp_MH_succs_2_partition, old_direct_simp_TPs=tmp_direct_simp_TPs, old_mesh_members=tmp_mesh_members, old_preds=tmp_preds, old_mesh_routes=tmp_mesh_routes)
        if obj_value > best_obj_value:
          best_obj_value = obj_value
          if get_network:
            best_network = network
        if MH_succs_2_partition_idx == 0:
          alignments_and_beams = [network.uplink_ant_alignments_rad, network.uplink_ant_nb_beams, copy.copy(network.downlink_ant_alignments_rad), copy.copy(network.downlink_ant_beams)]
          alignments_and_beams[2].pop(self.MH)
          alignments_and_beams[3].pop(self.MH)
          has_tmp_avg = True
          tmp_imp_arb_adj_matrix = network.imp_arb.adj_matrix
          tmp_MH_succs_2_partition = network.MH_succs_2_partition
          tmp_direct_simp_TPs = network.direct_TPs
          tmp_mesh_members = network.imp_arb.mesh_members
          if network.is_mixed:
            tmp_preds = network.imp_arb.preds
            tmp_mesh_routes = network.A_mesh_routes
          else:
            tmp_preds = -1
            tmp_mesh_routes = -1
    elif obj_func == "chan_avg" or obj_func == "chan_max":
      if len(MH_succs_2_partitions) <= 2:
        for MH_succs_2_partition_idx in range(len(MH_succs_2_partitions)):
          MH_succs_2_partition = MH_succs_2_partitions[MH_succs_2_partition_idx]
          chan_assignments = self.get_all_chan_assignments(MH_succs_2_partition)
          for chan_assignment in chan_assignments:
            network = Network(self, MH_succs_2_partition, chan_assignment)
            obj_value = network.get_obj_value(obj_func=obj_func, has_old_chan_avg=has_old_chan_avg, has_old_chan_max=has_old_chan_max, old_imp_arb_adj_matrix=old_imp_arb_adj_matrix, old_MH_succs_2_partition=old_MH_succs_2_partition, old_chan_assignment=old_chan_assignment, old_direct_simp_TPs=old_direct_simp_TPs, old_mesh_members=old_mesh_members, old_preds=old_preds, old_mesh_routes=old_mesh_routes)
            if obj_value > best_obj_value:
              best_obj_value = obj_value
              if get_network:
                best_network = network
      else:
        MH_succs = self.succ_lists[self.MH]
        channel_pred_direct_TPs = [dict(), dict(), ""]
        for channel in [-2,-3]:
          chan_assignment = {self.MH:[channel,-1]}
          nodes_to_visit = [MH_succ for MH_succ in MH_succs]
          while len(nodes_to_visit) > 0:
            new_node = nodes_to_visit.pop(0)
            nodes_to_visit += self.succ_lists[new_node]
            chan_assignment[new_node] = channel
          network = Network(self, [MH_succs, []], chan_assignment)
          arcs_and_effective_TPs = network.get_obj_value(obj_func=obj_func, return_arcs_and_effective_TPs=True, has_old_chan_avg=has_old_chan_avg, has_old_chan_max=has_old_chan_max, old_imp_arb_adj_matrix=old_imp_arb_adj_matrix, old_MH_succs_2_partition=old_MH_succs_2_partition, old_chan_assignment=old_chan_assignment, old_direct_simp_TPs=old_direct_simp_TPs, old_mesh_members=old_mesh_members, old_preds=old_preds, old_mesh_routes=old_mesh_routes)
          if not self.is_mixed:
            arcs, A_effective_TPs, B_effective_TPs, C_effective_TPs = arcs_and_effective_TPs
            for arc_idx in range(len(arcs)):
              pred,_ = arcs[arc_idx]
              if not pred in channel_pred_direct_TPs[channel]:
                channel_pred_direct_TPs[channel][pred] = [[],[],[]]
              channel_pred_direct_TPs[channel][pred][0].append(A_effective_TPs[arc_idx])
              channel_pred_direct_TPs[channel][pred][1].append(B_effective_TPs[arc_idx])
              channel_pred_direct_TPs[channel][pred][2].append(C_effective_TPs[arc_idx])
          else:
            AC_arcs, B_arcs, A_effective_TPs, B_effective_TPs, C_effective_TPs = arcs_and_effective_TPs
            non_mesh_arcs = True
            last_AC_mesh_idx = -1
            last_B_mesh_idx = -1
            for arc_idx in range(max(len(AC_arcs),len(B_arcs))):
              if non_mesh_arcs:
                AC_arc = AC_arcs[arc_idx]
                if AC_arc != B_arcs[arc_idx] or (AC_arc[0] in self.meshes and AC_arc[1] in self.meshes and len(set(self.meshes[AC_arc[0]]) & set(self.meshes[AC_arc[1]])) > 0):
                  non_mesh_arcs = False
              if non_mesh_arcs:
                pred,_ = AC_arcs[arc_idx]
                if not pred in channel_pred_direct_TPs[channel]:
                  channel_pred_direct_TPs[channel][pred] = [[],[],[]]
                channel_pred_direct_TPs[channel][pred][0].append(A_effective_TPs[arc_idx])
                channel_pred_direct_TPs[channel][pred][1].append(B_effective_TPs[arc_idx])
                channel_pred_direct_TPs[channel][pred][2].append(C_effective_TPs[arc_idx])
              else:
                if arc_idx < len(AC_arcs):
                  member1,member2 = AC_arcs[arc_idx]
                  if last_AC_mesh_idx in self.meshes[member1] and last_AC_mesh_idx in self.meshes[member2]:
                    mesh_idx = last_AC_mesh_idx
                  else:
                    mesh_idx = list(set(self.meshes[member1]) & set(self.meshes[member2]))[0]
                    last_AC_mesh_idx = mesh_idx
                  if not mesh_idx in channel_pred_direct_TPs[channel]:
                    channel_pred_direct_TPs[channel][mesh_idx] = [[],[],[]]
                  channel_pred_direct_TPs[channel][mesh_idx][0].append(A_effective_TPs[arc_idx])
                  channel_pred_direct_TPs[channel][mesh_idx][2].append(C_effective_TPs[arc_idx])
                if arc_idx < len(B_arcs):
                  member1,member2 = B_arcs[arc_idx]
                  if last_B_mesh_idx in self.meshes[member1] and last_B_mesh_idx in self.meshes[member2]:
                    mesh_idx = last_B_mesh_idx
                  else:
                    mesh_idx = list(set(self.meshes[member1]) & set(self.meshes[member2]))[0]
                    last_B_mesh_idx = mesh_idx
                  if not mesh_idx in channel_pred_direct_TPs[channel]:
                    channel_pred_direct_TPs[channel][mesh_idx] = [[],[],[]]
                  channel_pred_direct_TPs[channel][mesh_idx][1].append(B_effective_TPs[arc_idx])
        nb_nodes = self.prob_instance.nb_nodes
        avg_factor = 1/39
        A_factor = self.prob_params.obj_scenarios[0] # 1
        B_factor = self.prob_params.obj_scenarios[1] # 8/5
        C_factor = self.prob_params.obj_scenarios[2] # 1
        partial_tree_nodes = [self.MH] + MH_succs
        if not self.is_mixed:
          nbs_descs_inc_self = [0 for exp_node in range(nb_nodes)]
        else:
          nbs_descs_inc_self = [0 for imp_node in range(nb_nodes+int(np.max(self.adj_matrix)))]
        nodes_to_visit = []+MH_succs
        idx = 0
        while idx < len(nodes_to_visit):
          new_node = nodes_to_visit[idx]
          nodes_to_visit += self.succ_lists[new_node]
          idx += 1
        while len(nodes_to_visit) > 0:
          new_node = nodes_to_visit.pop(-1)
          if new_node >= 0:
            nb_descs_inc_self = 1
          else:
            nb_descs_inc_self = 0
          succs = self.succ_lists[new_node]
          for succ in succs:
            nb_descs_inc_self += nbs_descs_inc_self[succ]
          nbs_descs_inc_self[new_node] = nb_descs_inc_self
        if get_best_partitions:
          partition_best_obj_values = []
        for MH_succs_2_partition_idx in range(len(MH_succs_2_partitions)):
          MH_succs_2_partition = MH_succs_2_partitions[MH_succs_2_partition_idx]
          chan_assignments = self.get_all_chan_assignments(MH_succs_2_partition)
          if get_best_partitions:
            partition_best_obj_value = -1
          for chan_assignment in chan_assignments:
            tmp_MH_succs_2_partition = [part if len(part) > 0 and part[0] >= 0 else [] for part in MH_succs_2_partition]
            network = self.create_partial_network(tmp_MH_succs_2_partition, partial_tree_nodes, chan_assignment, [])
            arcs, A_effective_TPs, B_effective_TPs, C_effective_TPs = network.get_obj_value(obj_func=obj_func, return_arcs_and_effective_TPs=True)
            for arc_idx in range(len(arcs)):
              _,succ = arcs[arc_idx]
              B_effective_TPs[arc_idx] /= nbs_descs_inc_self[succ]
              C_effective_TPs[arc_idx] *= (nb_nodes - 1)/(nbs_descs_inc_self[succ] * (nb_nodes - nbs_descs_inc_self[succ]))
            for imp_node in chan_assignment.keys():
              if imp_node != self.MH and imp_node in channel_pred_direct_TPs[chan_assignment[imp_node]]:
                A_effective_TPs += channel_pred_direct_TPs[chan_assignment[imp_node]][imp_node][0]
                B_effective_TPs += channel_pred_direct_TPs[chan_assignment[imp_node]][imp_node][1]
                C_effective_TPs += channel_pred_direct_TPs[chan_assignment[imp_node]][imp_node][2]
            A_obj_value = min(A_effective_TPs) + avg_factor*mean(A_effective_TPs)
            B_obj_value = min(B_effective_TPs) + avg_factor*mean(B_effective_TPs)
            C_obj_value = (nb_nodes-1)*(min(C_effective_TPs) + avg_factor*mean(C_effective_TPs))
            obj_value = A_factor*A_obj_value + B_factor*B_obj_value + C_factor*C_obj_value
            if obj_value > best_obj_value:
              best_obj_value = obj_value
              if get_network:
                best_MH_succs_2_partition = MH_succs_2_partition
                best_chan_assignment = chan_assignment
            if get_best_partitions and obj_value > partition_best_obj_value:
              partition_best_obj_value = obj_value
          if get_best_partitions:
            partition_best_obj_values.append((MH_succs_2_partition,partition_best_obj_value))
        if get_best_partitions:
          partition_best_obj_values.sort(key=lambda pair: pair[1])
          return [partition for partition,_ in partition_best_obj_values]
        if get_network:
          best_network = Network(self, best_MH_succs_2_partition, best_chan_assignment)
          best_network.get_obj_value(obj_func=obj_func, has_old_chan_avg=has_old_chan_avg, has_old_chan_max=has_old_chan_max, old_imp_arb_adj_matrix=old_imp_arb_adj_matrix, old_MH_succs_2_partition=old_MH_succs_2_partition, old_chan_assignment=old_chan_assignment, old_direct_simp_TPs=old_direct_simp_TPs, old_mesh_members=old_mesh_members, old_preds=old_preds, old_mesh_routes=old_mesh_routes)
    # elif obj_func == "chan_avg_OLD" or obj_func == "chan_max_OLD":
    #   MH_succs = self.succ_lists[self.MH]
    #   MH_succs_partial_sols = dict()
    #   alignments_and_beams = []
    #   has_tmp_chan_avg = has_old_chan_avg
    #   has_tmp_chan_max = has_old_chan_max
    #   tmp_imp_arb_adj_matrix = old_imp_arb_adj_matrix
    #   tmp_MH_succs_2_partition = old_MH_succs_2_partition
    #   tmp_chan_assignment = old_chan_assignment
    #   tmp_direct_simp_TPs = old_direct_simp_TPs
    #   tmp_mesh_members = old_mesh_members
    #   tmp_preds = old_preds
    #   tmp_mesh_routes = old_mesh_routes
    #   for MH_succs_2_partition_idx in range(len(MH_succs_2_partitions)):
    #     MH_succs_2_partition = MH_succs_2_partitions[MH_succs_2_partition_idx]
    #     chan_assignments = self.get_all_chan_assignments(MH_succs_2_partition)
    #     for chan_assignment in chan_assignments:
    #       if len(MH_succs_partial_sols) > 0:
    #         alignments_and_beams = [dict(), dict(), dict(), dict()]
    #         tmp_chan_assignment = dict()
    #         tmp_direct_simp_TPs = np.zeros((self.prob_instance.nb_nodes,self.prob_instance.nb_nodes))
    #         if len(tmp_mesh_members) > 0:
    #           tmp_mesh_routes = dict()
    #         for MH_succ in MH_succs:
    #           if MH_succ in chan_assignment and MH_succ in MH_succs_partial_sols and chan_assignment[MH_succ] in MH_succs_partial_sols[MH_succ]:
    #             partial_sol = MH_succs_partial_sols[MH_succ][chan_assignment[MH_succ]]
    #             for align_idx in range(4):
    #               alignments_and_beams[align_idx] |= partial_sol[0][align_idx]
    #             tmp_chan_assignment |= partial_sol[1]
    #             tmp_direct_simp_TPs += partial_sol[2]
    #             if len(tmp_mesh_members) > 0:
    #               tmp_mesh_routes |= partial_sol[3]
    #       network = Network(self, MH_succs_2_partition, chan_assignment, alignments_and_beams=alignments_and_beams)
    #       obj_value = network.get_obj_value(obj_func=obj_func, has_old_chan_avg=has_tmp_chan_avg, has_old_chan_max=has_tmp_chan_max, old_imp_arb_adj_matrix=tmp_imp_arb_adj_matrix, old_MH_succs_2_partition=tmp_MH_succs_2_partition, old_chan_assignment=tmp_chan_assignment, old_direct_simp_TPs=tmp_direct_simp_TPs, old_mesh_members=tmp_mesh_members, old_preds=tmp_preds, old_mesh_routes=tmp_mesh_routes)
    #       if obj_value > best_obj_value:
    #         best_obj_value = obj_value
    #         if get_network:
    #           best_network = network
    #       if MH_succs_2_partition_idx < len(MH_succs_2_partitions) - 1:
    #         for MH_succ in MH_succs:
    #           if not MH_succ in MH_succs_partial_sols:
    #             if len(MH_succs_partial_sols) == 0:
    #               has_tmp_chan_avg = obj_func == "chan_avg"
    #               has_tmp_chan_max = obj_func == "chan_max"
    #               tmp_imp_arb_adj_matrix = self.adj_matrix
    #               tmp_mesh_members = self.mesh_members
    #               tmp_MH_succs_2_partition = [[],[]]
    #               if len(tmp_mesh_members) == 0:
    #                 tmp_preds = -1
    #                 tmp_mesh_routes = -1
    #               else:
    #                 tmp_preds = self.preds
    #             MH_succs_partial_sols[MH_succ] = dict()
    #           if MH_succ in chan_assignment and not chan_assignment[MH_succ] in MH_succs_partial_sols[MH_succ]:
    #             alignments_and_beams = [dict(), dict(), dict(), dict()]
    #             tmp_chan_assignment = dict()
    #             tmp_direct_simp_TPs = np.zeros((self.prob_instance.nb_nodes,self.prob_instance.nb_nodes))
    #             if len(tmp_mesh_members) > 0:
    #               tmp_mesh_routes = dict()
    #             nodes_to_visit = [MH_succ]
    #             while len(nodes_to_visit) > 0:
    #               imp_node = nodes_to_visit.pop(0)
    #               nodes_to_visit += self.succ_lists[imp_node]
    #               if imp_node != MH_succ and imp_node in network.uplink_ant_alignments_rad:
    #                 alignments_and_beams[0][imp_node] = network.uplink_ant_alignments_rad[imp_node]
    #                 alignments_and_beams[1][imp_node] = network.uplink_ant_nb_beams[imp_node]
    #               if imp_node in network.downlink_ant_alignments_rad:
    #                 alignments_and_beams[2][imp_node] = network.downlink_ant_alignments_rad[imp_node]
    #                 alignments_and_beams[3][imp_node] = network.downlink_ant_beams[imp_node]
    #               if imp_node in network.freq_assignment:
    #                 tmp_chan_assignment[imp_node] = network.freq_assignment[imp_node]
    #               tmp_direct_simp_TPs[imp_node,:] = network.direct_TPs[imp_node,:]
    #               if len(tmp_mesh_members) > 0:
    #                 if imp_node in network.A_mesh_routes:
    #                   tmp_mesh_routes[imp_node] = network.A_mesh_routes[imp_node]
    #             if len(tmp_mesh_members) == 0:
    #               MH_succs_partial_sols[MH_succ][chan_assignment[MH_succ]] = [alignments_and_beams, tmp_chan_assignment, tmp_direct_simp_TPs]
    #             else:
    #               MH_succs_partial_sols[MH_succ][chan_assignment[MH_succ]] = [alignments_and_beams, tmp_chan_assignment, tmp_direct_simp_TPs, tmp_mesh_routes]
    # elif obj_func == "1_freq_per_channel":
    #   for MH_succs_2_partition_idx in range(len(MH_succs_2_partitions)):
    #     MH_succs_2_partition = MH_succs_2_partitions[MH_succs_2_partition_idx]
    #     chan_assignments = self.get_all_chan_assignments(MH_succs_2_partition)
    #     for chan_assignment in chan_assignments:
    #       network = Network(self, MH_succs_2_partition, chan_assignment)
    #       obj_value = network.get_obj_value(obj_func=obj_func)
    #       if obj_value > best_obj_value:
    #         best_obj_value = obj_value
    #         if get_network:
    #           best_network = network
    elif obj_func == "greedy":
      if len(MH_succs_2_partitions) <= 64:
        for MH_succs_2_partition in MH_succs_2_partitions:
          chan_assignments = self.get_all_chan_assignments(MH_succs_2_partition)
          for chan_assignment in chan_assignments:
            with warnings.catch_warnings():
              warnings.simplefilter("ignore")
              network, obj_value = self.get_greedy_network(MH_succs_2_partition, chan_assignment)
            if obj_value > best_obj_value:
              best_obj_value = obj_value
              if get_network:
                best_network = network
      else:
        nb_GRASP_iterations = 1
        nb_local_search_iterations = len(self.succ_lists[self.MH]) - 4
        chan_avg_best_partitions = self.get_obj_value(obj_func="chan_avg", get_best_partitions=True)[-nb_GRASP_iterations:]
        for MH_succs_2_partition in reversed(chan_avg_best_partitions):
          chan_assignments = self.get_all_chan_assignments(MH_succs_2_partition)
          for chan_assignment in chan_assignments:
            with warnings.catch_warnings():
              warnings.simplefilter("ignore")
              network, obj_value = self.get_greedy_network(MH_succs_2_partition, chan_assignment)
            if obj_value > best_obj_value:
              best_obj_value = obj_value
              if get_network:
                best_network = network
          last_MH_succs_2_partition = -1
          for local_search_it in range(nb_local_search_iterations):
            neighb_MH_succs_2_partitions = ImpArborescence.get_neighbor_2_partitions(MH_succs_2_partition)
            best_neighb_obj_value = -1
            for neighb_MH_succs_2_partition in neighb_MH_succs_2_partitions:
              if neighb_MH_succs_2_partition != last_MH_succs_2_partition:
                chan_assignments = self.get_all_chan_assignments(MH_succs_2_partition)
                for chan_assignment in chan_assignments:
                  with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    network, obj_value = self.get_greedy_network(MH_succs_2_partition, chan_assignment)
                  if obj_value > best_obj_value:
                    best_obj_value = obj_value
                    if get_network:
                      best_network = network
                  if obj_value > best_neighb_obj_value:
                    best_neighb_obj_value = obj_value
                    best_neighb_MH_succs_2_partition = neighb_MH_succs_2_partition
            last_MH_succs_2_partition = MH_succs_2_partition
            MH_succs_2_partition = best_neighb_MH_succs_2_partition
    else: # obj_func == "full":
      for MH_succs_2_partition in MH_succs_2_partitions:
        freq_assignments = self.get_all_freq_assignments(MH_succs_2_partition=MH_succs_2_partition)
        for freq_assignment in freq_assignments:
          network = Network(self, MH_succs_2_partition, freq_assignment)
          obj_value = network.get_obj_value(obj_func="full")
          if obj_value > best_obj_value:
            best_obj_value = obj_value
            if get_network:
              best_network = network
    if best_obj_value > -1 and (obj_func == "full" or obj_func == "greedy"):
      self.obj_value = best_obj_value # float
      self.has_obj_value = True # boolean
    if get_network:
      return best_obj_value, best_network
    else:
      return best_obj_value

  """
  Returns a list of all valid partitions of the master hub's successors in 2 sets for the ImpArborescence (see Network for details).
  """
  def get_all_2_partitions(self):
    if not self.is_mixed:
      return ImpArborescence.get_all_2_partitions_recursive_tree([],[],self.succ_lists[self.MH], self.prob_params.max_nb_nodes_per_PTM_connection)
    else:
      if self.MH < 0:
        return [[self.mesh_members[self.MH],[]]]
      else:
        return ImpArborescence.get_all_2_partitions_recursive_mixed([],False,[],False,self.succ_lists[self.MH], self.prob_params.max_nb_nodes_per_PTM_connection)

  """
  Returns a list of all valid partitions of the master hub's successors in 2 sets for a strictly tree ImpArborescence (i.e. with no mesh clusters) where
    'part1' is a list of the successors of the first set that have been already assigned there;
    'part2' is a list of the successors of the second set that have been already assigned there;
    'undecided' is a list of the successors that have not been assigned already;
    'max_nb_nodes_per_PTM_connection' is the maximum number of connected nodes in a single PMP connection, given by ProbParameters.
  """
  @staticmethod
  def get_all_2_partitions_recursive_tree(part1, part2, undecided, max_nb_nodes_per_PTM_connection):
    if len(undecided)>0:
      new_node = undecided[0]
      partitions = ImpArborescence.get_all_2_partitions_recursive_tree(part1+[new_node],part2,undecided[1:],max_nb_nodes_per_PTM_connection)
      partitions += ImpArborescence.get_all_2_partitions_recursive_tree(part1,part2+[new_node],undecided[1:],max_nb_nodes_per_PTM_connection)
      return partitions
    else:
      not_duplicate = len(part1) > len(part2)
      if not not_duplicate and len(part1) == len(part2):
        min_node = min(part1+part2)
        not_duplicate = min_node in part2
      if not_duplicate and len(part1) <= max_nb_nodes_per_PTM_connection and len(part2) <= max_nb_nodes_per_PTM_connection:
        return [[part1, part2]]
      else:
        return []

  """
  Returns a list of all valid partitions of the master hub's successors in 2 sets for an ImpArborescence with mesh clusters where
    'part1' is a list of the successors of the first set that have been already assigned there;
    'part1_is_mesh' is whether 'part1' has been assigned a single successor that is a mesh index;
    'part2' is a list of the successors of the second set that have been already assigned there;
    'part2_is_mesh' is whether 'part2' has been assigned a single successor that is a mesh index;
    'undecided' is a list of the successors that have not been assigned already;
    'max_nb_nodes_per_PTM_connection' is the maximum number of connected nodes in a single PMP connection, given by ProbParameters.
  """
  @staticmethod
  def get_all_2_partitions_recursive_mixed(part1, part1_is_mesh, part2, part2_is_mesh, undecided, max_nb_nodes_per_PTM_connection):
    if len(undecided)>0:
      new_node = undecided[0]
      if new_node >= 0:
        if not part1_is_mesh:
          partitions = ImpArborescence.get_all_2_partitions_recursive_mixed(part1+[new_node],False,part2,part2_is_mesh,undecided[1:], max_nb_nodes_per_PTM_connection)
        else:
          partitions = []
        if not part2_is_mesh:
          partitions += ImpArborescence.get_all_2_partitions_recursive_mixed(part1,part1_is_mesh,part2+[new_node],False,undecided[1:], max_nb_nodes_per_PTM_connection)
        return partitions
      else:
        if not part1_is_mesh and len(part1) == 0:
          partitions = ImpArborescence.get_all_2_partitions_recursive_mixed([new_node],True,part2,part2_is_mesh,undecided[1:], max_nb_nodes_per_PTM_connection)
        else:
          partitions = []
        if not part2_is_mesh and len(part2) == 0:
          partitions += ImpArborescence.get_all_2_partitions_recursive_mixed(part1,part1_is_mesh,[new_node],True,undecided[1:], max_nb_nodes_per_PTM_connection)
        return partitions
    else:
      not_duplicate = len(part1) > len(part2)
      if not not_duplicate and len(part1) == len(part2):
        min_node = min(part1+part2)
        not_duplicate = min_node in part2
      if not_duplicate and len(part1) <= max_nb_nodes_per_PTM_connection and len(part2) <= max_nb_nodes_per_PTM_connection:
        return [[part1, part2]]
      else:
        return []
  
  """
  Returns a list of valid partitions of the master hub's successors in 2 sets that are neighbors of a current partition where
    'MH_succs_2_partition' is the current partition.
  """
  @staticmethod
  def get_neighbor_2_partitions(MH_succs_2_partition):
    neighb_MH_succs_2_partitions = []
    for i in range(2):
      for MH_succ_idx in range(len(MH_succs_2_partition[i])):
        if i == 0:
          neighb_MH_succs_2_partition = [MH_succs_2_partition[0][:MH_succ_idx]+MH_succs_2_partition[0][MH_succ_idx+1:],MH_succs_2_partition[1]+[MH_succs_2_partition[0][MH_succ_idx]]]
        else:
          neighb_MH_succs_2_partition = [MH_succs_2_partition[0]+[MH_succs_2_partition[1][MH_succ_idx]],MH_succs_2_partition[1][:MH_succ_idx]+MH_succs_2_partition[1][MH_succ_idx+1:]]
        if len(neighb_MH_succs_2_partition[0]) < len(neighb_MH_succs_2_partition[1]) or (len(neighb_MH_succs_2_partition[0]) == len(neighb_MH_succs_2_partition[1]) and min(neighb_MH_succs_2_partition[0]+neighb_MH_succs_2_partition[1]) in neighb_MH_succs_2_partition[0]):
          neighb_MH_succs_2_partition = [neighb_MH_succs_2_partition[1],neighb_MH_succs_2_partition[0]]
        neighb_MH_succs_2_partitions.append(neighb_MH_succs_2_partition)
    for MH0_succ_idx in range(len(MH_succs_2_partition[0])):
      for MH1_succ_idx in range(len(MH_succs_2_partition[1])):
        neighb_MH_succs_2_partition = [MH_succs_2_partition[0][:MH0_succ_idx]+[MH_succs_2_partition[1][MH1_succ_idx]]+MH_succs_2_partition[0][MH0_succ_idx+1:],MH_succs_2_partition[1][:MH1_succ_idx]+[MH_succs_2_partition[0][MH0_succ_idx]]+MH_succs_2_partition[1][MH1_succ_idx+1:]]
        if len(neighb_MH_succs_2_partition[0]) == len(neighb_MH_succs_2_partition[1]) and min(neighb_MH_succs_2_partition[0]+neighb_MH_succs_2_partition[1]) in neighb_MH_succs_2_partition[0]:
          neighb_MH_succs_2_partition = [neighb_MH_succs_2_partition[1],neighb_MH_succs_2_partition[0]]
        neighb_MH_succs_2_partitions.append(neighb_MH_succs_2_partition)
    return neighb_MH_succs_2_partitions

  """
  Returns a list of all valid frequency assignments for the ImpArborescence given a partition of the master hub's successors in 2 sets where
    'MH_succs_2_partition' is the partition.
  """
  def get_all_freq_assignments(self, MH_succs_2_partition):
    if self.prob_params.nb_freqs_per_channel == 1:
      return self.get_all_chan_assignments(MH_succs_2_partition)
    if not self.is_mixed:
      return self.get_all_freq_assignments_tree(MH_succs_2_partition=MH_succs_2_partition)
    else:
      return self.get_all_freq_assignments_mixed(MH_succs_2_partition=MH_succs_2_partition)

  """
  Returns a list of all valid frequency assignments for a strictly tree ImpArborescence (i.e. with no mesh clusters) given a partition of the master hub's successors in 2 sets where
    either
      'MH_succs_2_partition' is the partition;
    or
      'freq_assignment' is a partial frequency assignment;
      'current_nodes' is a list of nodes to assign a frequency that share the same 'current_channel';
      'next_nodes' is a list of nodes to assign a frequency that share the same next channel (1-'current_channel');
      'current_channel' corresponds to channel with
        'current_channel' = 0 --> channel 3+ (channel_index = -2);
        'current_channel' = 1 --> channel 4  (channel_index = -3).
  """
  def get_all_freq_assignments_tree(self, freq_assignment={}, current_nodes=[], next_nodes=[], current_channel=-1, MH_succs_2_partition=[]):
    if len(MH_succs_2_partition) == 2:
      init_current_nodes = MH_succs_2_partition[0]
      init_next_nodes = MH_succs_2_partition[1]
      
      freq_assignments = []
      for freq_index_0 in range(FreqMHz.nb):
        init_current_channel = 1-(-2-FreqMHz.channel_from_index(freq_index_0))
        if len(init_next_nodes) > 0:
          for freq_index_1 in range(2*init_current_channel,2*init_current_channel+2):
            init_freq_assignment = {self.MH:[freq_index_0,freq_index_1]}
            freq_assignments += self.get_all_freq_assignments_tree(freq_assignment=init_freq_assignment, current_nodes=init_current_nodes, next_nodes=init_next_nodes, current_channel=init_current_channel)
        else:
          init_freq_assignment = {self.MH:[freq_index_0,-1]}
          freq_assignments += self.get_all_freq_assignments_tree(freq_assignment=init_freq_assignment, current_nodes=init_current_nodes, next_nodes=init_next_nodes, current_channel=init_current_channel)
      return freq_assignments
    else:
      if len(current_nodes) > 0:
        new_node = current_nodes[0]
        new_node_succs = self.succ_lists[new_node]
        
        if len(new_node_succs) > 0:
          new_next_nodes = next_nodes + new_node_succs

          freq_assignments = []
          for freq_index in range(2*current_channel,2*current_channel+2):
            freq_assignments += self.get_all_freq_assignments_tree(freq_assignment=freq_assignment|{new_node:freq_index}, current_nodes=current_nodes[1:], next_nodes=new_next_nodes, current_channel=current_channel)
          return freq_assignments
        else:
          return self.get_all_freq_assignments_tree(freq_assignment=freq_assignment, current_nodes=current_nodes[1:], next_nodes=next_nodes, current_channel=current_channel)
      else:
        if len(next_nodes) > 0:
          return self.get_all_freq_assignments_tree(freq_assignment=freq_assignment, current_nodes=next_nodes, next_nodes=[], current_channel=1-current_channel)
        else:
          return [freq_assignment]
  
  """
  Returns a list of all valid frequency assignments for an ImpArborescence with mesh clusters given a partition of the master hub's successors in 2 sets where
    either
      'MH_succs_2_partition' is the partition;
    or
      'freq_assignment' is a partial frequency assignment;
      'current_nodes' is a list of nodes to assign a frequency that share the same 'current_channel';
      'next_nodes' is a list of nodes to assign a frequency that share the same next channel (1-'current_channel');
      'current_channel' corresponds to channel with
        'current_channel' = 0 --> channel 3+ (channel_index = -2);
        'current_channel' = 1 --> channel 4  (channel_index = -3).
  """
  def get_all_freq_assignments_mixed(self, freq_assignment={}, current_nodes=[], next_nodes=[], current_channel=-1, MH_succs_2_partition=[]):
    if len(MH_succs_2_partition) == 2:
      part1_is_mesh = MH_succs_2_partition[0][0] < 0
      if not part1_is_mesh:
        init_current_nodes = MH_succs_2_partition[0]
      else:
        init_current_nodes = self.succ_lists[MH_succs_2_partition[0][0]]

      part2_is_mesh = len(MH_succs_2_partition[1]) > 0 and MH_succs_2_partition[1][0] < 0
      if not part2_is_mesh:
        init_next_nodes = MH_succs_2_partition[1]
      else:
        init_next_nodes = self.succ_lists[MH_succs_2_partition[1][0]]
      
      freq_assignments = []
      for freq_index_0 in range(FreqMHz.nb):
        init_current_channel = 1-(-2-FreqMHz.channel_from_index(freq_index_0))
        if len(init_next_nodes) > 0:
          for freq_index_1 in range(2*init_current_channel,2*init_current_channel+2):
            if not part1_is_mesh:
              if not part2_is_mesh:
                init_freq_assignment = {self.MH:[freq_index_0,freq_index_1]}
              else:
                init_freq_assignment = {self.MH:[freq_index_0,freq_index_1], MH_succs_2_partition[1][0]:freq_index_1}
            else:
              if not part2_is_mesh:
                init_freq_assignment = {self.MH:[freq_index_0,freq_index_1], MH_succs_2_partition[0][0]:freq_index_0}
              else:
                init_freq_assignment = {self.MH:[freq_index_0,freq_index_1], MH_succs_2_partition[0][0]:freq_index_0, MH_succs_2_partition[1][0]:freq_index_1}
            freq_assignments += self.get_all_freq_assignments_mixed(freq_assignment=init_freq_assignment, current_nodes=init_current_nodes, next_nodes=init_next_nodes, current_channel=init_current_channel)
        else:
          if not part1_is_mesh:
            init_freq_assignment = {self.MH:[freq_index_0,-1]}
          else:
            init_freq_assignment = {self.MH:[freq_index_0,-1], MH_succs_2_partition[0][0]:freq_index_0}
          freq_assignments += self.get_all_freq_assignments_mixed(freq_assignment=init_freq_assignment, current_nodes=init_current_nodes, next_nodes=init_next_nodes, current_channel=init_current_channel)
      return freq_assignments
    else:
      if len(current_nodes) > 0:
        new_node = current_nodes[0]
        new_node_succs = self.succ_lists[new_node]
        
        if len(new_node_succs) > 0:
          new_node_succs_is_mesh = new_node_succs[0] < 0
          if not new_node_succs_is_mesh:
            new_next_nodes = next_nodes + new_node_succs
          else:
            new_next_nodes = next_nodes + self.succ_lists[new_node_succs[0]]

          freq_assignments = []
          for freq_index in range(2*current_channel,2*current_channel+2):
            if not new_node_succs_is_mesh:
              new_freq_assignment = freq_assignment|{new_node:freq_index}
            else:
              new_freq_assignment = freq_assignment|{new_node:freq_index, new_node_succs[0]:freq_index}
            freq_assignments += self.get_all_freq_assignments_mixed(freq_assignment=new_freq_assignment, current_nodes=current_nodes[1:], next_nodes=new_next_nodes, current_channel=current_channel)
          return freq_assignments
        else:
          return self.get_all_freq_assignments_mixed(freq_assignment=freq_assignment, current_nodes=current_nodes[1:], next_nodes=next_nodes, current_channel=current_channel)
      else:
        if len(next_nodes) > 0:
          return self.get_all_freq_assignments_mixed(freq_assignment=freq_assignment, current_nodes=next_nodes, next_nodes=[], current_channel=1-current_channel)
        else:
          return [freq_assignment]
  
  """
  Returns a list of all valid channel assignments for the ImpArborescence given a partition of the master hub's successors in 2 sets where
    'MH_succs_2_partition' is the partition.
  """
  def get_all_chan_assignments(self, MH_succs_2_partition):
    if not self.is_mixed:
      return self.get_all_chan_assignments_tree(MH_succs_2_partition)
    else:
      return self.get_all_chan_assignments_mixed(MH_succs_2_partition)

  """
  Returns a list of all valid channel assignments for a strictly tree ImpArborescence (i.e. with no mesh clusters) given a partition of the master hub's successors in 2 sets where
    'MH_succs_2_partition' is the partition.
  """
  def get_all_chan_assignments_tree(self, MH_succs_2_partition):
    MH_part_0_nodes = []
    MH_part_1_nodes_to_visit = []+MH_succs_2_partition[0]
    MH_part_1_nodes = []
    MH_part_0_nodes_to_visit = []+MH_succs_2_partition[1]
    while len(MH_part_0_nodes_to_visit) + len(MH_part_1_nodes_to_visit) > 0:
      if len(MH_part_0_nodes_to_visit) > 0:
        new_MH_part_0_node = MH_part_0_nodes_to_visit.pop(0)
        new_node_succs = self.succ_lists[new_MH_part_0_node]
        if len(new_node_succs) > 0:
          MH_part_0_nodes.append(new_MH_part_0_node)
          MH_part_1_nodes_to_visit += new_node_succs
      if len(MH_part_1_nodes_to_visit) > 0:
        new_MH_part_1_node = MH_part_1_nodes_to_visit.pop(0)
        new_node_succs = self.succ_lists[new_MH_part_1_node]
        if len(new_node_succs) > 0:
          MH_part_1_nodes.append(new_MH_part_1_node)
          MH_part_0_nodes_to_visit += new_node_succs
    chan_assignments = []
    for MH_part_0_chan in range(-3,-1):
      MH_part_1_chan = -5 - MH_part_0_chan
      if len(MH_succs_2_partition[1]) > 0:
        chan_assignment = {self.MH:[MH_part_0_chan,MH_part_1_chan]}
      else:
        chan_assignment = {self.MH:[MH_part_0_chan,-1]}
      for MH_part_0_node in MH_part_0_nodes:
        chan_assignment[MH_part_0_node] = MH_part_0_chan
      for MH_part_1_node in MH_part_1_nodes:
        chan_assignment[MH_part_1_node] = MH_part_1_chan
      chan_assignments.append(chan_assignment)
    return chan_assignments

  """
  Returns a list of all valid channel assignments for an ImpArborescence with mesh clusters given a partition of the master hub's successors in 2 sets where
    'MH_succs_2_partition' is the partition.
  """
  def get_all_chan_assignments_mixed(self, MH_succs_2_partition):
    if MH_succs_2_partition[0][0] >= 0:
      MH_part_0_nodes = []
      MH_part_1_nodes_to_visit = []+MH_succs_2_partition[0]
    else:
      MH_part_0_nodes = [MH_succs_2_partition[0][0]]
      MH_part_1_nodes_to_visit = []+self.succ_lists[MH_succs_2_partition[0][0]]
    if len(MH_succs_2_partition[1]) == 0 or MH_succs_2_partition[1][0] >= 0:
      MH_part_1_nodes = []
      MH_part_0_nodes_to_visit = []+MH_succs_2_partition[1]
    else:
      MH_part_1_nodes = [MH_succs_2_partition[1][0]]
      MH_part_0_nodes_to_visit = []+self.succ_lists[MH_succs_2_partition[1][0]]
    while len(MH_part_0_nodes_to_visit) + len(MH_part_1_nodes_to_visit) > 0:
      if len(MH_part_0_nodes_to_visit) > 0:
        new_MH_part_0_node = MH_part_0_nodes_to_visit.pop(0)
        new_node_succs = self.succ_lists[new_MH_part_0_node]
        if len(new_node_succs) > 0:
          if new_node_succs[0] >= 0:
            MH_part_0_nodes.append(new_MH_part_0_node)
            MH_part_1_nodes_to_visit += new_node_succs
          else:
            MH_part_0_nodes += [new_MH_part_0_node, new_node_succs[0]]
            MH_part_1_nodes_to_visit += self.succ_lists[new_node_succs[0]]
      if len(MH_part_1_nodes_to_visit) > 0:
        new_MH_part_1_node = MH_part_1_nodes_to_visit.pop(0)
        new_node_succs = self.succ_lists[new_MH_part_1_node]
        if len(new_node_succs) > 0:
          if new_node_succs[0] >= 0:
            MH_part_1_nodes.append(new_MH_part_1_node)
            MH_part_0_nodes_to_visit += new_node_succs
          else:
            MH_part_1_nodes += [new_MH_part_1_node, new_node_succs[0]]
            MH_part_0_nodes_to_visit += self.succ_lists[new_node_succs[0]]
    chan_assignments = []
    for MH_part_0_chan in range(-3,-1):
      MH_part_1_chan = -5 - MH_part_0_chan
      if len(MH_succs_2_partition[1]) > 0:
        chan_assignment = {self.MH:[MH_part_0_chan,MH_part_1_chan]}
      else:
        chan_assignment = {self.MH:[MH_part_0_chan,-1]}
      for MH_part_0_node in MH_part_0_nodes:
        chan_assignment[MH_part_0_node] = MH_part_0_chan
      for MH_part_1_node in MH_part_1_nodes:
        chan_assignment[MH_part_1_node] = MH_part_1_chan
      chan_assignments.append(chan_assignment)
    return chan_assignments
  
  """
  Returns a Netwok with a greedily optimized frequency assignment from the ImpArborescence given a partition of the master hub's successors in 2 sets and a channel assignment where
    'MH_succs_2_partition' is the partition;
    'chan_assignment' is the channel assignment.
  """
  def get_greedy_network(self, MH_succs_2_partition, chan_assignment):
    if self.prob_params.nb_freqs_per_channel == 1:
      network = Network(self, MH_succs_2_partition, chan_assignment)
      obj_value = network.get_obj_value(obj_func="full")
      return network, obj_value
    if not self.is_mixed:
      return self.get_greedy_network_tree(MH_succs_2_partition, chan_assignment)
    else:
      return self.get_greedy_network_mixed(MH_succs_2_partition, chan_assignment)

  """
  Returns a Netwok with a greedily optimized frequency assignment from a strictly tree ImpArborescence (i.e. with no mesh clusters) given a partition of the master hub's successors
  in 2 sets and a channel assignment where
    'MH_succs_2_partition' is the partition;
    'chan_assignment' is the channel assignment.
  """
  def get_greedy_network_tree(self, MH_succs_2_partition, chan_assignment):
    MH = self.MH
    succ_lists = self.succ_lists
    multibeam = self.prob_params.multibeam
    angles_from_to_rad = self.prob_instance.angles_from_to_rad
    pattern_exponents = self.prob_params.get_pattern_exponents()
    preds_to_visit = [MH]
    partial_tree_nodes = [MH]
    best_obj_value = float('-inf')
    freq_assignment = {}
    uplink_ant_alignments_rad = {}
    uplink_ant_nb_beams = {}
    #networks = []
    while len(preds_to_visit)>0:
      pred = preds_to_visit.pop(0)
      if pred == MH:
        best_freq_idx = [-1,-1]
        best_downlink_ant_alignments_rad = [-1,-1]
        best_downlink_ant_beams = [-1,-1]
        for i in range(2):
          succs = MH_succs_2_partition[i]
          if len(succs) > 0:
            preds_to_visit += succs
            partial_tree_nodes += succs
            loc_chan = chan_assignment[pred][i]
            best_worst_sig_strength = float('-inf')
            for loc_freq_idx in range(2*(-2 - loc_chan), 2*(-2 - loc_chan)+2):
              if not multibeam:
                antenna_types_3dB_beam_widths = self.prob_params.get_3dB_beam_width_deg(loc_freq_idx)
              else:
                antenna_types_3dB_beam_widths = -1
              loc_downlink_ant_alignment_rad, loc_downlink_ant_beams = Network.get_opt_dl_ant_config(pred, succs, multibeam, angles_from_to_rad, loc_freq_idx, antenna_types_3dB_beam_widths)
              loc_uplink_ant_alignments_rad = {}
              loc_uplink_ant_nb_beams = {}
              loc_worst_sig_strength = float('inf')
              for succ in succs:
                loc_uplink_ant_alignments_rad[succ] = angles_from_to_rad[succ,pred]
                loc_uplink_ant_nb_beams[succ] = 1
                sig_strength = self.prob_params.get_ul_dB_signal_strength(0, pattern_exponents, 1, 0, loc_freq_idx)
                sig_strength_pred_succ = self.prob_params.get_dl_lin_signal_strength(angles_from_to_rad[pred,succ], pattern_exponents, loc_downlink_ant_beams, loc_downlink_ant_alignment_rad, loc_freq_idx)
                sig_strength += 10*np.log10(sig_strength_pred_succ)
                if sig_strength < loc_worst_sig_strength:
                  loc_worst_sig_strength = sig_strength
              if loc_worst_sig_strength > best_worst_sig_strength:
                best_worst_sig_strength = loc_worst_sig_strength
                best_freq_idx[i] = loc_freq_idx
                best_uplink_ant_alignments_rad = loc_uplink_ant_alignments_rad
                best_uplink_ant_nb_beams = loc_uplink_ant_nb_beams
                best_downlink_ant_alignments_rad[i] = loc_downlink_ant_alignment_rad
                best_downlink_ant_beams[i] = loc_downlink_ant_beams
            if best_worst_sig_strength > float('-inf'):
              uplink_ant_alignments_rad = uplink_ant_alignments_rad|best_uplink_ant_alignments_rad
              uplink_ant_nb_beams = uplink_ant_nb_beams|best_uplink_ant_nb_beams
            else:
              best_freq_idx[i] = loc_freq_idx
              uplink_ant_alignments_rad = uplink_ant_alignments_rad|loc_uplink_ant_alignments_rad
              uplink_ant_nb_beams = uplink_ant_nb_beams|loc_uplink_ant_nb_beams
              best_downlink_ant_alignments_rad[i] = loc_downlink_ant_alignment_rad
              best_downlink_ant_beams[i] = loc_downlink_ant_beams
        freq_assignment[pred] = best_freq_idx
        downlink_ant_alignments_rad = {pred:best_downlink_ant_alignments_rad}
        downlink_ant_beams = {pred:best_downlink_ant_beams}
      else:
        succs = succ_lists[pred]
        if len(succs) > 0:
          chan = chan_assignment[pred]
          preds_to_visit += succs
          partial_tree_nodes += succs
          best_obj_value = float('-inf')
          for freq_idx in range(2*(-2 - chan), 2*(-2 - chan)+2):
            partial_network = self.create_partial_network(MH_succs_2_partition, partial_tree_nodes, freq_assignment|{pred:freq_idx}, [uplink_ant_alignments_rad|{}, uplink_ant_nb_beams|{}, downlink_ant_alignments_rad|{}, downlink_ant_beams|{}])
            partial_obj_value = partial_network.get_obj_value(obj_func="full")
            if partial_obj_value > best_obj_value:
              best_obj_value = partial_obj_value
              network = partial_network
          freq_assignment = network.freq_assignment
          uplink_ant_alignments_rad = network.uplink_ant_alignments_rad
          uplink_ant_nb_beams = network.uplink_ant_nb_beams
          downlink_ant_alignments_rad = network.downlink_ant_alignments_rad
          downlink_ant_beams = network.downlink_ant_beams
          #networks.append(network)
    if best_obj_value == float('-inf'):
      network = Network(self, MH_succs_2_partition, freq_assignment, [uplink_ant_alignments_rad, uplink_ant_nb_beams, downlink_ant_alignments_rad, downlink_ant_beams])
      best_obj_value = network.get_obj_value(obj_func="full")
    #nb_y, nb_x = MWMB_Plot.get_subplot_nb_y_and_nb_x(len(networks))
    #MWMB_Plot.subplot(networks, nb_y, nb_x)
    return network, best_obj_value

  """
  Returns a Netwok with a greedily optimized frequency assignment from an ImpArborescence ith mesh clusters given a partition of the master hub's successors in 2 sets and a channel
  assignment where
    'MH_succs_2_partition' is the partition;
    'chan_assignment' is the channel assignment.
  """
  def get_greedy_network_mixed(self, MH_succs_2_partition, chan_assignment):
    MH = self.MH
    succ_lists = self.succ_lists
    multibeam = self.prob_params.multibeam
    angles_from_to_rad = self.prob_instance.angles_from_to_rad
    pattern_exponents = self.prob_params.get_pattern_exponents()
    preds_to_visit = [MH]
    partial_tree_nodes = [MH]
    best_obj_value = float('-inf')
    freq_assignment = {}
    uplink_ant_alignments_rad = {}
    uplink_ant_nb_beams = {}
    #networks = []
    while len(preds_to_visit)>0:
      pred = preds_to_visit.pop(0)
      if pred == MH and pred >= 0:
        best_freq_idx = [-1,-1]
        best_downlink_ant_alignments_rad = [-1,-1]
        best_downlink_ant_beams = [-1,-1]
        for i in range(2):
          succs = MH_succs_2_partition[i]
          if len(succs) > 0:
            if succs[0] >= 0:
              preds_to_visit += succs
              partial_tree_nodes += succs
              loc_chan = chan_assignment[pred][i]
              best_worst_sig_strength = float('-inf')
              for loc_freq_idx in range(2*(-2 - loc_chan), 2*(-2 - loc_chan)+2):
                if not multibeam:
                  antenna_types_3dB_beam_widths = self.prob_params.get_3dB_beam_width_deg(loc_freq_idx)
                else:
                  antenna_types_3dB_beam_widths = -1
                loc_downlink_ant_alignment_rad, loc_downlink_ant_beams = Network.get_opt_dl_ant_config(pred, succs, multibeam, angles_from_to_rad, loc_freq_idx, antenna_types_3dB_beam_widths)
                loc_uplink_ant_alignments_rad = {}
                loc_uplink_ant_nb_beams = {}
                loc_worst_sig_strength = float('inf')
                for succ in succs:
                  loc_uplink_ant_alignments_rad[succ] = angles_from_to_rad[succ,pred]
                  loc_uplink_ant_nb_beams[succ] = 1
                  sig_strength = self.prob_params.get_ul_dB_signal_strength(0, pattern_exponents, 1, 0, loc_freq_idx)
                  sig_strength_pred_succ = self.prob_params.get_dl_lin_signal_strength(angles_from_to_rad[pred,succ], pattern_exponents, loc_downlink_ant_beams, loc_downlink_ant_alignment_rad, loc_freq_idx)
                  sig_strength += 10*np.log10(sig_strength_pred_succ)
                  if sig_strength < loc_worst_sig_strength:
                    loc_worst_sig_strength = sig_strength
                if loc_worst_sig_strength > best_worst_sig_strength:
                  best_worst_sig_strength = loc_worst_sig_strength
                  best_freq_idx[i] = loc_freq_idx
                  best_uplink_ant_alignments_rad = loc_uplink_ant_alignments_rad
                  best_uplink_ant_nb_beams = loc_uplink_ant_nb_beams
                  best_downlink_ant_alignments_rad[i] = loc_downlink_ant_alignment_rad
                  best_downlink_ant_beams[i] = loc_downlink_ant_beams
              if best_worst_sig_strength > float('-inf'):
                uplink_ant_alignments_rad = uplink_ant_alignments_rad|best_uplink_ant_alignments_rad
                uplink_ant_nb_beams = uplink_ant_nb_beams|best_uplink_ant_nb_beams
              else:
                best_freq_idx[i] = loc_freq_idx
                uplink_ant_alignments_rad = uplink_ant_alignments_rad|loc_uplink_ant_alignments_rad
                uplink_ant_nb_beams = uplink_ant_nb_beams|loc_uplink_ant_nb_beams
                best_downlink_ant_alignments_rad[i] = loc_downlink_ant_alignment_rad
                best_downlink_ant_beams[i] = loc_downlink_ant_beams
            else:
              mesh_idx = succs[0]
              succs = succ_lists[mesh_idx]
              preds_to_visit += succs
              partial_tree_nodes += succs
              if i == 0:
                curr_MH_succs_2_partition = [MH_succs_2_partition[0], []]
              else:
                curr_MH_succs_2_partition = MH_succs_2_partition
              loc_chan = chan_assignment[pred][i]
              best_downlink_ant_alignments_rad[i] = 0
              best_downlink_ant_beams[i] = []
              best_obj_value = float('-inf')
              for loc_freq_idx in range(2*(-2 - loc_chan), 2*(-2 - loc_chan)+2):
                if i == 0:
                  MH_freq_idcs = [loc_freq_idx,best_freq_idx[1]]
                else:
                  MH_freq_idcs = [best_freq_idx[0],loc_freq_idx]
                partial_network = self.create_partial_network(curr_MH_succs_2_partition, partial_tree_nodes, freq_assignment|{pred:MH_freq_idcs, mesh_idx:loc_freq_idx}, [uplink_ant_alignments_rad|{}, uplink_ant_nb_beams|{}, {pred:best_downlink_ant_alignments_rad}, {pred:best_downlink_ant_beams}])
                partial_obj_value = partial_network.get_obj_value(obj_func="full")
                if partial_obj_value > best_obj_value:
                  best_obj_value = partial_obj_value
                  best_freq_idx[i] = loc_freq_idx
                  network = partial_network
              freq_assignment = network.freq_assignment
              uplink_ant_alignments_rad = network.uplink_ant_alignments_rad
              uplink_ant_nb_beams = network.uplink_ant_nb_beams
        freq_assignment[pred] = best_freq_idx
        downlink_ant_alignments_rad = {pred:best_downlink_ant_alignments_rad}
        downlink_ant_beams = {pred:best_downlink_ant_beams}
      else:
        succs = succ_lists[pred]
        if len(succs) > 0:
          if pred == MH:
            downlink_ant_alignments_rad = {}
            downlink_ant_beams = {}
            chan = chan_assignment[pred][0]
            succ_is_mesh = False
          else:
            chan = chan_assignment[pred]
            succ_is_mesh = succs[0] < 0
            if succ_is_mesh:
              mesh_idx = succs[0]
              succs = succ_lists[mesh_idx]
          preds_to_visit += succs
          partial_tree_nodes += succs
          best_obj_value = float('-inf')
          for freq_idx in range(2*(-2 - chan), 2*(-2 - chan)+2):
            if not succ_is_mesh:
              new_freq_assignments = {pred:freq_idx}
            else:
              new_freq_assignments = {pred:freq_idx, mesh_idx:freq_idx}
            partial_network = self.create_partial_network(MH_succs_2_partition, partial_tree_nodes, freq_assignment|new_freq_assignments, [uplink_ant_alignments_rad|{}, uplink_ant_nb_beams|{}, downlink_ant_alignments_rad|{}, downlink_ant_beams|{}])
            partial_obj_value = partial_network.get_obj_value(obj_func="full")
            if partial_obj_value > best_obj_value:
              best_obj_value = partial_obj_value
              network = partial_network
          freq_assignment = network.freq_assignment
          uplink_ant_alignments_rad = network.uplink_ant_alignments_rad
          uplink_ant_nb_beams = network.uplink_ant_nb_beams
          downlink_ant_alignments_rad = network.downlink_ant_alignments_rad
          downlink_ant_beams = network.downlink_ant_beams
          #networks.append(network)
    if best_obj_value == float('-inf'):
      network = Network(self, MH_succs_2_partition, freq_assignment, [uplink_ant_alignments_rad, uplink_ant_nb_beams, downlink_ant_alignments_rad, downlink_ant_beams])
      best_obj_value = network.get_obj_value(obj_func="full")
    #nb_y, nb_x = MWMB_Plot.get_subplot_nb_y_and_nb_x(len(networks))
    #MWMB_Plot.subplot(networks, nb_y, nb_x)
    return network, best_obj_value

  """
  Returns a partial Network through a partial version of the ImpArborescence given a partition of the master hub's successors in 2 sets, a frequency assignment and a partial antenna
  configuration where
    'MH_succs_2_partition' is the partition;
    'partial_tree_nodes' is a list of the explicit nodes to be kept in the partial version of the ImpArborescence;
    'freq_assignment' is the frequency assignment;
    'partial_alignments_and_beams' is a partial antenna configuration (see Network for details.)
  """
  def create_partial_network(self, MH_succs_2_partition, partial_tree_nodes, freq_assignment, partial_alignments_and_beams):
    partial_adj_matrix = 1*self.adj_matrix
    for w in range(self.prob_instance.nb_nodes):
      if not w in partial_tree_nodes:
        partial_adj_matrix[w, :] = 0
        partial_adj_matrix[:, w] = 0
    partial_imp_arb = ImpArborescence(ImpTree(self.prob_instance, self.prob_params, np.abs(partial_adj_matrix)), self.MH)
    return Network(partial_imp_arb, MH_succs_2_partition, freq_assignment, partial_alignments_and_beams)


"""
Network
-------
Object representing a fully configured Network on a specific ProbInstance with specific ProbParameters.

  .imp_arb {ImpArborescence} --> associated implicit arborescence
  .is_mixed {boolean} --> whether the network contains mesh clusters (True) or not (False)
  .MH_succs_2_partition {list of 2 lists of implicit successors to the master hub} --> partition of the master hub's successors in 2 sets such that the first list is the first set
    and the second list is the second set
  .has_freq_assignment {boolean} --> whether the network has a channel/frequency assignment (True) or not (False)
  .uplink_ant_alignments_rad {dictionary from explicit node to associated angle} --> describes the uplink antenna alignment in radian of each explicit node that has one
  .uplink_ant_nb_beams {dictionary from explicit node to associated int} --> describes the number of activated beams of the uplink antenna of each explicit node that has one such that
    nb_beams = 1 --> one activated beam (multi-beam) / parabolic antenna (single-beam)
    nb_beams = 0 --> omni-directional mode (multi-beam) / omni-directional antenna (single-beam)
  .downlink_ant_alignments_rad {dictionary from explicit node to associated angle} --> describes the downlink antenna alignment in radian (of beam 0) of each explicit node that has one
    such that
    alignments_rad[master_hub] = [alignment_part1, alignment_part2]
    alignments_rad[other_predecessor] = alignment
  .downlink_ant_beams {dictionary from explicit node to associated list of beam indices} --> describes the activated beams of the downlink antenna of each explicit node that has one
    such that
    beams[master_hub] = [beam_list_part1, beam_list_part2]
    beams[other_predecessor] = beam_list
    beam_list = [] --> omni-directional mode (multi-beam) / omni-directional antenna (single-beam)
    beam_list = [0] --> one activated beam (multi-beam) / parabolic antenna (single-beam)
    beam_list = [0, beam2] --> two activated beam (multi-beam) / panel antenna (single-beam)
    beam_list = [0, beam2, beam3] --> three activated beam (multi-beam) / sector antenna (single-beam)
    beam_list = [0, beam2, beam3, ...] --> multiple activated beam (multi-beam) / ---
  .has_obj_value {boolean} --> whether its objective/pseudo-objective value has been evaluated and stored (True) or not (False)

  (defined if .has_freq_assignment)
    .freq_assignment {dictionary from implicit node (with successors) to associated frequency index} --> describes the channel of the downlink connections of each implicit nodes that
      have successors such that
        .freq_assignment[master_hub] = [freq_index_part1, freq_index_part2]
        .freq_assignment[other_predecessor] = freq_index

  (defined if .has_obj_value)
    .direct_TPs {numpy array of floats of shape ('nb_nodes', 'nb_nodes')} --> matrix containing the values of the direct throughputs such that
      a_uv is the direct throughput between node u and one of its successors v for a tree edge [u,v] (i.e. u and v are not in a same mesh cluster)
      a_uv = a_vu is the direct throughput in both directions between nodes u and v that are in a same mesh cluster
    .nbs_descs_inc_self {list of ints of size 'nb_nodes' if without mesh clusters or 'nb_nodes'+nb_mesh_clusters+1 otherwise} --> for each implicit node, its number of explicit
      descendants including itself
    .A_mesh_routes {dictionary from mesh index to associated dictionary from explicit node to associated dictionary from explicit node to associated list of tuples of explicit nodes}
      --> for each mesh index, for each pair of explicit nodes in that mesh cluster, the lists of edges that are used in the path between the two explicit nodes in that cluster for,
          at least, scenario A (if no other mesh routes are defined, these are also used for scenarios B and C)
    .B_mesh_TP_denoms {numpy array of ints of shape ('nb_nodes', 'nb_nodes')} --> matrix containing the denominators of the effective throughput computation (i.e. the number of data
      streams that go through each edge) for scenario B such that
        a_uv is the denominator associated with the tree edge [u,v] where u is the predecessor (i.e. u and v are not in a same mesh cluster)
        a_uv = a_vu is the denominator in both directions associated with the mesh edge [u,v] between nodes u and v that are in a same mesh cluster
    .C_mesh_TP_denoms {numpy array of ints of shape ('nb_nodes', 'nb_nodes')} --> matrix containing the denominators of the effective throughput computation (i.e. the number of data
      streams that go through each edge) for scenario C such that
        a_uv is the denominator associated with the tree edge [u,v] where u is the predecessor (i.e. u and v are not in a same mesh cluster)
        a_uv = a_vu is the denominator in both directions associated with the mesh edge [u,v] between nodes u and v that are in a same mesh cluster
    .A_obj_min {float} --> the minimum value of objective A for all edges (i.e. the bottleneck of scenario A)
    .A_obj_value {float} --> the objective A value of the network (taking into account the minimum and mean values)
    .B_obj_min {float} --> the minimum value of objective B for all edges (i.e. the bottleneck of scenario B)
    .B_obj_value {float} --> the objective B value of the network (taking into account the minimum and mean values)
    .C_obj_min {float} --> the minimum value of objective C for all edges (i.e. the bottleneck of scenario C)
    .C_obj_value {float} --> the objective C value of the network (taking into account the minimum and mean values)
    .obj_value {float} --> the aggregate objective value (taking into account the weighted values of each scenario)
    .A_lim_links {list of tuples of eplicit nodes} --> list of limiting edges of scenario A (the edges with the minimal value of objective A)
    .B_lim_links {list of tuples of eplicit nodes} --> list of limiting edges of scenario B (the edges with the minimal value of objective B)
    .C_lim_links {list of tuples of eplicit nodes} --> list of limiting edges of scenario C (the edges with the minimal value of objective C)
    
    (defined if evaluated with objective or greedy pseudo-objective and ProbParameters.mesh_high_traffic_routes_prioritization > 0)
      .C_mesh_routes = {dictionary from mesh index to associated dictionary from explicit node to associated dictionary from explicit node to associated list of tuples of explicit nodes}
        --> for each mesh index, for each pair of explicit nodes in that mesh cluster, the lists of edges that are used in the path between the two explicit nodes in that cluster for
            scenarios B and C
      .A_mesh_TP_denoms {numpy array of ints of shape ('nb_nodes', 'nb_nodes')} --> matrix containing the denominators of the effective throughput computation (i.e. the number of data
        streams that go through each edge) for scenario A such that
          a_uv is the denominator associated with the tree edge [u,v] where u is the predecessor (i.e. u and v are not in a same mesh cluster)
          a_uv = a_vu is the denominator in both directions associated with the mesh edge [u,v] between nodes u and v that are in a same mesh cluster
"""
class Network:
  """
  Initializes a Network object where
    'imp_arb' is the associated implicit arborescence;
    'MH_succs_2_partition' is the partition of the master hub's successors in 2 sets;
    'freq_assignment' (optional) is the dictionary that describes the channel of the downlink connections of each implicit nodes that have successors, such that
      .freq_assignment[master_hub] = [freq_index_part1, freq_index_part2];
      .freq_assignment[other_predecessor] = freq_index;
    'alignments_and_beams' (optional) is a list containing the alignments and beam configurations of all the antennas, respectively as
      [uplink_ant_alignments_rad, uplink_ant_nb_beams, downlink_ant_alignments_rad, downlink_ant_beams].
  """
  def __init__(self, imp_arb, MH_succs_2_partition, freq_assignment={}, alignments_and_beams=[]):
    self.imp_arb = imp_arb # ImpArborescence
    self.is_mixed = imp_arb.is_mixed # boolean
    self.MH_succs_2_partition = MH_succs_2_partition # [part1,part2] a partition of MH's successors
    self.has_freq_assignment = len(freq_assignment) > 0 # boolean
    if self.has_freq_assignment:
      self.freq_assignment = freq_assignment # dict : imp_node -> freq of successor links (MH -> [freq_part1,freq_part2])
    if True: # alignments and beams
      if len(alignments_and_beams) != 4:
        empty_align_beam = True
        uplink_ant_alignments_rad = {}
        uplink_ant_nb_beams = {}
        downlink_ant_alignments_rad = {}
        downlink_ant_beams = {}
      else:
        empty_align_beam = False
        uplink_ant_alignments_rad = copy.copy(alignments_and_beams[0])
        uplink_ant_nb_beams = copy.copy(alignments_and_beams[1])
        downlink_ant_alignments_rad = copy.copy(alignments_and_beams[2])
        downlink_ant_beams = copy.copy(alignments_and_beams[3])
      multibeam = self.imp_arb.prob_params.multibeam
      angles_from_to_rad = self.imp_arb.prob_instance.angles_from_to_rad
      for exp_node in range(self.imp_arb.prob_instance.nb_nodes):
        if exp_node != self.imp_arb.MH:
          if exp_node in self.imp_arb.preds:
            if empty_align_beam or not exp_node in uplink_ant_alignments_rad:
              pred = self.imp_arb.preds[exp_node]
              if pred >= 0:
                uplink_ant_alignments_rad[exp_node] = angles_from_to_rad[exp_node,pred]
                uplink_ant_nb_beams[exp_node] = 1
              else:
                uplink_ant_alignments_rad[exp_node] = 0
                uplink_ant_nb_beams[exp_node] = 0
            if exp_node in self.imp_arb.succ_lists:
              succs = self.imp_arb.succ_lists[exp_node]
              if len(succs) > 0 and (empty_align_beam or not exp_node in downlink_ant_alignments_rad or not exp_node in downlink_ant_beams):
                if self.has_freq_assignment:
                  freq = self.freq_assignment[exp_node]
                else:
                  freq = -1
                if not multibeam:
                  antenna_types_3dB_beam_widths = self.imp_arb.prob_params.get_3dB_beam_width_deg(freq)
                else:
                  antenna_types_3dB_beam_widths = -1
                downlink_ant_alignment_rad, exp_node_downlink_ant_beams = Network.get_opt_dl_ant_config(exp_node, succs, multibeam, angles_from_to_rad, freq, antenna_types_3dB_beam_widths)
                downlink_ant_alignments_rad[exp_node] = downlink_ant_alignment_rad
                downlink_ant_beams[exp_node] = exp_node_downlink_ant_beams
        else:
          if len(MH_succs_2_partition[0]) > 0 and (empty_align_beam or not exp_node in downlink_ant_alignments_rad or not exp_node in downlink_ant_beams):
            if self.has_freq_assignment:
              freq = self.freq_assignment[exp_node][0]
            else:
              freq = -1
            if not multibeam:
              antenna_types_3dB_beam_widths = self.imp_arb.prob_params.get_3dB_beam_width_deg(freq)
            else:
              antenna_types_3dB_beam_widths = -1
            downlink_ant_alignment_rad_0, downlink_ant_beams_0 = Network.get_opt_dl_ant_config(exp_node, MH_succs_2_partition[0], multibeam, angles_from_to_rad, freq, antenna_types_3dB_beam_widths)
            if len(MH_succs_2_partition[1]) > 0:
              if self.has_freq_assignment:
                freq = self.freq_assignment[exp_node][1]
              else:
                freq = -1
              if not multibeam:
                antenna_types_3dB_beam_widths = self.imp_arb.prob_params.get_3dB_beam_width_deg(freq)
              else:
                antenna_types_3dB_beam_widths = -1
              downlink_ant_alignment_rad_1, downlink_ant_beams_1 = Network.get_opt_dl_ant_config(exp_node, MH_succs_2_partition[1], multibeam, angles_from_to_rad, freq, antenna_types_3dB_beam_widths)
            else:
              downlink_ant_alignment_rad_1 = -1
              downlink_ant_beams_1 = [-1]
            downlink_ant_alignments_rad[exp_node] = [downlink_ant_alignment_rad_0, downlink_ant_alignment_rad_1]
            downlink_ant_beams[exp_node] = [downlink_ant_beams_0, downlink_ant_beams_1]
    self.uplink_ant_alignments_rad = uplink_ant_alignments_rad # dict : exp_node -> alignment of uplink antenna (rad)
    self.uplink_ant_nb_beams = uplink_ant_nb_beams # dict : exp_node -> nb activated beams (0 = omni)
    self.downlink_ant_alignments_rad = downlink_ant_alignments_rad # dict : exp_node -> alignment of downlink antenna (rad)
    self.downlink_ant_beams = downlink_ant_beams # dict : exp_node -> list of beams (empty list = omni])
    self.has_obj_value = False # boolean
  
  """
  Returns the alignment and activated beams of a downlink antenna given by the heuristic where
    'exp_node' is the node at which it is located;
    'succs' is a list of the successors to which the antenna should send signal;
    'multibeam' is whether we are in the multi-beam case (True) or the single-beam case (False), given by ProbParameters;
    'angles_from_to_rad' is the angles between every pair of explicit nodes, given by ProbInstance;
    'freq' is the frequency index of the connection;
    'antenna_types_3dB_beam_widths' is a list of the 3 dB beam widths of the single-beam antennas (omni not included).
  """
  @staticmethod
  def get_opt_dl_ant_config(exp_node, succs, multibeam, angles_from_to_rad, freq, antenna_types_3dB_beam_widths):
    if succs[0] < 0:
      downlink_ant_alignment_rad = 0.0
      downlink_ant_beams = []
    elif len(succs) == 1:
      downlink_ant_alignment_rad = angles_from_to_rad[exp_node,succs[0]]
      downlink_ant_beams = [0]
    else:
      succs_rad = [angles_from_to_rad[exp_node,succ] for succ in succs]
      succs_rad.sort()
      if multibeam:
        best_nb_activ_beams = 25
        for beam0_min_angle in succs_rad:
          beams_min_angles = (np.pi/12) * np.array(range(24)) + beam0_min_angle
          tmp_min_min_angle_beams = np.where(beams_min_angles > 2* np.pi)[0]
          if len(tmp_min_min_angle_beams) > 0:
            min_min_angle_beam = tmp_min_min_angle_beams[0]
          else:
            min_min_angle_beam = 0
          beams_min_angles[beams_min_angles > 2* np.pi] -= 2*np.pi
          succ_idx = 0
          nb_activ_beams = 0
          beam_nodes = {}
          for beam in list(range(min_min_angle_beam,24))+list(range(min_min_angle_beam)):
            while succ_idx < len(succs_rad) and succs_rad[succ_idx] < beams_min_angles[beam]:
              if not (beam-1) in beam_nodes:
                nb_activ_beams += 1
                beam_nodes[beam-1] = []
              beam_nodes[beam-1].append(succ_idx)
              succ_idx += 1
          while succ_idx < len(succs_rad) and succs_rad[succ_idx] < 2*np.pi:
            if not (min_min_angle_beam-1) in beam_nodes:
              nb_activ_beams += 1
              beam_nodes[min_min_angle_beam-1] = []
            beam_nodes[min_min_angle_beam-1].append(succ_idx)
            succ_idx += 1
          if nb_activ_beams < best_nb_activ_beams:
            best_nb_activ_beams = nb_activ_beams
            best_beams_min_angles = beams_min_angles
            best_beam_nodes = beam_nodes
        switch_to_omni_mode = len(best_beam_nodes) > 7
        if switch_to_omni_mode:
          downlink_ant_alignment_rad = 0.0
          downlink_ant_beams = []
        else:
          min_extra_angular_space = 2*np.pi
          for activ_beam in best_beam_nodes:
            extra_angular_space = best_beams_min_angles[activ_beam] - succs_rad[best_beam_nodes[activ_beam][-1]]
            if extra_angular_space < min_extra_angular_space:
              min_extra_angular_space = extra_angular_space
          downlink_ant_alignment_rad = best_beams_min_angles[0] - min_extra_angular_space/2 + np.pi/48
          if -1 in best_beam_nodes:
            best_beam_nodes[23] = best_beam_nodes.pop(-1)
          downlink_ant_beams = list(best_beam_nodes.keys())
      else:
        max_counterclockwise_angular_distance_rad = -1
        for succ_idx in range(len(succs_rad)):
          curr_succ_rad = succs_rad[succ_idx]
          if succ_idx < len(succs_rad)-1:
            next_succ_rad = succs_rad[succ_idx+1]
          else:
            next_succ_rad = succs_rad[0]
          if next_succ_rad >= curr_succ_rad:
            counterclockwise_angular_distance_rad = next_succ_rad - curr_succ_rad
          else:
            counterclockwise_angular_distance_rad = 2*np.pi + (next_succ_rad - curr_succ_rad)
          if counterclockwise_angular_distance_rad > max_counterclockwise_angular_distance_rad:
            max_counterclockwise_angular_distance_rad = counterclockwise_angular_distance_rad
            max_succ_idx = succ_idx
        min_beam_width_rad = 2*np.pi - max_counterclockwise_angular_distance_rad
        for antenna_type_idx in range(len(antenna_types_3dB_beam_widths)):
          if min_beam_width_rad <= (np.pi/180)*antenna_types_3dB_beam_widths[antenna_type_idx]:
            break
          if antenna_type_idx == len(antenna_types_3dB_beam_widths)-1:
            antenna_type_idx = 4
        if antenna_type_idx == 4:
          downlink_ant_alignment_rad = 0.0
          downlink_ant_beams = []
        else:
          curr_succ_rad = succs_rad[max_succ_idx]
          if max_succ_idx < len(succs_rad)-1:
            next_succ_rad = succs_rad[max_succ_idx+1]
          else:
            next_succ_rad = succs_rad[0]
          if next_succ_rad <= curr_succ_rad:
            downlink_ant_alignment_rad = (next_succ_rad + curr_succ_rad)/2
          elif curr_succ_rad >= 2*np.pi - next_succ_rad:
            downlink_ant_alignment_rad = (next_succ_rad + curr_succ_rad)/2 - np.pi
          else:
            downlink_ant_alignment_rad = (next_succ_rad + curr_succ_rad)/2 + np.pi
          if antenna_type_idx == 0:
            downlink_ant_beams = [0]
          elif antenna_type_idx == 1:
            downlink_ant_beams = [0,1]
          else:
            downlink_ant_beams = [0,1,2]
    return downlink_ant_alignment_rad, downlink_ant_beams

  """
  Defines equality of Network objects.
  """
  def __eq__(self, other):
    if isinstance(other, Network):
        return self.MH_succs_2_partition == other.MH_succs_2_partition and self.freq_assignment == other.freq_assignment and self.imp_arb == other.imp_arb
    else:
      return False
  
  """
  Prints a Network object.
  """
  def __repr__(self):
    string = "Full Solution :\n\t"
    string += str(self.imp_arb)+"\n\t"
    string += "Master Hub Successors Partition : "+str(self.MH_succs_2_partition)+"\n\t"
    string += "Uplink Antenna Alignments : "+str(self.uplink_ant_alignments_rad)+"\n\t"
    string += "Downlink Antenna Alignments : "+str(self.downlink_ant_alignments_rad)+"\n\t"
    string += "Downlink Activated Antenna Beams : "+str(self.downlink_ant_beams)
    if self.has_freq_assignment:
      string += "\n\tFrequency Assignment : "+str(self.freq_assignment)
    if self.has_obj_value:
      string += "\n\tDirect Throughputs : \n"+str(self.direct_TPs)
      if self.is_mixed:
        #string += "\n\tMesh Routes : "+str(self.A_mesh_routes)
        string += "\n\tMesh TP Denominators (All-->MH) : \n"+str(self.B_mesh_TP_denoms)
        string += "\n\tMesh TP Denominators (All<->All) : \n"+str(self.C_mesh_TP_denoms)
      string += "\n\tWorst Throughput (Any-->Any) : "+str(self.A_obj_min)
      string += "\n\tObjective Value (Any-->Any) : "+str(self.A_obj_value)
      string += "\n\tLimiting Connection Path (Any-->Any) : "+str(self.A_lim_links)
      string += "\n\tWorst Throughput (All-->MH) : "+str(self.B_obj_min)
      string += "\n\tObjective Value (All-->MH) : "+str(self.B_obj_value)
      string += "\n\tLimiting Connection Path (All-->MH)  : "+str(self.B_lim_links)
      string += "\n\tWorst Throughput (All<->All) : "+str(self.C_obj_min)
      string += "\n\tObjective Value (All<->All) : "+str(self.C_obj_value)
      string += "\n\tLimiting Connection Path (All<->All) : "+str(self.C_lim_links)
      string += "\n\tObjective Value (Aggregated) : "+str(self.obj_value)
    return string

  """
  Plots a Network as a set of nodes (master hub in green) connected by arrows (solid lines: tree edges, dotted lines: cluster membership) with transparent antenna beams with
  colors based on frequency indices (see FreqMHz.color_from_index(...) for details) where
    'ax' (optional) is the matplotlib axes object in which to plot;
    'pretitle' (optional) is the title string of the plot (ending in " F=obj_value" if the Network has an obj_value).
  """
  def show(self, ax=-1, pretitle=""):
    extra_border = 1
    line_width = 0.8
    has_axis = ax != -1
    if not has_axis:
      fig = plt.figure(1)
      ax = fig.add_axes([0.075,0.075,0.85,0.85])
    ax.set_aspect('equal')
    prob_params = self.imp_arb.prob_params
    multibeam = prob_params.multibeam
    nb_nodes = self.imp_arb.prob_instance.nb_nodes
    coordinates_km = self.imp_arb.prob_instance.coordinates_km
    adj_matrix = self.imp_arb.adj_matrix
    MH = self.imp_arb.MH
    has_freq_assignment = self.has_freq_assignment
    uplink_ant_alignments_rad = self.uplink_ant_alignments_rad
    uplink_ant_nb_beams = self.uplink_ant_nb_beams
    downlink_ant_alignments_rad = self.downlink_ant_alignments_rad
    downlink_ant_beams = self.downlink_ant_beams
    has_obj_value = self.has_obj_value
    if has_obj_value:
      direct_TPs = self.direct_TPs
      A_lim_links = self.A_lim_links
      B_lim_links = self.B_lim_links
      C_lim_links = self.C_lim_links
      obj_value = self.obj_value
    xlim0 = -extra_border
    xlim1 = np.max(coordinates_km[:,0])+extra_border
    ylim0 = -extra_border
    ylim1 = np.max(coordinates_km[:,1])+extra_border
    ax.set_xlim(xlim0,xlim1)
    ax.set_ylim(ylim0,ylim1)
    max_size = max(xlim1-xlim0,ylim1-ylim0)
    if has_obj_value:
      f_obj_string = "F_obj = "+str(round(100*obj_value)/100)
    else:
      f_obj_string = ""
    ax.set_title(pretitle+f_obj_string)
    # ARROWS
    for w1 in range(nb_nodes):
      for w2 in range(nb_nodes):
        if adj_matrix[w1,w2] > 0:
          dx = coordinates_km[w2,0] - coordinates_km[w1,0]
          dy = coordinates_km[w2,1] - coordinates_km[w1,1]
          radius = sqrt(pow(dx,2) + pow(dy,2))
          ratio_beg = (7/400*max_size)/radius
          #ratio_end = (radius - 7/200*max_size)/radius
          ratio_end = (radius - 7/400*max_size)/radius
          if adj_matrix[w1,w2] == 1:
            if has_freq_assignment:
              if w1 == self.imp_arb.MH:
                if w2 in self.MH_succs_2_partition[0]:
                  arc_freq_idx = self.freq_assignment[w1][0]
                else:
                  arc_freq_idx = self.freq_assignment[w1][1]
              else:
                arc_freq_idx = self.freq_assignment[w1]
            else:
              arc_freq_idx = -1
            arc_color = FreqMHz.color_from_index(arc_freq_idx)
            if has_obj_value and ((w1,w2) in A_lim_links or (w1,w2) in B_lim_links or (w1,w2) in C_lim_links):
              lw = 2*line_width
            else:
              lw = line_width
            if w1 == MH:
              for i in range(2):
                succs = self.MH_succs_2_partition[i]
                if w2 in succs:
                  dashed_arrow = len(succs) > 1
            else:
              succs = self.imp_arb.succ_lists[w1]
              dashed_arrow = len(succs) > 1
            if dashed_arrow:
              ls = (0,(3*line_width,6.25*line_width))
              lw *= 1.25
              #if lw == line_width:
              #  lw *= 1.5
              #else:
              #  lw *= 1.25
            else:
              ls = "-"
            #ax.arrow(coordinates_km[w1,0]+ratio_beg*dx, coordinates_km[w1,1]+ratio_beg*dy, ratio_end*dx, ratio_end*dy,length_includes_head=True,head_width=0.25,lw=lw,ls=ls,color=arc_color)
            ax.plot([coordinates_km[w1,0]+ratio_beg*dx,coordinates_km[w1,0]+ratio_end*dx],[coordinates_km[w1,1]+ratio_beg*dy,coordinates_km[w1,1]+ratio_end*dy],lw=lw,ls=ls,color=arc_color)
          elif w1 < w2 and (not has_obj_value or self.C_mesh_TP_denoms[w1,w2] > 0):
            if has_freq_assignment:
              mesh_idx = int(-1*adj_matrix[w1,w2])
              arc_freq_idx = self.freq_assignment[mesh_idx]
            else:
              arc_freq_idx = -1
            arc_color = FreqMHz.color_from_index(arc_freq_idx)
            if has_obj_value and ((w1,w2) in A_lim_links or (w1,w2) in B_lim_links or (w1,w2) in C_lim_links):
              lw = 3*line_width
            else:
              lw = 1.5*line_width
            ax.plot([coordinates_km[w1,0]+ratio_beg*dx,coordinates_km[w1,0]+ratio_end*dx],[coordinates_km[w1,1]+ratio_beg*dy,coordinates_km[w1,1]+ratio_end*dy],ls=":",lw=lw,color=arc_color)
          # DIRECT TPs
          if has_obj_value and (adj_matrix[w1,w2] == 1 or (w1 < w2 and self.C_mesh_TP_denoms[w1,w2] > 0)):
            direct_TP = direct_TPs[w1,w2]
            if (w1,w2) in A_lim_links:
              clr = 'olive'
            elif (w1,w2) in B_lim_links:
              clr = 'darkolivegreen'
            elif (w1,w2) in C_lim_links:
              clr = 'k'
            else:
              clr = 'gray'
            if clr != 'gray':
              ax.text(coordinates_km[w1,0]+0.5*dx,coordinates_km[w1,1]+0.5*dy,str(round(10*direct_TP)/10), color=clr)
            else:
              ax.text(coordinates_km[w1,0]+0.5*dx,coordinates_km[w1,1]+0.5*dy,str(round(10*direct_TP)/10), fontsize=9, color=clr)
    # NODES
    ax.scatter([coordinates_km[MH,0]],[coordinates_km[MH,1]], s=30, c='green')
    non_MH_coordinates = np.concatenate((coordinates_km[:MH,:],coordinates_km[MH+1:,:]))
    ax.scatter(non_MH_coordinates[:,0].tolist(),non_MH_coordinates[:,1].tolist(), s=30, c='k')
    # NODE LABELS
    for exp_node in range(nb_nodes):
      ax.text(coordinates_km[exp_node,0],coordinates_km[exp_node,1] - 5/100*max_size,str(exp_node),fontsize=8,color='k',ha='center')
    # BEAMS
    for exp_node in range(nb_nodes):
      if exp_node in self.imp_arb.preds or exp_node == self.imp_arb.MH:
        if exp_node in uplink_ant_alignments_rad:
          #uplink
          w1 = self.imp_arb.preds[exp_node]
          if has_freq_assignment:
            if w1 == self.imp_arb.MH:
              if exp_node in self.MH_succs_2_partition[0]:
                ant_freq_idx = self.freq_assignment[w1][0]
              else:
                ant_freq_idx = self.freq_assignment[w1][1]
            else:
              ant_freq_idx = self.freq_assignment[w1]
          else:
            ant_freq_idx = -1
          ant_color = FreqMHz.color_from_index(ant_freq_idx)
          if multibeam:
            half_beam_width_deg = 0.5*prob_params.get_3dB_beam_width_deg(ant_freq_idx)
          else:
            half_beam_width_deg = 0.5*prob_params.get_3dB_beam_width_deg(ant_freq_idx)[0]
          if uplink_ant_nb_beams[exp_node] == 0:
            wedge = mpatches.Wedge([coordinates_km[exp_node,0], coordinates_km[exp_node,1]], 0.075*max_size, 0, 359.99, ec="none", color = ant_color, alpha=0.3)
          else:
            mid_angle = (180/np.pi) * uplink_ant_alignments_rad[exp_node]
            wedge = mpatches.Wedge([coordinates_km[exp_node,0], coordinates_km[exp_node,1]], 0.10*max_size, mid_angle-half_beam_width_deg, mid_angle+half_beam_width_deg, ec="none", color = ant_color, alpha=0.3)
          ax.add_patch(wedge)
        if exp_node in downlink_ant_alignments_rad:
          #downlink
          if exp_node == self.imp_arb.MH:
            for i in range(2):
              ant_alignment = downlink_ant_alignments_rad[exp_node][i]
              if ant_alignment >= 0:
                if has_freq_assignment:
                  ant_freq_idx = self.freq_assignment[exp_node][i]
                else:
                  ant_freq_idx = -1
                ant_color = FreqMHz.color_from_index(ant_freq_idx)
                activ_beams = downlink_ant_beams[exp_node][i]
                if len(activ_beams) == 0:
                  wedge = mpatches.Wedge([coordinates_km[exp_node,0], coordinates_km[exp_node,1]], 0.075*max_size, 0, 359.99, ec="none", color = ant_color, alpha=0.3)
                  ax.add_patch(wedge)
                else:
                  if multibeam:
                    half_beam_width_deg = 0.5*prob_params.get_3dB_beam_width_deg(ant_freq_idx)
                    for activ_beam in activ_beams:
                      mid_angle = (180/np.pi) * ant_alignment + 15*activ_beam
                      wedge = mpatches.Wedge([coordinates_km[exp_node,0], coordinates_km[exp_node,1]], 0.10*max_size, mid_angle-half_beam_width_deg, mid_angle+half_beam_width_deg, ec="none", color = ant_color, alpha=0.3)
                      ax.add_patch(wedge)
                  else:
                    half_beam_width_deg = 0.5*prob_params.get_3dB_beam_width_deg(ant_freq_idx)[len(activ_beams)-1]
                    mid_angle = (180/np.pi) * ant_alignment
                    wedge = mpatches.Wedge([coordinates_km[exp_node,0], coordinates_km[exp_node,1]], 0.10*max_size, mid_angle-half_beam_width_deg, mid_angle+half_beam_width_deg, ec="none", color = ant_color, alpha=0.3)
                    ax.add_patch(wedge)
          else:
            if has_freq_assignment:
              ant_freq_idx = self.freq_assignment[exp_node]
            else:
              ant_freq_idx = -1
            ant_color = FreqMHz.color_from_index(ant_freq_idx)
            activ_beams = downlink_ant_beams[exp_node]
            if len(activ_beams) == 0:
              wedge = mpatches.Wedge([coordinates_km[exp_node,0], coordinates_km[exp_node,1]], 0.075*max_size, 0, 359.99, ec="none", color = ant_color, alpha=0.3)
              ax.add_patch(wedge)
            else:
              if multibeam:
                half_beam_width_deg = 0.5*prob_params.get_3dB_beam_width_deg(ant_freq_idx)
                for activ_beam in activ_beams:
                  mid_angle = (180/np.pi) * downlink_ant_alignments_rad[exp_node] + 15*activ_beam
                  wedge = mpatches.Wedge([coordinates_km[exp_node,0], coordinates_km[exp_node,1]], 0.10*max_size, mid_angle-half_beam_width_deg, mid_angle+half_beam_width_deg, ec="none", color = ant_color, alpha=0.3)
                  ax.add_patch(wedge)
              else:
                half_beam_width_deg = 0.5*prob_params.get_3dB_beam_width_deg(ant_freq_idx)[len(activ_beams)-1]
                mid_angle = (180/np.pi) * downlink_ant_alignments_rad[exp_node]
                wedge = mpatches.Wedge([coordinates_km[exp_node,0], coordinates_km[exp_node,1]], 0.10*max_size, mid_angle-half_beam_width_deg, mid_angle+half_beam_width_deg, ec="none", color = ant_color, alpha=0.3)
                ax.add_patch(wedge)
    if not has_axis:
      fig.canvas.draw()
      plt.show()

  """
  Returns the sizes of the Network's mesh clusters as a list of integers. If there are no mesh clusters, the returned list is empty.
  """
  def get_mesh_sizes(self):
    return self.imp_arb.get_mesh_sizes()

  """
  Returns the objective/pseudo-objective value of the ImpArborescence where
    'obj_func' is the identifier of the desired objective or pseudo-objective
      "full" --> real objective or accurate pseudo-objective (accurate evaluation of network greedy frequency assignment);
      "chan_avg" --> inaccurate but fast pseudo-objective;
      "chan_max" --> fast upper bound pseudo-objective (actual upper bound if no mesh cluster);
      "avg" --> very inaccurate but very fast pseudo-objective;
    'return_SINRs_after_direct_TPs' (optional) --> is whether to stop the evaluation early and only return the SINRs of the direct connections (True) or not (False);
    'return_arcs_and_effective_TPs' (optional) --> is whether to also return the effective throughput of every edge for every scenario (True) or not (False);
    'has_old_avg' (optional) is whether it is given the intermediary data for the "avg" pseudo-objective evaluation of a previous ImpTree neighbor (True) or not (False);
    'has_old_chan_avg' (optional) is whether it is given the intermediary data for the "chan_avg" pseudo-objective evaluation of a previous ImpTree neighbor (True) or not (False);
    'has_old_chan_max' (optional) is whether it is given the intermediary data for the "chan_max" pseudo-objective evaluation of a previous ImpTree neighbor (True) or not (False);
    'old_imp_arb_adj_matrix' (optional) is the adjacency matrix of the best ImpArborescence of a previous ImpTree neighbor;
    'old_MH_succs_2_partition' (optional) is the partition of the master hub's successors in 2 sets of the best pseudo-network of a previous ImpTree neighbor (see Network for details);
    'old_chan_assignment' (optional) is a dictionary that describes the channels of the downlink connections of the predecessors of the best pseudo-network of a previous neighbor (see
      Network for details);
    'old_direct_simp_TPs' (optional) is the direct throughputs of the best pseudo-network of a previous ImpTree neighbor (see Network for details);
    'old_mesh_members' (optional) is the 'mesh_members' of the best ImpArborescence of a previous ImpTree neighbor;
    'old_preds' (optional) is the 'preds' of the best ImpArborescence of a previous ImpTree neighbor;
    'old_mesh_routes' (optional) is a dictionary that describes the routes used between every pair of members for each cluster of the best pseudo-network of a previous neighbor (see
      Network for details).
  """
  def get_obj_value(self, obj_func="full", return_SINRs_after_direct_TPs=False, return_arcs_and_effective_TPs=False, has_old_avg=False, has_old_chan_avg=False, has_old_chan_max=False, old_imp_arb_adj_matrix=-1, old_MH_succs_2_partition=-1, old_chan_assignment=-1, old_direct_simp_TPs=-1, old_mesh_members=-1, old_preds=-1, old_mesh_routes=-1):
    if not self.has_obj_value:
      if len(self.MH_succs_2_partition[0]) == 0:
        self.obj_value = -1
        if return_SINRs_after_direct_TPs:
          return np.zeros((self.imp_arb.prob_instance.nb_nodes,self.imp_arb.prob_instance.nb_nodes))
        arcs_and_effective_TPs = ([],[],[],[])
      else:
        same_old_preds = []
        same_old_meshes = []
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          if obj_func == "full" or obj_func == "1_freq_per_channel":
            all_lin_simp_SINRs = self.get_all_linear_simp_SINRs()
            SINRs_dB = self.add_interference_to_simps(all_lin_simp_SINRs)
          else:
            if (has_old_avg and obj_func == "avg") or (has_old_chan_avg and obj_func == "chan_avg") or (has_old_chan_max and obj_func == "chan_max"):
              same_old_preds, same_old_meshes = self.get_same_old_preds_and_meshes(old_imp_arb_adj_matrix, old_MH_succs_2_partition, old_chan_assignment, old_mesh_members, old_preds)
            SINRs_dB = self.get_simp_SINRs_with_fade_margin_and_noise_power(same_old_preds, same_old_meshes, obj_func=obj_func)
        self.get_direct_TPs(SINRs_dB, same_old_preds, obj_func=="full", same_old_meshes, old_direct_simp_TPs)#, old_imp_arb_adj_matrix, old_MH_succs_2_partition, old_chan_assignment, old_mesh_members, old_preds)
        if return_SINRs_after_direct_TPs:
          return SINRs_dB
        arcs_and_effective_TPs = self.get_effective_TPs(obj_func=="full", same_old_meshes, old_mesh_routes, SINRs_dB)
      if return_arcs_and_effective_TPs:
        return arcs_and_effective_TPs
      # print(arcs_and_direct_TPs[0])
      # print(arcs_and_direct_TPs[-3])
      # print(arcs_and_direct_TPs[-1])
    return self.obj_value
  
  """
  Returns a list of explicit nodes whose direct outgoing (downlink) connections have not changed and a list of mesh clusters that have not changed since the previous ImpTree neighbor's
  pseudo-network where
    'old_imp_arb_adj_matrix' is the adjacency matrix of the best ImpArborescence of a previous ImpTree neighbor;
    'old_MH_succs_2_partition' is the partition of the master hub's successors in 2 sets of the best pseudo-network of a previous ImpTree neighbor (see Network for details);
    'old_chan_assignment' is a dictionary that describes the channels of the downlink connections of the predecessors of the best pseudo-network of a previous neighbor (see Network for
      details);
    'old_mesh_members' is the 'mesh_members' of the best ImpArborescence of a previous ImpTree neighbor;
    'old_preds' is the 'preds' of the best ImpArborescence of a previous ImpTree neighbor.
  """
  def get_same_old_preds_and_meshes(self, old_imp_arb_adj_matrix, old_MH_succs_2_partition, old_chan_assignment, old_mesh_members, old_preds):
    same_old_preds = []
    same_old_meshes = []
    for v in range(self.imp_arb.prob_instance.nb_nodes):
      if v == self.imp_arb.MH:
        if (old_preds != -1 and not v in old_preds) or (old_preds == -1 and np.min(old_imp_arb_adj_matrix[v,:]) == 0):
          for i in range(2):
            if len(self.MH_succs_2_partition[i]) > 0:
              if self.MH_succs_2_partition[i] == old_MH_succs_2_partition[0] and old_MH_succs_2_partition[0][0] >= 0:
                if not self.has_freq_assignment or self.freq_assignment[v][i] == old_chan_assignment[v][0]:
                  same_old_preds.append((v, i))
              elif self.MH_succs_2_partition[i] == old_MH_succs_2_partition[1] and old_MH_succs_2_partition[1][0] >= 0:
                if not self.has_freq_assignment or self.freq_assignment[v][i] == old_chan_assignment[v][1]:
                  same_old_preds.append((v, i))
      else:
        if ((old_preds != -1 and v in old_preds) or (old_preds == -1 and np.min(old_imp_arb_adj_matrix[v,:]) == -1)):
          succ_vector = self.imp_arb.adj_matrix[v,:] == 1
          if np.sum(succ_vector) > 0 and (succ_vector == (old_imp_arb_adj_matrix[v,:] == 1)).all():
            if not self.has_freq_assignment or (v in self.freq_assignment and v in old_chan_assignment and self.freq_assignment[v] == old_chan_assignment[v]):
              same_old_preds.append(v)
    if self.is_mixed:
      mesh_members = self.imp_arb.mesh_members
      for mesh_idx in mesh_members.keys():
        if mesh_idx in old_mesh_members and set(mesh_members[mesh_idx]) == set(old_mesh_members[mesh_idx]):
          if mesh_idx in self.imp_arb.preds:
            if mesh_idx in old_preds and self.imp_arb.preds[mesh_idx] == old_preds[mesh_idx]:
              if not self.has_freq_assignment or (mesh_idx in old_chan_assignment and self.freq_assignment[mesh_idx] == old_chan_assignment[mesh_idx]):
                same_old_meshes.append(mesh_idx)
          else:
            if not mesh_idx in old_preds:
              if not self.has_freq_assignment or (mesh_idx in old_chan_assignment and self.freq_assignment[mesh_idx] == old_chan_assignment[mesh_idx]):
                same_old_meshes.append(mesh_idx)
    return same_old_preds, same_old_meshes
  
  """
  Returns a matrix containing the SINRs in dB of the network for the "avg", "chan_avg" and "chan_max" pseudo-objectives where
    'same_old_preds' is a list of explicit nodes whose direct outgoing (downlink) connections have not changed since the previous ImpTree neighbor's pseudo-network;
    'same_old_meshes' is a list of mesh clusters that have not changed since the previous ImpTree neighbor's pseudo-network;
    'obj_func' is the identifier of the desired pseudo-objective
      "chan_avg" --> inaccurate but fast pseudo-objective;
      "chan_max" --> fast upper bound pseudo-objective (actual upper bound if no mesh cluster);
      "avg" --> very inaccurate but very fast pseudo-objective.
  """
  def get_simp_SINRs_with_fade_margin_and_noise_power(self, same_old_preds, same_old_meshes, obj_func="chan_avg"):
    nb_nodes = self.imp_arb.prob_instance.nb_nodes
    prob_params = self.imp_arb.prob_params
    angles_from_to_rad = self.imp_arb.prob_instance.angles_from_to_rad
    pattern_exponents = prob_params.get_pattern_exponents()
    if obj_func == "chan_max":
      pattern_exponents = [max(pattern_exponents[2:4]), max(pattern_exponents[0:2]), pattern_exponents[-1]]
    if obj_func != "avg":
      freq_assignment = self.freq_assignment
    noise_power_dBm = prob_params.noise_powers_dBm[0]
    SINRs_dB = np.zeros((nb_nodes,nb_nodes))
    path_losses_dB = self.imp_arb.prob_instance.path_losses_dB
    fade_margins_dB = self.imp_arb.prob_instance.fade_margins_dB
    if obj_func == "chan_max":
      path_losses_dB = np.concatenate((np.reshape(np.min(path_losses_dB[:,:,2:4],axis=2),(nb_nodes,nb_nodes,1)),np.reshape(np.min(path_losses_dB[:,:,0:2],axis=2),(nb_nodes,nb_nodes,1)),np.reshape(path_losses_dB[:,:,-1],(nb_nodes,nb_nodes,1))), axis=2)
      fade_margins_dB = np.concatenate((np.reshape(np.min(fade_margins_dB[:,:,2:4],axis=2),(nb_nodes,nb_nodes,1)),np.reshape(np.min(fade_margins_dB[:,:,0:2],axis=2),(nb_nodes,nb_nodes,1)),np.reshape(fade_margins_dB[:,:,-1],(nb_nodes,nb_nodes,1))), axis=2)
    nodes_to_visit = [self.imp_arb.MH]
    while len(nodes_to_visit) > 0:
      new_node = nodes_to_visit.pop(0)
      if new_node in self.imp_arb.succ_lists:
        succs = self.imp_arb.succ_lists[new_node]
        nodes_to_visit += succs
        if new_node >= 0:
          if new_node == self.imp_arb.MH:
            for i in range(2):
              if not (new_node,i) in same_old_preds:
                succs = self.MH_succs_2_partition[i]
                if len(succs) > 0:
                  if obj_func == "avg":
                    freq_idx = -1
                    max_gain_freq_idx = freq_idx
                  else:
                    freq_idx = freq_assignment[new_node][i]
                    if obj_func == "chan_max":
                      max_gain_freq_idx = 2*(-2 - freq_idx)+1
                    else: # "chan_avg"
                      max_gain_freq_idx = freq_idx
                  for succ in succs:
                    if succ >= 0:
                      sig_strength_new_node_succ = 10*np.log10(prob_params.get_dl_lin_signal_strength(angles_from_to_rad[new_node,succ], pattern_exponents, self.downlink_ant_beams[new_node][i], self.downlink_ant_alignments_rad[new_node][i], freq_idx, gain_freq_idx=max_gain_freq_idx))
                      sig_strength_succ_new_node = prob_params.get_ul_dB_signal_strength(angles_from_to_rad[succ,new_node], pattern_exponents, self.uplink_ant_nb_beams[succ], self.uplink_ant_alignments_rad[succ], freq_idx, gain_freq_idx=max_gain_freq_idx)
                      SINRs_dB[new_node, succ] = 30 + sig_strength_new_node_succ + sig_strength_succ_new_node - path_losses_dB[succ, new_node, freq_idx] - fade_margins_dB[succ, new_node, freq_idx] - noise_power_dBm
          else:
            if len(succs) > 0 and not new_node in same_old_preds:
              if obj_func == "avg":
                freq_idx = -1
                max_gain_freq_idx = freq_idx
              else:
                freq_idx = freq_assignment[new_node]
                if obj_func == "chan_max":
                  max_gain_freq_idx = 2*(-2 - freq_idx)+1
                else: # "chan_avg"
                  max_gain_freq_idx = freq_idx
              for succ in succs:
                if succ >= 0:
                  sig_strength_new_node_succ = 10*np.log10(prob_params.get_dl_lin_signal_strength(angles_from_to_rad[new_node,succ], pattern_exponents, self.downlink_ant_beams[new_node], self.downlink_ant_alignments_rad[new_node], freq_idx, gain_freq_idx=max_gain_freq_idx))
                  sig_strength_succ_new_node = prob_params.get_ul_dB_signal_strength(angles_from_to_rad[succ,new_node], pattern_exponents, self.uplink_ant_nb_beams[succ], self.uplink_ant_alignments_rad[succ], freq_idx, gain_freq_idx=max_gain_freq_idx)
                  SINRs_dB[new_node, succ] = 30 + sig_strength_new_node_succ + sig_strength_succ_new_node - path_losses_dB[succ, new_node, freq_idx] - fade_margins_dB[succ, new_node, freq_idx] - noise_power_dBm
    if self.is_mixed:
      mesh_members = self.imp_arb.mesh_members
      preds = self.imp_arb.preds
      for mesh_idx in mesh_members.keys():
        if not mesh_idx in same_old_meshes:
          if mesh_idx in preds:
            mesh_pred = preds[mesh_idx]
          else:
            mesh_pred = -1
          if obj_func == "avg":
            freq_idx = -1
            max_gain_freq_idx = freq_idx
          else:
            if mesh_idx == self.imp_arb.MH:
              freq_idx = freq_assignment[mesh_idx][0]
            else:
              freq_idx = freq_assignment[mesh_idx]
            if obj_func == "chan_max":
              max_gain_freq_idx = 2*(-2 - freq_idx)+1
            else: # "chan_avg"
              max_gain_freq_idx = freq_idx
          members = mesh_members[mesh_idx]
          for member_idx1 in range(len(members)):
            member1 = members[member_idx1]
            for member_idx2 in range(member_idx1+1,len(members)):
              member2 = members[member_idx2]
              if member1 == mesh_pred:
                if member1 == self.imp_arb.MH:
                  if self.MH_succs_2_partition[0][0] == mesh_idx:
                    sig_strength_1_2 = 10*np.log10(prob_params.get_dl_lin_signal_strength(angles_from_to_rad[member1,member2], pattern_exponents, self.downlink_ant_beams[member1][0], self.downlink_ant_alignments_rad[member1][0], freq_idx, gain_freq_idx=max_gain_freq_idx))
                  else:
                    sig_strength_1_2 = 10*np.log10(prob_params.get_dl_lin_signal_strength(angles_from_to_rad[member1,member2], pattern_exponents, self.downlink_ant_beams[member1][1], self.downlink_ant_alignments_rad[member1][1], freq_idx, gain_freq_idx=max_gain_freq_idx))
                else:
                  sig_strength_1_2 = 10*np.log10(prob_params.get_dl_lin_signal_strength(angles_from_to_rad[member1,member2], pattern_exponents, self.downlink_ant_beams[member1], self.downlink_ant_alignments_rad[member1], freq_idx, gain_freq_idx=max_gain_freq_idx))
              else:
                sig_strength_1_2 = prob_params.get_ul_dB_signal_strength(angles_from_to_rad[member1,member2], pattern_exponents, self.uplink_ant_nb_beams[member1], self.uplink_ant_alignments_rad[member1], freq_idx, gain_freq_idx=max_gain_freq_idx)
              if member2 == mesh_pred:
                if member2 == self.imp_arb.MH:
                  if self.MH_succs_2_partition[0][0] == mesh_idx:
                    sig_strength_2_1 = 10*np.log10(prob_params.get_dl_lin_signal_strength(angles_from_to_rad[member2,member1], pattern_exponents, self.downlink_ant_beams[member2][0], self.downlink_ant_alignments_rad[member2][0], freq_idx, gain_freq_idx=max_gain_freq_idx))
                  else:
                    sig_strength_2_1 = 10*np.log10(prob_params.get_dl_lin_signal_strength(angles_from_to_rad[member2,member1], pattern_exponents, self.downlink_ant_beams[member2][1], self.downlink_ant_alignments_rad[member2][1], freq_idx, gain_freq_idx=max_gain_freq_idx))
                else:
                  sig_strength_2_1 = 10*np.log10(prob_params.get_dl_lin_signal_strength(angles_from_to_rad[member2,member1], pattern_exponents, self.downlink_ant_beams[member2], self.downlink_ant_alignments_rad[member2], freq_idx, gain_freq_idx=max_gain_freq_idx))
              else:
                sig_strength_2_1 = prob_params.get_ul_dB_signal_strength(angles_from_to_rad[member2,member1], pattern_exponents, self.uplink_ant_nb_beams[member2], self.uplink_ant_alignments_rad[member2], freq_idx, gain_freq_idx=max_gain_freq_idx)
              SINR_dB = 30 + sig_strength_1_2 + sig_strength_2_1 - path_losses_dB[member1, member2, freq_idx] - fade_margins_dB[member1, member2, freq_idx] - noise_power_dBm
              SINRs_dB[member1, member2] = SINR_dB
              SINRs_dB[member2, member1] = SINR_dB
    return SINRs_dB # (nb_exp_nodes,nb_exp_nodes) SINRs only at pred,succ

  """
  Returns a tensor containing the SINRs in linear of the network and of the interfering nodes for the "full" objective and the "greedy" pseudo-objective.
  """
  def get_all_linear_simp_SINRs(self):
    nb_nodes = self.imp_arb.prob_instance.nb_nodes
    adj_matrix = self.imp_arb.adj_matrix
    is_mixed = self.is_mixed
    lin_SINRs = np.zeros((nb_nodes,nb_nodes,8))
    prob_params = self.imp_arb.prob_params
    angles_from_to_rad = self.imp_arb.prob_instance.angles_from_to_rad
    pattern_exponents = prob_params.get_pattern_exponents()
    path_losses_dB = self.imp_arb.prob_instance.path_losses_dB
    fade_margins_dB = self.imp_arb.prob_instance.fade_margins_dB
    nb_freqs_per_channel = prob_params.nb_freqs_per_channel
    if nb_freqs_per_channel > 1 and self.freq_assignment[self.imp_arb.MH][0] < 0:
      nb_freqs_per_channel = 1
    if nb_freqs_per_channel == 1:
      lin_SINRs = np.zeros((nb_nodes,nb_nodes,4))
      path_losses_dB = path_losses_dB[:,:,-3:]
    else:
      lin_SINRs = np.zeros((nb_nodes,nb_nodes,5))
      path_losses_dB = path_losses_dB[:,:,:4]
    for v1 in range(nb_nodes):
      if v1 == self.imp_arb.MH:
        MH0_freq_1,MH1_freq_1 = self.get_ul_dl_freqs(v1)
        for v2 in range(v1+1,nb_nodes):
          if v2 in self.imp_arb.preds:
            if nb_freqs_per_channel == 1:
              sig_strengths_1_2 = np.array([0,0,0])
              sig_strengths_2_1 = np.array([0,0,0])
            else:
              sig_strengths_1_2 = np.array([0,0,0,0])
              sig_strengths_2_1 = np.array([0,0,0,0])
            ul_freq_2,dl_freq_2 = self.get_ul_dl_freqs(v2)
            if ul_freq_2 != -1:
              if MH0_freq_1 == ul_freq_2:
                sig_strengths_1_2[MH0_freq_1] += prob_params.get_dl_lin_signal_strength(angles_from_to_rad[v1,v2], pattern_exponents, self.downlink_ant_beams[v1][0], self.downlink_ant_alignments_rad[v1][0], MH0_freq_1)
                sig_strengths_2_1[ul_freq_2] += pow(10,prob_params.get_ul_dB_signal_strength(angles_from_to_rad[v2,v1], pattern_exponents, self.uplink_ant_nb_beams[v2], self.uplink_ant_alignments_rad[v2], ul_freq_2)/10)
              if MH1_freq_1 == ul_freq_2:
                sig_strengths_1_2[MH1_freq_1] += prob_params.get_dl_lin_signal_strength(angles_from_to_rad[v1,v2], pattern_exponents, self.downlink_ant_beams[v1][1], self.downlink_ant_alignments_rad[v1][1], MH1_freq_1)
                sig_strengths_2_1[ul_freq_2] += pow(10,prob_params.get_ul_dB_signal_strength(angles_from_to_rad[v2,v1], pattern_exponents, self.uplink_ant_nb_beams[v2], self.uplink_ant_alignments_rad[v2], ul_freq_2)/10)
            if dl_freq_2 != -1:
              if MH0_freq_1 == dl_freq_2:
                sig_strengths_1_2[MH0_freq_1] += prob_params.get_dl_lin_signal_strength(angles_from_to_rad[v1,v2], pattern_exponents, self.downlink_ant_beams[v1][0], self.downlink_ant_alignments_rad[v1][0], MH0_freq_1)
                sig_strengths_2_1[dl_freq_2] += prob_params.get_dl_lin_signal_strength(angles_from_to_rad[v2,v1], pattern_exponents, self.downlink_ant_beams[v2], self.downlink_ant_alignments_rad[v2], dl_freq_2)
              if MH1_freq_1 == dl_freq_2:
                sig_strengths_1_2[MH1_freq_1] += prob_params.get_dl_lin_signal_strength(angles_from_to_rad[v1,v2], pattern_exponents, self.downlink_ant_beams[v1][1], self.downlink_ant_alignments_rad[v1][1], MH1_freq_1)
                sig_strengths_2_1[dl_freq_2] += prob_params.get_dl_lin_signal_strength(angles_from_to_rad[v2,v1], pattern_exponents, self.downlink_ant_beams[v2], self.downlink_ant_alignments_rad[v2], dl_freq_2)
            sig_strengths_1_2 = 10*np.log10(sig_strengths_1_2)
            sig_strengths_2_1 = 10*np.log10(sig_strengths_2_1)
            lin_SINRs_1_2 = np.power(10,(30 + sig_strengths_1_2 + sig_strengths_2_1 - path_losses_dB[v1,v2,:])/10)
            lin_SINRs[v1,v2,1:] = lin_SINRs_1_2
            lin_SINRs[v2,v1,1:] = lin_SINRs_1_2
            if v1 == self.imp_arb.preds[v2]:
              lin_SINRs[v1,v2,0] = 10*np.log10(lin_SINRs_1_2[ul_freq_2]) - fade_margins_dB[v1,v2,ul_freq_2]
            elif is_mixed and adj_matrix[v1,v2] >= 2:
              lin_SINRs[v1,v2,0] = 10*np.log10(lin_SINRs_1_2[ul_freq_2]) - fade_margins_dB[v1,v2,ul_freq_2]
              lin_SINRs[v2,v1,0] = lin_SINRs[v1,v2,0]
      else:
        if v1 in self.imp_arb.preds:
          ul_freq_1,dl_freq_1 = self.get_ul_dl_freqs(v1)
          for v2 in range(v1+1,nb_nodes):
            if v2 in self.imp_arb.preds or v2 == self.imp_arb.MH:
              if nb_freqs_per_channel == 1:
                sig_strengths_1_2 = np.array([0,0,0])
                sig_strengths_2_1 = np.array([0,0,0])
              else:
                sig_strengths_1_2 = np.array([0,0,0,0])
                sig_strengths_2_1 = np.array([0,0,0,0])
              if v2 == self.imp_arb.MH:
                MH0_freq_2,MH1_freq_2 = self.get_ul_dl_freqs(v2)
                if MH0_freq_2 != -1:
                  if ul_freq_1 == MH0_freq_2:
                    sig_strengths_1_2[ul_freq_1] += pow(10,prob_params.get_ul_dB_signal_strength(angles_from_to_rad[v1,v2], pattern_exponents, self.uplink_ant_nb_beams[v1], self.uplink_ant_alignments_rad[v1], ul_freq_1)/10)
                    sig_strengths_2_1[MH0_freq_2] += prob_params.get_dl_lin_signal_strength(angles_from_to_rad[v2,v1], pattern_exponents, self.downlink_ant_beams[v2][0], self.downlink_ant_alignments_rad[v2][0], MH0_freq_2)
                  if dl_freq_1 == MH0_freq_2:
                    sig_strengths_1_2[dl_freq_1] += prob_params.get_dl_lin_signal_strength(angles_from_to_rad[v1,v2], pattern_exponents, self.downlink_ant_beams[v1], self.downlink_ant_alignments_rad[v1], dl_freq_1)
                    sig_strengths_2_1[MH0_freq_2] += prob_params.get_dl_lin_signal_strength(angles_from_to_rad[v2,v1], pattern_exponents, self.downlink_ant_beams[v2][0], self.downlink_ant_alignments_rad[v2][0], MH0_freq_2)
                if MH1_freq_2 != -1:
                  if ul_freq_1 == MH1_freq_2:
                    sig_strengths_1_2[ul_freq_1] += pow(10,prob_params.get_ul_dB_signal_strength(angles_from_to_rad[v1,v2], pattern_exponents, self.uplink_ant_nb_beams[v1], self.uplink_ant_alignments_rad[v1], ul_freq_1)/10)
                    sig_strengths_2_1[MH1_freq_2] += prob_params.get_dl_lin_signal_strength(angles_from_to_rad[v2,v1], pattern_exponents, self.downlink_ant_beams[v2][1], self.downlink_ant_alignments_rad[v2][1], MH1_freq_2)
                  if dl_freq_1 == MH1_freq_2:
                    sig_strengths_1_2[dl_freq_1] += prob_params.get_dl_lin_signal_strength(angles_from_to_rad[v1,v2], pattern_exponents, self.downlink_ant_beams[v1], self.downlink_ant_alignments_rad[v1], dl_freq_1)
                    sig_strengths_2_1[MH1_freq_2] += prob_params.get_dl_lin_signal_strength(angles_from_to_rad[v2,v1], pattern_exponents, self.downlink_ant_beams[v2][1], self.downlink_ant_alignments_rad[v2][1], MH1_freq_2)
              else:
                ul_freq_2,dl_freq_2 = self.get_ul_dl_freqs(v2)
                if ul_freq_2 != -1:
                  if ul_freq_1 == ul_freq_2 and ul_freq_2 != -1:
                    sig_strengths_1_2[ul_freq_1] += pow(10,prob_params.get_ul_dB_signal_strength(angles_from_to_rad[v1,v2], pattern_exponents, self.uplink_ant_nb_beams[v1], self.uplink_ant_alignments_rad[v1], ul_freq_1)/10)
                    sig_strengths_2_1[ul_freq_2] += pow(10,prob_params.get_ul_dB_signal_strength(angles_from_to_rad[v2,v1], pattern_exponents, self.uplink_ant_nb_beams[v2], self.uplink_ant_alignments_rad[v2], ul_freq_2)/10)
                  if dl_freq_1 == ul_freq_2 and ul_freq_2 != -1:
                    sig_strengths_1_2[dl_freq_1] += prob_params.get_dl_lin_signal_strength(angles_from_to_rad[v1,v2], pattern_exponents, self.downlink_ant_beams[v1], self.downlink_ant_alignments_rad[v1], dl_freq_1)
                    sig_strengths_2_1[ul_freq_2] += pow(10,prob_params.get_ul_dB_signal_strength(angles_from_to_rad[v2,v1], pattern_exponents, self.uplink_ant_nb_beams[v2], self.uplink_ant_alignments_rad[v2], ul_freq_2)/10)
                if dl_freq_2 != -1:
                  if ul_freq_1 == dl_freq_2 and dl_freq_2 != -1:
                    sig_strengths_1_2[ul_freq_1] += pow(10,prob_params.get_ul_dB_signal_strength(angles_from_to_rad[v1,v2], pattern_exponents, self.uplink_ant_nb_beams[v1], self.uplink_ant_alignments_rad[v1], ul_freq_1)/10)
                    sig_strengths_2_1[dl_freq_2] += prob_params.get_dl_lin_signal_strength(angles_from_to_rad[v2,v1], pattern_exponents, self.downlink_ant_beams[v2], self.downlink_ant_alignments_rad[v2], dl_freq_2)
                  if dl_freq_1 == dl_freq_2 and dl_freq_2 != -1:
                    sig_strengths_1_2[dl_freq_1] += prob_params.get_dl_lin_signal_strength(angles_from_to_rad[v1,v2], pattern_exponents, self.downlink_ant_beams[v1], self.downlink_ant_alignments_rad[v1], dl_freq_1)
                    sig_strengths_2_1[dl_freq_2] += prob_params.get_dl_lin_signal_strength(angles_from_to_rad[v2,v1], pattern_exponents, self.downlink_ant_beams[v2], self.downlink_ant_alignments_rad[v2], dl_freq_2)
              sig_strengths_1_2 = 10*np.log10(sig_strengths_1_2)
              sig_strengths_2_1 = 10*np.log10(sig_strengths_2_1)
              lin_SINRs_1_2 = np.power(10,(30 + sig_strengths_1_2 + sig_strengths_2_1 - path_losses_dB[v1,v2,:])/10)
              lin_SINRs[v1,v2,1:] = lin_SINRs_1_2
              lin_SINRs[v2,v1,1:] = lin_SINRs_1_2
              if v2 != self.imp_arb.MH and v1 == self.imp_arb.preds[v2]:
                lin_SINRs[v1,v2,0] = 10*np.log10(lin_SINRs_1_2[ul_freq_2]) - fade_margins_dB[v1,v2,ul_freq_2]
              elif self.imp_arb.preds[v1] == v2:
                lin_SINRs[v2,v1,0] = 10*np.log10(lin_SINRs_1_2[ul_freq_1]) - fade_margins_dB[v1,v2,ul_freq_1]
              elif is_mixed and adj_matrix[v1,v2] >= 2:
                mesh_freq_idx = self.freq_assignment[int(-1*adj_matrix[v1,v2])]
                lin_SINRs[v1,v2,0] = 10*np.log10(lin_SINRs_1_2[mesh_freq_idx]) - fade_margins_dB[v1,v2,mesh_freq_idx]
                lin_SINRs[v2,v1,0] = lin_SINRs[v1,v2,0]
    return lin_SINRs # (nb_exp_nodes,nb_exp_nodes,FreqMHz.nb+2) 0: Topology (SINRs only at pred,succ); (if nb_freqs_per_channel==2) 1-4: Freq 0-3 (Symmetric); (else if nb_freqs_per_channel==1) 1-2: Chans -3,-2,empty (Symmetric)
  
  """
  Returns the uplink and downlink frequencies of the connections with respect to a specific explicit node where
    'exp_node' is the explicit node.
  """
  def get_ul_dl_freqs(self, exp_node):
    if exp_node == self.imp_arb.MH:
      if len(self.MH_succs_2_partition[1]) > 0:
        return self.freq_assignment[exp_node][0],self.freq_assignment[exp_node][1]
      else:
        return self.freq_assignment[exp_node][0],-1
    else:
      pred = self.imp_arb.preds[exp_node]
      if pred == self.imp_arb.MH:
        if exp_node in self.MH_succs_2_partition[0]:
          ul_freq = self.freq_assignment[pred][0]
        else:
          ul_freq = self.freq_assignment[pred][1]
      else:
        ul_freq = self.freq_assignment[pred]
      if exp_node in self.freq_assignment:
        dl_freq = self.freq_assignment[exp_node]
      else:
        dl_freq = -1
      return ul_freq,dl_freq

  """
  Returns a matrix containing the SINRs in dB (taking interference into account) of the network for the "full" objective and the "greedy" pseudo-objective where
    'all_lin_simp_SINRs' is the tensor of the SINRs of the connected and interfering nodes given by .get_all_linear_simp_SINRs(...).
  """
  def add_interference_to_simps(self, all_lin_simp_SINRs):
    top_log_simp_SINRs = all_lin_simp_SINRs[:,:,0]
    freq_lin_simp_SINRs = all_lin_simp_SINRs[:,:,1:]
    nb_nodes = self.imp_arb.prob_instance.nb_nodes
    noise_power_dBm = self.imp_arb.prob_params.noise_powers_dBm[0]
    SINRs_dB = np.zeros((nb_nodes,nb_nodes))   
    nodes_to_visit = [self.imp_arb.MH]
    while len(nodes_to_visit) > 0:
      new_pred = nodes_to_visit.pop(0)
      if new_pred in self.imp_arb.succ_lists:
        succs = self.imp_arb.succ_lists[new_pred]
        nodes_to_visit += succs
        if new_pred >= 0:
          if new_pred == self.imp_arb.MH:
            for i in range(2):
              freq_index = self.freq_assignment[new_pred][i]
              succs = self.MH_succs_2_partition[i]
              out_of_connection_nodes = list(set(range(nb_nodes)) - set(succs).union({new_pred}))
              lin_pred_interf = np.sum(freq_lin_simp_SINRs[new_pred,out_of_connection_nodes,freq_index])
              for succ in succs:
                if succ >= 0:
                  lin_succ_interf = np.sum(freq_lin_simp_SINRs[out_of_connection_nodes,succ,freq_index])
                  useful_sig_dBm = top_log_simp_SINRs[new_pred,succ]
                  interf_noise_dBm = 10*np.log10(lin_pred_interf + lin_succ_interf + np.power(10,noise_power_dBm/10))
                  SINRs_dB[new_pred, succ] = useful_sig_dBm-interf_noise_dBm
          else:
            if len(succs) > 0:
              freq_index = self.freq_assignment[new_pred]
              out_of_connection_nodes = list(set(range(nb_nodes)) - set(succs).union({new_pred}))
              lin_pred_interf = np.sum(freq_lin_simp_SINRs[new_pred,out_of_connection_nodes,freq_index])
              for succ in succs:
                if succ >= 0:
                  lin_succ_interf = np.sum(freq_lin_simp_SINRs[out_of_connection_nodes,succ,freq_index])
                  useful_sig_dBm = top_log_simp_SINRs[new_pred,succ]
                  interf_noise_dBm = 10*np.log10(lin_pred_interf + lin_succ_interf + np.power(10,noise_power_dBm/10))
                  SINRs_dB[new_pred, succ] = useful_sig_dBm-interf_noise_dBm
    if self.is_mixed:
      mesh_members = self.imp_arb.mesh_members
      for mesh_idx in mesh_members.keys():
        if mesh_idx == self.imp_arb.MH:
          freq_index = self.freq_assignment[mesh_idx][0]
        else:
          freq_index = self.freq_assignment[mesh_idx]
        members = mesh_members[mesh_idx]
        out_of_connection_nodes = list(set(range(nb_nodes)) - set(members))
        for member_idx1 in range(len(members)):
          member1 = members[member_idx1]
          lin_member1_interf = np.sum(freq_lin_simp_SINRs[member1,out_of_connection_nodes,freq_index])
          for member_idx2 in range(member_idx1+1,len(members)):
            member2 = members[member_idx2]
            lin_member2_interf = np.sum(freq_lin_simp_SINRs[out_of_connection_nodes,member2,freq_index])
            useful_sig_dBm = top_log_simp_SINRs[member1,member2]
            interf_noise_dBm = 10*np.log10(lin_member1_interf + lin_member2_interf + np.power(10,noise_power_dBm/10))
            SINRs_dB[member1, member2] = useful_sig_dBm-interf_noise_dBm
            SINRs_dB[member2, member1] = SINRs_dB[member1, member2]
    return SINRs_dB # (nb_exp_nodes,nb_exp_nodes) SINRs only at pred,succ

  """
  Computes the direct throughputs of the network where
    'same_old_preds' is a list of explicit nodes whose direct outgoing (downlink) connections have not changed since the previous ImpTree neighbor's pseudo-network;
    'do_full_mesh_evaluation' is whether the network is evaluated fully (True: "full"/"greedy") or approximately (False: "avg"/"chan_avg"/"chan_max");
    'same_old_meshes' is a list of mesh clusters that have not changed since the previous ImpTree neighbor's pseudo-network;
    'old_direct_simp_TPs' is the direct throughputs of the best pseudo-network of a previous ImpTree neighbor (see Network for details).
  """
  def get_direct_TPs(self, SINRs, same_old_preds, do_full_mesh_evaluation, same_old_meshes, old_direct_simp_TPs):
    nb_nodes = self.imp_arb.prob_instance.nb_nodes
    adj_matrix = self.imp_arb.adj_matrix
    TPs = np.zeros((nb_nodes,nb_nodes))
    SINR_threshold_throughput_table = self.imp_arb.prob_params.SINR_threshold_throughput_table
    if len(SINR_threshold_throughput_table) > 0:
      for table_idx in range(len(SINR_threshold_throughput_table)):
        if table_idx == 0:
          TPs[SINRs < SINR_threshold_throughput_table[table_idx][0]] = 0
        if table_idx < len(SINR_threshold_throughput_table)-1:
          TPs[(SINRs >= SINR_threshold_throughput_table[table_idx][0])*(SINRs < SINR_threshold_throughput_table[table_idx+1][0])] = SINR_threshold_throughput_table[table_idx][1]
        else:
          TPs[SINRs >= SINR_threshold_throughput_table[table_idx][0]] = SINR_threshold_throughput_table[table_idx][1]
    else:
      TPs[SINRs != 0] = 20 * np.log2(1 + np.power(10,SINRs[SINRs != 0]/10))
    MH = self.imp_arb.MH
    MH_succs_2_partition = self.MH_succs_2_partition
    succ_lists = self.imp_arb.succ_lists
    PTP_coeff = self.imp_arb.prob_params.PTP_coeff
    for u in range(nb_nodes):
      if u == MH:
        for i in range(2):
          succs = MH_succs_2_partition[i]
          if len(succs) == 1 and succs[0] >= 0: # PTP
            TPs[u,succs[0]] = PTP_coeff*TPs[u,succs[0]]
      else:
        if u in self.imp_arb.preds:
          succs = [node for node in succ_lists[u] if node >= 0]
          if len(succs) == 1: # PTP
            succ = succs[0]
            TPs[u,succ] = PTP_coeff*TPs[u,succ]
    for same_old_pred in same_old_preds:
      if type(same_old_pred) is tuple:
        MH = same_old_pred[0]
        i = same_old_pred[1]
        for v in self.MH_succs_2_partition[i]:
          TPs[MH,v] = old_direct_simp_TPs[MH,v]
      else:
        succs = succ_lists[same_old_pred]
        for succ in succs:
          TPs[same_old_pred,succ] = old_direct_simp_TPs[same_old_pred,succ]
    if self.is_mixed:
      mesh_collisions = self.imp_arb.prob_params.mesh_collisions
      Mesh_coeff = self.imp_arb.prob_params.Mesh_coeff
      mesh_members = self.imp_arb.mesh_members
      mesh_routes = {}
      for mesh_idx in mesh_members.keys():
        members = mesh_members[mesh_idx]
        if not mesh_idx in same_old_meshes:
          if not do_full_mesh_evaluation or not mesh_collisions:
            TPs[np.ix_(members,members)] *= Mesh_coeff
        else:
          for member_idx1 in range(len(members)):
            member1 = members[member_idx1]
            for member_idx2 in range(member_idx1+1,len(members)):
              member2 = members[member_idx2]
              TPs[member1,member2] = old_direct_simp_TPs[member1,member2]
              TPs[member2,member1] = old_direct_simp_TPs[member2,member1]
    self.direct_TPs = TPs # (nb_exp_nodes,nb_exp_nodes)

  """
  Computes the effective throughputs and objective or pseudo-objective values for every traffic scenario where
    'do_full_mesh_evaluation' is whether the network is evaluated fully (True: "full"/"greedy") or approximately (False: "avg"/"chan_avg"/"chan_max");
    'same_old_meshes' is a list of mesh clusters that have not changed since the previous ImpTree neighbor's pseudo-network;
    'old_mesh_routes' (optional) is a dictionary that describes the routes used between every pair of members for each cluster of the best pseudo-network of a previous neighbor (see
      Network for details);
    'SINRs' is a matrix containing the SINRs in dB of the network;
    'keep_previous_routes' (optional) is whether to keep the old mesh routes (True) or to recompute them (False).
  """
  def get_effective_TPs(self, do_full_mesh_evaluation, same_old_meshes, old_mesh_routes, SINRs, keep_previous_routes=False):
    obj_scenarios = self.imp_arb.prob_params.obj_scenarios
    mesh_high_traffic_routes_prioritization = self.imp_arb.prob_params.mesh_high_traffic_routes_prioritization
    mesh_collisions = self.imp_arb.prob_params.mesh_collisions
    nb_nodes = self.imp_arb.prob_instance.nb_nodes
    succ_lists = self.imp_arb.succ_lists
    preds = self.imp_arb.preds
    direct_TPs = self.direct_TPs
    nodes_to_visit = [self.imp_arb.MH]
    idx = 0
    while idx < len(nodes_to_visit):
      new_node = nodes_to_visit[idx]
      if new_node == self.imp_arb.MH:
        nodes_to_visit += self.MH_succs_2_partition[0]
        nodes_to_visit += self.MH_succs_2_partition[1]
      else:
        nodes_to_visit += succ_lists[new_node]
      idx += 1
    if not self.is_mixed:
      nbs_descs_inc_self = [0 for exp_node in range(nb_nodes)]
    else:
      nbs_descs_inc_self = [0 for imp_node in range(nb_nodes+int(np.max(self.imp_arb.adj_matrix)))]
    arcs = []
    A_effective_TPs = []
    B_effective_TPs = []
    C_effective_TPs = []
    while len(nodes_to_visit) > 0:
      new_node = nodes_to_visit.pop(-1)
      if new_node >= 0:
        nb_descs_inc_self = 1
      else:
        nb_descs_inc_self = 0
      succs = succ_lists[new_node]
      for succ in succs:
        nb_descs_inc_self += nbs_descs_inc_self[succ]
      nbs_descs_inc_self[new_node] = nb_descs_inc_self
      if new_node != self.imp_arb.MH:
        if new_node >= 0:
          pred = preds[new_node]
          if pred >= 0:
            arcs.append((pred,new_node))
            A_effective_TPs.append(direct_TPs[pred,new_node])
            B_effective_TPs.append(direct_TPs[pred,new_node]/nb_descs_inc_self)
            C_effective_TPs.append(direct_TPs[pred,new_node]/(nb_descs_inc_self * (nb_nodes - nb_descs_inc_self)))
    # print(nbs_descs_inc_self)
    self.nbs_descs_inc_self = nbs_descs_inc_self # [nb_nodes]
    if self.is_mixed:
      mesh_members = self.imp_arb.mesh_members
      A_mesh_routes = {}
      if not do_full_mesh_evaluation or mesh_high_traffic_routes_prioritization == 0:
        AC_arcs = copy.copy(arcs)
      else:
        C_mesh_routes = {}
        A_arcs = copy.copy(arcs)
        C_arcs = copy.copy(arcs)
      B_arcs = copy.copy(arcs)
      if do_full_mesh_evaluation and mesh_high_traffic_routes_prioritization != 0:
        A_mesh_TP_denoms = np.zeros((nb_nodes,nb_nodes))
      B_mesh_TP_denoms = np.zeros((nb_nodes,nb_nodes))
      C_mesh_TP_denoms = np.zeros((nb_nodes,nb_nodes))
      for mesh_idx in mesh_members.keys():
        members = mesh_members[mesh_idx]
        if mesh_idx in preds:
          mesh_pred = preds[mesh_idx]
        else:
          mesh_pred = -1
        B_TP_denoms = np.zeros((len(members),len(members)))
        C_TP_denoms = np.zeros((len(members),len(members)))
        if not keep_previous_routes or not hasattr(self, 'A_mesh_routes'):
          if not mesh_idx in same_old_meshes:
            # Floyd-Warshall
            new_mesh_TPs = np.zeros((len(members),len(members)))
            next_node_in_paths = -1*np.ones((len(members),len(members)))
            mesh_direct_TPs = direct_TPs[np.ix_(members,members)]
            non_zero_mask = mesh_direct_TPs != 0
            new_mesh_TPs[non_zero_mask] = mesh_direct_TPs[non_zero_mask]
            next_node_in_paths[non_zero_mask] = (np.indices((len(members),len(members)))[1])[non_zero_mask]
            for member_idx1 in range(len(members)):
              next_node_in_paths[member_idx1,member_idx1] = member_idx1
            for member_idx3 in range(len(members)):
              for member_idx1 in range(len(members)):
                for member_idx2 in range(member_idx1+1,len(members)):
                  loc_TP = min(new_mesh_TPs[member_idx1,member_idx3],new_mesh_TPs[member_idx3,member_idx2])
                  if loc_TP > new_mesh_TPs[member_idx1,member_idx2]:
                    new_mesh_TPs[member_idx2,member_idx1] = new_mesh_TPs[member_idx1,member_idx2] = loc_TP
                    next_node_in_paths[member_idx1,member_idx2] = next_node_in_paths[member_idx1,member_idx3]
                    next_node_in_paths[member_idx2,member_idx1] = next_node_in_paths[member_idx2,member_idx3]
            curr_mesh_routes = dict()
            for member_idx1 in range(len(members)):
              member1 = members[member_idx1]
              if not member1 in curr_mesh_routes:
                curr_mesh_routes[member1] = dict()
              for member_idx2 in range(member_idx1+1,len(members)):
                member2 = members[member_idx2]
                if not member2 in curr_mesh_routes:
                  curr_mesh_routes[member2] = dict()
                if new_mesh_TPs[member_idx1,member_idx2] > 0:
                  curr_path = []
                  curr_member_idx = member_idx1
                  while curr_member_idx != member_idx2:
                    old_member_idx = curr_member_idx
                    curr_member_idx = round(next_node_in_paths[curr_member_idx,member_idx2])
                    curr_path.append((old_member_idx,curr_member_idx))
                  curr_mesh_routes[member2][member1] = curr_mesh_routes[member1][member2] = [(members[old_member_idx],members[curr_member_idx]) for old_member_idx,curr_member_idx in curr_path]
                else:
                  curr_mesh_routes[member2][member1] = curr_mesh_routes[member1][member2] = [(member1,member2)]
            A_mesh_routes[mesh_idx] = curr_mesh_routes
          else:
            A_mesh_routes[mesh_idx] = old_mesh_routes[mesh_idx]
          if not do_full_mesh_evaluation or mesh_high_traffic_routes_prioritization == 0:
            for member_idx1 in range(len(members)):
              member1 = members[member_idx1]
              if mesh_pred >= 0 and member1 != mesh_pred:
                mesh_route = A_mesh_routes[mesh_idx][mesh_pred][member1]
                for mesh_edge in mesh_route:
                  member_idx3 = members.index(mesh_edge[0])
                  member_idx4 = members.index(mesh_edge[1])
                  B_TP_denoms[member_idx3,member_idx4] += nbs_descs_inc_self[member1]
                  B_TP_denoms[member_idx4,member_idx3] = B_TP_denoms[member_idx3,member_idx4]
              for member_idx2 in range(member_idx1+1,len(members)):
                member2 = members[member_idx2]
                mesh_route = A_mesh_routes[mesh_idx][member1][member2]
                for mesh_edge in mesh_route:
                  member_idx3 = members.index(mesh_edge[0])
                  member_idx4 = members.index(mesh_edge[1])
                  if member1 == mesh_pred:
                    C_TP_denoms[member_idx3,member_idx4] += nbs_descs_inc_self[member2] * (nb_nodes - nbs_descs_inc_self[mesh_idx])
                  elif member2 == mesh_pred:
                    C_TP_denoms[member_idx3,member_idx4] += nbs_descs_inc_self[member1] * (nb_nodes - nbs_descs_inc_self[mesh_idx])
                  else:
                    C_TP_denoms[member_idx3,member_idx4] += nbs_descs_inc_self[member1] * nbs_descs_inc_self[member2]
                  C_TP_denoms[member_idx4,member_idx3] = C_TP_denoms[member_idx3,member_idx4]
          else:
            A_TP_denoms = np.zeros((len(members),len(members)))
            for member_idx1 in range(len(members)):
              member1 = members[member_idx1]
              for member_idx2 in range(member_idx1+1,len(members)):
                member2 = members[member_idx2]
                mesh_route = A_mesh_routes[mesh_idx][member1][member2]
                for mesh_edge in mesh_route:
                  member_idx3 = members.index(mesh_edge[0])
                  member_idx4 = members.index(mesh_edge[1])
                  A_TP_denoms[member_idx4,member_idx3] = A_TP_denoms[member_idx3,member_idx4] = 1
            
            fixed_group_size = floor(pow(len(members)*(len(members)-1)/2, 1-mesh_high_traffic_routes_prioritization))
            nb_groups = ceil((len(members)*(len(members)-1)/2)/fixed_group_size)
            fixed_group_size = ceil((len(members)*(len(members)-1)/2)/nb_groups)
            mesh_pred_idx = members.index(mesh_pred)
            ordered_member_idcs = [(member_idx, nbs_descs_inc_self[members[member_idx]]) for member_idx in range(len(members))]
            ordered_member_idcs.sort(key=lambda pair:pair[1], reverse=True)
            ordered_members_idcs = []
            for ordered_member_idcs_idx1 in range(len(ordered_member_idcs)):
              member_idx1,_ = ordered_member_idcs[ordered_member_idcs_idx1]
              if member_idx1 != mesh_pred_idx:
                member1 = members[member_idx1]
                for ordered_member_idcs_idx2 in range(ordered_member_idcs_idx1+1, len(ordered_member_idcs)):
                  member_idx2,_ = ordered_member_idcs[ordered_member_idcs_idx2]
                  if member_idx2 != mesh_pred_idx:
                    member2 = members[member_idx2]
                    curr_route_weight = nbs_descs_inc_self[member1] * nbs_descs_inc_self[member2]
                    ordered_members_idcs.append((member_idx1,member_idx2,curr_route_weight))
            ordered_members_idcs.sort(key=lambda pair:pair[2], reverse=True)
            ordered_members_idcs = [(mesh_pred_idx,member_idx1,nbs_descs_inc_self1 * (nb_nodes - nbs_descs_inc_self[mesh_idx])) for member_idx1,nbs_descs_inc_self1 in ordered_member_idcs if member_idx1 != mesh_pred_idx] + ordered_members_idcs
            grouped_ordered_members_idcs = [([(ordered_members_idcs[0][0],ordered_members_idcs[0][1])], ordered_members_idcs[0][2])]
            for ordered_members_idcs_idx in range(1, len(ordered_members_idcs)):
              member_idx1, member_idx2, curr_route_weight = ordered_members_idcs[ordered_members_idcs_idx]
              if curr_route_weight == grouped_ordered_members_idcs[-1][1]:
                grouped_ordered_members_idcs[-1][0].append((member_idx1,member_idx2))
              else:
                grouped_ordered_members_idcs.append(([(member_idx1,member_idx2)], curr_route_weight))
            #print([(len(member_idcs_edge_list),weight) for member_idcs_edge_list, weight in grouped_ordered_members_idcs])
            new_grouped_ordered_members_idcs = []
            for member_idcs_edge_list, weight in grouped_ordered_members_idcs:
              if len(new_grouped_ordered_members_idcs) == 0 or len(new_grouped_ordered_members_idcs[-1][0]) == fixed_group_size:
                if len(member_idcs_edge_list) <= fixed_group_size:
                  new_grouped_ordered_members_idcs.append([member_idcs_edge_list, weight])
                else:
                  random.shuffle(member_idcs_edge_list)
                  while len(member_idcs_edge_list) > 0:
                    new_grouped_ordered_members_idcs.append([member_idcs_edge_list[:fixed_group_size], weight])
                    member_idcs_edge_list = member_idcs_edge_list[fixed_group_size:]
              else:
                remainder_group_size = fixed_group_size - len(new_grouped_ordered_members_idcs[-1][0])
                if len(member_idcs_edge_list) <= remainder_group_size:
                  new_grouped_ordered_members_idcs[-1][0] += member_idcs_edge_list
                else:
                  random.shuffle(member_idcs_edge_list)
                  new_grouped_ordered_members_idcs[-1][0] += member_idcs_edge_list[:remainder_group_size]
                  member_idcs_edge_list = member_idcs_edge_list[remainder_group_size:]
                  while len(member_idcs_edge_list) > 0:
                    new_grouped_ordered_members_idcs.append([member_idcs_edge_list[:fixed_group_size], weight])
                    member_idcs_edge_list = member_idcs_edge_list[fixed_group_size:]
            grouped_ordered_members_idcs = [(member_list_weight[0],member_list_weight[1]) for member_list_weight in new_grouped_ordered_members_idcs]
            #print([(len(member_list),weight) for member_list, weight in grouped_ordered_members_idcs])
            curr_mesh_routes = [[-1 for member_idx2 in range(len(members))] for member_idx1 in range(len(members))]
            for member_idcs_edge_list, max_curr_route_weight in grouped_ordered_members_idcs:
              # Floyd-Warshall
              new_mesh_TPs = np.zeros((len(members),len(members)))
              next_node_in_paths = -1*np.ones((len(members),len(members)))
              mesh_direct_TPs = direct_TPs[np.ix_(members,members)]/(max_curr_route_weight+C_TP_denoms)
              non_zero_mask = mesh_direct_TPs != 0
              new_mesh_TPs[non_zero_mask] = mesh_direct_TPs[non_zero_mask]
              next_node_in_paths[non_zero_mask] = (np.indices((len(members),len(members)))[1])[non_zero_mask]
              for member_idx1 in range(len(members)):
                next_node_in_paths[member_idx1,member_idx1] = member_idx1
              for member_idx3 in range(len(members)):
                for member_idx1 in range(len(members)):
                  for member_idx2 in range(member_idx1+1,len(members)):
                    loc_TP = min(new_mesh_TPs[member_idx1,member_idx3],new_mesh_TPs[member_idx3,member_idx2])
                    if loc_TP > new_mesh_TPs[member_idx1,member_idx2]:
                      new_mesh_TPs[member_idx2,member_idx1] = new_mesh_TPs[member_idx1,member_idx2] = loc_TP
                      next_node_in_paths[member_idx1,member_idx2] = next_node_in_paths[member_idx1,member_idx3]
                      next_node_in_paths[member_idx2,member_idx1] = next_node_in_paths[member_idx2,member_idx3]
              
              for member_idx1,member_idx2 in member_idcs_edge_list:
                if new_mesh_TPs[member_idx1,member_idx2] > 0:
                  curr_path = []
                  curr_member_idx = member_idx1
                  while curr_member_idx != member_idx2:
                    old_member_idx = curr_member_idx
                    curr_member_idx = round(next_node_in_paths[curr_member_idx,member_idx2])
                    curr_path.append((old_member_idx,curr_member_idx))
                  mesh_route = [(members[old_member_idx],members[curr_member_idx]) for old_member_idx,curr_member_idx in curr_path]
                else:
                  mesh_route = [(members[member_idx1],members[member_idx2])]
                curr_mesh_routes[member_idx1][member_idx2] = curr_mesh_routes[member_idx2][member_idx1] = mesh_route
                if member_idx1 == mesh_pred_idx:
                  curr_route_weight = nbs_descs_inc_self[members[member_idx2]] * (nb_nodes - nbs_descs_inc_self[mesh_idx])
                elif member_idx2 == mesh_pred_idx:
                  curr_route_weight = nbs_descs_inc_self[members[member_idx1]] * (nb_nodes - nbs_descs_inc_self[mesh_idx])
                else:
                  curr_route_weight = nbs_descs_inc_self[members[member_idx1]] * nbs_descs_inc_self[members[member_idx2]]
                for mesh_edge in mesh_route:
                  member_idx3 = members.index(mesh_edge[0])
                  member_idx4 = members.index(mesh_edge[1])
                  if member_idx1 == mesh_pred_idx:
                    B_TP_denoms[member_idx3,member_idx4] += nbs_descs_inc_self[members[member_idx2]]
                    B_TP_denoms[member_idx4,member_idx3] = B_TP_denoms[member_idx3,member_idx4]
                  elif member_idx2 == mesh_pred_idx:
                    B_TP_denoms[member_idx3,member_idx4] += nbs_descs_inc_self[members[member_idx1]]
                    B_TP_denoms[member_idx4,member_idx3] = B_TP_denoms[member_idx3,member_idx4]
                  C_TP_denoms[member_idx3,member_idx4] += curr_route_weight
                  C_TP_denoms[member_idx4,member_idx3] = C_TP_denoms[member_idx3,member_idx4]
              #print(C_TP_denoms)
            C_mesh_routes[mesh_idx] = {members[member_idx1]:{members[member_idx2]:curr_mesh_routes[member_idx1][member_idx2] for member_idx2 in range(len(members)) if member_idx1 != member_idx2} for member_idx1 in range(len(members))}
        else:
          A_mesh_routes = self.A_mesh_routes
          if not do_full_mesh_evaluation or mesh_high_traffic_routes_prioritization == 0:
            for member_idx1 in range(len(members)):
              member1 = members[member_idx1]
              if mesh_pred >= 0 and member1 != mesh_pred:
                mesh_route = A_mesh_routes[mesh_idx][mesh_pred][member1]
                for mesh_edge in mesh_route:
                  member_idx3 = members.index(mesh_edge[0])
                  member_idx4 = members.index(mesh_edge[1])
                  B_TP_denoms[member_idx3,member_idx4] += nbs_descs_inc_self[member1]
                  B_TP_denoms[member_idx4,member_idx3] = B_TP_denoms[member_idx3,member_idx4]
              for member_idx2 in range(member_idx1+1,len(members)):
                member2 = members[member_idx2]
                mesh_route = A_mesh_routes[mesh_idx][member1][member2]
                for mesh_edge in mesh_route:
                  member_idx3 = members.index(mesh_edge[0])
                  member_idx4 = members.index(mesh_edge[1])
                  if member1 == mesh_pred:
                    C_TP_denoms[member_idx3,member_idx4] += nbs_descs_inc_self[member2] * (nb_nodes - nbs_descs_inc_self[mesh_idx])
                  elif member2 == mesh_pred:
                    C_TP_denoms[member_idx3,member_idx4] += nbs_descs_inc_self[member1] * (nb_nodes - nbs_descs_inc_self[mesh_idx])
                  else:
                    C_TP_denoms[member_idx3,member_idx4] += nbs_descs_inc_self[member1] * nbs_descs_inc_self[member2]
                  C_TP_denoms[member_idx4,member_idx3] = C_TP_denoms[member_idx3,member_idx4]
          else:
            mesh_pred_idx = members.index(mesh_pred)
            C_mesh_routes = self.C_mesh_routes
            A_TP_denoms = np.zeros((len(members),len(members)))
            for member_idx1 in range(len(members)):
              member1 = members[member_idx1]
              for member_idx2 in range(member_idx1+1,len(members)):
                member2 = members[member_idx2]
                mesh_route = A_mesh_routes[mesh_idx][member1][member2]
                for mesh_edge in mesh_route:
                  member_idx3 = members.index(mesh_edge[0])
                  member_idx4 = members.index(mesh_edge[1])
                  A_TP_denoms[member_idx4,member_idx3] = A_TP_denoms[member_idx3,member_idx4] = 1
                mesh_route = C_mesh_routes[mesh_idx][member1][member2]
                if member_idx1 == mesh_pred_idx:
                  curr_route_weight = nbs_descs_inc_self[members[member_idx2]] * (nb_nodes - nbs_descs_inc_self[mesh_idx])
                elif member_idx2 == mesh_pred_idx:
                  curr_route_weight = nbs_descs_inc_self[members[member_idx1]] * (nb_nodes - nbs_descs_inc_self[mesh_idx])
                else:
                  curr_route_weight = nbs_descs_inc_self[members[member_idx1]] * nbs_descs_inc_self[members[member_idx2]]
                for mesh_edge in mesh_route:
                  member_idx3 = members.index(mesh_edge[0])
                  member_idx4 = members.index(mesh_edge[1])
                  if member_idx1 == mesh_pred_idx:
                    B_TP_denoms[member_idx3,member_idx4] += nbs_descs_inc_self[members[member_idx2]]
                    B_TP_denoms[member_idx4,member_idx3] = B_TP_denoms[member_idx3,member_idx4]
                  elif member_idx2 == mesh_pred_idx:
                    B_TP_denoms[member_idx3,member_idx4] += nbs_descs_inc_self[members[member_idx1]]
                    B_TP_denoms[member_idx4,member_idx3] = B_TP_denoms[member_idx3,member_idx4]
                  C_TP_denoms[member_idx3,member_idx4] += curr_route_weight
                  C_TP_denoms[member_idx4,member_idx3] = C_TP_denoms[member_idx3,member_idx4]
        if do_full_mesh_evaluation and mesh_collisions:
          collision_A_TP_denoms = np.zeros((len(members),len(members)))
          collision_B_TP_denoms = np.zeros((len(members),len(members)))
          collision_C_TP_denoms = np.zeros((len(members),len(members)))
          if len(self.imp_arb.prob_params.SINR_threshold_throughput_table) > 0:
            min_SINR_collision_threshold = self.imp_arb.prob_params.SINR_threshold_throughput_table[0][0] - 6
          else:
            min_SINR_collision_threshold = -4
          mesh_SINRs = SINRs[np.ix_(members,members)]
          for member_idx1 in range(len(members)):
            member1 = members[member_idx1]
            member1_collision_neighbor_idcs = set(np.where(mesh_SINRs[member_idx1,:] >= min_SINR_collision_threshold)[0])
            for member_idx2 in range(member_idx1+1,len(members)):
              member2 = members[member_idx2]
              mesh_route = A_mesh_routes[mesh_idx][member1][member2]
              for mesh_edge_idx in range(len(mesh_route)):
                mesh_edge = mesh_route[mesh_edge_idx]
                member_idx3 = members.index(mesh_edge[0])
                member_idx4 = members.index(mesh_edge[1])
                if mesh_edge_idx == 0 or mesh_edge_idx == len(mesh_route)-1:
                  new_A_TP_denom = min(len(mesh_route),2)
                else:
                  new_A_TP_denom = min(len(mesh_route),3)
                if new_A_TP_denom > collision_A_TP_denoms[member_idx3,member_idx4]:
                  collision_A_TP_denoms[member_idx4,member_idx3] = collision_A_TP_denoms[member_idx3,member_idx4] = new_A_TP_denom
              new_B_TP_denom = B_TP_denoms[member_idx1,member_idx2]
              new_C_TP_denom = C_TP_denoms[member_idx1,member_idx2]
              if new_B_TP_denom > 0 or new_C_TP_denom > 0:
                member2_collision_neighbor_idcs = set(np.where(mesh_SINRs[member_idx2,:] >= min_SINR_collision_threshold)[0])
                collision_neighbor_idcs = list((member1_collision_neighbor_idcs | member2_collision_neighbor_idcs) - {member_idx1,member_idx2})
                if new_B_TP_denom > 0:
                  collision_edges = set()
                  for collision_neighbor_idx in collision_neighbor_idcs:
                    collision_edges |= {tuple(sorted([collision_neighbor_idx,other_idx])) for other_idx in list(np.where(B_TP_denoms[collision_neighbor_idx,:] > 0)[0])}
                  for member_idx3,member_idx4 in collision_edges:
                    new_B_TP_denom += B_TP_denoms[member_idx3,member_idx4]
                  collision_B_TP_denoms[member_idx2,member_idx1] = collision_B_TP_denoms[member_idx1,member_idx2] = new_B_TP_denom
                if new_C_TP_denom > 0:
                  collision_edges = set()
                  for collision_neighbor_idx in collision_neighbor_idcs:
                    collision_edges |= {tuple(sorted([collision_neighbor_idx,other_idx])) for other_idx in list(np.where(C_TP_denoms[collision_neighbor_idx,:] > 0)[0])}
                  for member_idx3,member_idx4 in collision_edges:
                    new_C_TP_denom += C_TP_denoms[member_idx3,member_idx4]
                  collision_C_TP_denoms[member_idx2,member_idx1] = collision_C_TP_denoms[member_idx1,member_idx2] = new_C_TP_denom
          A_TP_denoms = collision_A_TP_denoms
          B_TP_denoms = collision_B_TP_denoms
          C_TP_denoms = collision_C_TP_denoms
        for member_idx1 in range(len(members)):
          member1 = members[member_idx1]
          for member_idx2 in range(member_idx1+1,len(members)):
            member2 = members[member_idx2]
            if member1 < member2:
              m1 = member1
              m2 = member2
            else:
              m1 = member2
              m2 = member1
            if not do_full_mesh_evaluation or mesh_high_traffic_routes_prioritization == 0:
              if C_TP_denoms[member_idx1,member_idx2] > 0:
                AC_arcs.append((m1,m2))
                if not do_full_mesh_evaluation or not mesh_collisions:
                  A_effective_TPs.append(direct_TPs[member1,member2])
                else:
                  A_effective_TPs.append(direct_TPs[member1,member2]/A_TP_denoms[member_idx1,member_idx2])
                C_effective_TPs.append(direct_TPs[member1,member2]/C_TP_denoms[member_idx1,member_idx2])
                C_mesh_TP_denoms[member2,member1] = C_mesh_TP_denoms[member1,member2] = C_TP_denoms[member_idx1,member_idx2]
            else:
              if A_TP_denoms[member_idx1,member_idx2] > 0:
                A_arcs.append((m1,m2))
                if not mesh_collisions:
                  A_effective_TPs.append(direct_TPs[member1,member2])
                else:
                  A_effective_TPs.append(direct_TPs[member1,member2]/A_TP_denoms[member_idx1,member_idx2])
                A_mesh_TP_denoms[member2,member1] = A_mesh_TP_denoms[member1,member2] = A_TP_denoms[member_idx1,member_idx2]
              if C_TP_denoms[member_idx1,member_idx2] > 0:
                C_arcs.append((m1,m2))
                C_effective_TPs.append(direct_TPs[member1,member2]/C_TP_denoms[member_idx1,member_idx2])
                C_mesh_TP_denoms[member2,member1] = C_mesh_TP_denoms[member1,member2] = C_TP_denoms[member_idx1,member_idx2]
            if B_TP_denoms[member_idx1,member_idx2] > 0:
              B_arcs.append((m1,m2))
              B_effective_TPs.append(direct_TPs[member1,member2]/B_TP_denoms[member_idx1,member_idx2])
              B_mesh_TP_denoms[member2,member1] = B_mesh_TP_denoms[member1,member2] = B_TP_denoms[member_idx1,member_idx2]
      self.A_mesh_routes = A_mesh_routes # dict : mesh -> dict : mesh_member1 -> dict : mesh_member2 -> list of mesh edges used between member1 and member2
      if do_full_mesh_evaluation and mesh_high_traffic_routes_prioritization != 0:
        self.C_mesh_routes = C_mesh_routes # dict : mesh -> dict : mesh_member1 -> dict : mesh_member2 -> list of mesh edges used between member1 and member2
        self.A_mesh_TP_denoms = A_mesh_TP_denoms # (nb_exp_nodes,nb_exp_nodes)
      self.B_mesh_TP_denoms = B_mesh_TP_denoms # (nb_exp_nodes,nb_exp_nodes)
      self.C_mesh_TP_denoms = C_mesh_TP_denoms # (nb_exp_nodes,nb_exp_nodes)
    avg_factor = 1/39
    self.A_obj_min = min(A_effective_TPs) # float
    A_obj_avg = mean(A_effective_TPs)
    self.A_obj_value = self.A_obj_min + avg_factor*A_obj_avg # float
    if len(B_effective_TPs) > 0:
      self.B_obj_min = min(B_effective_TPs) # float
      B_obj_avg = mean(B_effective_TPs)
    else:
      self.B_obj_min = B_obj_avg = 0
    self.B_obj_value = self.B_obj_min + avg_factor*B_obj_avg # float
    if len(C_effective_TPs) > 0:
      self.C_obj_min = (nb_nodes-1)*min(C_effective_TPs) # float
      C_obj_avg = mean(C_effective_TPs)
    else:
      self.C_obj_min = C_obj_avg = 0
    self.C_obj_value = self.C_obj_min + (nb_nodes-1)*avg_factor*C_obj_avg # float
    A_factor = obj_scenarios[0]
    B_factor = obj_scenarios[1]
    C_factor = obj_scenarios[2]
    self.obj_value = A_factor*self.A_obj_value + B_factor*self.B_obj_value + C_factor*self.C_obj_value # float
    self.has_obj_value = True # boolean
    if not self.is_mixed:
      self.A_lim_links = [arcs[idx] for idx, A_effective_TP in enumerate(A_effective_TPs) if A_effective_TP == self.A_obj_min] # list of tuples representing links in the limiting paths
      self.B_lim_links = [arcs[idx] for idx, B_effective_TP in enumerate(B_effective_TPs) if B_effective_TP == self.B_obj_min] # list of tuples representing links in the limiting paths
      self.C_lim_links = [arcs[idx] for idx, C_effective_TP in enumerate(C_effective_TPs) if (nb_nodes-1)*C_effective_TP == self.C_obj_min] # list of tuples representing links in the limiting paths
      return arcs, A_effective_TPs, B_effective_TPs, C_effective_TPs
    else:
      self.B_lim_links = [B_arcs[idx]  for idx, B_effective_TP in enumerate(B_effective_TPs) if B_effective_TP == self.B_obj_min] # list of tuples representing links in the limiting paths
      if not do_full_mesh_evaluation or mesh_high_traffic_routes_prioritization == 0:
        self.A_lim_links = [AC_arcs[idx] for idx, A_effective_TP in enumerate(A_effective_TPs) if A_effective_TP == self.A_obj_min] # list of tuples representing links in the limiting paths
        self.C_lim_links = [AC_arcs[idx] for idx, C_effective_TP in enumerate(C_effective_TPs) if (nb_nodes-1)*C_effective_TP == self.C_obj_min] # list of tuples representing links in the limiting paths
        return AC_arcs, B_arcs, A_effective_TPs, B_effective_TPs, C_effective_TPs
      else:
        self.A_lim_links = [A_arcs[idx] for idx, A_effective_TP in enumerate(A_effective_TPs) if A_effective_TP == self.A_obj_min] # list of tuples representing links in the limiting paths
        self.C_lim_links = [C_arcs[idx] for idx, C_effective_TP in enumerate(C_effective_TPs) if (nb_nodes-1)*C_effective_TP == self.C_obj_min] # list of tuples representing links in the limiting paths
        return A_arcs, B_arcs, C_arcs, A_effective_TPs, B_effective_TPs, C_effective_TPs

  