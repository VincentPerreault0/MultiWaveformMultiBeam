import pickle
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, cos, log10, log2, sqrt
from scipy.spatial import distance_matrix
from scipy.io import loadmat

"""
EXPLICIT NODES ARE IDENTIFIED BY AN INDEX BETWEEN 0 AND 'nb_nodes'-1 INCLUSIVELY.
"""



"""
FreqMHz
-------
Static class containing the possible frequency values and their associated channels.

freq_index = 0 --> 2000 MHz (channel_index = -2 --> band 3+)
freq_index = 1 --> 2400 MHz (channel_index = -2 --> band 3+)
freq_index = 2 --> 4500 MHz (channel_index = -3 --> band 4)
freq_index = 3 --> 5000 MHz (channel_index = -3 --> band 4)

freq_index = -2 --> 2200 MHz (band 3+ average)
freq_index = -3 --> 4500 MHz (band 4 average)
"""
class FreqMHz:
  nb = 4
  nb_channels = 2

  """
  Returns the frequency in MHz associated with the 'freq_index'.
  """
  @staticmethod
  def from_index(freq_index):
    if freq_index == 0:
      #return 1800
      return 2000
    elif freq_index == 1:
      return 2400
    elif freq_index == 2:
      return 4500
    elif freq_index == 3:
      return 5000
    elif freq_index == -2: # Band 3 average
      return 2200
    elif freq_index == -3: # Band 4 "average"
      return 4500
    else:  # average
      return 3500

  """
  Returns the channel index associated with the 'freq_index'.
  """
  @staticmethod
  def channel_from_index(freq_index):
    if freq_index < -1:
      return freq_index
    if freq_index < 0:
      return -1
    if freq_index <= 1:
      return -2
    else:
      return -3
  
  """
  Returns the color associated with the 'freq_index'.
  """
  @staticmethod
  def color_from_index(freq_index):
    if freq_index == -1:
      return 'k'
    elif freq_index == 0 or freq_index == -2:
      return 'blue'
    elif freq_index == 1:
      return 'darkblue'
    elif freq_index == 2 or freq_index == -3:
      return 'red'
    else:
      return 'darkred'


"""
ProbParameters
-------
Object containing all the parameters of the problem that are independent of the problem instance:

  .mixed = True {boolean} --> whether we consider PTP, PMP and Mesh (True) or only PTP and PMP (False)
  .multibeam = True {boolean} --> whether we consider multi-beam antennas (True) or single-beam antennas (False)
  .noise_powers_dBm = [-174 + 10*log10(20*1e6) + 10] {list of 1 float} --> receiver noise power in dBm assuming a 20 Mhz bandwidth and 10 dB noise figure
  .max_nb_nodes_per_PTM_connection = 10 {int} --> maximum number of connected nodes in a single PMP connection
  .nb_freqs_per_channel = 2 {int} --> number of frequencies per channels (either 2 or 1)
  .obj_scenarios = [1/13, 4/13, 8/13] {list of 3 floats} --> the respective weights of the three traffic scenarios A, B and C
  .SINR_threshold_throughput_table = [(2,6.5),(5,13),(9,19.5),(11,26),(15,39),(18,52),(20,58.5),(25,65),(29,78)] {list of tuples of (mininimum SINR,throughput)}
    --> the throughput lookup table used for direct throughput evaluation, if = [] then Shannon-Hartley theorem is used
  .PTP_coeff = 2 {float} --> the multiplicative coefficient of the direct throughput for PTP connections (relative to PMP)
  .Mesh_coeff = 0.5 {float} --> the multiplicative coefficient of the direct throughput for Mesh connections (relative to PMP)
  .mesh_high_traffic_routes_prioritization = 0 {float} --> referred to as xi in thesis, parameter for complexity of mesh routing algorithm (0: naive but fast, 1: through but slow)
  .mesh_collisions = False {boolean} --> whether mesh collisions are modeled directly (True) or via the Mesh_coeff (False)
"""
class ProbParameters:
  """
  Initializes a ProbParameters object where
    'mixed' (optional) is whether we consider PTP, PMP and Mesh (True) or only PTP and PMP (False);
    'multibeam' (optional) is whether we consider multi-beam antennas (True) or single-beam antennas (False);
    'max_nb_nodes_per_PTM_connection' (optional) is the maximum number of connected nodes in a single PMP connection;
    'nb_freqs_per_channel' (optional) is the number of frequencies per channels (either 2 or 1);
    'obj_scenarios' (optional) is the respective weights of the three traffic scenarios A, B and C;
    'SINR_threshold_throughput_table' (optional) is the throughput lookup table used for direct throughput evaluation, if = [] then Shannon-Hartley theorem is used;
    'PTP_coeff' (optional) is the multiplicative coefficient of the direct throughput for PTP connections (relative to PMP);
    'Mesh_coeff' (optional) is the multiplicative coefficient of the direct throughput for Mesh connections (relative to PMP);
    'mesh_high_traffic_routes_prioritization' (optional) is parameter for complexity of mesh routing algorithm (0: naive but fast, 1: through but slow) (xi in thesis);
    'mesh_collisions' (optional) is whether mesh collisions are modeled directly (True) or via the Mesh_coeff (False).
  """
  def __init__(self, mixed = True, multibeam = True, max_nb_nodes_per_PTM_connection = 10, nb_freqs_per_channel = 2, obj_scenarios = [1/13, 4/13, 8/13], SINR_threshold_throughput_table = [(2,6.5),(5,13),(9,19.5),(11,26),(15,39),(18,52),(20,58.5),(25,65),(29,78)], PTP_coeff = 2, Mesh_coeff = 0.5, mesh_high_traffic_routes_prioritization = 0, mesh_collisions = False):
    self.mixed = mixed # boolean
    self.multibeam = multibeam # boolean
    self.noise_powers_dBm = [-174 + 10*log10(20*1e6) + 10] # assuming 20Mhz Bandwidth and 10 dB Noise Figure
    self.max_nb_nodes_per_PTM_connection = max_nb_nodes_per_PTM_connection
    self.nb_freqs_per_channel = nb_freqs_per_channel # int
    self.obj_scenarios = obj_scenarios # list of weights for the 3 traffic scenarios [Any-->Any, All-->MH, All<->All]
    self.SINR_threshold_throughput_table = SINR_threshold_throughput_table # list of tuples (minimum_SINR_threshold,throughput)
    self.PTP_coeff = PTP_coeff # float
    self.Mesh_coeff = Mesh_coeff # float
    self.mesh_high_traffic_routes_prioritization = mesh_high_traffic_routes_prioritization # float    1 : one at a time (perfect),  0 : all at once (A),  0.5 : by groups of size sqrt(number of mesh edges)
    self.mesh_collisions = mesh_collisions # bool

  """
  Defines equality of ProbParameters objects.
  """
  def __eq__(self, other):
    if isinstance(other, ProbParameters):
        return self.mixed == other.mixed and self.multibeam == other.multibeam
    else:
      return False

  """
  How to print a ProbParameters object.
  """
  def __repr__(self):
    string = "Problem Parameters :\n\t"
    if self.mixed:
      string += "Mixed Topology\n\t"
    else:
      string += "Tree Topology\n\t"
    if self.multibeam:
      string += "Multi-Beam Antennas\n\t"
    else:
      string += "Single Beam Antennas\n\t"
    if self.nb_nodes <= 1:
      string += str(self.nb_nodes)+" Node\n\t"
    else:
      string += str(self.nb_nodes)+" Nodes\n\t"
    return string
  
  """
  Returns the maximum gain of the antenna (at the center if the beam) in dB for the 'freq_index' and for 'nb_activ_beams' activated beams.

  In the multi-beam case, 'nb_activ_beams' represents the number of activated beams (with 0 --> omni-mode).
  In the single-beam case, 'nb_activ_beams' represents the type of single-beam antenna:
    0 --> omni-directional antenna;
    1 --> parabolic antenna;
    2 --> panel antenna;
    3 --> sector antenna.
    """
  def get_max_gain_dB(self, freq_index, nb_activ_beams=-1):
    if self.multibeam:
      if nb_activ_beams == 0:
        #return 5 + 3.5*(FreqMHz.from_index(freq_index)-1300)/3700
        if FreqMHz.channel_from_index(freq_index) == -2:
          return 4
        else:
          return 6
      else:
        #return 7 + 8*(FreqMHz.from_index(freq_index)-1300)/3700 - 2*(nb_activ_beams-1)
        if FreqMHz.channel_from_index(freq_index) == -2:
          return 13 - 10*log10(nb_activ_beams)
        else:
          return 15 - 10*log10(nb_activ_beams)
    else:
      if nb_activ_beams == 0:
        if FreqMHz.channel_from_index(freq_index) == -2:
          return 6
        else:
          return 8
      elif nb_activ_beams == 1:
        if FreqMHz.channel_from_index(freq_index) == -2:
          return 21
        else:
          return 28
      elif nb_activ_beams == 2:
        if FreqMHz.channel_from_index(freq_index) == -2:
          return 16
        else:
          return 22
      else:
        if FreqMHz.channel_from_index(freq_index) == -2:
          return 14
        else:
          return 15

  """
  Returns a list of possible pattern exponents.

  In the multi-beam case, the list corresponds to the possible frequency indices in the order [0, 1, 2, 3, -3, -2, -1].
  In the single-beam case, the list contains 3 lists (for parabolic, panel and sector respectively) which correspond to the possible frequency indices as in multi-beam.
  """
  def get_pattern_exponents(self):
    if self.multibeam:
      beam_widths_rad = (np.pi/180)*np.array([self.get_3dB_beam_width_deg(freq_index) for freq_index in range(FreqMHz.nb)])
      exponents = -3 / (20*np.log10(np.cos(beam_widths_rad/2)))
      return exponents.tolist()+[exponents[2],np.mean(exponents[0:2]),np.mean(exponents)] # (FreqMhz.nb+3)
    else:
      beam_widths_rad = (np.pi/180)*np.array([self.get_3dB_beam_width_deg(freq_index) for freq_index in range(FreqMHz.nb)])
      beam_widths_rad = beam_widths_rad.T
      exponents = -3 / (20*np.log10(np.cos(beam_widths_rad/2)))
      return np.concatenate((exponents,np.reshape(exponents[:,2], (3,1)),np.reshape(np.mean(exponents[:,0:2],axis=1), (3,1)),np.reshape(np.mean(exponents,axis=1), (3,1))), axis=1).tolist() # (3, FreqMhz.nb+3)

  """
  Returns the 3 dB beam width (half-power beam width) in degrees.

  In the multi-beam case, the output is a single angle.
  In the single-beam case, the output is a list of 3 angles (for parabolic, panel and sector respectively).
  """
  def get_3dB_beam_width_deg(self, freq_index):
    if self.multibeam:
      #return 70 - 40*(FreqMHz.from_index(freq_index)-1300)/3700
      if FreqMHz.channel_from_index(freq_index) == -2:
        return 60
      else:
        return 50
    else:
      if FreqMHz.channel_from_index(freq_index) == -2:
        return [20, 40, 110]
      else:
        return [8, 16, 90]
  
  """
  Returns the signal strength in dB of the uplink antenna where
    'angle_from_to_rad' is the angle in radian from the uplink node to the downlink node;
    'pattern_exponents' are the pattern exponents returned by get_pattern_exponents(...);
    'ul_ant_nb_beams' is the number of activated beams of the uplink antenna (either 1 or 0 --> omni);
    'ul_ant_align' is the alignment of the uplink antenna in radian;
    'freq_idx' is the frequency index;
    'gain_freq_idx' (optional) is the frequency index used for the maximum gain, if different than the one used for the exponent.
  """
  def get_ul_dB_signal_strength(self, angle_from_to_rad, pattern_exponents, ul_ant_nb_beams, ul_ant_align, freq_idx, gain_freq_idx=-10):
    if gain_freq_idx == -10:
      gain_freq_idx = freq_idx
    if ul_ant_nb_beams == 1:
      v1_angle_mismatch = angle_from_to_rad - ul_ant_align
      if cos(v1_angle_mismatch) <= 0:
        return float('-inf')
      else:
        if self.multibeam:
          return self.get_max_gain_dB(gain_freq_idx,1)+20*pattern_exponents[freq_idx]*log10(max(cos(v1_angle_mismatch),0))
        else:
          return self.get_max_gain_dB(gain_freq_idx,1)+20*pattern_exponents[0][freq_idx]*log10(max(cos(v1_angle_mismatch),0))
    else:
      return self.get_max_gain_dB(gain_freq_idx,0)
  
  """
  Returns the signal strength in linear of the downlink antenna where
    'angle_from_to_rad' is the angle in radian from the downlink node to the uplink node;
    'pattern_exponents' are the pattern exponents returned by get_pattern_exponents(...);
    'dl_ant_beams' is the set of active beams (see get_max_gain_dB(...) for details);
    'dl_ant_align' is the alignment of the downlink antenna in radian;
    'freq_idx' is the frequency index;
    'gain_freq_idx' (optional) is the frequency index used for the maximum gain, if different than the one used for the exponent.
  """
  def get_dl_lin_signal_strength(self, angle_from_to_rad, pattern_exponents, dl_ant_beams, dl_ant_align, freq_idx, gain_freq_idx=-10):
    if gain_freq_idx == -10:
      gain_freq_idx = freq_idx
    if len(dl_ant_beams) > 0:
      sig_strength = 0
      if self.multibeam:
        for activ_beam in dl_ant_beams:
          v1_angle_mismatch = angle_from_to_rad - (dl_ant_align + (np.pi/12)*activ_beam)
          sig_strength += pow(10,self.get_max_gain_dB(gain_freq_idx,len(dl_ant_beams))/10)*pow(max(cos(v1_angle_mismatch),0),2*pattern_exponents[freq_idx])
      else:
        v1_angle_mismatch = angle_from_to_rad - dl_ant_align
        sig_strength += pow(10,self.get_max_gain_dB(gain_freq_idx,len(dl_ant_beams))/10)*pow(max(cos(v1_angle_mismatch),0),2*pattern_exponents[len(dl_ant_beams)-1][freq_idx])
      return sig_strength
    else:
      return pow(10,self.get_max_gain_dB(gain_freq_idx,0)/10)

  """
  Returns the direct throughput of an ideal connection with
    an averaged 'dist_dB' (sum of path loss and fade margin),
    a 'nb_beams' of either 1 (1 activated beam/parabolic) or 0 (omni-mode/omni-directional antenna) for both antennas,
    an averaged freq_index (= -1),
    perfect alignment for both antennas,
    and a PMP connection.
  """
  def get_perfect_signal_TP(self, dist_dB, nb_beams=1):
    SINR = 30 + 2*self.get_max_gain_dB(-1,nb_beams) - dist_dB - self.noise_powers_dBm[0]
    if len(self.SINR_threshold_throughput_table) > 0:
      for table_idx in range(len(self.SINR_threshold_throughput_table)):
        if table_idx == 0:
          if SINR < self.SINR_threshold_throughput_table[table_idx][0]:
            return 0
        if table_idx < len(self.SINR_threshold_throughput_table)-1:
          if SINR < self.SINR_threshold_throughput_table[table_idx+1][0]:
            return self.SINR_threshold_throughput_table[table_idx][1]
        else:
          return self.SINR_threshold_throughput_table[table_idx][1]
    else:
      return 20 * log2(1 + pow(10,SINR/10))


"""
ProbInstance
-------
Object containing all the parameters of the problem instance.

  .nb_nodes {int} --> number of nodes of the instance
  .coordinates_km {numpy array of floats of shape ('nb_nodes', 2)} --> (x,y) coordinates in km of the nodes
  .angles_from_to_rad {numpy array of floats of shape ('nb_nodes', 'nb_nodes')} --> angle_uv in radian from node u in the direction of node v (relative to the azimuth)
  .path_losses_dB {numpy array of floats of shape ('nb_nodes', 'nb_nodes', 7)} --> loss_uvf in dB between nodes u and v for freq_index f
  .fade_margins_dB {numpy array of floats of shape ('nb_nodes', 'nb_nodes', 7)} --> margin_uvf in dB between nodes u and v for freq_index f
"""
class ProbInstance:
  """
  Initializes a ProbInstance object where
    'coordinates_km' is the (x,y) coordinates in km of the nodes;
    'path_losses_dB' is the path loss in dB between every pair of nodes for every frequency index;
    'fade_margins_dB' is the fade margin in dB between every pair of nodes for every frequency index.
  """
  def __init__(self, coordinates_km, path_losses_dB, fade_margins_dB):
    self.coordinates_km = coordinates_km # (nb_nodes,2)
    nb_nodes = coordinates_km.shape[0]
    self.nb_nodes = nb_nodes # int
    angles_from_to_rad = np.zeros((nb_nodes,nb_nodes))
    for i in range(nb_nodes):
      angles = np.arctan2(coordinates_km[:,1]-coordinates_km[i,1], coordinates_km[:,0]-coordinates_km[i,0])
      angles[angles < 0] += 2*np.pi
      angles_from_to_rad[i,:] = angles
    self.angles_from_to_rad = angles_from_to_rad # (nb_nodes,nb_nodes)
    path_losses_dB = np.concatenate((path_losses_dB,np.reshape(path_losses_dB[:,:,2],(nb_nodes,nb_nodes,1)),np.reshape(np.mean(path_losses_dB[:,:,0:2],axis=2),(nb_nodes,nb_nodes,1)),np.reshape(np.mean(path_losses_dB,axis=2),(nb_nodes,nb_nodes,1))), axis=2)
    self.path_losses_dB = path_losses_dB # (nb_nodes,nb_nodes,FreqMhz.nb+3) avec Freqs = [0,1,2,3,-3,-2,-1]
    fade_margins_dB = np.concatenate((fade_margins_dB,np.reshape(fade_margins_dB[:,:,2],(nb_nodes,nb_nodes,1)),np.reshape(np.mean(fade_margins_dB[:,:,0:2],axis=2),(nb_nodes,nb_nodes,1)),np.reshape(np.mean(fade_margins_dB,axis=2),(nb_nodes,nb_nodes,1))), axis=2)
    self.fade_margins_dB = fade_margins_dB # (nb_nodes,nb_nodes,FreqMhz.nb+3) avec Freqs = [0,1,2,3,-3,-2,-1] 

  """
  Defines equality of ProbInstance objects.
  """
  def __eq__(self, other):
    if isinstance(other, ProbInstance):
        return (self.coordinates_km == other.coordinates_km).all() and (self.path_losses_dB == other.path_losses_dB).all() and (self.fade_margins_dB == other.fade_margins_dB).all()
    else:
      return False

  """
  Prints a ProbInstance object.
  """
  def __repr__(self):
    string = "Problem Instance :\n\t"
    if self.nb_nodes <= 1:
      string += str(self.nb_nodes)+" Node\n\t"
    else:
      string += str(self.nb_nodes)+" Nodes\n\t"
    string += "Coordinates : "+str(self.coordinates_km)+"\n\t"
    string += "Path Losses : "+str(np.moveaxis(self.path_losses_dB,2,0))+"\n\t"
    string += "Fade Margins : "+str(np.moveaxis(self.fade_margins_dB,2,0))
    return string

  """
  Plots a ProbInstance as a set of nodes where
    'ax' (optional) is the matplotlib axes object in which to plot;
    'title' (optional) is the title string of the plot;
    'show_PL_FM' (optional) decides whether to show the path losses and fade margins as edges of varying opacity.
  """
  def show(self, ax=-1, title="", show_PL_FM=False):
    extra_border = 1
    line_width = 1
    has_axis = ax != -1
    if not has_axis:
      fig = plt.figure(1)
      ax = fig.add_axes([0.075,0.075,0.85,0.85])
    coordinates_km = self.coordinates_km
    if show_PL_FM:
      PL_FM_dB = np.reshape(self.path_losses_dB[:,:,-1] + self.fade_margins_dB[:,:,-1],(self.nb_nodes,self.nb_nodes))
      PL_FM_dB -= np.min(PL_FM_dB)
      PL_FM_dB /= np.max(PL_FM_dB)
    ax.set_aspect('equal')
    xlim0 = -extra_border
    xlim1 = np.max(coordinates_km[:,0])+extra_border
    ylim0 = -extra_border
    ylim1 = np.max(coordinates_km[:,1])+extra_border
    ax.set_xlim(xlim0,xlim1)
    ax.set_ylim(ylim0,ylim1)
    max_size = max(xlim1-xlim0,ylim1-ylim0)
    ax.set_title(title)
    if show_PL_FM:
      for v1 in range(self.nb_nodes):
        for v2 in range(v1+1,self.nb_nodes):
          dx = coordinates_km[v2,0] - coordinates_km[v1,0]
          dy = coordinates_km[v2,1] - coordinates_km[v1,1]
          radius = sqrt(pow(dx,2) + pow(dy,2))
          ratio_beg = (7/400*max_size)/radius
          ratio_end = (radius - 7/400*max_size)/radius
          ax.plot([coordinates_km[v1,0]+ratio_beg*dx,coordinates_km[v1,0]+ratio_end*dx],[coordinates_km[v1,1]+ratio_beg*dy,coordinates_km[v1,1]+ratio_end*dy],lw=line_width,c=(1,0,0, PL_FM_dB[v1,v2]))
    ax.scatter(coordinates_km[:,0].tolist(),coordinates_km[:,1].tolist(), s=30, c='k')
    for exp_node in range(self.nb_nodes):
      ax.text(coordinates_km[exp_node,0],coordinates_km[exp_node,1] - 6/100*max_size,str(exp_node),color='gray',ha='center')
    if not has_axis:
      fig.canvas.draw()
      plt.show()

  """
  Creates a random ProbInstance where
    'nb_nodes' is the number of nodes;
    'avg_dist_km' is the average inter-node distance in km;
    'min_possible_dist_km' (optional) is the minimum inter-node distance in km;
    'max_possible_dist_km' (optional) is the maximum inter-node distance in km;
    'verbose' (optional) is whether to print intermediary steps (True) or not (False).
  """
  @staticmethod
  def create_random(nb_nodes = 10, verbose = False, min_possible_dist_km = -1, max_possible_dist_km = -1, avg_dist_km = 10):
    if min_possible_dist_km < 0:
      min_possible_dist_km = avg_dist_km / 5
    if max_possible_dist_km <= 0:
      max_possible_dist_km = avg_dist_km * 5
    min_dist_km = -float('inf')
    max_dist_km = float('inf')
    while min_dist_km < min_possible_dist_km or max_dist_km > max_possible_dist_km:
      if verbose:
        print("Minimum distance = "+str(min_dist_km))
        print("Maximum distance = "+str(max_dist_km))
      xs_km = np.sqrt(np.random.uniform(size=nb_nodes))*np.cos(2*np.pi*np.random.uniform(size=nb_nodes))
      ys_km = np.sqrt(np.random.uniform(size=nb_nodes))*np.sin(2*np.pi*np.random.uniform(size=nb_nodes))
      xys_km = np.stack([xs_km, ys_km], axis=1)

      distances_km = np.triu(distance_matrix(xys_km,xys_km),k=1)
      distances_km = distances_km[np.nonzero(distances_km)]
      avg_dist_ratio = avg_dist_km / np.mean(distances_km)

      xys_km = avg_dist_ratio * xys_km
      xys_km[:,0] = xys_km[:,0] - np.min(xys_km[:,0])
      xys_km[:,1] = xys_km[:,1] - np.min(xys_km[:,1])

      distances_matrix_km = distance_matrix(xys_km,xys_km)
      distances_km = np.triu(distances_matrix_km,k=1)
      distances_km = distances_km[np.nonzero(distances_km)]
      min_dist_km = np.min(distances_km)
      max_dist_km = np.max(distances_km)
    
    distances_matrix_km = np.ceil(distances_matrix_km)
    distances_matrix_km = distances_matrix_km.astype(int)
    pls_dB = np.zeros((nb_nodes,nb_nodes,FreqMHz.nb))
    fms_dB = np.zeros((nb_nodes,nb_nodes,FreqMHz.nb))
    for i in range(nb_nodes):
      for j in range(i+1,nb_nodes):
        dictmat = {}
        loadmat("PL_FM_ECDFs/PL_FM_atKM"+str(max(2,distances_matrix_km[i,j]))+".mat", mdict=dictmat)
        prob_PL = np.random.uniform(size=1)[0]
        prob_FM = np.random.uniform(size=1)[0]
        for freq_index in range(FreqMHz.nb):
          ecdf_PL = np.asarray(dictmat["ecdf_PL"][0,freq_index][0,0][2])
          ecdf_PL = np.stack([np.reshape(np.asarray(dictmat["ecdf_PL"][0,freq_index][0,0][1]),(ecdf_PL.shape[0])),np.reshape(np.asarray(dictmat["ecdf_PL"][0,freq_index][0,0][2]),(ecdf_PL.shape[0]))],axis=1)
          prob_index = np.where(ecdf_PL[:,1]<=prob_PL)[-1][-1]
          pl = ecdf_PL[prob_index,0]
          pls_dB[i,j,freq_index] = pl
          pls_dB[j,i,freq_index] = pl
          ecdf_FM = np.asarray(dictmat["ecdf_FM"][0,freq_index][0,0][2])
          ecdf_FM = np.stack([np.reshape(np.asarray(dictmat["ecdf_FM"][0,freq_index][0,0][1]),(ecdf_FM.shape[0])),np.reshape(np.asarray(dictmat["ecdf_FM"][0,freq_index][0,0][2]),(ecdf_FM.shape[0]))],axis=1)
          prob_index = np.where(ecdf_FM[:,1]<=prob_FM)[-1][-1]
          fm = ecdf_FM[prob_index,0]
          fms_dB[i,j,freq_index] = fm
          fms_dB[j,i,freq_index] = fm
    return ProbInstance(xys_km,pls_dB,fms_dB)

  """
  Creates a random ProbInstance in multiple scales of distance where
    'nb_nodes' is the number of nodes;
    'avg_dist_km' is the average inter-node distance in km of the reference scale;
    'min_possible_dist_km' (optional) is the minimum inter-node distance in km of the reference scale;
    'max_possible_dist_km' (optional) is the maximum inter-node distance in km of the reference scale;
    'coeffs' (optional) is a list of desired scales relative to the reference scale;
    'verbose' (optional) is whether to print intermediary steps (True) or not (False).
  """
  @staticmethod
  def create_mult_random(nb_nodes = 10, verbose = False, min_possible_dist_km = -1, max_possible_dist_km = -1, avg_dist_km = 10, coeffs = [0.25,0.5,1,2]):
    if min_possible_dist_km < 0:
      min_possible_dist_km = avg_dist_km / 5
    if max_possible_dist_km <= 0:
      max_possible_dist_km = avg_dist_km * 5

    raw_xs_km = np.sqrt(np.random.uniform(size=nb_nodes))*np.cos(2*np.pi*np.random.uniform(size=nb_nodes))
    raw_ys_km = np.sqrt(np.random.uniform(size=nb_nodes))*np.sin(2*np.pi*np.random.uniform(size=nb_nodes))
    raw_xys_km = np.stack([raw_xs_km, raw_ys_km], axis=1)

    distances_km = np.triu(distance_matrix(raw_xys_km,raw_xys_km),k=1)
    distances_km = distances_km[np.nonzero(distances_km)]
    avg_dist_ratio = avg_dist_km / np.mean(distances_km)

    xys_km = avg_dist_ratio * raw_xys_km
    xys_km[:,0] = xys_km[:,0] - np.min(xys_km[:,0])
    xys_km[:,1] = xys_km[:,1] - np.min(xys_km[:,1])

    distances_matrix_km = distance_matrix(xys_km,xys_km)
    distances_km = np.triu(distances_matrix_km,k=1)
    min_dist_km = np.min(distances_km[np.nonzero(distances_km)])
    max_dist_km = np.max(distances_km)

    while min_dist_km < min_possible_dist_km or max_dist_km > max_possible_dist_km:
      if verbose:
        print("Minimum distance = "+str(min_dist_km))
        print("Maximum distance = "+str(max_dist_km))
      
      if min_dist_km < min_possible_dist_km:
        conflicts = np.where(distances_km == min_dist_km)
        conflicts = [(conflicts[0][idx], conflicts[1][idx]) for idx in range(len(conflicts[0]))]
        worst_avg_dist_km = float('inf')
        for conflict in conflicts:
          for v in conflict:
            vec = np.stack([distances_km[v,:], distances_km[:,v].T])
            v_avg_dist_km = np.mean(vec[np.nonzero(vec)])
            if v_avg_dist_km < worst_avg_dist_km:
              worst_avg_dist_km = v_avg_dist_km
              worst_node = v
      else:
        conflicts = np.where(distances_km == max_dist_km)
        conflicts = [(conflicts[0][idx], conflicts[1][idx]) for idx in range(len(conflicts[0]))]
        worst_avg_dist_km = -1
        for conflict in conflicts:
          for v in conflict:
            vec = np.stack([distances_km[v,:], distances_km[:,v].T])
            v_avg_dist_km = np.mean(vec[np.nonzero(vec)])
            if v_avg_dist_km > worst_avg_dist_km:
              worst_avg_dist_km = v_avg_dist_km
              worst_node = v
      raw_xys_km[worst_node,0] = np.sqrt(np.random.uniform(size=1))*np.cos(2*np.pi*np.random.uniform(size=1))
      raw_xys_km[worst_node,1] = np.sqrt(np.random.uniform(size=1))*np.sin(2*np.pi*np.random.uniform(size=1))

      distances_km = np.triu(distance_matrix(raw_xys_km,raw_xys_km),k=1)
      distances_km = distances_km[np.nonzero(distances_km)]
      avg_dist_ratio = avg_dist_km / np.mean(distances_km)

      xys_km = avg_dist_ratio * raw_xys_km
      xys_km[:,0] = xys_km[:,0] - np.min(xys_km[:,0])
      xys_km[:,1] = xys_km[:,1] - np.min(xys_km[:,1])

      distances_matrix_km = distance_matrix(xys_km,xys_km)
      distances_km = np.triu(distances_matrix_km,k=1)
      min_dist_km = np.min(distances_km[np.nonzero(distances_km)])
      max_dist_km = np.max(distances_km)
    

    xys_matrices_km = []
    distances_matrices_km = []
    pls_matrices_dB = []
    fms_matrices_dB = []
    for coeff in coeffs:
      xys_matrices_km.append(coeff*xys_km)
      distances_matrices_km.append(distance_matrix(xys_matrices_km[-1],xys_matrices_km[-1]))
      pls_matrices_dB.append(np.zeros((nb_nodes,nb_nodes,FreqMHz.nb)))
      fms_matrices_dB.append(np.zeros((nb_nodes,nb_nodes,FreqMHz.nb)))
      
    for i in range(nb_nodes):
      for j in range(i+1,nb_nodes):
        prob_PL = np.random.uniform(size=1)[0]
        prob_FM = np.random.uniform(size=1)[0]
        for coeff_idx in range(len(coeffs)):
          dictmat = {}
          loadmat("PL_FM_ECDFs/PL_FM_atKM"+str(max(2,ceil(distances_matrices_km[coeff_idx][i,j])))+".mat", mdict=dictmat)
          for freq_index in range(FreqMHz.nb):
            ecdf_PL = np.asarray(dictmat["ecdf_PL"][0,freq_index][0,0][2])
            ecdf_PL = np.stack([np.reshape(np.asarray(dictmat["ecdf_PL"][0,freq_index][0,0][1]),(ecdf_PL.shape[0])),np.reshape(np.asarray(dictmat["ecdf_PL"][0,freq_index][0,0][2]),(ecdf_PL.shape[0]))],axis=1)
            prob_index = np.where(ecdf_PL[:,1]<=prob_PL)[-1][-1]
            pl = ecdf_PL[prob_index,0]
            pls_matrices_dB[coeff_idx][i,j,freq_index] = pl
            pls_matrices_dB[coeff_idx][j,i,freq_index] = pl
            ecdf_FM = np.asarray(dictmat["ecdf_FM"][0,freq_index][0,0][2])
            ecdf_FM = np.stack([np.reshape(np.asarray(dictmat["ecdf_FM"][0,freq_index][0,0][1]),(ecdf_FM.shape[0])),np.reshape(np.asarray(dictmat["ecdf_FM"][0,freq_index][0,0][2]),(ecdf_FM.shape[0]))],axis=1)
            prob_index = np.where(ecdf_FM[:,1]<=prob_FM)[-1][-1]
            fm = ecdf_FM[prob_index,0]
            fms_matrices_dB[coeff_idx][i,j,freq_index] = fm
            fms_matrices_dB[coeff_idx][j,i,freq_index] = fm
    
    prob_instances = []
    for coeff_idx in range(len(coeffs)):
      prob_instances.append(ProbInstance(xys_matrices_km[coeff_idx],pls_matrices_dB[coeff_idx],fms_matrices_dB[coeff_idx]))
    return prob_instances

  """
  Saves a ProbInstance as a 'filename'.pkl file in the folder "Problem_Instances" which must previously exist.
  """
  def save(self, filename): #overwrites file
    with open("Problem_Instances/"+filename+".pkl", "wb") as output:
      pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
  
  """
  Loads a ProbInstance from the file Problem_Instances/'filename'.pkl.
  """
  @staticmethod
  def load(filename):
    with open("Problem_Instances/"+filename+".pkl", "rb") as input:
      instance = pickle.load(input)
    return instance

  """
  Returns a copy of a ProbInstance object.
  """
  def copy(self):
    return ProbInstance(self.coordinates_km, self.path_losses_dB[:,:,:4], self.fade_margins_dB[:,:,:4], self.mixed, self._multibeam)




