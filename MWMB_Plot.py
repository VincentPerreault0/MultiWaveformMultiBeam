import matplotlib.pyplot as plt

"""
Returns the numbers of subplots on the y and x axes respectively where
  'nb' is the total number of desired subplots to be plotted in the same figure.
"""
def get_subplot_nb_y_and_nb_x(nb):
  if nb == 0:
    nb_y = 0
    nb_x = 0
  elif nb <= 2:
    nb_y = 1
    nb_x = 2
  elif nb <= 3:
    nb_y = 1
    nb_x = 3
  elif nb <= 6:
    nb_y = 2
    nb_x = 3
  elif nb <= 8:
    nb_y = 2
    nb_x = 4
  elif nb <= 12:
    nb_y = 3
    nb_x = 4
  elif nb <= 15:
     nb_y = 3
     nb_x = 5
  elif nb <= 20:
    nb_y = 4
    nb_x = 5
  elif nb <= 24:
    nb_y = 4
    nb_x = 6
  else:
    nb_y = 4
    nb_x = 7
  return nb_y,nb_x

"""
Plots one or multiple ImpTree/ImpArborescence/Network objects in the same figure where
  'list_treeArbSol' is a list of ImpTree/ImpArborescence/Network objects;
  'nb_y' (optional) is the number of subplots on the y axis;
  'nb_x' (optional) is the number of subplots on the x axis;
  'list_iter' (optional) is a list of iterations numbers, one for each of the ImpTree/ImpArborescence/Network objects;
  'fig_axs' (optional) is a tuple containing the figure object and a list of axes objects in which to plot the subplots, if they are given;
  'fig_number' (optional) is a number or identifier for the figure object to be created, if none is given;
  'simp_values' (optional) is a list of objective/pseudo-objective values, one for each of the ImpTree/ImpArborescence/Network objects;
  'MHs' (optional) is a list of master hubs, one for each of the ImpTree objects;
  'added_edges' (optional) is a list of added edges, one for each of the ImpTree objects;
  'dropped_edges' (optional) is a list of dropped edges, one for each of the ImpTree objects;
  'titles' (optional) is a list of titles, one for each of the ImpTree/ImpArborescence/Network objects;
"""
def subplot(list_treeArbSol, nb_y=-1, nb_x=-1, list_iter=-1, fig_axs=(-1,-1), fig_number=-1, simp_values=[], MHs=[], added_edges=[], dropped_edges=[], titles=-1):
  if nb_y==-1 and nb_x==-1:
    nb_y, nb_x = get_subplot_nb_y_and_nb_x(len(list_treeArbSol))
  has_list_iter = list_iter!=-1
  has_titles = titles != -1
  if fig_axs == (-1,-1):
    if fig_number < 0:
      fig, axs = plt.subplots(nb_y, nb_x, gridspec_kw={"left":0.075, "right":0.925, "bottom":0.075, "top":0.925})
    else:
      fig, axs = plt.subplots(nb_y, nb_x, num=fig_number, gridspec_kw={"left":0.075, "right":0.925, "bottom":0.075, "top":0.925})
  else:
    fig, axs = fig_axs
  for y in range(nb_y):
    for x in range(nb_x):
      i = nb_x*y + x
      if nb_y == 1:
        if nb_x == 1:
          if isinstance(axs, list):
            ax = axs[0]
          else:
            ax = axs
        else:
          ax = axs[x]
      else:
        if nb_x == 1:
          ax = axs[y]
        else:
          ax = axs[y][x]
      ax.clear()
      if i < len(list_treeArbSol):
        if has_list_iter or has_titles:
          pretitle = ""
          if has_list_iter:
            pretitle += str(list_iter[i])+") "
          if has_titles:
            pretitle += titles[i]
          if len(simp_values) > 0:
            pretitle += "f="+str(round(100*simp_values[i])/100)+"; "
          if len(MHs) == 0:
            list_treeArbSol[i].show(ax,pretitle)
          else:
            if len(added_edges) == 0 and len(dropped_edges) == 0:
              list_treeArbSol[i].show(ax,pretitle, MH=MHs[i])
            else:
              if len(added_edges) == len(list_treeArbSol)-1:
                if i == 0:
                  if i < len(dropped_edges):
                    list_treeArbSol[i].show(ax,pretitle, MH=MHs[i], dropped_edges=[dropped_edges[i]])
                  else:
                    list_treeArbSol[i].show(ax,pretitle, MH=MHs[i])
                else: 
                  if i < len(dropped_edges):
                    list_treeArbSol[i].show(ax,pretitle, MH=MHs[i], added_edges=[added_edges[i-1]], dropped_edges=[dropped_edges[i]])
                  else:
                    list_treeArbSol[i].show(ax,pretitle, MH=MHs[i], added_edges=[added_edges[i-1]])
              else:
                if i < len(dropped_edges):
                  list_treeArbSol[i].show(ax,pretitle, MH=MHs[i], added_edges=[added_edges[i]], dropped_edges=[dropped_edges[i]])
                else:
                  list_treeArbSol[i].show(ax,pretitle, MH=MHs[i], added_edges=[added_edges[i]])
        else:
          if len(MHs) == 0:
            list_treeArbSol[i].show(ax)
          else:
            if len(added_edges) == 0 and len(dropped_edges) == 0:
              list_treeArbSol[i].show(ax, MH=MHs[i])
            else:
              if len(added_edges) == len(list_treeArbSol)-1:
                if i == 0:
                  if i < len(dropped_edges):
                    list_treeArbSol[i].show(ax, MH=MHs[i], dropped_edges=[dropped_edges[i]])
                  else:
                    list_treeArbSol[i].show(ax, MH=MHs[i])
                else: 
                  if i < len(dropped_edges):
                    list_treeArbSol[i].show(ax, MH=MHs[i], added_edges=[added_edges[i-1]], dropped_edges=[dropped_edges[i]])
                  else:
                    list_treeArbSol[i].show(ax, MH=MHs[i], added_edges=[added_edges[i-1]])
              else:
                if i < len(dropped_edges):
                  list_treeArbSol[i].show(ax, MH=MHs[i], added_edges=[added_edges[i]], dropped_edges=[dropped_edges[i]])
                else:
                  list_treeArbSol[i].show(ax, MH=MHs[i], added_edges=[added_edges[i]])
  if nb_y == 1:
    if nb_x == 1:
      if isinstance(axs, list):
        axs[0].label_outer()
      else:
        axs.label_outer()
    else:
      for ax in axs:
        ax.label_outer()
  else:
    if nb_x == 1:
      for ax in axs:
        ax.label_outer()
    else:
      for axss in axs:
        for ax in axss:
          ax.label_outer()
  fig.canvas.draw()
  if fig_number < 0 and fig_axs == (-1,-1):
    plt.show()
  return fig, axs

"""
Prints a line in the console and in open text files where
  'line' is the line;
  'verbose' is whether to print in the console (True) or not (False);
  'files' (optional) is a list of open text files in which to print.
"""
def print_log(line, verbose, files=[]):
  if verbose:
    print(line)
  for file in files:
    file.write(str(line)+"\n")





    