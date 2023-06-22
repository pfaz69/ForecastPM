import os
import json
from matplotlib import pyplot

def create_dir_in_case(dir_):

    if not os.path.exists(dir_):
        os.makedirs(dir_)




def plot_graphs_llist(dir_data, file_name_stub, list_graphs, design = None):
    if design != None:
        dir_data = dir_data + "_" + design
    create_dir_in_case(dir_data)
    n = len(list_graphs)
    for i in range(n):
        pyplot.figure()
        pyplot.plot(list_graphs[i][0])

        if n == 1:
            filename = dir_data + "/" + file_name_stub + ".png"
        else:
            filename = dir_data + "/" + file_name_stub + "_" + str(i) + ".png"

        pyplot.savefig(filename)
        pyplot.close('all')

def plot_graphs_stacked(dir_, file_name_stub, design, graphs, highlight_id = None):
    dir_data = dir_ + "_" + design
    create_dir_in_case(dir_data)

    graph_ids = list(range(graphs.shape[1]))

   # Plot each column
    i = 1
    pyplot.figure()
    for i_val in list(graph_ids):
        pyplot.subplot(len(graph_ids), 1, i)
        pyplot.plot(graphs[:, 0], color = 'orange')
        if i_val > 0:
            if highlight_id == i_val:
                color_graph = 'royalblue'
            else:
                color_graph = 'seagreen'                
            pyplot.plot(graphs[:, i_val], color = color_graph)
        i += 1

    #pyplot.show()
    filename = dir_data + "/" + file_name_stub + ".png"
    pyplot.savefig(filename)
    pyplot.close('all')


def plot_bars(dir_, file_name_stub, design, indexes, cams, pers, model, do_log = True):
    # This function is application specific
    dir_data = dir_ + "_" + design
    create_dir_in_case(dir_data)
    pyplot.clf()
    pyplot.bar(indexes, cams, color = "orange")
    pyplot.bar(12, pers, color = "red")
    pyplot.bar(14, model, color = "blue")
    x1, y1 = [1, 14], [model, model]
    x2, y2 = [1, 12], [pers, pers]
    pyplot.plot(x1, y1, color = "blue", marker = 'o')
    pyplot.plot(x2, y2, color = "red", marker = 'o')   
    filename = dir_data + "/" + file_name_stub + ".png"
    pyplot.savefig(filename)
    pyplot.close('all')
    
    # Log
    if do_log:
        logfilename = dir_data + "/" + file_name_stub + ".txt"
        out_dict = {"cams":cams, "pers":pers, "model":model}
        with open(logfilename, 'w') as f:
            json.dump(out_dict, f, indent = 4, sort_keys=True)

    



def plot_multi_graphs(dir_, file_name_stub, design, graphs, labels, do_log = True, alpha = 1):
    dir_data = dir_ + "_" + design
    create_dir_in_case(dir_data)
    pyplot.clf()
    pyplot.grid(True)
    #pyplot.ylim(0, 1.4)
    for graph, label in zip(graphs, labels):
        pyplot.plot(graph, label=label, alpha=alpha, linewidth=1.0)
        pyplot.legend()
    #pyplot.show()
    filename = dir_data + "/" + file_name_stub + ".png"
    pyplot.savefig(filename)
    pyplot.close('all')
    
    # Log
    if do_log:
        logfilename = dir_data + "/" + file_name_stub + ".txt"
        for i, graph in enumerate(graphs):
            graphs[i] = graph.tolist()
        out_dict = {"graphs":graphs, "labels":labels}
        with open(logfilename, 'w') as f:
            json.dump(out_dict, f, indent = 4, sort_keys=True)
