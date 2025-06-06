import matplotlib.pyplot as plt
import networkx as nx
from ..utility.builtin_classes.dicts import update_default_dict

from frame_pipeline.matplotlib_funcs import get_plot_elements
from ..utility.general import get_public_attributes, convert_to_func


def plot_networkx_graph(input_graph, label_attributes=None,
                        pos_func=nx.drawing.nx_agraph.graphviz_layout, pos_kwargs=None,
                        ax=None, attribute_value_formatting_dict=None, default_formatting_dict=None,
                        edge_formatting_dict=None):
    default_pos_kwargs = {"prog": "dot"} if pos_func == nx.drawing.nx_agraph.graphviz_layout else {}
    edge_formatting_dict = {} if edge_formatting_dict is None else edge_formatting_dict

    pos_kwargs = dict(**default_pos_kwargs, **(pos_kwargs if pos_kwargs is not None else {}))
    pos = pos_func(input_graph, **pos_kwargs)
    default_formatting_dict = update_default_dict({}, default_formatting_dict, allow_new_attributes=True)

    if attribute_value_formatting_dict is None:
        attribute_value_formatting_dict = {}

    if label_attributes is None:
        labels = {node: node for node in input_graph.nodes}
    else:
        labels = {node: "\n".join([str(node_attrs[attribute]) for attribute in label_attributes]) for node, node_attrs
                  in dict(input_graph.nodes).items()}

    if ax is None:
        fig, ax = plt.subplots()

    remaining_nodes = set(input_graph.nodes)
    for (attribute, value), formatting in attribute_value_formatting_dict.items():
        valid_nodes = {n for (n, a) in nx.get_node_attributes(input_graph, attribute).items() if a == value}
        remaining_nodes -= valid_nodes
        nx.draw_networkx_nodes(input_graph, pos, nodelist=valid_nodes, ax=ax,
                               **update_default_dict(default_formatting_dict, formatting, allow_new_attributes=True))

    nx.draw_networkx_nodes(input_graph, pos, nodelist=remaining_nodes, ax=ax, **default_formatting_dict)
    nx.draw_networkx_edges(input_graph, pos, input_graph.edges, ax=ax)
    nx.draw_networkx_labels(input_graph, labels=labels, pos=pos, clip_on=False, ax=ax, **edge_formatting_dict)

    return ax


def processing_flow_elements_to_network_graph(processing_flow_elements, filter_only_included_elements=True):
    """

    :param processing_flow_elements:
    :type processing_flow_elements:
    :param filter_only_included_elements:
    :type filter_only_included_elements:
    :return:
    :rtype: nx.DiGraph
    """
    element_set = set(processing_flow_elements)
    network_graph = nx.DiGraph()

    for element in processing_flow_elements:
        public_attributes = get_public_attributes(element)
        network_graph.add_node(element, **public_attributes)

        filtered_upstream_elements = element_set.intersection(
            element.upstream_elements
        ) if filter_only_included_elements else element.upstream_elements
        for upstream_element in filtered_upstream_elements:
            network_graph.add_edge(upstream_element, element)

        filtered_downstream_elements = element_set.intersection(
            element.downstream_elements
        ) if filter_only_included_elements else element.downstream_elements
        for downstream_element in filtered_downstream_elements:
            network_graph.add_edge(element, downstream_element)

    return network_graph


def plot_undirected_pairwise_relationship_graph(input_values, relationship_function, input_labels=tuple(),
                                                outcome_colors=None, outcome_weights=None,
                                                outcome_labels=None, outcome_widths=None,
                                                ax=None, positions=None, label_font_size=12,
                                                node_colors="w", node_sizes=300):
    """Generates a networkx graph visualizing arbitrary pairwise relationships in an iterable

    This function takes an iterable input_values and calls a function on all pairs therein. The individual values are
    labeled with the input_labels and the outcome weights, labels, colors, and widths are all dependent on the value
    returned by the relationship_function. For this purpose all the parameters given for these features are
    translated to a function and called with the outcome of the relationship_function call.

    :param input_values: Iterable that contains all values that the pairs are drawn from.
    :type input_values: list or tuple
    :param relationship_function: Function that is called on all pairs within input_values
    :type relationship_function: function
    :param input_labels: Labels that will be used to annotate the nodes representing the input_values
    :type input_labels: list or tuple
    :param outcome_colors: Determines the color of the edge based on the output from relationship_function
    :type outcome_colors: dict or function or string or None
    :param outcome_weights: Determines the color of the edge based on the output from relationship_function
    :type outcome_weights: dict or function or int or float or None
    :param outcome_labels: Determines the color of the edge based on the output from relationship_function
    :type outcome_labels: dict or function or string or None
    :param outcome_widths: Determines the color of the edge based on the output from relationship_function
    :type outcome_widths: dict or function or int or float or None
    :param ax: Matplotlib Axes this graph will be plotted to
    :type ax: plt.Axes or None
    :param positions: List or tuple of coordinates that is given to networkx.draw() to determine node positions
    :type positions: list or tuple or None
    :param label_font_size: Font size of the node labels
    :type label_font_size: int
    :param node_colors: Colors of the nodes
    :type node_colors: list or tuple or string
    :param node_sizes: Sizes of the nodes
    :type node_sizes: list or tuple or int
    :return: plt.Axes with a networkx graph added to show the pairwise relations between the input_values
    :rtype: plt.Axes
    """

    outcome_dict = {}
    for i, value_1 in enumerate(input_values):
        for j, value_2 in enumerate(input_values):
            if i >= j:
                continue
            outcome_dict[(i, j)] = relationship_function(value_1, value_2)

    node_labels = dict(zip(range(len(input_labels)), input_labels)) if input_labels else None

    return plot_connections_from_pair_dict(outcome_dict, node_labels=node_labels,
                                           outcome_colors=outcome_colors, outcome_weights=outcome_weights,
                                           outcome_labels=outcome_labels, outcome_widths=outcome_widths,
                                           ax=ax, positions=positions, label_font_size=label_font_size,
                                           node_colors=node_colors, node_sizes=node_sizes)


def pair_dict_to_network_graph(pair_dict, directed=False,
                               edge_colors=None, edge_weights=None,
                               edge_labels=None, edge_widths=None,
                               node_colors=None, node_edgecolors=None,
                               node_sizes=None, node_labels=None):
    """
    Generates a networkx graph from a dictionary of pairs and their connections with the given parameters for the
    nodes and edges. The node_colors can be given as a dictionary that maps nodes to colors or as a list or tuple of
    colors. The node_edgecolor is the color of the border of the nodes. The node_sizes is the size of the nodes. The
    directed parameter determines if the graph is directed or not.

    :param pair_dict: A dict of (node1, node2) pairs and their connection values
    :type pair_dict: dict
    :param directed: Whether the graph is directed or not
    :type directed: bool
    :param edge_colors:
    :param edge_weights:
    :param edge_labels:
    :param edge_widths:
    :param node_colors:
    :param node_edgecolors:
    :param node_sizes:
    :param node_labels:
    :return:
    """
    if not directed:
        g = nx.Graph()
    else:
        g = nx.DiGraph()

    edge_dict = {
        "color": convert_to_func(edge_colors if edge_colors is not None else "blue"),
        "weight": convert_to_func(edge_weights if edge_weights is not None else lambda x: x),
        "width": convert_to_func(edge_widths if edge_widths is not None else 1),
        "label": convert_to_func(edge_labels if edge_labels is not None else lambda x: x)
    }

    for node_pair, connection_value in pair_dict.items():
        g.add_edge(node_pair[0], node_pair[1],
                   color=edge_dict["color"](connection_value),
                   weight=edge_dict["weight"](connection_value),
                   width=edge_dict["width"](connection_value),
                   label=edge_dict["label"](connection_value))

    node_dict = {
        "color": convert_to_func(node_colors if node_colors is not None else "white"),
        "edgecolor": convert_to_func(node_edgecolors if node_edgecolors is not None else "black"),
        "size": convert_to_func(node_sizes if node_sizes is not None else 300),
        "label": convert_to_func(node_labels if node_labels is not None else lambda x: x)
    }

    for node, node_attributes in g.nodes.data():
        node_attributes["color"] = node_dict["color"](node)
        node_attributes["edgecolor"] = node_dict["edgecolor"](node)
        node_attributes["size"] = node_dict["size"](node)
        node_attributes["label"] = node_dict["label"](node)

    return g


def plot_connections_from_pair_dict(pair_dict, directed=False, positions=None, ax=None, label_font_size=12,
                                    bbox_dict=None, connection_style_dict=None, **pair_dict_to_network_kwargs):
    """
    Generates a networkx graph from a dictionary of pairs and their connections with the given parameters for the
    nodes and edges. The positions of the nodes can be given as a dictionary or a function that takes the graph as
    input and returns a dictionary of positions. Same goes for the node_labels, outcome_colors, outcome_weights,
    outcome_labels, and outcome_widths. The node_colors can be given as a dictionary that maps nodes to colors or as
    a list or tuple of colors. The node_edgecolor is the color of the border of the nodes. The node_sizes is the size
    of the nodes. The directed parameter determines if the graph is directed or not. The bbox_dict is a dictionary
    that determines the style of the node labels.

    :param pair_dict:
    :type pair_dict:
    :param positions:
    :type positions:
    :param ax:
    :type ax:
    :param label_font_size:
    :type label_font_size:
    :param directed:
    :type directed:
    :param bbox_dict:
    :type bbox_dict:
    :param connection_style_dict:
    :type connection_style_dict:
    :return:
    :rtype:
    """
    g = pair_dict_to_network_graph(pair_dict, directed=directed, **pair_dict_to_network_kwargs)

    edges = g.edges()
    colors = [g[u][v]['color'] for u, v in edges]
    widths = [g[u][v]['width'] for u, v in edges]
    edge_labels = {(u, v): g[u][v]['label'] for u, v in edges}

    nodes_data = g.nodes.data()
    node_colors = [node_data["color"] for _node, node_data in nodes_data]
    node_edgecolor = [node_data["edgecolor"] for _node, node_data in nodes_data]
    node_sizes = [node_data["size"] for _node, node_data in nodes_data]
    node_labels = {node: node_data["label"] for node, node_data in nodes_data}

    plot_ax = ax if ax is not None else plt.subplots()[1]
    original_artists = set(get_plot_elements(plot_ax))

    if callable(positions):
        positions = positions(g)
    positions = nx.spring_layout(g) if positions is None else positions

    default_bbox_dict = {"facecolor": "none", "edgecolor": 'none', "boxstyle": 'round,pad=0.2'}
    bbox_dict = update_default_dict(default_bbox_dict, bbox_dict)

    nx.draw_networkx_nodes(g, pos=positions,
                           node_color=node_colors, edgecolors=node_edgecolor,
                           linewidths=1.0, node_size=node_sizes, ax=plot_ax)

    if node_labels:
        nx.draw_networkx_labels(g, pos=positions, ax=plot_ax, labels=node_labels, font_size=label_font_size,
                                bbox=bbox_dict)

    default_connection_style_dict = dict(connectionstyle="arc3,rad=0.1")
    connection_style_dict = {} if connection_style_dict is None else connection_style_dict
    connection_style_dict = {**default_connection_style_dict, **connection_style_dict}
    connection_style_dict = connection_style_dict if directed else {}

    alphas = [0 if color == "none" else 0.3 for color in colors]
    nx.draw_networkx_edges(g, pos=positions, ax=plot_ax, edge_color=colors, width=widths,
                           node_size=node_sizes, arrows=directed, alpha=alphas, **connection_style_dict)

    edge_labels = {key: value for key, value in edge_labels.items() if value is not None}
    label_pos_dict = dict(label_pos=0.3) if directed else {}
    nx.draw_networkx_edge_labels(g, pos=positions, ax=plot_ax, edge_labels=edge_labels, font_size=label_font_size,
                                 bbox=bbox_dict, **label_pos_dict, **connection_style_dict)

    current_artists = set(get_plot_elements(plot_ax))
    return plot_ax, current_artists - original_artists, g
