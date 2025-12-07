## @package plot
# @brief Functions for plotting solutions and gradients over a given domain.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import BoundaryNorm



##
# @param x (np.ndarray): x coordinates of the vertices.
# @param y (np.ndarray): y coordinates of the vertices.
# @param cells (np.ndarray): connectivity of the mesh cells.
# @param grad (np.ndarray): quantity defined per cell to be plotted.
# @param title (str): Title of the plot.
# @param xlabel (str): Label for the x-axis.
# @param ylabel (str): Label for the y-axis.
# @param colorbar_label (str): Label for the color bar.
# @param cmap (str): Colormap to use for plotting.
# @param sharp_color_range (tuple): Optional range to create sharp color transitions.
# @param plot_triangulation (bool): Whether to overlay the mesh triangulation.
# @param postpone_show (bool): Whether to postpone the plt.show() call.
# @note Add optional arguments for customizing the plot (e.g., color map, title, labels).
def cells_plot(x: np.ndarray,
               y: np.ndarray, 
               cells: np.ndarray, 
               sol: np.ndarray, 
               title: str ="Title", 
               xlabel: str ="x", 
               ylabel: str ="y", 
               colorbar_label: str ="Quantity",
               cmap: str ='RdBu_r',
               sharp_color_range: tuple = None,
               plot_triangulation: bool = True,
               postpone_show: bool = False) -> None:
    """It plots the provided solution over the domain using cell-based data."""
    
    # Create triangulation object
    triang = tri.Triangulation(x, y, triangles = cells)

    # Plot the solution using tripcolor
    if sharp_color_range is not None:
        # Define custom color normalization with sharp transitions in specified range
        bounds = np.concatenate([
            np.linspace(sol.min(), sharp_color_range[0], 50, endpoint=False),
            np.linspace(sharp_color_range[0], sharp_color_range[1], 150),
            np.linspace(sharp_color_range[1], sol.max(), 50)
        ])
        norm = BoundaryNorm(boundaries=bounds, ncolors=256, clip=True)
        plt.tripcolor(
            triang,
            facecolors=sol,
            shading='flat',
            cmap=cmap,
            norm=norm,
        )
    else:
        plt.tripcolor(
            triang,
            facecolors=sol,
            shading='flat',
            cmap=cmap
        )
    # plot also the connectivity (mesh)
    if plot_triangulation:
        plt.triplot(triang, color='lightgrey', linewidth=0.5, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.colorbar(label=colorbar_label)
    if not postpone_show:
        plt.show()

##
# @param x (np.ndarray): x coordinates of the vertices.
# @param y (np.ndarray): y coordinates of the vertices.
# @param cells (np.ndarray): connectivity of the mesh cells.
# @param sol (np.ndarray): quantity defined per vertex to be plotted.
# @param title (str): Title of the plot.
# @param xlabel (str): Label for the x-axis.
# @param ylabel (str): Label for the y-axis.
# @param colorbar_label (str): Label for the color bar.
# @param cmap (str): Colormap to use for plotting.
# @param sharp_color_range (tuple): Optional range to create sharp color transitions.
# @param plot_triangulation (bool): Whether to overlay the mesh triangulation.
# @param postpone_show (bool): Whether to postpone the plt.show() call.
# @note Add optional arguments for customizing the plot (e.g., color map, title, labels).
def vertices_plot(x: np.ndarray, 
                  y: np.ndarray,
                  cells: np.ndarray,
                  sol: np.ndarray,
                  title: str ="Title",
                  xlabel: str ="x",
                  ylabel: str ="y",
                  colorbar_label: str ="Quantity",
                  cmap: str ='RdBu_r',
                  sharp_color_range: tuple = None, 
                  plot_triangulation: bool = True,
                  postpone_show: bool = False) -> None:
    """It plots the specified solution over the domain using vertex-based data."""
    
    triang = tri.Triangulation(x, y, triangles = cells)

    if sharp_color_range is not None:
        bounds = np.concatenate([
            np.linspace(sol.min(), sharp_color_range[0], 50, endpoint=False),
            np.linspace(sharp_color_range[0], sharp_color_range[1], 150),
            np.linspace(sharp_color_range[1], sol.max(), 50)
        ])
        norm = BoundaryNorm(boundaries=bounds, ncolors=256, clip=True)
        plt.tricontourf(
            triang,
            sol,
            levels=256,
            cmap=cmap,
            norm=norm,
        )
    else:
        plt.tricontourf(
            triang,
            sol,
            levels=256,
            cmap=cmap
        )

    # plot also the connectivity (mesh)
    if plot_triangulation:
        plt.triplot(triang, color='lightgrey', linewidth=0.5, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.colorbar(label=colorbar_label)
    if not postpone_show:
        plt.show()


##
# @param x (np.ndarray): x coordinates of the vertices.
# @param y (np.ndarray): y coordinates of the vertices.
# @param cells (np.ndarray): connectivity of the mesh cells.
# @param title (str): Title of the plot.
# @param xlabel (str): Label for the x-axis.
# @param ylabel (str): Label for the y-axis.
# @param plot_triangulation (bool): Whether to overlay the mesh triangulation.
# @param postpone_show (bool): Whether to postpone the plt.show() call.
def domain_plot(x: np.ndarray, 
            y: np.ndarray,
            cells: np.ndarray,
            title: str ="Title",
            xlabel: str ="x",
            ylabel: str ="y",
            plot_triangulation: bool = True,
            postpone_show: bool = False) -> None:
    """It plots only the domain and the mesh."""
    triang = tri.Triangulation(x, y, triangles = cells)
    triangles = triang.triangles
    neighbors = triang.neighbors

    # plot also the connectivity (mesh)
    if plot_triangulation:
        plt.triplot(triang, color='lightgrey', linewidth=0.5, alpha=0.5)

    boundary_edges = []

    # Each triangle has 3 edges.
    # If a neighbor is -1, that edge is on the boundary.
    for t_idx, tri_nodes in enumerate(triangles):
        for edge_local, neigh in enumerate(neighbors[t_idx]):
            if neigh == -1:  # boundary edge
                i = tri_nodes[edge_local]
                j = tri_nodes[(edge_local + 1) % 3]
                boundary_edges.append((i, j))

    # Remove duplicates while keeping order
    boundary_edges = list(dict.fromkeys(tuple(sorted(e)) for e in boundary_edges))

    # Plot ONLY boundary
    for i, j in boundary_edges:
        plt.plot([x[i], x[j]], [y[i], y[j]], color='black', linewidth=1.0)
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if not postpone_show:
        plt.show()

##
# @param x (np.ndarray): x coordinates of the vertices.
# @param y (np.ndarray): y coordinates of the vertices.
# @param cells (np.ndarray): connectivity of the mesh cells.
# @param sol (np.ndarray): quantity defined per vertex or per cell to be plotted.
# @param title (str): Title of the plot.
# @param xlabel (str): Label for the x-axis.
# @param ylabel (str): Label for the y-axis.
# @param colorbar_label (str): Label for the color bar.
# @param cmap (str): Colormap to use for plotting.
# @param sharp_color_range (tuple): Optional range to create sharp color transitions.
# @param plot_triangulation (bool): Whether to overlay the mesh triangulation.
# @param postpone_show (bool): Whether to postpone the plt.show() call.
# @note If you set the solution to None you can plot only the domain.
def plot(x: np.ndarray, 
         y: np.ndarray, 
         cells: np.ndarray, 
         sol: np.ndarray,
         title: str ="Title", 
         xlabel: str ="x", 
         ylabel: str ="y", 
         colorbar_label: str ="Quantity",
         cmap: str ='RdBu_r',
         sharp_color_range: tuple = None,
         plot_triangulation: bool = True,
         postpone_show: bool = False) -> None:
    """It plots the provided solution over the domain."""

    # Check validity of input data
    if len(x) == 0 or len(y) == 0 or len(cells) == 0 or (sol is not None and len(sol) == 0):
        print("No data to plot.")
        return
    
    if len(x) != len(y):
        print("Inconsistent lengths between x and y coordinates.")
        if sol is not None and len(x) != len(sol):
            # check if the solution is given in terms of cells
            if len(cells) % len(sol) != 0:
                print("Inconsistent data lengths between coordinates and solution.")
                return
        return
    
    plt.figure()
    
    if sol is None:
        # Plot only the domain
        domain_plot(x, y, cells, title, xlabel, ylabel, plot_triangulation, postpone_show)
    elif len(cells) % len(sol) == 0:
        # Plot using the cells
        cells_plot(x, y, cells, sol, title, xlabel, ylabel, colorbar_label, cmap, sharp_color_range, plot_triangulation, postpone_show)
    else:
        # Plot using the vertices
        vertices_plot(x, y, cells, sol, title, xlabel, ylabel, colorbar_label, cmap, sharp_color_range, plot_triangulation, postpone_show)

##
# @param file (h5py.File): h5py file object containing the solution data.
# @param postpone_show (bool): Whether to postpone the plt.show() call.
def plot_domain(file, postpone_show=False):
    """It plots the domain and the mesh."""
    plot(x = file["x"][:], 
         y = file["y"][:], 
         cells = file["cells"][:], 
         sol = None,
         title ="Domain and mesh", 
         xlabel ="x", 
         ylabel ="y", 
         colorbar_label ="",
         cmap ='RdBu_r',
         sharp_color_range = None,
         plot_triangulation = True,
         postpone_show = postpone_show)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

## 
# @param file (h5py.File): h5py file object containing the solution data.
# @param postpone_show (bool): Whether to postpone the plt.show() call.
def plot_potential(file, postpone_show=False):
    """It plots the electrostatic potential over the domain."""
    plot(x = file["x"][:], 
         y = file["y"][:], 
         cells = file["cells"][:], 
         sol = file["potential"][:],
         title ="Electrostatic potential", 
         xlabel ="x", 
         ylabel ="y", 
         colorbar_label ="Potential",
         cmap ='RdBu_r',
         sharp_color_range = None,
         plot_triangulation = True,
         postpone_show = postpone_show)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

## 
# @param file (h5py.File): h5py file object containing the solution data.
# @param postpone_show (bool): Whether to postpone the plt.show() call.
def plot_error(file, postpone_show=False):
    """It plots the electrostatic potential over the domain."""
    plot(x = file["x"][:], 
         y = file["y"][:], 
         cells = file["cells"][:], 
         sol = file["se"][:],
         title ="Squared error in electrostatic potential prediction (RMSE {:.2e})".format(np.sqrt(np.mean(file["se"][:]))), 
         xlabel ="x", 
         ylabel ="y", 
         colorbar_label ="Squared Error",
         cmap ='RdBu_r',
         sharp_color_range = None,
         plot_triangulation = True,
         postpone_show = True)
    plot(x = file["x"][:], 
         y = file["y"][:], 
         cells = file["cells"][:], 
         sol = file["ae"][:],
         title ="Absolute error in electrostatic potential prediction (MAE {:.2e})".format(np.mean(file["ae"][:])),
         xlabel ="x", 
         ylabel ="y", 
         colorbar_label ="Absolute Error",
         cmap ='RdBu_r',
         sharp_color_range = None,
         plot_triangulation = True,
         postpone_show = postpone_show)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

## 
# @param file (h5py.File): h5py file object containing the solution data.
# @param postpone_show (bool): Whether to postpone the plt.show() call.
def plot_potential_pred(file, postpone_show=False):
    """It plots the electrostatic potential over the domain."""
    plot(x = file["x"][:], 
         y = file["y"][:], 
         cells = file["cells"][:], 
         sol = file["potential_pred"][:],
         title ="Predicted electrostatic potential", 
         xlabel ="x", 
         ylabel ="y", 
         colorbar_label ="Potential",
         cmap ='RdBu_r',
         sharp_color_range = None,
         plot_triangulation = True,
         postpone_show = postpone_show)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    
##
# @param file (h5py.File): h5py file object containing the solution data.
# @param postpone_show (bool): Whether to postpone the plt.show() call.
def plot_grad_x(file, postpone_show=False):
    """It plots the x component of the gradient of the electrostatic potential."""
    plot(x = file["x"][:], 
         y = file["y"][:], 
         cells = file["cells"][:], 
         sol = file["grad_x"][:],
         title ="x component of the gradient of the electrostatic potential", 
         xlabel ="x", 
         ylabel ="y", 
         colorbar_label ="grad_x",
         cmap ='RdBu_r',
         sharp_color_range =  None,
         plot_triangulation = True,
         postpone_show = postpone_show)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

##
# @param file (h5py.File): h5py file object containing the solution data.
# @param postpone_show (bool): Whether to postpone the plt.show() call.
def plot_grad_y(file, postpone_show=False):
    """It plots the y component of the gradient of the electrostatic potential."""
    plot(x = file["x"][:], 
         y = file["y"][:], 
         cells = file["cells"][:], 
         sol = file["grad_y"][:],
         title ="y component of the gradient of the electrostatic potential", 
         xlabel ="x", 
         ylabel ="y", 
         colorbar_label ="grad_y",
         cmap ='RdBu_r',
         sharp_color_range = (-0.7, -0.5),
         plot_triangulation = True,
         postpone_show = postpone_show)
    ax = plt.gca()
    ax.set_xlim(-55, 55)
    ax.set_ylim(-25, 25)
    ax.set_aspect('equal', adjustable='box')
    
##
# @param file (h5py.File): h5py file object containing the solution data.
# @param postpone_show (bool): Whether to postpone the plt.show() call.
def plot_normal_derivative(file, postpone_show=False):   
    """It plots the normal derivative of the potential on the upper plate as arrows.""" 
    plot_domain(file, postpone_show=True)
    normal_derivative = file["normal_derivatives_plate"][:]
    points = file["midpoints_plate"][:]
    normals = file["normal_vectors_plate"][:]
    U = normal_derivative * normals[:, 0]
    V = normal_derivative * normals[:, 1]
    # Add the plot arrows from midpoints in the direction of the normals scaled by the normal derivative
    for i in range(len(points)):
        plt.arrow(points[i, 0], points[i, 1], U[i], V[i], head_width=0.05, head_length=0.05, fc='r', ec='r')
    plt.title("Normal derivative of the potential on the upper plate")
    ax = plt.gca()
    ax.set_xlim(-55, 55)
    ax.set_ylim(-25, 25)
    ax.set_aspect('equal', adjustable='box')
    if not postpone_show:
        plt.show()

##
# @param file (h5py.File): h5py file object containing the solution data.
def summary_plot(file):
    """It creates a summary plot with all relevant plots."""
    plot_domain(file, postpone_show=True)
    plot_potential(file, postpone_show=True)
    plot_grad_x(file, postpone_show=True)
    plot_grad_y(file, postpone_show=True)
    plot_normal_derivative(file, postpone_show=True)
    plt.show()