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
# @param sol (np.ndarray): quantity defined per vertex or per cell to be plotted.
# @param title (str): Title of the plot.
# @param xlabel (str): Label for the x-axis.
# @param ylabel (str): Label for the y-axis.
# @param colorbar_label (str): Label for the color bar.
# @param cmap (str): Colormap to use for plotting.
# @param sharp_color_range (tuple): Optional range to create sharp color transitions.
# @param plot_triangulation (bool): Whether to overlay the mesh triangulation.
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
         plot_triangulation: bool = True) -> None:

    # Check validity of input data
    if len(x) == 0 or len(y) == 0 or len(cells) == 0 or len(sol) == 0:
        print("No data to plot.")
        return
    if len(x) != len(y):
        print("Inconsistent lengths between x and y coordinates.")
        if len(x) != len(sol):
            # check if the solution is given in terms of cells
            if len(cells) % len(sol) != 0:
                print("Inconsistent data lengths between coordinates and solution.")
                return
        return
    
    if len(cells) % len(sol) == 0:
        # Plot using the cells
        cells_plot(x, y, cells, sol, title, xlabel, ylabel, colorbar_label, cmap, sharp_color_range, plot_triangulation)
    else:
        # Plot using the vertices
        vertices_plot(x, y, cells, sol, title, xlabel, ylabel, colorbar_label, cmap, sharp_color_range, plot_triangulation)

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
               plot_triangulation: bool = True) -> None:
    
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
    plt.axis('equal')
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
                  plot_triangulation: bool = True) -> None:
    
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
    plt.triplot(triang, color='lightgrey', linewidth=0.5, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.colorbar(label=colorbar_label)
    plt.axis('equal')
    plt.show()