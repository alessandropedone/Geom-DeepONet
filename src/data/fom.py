## @package dataset_generation
# @brief Full order model that solves the PDE and saves the solution and the gradient of the solution in a .h5 file.

##
# @param mesh (str): path to the mesh file.
# @param data_folder (str): path to the data folder.



def fom(mesh: str, data_folder: str = "data"):
    """
    Full order model that solves the PDE and saves the solution and the gradient of the solution in a .h5 file.
    """

    if mesh.is_file() and mesh.suffix == ".msh":
        
        # Read the mesh from the .msh file
        from mpi4py import MPI
        from dolfinx.io import gmshio
        domain, cell_tags, facet_tags = gmshio.read_from_msh(mesh, MPI.COMM_WORLD, 0, gdim=2)

        # Define finite element function space
        from dolfinx.fem import functionspace
        import numpy as np
        V = functionspace(domain, ("Lagrange", 1))

        # Identify the boundary (create facet to cell connectivity required to determine boundary facets)
        from dolfinx import default_scalar_type
        from dolfinx.fem import (Constant, dirichletbc, locate_dofs_topological)
        from dolfinx.fem.petsc import LinearProblem
        tdim = domain.topology.dim
        fdim = tdim - 1
        domain.topology.create_connectivity(fdim, tdim)

        # Find facets marked with 10, 11, 12 (the two plates)
        facets_rect1 = np.concatenate([facet_tags.find(10), facet_tags.find(11)])
        facets_rect2 = facet_tags.find(12)

        # Locate degrees of freedom
        dofs_rect1 = locate_dofs_topological(V, fdim, facets_rect1)
        dofs_rect2 = locate_dofs_topological(V, fdim, facets_rect2)

        # Define different Dirichlet values
        u_rect1 = Constant(domain, 0.0)
        u_rect2 = Constant(domain, 1.0)

        # Create BCs
        bc1 = dirichletbc(u_rect1, dofs_rect1, V)
        bc2 = dirichletbc(u_rect2, dofs_rect2, V)

        bcs = [bc1, bc2]

        # Trial and test functions
        import ufl
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # Source term
        from dolfinx import default_scalar_type
        from dolfinx import fem
        f = fem.Constant(domain, default_scalar_type(0.0))

        # Variational problem
        a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = f * v * ufl.dx

        # Assemble the system
        from dolfinx.fem.petsc import LinearProblem
        problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()


        import ufl

        # Define the vector function space for the gradient
        V_grad = fem.functionspace(domain, ("DG", 0, (domain.geometry.dim, )))

        # Define the trial and test functions for the vector space
        u_vec = ufl.TrialFunction(V_grad)
        v_vec = ufl.TestFunction(V_grad)

        # Define the gradient of the solution
        grad_u = ufl.grad(uh)

        # Define the bilinear and linear forms
        a_grad = ufl.inner(u_vec, v_vec) * ufl.dx
        L_grad = ufl.inner(grad_u, v_vec) * ufl.dx

        # Assemble the system
        from dolfinx.fem.petsc import LinearProblem
        problem_grad = LinearProblem(a_grad, L_grad, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        grad_uh = problem_grad.solve()

        # Extract gradient components
        dim = domain.geometry.dim
        grad_x_uh = grad_uh.x.array[0::dim]
        grad_y_uh = grad_uh.x.array[1::dim]
        
        # Find all dofs in the function spaces
        dofs_uh = np.arange(V.dofmap.index_map.size_local)
        dofs_grad_uh = np.arange(V_grad.dofmap.index_map.size_local)

        
        # Plot solution and gradient components (for debugging)
        import matplotlib.pyplot as plt

        # Get potential values and coordinates
        potential = np.array(uh.x.array[dofs_uh])
        dofs_c = V.tabulate_dof_coordinates()[dofs_uh]
        x = np.array(dofs_c[:, 0])
        y = np.array(dofs_c[:, 1])

        # Get boundary coordinates for each tag separately
        def get_boundary_coords(V, fdim, facet_tags, tag):
            facets = facet_tags.find(tag)
            dofs = locate_dofs_topological(V, fdim, facets)
            coords = V.tabulate_dof_coordinates()[dofs]
            return np.array(coords[:, 0]), np.array(coords[:, 1])

        # Get coordinates for each boundary
        # Hole 1: tags 10 and 11 combined
        x_hole1_tag10, y_hole1_tag10 = get_boundary_coords(V, fdim, facet_tags, 10)
        x_hole1_tag11, y_hole1_tag11 = get_boundary_coords(V, fdim, facet_tags, 11)
        x_hole1 = np.concatenate([x_hole1_tag10, x_hole1_tag11])
        y_hole1 = np.concatenate([y_hole1_tag10, y_hole1_tag11])

        # Hole 2: tag 12
        x_hole2, y_hole2 = get_boundary_coords(V, fdim, facet_tags, 12)

        # Outer boundary: tag 20
        x_outer, y_outer = get_boundary_coords(V, fdim, facet_tags, 20)

        # Plot potential with boundary overlays
        # get domain connectivity matrix
        cells = domain.topology.connectivity(tdim, 0).array.reshape(-1, tdim + 1)
        import matplotlib.tri as tri
        triang = tri.Triangulation(x, y, triangles = cells)
        plt.tricontourf(triang, potential, levels=100, cmap='viridis')
        # plot also the connectivity (mesh)
        plt.triplot(triang, color='lightgrey', linewidth=0.5, alpha=0.5)
        plt.title('Potential Distribution with Boundaries')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.colorbar(label="Potential")
        plt.axis('equal')
        plt.show()
        
        print("Shape of uh vector:", uh.x.array.shape)
        print("Shape of grad_uh vector:", grad_uh.x.array.shape)
        print("Number of dofs in V:", len(dofs_uh))
        print("Number of dofs in V_grad:", len(dofs_grad_uh))






import pathlib

mesh_folder_path = pathlib.Path("data/msh")
mesh = list(mesh_folder_path.iterdir())  # Process only the first mesh for testing
fom(mesh[0], data_folder="data")


