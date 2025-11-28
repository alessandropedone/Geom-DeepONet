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
        V_vec = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

        # Define the trial and test functions for the vector space
        u_vec = ufl.TrialFunction(V_vec)
        v_vec = ufl.TestFunction(V_vec)

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
        

        
        # SAVE

        # Save the gradient values in the normal_derivative_potential.csv file
        
        # Read the normal_derivative_potential.csv file
        import pandas as pd
        j = int(mesh.stem)
        df = pd.read_csv(f"{data_folder}/normal_derivative_potential.csv")
        df2 = pd.read_csv(f"{data_folder}/coordinates.csv")
        # Get the coordinates of center of the lower edge of the upper plate, 
        # to calculate the coordinates of the points belonging to it
        center_y = df.iloc[j-1, 2] / 2
        print("mesh number: ", j, " center_y: ", center_y)
        center_x = 0.0

        # Sort the coordinates and the corresponding grad_uh values, then save grad_y into normal_derivative_potential_{j}.csv
        coords = np.sign(x_coords) * np.sqrt((x_coords-center_x)**2 + (y_coords-center_y)**2)
        sorted_indices = np.argsort(coords)
        coords = coords[sorted_indices]
        grad_y_uh_plate = grad_y_uh_plate[sorted_indices]
        grad_x_uh_plate = grad_x_uh_plate[sorted_indices]

        # Compute the normale derivative of the potential
        angle = df.iloc[j-1, 3] * np.pi / 180
        normal_der = - grad_x_uh_plate * np.sin(angle) + grad_y_uh_plate * np.cos(angle) 

        # Save the sorted grad values into a temp CSV file
        start_col = 4
        needed_cols = start_col + len(coords)
        df.iloc[j-1, start_col:needed_cols] = normal_der
        # Copy the jth line of normal_derivative_potential.csv into normal_derivative_potential_{j}.csv
        df.iloc[[j-1]].to_csv(f"{data_folder}/temp/normal_derivative_potential_{j}.csv", index=False)             

        # Save the sorted coordinates into a temp CSV file
        start_col = 4
        needed_cols = start_col + len(coords)
        df.iloc[j-1, start_col:needed_cols] = coords
        # Copy the jth line of normal_derivative_potential.csv into coordinates_{j}.csv
        df.iloc[[j-1]].to_csv(f"{data_folder}/temp/coordinates_{j}.csv", index=False)

        # Create a folder to save the results if it doesn't exist
        from pathlib import Path
        results_folder = Path(f"{data_folder}/results")
        results_folder.mkdir(exist_ok=True, parents=True)

        # Save solution and the gradient of the solution in a .h5 file

        # Find all DOFs in the function space
        dofs = np.arange(V.dofmap.index_map.size_local)

        # Extract the x-coordinates and the y-coordinates of the DOFs
        dofs_c = V.tabulate_dof_coordinates()[dofs]
        x_coords = np.array(dofs_c[:, 0])
        y_coords = np.array(dofs_c[:, 1])

        # Evaluate the function at those DOFs
        dim = domain.geometry.dim
        pval = np.array(uh.x.array[dofs])
        fval_x = grad_uh.x.array[0::dim]
        fval_y = grad_uh.x.array[1::dim]
        fval_x = np.array(fval_x[dofs])
        fval_y = np.array(fval_y[dofs])
        
        # Save the results in a .h5 file
        import os
        import h5py
        base_name = os.path.splitext(os.path.basename(mesh))[0]
        filename = results_folder / f"{base_name}_solution.h5"
        with h5py.File(filename, "w") as file:
            file.create_dataset("coord_x", data=x_coords)
            file.create_dataset("coord_y", data=y_coords)
            file.create_dataset("potential", data=pval)
            file.create_dataset("grad_x", data=fval_x)
            file.create_dataset("grad_y", data=fval_y)


        """
        In the case we want to extract the gradient only on a specific facet, e.g., the lower horizontal edge of the upper plate:
        # Find facets with tag 10 (lower horizontal edge of the upper plate)
        facets10 = facet_tags.find(10)
        dofs10 = locate_dofs_topological(V, fdim, facets10)

        # Extract the x-coordinates and the y-coordinates of the DOFs
        x_dofs = V.tabulate_dof_coordinates()[dofs10]
        x_coords = x_dofs[:, 0]
        y_coords = x_dofs[:, 1]

        # Evaluate grad_uh at those DOFs
        grad_x_uh_plate = grad_x_uh_values[dofs10]
        grad_y_uh_plate = grad_y_uh_values[dofs10]"""