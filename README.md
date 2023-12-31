# Solving-Poisson-equation-using-Successive-Over-Relaxation
Introduction:

The aim of this project is to solve the 2D Poisson equation using the iterative Jacobi method Successive Over Relaxation (SOR), a popular iterative technique for solving partial differential equations. The Poisson equation is a fundamental partial differential equation (PDE) that arises in a wide range of scientific and engineering applications, including fluid dynamics, electrostatics, and heat conduction. Given the importance of the Poisson equation, it is essential to develop efficient numerical methods to solve it in various contexts.

Our thesis is to implement the SOR method, which helps in solving the Poisson equation both in serial and hybrid models (MPI+OpenMP) and to prove that the hybrid model has lower run time compared to the serial. The structure of the report is organized as follows: Introduction, Methods, Results, Conclusions, and References.

Background on Poisson Equation and Methods:

The Poisson equation is a second-order partial differential equation (PDE) that arises in various scientific and engineering applications, such as heat conduction, fluid flow, and electrostatics. It is given by the following expression:

Δu = σ(x)
Where Δ denotes the Laplace operator, σ(x) is the “source term,” and is often zero, either everywhere or everywhere except for some specific region. Solving the Poisson equation is crucial for understanding and predicting the behavior of many physical systems.

Successive Over-Relaxation (SOR) is an iterative method used to solve linear systems of equations, particularly those arising from the discretization of partial differential equations, such as the Poisson and Laplace equations. The SOR method is an improvement upon the Gauss-Seidel method, which is itself an enhancement of the Jacobi method. The key idea behind SOR is to introduce a relaxation factor (ω) to the Gauss-Seidel method to accelerate the convergence of the solution. The relaxation factor is a scalar value, usually chosen between 1 and 2, that controls the amount of "over-relaxation" applied to each iteration. By tuning the relaxation factor, the SOR method can achieve faster convergence compared to the Gauss-Seidel method, potentially reducing the number of iterations required to reach an acceptable solution.

One of the challenges in applying the Jacobi method to solve the Poisson equation is the computational complexity, particularly when dealing with large-scale problems or high-resolution simulations. To overcome this limitation, various strategies for parallelization and halo exchange have been proposed in the literature. Parallelization enables the distribution of computation across multiple processors, while halo exchange facilitates the communication of data between neighboring processors. These techniques can significantly improve the computational efficiency and scalability of the SOR method.

A key aspect of implementing parallelization and halo exchange is the use of efficient communication schemes and topologies. Message Passing Interface (MPI) is a widely used standard for parallel programming, providing a set of functions for exchanging messages between processors in a parallel computing environment. In particular, MPI Cartesian topology and communication functions allow for the creation of a structured grid of processors, finding the neighbors, simplifying the communication patterns, and improving performance.

Another important aspect of parallel programming with MPI is ensuring synchronization between processors. MPI_Barrier is a function that provides a synchronization mechanism, making sure that all processes have reached the same point in their execution before proceeding. This can be crucial for maintaining consistency in the Jacobi method's iterative process.

In addition to parallelization and communication strategies, efficient memory management is also essential for the successful implementation of the Jacobi method. Dynamic memory allocation in C, such as malloc, calloc, free, and realloc, allows for the allocation and deallocation of memory during runtime. By using these functions, it is possible to optimize memory usage and prevent memory leaks, leading to a more efficient and robust implementation of the Jacobi method.

In summary, the background study for the project "Solving Poisson Equation using SOR" involves understanding the Poisson equation, the Jacobi method, and various computational techniques for improving the efficiency and accuracy of the method. The literature review encompasses sources related to the mathematical foundations of the Poisson equation and Jacobi method, parallelization and halo exchange strategies, communication schemes and topologies, synchronization mechanisms, and dynamic memory allocation
