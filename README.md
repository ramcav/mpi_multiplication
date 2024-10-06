# MPI Matrix Multiplication

This program performs matrix multiplication in parallel using MPI (Message Passing Interface). The program takes two matrices `A` (of size M x K) and `B` (of size K x N) and computes the resultant matrix `C` (of size M x N) using parallel processing. Each process is responsible for a portion of the rows of matrix `A`, and the entire matrix `B` is broadcast to all processes.

## Dependencies

To run this program, you need the following:

* An MPI implementation (such as OpenMPI or MPICH).
*  C compiler (such as gcc) with MPI support.

## Installing OpenMPI (if not already installed)

On Ubuntu/Debian systems, you can install OpenMPI with the following command:

```bash
sudo apt-get update
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
```

Or for macOs

```bash
brew install open-mpi
```

## Compiling

You can compile the program using mpicc, the MPI-enabled C compiler that comes with MPI.

```bash
mpicc -o matrix_multiplication_mpi matrix_multiplication_mpi.c
```

## Running the program

To run the program using multiple processes, use the following command:

```bash
mpiexec -np 4 ./matrix_multiplication_mpi
```

### Explanation

Use the np flag to specify the number of processes to be ran

### Output

The output of the execution will be contained within a .txt file ```matrix_output.txt``` that will hold the result of the computation, as well as the computing time it took.

## Additional Info

### Matrix dimensions

The dimensions of the matrixes are hard coded into the source file.

```c
#define M 500  // Number of rows in A
#define K 300  // Number of columns in A and rows in B
#define N 500  // Number of columns in B
```

They can be modified indefinitely and the program should work as expected for reasonable input sizes.

### MPI functions used

* MPI_Scatterv: Scatters rows of matrix A to all processes.
* MPI_Bcast: Broadcasts the entire matrix B to all processes.
* MPI_Gatherv: Gathers the computed parts of matrix C from all processes.

The regular initialization steps for any MPI program are also ran before using the ones above.
