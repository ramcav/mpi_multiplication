#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 500  // Number of rows in A
#define K 300 // Number of columns in A and rows in B
#define N 500  // Number of columns in B

// Function to initialize matrices A and B
void initialize_matrices(int A[M][K], int B[K][N]) {
    // Seed!
    srand(time(NULL));

    // Initialize matrix A
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[i][j] = rand() % 10;
        }
    }

    // Initialize matrix B
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B[i][j] = rand() % 10;
        }
    }
}

// Print a matrix to a file
void print_matrix_to_file(FILE *file, int rows, int cols, int matrix[rows][cols]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%5d ", matrix[i][j]);  // Print to file instead of stdout
        }
        fprintf(file, "\n");
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Open a file to write the output
    FILE *file;
    if (rank == 0) {
        file = fopen("matrix_output.txt", "w");
        if (file == NULL) {
            printf("Error opening file!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Matrices and buffers
    int A[M][K], B[K][N], C[M][N];
    
    // Buffer for the scattered rows of A
    int local_A[M/size + 1][K];

    // Buffer to hold the local part of C
    int local_C[M/size + 1][N];

    // // Each process gets at least this many rows
    int local_rows = M / size;  

    // However, they can get more if the division remainder is not 0
    int extra_rows = M % size;

    // Define counts and displacements for scattering and gathering.
    // The counts array is an array representing how much elements is each process going to be working with.
    // As you can see, the counts is dynamically assigned depending on if the process is working with extra (uneven rows)
    // or if it is working with even rows.

    // The displs array (displacements) indicates where exactly should each process start looking for the data
    // So, for example, if we pass an index of i to the matrix, it indicates that it should start looking for
    // the data in matrix A starting at index i.

    int counts[size], displs[size];
    int displ = 0;

    for (int i = 0; i < size; i++) {
        counts[i] = (i < extra_rows) ? (local_rows + 1) * K : local_rows * K;
        displs[i] = displ;
        displ += counts[i];
    }

    if (rank == 0) {
        // Initialize matrices A and B in the root process
        initialize_matrices(A, B);

        // Print A and B to the file on the root process
        fprintf(file, "Matrix A (M x K):\n");
        print_matrix_to_file(file, M, K, A);
        fprintf(file, "\nMatrix B (K x N):\n");
        print_matrix_to_file(file, K, N, B);
        fprintf(file, "\n");
    }

    // Then, the displacements and counts are passed to Scatterv, which will scatter the data and work
    // between the processes. I'm only passing the data in A to all processes, as then, every single row
    // passed to A will need to be computed along with the whole matrix B
    MPI_Scatterv(A, counts, displs, MPI_INT, local_A, counts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    // B is completely broadcasted to all processes in order to compute the partial multiplications
    // With the chunks of A
    MPI_Bcast(B, K * N, MPI_INT, 0, MPI_COMM_WORLD);

    // **Start timing the computation**
    double start_time = MPI_Wtime();

    // Here the local multiplication of A times B is done for each process
    int rows = counts[rank] / K; 

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i][j] = 0;
            for (int k = 0; k < K; k++) {
                local_C[i][j] += local_A[i][k] * B[k][j];
            }
        }
    }

    // **Start timing the computation**
    double end_time = MPI_Wtime();
    double time = end_time - start_time; 

    // Calculate the displacements and counts to gather all the data again
    displ = 0;
    for (int i = 0; i < size; i++) {
        counts[i] = (i < extra_rows) ? (local_rows + 1) * N : local_rows * N;
        displs[i] = displ;
        displ += counts[i];
    }

    // Gather the local parts of C from all processes. This is done
    MPI_Gatherv(local_C, rows * N, MPI_INT, C, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Print the result
        fprintf(file, "Resultant Matrix C took %f time to be computed:\n", time);
        print_matrix_to_file(file, M, N, C);
        fclose(file);
    }

    MPI_Finalize();
    return 0;
}
