#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>

#include "MatUtil.h"

//#define DEBUG
//#define INFO

#ifdef DEBUG
#define DEBUG_LOG(fmt, ...)	printf(fmt, ##__VA_ARGS__)
#define PRINT_MAT(mat, n) 	print_mat(mat, n)
#else
#define DEBUG_LOG(fmt, ...)
#define PRINT_MAT(mat, n)
#endif

#ifdef INFO 
#define INFO_LOG(fmt, ...)	printf(fmt, ##__VA_ARGS__)
#else
#define INFO_LOG(fmt, ...) 	
#endif


struct timeval tvBefore, tvAfter;
int root, rank, commSize;
size_t n;


void MPI_APSP(int *mat, const size_t n)
{
   int rowCount = n / commSize;
   int partSize = rowCount * n;
   int *krow = NULL;
   int *local = (int*)malloc(sizeof(MPI_INT) * n * rowCount);
   int *rowStorage = (int*)malloc(sizeof(MPI_INT) * n);

   MPI_Scatter(mat, partSize, MPI_INT, local, partSize, MPI_INT, root, MPI_COMM_WORLD);

   for(int k = 0; k < n; k++)
   {
      int *irow = local;
      int krowOwner = k / rowCount;
      
      if(krowOwner == rank)
      {
         krow = local + ((k % rowCount) * n);
         MPI_Bcast(krow, n, MPI_INT, krowOwner, MPI_COMM_WORLD);
      }
      else
      {
         MPI_Bcast(rowStorage, n, MPI_INT, krowOwner, MPI_COMM_WORLD);
         krow = rowStorage;
      }

      for(int r = 0; r < rowCount; r++)
      {
         for(int j = 0; j < n; j++)
         {
            if(irow[k] != -1 && krow[j] != -1)
            {
               int sum = irow[k] + krow[j];
               if(irow[j] == -1 || irow[j] > sum) 
               {
                  INFO_LOG("Process %d replace %d with %d + %d\n", rank, irow[j], irow[k], krow[j]);
                  irow[j] = sum;
               }
            }      
         }
         irow += n;
      }
   }

   MPI_Gather(local, partSize, MPI_INT, mat, partSize, MPI_INT, root, MPI_COMM_WORLD);
   free(local);
   free(rowStorage);
}

void master(int argc, char **argv)
{
   //generate a random matrix.
   int *mat = (int*)malloc(sizeof(int)*n*n);
   GenMatrix(mat, n);

   DEBUG_LOG("Original matrix.\n");
   PRINT_MAT(mat, n);

   //compute the reference result.
   int *ref = (int*)malloc(sizeof(int)*n*n);
   memcpy(ref, mat, sizeof(int)*n*n);
   gettimeofday(&tvBefore, NULL);
   ST_APSP(ref, n);
   gettimeofday(&tvAfter, NULL);
   printf("Elapsed time = %ld usecs\n", (tvAfter.tv_sec - tvBefore.tv_sec) * 1000000 + tvAfter.tv_usec - tvBefore.tv_usec);

   DEBUG_LOG("Sequential algortihm reference.\n");
   PRINT_MAT(ref, n);

   //compute your results
   int *result = (int*)malloc(sizeof(int)*n*n);
   memcpy(result, mat, sizeof(int)*n*n);
   gettimeofday(&tvBefore, NULL);
   MPI_APSP(result, n);
   gettimeofday(&tvAfter, NULL);
   printf("Elapsed time = %ld usecs\n", (tvAfter.tv_sec - tvBefore.tv_sec) * 1000000 + tvAfter.tv_usec - tvBefore.tv_usec);

   DEBUG_LOG("Parallelized algorithm result.\n");
   PRINT_MAT(result, n);

   //compare your result with reference result
   if(CmpArray(result, ref, n*n))
      printf("Your result is correct.\n");
   else
      printf("Your result is wrong.\n");
}

void node(int argc, char **argv)
{
   MPI_APSP(NULL, n);
}

int main(int argc, char **argv)
{
   if(argc != 2)
   {
      printf("Usage: test {N}\n");
      exit(-1);
   }

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &commSize);

   n = atoi(argv[1]);
   root = commSize - 1;	

   if(rank == root) master(argc, argv);
   else node(argc, argv);

   MPI_Finalize();
   return 0;
}
