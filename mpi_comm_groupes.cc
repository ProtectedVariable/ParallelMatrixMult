#include <iostream>
#include <cstdlib>
#include <cmath>
#include <unistd.h>
#include <mpi.h>
#include <cstring>
using namespace std;


int main(int argc,char** argv) {
   MPI_Init(&argc,&argv);
   int nbPE,myPE;
   MPI_Comm_size(MPI_COMM_WORLD,&nbPE);
   MPI_Comm_rank(MPI_COMM_WORLD,&myPE);
   MPI_Comm comm;
   // nbPE = 8
   MPI_Comm_split(MPI_COMM_WORLD,myPE%3,myPE,&comm);
   // partition 0 de comm: 0 3 6, identifiants locaux: 0 1 2
   // partition 1 de comm: 1 4 7, identifiants locaux: 0 1 2
   // partition 2 de comm: 2 5, identifiants locaux: 0 1
   int localPE;
   MPI_Comm_rank(comm,&localPE);
   int a = 666*myPE;
   usleep(myPE*10000);
   cout << myPE << ": " << localPE << ": " << a << endl;
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Bcast(&a,1,MPI_INT,0,comm);
   usleep(myPE*10000);
   cout << myPE << ": " << localPE << ": " << a << endl;
   MPI_Finalize();
   return 0;
}

   
   
