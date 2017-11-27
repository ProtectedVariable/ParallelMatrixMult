#include <iostream>
#include <cstdlib>
#include <cmath>
#include <unistd.h>
#include <mpi.h>
#include <cstring>
using namespace std;

/*
Les sous-matrices <mat> de dimension <nloc>x<nloc> sur les <nbPE> processeurs
du groupe de communication <comm> sont rassemblées sur le processeur 0
qui les affiche en une grande matrice de dimension (<p>*<nloc>) x (<p>*<nloc>)
*/
void printAll(int* mat,int nloc,MPI_Comm comm,string label) { 
   int nbPE,globalPE;
   MPI_Comm_size(comm,&nbPE);
   MPI_Comm_rank(MPI_COMM_WORLD,&globalPE);
   int* recv_buf = new int[nbPE*nloc*nloc];
   MPI_Gather(mat,nloc*nloc,MPI_INT,recv_buf,nloc*nloc,MPI_INT,0,comm);
   if (globalPE == 0) {
      int p = sqrt(nbPE+0.1);
      cout << label;
      for (int global=0;global<(p*nloc)*(p*nloc);global++) {
         int global_i = global/(p*nloc); int global_j = global%(p*nloc);
         int pe_i = global_i/nloc; int pe_j = global_j/nloc;
         int local_i = global_i%nloc; int local_j = global_j%nloc;
         int pe = pe_i*p+pe_j; int local = local_i*nloc+local_j;
         cout << recv_buf[pe*nloc*nloc + local] << " ";
         if ((global+1)%(p*nloc) == 0) cout << endl;
      }
   }
   delete recv_buf; 
}
/* 
Allocation et initialisation aléatoire d'une matrice de dimension <size>x<size>
avec des valeurs entières entre <inf> et <sup>; cette matrice est retournée
*/
int* randomInit(int size,int inf,int sup) {
   int* mat = new int[size*size];
   for (int i=0;i<size*size;i++) mat[i] = inf+rand()%(sup-inf);
   return mat;
}

/*
Algorithmes de multiplication matricielle parallèle
*/
void fox(int* matLocA,int* matLocB,int* matLocC,int nloc) { /*à compléter*/ }
void cannon(int* matLocA,int* matLocB,int* matLocC,int nloc) { /*à compléter*/ }
void dns(int* matLocA,int* matLocB,int* matLocC,int nloc) { /*à compléter*/ }

int main(int argc,char** argv) {
   MPI_Init(&argc,&argv);
   int nbPE,myPE;
   MPI_Comm_size(MPI_COMM_WORLD,&nbPE);
   MPI_Comm_rank(MPI_COMM_WORLD,&myPE);
   MPI_Comm comm_i_cte = MPI_COMM_WORLD, 
            comm_j_cte = MPI_COMM_WORLD,
            comm_k_cte = MPI_COMM_WORLD;
   
   int algo = atoi(argv[1]); // 1 = fox, 2 = cannon, 3 = dns
   srand(atoi(argv[2])+myPE);   
   int nloc = atoi(argv[3]);

   // sous-matrices de dimension <nloc> x <nloc> 
   // des matrices A, B et C de dimension (<p>*<nloc>) x (<p>*<nloc>)
   int* matLocA = randomInit(nloc,-10,10);
   int* matLocB = randomInit(nloc,-10,10);   
   int* matLocC = new int[nloc*nloc];   

   switch(algo) {
   case 1: fox(matLocA,matLocB,matLocC,nloc); break;
   case 2: cannon(matLocA,matLocB,matLocC,nloc); break;
   case 3: // p^3 processeurs P_{ijk} avec i,j,k = 0,...,p-1
      int p = pow(nbPE+0.1,1.0/3.0);
      int j = (myPE/p)%p;  // A_{ik} sur les P_{i0k}
      MPI_Comm_split(MPI_COMM_WORLD,j,myPE,&comm_j_cte);
      int i = myPE/(p*p); // B_{kj} sur les P_{0jk} 
      MPI_Comm_split(MPI_COMM_WORLD,i,(myPE%p)*p+myPE/p,&comm_i_cte);
      int k = myPE%p;      // C_{ij} sur les P_{ij0} 
      MPI_Comm_split(MPI_COMM_WORLD,k,myPE,&comm_k_cte); 
      dns(matLocA,matLocB,matLocC,nloc);
   }
   printAll(matLocA,nloc,comm_j_cte,"%matrice complete A\n");
   printAll(matLocB,nloc,comm_i_cte,"%matrice complete B\n");
   printAll(matLocC,nloc,comm_k_cte,"%matrice complete C\n");

   MPI_Finalize();
   delete matLocA, matLocB, matLocC;
   return 0;
}

/*
Exemple pour l'algorithme DNS avec p=3 (p^3 = 27 processeurs, pe = 0,1,...,p^3-1)

                              j  
                             ^
                            /
                           /----->  
                           |    k
                         i |           j
                           v          ^
                                     /
                                    /
                  /|               /--------------------------/|
                 / | ...          /        /        /        / |
                /  |             /  B02   /  B12   /  B22   /  | 
               /   |            /        /        /        /   |
              /|C02|           /--------------------------/|   |
             / |  /|          /        /        /        / |  /|
            /  | / | ...     /  B01   /  B11   /  B21   /  | / |    
           /   |/  |        /        /        /        /   |/  |
          /|C01|C12|       /--------------------------/|   |   |
         / |  /|  /|      /        /        /        / |  /|  /|
        /  | / | / | ... /  B00   /  B10   /  B20   /  | / | / |
       /   |/  |/  |    /        /        /        /   |/  |/  |
      /C00 |C11|C22/   /----------------------------> k|   |   /
      |   /|  /|  /    |        |        |        |   /|  /|  /
      |  / | / | / ... |  A00   |  A01   |  A02   |  / | / | /
      | /  |/  |/      |        |        |        | /  |/  |/
      |/C10|C21/       |--------|--------|--------|/   |   /
      |   /|  /        |        |        |        |   /|  /
      |  / | /  ...    |  A10   |  A11   |  A12   |  / | /
      | /  |/          |        |        |        | /  |/
      |/C20|           |--------|--------|--------|/   |
      |   /            |        |        |        |   /
      |  /  ...        |  A20   |  A21   |  A22   |  /
      | /              |        |        |        | /
      |/               |--------|--------|--------|/  
                       |
                       |
                       v
                       
                       i

Initialement, Aik est sur le processeur P_{i0k} et Bkj est sur P_{0jk}
Broadcast de Aik dans la direction j par P_{i0k} aux p processeurs P_{ijk}
Broadcast de Bkj dans la direction i par P_{0jk} aux p processeurs P_{ijk}
Multiplication Aik*Bkj par le processeur P_{ijk}
Réduction dans la direction k en rapatriant et sommant les Aik*Bkj:
Cij = Sum_{k=0}^{p-1} Aik*Bkj sur P_{ij0}
 
=======================================================
i = pe/9

i=0            
     6--7--8   
    /| /| /|   
   3--4--5     
  /| /| /|      
 0--1--2       
 |  |  |
 
i=1
      |   |   |
     15--16--17
    |/  |/ | /|
   12--13--14 |
  |/  |/  |/| /
  9--10--11 |/
  |   |   | /
  |___|___|/
 
i=2
      |   |   |
     24--25--26
    |/  |/ | /
   21--22--23 
  |/  |/  |/
 18--19--20
 
=======================================================
j = (pe/3)%3

j=0

   /   /   /
  0---1---2
  |/  |/  |/
  9--10--11
  |/  |/  |/
 18--19--20

j=1

   /   /   /
  3---4---5
 /|/  |/  |/
 12--13--14
 /|/  |/  |/
 21--22--23
 /   /   /
 
j=2

  6---7---8
 /|  /|  /| 
 15--16--17
 /|  /|  /|
 24--25--26
 /   /   /
 
=======================================================
k = pe%3

k=0            
         6--- 
        /|
       / |
      3- 15--  
     /| /|  
    / |/ |
   0- 12 24--      
   | /| /
   |/ |/
   9- 21--
   | /
   |/
   18--

k=1
         ---7--- 
           /|
          / |
      ---4- 16--  
        /| /|  
       / |/ |
   ---1- 13 25--      
      | /| /
      |/ |/
   --10 -22--
      | /
      |/
   --19--
 
k=2
            ---8 
              /|
             / |
         ---5  17 
           /| /|  
          / |/ |
      ---2  14 26      
         | /| /
         |/ |/
      --11 -23
         | /
         |/
      --20
      
===========================================================
Exemple avec p = 3

    	0	1	2	3	4	5	6	7	8	
---------------------------------
i=0:	0	1	2	3	4	5	6	7	8	    i = pe/(p*p)
i=1:	9	10	11	12	13	14	15	16	17	
i=2:	18	19	20	21	22	23	24	25	26
      j*p+k = pe%(p*p) 

    	0	1	2	3	4	5	6	7	8	
---------------------------------
j=0:	0	1	2	9	10	11	18	19	20	    j = (pe/p)%p ou j = (pe%(p*p))/p
j=1:	3	4	5	12	13	14	21	22	23	
j=2:	6	7	8	15	16	17	24	25	26	
      i*p+k = (pe/(p*p))*p+pe%p

    	0	1	2	3	4	5	6	7	8	
---------------------------------
k=0:	0	3	6	9	12	15	18	21	24	    k = pe%p
k=1:	1	4	7	10	13	16	19	22	25	
k=2:	2	5	8	11	14	17	20	23	26	
      i*p+j = pe/p

*/
