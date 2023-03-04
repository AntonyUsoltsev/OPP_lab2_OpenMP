# OPP_lab2_OpenMP
Parallelization of a program using OpenMp

Compile command

    gcc -fopenmp task1.c -o task1
    gcc -fopenmp task2.c -o task2

Run command:
    
    OMP_NUM_THREADS=num ./task1
    OMP_NUM_THREADS=num ./task2
    *num - number of threads on which need to run the program
    

