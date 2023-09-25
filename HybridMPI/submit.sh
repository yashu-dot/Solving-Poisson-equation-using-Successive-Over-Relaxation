#!/bin/bash

for j in {1,2,4,8,16,32};do
	echo $j
	for i in {1,2,4,8,16,32,64,128}; do
		echo $i
		export OMP_NUM_THREADS=$i
		mpiexec -n $j ./jacobi_hybrid_mpi.x -c < input
	done
done