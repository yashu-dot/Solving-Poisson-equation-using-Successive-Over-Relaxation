#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>


double checkSolution(double xStart, double yStart,
                     int maxXCount, int maxYCount,
                     double *u,
                     double deltaX, double deltaY,
                     double alpha, double *fx_thing, double *fy_thing)
{
#define U(XX,YY) u[(YY)*maxXCount+(XX)]
    int x, y;
    double fX, fY;
    double localError, error = 0.0;

    for (y = 1; y < (maxYCount-1); y++)
    {
        for (x = 1; x < (maxXCount-1); x++)
        {
            localError = U(x,y) - fx_thing[x-1]*fy_thing[y-1];
            error += localError*localError;
        }
    }
    return sqrt(error)/((maxXCount-2)*(maxYCount-2));
}


int main(int argc, char **argv)
{
    int n, m, mits;
    double alpha, tol, relax;
    double maxAcceptableError;
    double error;
    double *u, *u_old, *tmp;
    int allocCount;
    int iterationCount, maxIterationCount;
    double t1, t2;

    scanf("%d,%d", &n, &m);

    scanf("%lf", &alpha);

    scanf("%lf", &relax);

    scanf("%lf", &tol);

    scanf("%d", &mits);



    allocCount = (n+2)*(m+2);

    u = 	(double*)calloc(allocCount, sizeof(double));
    u_old = (double*)calloc(allocCount, sizeof(double));

    
    maxIterationCount = mits;
    maxAcceptableError = tol;


    double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;

    double deltaX = (xRight-xLeft)/(n-1);
    double deltaY = (yUp-yBottom)/(m-1);

    iterationCount = 0;
    error = HUGE_VAL;

    double *fx_thing = (double*)malloc(n*sizeof(double));
    double *fy_thing = (double*)malloc(m*sizeof(double));

    if (u == NULL || u_old == NULL || fx_thing == NULL || fy_thing == NULL) {
        printf("Not enough memory for two %ix%i matrices\n", n+2, m+2);
        exit(1);
    }



    int x,y;
    for (x = 1; x < n+1; x++) {
        double fX = xLeft + (x-1)*deltaX;
        fx_thing[x-1] = 1.0-fX*fX;
    }

    for (y = 1; y < m+1; y++) {
        double fY = yBottom + (y-1)*deltaY;
        fy_thing[y-1] = 1.0-fY*fY;
    }

    double cx = 1.0/(deltaX*deltaX);
    double cy = 1.0/(deltaY*deltaY);
    double cc = -2.0*cx-2.0*cy-alpha;

    #define SRC(XX,YY) u_old[(YY)*(n+2)+(XX)]
    #define DST(XX,YY) u[(YY)*(n+2)+(XX)]
    double updateVal, f;

    clock_t start = clock(), diff;
    
    MPI_Init(NULL,NULL);
    t1 = MPI_Wtime();

    
    while (iterationCount < maxIterationCount && error > maxAcceptableError)
    {    	
        error = 0.0;
        for (y = 1; y < m+1; y++)
        {
            for (x = 1; x < n+1; x++)
            {
                f = -alpha*fx_thing[x-1]*fy_thing[y-1] - 2.0*fx_thing[x-1] - 2.0*fy_thing[y-1];
                updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx +
                                (SRC(x,y-1) + SRC(x,y+1))*cy +
                                SRC(x,y)*cc - f
                            )/cc;
                DST(x,y) = SRC(x,y) - relax*updateVal;
                error += updateVal*updateVal;
            }
        }
        error = sqrt(error)/(n*m);
        iterationCount++;
       
        tmp = u_old;
        u_old = u;
        u = tmp;
    }

    t2 = MPI_Wtime();
    printf( "Iterations=%3d Elapsed MPI Wall time is %f\n", iterationCount, t2 - t1 ); 
    MPI_Finalize();
    
    
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
    printf("Residual %g\n",error);

  
    double absoluteError = checkSolution(xLeft, yBottom,
                                         n+2, m+2,
                                         u_old,
                                         deltaX, deltaY,
                                         alpha, fx_thing, fy_thing);
    printf("The error of the iterative solution is %g\n", absoluteError);

    free(fy_thing);
    free(fx_thing);

    return 0;
}
