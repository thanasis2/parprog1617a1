#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Compile with: gcc -O2 -Wall float-triad-normal.c -o float-triad-normal -DN=.. -DR=..

// use -DN=.. and -DR=... at compilation
//#define N 1000000
//#define M 10

void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}

int main() {
float *a,*b,*c,*d;
int i,j;
double ts,te,mflops;

  // allocate test arrays
  a = (float *)malloc(N*sizeof(float));
  if (a==NULL) exit(1);
  b = (float *)malloc(N*sizeof(float));
  if (b==NULL) { free(a); exit(1); }
  c = (float *)malloc(N*sizeof(float));
  if (c==NULL) { free(a); free(b); exit(1); }
  d = (float *)malloc(N*sizeof(float));
  if (d==NULL) { free(a); free(b); free(c); exit(1); }
  
  //initialize all arrays - cache warm-up
  for (i=0;i<N*M;i++) {
    a[i]=0.0; b[i]=1.0; c[i]=2.0; d[i]=3.0;
  }
 
  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // do triad artificial work
  for (j=0;j<R;j++) {
    for (i=0;i<N;i++) {
      a[i] = b[i]+c[i]*d[i];
    }
  }
 
  // get ending time
  get_walltime(&te);
  
  // compute mflops/sec (2 operations per R*N passes)
  mflops = (R*N*2.0)/((te-ts)*1e6);
  
  printf("MFLOPS/sec = %f\n",mflops);
  
  // free arrays
  free(a); free(b); free(c); free(d);
  
  return 0;
}

