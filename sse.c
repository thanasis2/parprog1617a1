// Compile with  gcc -O2 week1.c -o week1 -DN=12 -DR=1000000
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <emmintrin.h> // SSE2

#define N 1000000
#define M 10

void get_walltime(double *wct) {
	struct timeval tp;

	gettimeofday(&tp, NULL);
	*wct = (double)(tp.tv_sec + tp.tv_usec / 1000000.0);
}

int main() {
	int row, col, i,j,k;
	double ts, te, time, t;

	float *a;
	float *b;
	float *c, *d, *ar0, *bc0, *bc1, *bc2, *bc3, *cr0, *tar0;
	__m128 a0, b0, b1, b2, b3, c0, c1, c2, c3, t0, t1, t2, *va, *vb, *vc, *vd;
	
	i = posix_memalign((void **)&a, 16, N* N * sizeof(float));
	if(i != 0) {
		exit(1);
	}
	i = posix_memalign((void **)&b, 16, N* N * sizeof(float));
	if(i != 0) {
		free(a);
		exit(1);
	}
	i = posix_memalign((void **)&c, 16, N* N * sizeof(float));
	if(i != 0) {
		free(a);
		free(b);
		exit(1);
	}
	i = posix_memalign((void **)&d, 16, N* N * sizeof(float));
	if(i != 0) {
		free(a);
		free(b);
		free(c);
		exit(1);
	}
	
	for(i = 0;i < N*N;i++) { // Initialise
		a[i] = 0.0;
		b[i] = 1.0;
		c[i] = 2.0;
		d[i] = 3.0;
	}
	
	get_walltime(&ts); // Start clock
	ar0=a;
	tar0=a;
	bc0=b;
	bc1=b+N;
	bc2=b+2*N;
	bc3=b+3*N;
	cr0=c;
	for(k=0;k<N;k++){
		for(j=0;j<N;j+=4){
			c0=_mm_xor_ps(c0,c0);
			c1=_mm_xor_ps(c1,c1);
			c2=_mm_xor_ps(c2,c2);
			c3=_mm_xor_ps(c3,c3);

			for(i=0;i<N;i+=4){
				a0 = _mm_load_ps(ar0);
				b0 = _mm_load_ps(bc0);
				b1 = _mm_load_ps(bc1);
				b2 = _mm_load_ps(bc2);
				b3 = _mm_load_ps(bc3);
		
				c0=_mm_add_ps(c0, _mm_mul_ps(a0, b0));
				c1=_mm_add_ps(c1, _mm_mul_ps(a0, b1));
				c2=_mm_add_ps(c2, _mm_mul_ps(a0, b2));
				c3=_mm_add_ps(c3, _mm_mul_ps(a0, b3));

				ar0 +=4;
				bc0 +=4;
				bc1 +=4;
				bc2 +=4;
				bc3 +=4;
			}
		t0=_mm_add_ps(_mm_unpackhi_ps(c0,c1), _mm_unpacklo_ps(c0,c1));
		t1=_mm_add_ps(_mm_unpackhi_ps(c2,c3), _mm_unpacklo_ps(c2,c3));
		t2=_mm_add_ps(_mm_unpackhi_ps(t0,t1), _mm_unpacklo_ps(t0,t1));
		_mm_store_ps(cr0,_mm_shuffle_ps(t2,t2,0xd8));
		cr0 +=4;
		ar0=tar0;
		bc0 = bc3;
		bc1 = bc0+N;
		bc2 = bc1+N;
		bc3 = bc2+N;
		}
		tar0+=N;
		ar0=tar0;
		bc0=b;
		bc1=b+N;
		bc2=b+2*N;
		bc3=b+3*N;
	}
	get_walltime(&te); // Stop clock

	mflops=(R*N*2.0)/((te-ts)*1e6);
	printf("MFLOPS/sec = %f\n", mflops);
	time = te - ts;
	printf("time: %f\n", time);

	free(a);
	free(b);
	free(c);
	free(d);
	
	return 0;
}

