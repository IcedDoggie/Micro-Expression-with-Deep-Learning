
// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2011, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// All rights reserved.

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifndef DISABLE_OMP
#include <omp.h>
#endif//DISABLE_OMP

#include "iio.h"

#include "tvl1flow_lib.c"


#define PAR_DEFAULT_OUTFLOW "flow.flo"
#define PAR_DEFAULT_NPROC   0
#define PAR_DEFAULT_TAU     0.25
#define PAR_DEFAULT_LAMBDA  0.15
#define PAR_DEFAULT_THETA   0.3
#define PAR_DEFAULT_NSCALES 100
#define PAR_DEFAULT_ZFACTOR 0.5
#define PAR_DEFAULT_NWARPS  5
#define PAR_DEFAULT_EPSILON 0.01
#define PAR_DEFAULT_VERBOSE 0


/**
 *
 *  Function to read images using the iio library
 *  It always returns an allocated the image.
 *
 */
static float *read_image(const char *filename, int *w, int *h)
{
	float *f = iio_read_image_float(filename, w, h);
	if (!f)
		fprintf(stderr, "ERROR: could not read image from file "
				"\"%s\"\n", filename);
	return f;
}


/**
 *
 *  Main program:
 *   This program reads the following parameters from the console and
 *   then computes the optical flow:
 *   -nprocs      number of threads to use (OpenMP library)
 *   -I0          first image
 *   -I1          second image
 *   -tau         time step in the numerical scheme
 *   -lambda      data term weight parameter
 *   -theta       tightness parameter
 *   -nscales     number of scales in the pyramidal structure
 *   -zfactor     downsampling factor for creating the scales
 *   -nwarps      number of warps per scales
 *   -epsilon     stopping criterion threshold for the iterative process
 *   -out         name of the output flow field
 *   -verbose     switch on/off messages
 *
 */
int main(int argc, char *argv[])
{
	if (argc < 3) {
		fprintf(stderr, "Usage: %s I0 I1 [out "
		//                       0 1  2   3
		"nproc tau lambda theta nscales zfactor nwarps epsilon "
		//  4  5   6      7     8       9       10     11
		"verbose]\n", *argv);
		// 12
		return EXIT_FAILURE;
	}

	//read the parameters
	int i = 1;
	char* image1_name  = argv[i]; i++;
	char* image2_name  = argv[i]; i++;
	char* outfile = (argc>i)? argv[i]: PAR_DEFAULT_OUTFLOW;       i++;
	int   nproc   = (argc>i)? atoi(argv[i]): PAR_DEFAULT_NPROC;   i++;
	float tau     = (argc>i)? atof(argv[i]): PAR_DEFAULT_TAU;     i++;
	float lambda  = (argc>i)? atof(argv[i]): PAR_DEFAULT_LAMBDA;  i++;
	float theta   = (argc>i)? atof(argv[i]): PAR_DEFAULT_THETA;   i++;
	int   nscales = (argc>i)? atoi(argv[i]): PAR_DEFAULT_NSCALES; i++;
	float zfactor = (argc>i)? atof(argv[i]): PAR_DEFAULT_ZFACTOR; i++;
	int   nwarps  = (argc>i)? atoi(argv[i]): PAR_DEFAULT_NWARPS;  i++;
	float epsilon = (argc>i)? atof(argv[i]): PAR_DEFAULT_EPSILON; i++;
	int   verbose = (argc>i)? atoi(argv[i]): PAR_DEFAULT_VERBOSE; i++;

	//check parameters
	if (nproc < 0) {
		nproc = PAR_DEFAULT_NPROC;
		if (verbose) fprintf(stderr, "warning: "
				"nproc changed to %d\n", nproc);
	}
	if (tau <= 0 || tau > 0.25) {
		tau = PAR_DEFAULT_TAU;
		if (verbose) fprintf(stderr, "warning: "
				"tau changed to %g\n", tau);
	}
	if (lambda <= 0) {
		lambda = PAR_DEFAULT_LAMBDA;
		if (verbose) fprintf(stderr, "warning: "
				"lambda changed to %g\n", lambda);
	}
	if (theta <= 0) {
		theta = PAR_DEFAULT_THETA;
		if (verbose) fprintf(stderr, "warning: "
				"theta changed to %g\n", theta);
	}
	if (nscales <= 0) {
		nscales = PAR_DEFAULT_NSCALES;
		if (verbose) fprintf(stderr, "warning: "
				"nscales changed to %d\n", nscales);
	}
	if (zfactor <= 0 || zfactor >= 1) {
		zfactor = PAR_DEFAULT_ZFACTOR;
		if (verbose) fprintf(stderr, "warning: "
				"zfactor changed to %g\n", zfactor);
	}
	if (nwarps <= 0) {
		nwarps = PAR_DEFAULT_NWARPS;
		if (verbose) fprintf(stderr, "warning: "
				"nwarps changed to %d\n", nwarps);
	}
	if (epsilon <= 0) {
		epsilon = PAR_DEFAULT_EPSILON;
		if (verbose) fprintf(stderr, "warning: "
				"epsilon changed to %f\n", epsilon);
	}

#ifndef DISABLE_OMP
	if (nproc > 0)
		omp_set_num_threads(nproc);
#endif//DISABLE_OMP

	// read the input images
	int    nx, ny, nx2, ny2;
	float *I0 = read_image(image1_name, &nx, &ny);
	float *I1 = read_image(image2_name, &nx2, &ny2);

	//read the images and compute the optical flow
	if (nx == nx2 && ny == ny2)
	{
		//Set the number of scales according to the size of the
		//images.  The value N is computed to assure that the smaller
		//images of the pyramid don't have a size smaller than 16x16
		const float N = 1 + log(hypot(nx, ny)/16.0) / log(1/zfactor);
		if (N < nscales)
			nscales = N;

		if (verbose)
			fprintf(stderr,
				"nproc=%d tau=%f lambda=%f theta=%f nscales=%d "
				"zfactor=%f nwarps=%d epsilon=%g\n",
				nproc, tau, lambda, theta, nscales,
				zfactor, nwarps, epsilon);

		//allocate memory for the flow
		float *u = xmalloc(2 * nx * ny * sizeof*u);
		float *v = u + nx*ny;;

		//compute the optical flow
		Dual_TVL1_optic_flow_multiscale(
				I0, I1, u, v, nx, ny, tau, lambda, theta,
				nscales, zfactor, nwarps, epsilon, verbose
		);

		//save the optical flow
		iio_save_image_float_split(outfile, u, nx, ny, 2);

		//delete allocated memory
		free(I0);
		free(I1);
		free(u);
	} else {
		fprintf(stderr, "ERROR: input images size mismatch "
				"%dx%d != %dx%d\n", nx, ny, nx2, ny2);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
