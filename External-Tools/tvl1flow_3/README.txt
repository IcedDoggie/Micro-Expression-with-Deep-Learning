TVL1FLOW

A program for optical flow estimation based on total variation and the L1 norm.

This program is part of an IPOL publication:
http://www.ipol.im/pub/algo/smf_tl1_optical_flow_estimation

Javier Sánchez Pérez <jsanchez@dis.ulpgc.es> CTIM, Universidad de Las Palmas de Gran Canaria
Enric Meinhardt Llopis <enric.meinhardt@cmla.ens-cachan.fr> CMLA, ENS Cachan
Gabriele Facciolo <gabriele.facciolo@cmla.ens-cachan.fr> CMLA, ENS Cachan

Version 1, released on November 18, 2011

This software is distributed under the terms of the BSD license (see file
license.txt)

Required libraries: libpng, libtiff

Compilation instructions: run "make" to produce an executable named "tvl1flow"

Usage instructions:

./tvl1flow I0.png I1.png [out.flo NPROCS TAU LAMBDA THETA NSCALES ZOOM NWARPS EPSILON VERBOSE]

where the parameters between brackets are optional and

I0.png: first input image
I1.png: second input image
out.flo: output optical flow

NPROCS is the number of processors to use (NPROCS=0, all processors available)
TAU is the time step (e.g., 0.25)
LAMBDA is the data attachment weight (e.g., 0.15)
THETA is the tightness of the relaxed functional (e.g., 0.3)
NSCALES is the requested number of scales (e.g., 5)
ZOOM is the zoom factor between each scale (e.g., 0.5)
NWARPS is the number of warps per iteration (e.g., 5)
EPSILON is the stopping criterion threshold (e.g., 0.01)
VERBOSE is for verbose mode (e.g., 1 for verbose)

Simple example:

./tvl1flow I0.png I1.png

Example with reasonable default parameters:

./tvl1flow I0.png I1.png out.flo 0 0.25 0.15 0.3 5 0.5 5 0.01

