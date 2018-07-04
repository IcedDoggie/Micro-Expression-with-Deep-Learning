for i in "c99" "clang" "sunc99" "gcc -std=c99" "icc -std=c99 -Wno-unknown-pragmas" ; do
	make clean
	make OMPFLAGS="" CFLAGS=-DDISABLE_OMP CC="$i"
done
