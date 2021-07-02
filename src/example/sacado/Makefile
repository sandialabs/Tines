# your kokkos install directory
KOKKOS_INSTALL_PATH=$(HOME)/Work/lib/kokkos/install/coffee/release

EXE = Tines_ToySacadoRayleighQuotient

# CXX and CXX flags may need to be modified according to users compiler setting
# openmp flag is required if kokkos openmp space is enalbed when kokkos is installed
CXX = clang++-mp-11
CXXFLAGS = -std=c++14 -fopenmp

CPPFLAGS = \
	-I../../core/sacado \
	-I../../core/sacado/mpl \
	-I../../core/sacado/experimental \
	-I$(KOKKOS_INSTALL_PATH)/include
LINK = ${CXX}

# for some architecture, kokkos library is installed under lib64 directory
# please assure that. with opnemp enabled, an openmp flag is also required.
LINKFLAGS = -fopenmp -L$(KOKKOS_INSTALL_PATH)/lib -lkokkoscore -lkokkoscontainers


default: $(EXE).o
	$(LINK) $(LINKFLAGS) $(EXE).o -o $(EXE).x

%.o:%.cpp 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

clean: 
	rm -f *.o *.x
