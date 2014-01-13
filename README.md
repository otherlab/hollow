Hollow: Solid simulation based on Geode, PETSc, and PetIGA
==========================================================

Hollow is a solid simulator for use on various engineering engineering projects
at Otherlab.  The license is standard two-clause BSD (see the included `LICENSE`
file or [LICENSE](https://github.com/otherlab/hollow/blob/master/LICENSE)).

For questions or discussion, email geode-dev@googlegroups.com.

### Dependencies

Like geode, hollow is a mixed python/c++ codebase.  The dependencies are

* [geode](https://github.com/otherlab/geode): Otherlab computational geometry library
* [petsc](http://www.mcs.anl.gov/petsc): Portable Extensible Toolkit for Scientific Computation
* [petiga](https://bitbucket.org/dalcinl/petiga): High performance isogeometric analysis
* [igakit](https://bitbucket.org/dalcinl/igakit): Geometry generation and visualization for PetIGA
* [mayavi2](http://docs.enthought.com/mayavi/mayavi): Python scientific data visualization

### Installation

Install geode and its dependencies via the instructions at https://github.com/otherlab/geode.

Installation of mayavi and mercurial (for petiga and igakit) varies with platform:

    # Debian/Ubuntu
    sudo apt-get install mayavi2 mercurial flex gfortran openmpi-bin libopenmpi-dev

    # Homebrew (recommended for Mac)
    brew install homebrew/versions/vtk5 --with-qt
    brew install openmpi freetype
    pip install --upgrade mayavi mercurial python-dateutil matplotlib

    # MacPorts (not recommended)
    sudo port -v install vtk5 +python27
    sudo port -v install mercurial py27-mayavi
    sudo ln -s /opt/local/bin/gfortran-mp-4.7 /opt/local/bin/gfortran # So that python can find it
    sudo port -v install mpich-default
    sudo port select --set mpich mpich-mp-fortran gcc48

Hollow depends on unreleased features of petsc, so a specific branch is required:

    # Download petsc
    git clone https://bitbucket.org/petsc/petsc.git
    cd petsc

    # Switch to the hollow branch
    git checkout -b hollow origin/irving/hollow

    # Configure and build debug and release versions.
    # IMPORTANT: On MacPorts, add --with-mpi-dir=/opt/local
    ./configure --with-petsc-arch=debug   --with-debugging=1
    ./configure --with-petsc-arch=release --with-debugging=0
    export PETSC_DIR=`pwd`
    make PETSC_ARCH=debug
    make PETSC_ARCH=release
    cd ..

Download and build petiga and igakit:

    # Download petiga and igakit
    git clone git@github.com:otherlab/petiga.git
    hg clone https://bitbucket.org/dalcinl/igakit

    # Build petiga (uses PETSC_DIR environment variable set above)
    cd petiga
    make PETSC_ARCH=debug
    make PETSC_ARCH=release

    # Build and install igakit
    cd ../igakit
    sudo python setup.py install

Download, configure, build, and test hollow:

    # Download hollow
    git clone git@github.com:otherlab/hollow.git
    cd hollow

    # If you used a nontrivial `config.py` file for geode, reuse it via
    ln -s <path-to-geode>/config.py

    # Add the following lines to hollow/config.py
    petsc_base = '<path-to-petsc>'
    petiga_base = '<path-to-petiga>'
    petsc_include = [petsc_base+'/include',petsc_base+'/$type/include']
    petsc_libpath = [petsc_base+'/$type/lib']
    petiga_include = [petiga_base+'/include']
    petiga_libpath = [petiga_base+'/$type/lib']

    # Build hollow and setup development symlinks (add type=debug for debug)
    scons -j5
    sudo scons develop
    sudo python setup.py develop

    # Run unit tests
    py.test

### Usage

To simulate a bent tube, run something like

    bend-tube -o output --length 10 --thickness .25

To visualize the results, run

    bend-tube -o output --view 1

This can be run anytime during the simulation, and will show the most recent frame.
To see a different frame, add `--frame 7` or the like.
