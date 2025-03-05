# SLURM and Module System on HPC

The Lmod modules system on the HPC system enables users to easily set their environments for selected software and to choose versions if appropriate.

The Lmod system is hierarchical; not every module is available in every environment. We provide a core environment that contains most of the software installed by Research Computing staff, but software that requires a compiler or MPI is not in that environment, and a compiler must first be loaded.

## Basic Commands

### List all available software
```bash
module avail
```
Use `module key` to list all modules in a particular category. The current choices are:

- base, bio, cae, chem, compiler, data, debugger, devel, geo, ide, lang, lib, math, mpi, numlib, perf, phys, system, toolchain, tools, vis, licensed

Example:
```bash
module key bio
```

### Load the environment for a particular package
```bash
module load thepackage
```
If you do not specify a version, the system default is loaded. For example, to load the default version of our Python distribution, run:
```bash
module load miniforge
```
To specify a particular version explicitly:
```bash
module load gcc/13.3.0
```

### Remove a module
```bash
module unload thepackage
```

### List all modules loaded in the current shell
```bash
module list
```

### Change from one version to another
```bash
module swap oldpackage newpackage
```
For example:
```bash
module swap gcc/11.4.0 intel/2023.1
```

### Clear all loaded modules
```bash
module purge
```

### Finding prerequisites
```bash
module spider
```
For a specific package:
```bash
module spider hdf5
```

To check specific versions of a package, e.g., R:
```bash
module spider R
```
Example output:
```
Versions:
  R/3.2.1
  R/3.4.4
  R/3.5.3
  R/3.6.3
  R/4.0.0
```
To see how to load a particular version:
```bash
module spider R/3.6.3
```
If prerequisites are required, load them first:
```bash
module load gcc/7.1.0
module load openmpi/3.1.4
module load R/3.6.3
```

## Modules Best Practices

- **Start with a clean slate:** Run `module purge` before beginning your workflow.
- **Specify module versions:** Avoid relying on defaults, as they may change.
- **Use `module spider`** to check available versions.

## Advanced Usage

### Using a Bash Script for Modules
Instead of adding modules to `.bashrc`, use a separate script:
```bash
module purge
module load gcc/7.1.0
module load openmpi/3.1.4
module load R/3.6.3
```
Run it as needed:
```bash
source mymodules.sh
```

## Modules in Job Scripts
In SLURM job scripts, include modules before executing the job. Example:
```bash
#!/bin/bash
#SBATCH -p standard
#SBATCH -A MyAcct
#SBATCH -n 1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=4000

module purge
module load gcc/7.1.0
module load openmpi/3.1.4
module load R/3.6.3

Rscript myScript.R
```

## Creating Your Own Modules
If installing your own software, create custom modules in your home directory:
```bash
mkdir $HOME/modulefiles
```
Download and install software, ensuring installation is in a non-administrator directory:
```bash
wget https://www.kernel.org/pub/software/scm/git/git-2.6.2.tar.gz
tar xf git-2.6.2.tar.gz
cd git-2.6.2
./configure --prefix=$HOME/git/2.6.2
make
make install
```
Create a modulefile:
```bash
mkdir -p $HOME/modulefiles/git
cd $HOME/modulefiles/git
```
Create `2.6.2.lua` with the following content:
```lua
local home    = os.getenv("HOME")
local version = myModuleVersion()
local pkgName = myModuleName()
local pkg     = pathJoin(home, pkgName, version, "bin")
prepend_path("PATH", pkg)
```
Use the module:
```bash
module use $HOME/modulefiles
module load git
which git
```

### Finding Modules with the Same Name
```bash
module avail git
```
If multiple versions exist, the highest is loaded by default.
```bash
module load git
```

For more details, refer to the HPC documentation.

