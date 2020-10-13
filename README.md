# Neural Network FDK algorithm

Support code for the Neural Network FDK algorithm paper

This paragraph should contain a high-level description of the package, with a
brief overview of its features and limitations.


* Free software: GNU General Public License v3
* Documentation: [https://mjlagerwerf.github.io/nn_fdk]


## Getting Started

It takes a few steps to setup Neural Network FDK algorithm on your
machine. We recommend installing
[Anaconda package manager](https://www.anaconda.com/download/) for
Python 3.


### Installing from source

To install Neural Network FDK algorithm, simply clone this GitHub
project. Go to the cloned directory and run PIP installer:
```
git clone https://github.com/mjlagerwerf/nn_fdk.git
cd nn_fdk
make install

git clone https://github.com/mjlagerwerf/nn_fdk.git
cd nn_fdk
make install
```
Instal it as editable 
```
git clone https://github.com/mjlagerwerf/nn_fdk.git
cd nn_fdk
make install_dev

git clone https://github.com/mjlagerwerf/nn_fdk.git
cd nn_fdk
make install_dev
```

If you want to make use of the U-net and MSD-net functionalities follow the following instructions.

Create a conda environment with the following packages. Note that the cudatoolkit should coincide with the version on your machine.

```
conda create -n nnfdk -c conda-forge -c aahendriksen -c pytorch msd_pytorch torchvision msdnet cudatoolkit=10.1 -y
```
Activate the new environment and install the ddf_fdk and nn_fdk packages

```
conda activate nnfdk

git clone https://github.com/mjlagerwerf/nn_fdk.git
cd nn_fdk
make install

git clone https://github.com/mjlagerwerf/nn_fdk.git
cd nn_fdk
make install
```

### Running the examples

To learn more about the functionality of the package check out our
examples folder.

## Authors and contributors

* **Rien Lagerwerf** - *Initial work*

See also the list of [contributors](https://github.com/mjlagerwerf/nn_fdk/contributors) who participated in this project.

## How to contribute

Contributions are always welcome. Please submit pull requests against the `master` branch.

If you have any issues, questions, or remarks, then please open an issue on GitHub.

## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE.md](LICENSE.md) file for details.
