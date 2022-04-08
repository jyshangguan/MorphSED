# MorphSED
This is a test project

# dependencies
* [libprofit](#https://github.com/ICRAR/libprofit)
  * clone the repo, compile and install it
    * `git clone https://github.com/ICRAR/libprofit.git`
    * `cd libprofit && mkdir build && cmake .. && make`
      * dependencies: `gsl`
    * `sudo make install`

# install
* `make install` to install
* `make develop` to install in develop mode

# configs

you should set these env variables
```
export MorphSED_DATA_PATH=<path-to-MorphSED-DATA>

# the templates are in the directory of `MorphSED_DATA_PATH/templates/*`

