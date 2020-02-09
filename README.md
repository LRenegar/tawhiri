# T&#257;whirim&#257;tea

[![Documentation Status](https://readthedocs.org/projects/tawhiri/badge/?version=latest)](https://readthedocs.org/projects/tawhiri/?badge=latest)

## Introduction

Tawhiri is a trajectory prediction software for high-altitude balloons originally
developed by the Cambridge University Spaceflight (CUSF) team behind
http://predict.habhub.org. This version has been significantly modified from the
original CUSF code to implement Monte Carlo capabilities in order to bettter
account for uncertainty in the balloon's trajectory.

The name comes from a
[M&#257;ori](http://en.wikipedia.org/wiki/M%C4%81ori_people)
god of weather, which rather aptly
&ldquo;drove Tangaroa and his progeny into the sea &rdquo;
[(WP)](http://en.wikipedia.org/wiki/Tawhiri).

## More information

Please see the [CUSF wiki](http://www.cusf.co.uk/wiki/), which contains pages
on [Tawhiri](http://www.cusf.co.uk/wiki/tawhiri:start) and [prediction in
general](http://www.cusf.co.uk/wiki/landing_predictor).

[More detailed API and setup documentation](http://tawhiri.cusf.co.uk/).

## Setup

See the deployment example in the ```deploy``` folder for a more comprehensive example.

```bash
$ apt install libgrib-api-dev libevent-dev libpng-dev libeccodes-dev
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install numpy wheel
$ pip install pyproj
$ pip install pygrib gevent
$ pip install -r requirements.txt
$ python setup.py build_ext --inplace
```

The last line (re-)builds the Cython extensions, and needs to be run again
after modifying any `.pyx` files.


## License

Tawhiri is Copyright 2014 (see AUTHORS & individual files) and licensed under
the [GNU GPL 3](http://gplv3.fsf.org/) (see LICENSE).
