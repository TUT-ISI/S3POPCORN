# POPCORN Sentinel-3 Synergy aerosol parameter post-process correction

A Python script to post-process correct the Sentinel-3 Synergy (SY_2_SYN) aerosol data products.
Includes both the accuracy correction and spatial anomaly correction.

* Developed by: Finnish Meteorological Institute and University of Eastern Finland
* Development of the algorithm was funded by the European Space Agency EO science for society programme via POPCORN project.
* Contact info: Antti Lipponen (antti.lipponen@fmi.fi)

## Script use

```
USE: python S3POPCORN.py BASEDIR OUTPUTDIR SYNFILE
    BASEDIR=directory under which the Sentinel-3 data will be looked for (uses recursive search). For example ".".
    OUTPUTDIR=directory in which the post-process corrected data will be saver
    SYNFILE=name of the zip file that contains the Synergy data

    Sentinel-3 level-1 and level-2 data products are expected to be stored in zip format.
```

The script requires you have the following Sentinel-3 data available (in zip format) under the BASEDIR:
* SY_2_SYN____
* OL_1_ERR____
* SL_1_RBT____

### Use example
```console
foo@bar:~$ python S3POPCORN.py . S3POPCORNoutput S3A_SY_2_SYN____20190216T093921_20190216T094221_20190218T024047_0179_041_250_2160_LN2_O_NT_002.zip
...
```

### Python dependencies

The script depends for example on the following packages:
* Numpy
* Scipy
* Pytorch
* Pytorch-lightning
* netCDF4
* Scikit-learn

## Satellite data

Sentinel-3 satellite data can be downloaded free of charge for example from the Copernicus Open Access Hub (https://scihub.copernicus.eu/)
