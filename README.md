# Epidemiological Foci Relating Infections by Distance (EpiFRIenDs)

> Author: **Arnau Pujol**  
> Year: **2022**  
> Version: **1.1**  

This repository contains the EpiFRIenDs software to detect and analyse foci
(clusters, outbreaks or hotspots) of infections from a given disease.

This software is fully open source and all are welcome to use or modify it
for any purpose. We would kindly request that any scientific publications
making use of this software cite
**Pujol A., Brokhattingen N., Matambisso G., et al (in prep.)**.


Software requirements:
----------------------
All the packages that are required are:
- numpy
- scipy
- pandas
- geopandas

The Jupyter notebook examples also require matplotlib.

Installation:
----------------------
To install the repository, first you have to clone it to your local machine.
Then, from its current directory, you can simply run:

```
$ python setup.py install
```

In the text file `requirements.txt` the required packages are specified and
installed if needed. This version have been proven to work for Python 3.8.12

Structure of the repository:
----------------------------
The repository contains two main directories:
- epifriends: where the code is stored
- examples: where some examples of the software implementation are shown in
Jupyter notebooks.

How to use it:
----------------------------
The files epifriends.py and utils.py from the directory epifriends contains all
the main methods(functions) that can be called within EpiFRIenDs. The main
methods of the algorithm are defined in epifriends.py and are:
- dbscan: from some input position, linking distance and minimum number of
neighbours, this function finds DBSCAN clusters and assigns a cluster ID for
each position, with 0 meaning that the position does not belong to any
cluster.
- catalogue: from some input positions, test results, linking distance and
minimum number of neighbours, this function detects the EpiFRIenDs foci and
outputs a catalogue of them with its associated data.
- add_tempoal_id: from a list of EpiFRIenDs catalogues (each element of the  
list representing a time frame) and some linking times and distances, this
function assigns a temporal ID to the foci, assigning the same temporal ID
to foci form different time frames when they are close in time and space.

#### Examples:

Examples can be found in the directory `examples`, in the following Jupyter
notebooks:

- epifriends_on_different_distributions.ipynb: this notebook generates three
sets of artificial data and shows how to run EpiFRIenDs to detect foci on them.

- temporal_analysis.ipynb: this notebook generates mock data catalogues with a
date time for each sample. Then, it shows how to run EpiFRIenDs in different
time frames of the data set, how to link foci from different time frames to
assign them the same temporal ID and how to estimate the lifetime of the foci.
