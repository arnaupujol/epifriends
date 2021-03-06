# Epidemiological Foci Relating Infections by Distance (EpiFRIenDs)

> Author: **Arnau Pujol**  
> Year: **2022**  
> Version: **1.0**  

This repository contains the EpiFRIenDs software to detect and analyse foci
(clusters, outbreaks or hotspots) of infections from a given disease.

Software requirements:
----------------------
All the packages that are required so that all the codes can run are:
- numpy
- scipy
- pandas
- geopandas

Installation:
----------------------
To install the repository, first you have to clone it to your local machine.
Then you can simply run:

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
The file epifriends.py from the directory epifriends contains all the methods
(functions) that can be called within EpiFRIenDs. The main methods of the
algorithm are:
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
