## Welcome to SAMPDI-3Dv2

SAMPDI-3Dv2 is a new version of the SAMPDI-3D trained on a larger dataset of mutation in protein and DNA in protein-DNA complex with an extended feature set and uses a gradient boosting decision tree machine learning algorithm that predicts the change in binding free energy due to mutation in protein (single amino acid) or DNA (a single base-pair). This method uses two different models:
 1. Mutations in protein: It trained on single amino acid mutation in protein protein and associated free energy change, using a variety of features derived from protein-DNA complex structure, to make predictions for single amino acid mutations.
 2. Mutation in DNA: It trained on single base-pair mutation in DNA and associated free energy change, using avariety of features derived from protein-DNA complex structure, to make prediction for single base-pair change.

We  have  installed  and  tested  sampdi3dv2  on  a  machine  running  Ubuntu  24.04 equpped with anaconda3   on  this  machine  to  manage  python  environments.

## Dependencies
We can keep the dependences in two group: (1) third-party tool dependencies; and (2) python dependencies.
### Third-party tools: Acquiring
SAMPDI-3Dv2 uses below listed third-party tools to analyze structure and these are used for calculating features used for making the prediction from the input structure file (PDB v 3.0).

 - DSSP: Download  it from  https://swift.cmbi.umcn.nl/gv/dssp/ and install using the instructions provided on the their website. We have used DSSP which is distribued with name mkdssp (v2.0.4).

- X3DNA-DSSR: Download it from  http://forum.x3dna.org/site-announcements/download-instructions/ and install after getting appropriate license. We have used x3dna-dssr (v2.4.5) under academic license in our work.

- Scwrl4: Check  link  http://dunbrack.fccc.edu/lab/scwrl  to  request  a  license (free for  academic  users) and acquire a copy to install. We have used Scwrl4 (v4.0.2) in our work.

- PSI-BLAST: Check  the  link  https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/  to  download  a  copy  of  psiblast. We have used phiblast (v2.10.0) in our work.

- UniRef50: Download  uniref50  from  https://www.uniprot.org/help/downloads  and  build  the  database  to  use.

### Third-party tools: Configuration
Users should create a directory ``software`` in the root-folder of this package, and place the executables of above mentioned 
third party tools inside ``software`` folder, and adjust the paths accordinaly in the ``configuration.py`` file in this folder.

##### Configuring DSSP, X3DNA-DSSR and Scwrl4
```python

    config["dssp_exe"] = f"{base_location}/software/dssp"
    config["x3dna_exe"] = f"{base_location}/software/x3dna-dssr"
    config["scwrl4_exe"] = f"{base_location}/software/scwrl4/Scwrl4"
```
Notes: 
- The ``base_location`` used above referes to the root-folder of the package and ``{base_location}/software/`` referrs to its subfolder ``software``, that we created while startining configuration.
- Scwrl4 is installed in folder ``{base_location}/software/scwrl4``, users may install it in the recommended location (then no need to edit the location of ``scwrl4_exe``), or even at different location but they must set the path to absolute path of the ``Scwrl4`` executable to the ``scwrl4_exe``.

##### Configuring PSI-BLAST and UniRef50
Users can install blast and configure uniref50 in a common path and to use it, user can export the environemnt variable ``SAMPDI3Dv2_HOME``. Alternatively users can comment line: 22. and provide absolute path to install directory of blast and uniref50 databse in lines 30 and 31, respectively.

### Python packages

SAMPDI-3Dv2 requires following packages and mentioned versions to be installed in the Python (v3.11.11) to work
-   numpy (v1.26.4)
-   pandas (v2.2.3)
-   prody (v2.4.1)
-   xgboost (v2.1.1)
-   joblib (v1.4.2)
-   pdb-tools: a fork of it from https://github.com/shaileshp51/pdb-tools and install it. This version handles alternate positions of the atoms from the PDB differently (alternate positons are considered based on residue as well as atom level) compared to original pdb-tools. We use only the highest occupancy alternate coordinate set in out work.

### Use a conda environment 
Users can alternatively try creating a conda environment from the environemnt.yaml and then install the pdb-tools as described: 
##### Create a conda environment using the environment.yml file
``conda  env  create  --file  environment.yml``

##### Activate the newly created python environment using
``conda  activate  py311_sampdi3dv2``

##### pull pdb-tools with updated ``pdb_selaltloc`` from repository https://github.com/shaileshp51/pdb-tools
```bash

cd  pdb-tools
python  setup.py  install
```

This choice simplifies the process of preparing the python environment.

## Help and Example
Users should activate the ``environment`` created for ``sampdi3dv2``, that will be ``py311_sampdi3dv2``, if created using ``environment.yml``.
Users can get the help message that details all the available options in sampdi3dv2 as follows:
```
python sampdi3d.py --help
```
Note: When users are not inside the ``sampdi3dv2`` folder or its subfolders, they should use the absoulte path of the executable ``sampdi3d.py``.

Two examples one for each of protein and DNA mutations are provided in the ``example`` folder that can be run as detailed below:

##### Examples of predicting binding free energy change due to single amino acids mutation for a protein-DNA complex
change directory to examples/protein
``cd  examples/protein``

``python  ../../sampdi3d.py  -i  1az0.pdb  -cc  ACD  -d  protein  -j  1az0_A  -f  1az0_A_mt_list.txt  -k  fasta  pdb  pssm  csv``

##### Examples of predicting binding free energy change due to single base/base-pair mutation for a protein-DNA complex
change directory to examples/dna
``cd  ../../``
``cd  examples/dna``
``python  ../../sampdi3d.py  -i  3wpd.pdb  -cc  AC  -d  dna  -j  3wpd_C  -f  3wpd_A_mt_list.txt  -k  fasta  pdb  pssm  csv``

## Publication
Rimal, P.; Paul, S.K.; Panday, S.K.; Alexov, E. Further Development of SAMPDI-3D: A Machine Learning Method for Predicting Binding Free Energy Changes Caused by Mutations in Either Protein or DNA. Genes 2025, 16, 101. https://doi.org/10.3390/genes16010101 
