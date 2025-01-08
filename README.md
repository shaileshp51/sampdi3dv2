## Welcome to SAMPDI-3Dv2

SAMPDI-3Dv2 is a new version of the SAMPDI-3D trained on a larger dataset of mutation in protein and DNA in protein-DNA complex with an extended feature set and uses a gradient boosting decision tree machine learning algorithm that predicts the change in binding free energy due to mutation in protein (single amino acid) or DNA (a single base-pair). This method uses two different models:
 1. Mutations in protein: It trained on single amino acid mutation in protein protein and associated free energy change, using a variety of features derived from protein-DNA complex structure, to make predictions for single amino acid mutations.
 2. Mutation in DNA: It trained on single base-pair mutation in DNA and associated free energy change, using avariety of features derived from protein-DNA complex structure, to make prediction for single base-pair change.

We  have  installed  and  tested  sampdi3dv2  on  a  machine  running  Ubuntu  24.04 equpped with anaconda3   on  this  machine  to  manage  python  environments.

## Dependencies
We can keep the dependences in two group: (1) third-party tool dependencies; and (2) python dependencies.
### Third-party tools
SAMPDI-3Dv2 uses below listed third-party tools to calculated features used for making the prediction from the input structure file (PDB v 3.0).

 - DSSP: Download  it from  https://swift.cmbi.umcn.nl/gv/dssp/ and install using the instructions provided on the their website. We have used DSSP which is distribued with name mkdssp (v2.0.4).

- X3DNA-DSSR: Download it from  http://forum.x3dna.org/site-announcements/download-instructions/ and install after getting appropriate license. We have used x3dna-dssr (v2.4.5) under academic license in our work.

- Scwrl4: Check  link  http://dunbrack.fccc.edu/lab/scwrl  to  request  a  license (free for  academic  users) and acquire a copy to install. We have used Scwrl4 (v4.0.2) in our work.

- PSI-BLAST: Check  the  link  https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/  to  download  a  copy  of  psiblast. We have used phiblast (v2.10.0) in our work.

- UniRef50: Download  uniref50  from  https://www.uniprot.org/help/downloads  and  build  the  database  to  use.

### Python packages

SAMPDI-3Dv2 requires following packages and mentioned versions to be installed in the Python (v3.11.11) to work
-   numpy (1.26.4)
-   pandas (2.2.3)
-   prody (2.4.1)
-   xgboost (2.1.1)
-   pdb-tools: a fork of it from https://github.com/shaileshp51/pdb-tools and install it. This version handles alternate positions of the atoms from the PDB differently (alternate positons are considered based on residue as well as atom level) compared to original pdb-tools. We use only the highest occupancy alternate coordinate set in out work.

### Use a conda environment 
Users can alternatively try creating a conda environment from the environemnt.yaml and then install the pdb-tools as described: 
##### Create a conda environment using the environment.yml file
``conda  env  create  --file  environment.yml``

##### Activate the newly created python environment using
``conda  activate  py311_sampdi3dv2``

##### pull pdb-tools with updated ``pdb_selaltloc`` from repository https://github.com/shaileshp51/pdb-tools
``cd  pdb-tools``
``python  setup.py  install``

This choice simplifies the process of preparing the python environment.

## Example
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
Comming soon...
