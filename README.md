# StackBuilder
A tool to generate molecular dimers/stacks/&amp;c. with provided twist and slide angles.

## Usage
The python3 script should be called to operate on an .xyz formatted structure:````python stack_builder mol.xyz````

A list of options is provided by ````-h````, and includes the functionality to set:
- Number of layers in the stack ````-n 5````
- Layer-to-layer twist angle in degrees ````-t 30````
- Layer-to-layer slide angle in degrees ````-s 30````
- Distance between layers' centroids in Angstrom ````-r 3.5````
- If the twist angle alternates direction layer-to-layer ````-t_a````
- If the slide angle alternates direction layer-to-layer ````-s_a````
- Output file-name ````-o myname````
- If the molecule is aligned to the xy-plane ````-align````
- If the molecule is flipped about the xy-plane ````-f````

E.g. ````python stack_builder.py mol.xyz -n 5 -t 90 -t_a -f -o name````
