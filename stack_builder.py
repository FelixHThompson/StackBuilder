import numpy as np
import math
from scipy.spatial.transform import Rotation
import copy
import argparse

def load_xyz(file_path):
    """Load .xyz coordinate file into a useful list.

    Return a list of atoms comprising a provided .xyz file. To allow future
    processing, each element is presented as [<name>, [<coords>], <number>],
    where <name> is the element name ("H"/"He"/"Li"), [<coords>] is an array
    containing the x/y/z position of this atom, <number> is the number of the
    atom in the file (counting from 0).

    Parameters
    ----------
    file_path : string
        Address of the .xyz file.

    Returns
    -------
    list(string, np.array[])
        Formatted list of atom positions
    """

    file = open(file_path)
    file_contents = file.readlines()
    file.close()

    i = 0
    structure = []

    for lines in file_contents:
    # First two lines of file are descriptive and skipped over
        if i > 1:
            split_line = lines.split()
            if len(split_line) > 0:
                coords = np.asarray([float(split_line[1]), float(split_line[2]),
                                    float(split_line[3])])

                if len(split_line) > 2:
                    structure += [[split_line[0], coords, i-2]]
        i += 1

    return structure

def save_xyz(structure, file_path):
    """Save a formatted list of atoms as an .xyz file.

    Parameters
    ----------
    structure : list(string, np.array[])
        Formatted list of atom positions
    file_path : string
        Address of the .xyz file being saved
    """

    file = open(file_path, 'w+')
    file.write('{}\n\n'.format(len(structure)))
    for atom in structure:
        file.write('{} {:20.16f} {:20.16f} {:20.16f}\n'.format(atom[0],
                   atom[1][0], atom[1][1], atom[1][2]))
    file.close()

def rotate_structure(structure, R):
    """Apply a Rotation object to the coordinates of each atom in a structure.

    Parameters
    ----------
    structure : list(string, np.array[])
        Formatted list of atom positions
    R : scipy.spatial.transform.rotation.Rotation object
        The rotation object to be applied

    Returns
    -------
    list(string, np.array[])
        Formatted list of atom positions, having been rotated
    """

    for count in range(0, len(structure)):
        structure[count][1] = R.apply(structure[count][1])

    return structure

def translate_structure(structure, trans_vec):
    """Translate the coordinates of each atom in a structure by some vector.

    Parameters
    ----------
    structure : list(string, np.array[])
        Formatted list of atom positions
    trans_vec : 1x3 np.array[] of floats
        The translation vector to be applied

    Returns
    -------
    list(string, np.array[])
        Formatted list of atom positions, having been translated
    """

    for count in range(0, len(structure)):
        structure[count][1] = structure[count][1] + trans_vec

    return structure

parser = argparse.ArgumentParser(description='Molecular stack builder:')
parser.add_argument("mol", default="mol.xyz",
                    help="Monomer coordinate file [.xyz formatting]")
parser.add_argument("-n", "-number", default=3, required=False,
                    help="Number of layers")
parser.add_argument("-t", "-twist", default=0, required=False, type=float,
                    help="Twist angle")
parser.add_argument("-r", "-separation", default=3.5, required=False, type=float,
                    help="Vertical layer separation")
parser.add_argument("-s", "-slide", default=90, required=False,
                    help="Slide angle")
parser.add_argument("-o", "-output", default="", required=False,
                     help="Filename of created stack")
parser.add_argument("-t_a", action='store_true', required=False,
                    help="Alternate rotation direction")
parser.add_argument("-s_a", action='store_true', required=False,
                    help="Alternate sliding direction")
# parser.add_argument("-align", action='store_true', required=False,
#                     help="Align molecule to xy-plane")
args = parser.parse_args()

monomer = load_xyz(args.mol)

stack = copy.deepcopy(monomer)
prev_layer = copy.deepcopy(monomer)
next_translation = np.asarray([float(args.r) * math.cos(float(args.s) * np.pi / 180), 0.0, float(args.r) * math.sin(float(args.s) * np.pi / 180)])

running_rotation = 0.0
running_translation = np.asarray([float(args.r) * math.cos(float(args.s) * np.pi / 180), 0.0, float(args.r) * math.sin(float(args.s) * np.pi / 180)])
running_translation = np.asarray([0.0, 0.0, 0.0])
next_rotation = Rotation.from_euler('z', 0, degrees=True)

for x in range(0, int(args.n) - 1):
    working_layer = copy.deepcopy(monomer)
    prev_rotation = next_rotation
    if not args.t_a:
        running_rotation += args.t
    else:
        running_rotation += ((-1) ** x) * args.t
    next_rotation = Rotation.from_euler('z', -running_rotation, degrees=True)
    working_layer = rotate_structure(working_layer, next_rotation)
    if not args.s_a:
        running_translation += prev_rotation.apply(np.asarray([float(args.r) * math.cos(float(args.s) * np.pi / 180), 0.0, float(args.r) * math.sin(float(args.s) * np.pi / 180)]))
    else:
        running_translation += prev_rotation.apply(np.asarray([((-1) ** x) * float(args.r) * math.cos(float(args.s) * np.pi / 180), 0.0, float(args.r) * math.sin(float(args.s) * np.pi / 180)]))
    print(running_translation)
    working_layer = translate_structure(working_layer, running_translation)
    stack += copy.deepcopy(working_layer)

if args.o == "":
    output_name = "stack"
else:
    output_name = args.o
save_xyz(stack, output_name+".xyz")

# Handle range of inputs? Tbh all single line in terminal, so just write a
#   script that interfaces with it if needs be?
# Be great to be able to pass lists of angles for each layer.
# Include molecule alignment?
