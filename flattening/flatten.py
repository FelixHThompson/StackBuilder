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

def plane_rmsd(structure):
    """
    """

    sq_residuals = 0
    for atom in structure:
        sq_residuals += atom[1][2] ** 2

    return sq_residuals

def find_centroid(structure):
    """
    """

    coords_sum = np.asarray([0.0, 0.0, 0.0])
    for atom in structure:
        coords_sum += atom[1]
    centroid = coords_sum / len(structure)

    return centroid

parser = argparse.ArgumentParser(description='Molecular stack builder:')
parser.add_argument("mol", default="mol.xyz",
                    help="Monomer coordinate file [.xyz formatting]")
args = parser.parse_args()

mol = load_xyz(args.mol)

## Add some selection for only mapping based on C's for example
lowest = [plane_rmsd(mol), [0.0,0.0,0.0]]
print(find_centroid(mol))
mol = translate_structure(mol, -1 * find_centroid(mol))
save_xyz(mol, "centred.xyz")
print(plane_rmsd(mol))



carryon = True
xlb = 0
xub = 180
ylb = 0
yub = 180
zlb = 0
zub = 180
step = 20
while carryon == True:
    x = xlb
    while x <= xub:
        y = ylb
        while y <= yub:
            z = zlb
            #print("Currently: {}/{}/{}".format(x, y, z))
            while z <= zub:
                t_mol = copy.deepcopy(mol)
                t_xyz = plane_rmsd(rotate_structure(t_mol, Rotation.from_euler('xyz', [x, y, z], degrees=True)))
                if t_xyz < lowest[0]:
                    lowest = [t_xyz, [x, y, z]]
                    print("New: ")
                    print(lowest)
                z += step
            y += step
        x += step
    step = step * 0.5
    xlb = lowest[1][0] - 2 * step
    xub = lowest[1][0] + 2 * step
    ylb = lowest[1][1] - 2 * step
    yub = lowest[1][1] + 2 * step
    zlb = lowest[1][2] - 2 * step
    zub = lowest[1][2] + 2 * step

    print("This pass finds: ")
    print(lowest)
    print("New step: "+str(step))
    print("x-bounds: "+str(xlb)+", "+str(xub))
    print("y-bounds: "+str(ylb)+", "+str(yub))
    print("z-bounds: "+str(zlb)+", "+str(zub))

    if step <= 0.05:
        carryon = False


print(plane_rmsd(mol))

R = Rotation.from_euler('xyz', lowest[1], degrees=True)
mol = rotate_structure(mol, R)
print(plane_rmsd(mol))
save_xyz(mol, "temp.xyz")
