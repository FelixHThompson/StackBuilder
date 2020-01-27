import numpy as np
import math
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
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
            line_bits = lines.split()
            if len(line_bits) > 0:
                coords = np.asarray([float(line_bits[1]), float(line_bits[2]),
                                    float(line_bits[3])])

                if len(line_bits) > 2:
                    structure += [[line_bits[0], coords, i-2]]
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

def find_centroid(structure):
    """Find the centroid of a given structure.

    Parameters
    ----------
    structure : list(string, np.arrray[])
        Formatted list of atom positions

    Returns
    -------
    np.array[]
        Coordinates of structure's centroid
    """

    coords_sum = np.asarray([0.0, 0.0, 0.0])
    for atom in structure:
        coords_sum += atom[1]
    centroid = coords_sum / len(structure)

    return centroid

def rmsd_angle(angles, structure):
    """Find the root mean squared displacement of a structure to a plane.

    Conventionally the RMSD might be taken to the xy-plane, as such making
      each displacement simply the z-coordinate of each position. This process
      can be done for a general plane passing through the origin. Such a plane
      can be defined by a unit vector orthogonal to the plane, with the vector
      defined by the spherical angles theta and phi. At each position the
      displacement is found from the dot-product of the orthogonal vector and
      the vector defining the position.

    Constructed in this way, this function is used to carry out a scipy
      minimization, finding the vector (and therefore plane) for which the
      RMSD is minimized -- i.e. the plane in which the structure is most flat.

    Parameters
    ----------
    angles : list(float, float)
        The spherical angles theta and phi
    structure : list(string, np.arrray[])
        Formatted list of atom positions

    Returns
    -------
    float
        RMSD of the system to a given plane
    """

    theta = angles[0]
    phi = angles[1]

    # From our spherical angles, a plane-defining vector is formed
    n_prime = np.asarray([np.sin(theta) * np.cos(phi),
                          np.sin(theta) * np.sin(phi),
                          np.cos(theta)])

    run_rmsd = 0
    for atom in structure:
        # Summing the square displacements of each position to this plane
        run_rmsd += abs(np.dot(n_prime, atom[1])) ** 2

    final_rmsd = (run_rmsd / len(structure)) ** 0.5

    return final_rmsd

def Rodrigues(angles, structure):
    """Apply the Rodrigues' rotation formula to a given structure.

    This formula allows the rotation of a vector by a certain angle about a
      certain axis. Here, each vector is an atomic position, the angle the
      angle between the z-axis and our minimising RMSD vector, and the axis is
      the vector orthogonal to the z-axis and said RMSD vector.

    Parameters
    ----------
    angles : list(float, float)
        The spherical angles theta and phi
    structure : list(string, np.arrray[])
        Formatted list of atom positions

    Returns
    -------
    list(string, np.array[])
        Formatted list of atom positions, having been rotated
    """

    theta, phi = angles

    # From our spherical angles, a plane-defining vector is formed
    n_prime = np.asarray([np.sin(theta) * np.cos(phi),
                          np.sin(theta) * np.sin(phi),
                          np.cos(theta)])
    init = np.asarray([0.0, 0.0, 1.0])

    # Where k is a vector orthogonal to k and the z-axis
    k = np.cross(init, n_prime)
    k = k / np.linalg.norm(k)
    # Where alpha is the angle between k and the z-axis
    alpha = -1 * np.arccos(np.dot(init, n_prime))

    for x in range(0, len(structure)):
        v = structure[x][1]
        # The Rodrigues' Rotation Formula on each position
        structure[x][1] = (v * np.cos(alpha) + np.cross(k, v) * np.sin(alpha)
                          + k * (np.dot(k, v)) * (1 - np.cos(alpha)))

    return structure

parser = argparse.ArgumentParser(description='Molecule flattener:')
parser.add_argument("mol", default="mol.xyz",
                    help="Coordinate file [.xyz]")
parser.add_argument("-f", "-flip", action='store_true', required=False,
                    help="Flip molecule about z-axis")
args = parser.parse_args()

mol = load_xyz(args.mol)

## Add some selection for only mapping based on C's for example
mol = translate_structure(mol, -1 * find_centroid(mol))
save_xyz(mol, "centred.xyz")

# Find the spherical angles that define the plane to which the structure's
#   RMSD is minimized.
res = minimize(rmsd_angle, x0=[0.23 * np.pi, np.pi], args=mol, method='TNC',
               bounds=((0, 0.5 * np.pi), (0, 2 * np.pi)))
mol = Rodrigues(res.x, mol)
# If the user specifies, flip the molecule over the xy-plane
if args.f:
    mol = Rodrigues([0.5 * np.pi, 0], mol)
    mol = Rodrigues([0.5 * np.pi, 0], mol)

save_xyz(mol, "flattened.xyz")
