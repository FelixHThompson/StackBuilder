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
    if np.linalg.norm(k) != 0:
        k = k / np.linalg.norm(k)
    # Where alpha is the angle between k and the z-axis
    alpha = -1 * np.arccos(np.dot(init, n_prime))

    for x in range(0, len(structure)):
        v = structure[x][1]
        # The Rodrigues' Rotation Formula on each position
        structure[x][1] = (v * np.cos(alpha) + np.cross(k, v) * np.sin(alpha)
                          + k * (np.dot(k, v)) * (1 - np.cos(alpha)))

    return structure

parser = argparse.ArgumentParser(description='Molecular stack builder:')
parser.add_argument("mol", default="mol.xyz",
                    help="Monomer coordinate file [.xyz formatting]")
parser.add_argument("-n", "-number", default=3, required=False, type=int,
                    help="Number of layers")
parser.add_argument("-t", "-twist", default="0", required=False,
                    help="Twist angle")
parser.add_argument("-r", "-separation", default=3.5, required=False,
                    type=float, help="Vertical layer separation")
parser.add_argument("-s", "-slide", default="90", required=False,
                    help="Slide angle")
parser.add_argument("-o", "-output", default="", required=False,
                     help="Filename of created stack")
parser.add_argument("-t_a", action='store_true', required=False,
                    help="Alternate rotation direction")
parser.add_argument("-s_a", action='store_true', required=False,
                    help="Alternate sliding direction")
parser.add_argument("-align", action='store_true', required=False,
                    help="Align molecule to xy-plane")
parser.add_argument("-f", "-flip", action='store_true', required=False,
                    help="Flip molecule about z-axis")
args = parser.parse_args()

monomer = load_xyz(args.mol)

monomer = translate_structure(monomer, -1 * find_centroid(monomer))

if args.align:
    # Here one could add some selection for only atom-specific mapping
    # Find the spherical angles that define the plane to which the structure's
    #   RMSD is minimized.
    res = minimize(rmsd_angle, x0=[0.23 * np.pi, np.pi], args=monomer, method='TNC',
                   bounds=((0, 0.5 * np.pi), (0, 2 * np.pi)))
    # Rotate the structure according to those angles
    monomer = Rodrigues(res.x, monomer)

# If the user specifies, flip the molecule over the xy-plane
if args.f:
    monomer = Rodrigues([0.5 * np.pi, 0], monomer)
    monomer = Rodrigues([0.5 * np.pi, 0], monomer)

# Angle input parsing:
wrong_no_inputs = False
if len(args.t.split(" ")) > 1:
    rots = [float(x) for x in args.t.split()]
    if len(rots) < args.n - 1:
        padding = [0.0 for x in range(0, args.n - 1 - len(rots))]
        [rots.append(x) for x in padding]
        print(" >> Too few twist angles.")
    if len(rots) > args.n - 1:
        rots = rots[:args.n - 1]
        print(" >> Too many twist angles.")
else:
    rots = [float(args.t) for x in range(0, args.n - 1)]

if len(args.s.split(" ")) > 1:
    slides = [float(x) for x in args.s.split()]
    if len(slides) < args.n - 1:
        padding = [0.0 for x in range(0, args.n - 1 - len(slides))]
        [slides.append(x) for x in padding]
        print(" >> Too few slide angles.")
    if len(slides) > args.n - 1:
        slides = slides[:args.n - 1]
        print(" >> Too many slide angles.")
else:
    slides = [float(args.s) for x in range(0, args.n - 1)]

# Applying angle alternation:
if len(rots) > 1:
    if args.t_a:
        for x in range(1, len(rots), 2):
            rots[x] *= -1

if len(slides) > 1:
    if args.s_a:
        for x in range(1, len(slides), 2):
            slides[x] = 180 - slides[x]

print(rots)
print(slides)


# Deepcopy due to mutability of copy()-ied sublists
stack = copy.deepcopy(monomer)

# We add the rotations of each step together such that only a single rotation
#   is needed to be carried out.
# The translation of each layer is dependent on the rotations of all previous
#   layers, so we must keep track of it each loop.
running_rotation = 0.0
running_translation = np.asarray([0.0, 0.0, 0.0])
next_rotation = Rotation.from_euler('z', 0, degrees=True)

for x in range(0, int(args.n) - 1):
    working_layer = copy.deepcopy(monomer)
    prev_rotation = next_rotation


    running_rotation += rots[x]

    next_rotation = Rotation.from_euler('z', -running_rotation, degrees=True)
    working_layer = rotate_structure(working_layer, next_rotation)

    # We add the next translation which has been rotated by the previous
    #   layer-to-layer rotation, keeping the point of reference as the centre
    #   of the previous layer.


    running_translation += prev_rotation.apply(np.asarray([args.r * math.cos(slides[x] * np.pi / 180), 0.0, args.r * math.sin(slides[x] * np.pi / 180)]))

    working_layer = translate_structure(working_layer, running_translation)
    stack += copy.deepcopy(working_layer)

# File name/saving
if args.o == "":
    output_name = "stack"
else:
    output_name = args.o
save_xyz(stack, output_name+".xyz")
