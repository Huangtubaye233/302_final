import numpy as np
from scipy import ndimage

# Size of the binary mask (8x8 grid of voxels)
MASK_DIM = 8 

# Scale of the robot (edge length of a voxel)
# NOTE: this is very important as the simulator physics are configured to use this scale, more or less.
SCALE = 0.1 
DEFAULT_MATERIAL_TYPE = 1
DEFAULT_N_MATERIAL_TYPES = 3

def clone_robot(robot):
    cloned = {}
    for key, value in robot.items():
        if isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        else:
            cloned[key] = value
    return cloned


def sample_spring_types(n_springs, n_material_types=DEFAULT_N_MATERIAL_TYPES):
    return np.random.randint(0, n_material_types, size=(n_springs,), dtype=np.int32)


def fixed_spring_types(n_springs, material_type=DEFAULT_MATERIAL_TYPE):
    return np.full((n_springs,), int(material_type), dtype=np.int32)


def mutate_spring_types(spring_types, p_flip=0.08, n_material_types=DEFAULT_N_MATERIAL_TYPES):
    spring_types = spring_types.astype(np.int32).copy()
    if n_material_types <= 1 or spring_types.size == 0:
        return spring_types
    flip_mask = np.random.uniform(0.0, 1.0, size=spring_types.shape) < p_flip
    if not np.any(flip_mask):
        return spring_types
    old = spring_types[flip_mask]
    delta = np.random.randint(1, n_material_types, size=old.shape, dtype=np.int32)
    spring_types[flip_mask] = (old + delta) % n_material_types
    return spring_types


def robot_from_mask(
    mask,
    spring_types=None,
    evolve_material=True,
    n_material_types=DEFAULT_N_MATERIAL_TYPES,
    fixed_material_type=DEFAULT_MATERIAL_TYPE,
):
    masses, springs = mask_to_robot(mask)
    masses = masses * SCALE # NOTE: scale of the robot geometry is KEY to stable simulation!
    if spring_types is None:
        if evolve_material:
            spring_types = sample_spring_types(springs.shape[0], n_material_types=n_material_types)
        else:
            spring_types = fixed_spring_types(springs.shape[0], material_type=fixed_material_type)
    spring_types = np.asarray(spring_types, dtype=np.int32)
    assert spring_types.shape[0] == springs.shape[0], "spring_types length must match n_springs"
    return {
        "n_masses": masses.shape[0],
        "n_springs": springs.shape[0],
        "masses": masses,
        "springs": springs,
        "spring_types": spring_types,
        "mask": mask,
    }

# Randomly sample a binary mask of size MASK_DIM x MASK_DIM
# Convert the binary mask to a mass-spring robot geometry
# The parameter p is by default set to 0.55, which is the probability of a voxel being filled.
# This is a manually tuned value that seems to produce a variety of different robot geometries.
def sample_robot(
    p=0.55,
    evolve_material=True,
    n_material_types=DEFAULT_N_MATERIAL_TYPES,
    fixed_material_type=DEFAULT_MATERIAL_TYPE,
):
    mask = sample_mask(p)
    return robot_from_mask(
        mask,
        evolve_material=evolve_material,
        n_material_types=n_material_types,
        fixed_material_type=fixed_material_type,
    )

# Convert a voxel position to a list of mass coordinates
# Each voxel has a mass located at each of its four corners
def voxel_to_masses(row, col):
    return [
        [row, col],
        [row, col+1],
        [row+1, col],
        [row+1, col+1],
    ]

# Convert a binary mask to a mass-spring robot geometry
# Each voxel is represented by 4 masses and 6 springs
# Masses are located at the corners of the voxel
# Springs connect adjacent masses along the edges and diagonals of the voxel
def mask_to_robot(mask):
    spring_connections = [
        [0, 1], # bottom left (bl) to bottom right (br)
        [0, 2], # bl to top left (tl)
        [1, 3], # br to top right (tr)
        [2, 3], # tl to tr
        [0, 3], # bl to tr
        [1, 2], # br to tl
    ]
    masses = []
    springs = []
    rows, cols = np.where(mask)
    n_voxels = len(rows)
    for i in range(n_voxels):
        row = rows[i]
        col = cols[i]
        coords = voxel_to_masses(row, col)
        for c in coords:
            if c not in masses: # NOTE: make sure to avoid duplicates!
                masses.append(c)
        for a, b, in spring_connections:
            ca = coords[a]
            cb = coords[b]
            ia = masses.index(ca)
            ib = masses.index(cb)
            s = [min(ia, ib), max(ia, ib)]
            if s not in springs: # NOTE: make sure to avoid duplicates!
                springs.append(s)
    masses = np.array(masses, dtype=np.float32) # Numpy array of shape (n_masses, 2)
    springs = np.array(springs, dtype=np.int32) # Numpy array of shape (n_springs, 2)
    return masses, springs

# Sample a binary mask of size MASK_DIM x MASK_DIM
# Select the largest connected component in the mask
# Zero out the rest of the mask
# Shift the largest component to the bottom left corner of the mask
def normalize_mask(mask):
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return None
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_component = np.argmax(component_sizes) + 1
    mask = (labeled == largest_component)
    rows, cols = np.where(mask)
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    component = mask[min_row:max_row+1, min_col:max_col+1]
    new_mask = np.zeros((MASK_DIM, MASK_DIM), dtype=int)
    component_height, component_width = component.shape
    new_mask[MASK_DIM - component_height:MASK_DIM, 0:component_width] = component.astype(int)
    return new_mask

def sample_mask(p):
    mask = np.random.uniform(0.0, 1.0, size=(MASK_DIM, MASK_DIM))
    mask = mask < p
    normalized = normalize_mask(mask)
    if normalized is None: # If the mask is empty, try again
        return sample_mask(p)
    return normalized

def mutate_robot(
    parent,
    p_flip=0.08,
    material_flip=0.08,
    evolve_shape=True,
    evolve_material=True,
    n_material_types=DEFAULT_N_MATERIAL_TYPES,
    fixed_material_type=DEFAULT_MATERIAL_TYPE,
    max_attempts=5,
    fill_p=0.55,
):
    if not evolve_shape:
        child = clone_robot(parent)
        if evolve_material:
            child["spring_types"] = mutate_spring_types(
                child["spring_types"],
                p_flip=material_flip,
                n_material_types=n_material_types,
            )
        else:
            child["spring_types"] = fixed_spring_types(
                child["n_springs"],
                material_type=fixed_material_type,
            )
        return child

    parent_mask = parent.get("mask", None)
    for _ in range(max_attempts):
        if parent_mask is None:
            mask = sample_mask(fill_p)
        else:
            mask = parent_mask.copy()
            flip = np.random.uniform(0.0, 1.0, size=mask.shape) < p_flip
            mask = np.logical_xor(mask, flip)
        normalized = normalize_mask(mask)
        if normalized is not None:
            return robot_from_mask(
                normalized,
                evolve_material=evolve_material,
                n_material_types=n_material_types,
                fixed_material_type=fixed_material_type,
            )
    return sample_robot(
        p=fill_p,
        evolve_material=evolve_material,
        n_material_types=n_material_types,
        fixed_material_type=fixed_material_type,
    )

def crossover_robots(
    parent_a,
    parent_b,
    p_swap=0.5,
    p_flip=0.03,
    material_swap=0.5,
    material_flip=0.03,
    evolve_shape=True,
    evolve_material=True,
    n_material_types=DEFAULT_N_MATERIAL_TYPES,
    fixed_material_type=DEFAULT_MATERIAL_TYPE,
    max_attempts=5,
    fill_p=0.55,
):
    if not evolve_shape:
        child = clone_robot(parent_a)
        if evolve_material:
            a_types = parent_a.get("spring_types", fixed_spring_types(parent_a["n_springs"], fixed_material_type))
            b_types = parent_b.get("spring_types", fixed_spring_types(parent_b["n_springs"], fixed_material_type))
            if a_types.shape[0] == b_types.shape[0]:
                swap = np.random.uniform(0.0, 1.0, size=a_types.shape) < material_swap
                child_types = np.where(swap, a_types, b_types).astype(np.int32)
            else:
                child_types = a_types.astype(np.int32).copy()
            child["spring_types"] = mutate_spring_types(
                child_types,
                p_flip=material_flip,
                n_material_types=n_material_types,
            )
        else:
            child["spring_types"] = fixed_spring_types(child["n_springs"], material_type=fixed_material_type)
        return child

    mask_a = parent_a.get("mask", None)
    mask_b = parent_b.get("mask", None)
    if mask_a is None or mask_b is None:
        return sample_robot(
            p=fill_p,
            evolve_material=evolve_material,
            n_material_types=n_material_types,
            fixed_material_type=fixed_material_type,
        )
    for _ in range(max_attempts):
        swap = np.random.uniform(0.0, 1.0, size=mask_a.shape) < p_swap
        child_mask = np.where(swap, mask_a, mask_b)
        if p_flip > 0.0:
            flip = np.random.uniform(0.0, 1.0, size=child_mask.shape) < p_flip
            child_mask = np.logical_xor(child_mask, flip)
        normalized = normalize_mask(child_mask)
        if normalized is not None:
            return robot_from_mask(
                normalized,
                evolve_material=evolve_material,
                n_material_types=n_material_types,
                fixed_material_type=fixed_material_type,
            )
    return sample_robot(
        p=fill_p,
        evolve_material=evolve_material,
        n_material_types=n_material_types,
        fixed_material_type=fixed_material_type,
    )