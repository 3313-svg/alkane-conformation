import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import py3Dmol
import uuid
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D

# Page Setup
st.set_page_config(layout="wide")
st.title("Alkane & Cycloalkane Conformational Analysis Simulator")

# --- Initialize Session States Safely ---
if 'theta' not in st.session_state:
    st.session_state.theta = 180

# --- Callback function to update theta safely without StreamlitAPIException ---
def update_theta(val):
    st.session_state.theta = val

# --- Helper Function with Container Wrapping to Fix Caching Bug ---
def render_robust_3d_view(view, height=320, width=450, unique_key=""):
    """
    Renders the py3Dmol canvas inside a keyed st.container.
    """
    html_string = view._make_html()
    cache_buster = uuid.uuid4().hex
    html_string += f"\n"

    with st.container(key=unique_key):
        components.html(html_string, height=height, width=width)

# --- Definitions and Functions ---
alkyl_labels = {
    0: "H", 1: r"CH$_3$", 2: r"C$_2$H$_5$", 3: r"C$_3$H$_7$", 
    4: r"C$_4$H$_9$", 5: r"C$_5$H$_{11}$", 6: r"C$_6$H$_{13}$", 7: r"C$_7$H$_{15}$"
}

alkanes = {
    "Butane (n-Butane)": "CCCC",
    "Isobutane (2-Methylpropane)": "CC(C)C",
    "Pentane (n-Pentane)": "CCCCC",
    "Isopentane (2-Methylbutane)": "CCC(C)C",
    "Neopentane (2,2-Dimethylpropane)": "CC(C)(C)C",
    "Hexane (n-Hexane)": "CCCCCC",
    "2-Methylpentane": "CCCC(C)C",
    "3-Methylpentane": "CCC(C)CC",
    "2,2-Dimethylbutane": "CCC(C)(C)C",
    "2,3-Dimethylbutane": "CC(C)C(C)C",
    "Heptane (n-Heptane)": "CCCCCCC"
}

cycloalkanes_data = {
    "Cyclopropane (C3)": {
        "smiles": "C1CC1",
        "steps": ["Planar"],
        "energies": [0.0],
        "coords": {"Planar": [(0.0, 0.86, 0.0), (0.77, -0.43, 0.0), (-0.77, -0.43, 0.0)]},
        "eng_types": ["Planar"]
    },
    "Cyclobutane (C4)": {
        "smiles": "C1CCC1",
        "steps": ["Puckered Conformation 1", "Planar", "Puckered Conformation 2"],
        "energies": [0.0, 6.0, 0.0],
        "coords": {
            "Puckered 1": [(0.75, 0.75, 0.25), (0.75, -0.75, -0.25), (-0.75, -0.75, 0.25), (-0.75, 0.75, -0.25)],
            "Planar": [(0.77, 0.77, 0.0), (0.77, -0.77, 0.0), (-0.77, -0.77, 0.0), (-0.77, 0.77, 0.0)],
            "Puckered 2": [(0.75, 0.75, -0.25), (0.75, -0.75, 0.25), (-0.75, -0.75, -0.25), (-0.75, 0.75, -0.25)]
        },
        "eng_types": ["Puckered 1", "Planar", "Puckered 2"]
    },
    "Cyclopentane (C5)": {
        "smiles": "C1CCCC1",
        "steps": ["Envelope Form 1", "Half-Chair", "Envelope Form 2"],
        "energies": [0.0, 2.0, 0.0],
        "coords": {
            "Envelope 1": [(0.0, 1.2, 0.5), (1.23, 0.4, -0.1), (0.76, -1.05, -0.1), (-0.76, -1.05, -0.1), (-1.23, 0.4, -0.1)],
            "Half-Chair": [(0.0, 1.25, 0.3), (1.2, 0.35, -0.3), (0.76, -1.05, 0.0), (-0.76, -1.05, 0.0), (-1.23, 0.4, 0.0)],
            "Envelope 2": [(0.0, 1.2, -0.5), (1.23, 0.4, 0.1), (0.76, -1.05, 0.1), (-0.76, -1.05, 0.1), (-1.23, 0.4, 0.1)]
        },
        "eng_types": ["Envelope 1", "Half-Chair", "Envelope 2"]
    },
    "Cyclohexane (C6)": {
        "smiles": "C1CCCCC1",
        "steps": ["Chair 1", "Half-Chair 1", "Twist-Boat 1", "Boat", "Twist-Boat 2", "Half-Chair 2", "Chair 2"],
        "energies": [0.0, 45.0, 23.0, 29.0, 23.0, 45.0, 0.0],
        "coords": {
            "Chair 1": [(0.0, 1.4, 0.5), (1.2, 0.7, -0.5), (1.2, -0.7, 0.5), (0.0, -1.4, -0.5), (-1.2, -0.7, 0.5), (-1.2, 0.7, -0.5)],
            "Half-Chair 1": [(0.0, 1.4, 0.5), (1.2, 0.7, 0.0), (1.2, -0.7, 0.0), (0.0, -1.4, -0.5), (-1.2, -0.7, 0.0), (-1.2, 0.7, 0.0)],
            "Twist-Boat 1": [(0.0, 1.4, 0.3), (1.2, 0.7, -0.5), (1.2, -0.7, -0.1), (0.0, -1.4, 0.3), (-1.2, -0.7, -0.5), (-1.2, 0.7, -0.1)],
            "Boat": [(0.0, 1.4, 0.5), (1.2, 0.7, -0.5), (1.2, -0.7, -0.5), (0.0, -1.4, 0.5), (-1.2, -0.7, -0.5), (-1.2, 0.7, -0.5)],
            "Twist-Boat 2": [(0.0, 1.4, 0.3), (1.2, 0.7, -0.1), (1.2, -0.7, -0.5), (0.0, -1.4, 0.3), (-1.2, -0.7, -0.1), (-1.2, 0.7, -0.5)],
            "Half-Chair 2": [(0.0, 1.4, 0.5), (1.2, 0.7, 0.0), (1.2, -0.7, 0.0), (0.0, -1.4, -0.5), (-1.2, -0.7, 0.0), (-1.2, 0.7, 0.0)],
            "Chair 2": [(0.0, 1.4, -0.5), (1.2, 0.7, 0.5), (1.2, -0.7, -0.5), (0.0, -1.4, 0.5), (-1.2, -0.7, -0.5), (-1.2, 0.7, 0.5)]
        },
        "eng_types": ["Chair 1", "Half-Chair 1", "Twist-Boat 1", "Boat", "Twist-Boat 2", "Half-Chair 2", "Chair 2"]
    }
}

def compute_iupac_labels(mol):
    adj = {}
    c_indices = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            idx = atom.GetIdx()
            c_indices.append(idx)
            adj[idx] = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'C']
            
    if not c_indices:
        return {}
        
    terminals = [i for i in c_indices if len(adj[i]) <= 1]
    if not terminals:  
        terminals = c_indices
        
    all_paths = []
    def dfs(curr, target, path, visited):
        if curr == target:
            all_paths.append(list(path))
            return
        for nbr in adj[curr]:
            if nbr not in visited:
                visited.add(nbr)
                path.append(nbr)
                dfs(nbr, target, path, visited)
                path.pop()
                visited.remove(nbr)

    for i in range(len(terminals)):
        for j in range(i + 1, len(terminals)):
            start, end = terminals[i], terminals[j]
            dfs(start, end, [start], {start})
            
    if not all_paths:
        return {idx: f"C{i+1}" for i, idx in enumerate(c_indices)}
        
    max_len = max(len(p) for p in all_paths)
    longest_paths = [p for p in all_paths if len(p) == max_len]
    
    best_path = None
    best_locants = None
    best_num_branches = -1
    best_spread = -1
    
    for path in longest_paths:
        for p in [path, path[::-1]]:
            locants = []
            for step_idx, c_idx in enumerate(p):
                for nbr in adj[c_idx]:
                    if nbr not in p:
                        locants.append(step_idx + 1)
            locants.sort()
            
            num_branches = len(locants)
            spread = max(p) - min(p) 
            
            if best_path is None:
                best_path = p
                best_locants = locants
                best_num_branches = num_branches
                best_spread = spread
            else:
                if num_branches > best_num_branches:
                    best_path = p
                    best_locants = locants
                    best_num_branches = num_branches
                    best_spread = spread
                elif num_branches == best_num_branches:
                    if locants < best_locants:
                        best_path = p
                        best_locants = locants
                        best_spread = spread
                    elif locants == best_locants:
                        if spread > best_spread:
                            best_path = p
                            best_locants = locants
                            best_spread = spread

    mapping = {}
    for step_idx, c_idx in enumerate(best_path):
        mapping[c_idx] = f"C{step_idx + 1}"
        
    branch_counter = len(best_path) + 1
    for c_idx in c_indices:
        if c_idx not in mapping:
            mapping[c_idx] = f"C{branch_counter}"
            branch_counter += 1
            
    return mapping

def get_substituent_label(mol, nbr_idx, root_idx):
    atom = mol.GetAtomWithIdx(nbr_idx)
    if atom.GetSymbol() == 'H':
        return "H"
    
    visited = {root_idx}
    queue = [nbr_idx]
    c_count = 0
    while queue:
        curr = queue.pop(0)
        if curr not in visited:
            visited.add(curr)
            if mol.GetAtomWithIdx(curr).GetSymbol() == 'C':
                c_count += 1
            for n in mol.GetAtomWithIdx(curr).GetNeighbors():
                if n.GetIdx() not in visited:
                    queue.append(n.GetIdx())
    return alkyl_labels.get(c_count, r"C$_n$H$_{2n+1}$")

def draw_2d_newman(ax, front_subs, back_subs, text_color="black"):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    
    for angle, label in back_subs:
        rad = np.radians(angle)
        ax.plot([0, 1.15 * np.cos(rad)], [0, 1.15 * np.sin(rad)], color="darkgray", lw=2.5, zorder=1)
    
    circle = plt.Circle((0, 0), 0.5, edgecolor=text_color, facecolor="white", fill=True, lw=2, zorder=2)
    ax.add_patch(circle)
    
    for angle, label in front_subs:
        rad = np.radians(angle)
        ax.plot([0, 1.0 * np.cos(rad)], [0, 1.0 * np.sin(rad)], color=text_color, lw=3.5, zorder=3)
        
    ax.scatter(0, 0, color=text_color, s=90, zorder=4)
    
    for angle, label in front_subs:
        rad = np.radians(angle)
        x = 1.32 * np.cos(rad)
        y = 1.32 * np.sin(rad)
        ax.text(x, y, label, color=text_color, fontsize=12, fontweight="black", ha="center", va="center", zorder=5)
        
    for angle, label in back_subs:
        rad = np.radians(angle)
        x = 1.48 * np.cos(rad)
        y = 1.48 * np.sin(rad)
        ax.text(x, y, label, color="gray", fontsize=11, fontweight="bold", ha="center", va="center", zorder=5)

def calculate_dynamic_energy(angle_deg, front_size, back_size):
    rad = np.radians(angle_deg)
    if front_size == 0 or back_size == 0:
        return 7.0 + 7.0 * np.cos(3 * rad)
    else:
        factor = 1.0
        if front_size > 1 or back_size > 1:
            factor = 1.15
        
        v_offset = 9.435 * factor
        v_cos1 = 2.805 * factor
        v_cos2 = -0.90 * factor
        v_cos3 = 5.73 * factor
        return v_offset + v_cos1 * np.cos(rad) + v_cos2 * np.cos(2 * rad) + v_cos3 * np.cos(3 * rad)

def get_fragment_size(mol, start_idx, avoid_idx):
    visited = {avoid_idx}
    queue = [start_idx]
    size = 0
    while queue:
        curr = queue.pop(0)
        if curr not in visited:
            visited.add(curr)
            if mol.GetAtomWithIdx(curr).GetSymbol() == 'C':
                size += 1
            for nbr in mol.GetAtomWithIdx(curr).GetNeighbors():
                if nbr.GetIdx() not in visited:
                    queue.append(nbr.GetIdx())
    return max(0, size - 1)

_conformation_descriptions = [
    ("Twist-Boat", "A slightly twisted version of the boat form that relieves some torsional strain. More stable than the boat, but less stable than the chair."),
    ("Half-Chair", "A transition state between the chair and boat forms, with some ring atoms out of plane, giving it significant strain energy."),
    ("Boat", "Two 'prow' carbons point the same direction, causing flagpole steric repulsion and torsional strain, giving it high energy."),
    ("Chair", "The most stable form: all bond angles are close to the ideal 109.5 degrees, minimizing both angle strain and torsional strain."),
    ("Envelope", "One ring carbon sits out of the plane formed by the rest, the characteristic shape cyclopentane adopts to reduce angle strain."),
    ("Planar", "All atoms lie in a single plane. Relatively stable for small rings, but disfavored in larger rings due to torsional strain."),
    ("Puckered", "The ring bends out of plane, which reduces torsional strain compared to a fully planar form and is therefore more stable."),
]

def get_conformation_description(name):
    for keyword, desc in _conformation_descriptions:
        if keyword in name:
            return desc
    return "The relative stability of this ring conformation is determined by the balance between angle strain and torsional strain."

def generate_cycloalkane_conformer(smiles, coords):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    conf = mol.GetConformer()
    
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(x, y, z))
        
    prop = AllChem.MMFFGetMoleculeProperties(mol)
    if prop:
        ff = AllChem.MMFFGetMoleculeForceField(mol, prop)
        if ff:
            for i in range(len(coords)):
                ff.AddFixedPoint(i)
            ff.Minimize(maxIts=500)
    return mol


# =========================================================================
# MAIN VIEW SELECTOR
# =========================================================================
view_choice = st.radio(
    "Select View",
    ["🧬 Non-Cyclic Alkanes (Single Bond Rotation)", "🔄 Cyclic Alkanes (Ring Conformations)"],
    horizontal=True,
    key="main_view_choice"
)
st.markdown("---")


# =========================================================================
# VIEW 1 FRAGMENT FUNCTION
# =========================================================================
@st.fragment
def render_non_cyclic_view():
    col_sidebar, col_main = st.columns([1, 3])
    
    with col_sidebar:
        st.header("Simulator Settings")
        mol_choice = st.selectbox("1. Select Alkane Molecule", list(alkanes.keys()), key="tab1_mol_choice")
        smiles = alkanes[mol_choice]
        mol_base = Chem.MolFromSmiles(smiles)

        iupac_labels = compute_iupac_labels(mol_base)

        bond_options = []
        bond_indices = []
        for bond in mol_base.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                a1 = bond.GetBeginAtomIdx()
                a2 = bond.GetEndAtomIdx()
                if mol_base.GetAtomWithIdx(a1).GetSymbol() == 'C' and mol_base.GetAtomWithIdx(a2).GetSymbol() == 'C':
                    label1 = iupac_labels.get(a1, f"C{a1+1}")
                    label2 = iupac_labels.get(a2, f"C{a2+1}")
                    bond_options.append(f"{label1}-{label2}")
                    bond_indices.append((a1, a2))

        bond_choice_str = st.selectbox("2. Select Bond for Newman Projection", bond_options, key="tab1_bond_choice")
        bond_index = bond_options.index(bond_choice_str)
        idx2, idx3 = bond_indices[bond_index]

        temp = st.slider("3. Temperature (Kelvin)", min_value=100, max_value=600, value=298, step=10, key="tab1_temp")

        front_group_size = get_fragment_size(mol_base, idx2, idx3)
        back_group_size = get_fragment_size(mol_base, idx3, idx2)

        st.markdown("---")
        st.markdown("**4. Adjust Dihedral Angle**")

        st.slider("Slider:", 0, 360, key="theta")

        col_r1_1, col_r1_2, col_r1_3 = st.columns(3)
        col_r1_1.button("0° 💥", use_container_width=True, key="btn_0", on_click=update_theta, args=(0,))
        col_r1_2.button("60° 🍀", use_container_width=True, key="btn_60", on_click=update_theta, args=(60,))
        col_r1_3.button("120° 💥", use_container_width=True, key="btn_120", on_click=update_theta, args=(120,))

        col_r2_1, col_r2_2, col_r2_3 = st.columns(3)
        col_r2_1.button("180° ✨", use_container_width=True, key="btn_180", on_click=update_theta, args=(180,))
        col_r2_2.button("240° 💥", use_container_width=True, key="btn_240", on_click=update_theta, args=(240,))
        col_r2_3.button("300° 🍀", use_container_width=True, key="btn_300", on_click=update_theta, args=(300,))

    with col_main:
        R_gas = 0.008314  
        all_angles = np.arange(0, 361, 1)
        all_energies = np.array([calculate_dynamic_energy(a, front_group_size, back_group_size) for a in all_angles])

        min_energy = np.min(all_energies)
        boltzmann_factors = np.exp(-(all_energies - min_energy) / (R_gas * temp))
        probabilities = boltzmann_factors / np.sum(boltzmann_factors)

        current_theta = st.session_state.theta
        current_energy = calculate_dynamic_energy(current_theta, front_group_size, back_group_size)
        current_prob = probabilities[int(current_theta) % 360]

        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("3D Structure (Interactive: Rotate & Zoom)")
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            
            AllChem.EmbedMolecule(mol, randomSeed=42)
            try: AllChem.MMFFOptimizeMolecule(mol)
            except: pass
            
            front_neighbors = [a.GetIdx() for a in mol.GetAtomWithIdx(idx2).GetNeighbors() if a.GetIdx() != idx3]
            back_neighbors = [a.GetIdx() for a in mol.GetAtomWithIdx(idx3).GetNeighbors() if a.GetIdx() != idx2]
            
            f_carbons = sorted([i for i in front_neighbors if mol.GetAtomWithIdx(i).GetSymbol() == 'C'])
            ref_idx1 = f_carbons[0] if f_carbons else sorted(front_neighbors)[0]
            
            b_carbons = sorted([i for i in back_neighbors if mol.GetAtomWithIdx(i).GetSymbol() == 'C'])
            ref_idx4 = b_carbons[0] if b_carbons else sorted(back_neighbors)[0]
            
            AllChem.SetDihedralDeg(mol.GetConformer(), ref_idx1, idx2, idx3, ref_idx4, float(current_theta))
            
            conf = mol.GetConformer()
            pos2 = conf.GetAtomPosition(idx2)
            T_matrix = np.eye(4, dtype=np.double)
            T_matrix[0, 3] = -pos2.x; T_matrix[1, 3] = -pos2.y; T_matrix[2, 3] = -pos2.z
            rdMolTransforms.TransformConformer(conf, T_matrix)
            
            pos3_new = conf.GetAtomPosition(idx3)
            bond_vector = np.array([pos3_new.x, pos3_new.y, pos3_new.z])
            bond_vector = bond_vector / np.linalg.norm(bond_vector)
            
            target_vector = np.array([0.0, 0.0, -1.0])
            axis = np.cross(bond_vector, target_vector)
            axis_len = np.linalg.norm(axis)
            
            R_matrix = np.eye(4, dtype=np.double)
            if axis_len > 1e-6:
                axis = axis / axis_len
                angle = np.arccos(np.clip(np.dot(bond_vector, target_vector), -1.0, 1.0))
                ca = np.cos(angle); sa = np.sin(angle); kx, ky, kz = axis
                R_matrix[0:3, 0:3] = np.array([
                    [ca + kx*kx*(1-ca), kx*ky*(1-ca) - kz*sa, kx*kz*(1-ca) + ky*sa],
                    [ky*kx*(1-ca) + kz*sa, ca + ky*ky*(1-ca), ky*kz*(1-ca) - kx*sa],
                    [kz*kx*(1-ca) - ky*sa, kz*ky*(1-ca) + kx*sa, ca + kz*kz*(1-ca)]
                ])
            rdMolTransforms.TransformConformer(conf, R_matrix)
            
            pos1_new = conf.GetAtomPosition(ref_idx1)
            current_2d_angle = np.arctan2(pos1_new.y, pos1_new.x)
            desired_2d_angle = np.pi / 2.0
            twist_angle = desired_2d_angle - current_2d_angle
            
            Twist_matrix = np.eye(4, dtype=np.double)
            ct = np.cos(twist_angle); st_val = np.sin(twist_angle)
            Twist_matrix[0:3, 0:3] = np.array([[ct, -st_val, 0], [st_val, ct, 0], [0, 0, 1]])
            rdMolTransforms.TransformConformer(conf, Twist_matrix)

            front_subs = []
            for nbr in front_neighbors:
                pos = conf.GetAtomPosition(nbr)
                angle = np.degrees(np.arctan2(pos.y, pos.x)) % 360
                label = get_substituent_label(mol, nbr, idx2)
                front_subs.append((angle, label))
                
            back_subs = []
            for nbr in back_neighbors:
                pos = conf.GetAtomPosition(nbr)
                angle = np.degrees(np.arctan2(pos.y, pos.x)) % 360
                label = get_substituent_label(mol, nbr, idx3)
                back_subs.append((angle, label))

            xyz = Chem.MolToXYZBlock(mol)
            view = py3Dmol.view(width=560, height=440)
            view.addModel(xyz, 'xyz')
            view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
            
            num_atoms = mol.GetNumAtoms()
            for i in range(num_atoms):
                if mol.GetAtomWithIdx(i).GetSymbol() == 'C':
                    pos = conf.GetAtomPosition(i)
                    iupac_text = iupac_labels.get(i, f"C{i+1}")
                    view.addLabel(iupac_text, {
                        'position': {'x': pos.x, 'y': pos.y, 'z': pos.z},
                        'fontColor': 'white', 'backgroundColor': 'black', 'fontsize': 12, 'backgroundOpacity': 0.7, 'alignment': 'center'
                    })
            view.zoomTo()
            view.update()
            
            render_robust_3d_view(view, height=440, width=560, unique_key=f"linear_view_{mol_choice}_{bond_choice_str}_{current_theta}")

        with col2:
            st.subheader("Newman Projection")
            fig_newman, ax_newman = plt.subplots(figsize=(4.2, 4.2), facecolor="white")
            draw_2d_newman(ax_newman, front_subs, back_subs, text_color="black")
            st.pyplot(fig_newman)

        st.markdown("---")
        col_graph1, col_graph2 = st.columns(2)

        with col_graph1:
            st.subheader("Strain Energy vs. Dihedral Angle")
            fig_energy = go.Figure()
            fig_energy.add_trace(go.Scatter(x=all_angles, y=all_energies, mode='lines', line=dict(color='#FF4B4B', width=3)))
            fig_energy.add_trace(go.Scatter(x=[current_theta], y=[current_energy], mode='markers', marker=dict(color='black', size=12, line=dict(color='white', width=2))))
            fig_energy.update_layout(xaxis=dict(title="Dihedral Angle (°)", range=[0, 360], dtick=60), yaxis=dict(title="Relative Energy (kJ/mol)", range=[0, max(all_energies) * 1.1]), margin=dict(l=40, r=40, t=20, b=30), height=260, template="plotly_white", showlegend=False)
            st.plotly_chart(fig_energy, use_container_width=True, key="plot_energy_linear")

        with col_graph2:
            st.subheader("Conformer Probability Distribution vs. Dihedral Angle")
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Scatter(x=all_angles, y=probabilities, mode='lines', line=dict(color='#0068C9', width=3)))
            fig_prob.add_trace(go.Scatter(x=[current_theta], y=[current_prob], mode='markers', marker=dict(color='black', size=12, line=dict(color='white', width=2))))
            fig_prob.update_layout(xaxis=dict(title="Dihedral Angle (°)", range=[0, 360], dtick=60), yaxis=dict(title="Raw Probability Value (0.0 - 1.0)", range=[0, max(probabilities) * 1.15]), margin=dict(l=40, r=40, t=20, b=30), height=260, template="plotly_white", showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True, key="plot_prob_linear")

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Current Angle", f"{current_theta}°")
        col_m2.metric("Total Strain Energy", f"{current_energy:.2f} kJ/mol")
        col_m3.metric("Probability Density", f"{current_prob:.5f}")


# =========================================================================
# VIEW 2 FRAGMENT FUNCTION
# =========================================================================
@st.fragment
def render_cyclic_view():
    col_sidebar, col_main = st.columns([1, 3])

    with col_sidebar:
        st.header("Simulator Settings")
        cyc_mol_choice = st.selectbox("1. Select Cycloalkane", list(cycloalkanes_data.keys()), key="tab2_mol_choice")
        cyc_data = cycloalkanes_data[cyc_mol_choice]

        flip_steps = cyc_data["steps"]
        flip_energies = cyc_data["energies"]
        eng_types = cyc_data["eng_types"]

        st.markdown("---")
        st.markdown("**2. Conformation Selection**")
        if len(flip_steps) > 1:
            ring_step = st.slider("Conformation Transition (Progress):", min_value=0, max_value=len(flip_steps) - 1, value=0, key="tab2_slider")
        else:
            ring_step = 0
            st.info("💡 Cyclopropane is geometrically constrained and highly rigid due to ring strain. It lacks conformational flexibility and exists only in a planar configuration.")

        cyc_temp = st.slider("3. Temperature (Kelvin)", min_value=100, max_value=600, value=298, step=10, key="tab2_temp")

        current_cyc_state = flip_steps[ring_step]
        current_cyc_energy = flip_energies[ring_step]

    with col_main:
        R_gas = 0.008314
        cyc_energies_arr = np.array(flip_energies)
        min_cyc_energy = np.min(cyc_energies_arr)
        cyc_boltzmann_factors = np.exp(-(cyc_energies_arr - min_cyc_energy) / (R_gas * cyc_temp))
        cyc_probabilities = cyc_boltzmann_factors / np.sum(cyc_boltzmann_factors)
        current_cyc_prob = cyc_probabilities[ring_step]

        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("3D Structure (Interactive: Rotate & Zoom)")
            cyc_mol = generate_cycloalkane_conformer(cyc_data["smiles"], cyc_data["coords"][eng_types[ring_step]])

            view_cyc = py3Dmol.view(width=560, height=440)
            view_cyc.addModel(Chem.MolToXYZBlock(cyc_mol), 'xyz')
            view_cyc.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
            view_cyc.zoomTo()
            view_cyc.update()

            render_robust_3d_view(view_cyc, height=440, width=560, unique_key=f"cyclic_view_{cyc_mol_choice}_{ring_step}")

        with col2:
            st.subheader("Conformation Note")
            st.markdown(f"**{current_cyc_state}**")
            st.markdown(get_conformation_description(current_cyc_state))

        st.markdown("---")
        col_graph1, col_graph2 = st.columns(2)

        with col_graph1:
            st.subheader("Energy Profile Diagram")
            fig_cyc = go.Figure()

            if len(flip_steps) > 1:
                fig_cyc.add_trace(go.Scatter(
                    x=list(range(len(flip_steps))),
                    y=flip_energies,
                    mode='lines+markers',
                    line=dict(color='#2ca02c', width=4, shape='spline'),
                    marker=dict(size=8, color='white', line=dict(width=2, color='#2ca02c'))
                ))
                fig_cyc.add_trace(go.Scatter(
                    x=[ring_step],
                    y=[current_cyc_energy],
                    mode='markers',
                    marker=dict(color='black', size=12, line=dict(color='white', width=2))
                ))
                fig_cyc.update_layout(
                    xaxis=dict(title="Conformation Stage", tickmode='array', tickvals=list(range(len(flip_steps))), ticktext=flip_steps),
                    yaxis=dict(title="Relative Energy (kJ/mol)", range=[-2, max(flip_energies) * 1.2 + 2]),
                    margin=dict(l=40, r=40, t=20, b=30),
                    height=260,
                    template="plotly_white",
                    showlegend=False
                )
            else:
                fig_cyc.add_trace(go.Scatter(
                    x=[0],
                    y=[0],
                    mode='markers',
                    marker=dict(color='black', size=12, line=dict(color='white', width=2))
                ))
                fig_cyc.update_layout(
                    xaxis=dict(title="Conformation Stage", tickmode='array', tickvals=[0], ticktext=flip_steps),
                    yaxis=dict(title="Relative Energy (kJ/mol)", range=[-2, 5]),
                    margin=dict(l=40, r=40, t=20, b=30),
                    height=260,
                    template="plotly_white",
                    showlegend=False
                )

            st.plotly_chart(fig_cyc, use_container_width=True, key="plot_energy_cyclic")

        with col_graph2:
            st.subheader("Conformer Probability Distribution")
            fig_cyc_prob = go.Figure()

            if len(flip_steps) > 1:
                fig_cyc_prob.add_trace(go.Scatter(
                    x=list(range(len(flip_steps))),
                    y=cyc_probabilities,
                    mode='lines+markers',
                    line=dict(color='#0068C9', width=4, shape='spline'),
                    marker=dict(size=8, color='white', line=dict(width=2, color='#0068C9'))
                ))
                fig_cyc_prob.add_trace(go.Scatter(
                    x=[ring_step],
                    y=[current_cyc_prob],
                    mode='markers',
                    marker=dict(color='black', size=12, line=dict(color='white', width=2))
                ))
                fig_cyc_prob.update_layout(
                    xaxis=dict(title="Conformation Stage", tickmode='array', tickvals=list(range(len(flip_steps))), ticktext=flip_steps),
                    yaxis=dict(title="Raw Probability Value (0.0 - 1.0)", range=[0, max(cyc_probabilities) * 1.15]),
                    margin=dict(l=40, r=40, t=20, b=30),
                    height=260,
                    template="plotly_white",
                    showlegend=False
                )
            else:
                fig_cyc_prob.add_trace(go.Scatter(
                    x=[0],
                    y=[1.0],
                    mode='markers',
                    marker=dict(color='black', size=12, line=dict(color='white', width=2))
                ))
                fig_cyc_prob.update_layout(
                    xaxis=dict(title="Conformation Stage", tickmode='array', tickvals=[0], ticktext=flip_steps),
                    yaxis=dict(title="Raw Probability Value (0.0 - 1.0)", range=[0, 1.15]),
                    margin=dict(l=40, r=40, t=20, b=30),
                    height=260,
                    template="plotly_white",
                    showlegend=False
                )

            st.plotly_chart(fig_cyc_prob, use_container_width=True, key="plot_prob_cyclic")

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Current Conformation", current_cyc_state)
        col_m2.metric("Relative Strain Energy", f"{current_cyc_energy:.2f} kJ/mol")
        col_m3.metric("Probability Density", f"{current_cyc_prob:.5f}")


# =========================================================================
# CONDITIONAL EXECUTION (Calls the corresponding Fragment)
# =========================================================================
if view_choice == "🧬 Non-Cyclic Alkanes (Single Bond Rotation)":
    render_non_cyclic_view()
elif view_choice == "🔄 Cyclic Alkanes (Ring Conformations)":
    render_cyclic_view()
