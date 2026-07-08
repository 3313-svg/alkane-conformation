import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import py3Dmol
import time
from stmol import showmol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms

# Page Setup
st.set_page_config(layout="wide")
st.title("알케인 형태 이성질체: 회전 각도에 따른 에너지 및 확률 분포")

# --- Initialize Session States Safely ---
if 'theta' not in st.session_state:
    st.session_state.theta = 180
if 'animating' not in st.session_state:
    st.session_state.animating = False

# --- Definitions and Functions ---
# Proper chemical formulas with mathematical subscript formatting
alkyl_labels = {
    0: "H", 
    1: r"CH$_3$", 
    2: r"C$_2$H$_5$", 
    3: r"C$_3$H$_7$", 
    4: r"C$_4$H$_9$", 
    5: r"C$_5$H$_{11}$", 
    6: r"C$_6$H$_{13}$", 
    7: r"C$_7$H$_{15}$"
}

def get_substituent_label(mol, nbr_idx, root_idx):
    """Dynamically determines the alkyl formula for a specific branch vertex."""
    atom = mol.GetAtomWithIdx(nbr_idx)
    if atom.GetSymbol() == 'H':
        return "H"
    
    # Count total carbons belonging purely to this specific substituent branch
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
    """Draws a dynamically generated Newman projection using exact 3D angles."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    
    # 1. Draw Back Bonds (drawn out from center, layered behind front circle)
    for angle, label in back_subs:
        rad = np.radians(angle)
        ax.plot([0, 1.15 * np.cos(rad)], [0, 1.15 * np.sin(rad)], color="darkgray", lw=2.5, zorder=1)
    
    # 2. Draw Front Central Circle
    circle = plt.Circle((0, 0), 0.5, edgecolor=text_color, facecolor="white", fill=True, lw=2, zorder=2)
    ax.add_patch(circle)
    
    # 3. Draw Front Bonds (radiate strictly from the center hub)
    for angle, label in front_subs:
        rad = np.radians(angle)
        ax.plot([0, 1.0 * np.cos(rad)], [0, 1.0 * np.sin(rad)], color=text_color, lw=3.5, zorder=3)
        
    # Central intersection vertex point
    ax.scatter(0, 0, color=text_color, s=90, zorder=4)
    
    # 4. Render Front Labels
    for angle, label in front_subs:
        rad = np.radians(angle)
        x = 1.32 * np.cos(rad)
        y = 1.32 * np.sin(rad)
        ax.text(x, y, label, color=text_color, fontsize=12, fontweight="black", ha="center", va="center", zorder=5)
        
    # 5. Render Back Labels
    for angle, label in back_subs:
        rad = np.radians(angle)
        x = 1.48 * np.cos(rad)
        y = 1.48 * np.sin(rad)
        ax.text(x, y, label, color="gray", fontsize=11, fontweight="bold", ha="center", va="center", zorder=5)

def calculate_dynamic_energy(angle_deg, front_size, back_size):
    """Calculates torsional potential strain energy ensuring a strict >= 0.0 baseline."""
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

# --- Left Sidebar: User Controls ---
st.sidebar.header("Simulation Settings")

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

mol_choice = st.sidebar.selectbox("1. 알케인 분자 선택", list(alkanes.keys()))
smiles = alkanes[mol_choice]
mol_base = Chem.MolFromSmiles(smiles)

bond_options = []
bond_indices = []
for bond in mol_base.GetBonds():
    if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if mol_base.GetAtomWithIdx(a1).GetSymbol() == 'C' and mol_base.GetAtomWithIdx(a2).GetSymbol() == 'C':
            bond_options.append(f"C{a1+1}-C{a2+1}")
            bond_indices.append((a1, a2))

bond_choice_str = st.sidebar.selectbox("2. 뉴먼 투영법 적용할 결합 선택", bond_options)
bond_index = bond_options.index(bond_choice_str)
idx2, idx3 = bond_indices[bond_index]

temp = st.sidebar.slider("3. 온도 (Kelvin)", min_value=100, max_value=600, value=298, step=10)

front_group_size = get_fragment_size(mol_base, idx2, idx3)
back_group_size = get_fragment_size(mol_base, idx3, idx2)

# --- UPDATED LAYOUT ORDER: Manual Angle Adjustments positioned higher up ---
st.sidebar.markdown("---")
st.sidebar.markdown("**4. 회전 각도 조정**")

theta_manual = st.sidebar.slider("슬라이드바:", 0, 360, value=int(st.session_state.theta))
if not st.session_state.animating:
    st.session_state.theta = theta_manual

# 6-Button Grid Layout underneath the primary slider
col_r1_1, col_r1_2, col_r1_3 = st.sidebar.columns(3)
if col_r1_1.button("0°", use_container_width=True, help="Fully Eclipsed"): 
    st.session_state.theta = 0
if col_r1_2.button("60°", use_container_width=True, help="Gauche / Staggered"): 
    st.session_state.theta = 60
if col_r1_3.button("120°", use_container_width=True, help="Eclipsed"): 
    st.session_state.theta = 120

col_r2_1, col_r2_2, col_r2_3 = st.sidebar.columns(3)
if col_r2_1.button("180°", use_container_width=True, help="Anti / Staggered"): 
    st.session_state.theta = 180
if col_r2_2.button("240°", use_container_width=True, help="Eclipsed"): 
    st.session_state.theta = 240
if col_r2_3.button("300°", use_container_width=True, help="Gauche / Staggered"): 
    st.session_state.theta = 300

# --- UPDATED LAYOUT ORDER: Animation Drive sent to the bottom ---
st.sidebar.markdown("---")
st.sidebar.markdown("**5. 자동 애니메이션**")

col_play, col_stop = st.sidebar.columns(2)
if col_play.button("▶️ Play", use_container_width=True):
    st.session_state.animating = True
if col_stop.button("⏹️ Stop", use_container_width=True):
    st.session_state.animating = False

# --- THERMODYNAMIC PROBABILITY ENGINE (Boltzmann Calculation) ---
R_gas = 0.008314  
all_angles = np.arange(0, 361, 1)
all_energies = np.array([calculate_dynamic_energy(a, front_group_size, back_group_size) for a in all_angles])

min_energy = np.min(all_energies)
boltzmann_factors = np.exp(-(all_energies - min_energy) / (R_gas * temp))
probabilities = boltzmann_factors / np.sum(boltzmann_factors)

current_theta = st.session_state.theta
current_energy = calculate_dynamic_energy(current_theta, front_group_size, back_group_size)
current_prob = probabilities[int(current_theta) % 360]

# --- RENDER LAYOUT SYSTEM ---
col1, col2 = st.columns(2)

# Top Left: Molecular Viewer
with col1:
    st.subheader(f"3D 구조(카메라 각도 조정 + 확대/축소 가능)")
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
        R_matrix[0:3, 0:3] = np.array([[ca + kx*kx*(1-ca), kx*ky*(1-ca) - kz*sa, kx*kz*(1-ca) + ky*sa],[ky*kx*(1-ca) + kz*sa, ca + ky*ky*(1-ca), ky*kz*(1-ca) - kx*sa],[kz*kx*(1-ca) - ky*sa, kz*ky*(1-ca) + kx*sa, ca + kz*kz*(1-ca)]])
    rdMolTransforms.TransformConformer(conf, R_matrix)
    
    pos1_new = conf.GetAtomPosition(ref_idx1)
    current_2d_angle = np.arctan2(pos1_new.y, pos1_new.x)
    desired_2d_angle = np.pi / 2.0
    twist_angle = desired_2d_angle - current_2d_angle
    
    Twist_matrix = np.eye(4, dtype=np.double)
    ct = np.cos(twist_angle); st_val = np.sin(twist_angle)
    Twist_matrix[0:3, 0:3] = np.array([[ct, -st_val, 0], [st_val, ct, 0], [0, 0, 1]])
    rdMolTransforms.TransformConformer(conf, Twist_matrix)

    # Dynamic coordinate parsing for the custom Newman projection
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
    view = py3Dmol.view(width=450, height=320)
    view.addModel(xyz, 'xyz')
    view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
    
    num_atoms = mol.GetNumAtoms()
    for i in range(num_atoms):
        if mol.GetAtomWithIdx(i).GetSymbol() == 'C':
            pos = conf.GetAtomPosition(i)
            view.addLabel(f'C{i+1}', {
                'position': {'x': pos.x, 'y': pos.y, 'z': pos.z},
                'fontColor': 'white', 'backgroundColor': 'black', 'fontsize': 12, 'backgroundOpacity': 0.7, 'alignment': 'center'
            })
    view.zoomTo()
    showmol(view, height=320, width=450)

# Top Right: Newman Diagram
with col2:
    st.subheader(f"뉴먼 투영법 다이어그램")
    fig_newman, ax_newman = plt.subplots(figsize=(4.2, 4.2), facecolor="white")
    draw_2d_newman(ax_newman, front_subs, back_subs, text_color="black")
    st.pyplot(fig_newman)

# --- SPLIT ANALYSIS PLOTS SECTION ---
st.markdown("---")
col_graph1, col_graph2 = st.columns(2)

# Graph 1: Torsional Potential Energy vs Angle
with col_graph1:
    st.subheader("회전 각도에 따른 스트레인 에너지")
    fig_energy = go.Figure()
    
    fig_energy.add_trace(go.Scatter(
        x=all_angles, y=all_energies,
        mode='lines', line=dict(color='#FF4B4B', width=3),
        name='Potential Energy'
    ))
    fig_energy.add_trace(go.Scatter(
        x=[current_theta], y=[current_energy],
        mode='markers', marker=dict(color='black', size=12, line=dict(color='white', width=2)),
        name='Current State', showlegend=False
    ))
    
    fig_energy.update_layout(
        xaxis=dict(title="Dihedral Angle (°)", range=[0, 360], tickmode='linear', tick0=0, dtick=60),
        yaxis=dict(title="Relative Energy (kJ/mol)", range=[0, max(all_energies) * 1.1]),
        margin=dict(l=40, r=40, t=20, b=40), height=350, template="plotly_white"
    )
    st.plotly_chart(fig_energy, use_container_width=True)

# Graph 2: Thermodynamic Probability Population vs Angle
with col_graph2:
    st.subheader("회전 각도에 따른 형태 이성질체 분포 확률")
    fig_prob = go.Figure()
    
    fig_prob.add_trace(go.Scatter(
        x=all_angles, y=probabilities,
        mode='lines', line=dict(color='#0068C9', width=3),
        name='Probability'
    ))
    fig_prob.add_trace(go.Scatter(
        x=[current_theta], y=[current_prob],
        mode='markers', marker=dict(color='black', size=12, line=dict(color='white', width=2)),
        name='Current State', showlegend=False
    ))
    
    fig_prob.update_layout(
        xaxis=dict(title="Dihedral Angle (°)", range=[0, 360], tickmode='linear', tick0=0, dtick=60),
        yaxis=dict(title="Raw Probability Value (0.0 - 1.0)", range=[0, max(probabilities) * 1.15]),
        margin=dict(l=40, r=40, t=20, b=40), height=350, template="plotly_white"
    )
    st.plotly_chart(fig_prob, use_container_width=True)

# --- 4. DATA TELEMETRY READOUTS ---
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("현재 회전 각도", f"{current_theta}°")
col_m2.metric("총 스트레인 에너지", f"{current_energy:.2f} kJ/mol")
col_m3.metric("확률", f"{current_prob:.5f}")

# --- RE-RUN ANIMATION DRIVER ---
if st.session_state.animating:
    time.sleep(0.01)
    st.session_state.theta = (st.session_state.theta + 6) % 360
    st.rerun()
