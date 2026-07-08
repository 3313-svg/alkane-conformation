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
st.title("🔬 Alkane Conformation: Energy vs. Raw Probability Analysis")

# --- Initialize Session States Safely ---
if 'theta' not in st.session_state:
    st.session_state.theta = 180
if 'animating' not in st.session_state:
    st.session_state.animating = False

# --- Definitions and Functions ---
alkyl_labels = {
    0: "H", 1: "Me", 2: "Et", 3: "Pr", 
    4: "Bu", 5: "Pe", 6: "Hx", 7: "Hp"
}

def draw_2d_newman(ax, front_group, back_group, angle_deg, text_color="black"):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    
    circle = plt.Circle((0, 0), 0.5, edgecolor=text_color, facecolor="white", fill=True, lw=2, zorder=2)
    ax.add_patch(circle)
    
    front_angles = [90, 210, 330]
    ax.plot([0, np.cos(np.radians(front_angles[0]))], [0, np.sin(np.radians(front_angles[0]))], color=text_color, lw=4.5, zorder=3)
    for a in front_angles[1:]:
        ax.plot([0, np.cos(np.radians(a))], [0, np.sin(np.radians(a))], color=text_color, lw=3, zorder=3)
        
    ax.scatter(0, 0, color=text_color, s=80, zorder=4)
        
    back_angles = [(a - angle_deg) % 360 for a in front_angles]
    r = 1.2
    rad0 = np.radians(back_angles[0])
    ax.plot([0, r*np.cos(rad0)], [0, r*np.sin(rad0)], color="gray", lw=2, zorder=1)
    for a in back_angles[1:]:
        rad = np.radians(a)
        ax.plot([0, np.cos(rad)], [0, np.sin(rad)], color="gray", lw=1.5, zorder=1)
        
    label_dist = 1.3
    front_labels = [front_group, "H", "H"]
    for i, a in enumerate(front_angles):
        rad = np.radians(a)
        x = label_dist * np.cos(rad)
        y = label_dist * np.sin(rad)
        ax.text(x, y, front_labels[i], color=text_color, fontsize=13, fontweight="black", ha="center", va="center", zorder=5)
        
    label_dist_back = 1.45
    back_labels = [back_group, "H", "H"]
    for i, a in enumerate(back_labels):
        rad = np.radians(back_angles[i])
        x = label_dist_back * np.cos(rad)
        y = label_dist_back * np.sin(rad)
        ax.text(x, y, back_labels[i], color="gray", fontsize=11, fontweight="bold", ha="center", va="center", zorder=5)

def calculate_dynamic_energy(angle_deg, front_size, back_size):
    rad = np.radians(angle_deg)
    if front_size == 0 or back_size == 0:
        return 7.0 + 7.0 * np.cos(3 * rad)
    else:
        factor = 1.0
        if front_size > 1 or back_size > 1:
            factor = 1.15
        v_offset = 9.31 * factor
        v_cos1 = 2.805 * factor
        v_cos2 = -0.90 * factor
        v_cos3 = 5.73 * factor
        return v_offset + v_cos1 * np.cos(rad) + v_cos2 * np.cos(2 * rad) + v_cos3 * np.cos(3 * rad)

# --- Left Sidebar: User Controls ---
st.sidebar.header("Simulation Settings")

alkanes = {"Butane": "CCCC", "Pentane": "CCCCC", "Hexane": "CCCCCC", "Heptane": "CCCCCCC"}
mol_choice = st.sidebar.selectbox("1. Select Molecule", list(alkanes.keys()))
smiles = alkanes[mol_choice]
num_carbons = len(smiles)

bond_options = [f"C{i}-C{i+1}" for i in range(1, num_carbons)]
bond_choice = st.sidebar.selectbox("2. Select Bond to View", bond_options)

temp = st.sidebar.slider("3. Temperature (Kelvin)", min_value=100, max_value=600, value=298, step=10)

# --- Define dynamic atomic mappings and sizes ---
c_start_label = int(bond_choice.split('-')[0][1:]) 
c_start = c_start_label - 1 
c_end = c_start + 1       

front_group_size = c_start
back_group_size = num_carbons - 1 - c_end
front_group_name = alkyl_labels[front_group_size]
back_group_name = alkyl_labels[back_group_size]

idx2 = c_start
idx3 = c_end

# --- Animation Controls Configuration ---
st.sidebar.markdown("---")
st.sidebar.markdown("**4. Animation Controls**")

col_play, col_stop = st.sidebar.columns(2)
if col_play.button("▶️ Play Animation", use_container_width=True):
    st.session_state.animating = True
if col_stop.button("⏹️ Stop Animation", use_container_width=True):
    st.session_state.animating = False

st.sidebar.markdown("---")
st.sidebar.markdown("**5. Manual Override Controls**")

if front_group_size == 0 or back_group_size == 0:
    if st.sidebar.button("🔴 Peak: Eclipsed (0°)"): st.session_state.theta = 0
    if st.sidebar.button("🟢 Valley: Staggered (60°)"): st.session_state.theta = 60
    if st.sidebar.button("🟢 Valley: Staggered (180°)"): st.session_state.theta = 180
else:
    if st.sidebar.button("🔴 Peak: Fully Eclipsed (0°)"): st.session_state.theta = 0
    if st.sidebar.button("🟢 Valley: Gauche Staggered (60°)"): st.session_state.theta = 60
    if st.sidebar.button("🟢 Valley: Anti Staggered (180°)"): st.session_state.theta = 180

theta_manual = st.sidebar.slider("Fine-tune Angle manually:", 0, 360, value=int(st.session_state.theta))
if not st.session_state.animating:
    st.session_state.theta = theta_manual

# --- THERMODYNAMIC PROBABILITY ENGINE (Boltzmann Calculation) ---
R_gas = 0.008314  
all_angles = np.arange(0, 361, 1)
all_energies = np.array([calculate_dynamic_energy(a, front_group_size, back_group_size) for a in all_angles])

min_energy = np.min(all_energies)
boltzmann_factors = np.exp(-(all_energies - min_energy) / (R_gas * temp))

# FIX: Removed the '* 100' multiplication to yield raw decimal values strictly between 0 and 1
probabilities = boltzmann_factors / np.sum(boltzmann_factors)

current_theta = st.session_state.theta
current_energy = calculate_dynamic_energy(current_theta, front_group_size, back_group_size)
current_prob = probabilities[int(current_theta) % 360]

# --- RENDER LAYOUT SYSTEM ---
col1, col2 = st.columns(2)

# Top Left: Molecular Viewer
with col1:
    st.subheader(f"Interactive 3D Structure")
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

    xyz = Chem.MolToXYZBlock(mol)
    view = py3Dmol.view(width=450, height=320)
    view.addModel(xyz, 'xyz')
    view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
    
    for i in range(num_carbons):
        pos = conf.GetAtomPosition(i)
        view.addLabel(f'C{i+1}', {
            'position': {'x': pos.x, 'y': pos.y, 'z': pos.z},
            'fontColor': 'white', 'backgroundColor': 'black', 'fontsize': 12, 'backgroundOpacity': 0.7, 'alignment': 'center'
        })
    view.zoomTo()
    showmol(view, height=320, width=450)

# Top Right: Newman Diagram
with col2:
    st.subheader(f"Newman Projection")
    fig_newman, ax_newman = plt.subplots(figsize=(4.2, 4.2), facecolor="white")
    draw_2d_newman(ax_newman, front_group_name, back_group_name, current_theta, text_color="black")
    st.pyplot(fig_newman)

# --- SPLIT ANALYSIS PLOTS SECTION ---
st.markdown("---")
col_graph1, col_graph2 = st.columns(2)

# Graph 1: Torsional Potential Energy vs Angle
with col_graph1:
    st.subheader("📈 Potential Energy Profile")
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
        yaxis=dict(title="Relative Energy (kJ/mol)"),
        margin=dict(l=40, r=40, t=20, b=40), height=350, template="plotly_white"
    )
    st.plotly_chart(fig_energy, use_container_width=True)

# Graph 2: Thermodynamic Probability Population vs Angle (Raw Decimals)
with col_graph2:
    st.subheader("📊 Conformation Probability Density")
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
col_m1.metric("Current Angle", f"{current_theta}°")
col_m2.metric("Steric Strain Energy", f"{current_energy:.2f} kJ/mol")
col_m3.metric("Raw Boltzmann Probability", f"{current_prob:.5f}")

# --- RE-RUN ANIMATION DRIVER ---
if st.session_state.animating:
    time.sleep(0.01)
    st.session_state.theta = (st.session_state.theta + 6) % 360
    st.rerun()