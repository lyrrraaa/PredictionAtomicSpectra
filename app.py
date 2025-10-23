import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Physical Constants
# ----------------------------
h = 6.626e-34
c = 3.0e8
eV = 1.602e-19
d_default = 1.0 / (600e3)  # 600 lines/mm → 1.6667e-6 m

# ----------------------------
# Utility Functions
# ----------------------------
def wavelength_to_rgb(wl):
    wl = float(wl)
    if wl < 380 or wl > 780:
        return (0.0, 0.0, 0.0)
    if wl < 440:
        r, g, b = -(wl - 440) / 60, 0.0, 1.0
    elif wl < 490:
        r, g, b = 0.0, (wl - 440) / 50, 1.0
    elif wl < 510:
        r, g, b = 0.0, 1.0, -(wl - 510) / 20
    elif wl < 580:
        r, g, b = (wl - 510) / 70, 1.0, 0.0
    elif wl < 645:
        r, g, b = 1.0, -(wl - 645) / 65, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    factor = 1.0 if 420 <= wl <= 700 else 0.3
    return (max(0.0, r * factor), max(0.0, g * factor), max(0.0, b * factor))

def photon_energy_eV(wl_nm):
    wl_m = wl_nm * 1e-9
    E_J = h * c / wl_m
    return E_J / eV

def zone_label(wl, zones):
    for color, (low, high) in zones.items():
        if low <= wl < high:
            return color
    return "None"

# ----------------------------
# Streamlit Configuration
# ----------------------------
st.set_page_config(page_title="LED Spectrum Visualizer", layout="wide")
st.title("💡 LED Spectrum Visualizer")
st.caption("Visualize LED emission spectrum using diffraction geometry, wavelength, photon energy, and color zones.")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
d_input = st.sidebar.number_input("Grating spacing (d, m)", value=float(d_default), format="%.6e")
sigma_nm = st.sidebar.slider("Gaussian width per peak (σ, nm)", 2.0, 80.0, 12.0)
normalize = st.sidebar.checkbox("Normalize intensity (max = 1)", value=True)
show_peaks = st.sidebar.checkbox("Show λ markers on graph", value=True)
st.sidebar.markdown("---")

# ----------------------------
# Editable Visible Spectrum Zones (6 colors only)
# ----------------------------
st.sidebar.subheader("🎨 Editable Visible Spectrum Ranges (nm)")

zones = {
    "Violet": [
        st.sidebar.number_input("Violet min", 380, 800, 405),
        st.sidebar.number_input("Violet max", 380, 800, 410)
    ],
    "Blue": [
        st.sidebar.number_input("Blue min", 380, 800, 415),
        st.sidebar.number_input("Blue max", 380, 800, 470)
    ],
    "Green": [
        st.sidebar.number_input("Green min", 380, 800, 495),
        st.sidebar.number_input("Green max", 380, 800, 530)
    ],
    "Yellow": [
        st.sidebar.number_input("Yellow min", 380, 800, 550),
        st.sidebar.number_input("Yellow max", 380, 800, 590)
    ],
    "Orange": [
        st.sidebar.number_input("Orange min", 380, 800, 599),
        st.sidebar.number_input("Orange max", 380, 800, 629)
    ],
    "Red": [
        st.sidebar.number_input("Red min", 380, 800, 600),
        st.sidebar.number_input("Red max", 380, 800, 700)
    ],
}

st.sidebar.markdown("You can adjust wavelength limits for each color zone above.")

# ----------------------------
# Display All Color Zones
# ----------------------------
st.markdown("### 🎨 All Color Zones (Editable in Sidebar)")
cols = st.columns(len(zones))
for i, (color, (low, high)) in enumerate(zones.items()):
    rgb = wavelength_to_rgb((low + high) / 2)
    css = f"background: rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}); height:60px; border-radius:6px; border:1px solid #ccc;"
    with cols[i]:
        st.markdown(f"<div style='{css}'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='text-align:center'><small>{color}: {low}–{high} nm</small></div>",
            unsafe_allow_html=True
        )

# ----------------------------
# Input Diffraction Data
# ----------------------------
st.subheader("📋 Input Diffraction Data (in mm)")
default = pd.DataFrame({"X (mm)": [20, 25, 30, 35, 40], "Y (mm)": [100]*5})
data = st.data_editor(default, num_rows="dynamic", use_container_width=True)

# ----------------------------
# Calculations
# ----------------------------
X_m = data["X (mm)"].values * 1e-3
Y_m = data["Y (mm)"].values * 1e-3
theta_rad = np.arctan2(X_m, Y_m)
theta_deg = np.degrees(theta_rad)
lambda_m = d_input * np.sin(theta_rad)
lambda_nm = np.where(lambda_m > 0, lambda_m * 1e9, np.nan)
E_eV = [photon_energy_eV(wl) if not np.isnan(wl) else np.nan for wl in lambda_nm]
color_zones = [zone_label(wl, zones) for wl in lambda_nm]

results = pd.DataFrame({
    "X (mm)": data["X (mm)"],
    "Y (mm)": data["Y (mm)"],
    "θ (°)": theta_deg,
    "λ (nm)": lambda_nm,
    "E (eV)": E_eV,
    "Color Zone": color_zones
})

st.subheader("📊 Computed LED Emission Data")
st.dataframe(results.style.format({"θ (°)": "{:.2f}", "λ (nm)": "{:.1f}", "E (eV)": "{:.2f}"}), use_container_width=True)

# ----------------------------
# Spectrum Visualization
# ----------------------------
st.subheader("📈 Emission Spectrum")
wavelengths = np.linspace(380, 780, 1000)
I = np.zeros_like(wavelengths)
valid_wls = [wl for wl in lambda_nm if not np.isnan(wl)]
for wl in valid_wls:
    I += np.exp(-0.5 * ((wavelengths - wl) / sigma_nm) ** 2)
if normalize and I.max() > 0:
    I /= I.max()

bg_rgb = np.array([wavelength_to_rgb(w) for w in wavelengths])
bg_img = np.tile(bg_rgb.reshape(1, len(wavelengths), 3), (40, 1, 1))

fig, ax = plt.subplots(figsize=(12, 4))
ax.imshow(bg_img, extent=[380, 780, 0, 1], aspect='auto', origin='lower')
ax.plot(wavelengths, I, color='black', linewidth=1.8)
if show_peaks:
    for wl in valid_wls:
        ax.axvline(wl, color=wavelength_to_rgb(wl), linewidth=1.2, alpha=0.9)
        ax.text(wl, 1.02, f"{wl:.0f} nm", rotation=90, ha="center", fontsize=8)
ax.set_xlim(380, 780)
ax.set_ylim(0, 1.1)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Relative Intensity")
ax.grid(alpha=0.25)
st.pyplot(fig)

# ----------------------------
# Photon Energy vs Wavelength
# ----------------------------
st.subheader("⚡ Photon Energy vs Wavelength")
fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.plot(wavelengths, [photon_energy_eV(w) for w in wavelengths], color="gray", linestyle="--", alpha=0.6)
ax2.scatter(results["λ (nm)"], results["E (eV)"],
            color=[wavelength_to_rgb(max(380, min(780, wl))) for wl in results["λ (nm)"]],
            edgecolors="k", s=90)
ax2.set_xlim(380, 780)
ax2.set_xlabel("Wavelength (nm)")
ax2.set_ylabel("Photon Energy (eV)")
ax2.grid(alpha=0.3)
st.pyplot(fig2)

# ----------------------------
# Interpretation
# ----------------------------
st.markdown("---")
st.subheader("📘 Simulation Interpretation")
st.write(
    "- This simulation demonstrates how **LED colors** correspond to specific wavelengths and photon energies.\n"
    "- Input **X (mm)** and **Y (mm)** represent diffraction geometry: X is lateral displacement, Y is grating–screen distance.\n"
    "- Wavelengths are computed using the **grating equation** λ = d·sinθ, where θ = arctan(X/Y).\n"
    "- The main spectrum shows Gaussian peaks at the computed λ values.\n"
    "- The energy plot confirms the **inverse relationship**: shorter wavelength → higher photon energy.\n"
)
