#plot l1b vs
import numpy as np
import self
from scipy.constants import sigma

from common.io.writeToa import writeToa, readToa
from ism.src.detectionPhase import detectionPhase

#ISM-001

#raw data
toa_0=readToa("C:\\Master\\aa\\EODP\\EODP-TS-ISM\\output","ism_toa_isrf_VNIR-0.nc")
toa_1=readToa("C:\\Master\\aa\EODP\EODP-TS-ISM\\output","ism_toa_isrf_VNIR-1.nc")
toa_2=readToa("C:\\Master\\aa\EODP\\EODP-TS-ISM\\output","ism_toa_isrf_VNIR-2.nc")
toa_3=readToa("C:\\Master\\aa\EODP\\EODP-TS-ISM\\output","ism_toa_isrf_VNIR-3.nc")

#Myoutputs
toa_output_0=readToa("C:\\Master\\aa\\EODP\\EODP-TS-ISM\\myoutput","ism_toa_isrf_VNIR-0.nc")
toa_output_1=readToa("C:\\Master\\aa\EODP\EODP-TS-ISM\\myoutput","ism_toa_isrf_VNIR-1.nc")
toa_output_2=readToa("C:\\Master\\aa\EODP\\EODP-TS-ISM\\myoutput","ism_toa_isrf_VNIR-2.nc")
toa_output_3=readToa("C:\\Master\\aa\EODP\\EODP-TS-ISM\\myoutput","ism_toa_isrf_VNIR-3.nc")

#Exercise 1

def check_band(toa_my, toa_ref, tol=0.01):
    dif= np.abs(toa_ref-toa_my) / np.maximum(toa_ref,1e-12) * 100
    mu = np.mean(dif)
    sigma=np.std(dif)
    threshold = mu + 3*sigma
    ok = threshold < tol
    return mu, sigma, threshold, ok
mu0, sigma0, threshold0, ok0=check_band(toa_0,toa_output_0, tol=0.01)
mu1, sigma1, threshold1, ok1=check_band(toa_1,toa_output_1, tol=0.01)
mu2, sigma2, threshold2, ok2=check_band(toa_2,toa_output_2, tol=0.01)
mu3, sigma3, threshold3, ok3=check_band(toa_3,toa_output_3, tol=0.01)
print("Criteria 1")
print("Band 0: mean=", mu0, "std=", sigma0, " threshold=", threshold0, "Is it below 0.01%?? ", ok0)
print("Band 1: mean=", mu1, "std=", sigma1, " threshold=", threshold1, "Is it below 0.01%?? ", ok1)
print("Band 2: mean=", mu2, "std=", sigma2, " threshold=", threshold2, "Is it below 0.01%?? ", ok2)
print("Band 3: mean=", mu3, "std=", sigma3, " threshold=", threshold3, "Is it below 0.01%?? ", ok3)

#Exercise 2

#Optical Output
toa_optical_0=readToa("C:\\Master\\aa\\EODP\\EODP-TS-ISM\\output","ism_toa_optical_VNIR-0.nc")
toa_optical_1=readToa("C:\\Master\\aa\EODP\EODP-TS-ISM\\output","ism_toa_optical_VNIR-1.nc")
toa_optical_2=readToa("C:\\Master\\aa\EODP\\EODP-TS-ISM\\output","ism_toa_optical_VNIR-2.nc")
toa_optical_3=readToa("C:\\Master\\aa\EODP\\EODP-TS-ISM\\output","ism_toa_optical_VNIR-3.nc")


#Myoutputs_optical
toa_optical_output_0=readToa("C:\\Master\\aa\\EODP\\EODP-TS-ISM\\myoutput","ism_toa_optical_VNIR-0.nc")
toa_optical_output_1=readToa("C:\\Master\\aa\EODP\EODP-TS-ISM\\myoutput","ism_toa_optical_VNIR-1.nc")
toa_optical_output_2=readToa("C:\\Master\\aa\EODP\\EODP-TS-ISM\\myoutput","ism_toa_optical_VNIR-2.nc")
toa_optical_output_3=readToa("C:\\Master\\aa\EODP\\EODP-TS-ISM\\myoutput","ism_toa_optical_VNIR-3.nc")

mu0, sigma0, threshold0, ok0=check_band(toa_optical_0,toa_optical_output_0, tol=0.01)
mu1, sigma1, threshold1, ok1=check_band(toa_optical_1,toa_optical_output_1, tol=0.01)
mu2, sigma2, threshold2, ok2=check_band(toa_optical_2,toa_optical_output_2, tol=0.01)
mu3, sigma3, threshold3, ok3=check_band(toa_optical_3,toa_optical_output_3, tol=0.01)
print("Criteria 2")
print("Band 0: mean=", mu0, "std=", sigma0, " threshold=", threshold0, "Is it below 0.01%?? ", ok0)
print("Band 1: mean=", mu1, "std=", sigma1, " threshold=", threshold1, "Is it below 0.01%?? ", ok1)
print("Band 2: mean=", mu2, "std=", sigma2, " threshold=", threshold2, "Is it below 0.01%?? ", ok2)
print("Band 3: mean=", mu3, "std=", sigma3, " threshold=", threshold3, "Is it below 0.01%?? ", ok3)

#Exercise 3

#Variables del siestma

D = 0.15
f = 0.5262
Tr = 0.99
factor = Tr * (np.pi/4) * (D/f)**2
print("Criteria 3")
print("The factor value is=",factor)


#ISM-002

toa_0=readToa("C:\\Master\\aa\\EODP\\EODP-TS-ISM\\output","ism_toa_VNIR-0.nc")
toa_1=readToa("C:\\Master\\aa\EODP\EODP-TS-ISM\\output","ism_toa_VNIR-1.nc")
toa_2=readToa("C:\\Master\\aa\EODP\\EODP-TS-ISM\\output","ism_toa_VNIR-2.nc")
toa_3=readToa("C:\\Master\\aa\EODP\\EODP-TS-ISM\\output","ism_toa_VNIR-3.nc")

#Myoutputs
toa_output_00=readToa("C:\\Master\\aa\\EODP\\EODP-TS-ISM\\myoutput","ism_toa_VNIR-0.nc")
toa_output_11=readToa("C:\\Master\\aa\EODP\EODP-TS-ISM\\myoutput","ism_toa_VNIR-1.nc")
toa_output_22=readToa("C:\\Master\\aa\EODP\\EODP-TS-ISM\\myoutput","ism_toa_VNIR-2.nc")
toa_output_33=readToa("C:\\Master\\aa\EODP\\EODP-TS-ISM\\myoutput","ism_toa_VNIR-3.nc")

mu0, sigma0, threshold0, ok0=check_band(toa_0,toa_output_00, tol=0.01)
mu1, sigma1, threshold1, ok1=check_band(toa_1,toa_output_11, tol=0.01)
mu2, sigma2, threshold2, ok2=check_band(toa_2,toa_output_22, tol=0.01)
mu3, sigma3, threshold3, ok3=check_band(toa_3,toa_output_33, tol=0.01)

print("Criteria 1")
print("Band 0: mean=", mu0, "std=", sigma0, " threshold=", threshold0, "Is it below 0.01%?? ", ok0)
print("Band 1: mean=", mu1, "std=", sigma1, " threshold=", threshold1, "Is it below 0.01%?? ", ok1)
print("Band 2: mean=", mu2, "std=", sigma2, " threshold=", threshold2, "Is it below 0.01%?? ", ok2)
print("Band 3: mean=", mu3, "std=", sigma3, " threshold=", threshold3, "Is it below 0.01%?? ", ok3)

print("Criteria 2")



#ISM-003
print("\n=== ISM-003: Conversion Factors and Units ===")

from ism.src.detectionPhase import detectionPhase

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:\Master\aa\EODP\EODP_Oriol_Espejo\EODP_Oriol_Espejo\auxiliary'
indir = r"C:\Master\aa\EODP\EODP-TS-ISM\input\gradient_alt100_act150" # small scene CLass 3
#indir = r"C:\\Master\\aa\\EODP\\EODP-TS-E2E\\sgm_out"
outdir = r"C:\\Master\\aa\\EODP\\EODP-TS-ISM\output" #Class 3
#outdir = r"C:\\Master\\aa\\EODP\\EODP-TS-E2E\\myismoutputs"

dp = detectionPhase(auxdir, indir, outdir)
config = dp.ismConfig
constants = dp.constants

# Parámetros
area_pix = config.pix_size ** 2   # [m²]
t_int = config.t_int              # [s]
QE = config.QE                    # [e-/photon]
h = constants.h_planck            # [J·s]
c = constants.speed_light         # [m/s]

conv_irrad_to_phot = []

# Irradiance → Photons (por banda)
for wv in config.wv:
    E_photon = (h * c) / wv                  # Energía de un fotón [J]
    Ein = (t_int * area_pix * 1e-3)          # 1 mW/m² → J
    conv = Ein / E_photon                    # [photons / (mW/m²)]
    conv_irrad_to_phot.append(conv)

# Otras conversiones
conv_phot_to_e = QE                          # [e-/photon]
conv_e_to_v = config.OCF               # [V/e-]
conv_v_to_dn = (1/(config.max_voltage - config.min_voltage)) * (2**config.bit_depth - 1)

# Mostrar resultados
print("\nConversion Factors:")
print("Irradiance → Photons =", np.round(conv_irrad_to_phot, 2), "→ Units: Photons [p+]")
print("Photons → Electrons =", conv_phot_to_e, "→ Units: Electrons [e-]")
print("Electrons → Volts =", f"{conv_e_to_v:.3e}", "→ Units: Volts [V]")
print("Volts → Digital =", f"{conv_v_to_dn:.2f}", "→ Units: Digital Numbers [-]")

print("\nUnits at each stage:")
print("After Irradiance → Photons: Photons [p+]")
print("After Photons → Electrons: Electrons [e-]")
print("After Electrons → Volts: Volts [V]")
print("After Volts → Digital: Digital Numbers [-]\n")

print("Criteria 3")

# Rutas de salida de la detección (TOA en e-)


FWC = 420000

# Lista de bandas ya cargadas
toa_bands = [toa_0, toa_1, toa_2, toa_3]
band_names = ['VNIR-0', 'VNIR-1', 'VNIR-2', 'VNIR-3']

saturated_percent = []

# Revisión de saturación considerando los píxeles recortados al FWC
for i, toa in enumerate(toa_bands):
    n_pixels = toa.size
    saturated_pixels = np.sum(toa >= FWC)  # todos los píxeles que llegaron al FWC
    pct_saturated = saturated_pixels / n_pixels * 100
    saturated_percent.append(pct_saturated)
    print(f"Band {band_names[i]}: Saturated pixels = {saturated_pixels}, Percentage = {pct_saturated:.2f}%")
