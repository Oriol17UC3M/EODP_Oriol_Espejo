#plot l1b vs
import numpy as np
from scipy.constants import sigma

from common.io.writeToa import writeToa, readToa

#raw data
toa_0=readToa("C:\\Master\\aa\\EODP\\EODP-TS-ISM\\output","ism_toa_isrf_VNIR-0.nc")
toa_1=readToa("C:\\Master\\aa\EODP\EODP-TS-ISM\\output","ism_toa_isrf_VNIR-1.nc")
toa_2=readToa("C:\\Master\\aa\EODP\\EODP-TS-ISM\\output","ism_toa_isrf_VNIR-2.nc")
toa_3=readToa("C:\\Master\\aa\EODP\\EODP-TS-ISM\\output","ism_toa_isrf_VNIR-3.nc")

#outputs
toa_output_0=readToa("C:\\Master\\aa\\EODP\\EODP-TS-L1B\\output","l1b_toa_VNIR-0.nc")
toa_output_1=readToa("C:\\Master\\aa\\EODP\\EODP-TS-L1B\\output","l1b_toa_VNIR-1.nc")
toa_output_2=readToa("C:\\Master\\aa\\EODP\\EODP-TS-L1B\\output","l1b_toa_VNIR-2.nc")
toa_output_3=readToa("C:\\Master\\aa\\EODP\\EODP-TS-L1B\\output","l1b_toa_VNIR-3.nc")

#My results
toa_lib_0=readToa("C:\\Master\\aa\\EODP\\EODP-TS-L1B\\results","l1b_toa_VNIR-0.nc")
toa_lib_1=readToa("C:\\Master\\aa\\EODP\\EODP-TS-L1B\\results","l1b_toa_VNIR-1.nc")
toa_lib_2=readToa("C:\\Master\\aa\\EODP\\EODP-TS-L1B\\results","l1b_toa_VNIR-2.nc")
toa_lib_3=readToa("C:\\Master\\aa\\EODP\\EODP-TS-L1B\\results","l1b_toa_VNIR-3.nc")

#My results not equalized
toa_lib_0_noeq=readToa("C:\\Master\\aa\\EODP\\EODP-TS-L1B\\results_noequalization","l1b_toa_VNIR-0.nc")
toa_lib_1_noeq=readToa("C:\\Master\\aa\\EODP\\EODP-TS-L1B\\results_noequalization","l1b_toa_VNIR-1.nc")
toa_lib_2_noeq=readToa("C:\\Master\\aa\\EODP\\EODP-TS-L1B\\results_noequalization","l1b_toa_VNIR-2.nc")
toa_lib_3_noeq=readToa("C:\\Master\\aa\\EODP\\EODP-TS-L1B\\results_noequalization","l1b_toa_VNIR-3.nc")

#Exercise 1
def check_band(toa_my, toa_ref, tol=0.01):
    dif= np.abs(toa_ref-toa_my) / np.maximum(toa_ref,1e-12) * 100
    mu = np.mean(dif)
    sigma=np.std(dif)
    threshold = mu + 3*sigma
    ok = threshold < tol
    return mu, sigma, threshold, ok
mu0, sigma0, threshold0, ok0=check_band(toa_lib_0,toa_output_0, tol=0.01)
mu1, sigma1, threshold1, ok1=check_band(toa_lib_0,toa_output_0, tol=0.01)
mu2, sigma2, threshold2, ok2=check_band(toa_lib_0,toa_output_0, tol=0.01)
mu3, sigma3, threshold3, ok3=check_band(toa_lib_0,toa_output_0, tol=0.01)
print("Band 0: mean=", mu0, "std=", sigma0, " threshold=", threshold0, "Is it below 0.01%?? ", ok0)
print("Band 1: mean=", mu1, "std=", sigma1, " threshold=", threshold1, "Is it below 0.01%?? ", ok1)
print("Band 2: mean=", mu2, "std=", sigma2, " threshold=", threshold2, "Is it below 0.01%?? ", ok2)
print("Band 3: mean=", mu3, "std=", sigma3, " threshold=", threshold3, "Is it below 0.01%?? ", ok3)

#Exercise 2
import matplotlib.pyplot as plt
plt.plot(toa_0[0,:],label='isrf')
plt.plot(toa_lib_0[0,:], label='myresults')
plt.title("Raw data vs processed data")
plt.show()
plt.plot(toa_1[0,:],label='isrf')
plt.plot(toa_lib_1[0,:], label='myresults')
plt.title("Raw data vs processed data")
plt.show()
plt.plot(toa_2[0,:],label='isrf')
plt.plot(toa_lib_2[0,:], label='myresults')
plt.title("Raw data vs processed data")
plt.show()
plt.plot(toa_3[0,:],label='isrf')
plt.plot(toa_lib_3[0,:], label='myresults')
plt.title("Raw data vs processed data")
plt.show()

#Exercise 3
plt.plot(toa_lib_0_noeq[0,:],label='myresults not equalized')
plt.plot(toa_lib_0[0,:], label='myresults')
plt.title("Equalized output vs not equalized output")
plt.show()
plt.plot(toa_lib_1_noeq[0,:],label='myresults not equalized')
plt.plot(toa_lib_1[0,:], label='myresults')
plt.title("Equalized output vs not equalized output")
plt.show()
plt.plot(toa_lib_2_noeq[0,:],label='myresults not equalized')
plt.plot(toa_lib_2[0,:], label='myresults')
plt.title("Equalized output vs not equalized output")
plt.show()
plt.plot(toa_lib_3_noeq[0,:],label='myresults not equalized')
plt.plot(toa_lib_3[0,:], label='myresults')
plt.title("Equalized output vs not equalized output")
plt.show()