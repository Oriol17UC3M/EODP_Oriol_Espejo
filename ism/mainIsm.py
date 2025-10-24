
# MAIN FUNCTION TO CALL THE ISM MODULE

from ism.src.ism import ism

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:\Master\aa\EODP\EODP_Oriol_Espejo\EODP_Oriol_Espejo\auxiliary'
indir = r"C:\Master\aa\EODP\EODP-TS-ISM\input\gradient_alt100_act150" # small scene CLass 3
#indir = r"C:\\Master\\aa\\EODP\\EODP-TS-E2E\\sgm_out"
outdir = r"C:\\Master\\aa\\EODP\\EODP-TS-ISM\myoutput" #Class 3
#outdir = r"C:\\Master\\aa\\EODP\\EODP-TS-E2E\\myismoutputs"

# Initialise the ISM
myIsm = ism(auxdir, indir, outdir)
myIsm.processModule()
