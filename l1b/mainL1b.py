
# MAIN FUNCTION TO CALL THE L1B MODULE

from l1b.src.l1b import l1b

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:\Master\aa\EODP\EODP_Oriol_Espejo\EODP_Oriol_Espejo\auxiliary'
indir = r"C:\Master\aa\EODP\EODP-TS-L1B\input"
outdir = r"C:\\Master\\aa\\EODP\\EODP-TS-L1B\\results"

# Initialise the ISM
myL1b = l1b(auxdir, indir, outdir)
myL1b.processModule()
