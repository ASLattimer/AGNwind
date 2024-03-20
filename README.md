# AGNwind
Data files and assorted code used in Lattimer &amp; Cranmer 2024, "A Self-Consistent Treatment of the Line-Driving Radiation Force for Active Galactic Nuclei Outflows: New Prescriptions for Simulations."  Requires QSOSED (https://github.com/arnauqb/qsosed).

Note: the python version of QSOSED is no longer maintained by the author (see the Julia version for the most current version); the code provided here requires replacing the sed.py file in QSOSED's pyagn folder with the modified sed.py file provided here.

support_files: provides data files used in the IEHI subroutine used in the calculation of the ionization balance

Initial_Parms: defines initial constants, parameters, and functions

Loop_Parallels: implements parallelization of the line of sight calculation code

qbar_knn_model.sav: K nearest neighbors model for use in estimating the value of M(t)


