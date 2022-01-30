# aerothermoTables
Aerothermodynamic tables developed to ease the exercise solution from the Aerothermodynamic course in Politecnico di Milano.
The source code is based on Python3 scripts and has been runned on PowerShell. 
The main is "aeroTab.py", while "aeroLib.py" contains all the functions.
To make it work you need to call "aeroTab.py" followed by the commands.
In the following you can find the instractions and some examples.

You can run on PowerShell by following these steps:
- STEP 1: 
  - Open Powershell
- STEP 2: 
  - Select the directory in which you have saved the scripts: 
  - i.e. type :
    PS: cd "<PATH>"
- STEP 3:
  - Study the command menu:
    PS : python .\aeroTab.py -h
 - STEP 4: 
  - Perform some calculations:
    PS : python .\aeroTab.py -o "2.4 , 18"  -show 1

Available commands:

COMMANDS       :
 -h              : help command .
 -i "M"          : compute the Isoentropic Table given the Mach numer.
 -ip "P/P*"      : compute Mach number given pressur ratio (P/P0), for Isoentropic Flow.
 -ia "A/A*"      : compute Mach number given area ratio (A/A*) in a Convergent-Divergent nozzle.
 -f "M"          : compute Fanno Table (with friction, no heat) given Mach number M.
 -fg "g(M)"      : compute Mach number given Fanno parameter g(M)=4fL/d, for a Fanno flow.
 -r "M"          : compute Rayleigh Table (Heat Addition, friction-less) given Mach number M.
 -rt "b(M)"      : compute Rayleigh Table given T0/T0*, for a Rayleigh flow .
 -n "M1"         : compute Normal Shocks Table given inlet Mach numberYes  M1
 -p "M"          : compute Prandtl-Meyer Function and Mach Angle  given Mach number
 -o "M , theta"  : compute the Oblique Shocks deflection angle beta, given Mach number and deflector angle Theta (deg)

SETTINGS   :
 -g "gam"    : set a value for gamma, default is gamma = 1.4 (dry air)
 -show "1/0" : show plots (useful for debugging), default is 0.
 
 Example of usage: 
 1. Compute the Isoentropic Table given the Mach Number:
    PS : python .\aeroTab.py -i 1.3
    
    Expected Output: 
          Number of Inputs :  2
          --- ISO-S TABLE, gam =  1.4  ---
          M =  1.3
          T/T0 = 0.7473841554559043
          P/P0 = 0.360913895409136
          rho/rho0 = 0.4829027920574239
          A/A* = 1.0663045192307699
          a/a0 = 0.864513826064051

          DONE. Exiting.
  
 2. Compute the Normal Shock Table given M1:
    PS : python .\aeroTab.py -n "1.598" 
  
    Expected Output: 
        Number of Inputs :  2
        --- NORMAL SHOCK TABLE, gam =  1.4  ---
        M1 =  1.598
        M2 =  0.6690403052154059
        rho2/rho1 = 2.0283859201514933
        u2/u1 = 0.4930028305093507
        T2/T1 = 1.3865891949151081
        P2/P1 = 2.812538
        T02/T01 = 1
        P02/P01 = 1.8931716001150318
        A2*/A1* = 0.5282141354429987
        c2/c1 = 1.1775352202440095

        DONE. Exiting.
 
