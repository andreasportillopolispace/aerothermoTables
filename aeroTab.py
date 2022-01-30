import sys
from aeroLib import *
nInput=len(sys.argv)-1
print('Number of Inputs : ', nInput )

if nInput >= 2:
    comm=[]
    value=[]
 
    for i in range(1, nInput+1, 2) : # extract commands
        comm.append(sys.argv[i] )  
   
    for i in range(2, nInput+1,2) : #extract values associated to commands
        value.append(sys.argv[i])
        
    if len(comm)!=len(value): 
        print('Mismatch between command and value inputs.') # check if correspondance between value and command. Exception : -h dont need value.

    # Settings:
    # Default
    gam=1.4
    p=0
    for i in range(0, len(comm), 1) : # Acquire Settings
        if comm[i]== '-g' :
            gam=float(extract_numbers(value[i])[0])
        if comm[i]== '-show' :
            p=float(extract_numbers(value[i])[0])
        if comm[i]== '-h' :
            menuScript()
            ret=1
            break
         
    for i in range(0, len(comm), 1) : # Acquire Commands
        if comm[i]== '-ia' : 
            areaRatio=float(extract_numbers(value[i])[0])
            # areaRatio=float(value[i])
            ret= areaRatioScript(areaRatio, gam, p)
        elif comm[i]== '-i' :
            M = float(value[i])
            ret= isoSairTab(M, gam, p)
        elif comm[i]== '-ip' : 
            Pratio = float(extract_numbers(value[i])[0])
            ret= pressureRatioScript(Pratio, gam, p)
        elif comm[i]== '-f' : 
            M = float(extract_numbers(value[i])[0])
            ret= fannoTab(M, gam, p)
        elif comm[i]== '-fg' : 
            g = float(extract_numbers(value[i])[0])
            ret= fannoScriptInv(g, gam, p)
        elif comm[i]== '-r' : 
            g = float(extract_numbers(value[i])[0])
            ret= rayleighTab(g, gam, p)
        elif comm[i]== '-rt' : 
            b = float(extract_numbers(value[i])[0])
            ret= rayleighScriptInv(b, gam, p) 
        elif comm[i]== '-n' : 
            M1 = float(extract_numbers(value[i])[0])
            ret= normalShockTab(M1, gam, p) 
        elif comm[i]== '-p' : 
            M = float(extract_numbers(value[i])[0])
            ret= prandtlMeyerTab(M, gam, p) 
        elif comm[i]== '-o' : 
            obliqueValues=extract_numbers(value[i])
            M = float(obliqueValues[0])
            theta = float(obliqueValues[1])
            ret= obliqueShockTab(M, theta, gam, p) 
        elif comm[i]== '-g' :
            ret=0 #treated in settings section
        elif comm[i]== '-show' :
            ret=0 #treated in settings section
        elif comm[i]== '-h' :
            ret=0 #treated in settings section
        else :
            print('COMMAND NOT RECOGNIZED')
            ret=1
            
elif nInput == 1 :
    if sys.argv[1]=='-h' : # Help Menu
        menuScript()
        ret=0
    else :
        print('/!\ You need to include a command : type ''-h'' for help.')
        ret=1
else:
    print('/!\ You need to include a command, type ''-h'' for help.')
    ret=1

if ret==0 :
    print('\nDONE. Exiting.')
    print('\n')
else : 
    print('Return not 0 !')
    print('\n')

exit()
