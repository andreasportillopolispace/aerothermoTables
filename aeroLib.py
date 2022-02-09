import numpy as np
import matplotlib.pyplot as plt

#
def menuScript() :
    print('--- HELP MENU---')
    print('\nCOMMANDS       : ')
    print(' -h              : help command .')
    print(' -i "M"          : compute the Isoentropic Table given the Mach numer.')
    print(' -ip "P/P*"      : compute Mach number given pressur ratio (P/P0), for Isoentropic Flow.')
    print(' -ia "A/A*"      : compute Mach number given area ratio (A/A*) in a Convergent-Divergent nozzle.')
    print(' -f "M"          : compute Fanno Table (with friction, no heat) given Mach number M.')
    print(' -fg "g(M)"      : compute Mach number given Fanno parameter g(M)=4fL/d, for a Fanno flow.')
    print(' -r "M"          : compute Rayleigh Table (Heat Addition, friction-less) given Mach number M. ')
    print(' -rt "b(M)"      : compute Rayleigh Table given T0/T0*, for a Rayleigh flow . ')
    print(' -n "M1"         : compute Normal Shocks Table given inlet Mach numberYes  M1')
    print(' -p "M"          : compute Prandtl-Meyer Function and Mach Angle  given Mach number')
    print(' -o "M , theta"  : compute the Oblique Shocks deflection angle beta, given Mach number and deflector angle Theta (deg) ')
    print('\nSETTINGS   :')
    print(' -g "gam"    : set a value for gamma, default is gamma = 1.4 (dry air)')
    print(' -show "1/0" : show plots (useful for debugging), default is 0.')
    print('---\n')

#
def isoSairTab(M, gam, p) :
    delta=0.5*(gam-1)

    Tratio=(1+delta*M**2)**-1 # temperature ratio
    Pratio=Tratio**(gam/(gam-1)) # pressure ratio
    RhoRatio=Tratio**(1/(gam-1)) # density ratio
    if M!=0 :
        areaRatio=(1/M)*((2)/(gam+1)*(1 + delta*M**2))**((gam+1)/(2*(gam-1)))#Area Ratio
    else : 
        areaRatio="Inf"
    aRatio=Tratio**(0.5) # soundspeed ratio

    print("--- ISO-S TABLE, gam = ", gam, " ---")
    print("M = ", M)
    print("T/T0 =", Tratio )
    print("P/P0 =", Pratio )
    print("rho/rho0 =", RhoRatio )
    print("A/A* =", areaRatio )
    print("a/a0 =", aRatio )
    return 0

#
def fannoTab(M, gam, p) :
    delta=0.5*(gam-1)
    g= fannoFunc(M, gam)
    Tratio=(0.5*(gam+1))*(1+delta*M**2)**-1 # temperature ratio
    Pratio=Tratio**(0.5) / M # static pressure ratio
    RhoRatio=Tratio**(-0.5) / M # density ratio
    P0ratio=((1+delta*M**2)/ (0.5*(gam+1)))**((gam+1)/(2*(gam-1))) / M # total pressure ratio
    
    print("--- FANNO TABLE, gam = ", gam, " ---")
    print("M = ", M)
    print("g(M) =", g )
    print("T/T* =", Tratio )
    print("P/P* =", Pratio )
    print("rho/rho* =", RhoRatio )
    print("P0/P0* =", P0ratio )

    #PLOT
    if p :
        mach = np.arange(0.02, 3., 0.01)
        plt.plot(mach, fannoFunc(mach, gam), 'r:'), 
        plt.grid(),
        plt.plot(M, fannoFunc(M, gam), 'bo')
        plt.legend(["Fanno Line", "M"])
        # plt.show()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")
    return 0

# 
def rayleighTab(M, gam, p) :
    delta=0.5*(gam-1)
    T0ratio= rayleighFunc(M, gam)
    Pratio= (gam+1)/(1+gam*M**2) # static pressure ratio
    P0ratio= Pratio*(2*((1+delta*M**2))/(gam+1))**(gam/(gam-1)) # total pressure ratio
    Tratio= M**2 * Pratio**2 # static temperature ratio
    RhoRatio= ((1+gam*M**2)/((gam+1)*M**2))# density ratio

    
    print("--- RAYLEIGH TABLE, gam = ", gam, " ---")
    print("M = ", M)
    print("T0/T0* =", T0ratio )
    print("P0/P0* =", P0ratio )
    print("rho/rho* =", RhoRatio )
    print("T/T* =", Tratio )
    print("P/P* =", Pratio )
    return 0

#
def normalShockTab(M1, gam, p) :
    delta=0.5*(gam-1)
    T0ratio= 1
    P0ratio= ((M1**2*(gam+1))**((gam+1)/gam))/(2*(1+delta*M1**2)*(2*gam*M1**2 + 1 - gam)**(1/gam)) # static pressure ratio
    Pratio=(2*gam*M1**2 + 1 - gam)/(gam+1) # total pressure ratio
    RhoRatio= ((gam+1)*M1**2)/(2*(1+delta*M1**2))# density ratio
    uRatio= RhoRatio**-1
    Tratio= Pratio/RhoRatio  # static temperature ratio
    M2 = ((2*(1+delta*M1**2))/(2*gam*M1**2 + 1 - gam))**0.5
    cRatio = Tratio**0.5
    aRatioCrit = P0ratio**-1

    print("--- NORMAL SHOCK TABLE, gam = ", gam, " ---")
    print("M1 = ", M1)
    print("M2 = ", M2)
    print("rho2/rho1 =", RhoRatio )
    print("u2/u1 =", uRatio )
    print("T2/T1 =", Tratio )
    print("P2/P1 =", Pratio )
    print("T02/T01 =", T0ratio )
    print("P02/P01 =", P0ratio )
    print("A2*/A1* =", aRatioCrit )
    print("c2/c1 =", cRatio )
    return 0
#
def prandtlMeyerTab(M, gam, p):
    nu=PMfunc(M, gam)
    mu=np.arcsin(M**-1)
    print("--- PRANDTL-MEYER FUNC AND MACH ANGLE TABLE, gam = ", gam, " ---")
    print("M = ", M)
    print("(rad) nu = ", nu )
    print("(deg) nu = ", nu*180/np.pi )
    print("(rad) mu = ", mu )
    print("(deg) mu = ", mu*180/np.pi )

        #PLOT
    if p :
        mach = np.arange(1.0, 10., 0.01)
        plt.plot(mach, PMfunc(mach, gam), 'r'), 
        plt.grid(),
        plt.legend(["nu(M)"])
        # plt.show()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")
    return 0

#
def obliqueShockTab(M, theta, gam, p) :
    beta0_w=0.0001
    beta0_s=89*np.pi/180
    theta_rad=theta*np.pi/180
    param = [M, theta_rad, gam] 
    func = lambda beta, param : obliqueShockFunc(param[0], param[1], beta, param[2])
    beta_w=fzeroNR(func, beta0_w, param )
    beta_s=fzeroNR(func, beta0_s, param )

    # Find max value of the function
    db=0.0001
    param2 = [M, 0, gam] 
    func_theta_max = lambda beta, param : (func(beta+db, param) - func(beta-db, param) )/ db/2
    beta_theta_max=abs(fzeroNR(func_theta_max, beta0_s, param2 )) # minus sign introduced because of the way oliqueShockFunc is defined
    theta_max=abs(np.arctan(func(beta_theta_max, param2 ))) # minus sign introduced because of the way oliqueShockFunc is defined

    print("--- OBLIQUE SHOCK TABLE, gam = ", gam, " ---")
    if theta_rad <= theta_max :
        print("M = ", M)
        print("[deg] theta = ", theta)    
        print("[deg] theta_max = ", theta_max*180/np.pi)    
        print("[deg] beta_max = ", beta_theta_max*180/np.pi)
        print("[deg] beta_weak = ", beta_w*180/np.pi )    
        print("[deg] beta_strong = ", beta_s*180/np.pi )
        print("")
        print("[rad] theta = ", theta_rad)
        print("[rad] theta_max = ", theta_max)
        print("[rad] beta_max = ", beta_theta_max)
        print("[rad] beta_weak = ", beta_w )
        print("[rad] beta_strong= ", beta_s )
        if p :
            beta_min=fzeroNR(func, beta0_w, param2 )
            b = np.arange(beta_min, np.pi/2, 0.001)
            plt.plot(b*180/np.pi, -np.arctan(func(b, param2))*180/np.pi, 'r'), 
            plt.plot(beta_theta_max*180/np.pi, -np.arctan(func(beta_theta_max, param2))*180/np.pi, 'ro')
            plt.plot(beta_w*180/np.pi, -np.arctan(func(beta_w, param2))*180/np.pi, 'go')
            plt.plot(beta_s*180/np.pi, -np.arctan(func(beta_s, param2))*180/np.pi, 'bo')
            plt.grid(),
            plt.legend(["theta(beta)","MAX", "Weak Solution", "Strong Solution"])
            plt.xlabel("beta")
            plt.ylabel("theta")
            # plt.show()
            plt.draw()
            plt.pause(0.001)
            input("Press [enter] to continue.")

    else :
        print("/!\ NO SOLUTION for this Mach number and Theta value !")
        print("M = ", M)
        print("[deg] theta = ", theta)    
        print("[deg] theta_max = ", theta_max*180/np.pi)  
        print("[deg] beta_max = ", beta_theta_max*180/np.pi)  
        print("")
        print("[rad] theta = ", theta_rad)
        print("[rad] theta_max = ", theta_max)
        print("[rad] beta_max = ", beta_theta_max)
        if p :
            beta_min=fzeroNR(func, beta0_w, param2 )
            b = np.arange(beta_min, np.pi/2, 0.001)
            plt.plot(b*180/np.pi, -np.arctan(func(b, param2))*180/np.pi, 'r'), 
            plt.plot(beta_theta_max*180/np.pi, -np.arctan(func(beta_theta_max, param2))*180/np.pi, 'ro')
            plt.grid(),
            plt.legend(["theta(beta)","MAX"])
            plt.xlabel("beta")
            plt.ylabel("theta")
            # plt.show()
            plt.draw()
            plt.pause(0.001)
            input("Press [enter] to continue.")
    return 0

def obliqueShockFunc(M, theta, beta, gam) :

    func=  np.tan(theta) - 2*( (M*np.sin(beta))**2 -1)/( np.tan(beta) * ((gam + np.cos(2*beta))*M**2 + 2) )

    return func

#
def pressureRatioScript(Pratio, gam, p) :
    param=[Pratio,gam]
    #PLOT
    if p :
        mach = np.arange(0.2, 10., 0.01)
        plt.plot(mach, pressureRatioFunc(mach, param), 'r:'), 
        plt.grid(),
        plt.legend(["P/P0"])
        # plt.show()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")

    M=fzeroNR(pressureRatioFunc, 1, param )
    isoSairTab(M, gam, p)  
    return 0
#
def pressureRatioFunc(M, param) :
    Pratio=param[0]
    gam=param[1]
    delta= 0.5*(gam-1)
    func= Pratio - (1+delta*M**2)**(gam/(1-gam)) # pressure ratio
    return func

#
def areaRatioScript(areaRatio, gam, p) :
    if areaRatio<1:
        print('A/A* must be greater than 1 .')
        exit()

    param=[areaRatio,gam]
    M_sub=fzeroNR(areaRatioFunc, 0.1, param )
    M_sup=fzeroNR(areaRatioFunc, 1.1, param )
    print('\nM(A/A*)_subsonic = ', M_sub )
    isoSairTab(M_sub, gam, p)
    print('\nM(A/A*)_supersonic = ', M_sup )
    isoSairTab(M_sup, gam, p)

    #PLOT
    if p :  
        mach = np.arange(0.2, 3., 0.01)
        plt.plot(mach, areaRatioFunc(mach, param), 'r:'), 
        plt.grid(),
        plt.plot(M_sub, areaRatioFunc(M_sub, param), 'bo')
        plt.plot(M_sup, areaRatioFunc(M_sup, param), 'go')
        plt.legend(["A/A*", "M_sub", "M_sup"])
        # plt.show()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")

    return 0

# 
def areaRatioFunc(M, param):
    areaRatio=float(param[0])
    gam=float(param[1])
    delta=0.5*(gam-1)
    func= (1/M)*((2)/(gam+1)*(1 + delta*M**2))**((gam+1)/(2*(gam-1))) - areaRatio#Area Ratio
    return func

#

def fannoScriptInv(g, gam, p) :   
    param=[g, gam]

    M_sub=fzeroNR(fannoFuncInv, 0.1, param )
    M_sup=fzeroNR(fannoFuncInv, 1.1, param )
    print('\nM(g)_subsonic = ', M_sub )
    fannoTab(M_sub, gam, p)
    print('\nM(g)_supersonic = ', M_sup )
    fannoTab(M_sup, gam, p)

    #PLOT
    if p :
        mach = np.arange(0.02, 8., 0.01)
        plt.plot(mach, fannoFuncInv(mach, param), 'r:'), plt.grid(),
        plt.plot(M_sub, fannoFuncInv(M_sub, param), 'bo')
        plt.plot(M_sup, fannoFuncInv(M_sup, param), 'go')
        plt.legend(["Fanno", "M_sub", "M_sup"])
        # plt.show()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")
    return 0

#
def rayleighScriptInv(b, gam, p) :
    param=[b, gam]

    M_sub=fzeroNR(rayleighFuncInv, 0.2, param )
    M_sup=fzeroNR(rayleighFuncInv, 1.1, param )
    print('\nM(g)_subsonic = ', M_sub )
    rayleighTab(M_sub, gam, p)
    print('\nM(g)_supersonic = ', M_sup )
    rayleighTab(M_sup, gam, p)

    #PLOT
    if p :
        mach = np.arange(0.02, 8., 0.01)
        plt.plot(mach, rayleighFuncInv(mach, param), 'r:'), plt.grid(),
        plt.plot(M_sub, rayleighFuncInv(M_sub, param), 'bo')
        plt.plot(M_sup, rayleighFuncInv(M_sup, param), 'go')
        plt.legend(["Rayleigh", "M_sub", "M_sup"])
        # plt.show()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")
        return 0

# 
def fannoFunc(M, gam):
    delta=0.5*(gam-1)
    func= (1-M**2)/(gam*M**2) + (gam+1)/(2*gam)*np.log((M**2 * (gam+1))/(2*(1 + delta*M**2))) # Fanno function
    return func

# 
def fannoFuncInv(M, param):
    g=float(param[0])
    gam=float(param[1])
    func= g - fannoFunc(M, gam) # Fanno function
    return func

#
def rayleighFunc(M, gam) :
    delta=0.5*(gam-1)
    b=(2*M**2 *(gam+1)*(1+delta*M**2) )/((1+gam*M**2)**2) # total temperature ratio
    return b
#
def rayleighFuncInv(M, param) :
    b=param[0]
    gam=param[1]
    delta=0.5*(gam-1)
    func=b-(2*M**2 *(gam+1)*(1+delta*M**2) )/((1+gam*M**2)**2) # total temperature ratio
    return func
#
def PMfunc(M, gam): #rad
    func =((gam+1)/(gam-1))**0.5  * np.arctan( ((gam-1)*(M**2-1)/(gam+1))**0.5) -np.arctan( ( M**2 -1 )**0.5)  
    return func

#
def fzeroNR(f, x0, param):
    tol=1e-12
    nMax=1000
    eps=1
    n=0
    xOld=x0
    dx=0.000001

    while n<nMax and eps>tol :
        x=xOld - (f(xOld, param)*dx)/(f(xOld+dx, param) - f(xOld, param))       
        eps =abs((x - xOld)/xOld)
        xOld=x
        n+=1
    #print('n: ', n)
    if n>=nMax:
        print('fzeroNR: Max n of iteration reached.')

    return x



def extract_numbers(s):
    # SOURCE :  Warren Weckesset → https://stackoverflow.com/users/1217358/warren-weckesser
    # This function is utilized to avoid regex library import.
    """
    Extract numbers from a string.

    Examples
    --------
    >>> extract_numbers("Hello4.2this.is random 24 text42")
    [4.2, 24, 42]

    >>> extract_numbers("2.3+45-99")
    [2.3, 45, -99]

    >>> extract_numbers("Avogadro's number, 6.022e23, is greater than 1 million.")
    [6.022e+23, 1]
    """
    numbers = []
    current = ''
    for c in s.lower() + '!':
        if (c.isdigit() or
            (c == 'e' and ('e' not in current) and (current not in ['', '.', '-', '-.'])) or
            (c == '.' and ('e' not in current) and ('.' not in current)) or
            (c == '+' and current.endswith('e')) or
            (c == '-' and ((current == '') or current.endswith('e')))):
            current += c
        else:
            if current not in ['', '.', '-', '-.']:
                if current.endswith('e'):
                    current = current[:-1]
                elif current.endswith('e-') or current.endswith('e+'):
                    current = current[:-2]
                numbers.append(current)
            if c == '.' or c == '-':
                current = c
            else:
                current = ''

    # Convert from strings to actual python numbers.
    numbers = [float(t) if ('.' in t or 'e' in t) else int(t) for t in numbers]

    return numbers
