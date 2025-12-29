import  CoolProp.CoolProp as CP
import pandas as pd
import numpy as np

fluid = "Air"
temps = np.arange(290, 641, 5)

data = []
for T in temps:
    rho = CP.PropsSI("D", "T", T, "P", 101325, fluid)
    cp  = CP.PropsSI("Cpmass", "T", T, "P", 101325, fluid)
    k   = CP.PropsSI("L", "T", T, "P", 101325, fluid)
    mu  = CP.PropsSI("V", "T", T, "P", 101325, fluid)

    nu = mu / rho
    alpha = k / (rho * cp)

    data.append([T, rho, cp, k, mu, nu, alpha])

df = pd.DataFrame(data, columns=["T [K]", "Density [kg/m3]", "cp [J/kgK]",
                                 "k [W/mK]", "mu [Pa s]", "nu [m2/s]", "alpha [m2/s]"])
df.to_csv("air_properties_coolprop.csv", index=False)
