from __future__ import annotations
import fenics
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import meshio
import gmsh
import subprocess
import argparse
import math
from pathlib import Path
import re
from typing import Optional, Tuple

import gmsh

EXPERIMENTS_JSON_PATH = "experiments.json"
SCHEMA_JSON_PATH = "experiments.schema.json"
TRIG_XDMF_PATH = "data/plume.xdmf"         # cell mesh (Grid)
FACETS_XDMF_PATH = "data/plume_mt.xdmf"    # facet tags (Grid)
OUTPUT_XDMF_PATH_WIRE = "data/base/wire_temperature.xdmf" # output file for wire temperature
OUTPUT_XDMF_PATH_TEMP = "data/base/temperature.xdmf" # output file for wire temperature
OUTPUT_XDMF_PATH_AIR_T = "data/base/air_temperature.xdmf" # output file for wire temperature
OUTPUT_XDMF_PATH_AIR_P = "data/base/air_pressure.xdmf" # output file for wire temperature
OUTPUT_XDMF_PATH_AIR_V = "data/base/air_velocity.xdmf" # output file for wire temperature
OUTPUT_XDMF_PATH_AIR_PVT = "data/base/air_pvt.xdmf" # output file for wire temperature
MESH_NAME = "Grid"                              # XDMF mesh name used when writing
GEOM_FILE = "data/geom.geo"                          # Gmsh geometry file
MSH_FILE = "data/plume.msh"                # Gmsh mesh file
MSH_FILE_C = "data/plume_coarse.msh"                # Gmsh mesh file
ELEM = "triangle"                               # element type

WIRE_TAG = 10                                   # wire's cell tag id
AIR_TAG = 11                                    # air's cell tag id
SYMMETRY_AIR_TAG = 100                          # wire/air interface facet tag id 
OUTER_AIR_TAG = 101                             # wire/air interface facet tag id 
INTERFACE_TAG = 102                             # wire/air interface facet tag id 
PRINT_TAG_SUMMARY = True                        # print summary after reading mesh
UNIFORM_Q = True                                # use uniform heat generation

# Material and model params (wire)
k_wire = 16.0                                   # W/(m·K)  thermal conductivity
h_conv = 15.0                                   # W/(m²·K) effective convection at wire boundary
q_wire = 9.75                                       # W/m³ heat generation

k_air = 0.0257                                  # W/(m·K)  thermal conductivity
rho_air = 1.1614                                # kg/m³   density
q_air = 0.0                                     # W/m³    heat generation
mu_air = 1.85e-5                                # kg/(m·s) dynamic viscosity
cp_air = 1007.0                                 # J/(kg·K) specific heat capacity
beta_air = 3.4e-3                               # 1/K volumetric thermal expansion coefficient

# Heating options
I = 1.0                                         # A
sigma_e = 1.0e6                                 # S/m (example)
D_wire = 0.075e-3                                  # m
T_ambient = 292.95                              # K (19.8°C)
g = 9.81                                    # m/s² gravitational acceleration

# Stabilization / iteration
max_it = 20
rtol = 1e-8

