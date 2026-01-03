from imports import *

def plot_mesh(mesh,  title="Mesh", cmap="", colorbar=False, mode="", figsize=(8,6)):
    plt.figure(figsize=figsize)
    if mode=="":
        p=fenics.plot(mesh)
    elif mode=="glyphs":
        p=fenics.plot(mesh, mode=mode)
    plt.title(title)
    if cmap!="": plt.set_cmap(cmap)
    if colorbar: 
        cb = plt.colorbar(p,cmap=plt.get_cmap() if cmap!="" else None, orientation='vertical')
        cb.set_label('Temperature (°C)', labelpad=10)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()
