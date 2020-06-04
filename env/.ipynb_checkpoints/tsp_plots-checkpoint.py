import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches



def plot_result_progress_cross_entropy(batch, average, best):
    """Line plot of average and best rewards over time"""

    plt.plot(batch, average, label='Average batch reward')
    plt.plot(batch, best, label='Best batch reward')
    plt.xlabel('Run')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
    
    
def plot_result_progress(distance):
    """Line plot of average and best rewards over time"""
    
    plt.plot(distance)
    plt.xlabel('Run')
    plt.ylabel('Reward')
    plt.show()
    
    
def plot_route(route, route_co_ords):
    """Plot points and best route found between points"""
    
    # Separate x and y co-ordinates
    xCo = [val[0] for val in route_co_ords]
    yCo = [val[1] for val in route_co_ords]
    
  
    # Create figure
    fig = plt.figure(figsize=(8, 5))

   # Plot points to vist
    ax1 = fig.add_subplot(121)
    ax1.scatter(xCo, yCo)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    
    texts = []
    for i, txt in enumerate(route):
        texts.append(ax1.text(xCo[i] + 1, yCo[i] + 1, txt))

    # Plot best route found between points
    verts = [None] * int(len(route) + 1)
    codes = [None] * int(len(route) + 1)
    for i in range(0, len(route)):
        verts[i] = xCo[i], yCo[i]
        if i == 0:
            codes[i] = Path.MOVETO
        else:
            codes[i] = Path.LINETO
    verts[len(route)] = xCo[route[0]], yCo[route[0]]
    codes[len(route)] = Path.CLOSEPOLY

    path = Path(verts, codes)

    ax2 = fig.add_subplot(122)
    patch = patches.PathPatch(path, facecolor='none', lw=0)
    ax2.add_patch(patch)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)

    # give the points a label
    xs, ys = zip(*verts)
    ax2.plot(xs, ys, 'x--', lw=2, color='black', ms=10)

    # Display plot    
    plt.tight_layout(pad=4)
    plt.show()
    return