import matplotlib.pyplot as plt

def projection3D(colour, x, y, z, ex = None, ey = None, ez = None):
    ax = plt.axes(projection='3d')
    ax.plot(x, y, z, colour)
    if ex != ey != ez != None:
        ax.plot(ex, ey, ez, 'blue')
    plt.show()

def projection2D(text, label, colour, x, y, ex = None, ey = None):
    plt.title(text)
    plt.plot(x, y, colour, label = label)
    if ex != ey != None:
        plt.plot(ex, ey, 'blue', label = 'original')
    plt.legend()
    plt.show()

def dependance_on_time(text, label, colour, t, x, ex = None):
    plt.title(text)
    plt.plot(t, x, colour, label = label)
    if ex != None:
        plt.plot(t, ex, 'blue', label = 'original')
    plt.legend()    
    plt.show()
    
def show_interval(text, t, x, ex, y, ey, z, ez):
    projection3D('red', x, y, z, ex, ey, ez)
    
    dependance_on_time('Dependance X(t)', text, 'red', t, x, ex)  
    dependance_on_time('Dependance Y(t)', text, 'red', t, y, ey)
    dependance_on_time('Dependance Z(t)', text, 'red', t, z, ez)

    projection2D('Function projection on the plane XY', text, 'red', x, y, ex, ey)
    projection2D('Function projection on the plane XZ', text, 'red', x, z, ex, ez)
    projection2D('Function projection on the plane YZ', text, 'red', y, z, ey, ez)

def show_dataset(t, x, y, z):
    projection3D('blue', x, y, z)

    dependance_on_time('X(t)', 'dataset', 'blue', t, x)
    dependance_on_time('Y(t)', 'dataset', 'blue', t, y)
    dependance_on_time('Z(t)', 'dataset', 'blue', t, z)

    projection2D('Function projection on the plane XY', 'dataset', 'blue', x, y)
    projection2D('Function projection on the plane XZ', 'dataset', 'blue', x, z)
    projection2D('Function projection on the plane YZ', 'dataset', 'blue', y, z)
