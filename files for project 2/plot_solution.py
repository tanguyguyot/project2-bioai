import matplotlib.pyplot as plt
import json

# get routes and coordinates
with open('testplot9.json', 'r') as file:
    data = json.load(file)

solution = data["routes"]
coordinates = data["coordinates"]

def visualize_patients(coordinates):
    coords_x = [coord["x"] for coord in coordinates.values()]
    coords_y = [coord["y"] for coord in coordinates.values()]
    plt.plot(coords_x, coords_y, 'ro')
    plt.show()
    return 

def plot_solution(solution, coordinates):
    for (nurse, route) in solution.items():
        # x : coordinates[route[i]][0]
        # y : coordinates[route[i]][1]
        # lists of x and y coordinates
        x = [coordinates[str(route[i])]["x"] for i in range(len(route))]
        y = [coordinates[str(route[i])]["y"] for i in range(len(route))]
        plt.plot(x, y, marker='o', label='Nurse ' + str(nurse))
    plt.legend()
    plt.show()
    return

plot_solution(solution, coordinates)
print("end")