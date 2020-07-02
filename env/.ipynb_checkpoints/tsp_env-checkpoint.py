import numpy as np
import scipy.spatial as spacial

from math import sqrt

class TspEnv:
    """
    A Travelling Salesman Environment.

    Any environment needs:
    * An initialise (reset) method that returns the initial observations,
        reward, whether state is terminal, additional information.
    * A reset
    * A state space
    * A way to denote possible actions
    * A way to make sure the move is legal
    * A way to affect environment
    * A step function that returns the new observations, reward,
        whether state is terminal, additional information
    * A way to render the environment.

    Methods:
    --------

    __innit__:
        Constructor method.
    is_terminal_state:
        Check whether all cities visited
    render:
        Display state (grid showing which cities visited or unvisited
    reset:
        Initialise environment (including TspState object),
        and return obs, reward, terminal, info
    step:
        Take an action. Update state. Return obs, reward, terminal, info


    Attributes:
    ----------

    action_space:
        Number of cities (integer)
    number_of_cities:
        Number of cities to be visited (integer)
    observation_space:
        Cities visited (NumPy 0/1 array)
    render_game:
        Show game grid
    """

    def __init__(self, number_of_cities = 6, grid_dimensions = (100,100),
                 render_game = False):
        """
        Constructor class for TSP environment
        """

        self.action_space = np.zeros(number_of_cities)
        self.grid_dimensions = grid_dimensions
        self.max_possible_distance = sqrt(grid_dimensions[0]**2 + grid_dimensions[1]**2)
        self.number_of_cities = number_of_cities
        self.observation_space = np.zeros(number_of_cities)
        self.render_game = render_game


    def is_terminal_state(self, action):
        """Check if current state is terminal. All cities complete and agent
        returns to city 0"""

        is_terminal = False
        if (self.state.visited_status.sum() == self.number_of_cities) and (
                action ==0):
            is_terminal = True

        return is_terminal


    def render(self):
        """Show which cities visited and current city"""

        # TODO: REPLACE THIS WITH MATPLOTLIB OUTPUT

        grid = np.zeros(self.grid_dimensions)
        # Add unvisited cities as 1, visited cities as 2
        for city in range(self.number_of_cities):
            city_grid_ref = self.state.city_locations[city]
            if self.state.visited_status[city] == 0:
                grid[city_grid_ref] = 1
            else:
                grid[city_grid_ref] = 2

        # Print
        print (grid)


    def reset(self):
        """Initialise model and return observations"""

        self.state = TspState(self.number_of_cities, self.grid_dimensions)

        # Obs = array of visited cities and on-ehot array of current city
        obs = np.zeros(self.number_of_cities)
        obs[0] = 1
        obs = np.concatenate((self.state.visited_status, obs))
        reward = 0
        is_terminal = self.is_terminal_state(0)

        if self.render_game:
            self.render()

        # return city order chosen as info
        info = self.state.visited_order

        return obs, reward, is_terminal, info


    def step(self, action):
        """Make agent take a step"""

        # ToDo check action is legal (in action space)

        self.state.visited_order.append(action)
        
        # Get reward if new city visited (max reward = max possible distance):
        if self.state.visited_status[action] == 0:
            reward = self.max_possible_distance 
        else:
            reward = 0            
        # Subtract distance travelled from reward
        distance = self.state.distances[self.state.agent_location, action]
        reward -= distance
        # Update agent location is state
        self.state.agent_location = action
        # Update visted_status
        self.state.visited_status[action] = 1
        # Check whether all cities visited and returned home (reward with extra reward)
        terminal = self.is_terminal_state(action)
        if terminal:
            reward += self.max_possible_distance
        # Obs = array of visited cities and on-ehot array of current city
        obs = np.zeros(self.number_of_cities)
        obs[action]= 1
        obs = np.concatenate((self.state.visited_status, obs))
        # return city order chosen as info
        info = self.state.visited_order

        if self.render_game:
            self.render()

        return obs, reward, terminal, info


class TspState:
    """TSP state object.

    Methods:
    --------

    __innit__
        Constructor method.

    Attributes:
    -----------

    city_locations:
        List of city x,y, locations
    distances:
        Dictionary of distance between two cities (index = (from, to))
        Distance (cost) of staying in the same city = 100
    visited_order:
        List of actions (cities visited) by agent. Can contain duplicates
        if agent returned to a a city.
    visited_status:
        Array showing if cities unvisited (0) or visited (1)

    The state is set up with the agent at city 0 (which is marked as
    visited)"""

    def __init__(self, number_of_cities, grid_dimensions):
        """
        Constructor method for TSP state.

        """

        self.agent_location = 0
        self.city_locations = []
        self.distances = dict()
        self.visited_order = [0]
        self.visited_status = np.zeros(number_of_cities)
        # Set city 0 as visited
        self.visited_status[0] = 1

        # Set up cities in grid
        grid_squares = grid_dimensions[0] * grid_dimensions[1]
        np.random.seed(42)
        city_grid_squares = np.random.choice(grid_squares, number_of_cities)
        for city in city_grid_squares:
            x = city % grid_dimensions[0]
            y = city // grid_dimensions[0]
            self.city_locations.append((x,y))

        # Set up distances
        for start in range(number_of_cities):
            for end in range(number_of_cities):
                start_loc = self.city_locations[start]
                end_loc = self.city_locations[end]
                # Set distance (cost) to 100 if start and end same city
                if start == end:
                    distance = 150
                else:
                    distance = spacial.distance.euclidean(start_loc, end_loc)
                self.distances[(start, end)] = distance
            
    
    def calculate_distance(self, route):
        """Calculate total distance for a given route"""
        
        total_distance = 0
        for i in range(len(route)-1):
            distance = self.distances[(route[i], route[i+1])]
            total_distance += distance
            
        return total_distance
        
        
            