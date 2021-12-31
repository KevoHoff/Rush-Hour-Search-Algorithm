class Car:
    def __init__(self, i, j, L, horiz):
        """
        Parameters
        i: int
            Row of the car
        j: int
            Column of the car
        L: int
            Length of the car
        horiz: boolean
            True if the car is horizontal, false
            if the car is vertical
        """
        self.i = i
        self.j = j
        self.L = L
        self.horiz = horiz

class State:
    def __init__(self):
        self.N = 0 # Our cars are on an NxN grid
        self.cars = [] # The first car is the red car
        self.goal = [0, 0] # The state that our red car needs to reach
        self.prev = None # Pointers to previous states (use later)

    def clone(self):
        """
        Make a deep copy of this state

        Return
        ------
        State: Deep copy of this state
        """
        s = State()
        s.N = self.N
        for c in self.cars:
            s.cars.append(Car(c.i, c.j, c.L, c.horiz))
        s.goal = self.goal.copy()
        return s

    def load_puzzle(self, filename):
        """
        Load in a puzzle from a text file
        
        Parameters
        ----------
        filename: string
            Path to puzzle
        """
        fin = open(filename)
        lines = fin.readlines()
        fin.close()
        self.N = int(lines[0])
        self.goal = [int(k) for k in lines[1].split()]
        for line in lines[2::]:
            fields = line.rstrip().split()
            i, j, L = int(fields[0]), int(fields[1]), int(fields[3])
            horiz = True
            if "v" in fields[2]:
                horiz = False
            self.cars.append(Car(i, j, L, horiz))

    def get_state_grid(self):
        """
        Return an NxN 2D list corresponding to this state.  Each
        element has a number corresponding to the car that occupies 
        that cell, or is a -1 if the cell is empty

        Returns
        -------
        list of list: The grid of numbers for the state
        """
        grid = [[-1]*self.N for i in range(self.N)]
        for idx, c in enumerate(self.cars):
            di = 0
            dj = 0
            if c.horiz:
                dj = 1
            else:
                di = 1
            i, j = c.i, c.j
            for k in range(c.L):
                grid[i][j] = idx
                i += di
                j += dj
        return grid
    
    def __str__(self):
        """
        Get a string representing the state

        Returns
        -------
        string: A string representation of this state
        """
        s = ""
        grid = self.get_state_grid()
        for i in range(self.N):
            for j in range(self.N):
                s += "%5s"%grid[i][j]
            s += "\n"
        return s
    
    def get_state_hashable(self):
        """
        Return a shorter string without line breaks that can be
        used to hash the state

        Returns
        -------
        string: A string representation of this state
        """
        s = ""
        grid = self.get_state_grid()
        for i in range(self.N):
            for j in range(self.N):
                s += "{}".format(grid[i][j])
        return s
    
    def get_neighbors(self):
        """
        Return a list of states that could be achieved by a single
        move of a car in the original state
        
        Returns
        -------
        list: A list containing States
        """
        grid = self.get_state_grid()
        states = []
        for i, car in enumerate(self.cars):
            if car.horiz == True:
                if car.j > 0:
                    if grid[car.i][car.j-1] == -1:
                        new_state = self.clone()
                        new_state.cars[i].j = car.j-1
                        states.append(new_state)
                if car.j+car.L < self.N:
                    if grid[car.i][car.j+car.L] == -1:
                        new_state = self.clone()
                        new_state.cars[i].j = car.j+1
                        states.append(new_state)
            else:
                if car.i > 0:
                    if grid[car.i-1][car.j] == -1:
                        new_state = self.clone()
                        new_state.cars[i].i = car.i-1
                        states.append(new_state)
                if car.i+car.L < self.N:
                    if grid[car.i+car.L][car.j] == -1:
                        new_state = self.clone()
                        new_state.cars[i].i = car.i+1
                        states.append(new_state)

        return states   
    
    def __lt__(self, other):
        """
        Overload the less than operator so that ties can
        be broken automatically in a heap without crashing
        Parameters
        ----------
        other: State
            Another state
        
        Returns
        -------
        Result of < on string comparison of __str__ from self
        and other
        """
        return str(self) < str(other)
            
    def plot(self):
        """
        Create a new figure and plot the state of this puzzle,
        coloring the cars by different colors
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.colors import ListedColormap
        c = cm.get_cmap("Paired", len(self.cars))
        colors = [[1, 1, 1, 1], [1, 0, 0, 1]]
        colors = colors + c.colors.tolist()
        cmap = ListedColormap(colors)
        grid = self.get_state_grid()
        grid = np.array(grid)
        plt.imshow(grid, interpolation='none', cmap=cmap)
        
    
    def is_goal(self):
        """
        Idenitifies if the current state is in a solved state
        
        Returns
        -------
        bool: True is the state is solved, false otherwise
        """
        is_at_goal = False
        grid = self.get_state_grid()
        if grid[self.goal[0]][self.goal[1]] == 0:
            is_at_goal = True
        return is_at_goal
    
    def get_heuristic(self):
        """
        Calculates the A* heuristic (blocking heuristic) for a given State.
        
        Returns
        -------
        int: 0 if goal state, 1 is not goal but not cars 
             blocking, 2 otherwise
        """
        h = 0
        if not self.is_goal():
            h = 1
        
            start = self.cars[0]

            diff_loc = self.get_goal_direction(start)

            grid = self.get_state_grid()

            loc = [start.i + (not start.horiz * start.L),
                        start.j + (start.horiz * start.L)]

            while loc[0] < self.N and loc[1] < self.N:
                if grid[loc[0]][loc[1]] != -1:
                    h = 2
                loc[0] += diff_loc[0]
                loc[1] += diff_loc[1]
        
        return h
    
    def get_goal_direction(self, car):
        """
        Finds the direction that a starting car has to
        move in a 2x2 grid to reach the goal state
        
        Returns
        -------
        tuple: size of 2, first element is change in row,
               second element is change in column
        """
        diff_loc = [self.goal[0] - car.i, self.goal[1] - car.j]

        for i in range(len(diff_loc)):
            if diff_loc[i] != 0:
                diff_loc[i] = int(diff_loc[i]/diff_loc[i])
        return diff_loc

    def solve(self):
        """
        BFS Graph Search approach to solving Rush Hour puzzle
        
        Returns
        -------
        list: a collection of States that keeps track of the
              optimal solution's movement path to reach goal
              state
              
        int: expanded number of nodes in queue during search
        
        int: the number of nodes in the optimal solution search
             path
        """
        queue = [self]
        visited = set([])
        steps = []
        reachedGoal = False
        
        expanded = 0
        while not reachedGoal and len(queue) > 0:
            state = queue.pop(0)
            hashable = state.get_state_hashable()
            if state.is_goal():
                expanded += 1
                reachedGoal = True
            elif hashable not in visited:
                expanded += 1
                for n in state.get_neighbors():
                    queue.append(n)
                    n.prev = state
                visited.add(hashable)
        
        current = state
        is_start = False
        condensed = 0
        while not is_start:
            steps.insert(0, current)
            condensed += 1
            if current != self:
                current = current.prev
            else:
                is_start = True
        
        return steps, expanded, condensed
    
    def solve_astar(self):
        """
        BFS A* approach to solving Rush Hour puzzle using the
        blocking heuristic
        
        Returns
        -------
        list: a collection of States that keeps track of the
              optimal solution's movement path to reach goal
              state
              
        int: expanded number of nodes in queue during search
        
        int: the number of nodes in the optimal solution search
             path
        """
        from heapq import heappush, heappop
        
        queue = [(self.get_heuristic(), self.get_state_hashable(), self, 0)]
        visited = set([])
        steps = []
        reachedGoal = False
        
        expanded = 0
        while not reachedGoal and len(queue) > 0:
            (h, hashable, state, cost) = heappop(queue)
            if state.is_goal():
                expanded += 1
                reachedGoal = True
            else:
                expanded += 1
                visited.add(hashable)
                for n in state.get_neighbors():
                    costn = cost + 1 + n.get_heuristic()
                    n_hashable = n.get_state_hashable()
                    n.prev = state
                    q_hash = [el[1] for el in queue]
                    if not n_hashable in q_hash:
                        if not n_hashable in visited: 
                            heappush(queue, (costn, n_hashable, n, cost + 1))
                    else:
                        for q in queue:
                            if hashable in q and h < q[0]:
                                q[0] = h                                      

        current = state
        is_start = False
        condensed = 0
        while not is_start:
            steps.insert(0, current)
            condensed += 1
            if current != self:
                current = current.prev
            else:
                is_start = True
        
        return steps, expanded, condensed
    
    def get_my_heuristic(self):
        """
        Calculates MY A* heuristic (number of blocking + 
        distance heuristic) for a given State.
        
        Returns
        -------
        int: 0 if goal, otherwise the number of cars blocking
        the starting car from finishing plus the distance
        of the start car from the goal state
        """
        h = 0
        if not self.is_goal():
            start = self.cars[0]

            diff_loc = self.get_goal_direction(start)

            grid = self.get_state_grid()

            loc = [start.i + (not start.horiz * start.L),
                        start.j + (start.horiz * start.L)]

            while loc[0] < self.N and loc[1] < self.N:
                if grid[loc[0]][loc[1]] != -1:
                    h += 1
                    car = grid[loc[0]][loc[1]]
                    tp = loc.copy() # top pointer
                    """
                    Note the below commented code makes the search algorithm
                    inconsistent
                    """
                    
#                     while tp[0] > 0 and tp[1] > 0 and grid[tp[0]][tp[1]] == car: 
#                         tp[0] -= diff_loc[1]
#                         tp[1] -= diff_loc[0]
#                         if grid[tp[0]][tp[1]] != car and grid[tp[0]][tp[1]] != -1:
#                             h+=1
                        
                    
#                     bp = loc.copy() # bottom pointer
                    
#                     while bp[0] <= self.N-1 and bp[1] <= self.N-1 and grid[bp[0]][bp[1]] == car:
#                         bp[0] += diff_loc[1]
#                         bp[1] += diff_loc[0]
#                         if grid[bp[0]][bp[1]] != car and grid[bp[0]][bp[1]] != -1:
#                             h+=1
                           
                h += 1
                loc[0] += diff_loc[0]
                loc[1] += diff_loc[1]

        
        return h
    
    def solve_myastar(self):
        """
        BFS MY A* approach to solving Rush Hour puzzle using the
        blocking heuristic
        
        Returns
        -------
        list: a collection of States that keeps track of the
              optimal solution's movement path to reach goal
              state
              
        int: expanded number of nodes in queue during search
        
        int: the number of nodes in the optimal solution search
             path
        """
        from heapq import heappush, heappop
        
        queue = [(self.get_my_heuristic(), self.get_state_hashable(), self, 0)]
        visited = set([])
        steps = []
        reachedGoal = False
        
        expanded = 0
        while not reachedGoal and len(queue) > 0:
            (h, hashable, state, cost) = heappop(queue)
            if state.is_goal():
                expanded += 1
                reachedGoal = True
            else:
                expanded += 1
                visited.add(hashable)
                for n in state.get_neighbors():
                    costn = cost + 1 + n.get_my_heuristic()
                    n_hashable = n.get_state_hashable()
                    n.prev = state
                    q_hash = [el[1] for el in queue]
                    if not n_hashable in q_hash:
                        if not n_hashable in visited: 
                            heappush(queue, (costn, n_hashable, n, cost + 1))
                    else:
                        for q in queue:
                            if hashable in q and h < q[0]:
                                q[0] = h                                      

        current = state
        is_start = False
        condensed = 0
        while not is_start:
            steps.insert(0, current)
            condensed += 1
            if current != self:
                current = current.prev
            else:
                is_start = True
        
        return steps, expanded, condensed
            
            
        
