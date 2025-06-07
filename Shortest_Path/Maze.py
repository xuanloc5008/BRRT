import time
import random
from env import *

DELAY = 0.01
class Maze:
    def __init__(self, board: Board):
        self.board = board
        self.passages = set()  #set of paths
        self.start = self.board.start  
        self.goal = self.board.target  
        self.walls = set()  # set of walls

    def initialize(self):
        """
        Create following information for solver:
        1. board.wall: set all cells as wall except for start and target.
        2. passages: set start and target as passages to record maze path.
        3. board.frontiers: add frontiers of start and target to board.frontiers.
        """
        # set every cell to be wall first
        self.board.wall = {
            (i, j) for i in range(self.board.v_cells)
                   for j in range(self.board.h_cells)
        }

        # Add start and target to passages
        self.passages.add(self.start)
        self.passages.add(self.goal)

        # Remove start and target from walls
        self.board.wall = self.board.wall.difference(self.passages)

        # Initialize frontiers
        self.board.frontiers = self.get_frontiers(self.start)

    def get_frontiers(self, state: tuple) -> set:
        """
        Return frontiers of a state. A frontier cell is a wall
        with distance 2 (in straight) from the given state.
        """
        x, y = state
        frontiers = {(x - 2, y), (x + 2, y), (x, y - 2), (x, y + 2)}
        temp_frontiers = frontiers.copy()
        for row, col in temp_frontiers:
            if (row < 0 or row >= self.board.v_cells) or (col < 0 or col >= self.board.h_cells) or \
               (row, col) in self.passages or (row, col) not in self.board.wall:
                frontiers.remove((row, col))

        return frontiers

    def frontier_neighbor(self, frontier: tuple) -> tuple:
        """
        Randomly pick a cell which is in distance 2 to cells in passages
        from the chosen frontier.
        """
        t = int(time.time())
        random.seed(t)

        x, y = frontier
        neighbors = {(x - 2, y), (x + 2, y), (x, y - 2), (x, y + 2)}

        temp_neighbors = neighbors.copy()
        for cell in temp_neighbors:
            if cell not in self.passages:
                neighbors.remove(cell)

        neighbor = random.choice(list(neighbors))
        return neighbor

    def connect_cell(self, cell_1: tuple, cell_2: tuple):
        """
        Connecting cells by changing the wall cell between 
        passages and chosen frontier_neighbor to be part of the maze.
        """
        x1, y1 = cell_1
        x2, y2 = cell_2

        x_diff = x1 - x2
        y_diff = y1 - y2

        if x_diff != 0 and y_diff == 0:
            x_conn = (x1 + x2) // 2
            y_conn = y1

        elif y_diff != 0 and x_diff == 0:
            y_conn = (y1 + y2) // 2
            x_conn = x1

        if (x_conn, y_conn) in self.board.wall:
            self.passages.add((x_conn, y_conn))
            self.board.wall.remove((x_conn, y_conn))

    def generate(self):
        """
        Main function to generate maze using randomized Prim's algorithm.
        """
        if not self.board.frontiers:
            raise ValueError("Use initialize function first")

        while self.board.frontiers:
            t = int(time.time())
            random.seed(t)
            time.sleep(DELAY)
            self.board.draw_board(return_cells=False)
            
            # Randomly select a cell in frontier as part of the maze
            frontier = random.choice(list(self.board.frontiers))
            self.passages.add(frontier)

            # Randomly select a frontier neighbor and connect path
            neighbor = self.frontier_neighbor(frontier)
            self.connect_cell(frontier, neighbor)
            
            # Add new frontiers to the board.frontiers
            next_frontiers = self.get_frontiers(frontier)
            self.board.frontiers = self.board.frontiers | next_frontiers
            
            # Remove cell from frontier and board.wall
            self.board.frontiers.remove(frontier)
            self.board.wall.remove(frontier)

            pygame.display.flip()


    def get_maze_info(self):
        """
        Return the information about start, goal, and walls.
        """
        return {
            "start": self.start,
            "goal": self.goal,
            "walls": self.board.wall,
        }