import time
import heapq
import pygame
import random
from env import *
from Queue import *
from collections import defaultdict
from abc import ABCMeta, abstractmethod
import math

INF = float('inf')
DELAY = 0.01
DISTANCE = 1

class Search(metaclass=ABCMeta):
    """
    class for search algorithms
    """
    @abstractmethod
    def solver(self):
        """
        Solver to find shortest path between start and target node
        """
        pass
    
    @abstractmethod
    def initialize(self):
        """
        Create information required for solver
        """
        pass

    def output(self):
        # get cells first in case path to be drawn directly
        cells = self.board.draw_board()   

        # derive shortest path starting from target node and reverse it
        node = self.target_node
        while node.parent is not None:
            self.board.path.append(node.state)
            node = node.parent
        self.board.path.reverse()

        # draw shortest path step by step
        color = self.board.colors["p_yellow"]                     
        for i, j in self.board.path:
            time.sleep(1.5*DELAY)
            rect = cells[i][j]
            pygame.draw.rect(self.board.screen, color, rect)
            pygame.display.flip()
def sample_points_in_circle(center, radius, num_samples, board):
    cx, cy = center
    candidates = []
    for _ in range(num_samples):
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(0, radius)
        dx = int(round(r * math.cos(angle)))
        dy = int(round(r * math.sin(angle)))
        nx, ny = cx + dx, cy + dy
        if 0 <= nx < board.v_cells and 0 <= ny < board.h_cells:
            if (nx, ny) not in board.wall:
                candidates.append((nx, ny))
    return candidates

class BRRT(Search):
    def __init__(self, board: Board, max_iter=10000, step_size=1):
        self.board = board
        self.max_iter = max_iter
        self.step_size = step_size
        self.find = False
        self.takenNodes = set()  # To track taken nodes


    def initialize(self):
        self.start_tree = {self.board.start: None}
        self.goal_tree = {self.board.target: None}
        self.start_nodes = [self.board.start]
        self.goal_nodes = [self.board.target]
        print("Taken nodes initialized.", self.takenNodes)
        print("BRRT initialized with start at {} and target at {}".format(self.board.start, self.board.target))

    def add_to_tree(self, tree, tree_nodes, child, parent, tree_name="unknown"):
        if child != parent:  # Prevent self-loop
            tree[child] = parent
            tree_nodes.append(child)
            self.takenNodes.add(child)  # Mark this node as taken           
            print(f"[Taken Nodes] ({tree_name}) Node {child} added to taken nodes.") 
            print(f"[Tree Build] ({tree_name}) Added node {child} with parent {parent}")
            print(f"[Tree Build] ({tree_name}) {parent} -> {child}")
            print(f"[Tree Build] ({tree_name}) Current taken nodes: {self.takenNodes}")
        else:
            print(f"[Warning] ({tree_name}) Attempted to add self-loop at node {child}, skipped.")


    def extend_tree(self, tree_nodes, tree, other_tree):
        rand_point = (random.randint(0, self.board.v_cells - 1),
                      random.randint(0, self.board.h_cells - 1))

        nearest = min(tree_nodes, key=lambda n: abs(n[0] - rand_point[0]) + abs(n[1] - rand_point[1]))

        if nearest in self.board.wall:
            return None

        direction = (rand_point[0] - nearest[0], rand_point[1] - nearest[1])
        print(f"[Extend Tree] Nearest node to random point {rand_point} is {nearest} with direction {direction}")
        step = (
            nearest[0] + self.step_size * (1 if direction[0] > 0 else -1 if direction[0] < 0 else 0),
            nearest[1] + self.step_size * (1 if direction[1] > 0 else -1 if direction[1] < 0 else 0)
        )
        if 0 <= step[0] < self.board.v_cells and 0 <= step[1] < self.board.h_cells:
            if step not in self.board.wall:
                if step in tree or step in other_tree or step in self.takenNodes:
                    print(f"[Warning] Node {step} already exists in tree or taken nodes, skipping.")
                    return None  # Avoid re-adding nodes
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor = (step[0] + dx, step[1] + dy)
                    if neighbor in other_tree:
                        tree_name = "start" if tree is self.start_tree else "goal"
                        self.add_to_tree(tree, tree_nodes, step, nearest, tree_name=tree_name)
                        if step not in other_tree:
                            other_tree[step] = neighbor

                        if neighbor not in tree:
                            tree[neighbor] = step

                        print(f"[Connection] Trees connected at node {step}")
                        return step

                # Expand current tree if no connection
                tree_name = "start" if tree is self.start_tree else "goal"
                self.add_to_tree(tree, tree_nodes, step, nearest, tree_name=tree_name)
        return None

    def draw_tree(self, tree, cells, color):
        for node in tree:
            i, j = node
            rect = cells[i][j]
            pygame.draw.rect(self.board.screen, color, rect)

    def draw_start_and_goal(self, cells):
        start_i, start_j = self.board.start
        target_i, target_j = self.board.target
        pygame.draw.rect(self.board.screen, self.board.colors["red"], cells[start_i][start_j])   # Start
        pygame.draw.rect(self.board.screen, self.board.colors["blue"], cells[target_i][target_j]) # Target

    def solver(self):
        self.initialize()
        print("Starting BRRT solver...")
        cells = self.board.draw_board()
        self.draw_start_and_goal(cells)
        pygame.display.flip()

        toggle = True  # True: start_tree, False: goal_tree

        for _ in range(self.max_iter):
            pygame.event.pump()
            time.sleep(DELAY)

            if toggle:
                new_node = self.extend_tree(self.start_nodes, self.start_tree, self.goal_tree)
                if new_node:
                    self.log_common_nodes()
                    self.find = True
                    self.path = self.build_path(new_node)
                    print(f"[SUCCESS] Path built successfully from start to goal: {self.path}")
                    self.output()
                    break
                self.draw_tree(self.start_tree, cells, self.board.colors["green"])
            else:
                new_node = self.extend_tree(self.goal_nodes, self.goal_tree, self.start_tree)
                if new_node:
                    self.log_common_nodes()
                    self.find = True
                    self.path = self.build_path(new_node)
                    print(f"[SUCCESS] Path built successfully from goal to start: {self.path}")
                    self.output()
                    break
                self.draw_tree(self.goal_tree, cells, self.board.colors["purple"])

            toggle = not toggle  # Đổi lượt giữa 2 cây
            self.draw_start_and_goal(cells)
            pygame.display.flip()

        if self.find:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                time.sleep(0.1)

    def build_path(self, connect_node):
        print(f"Building path from connection node: {connect_node}")
        print(f"In start_tree: {connect_node in self.start_tree}")
        print(f"In goal_tree: {connect_node in self.goal_tree}")

        def trace_path(tree, node, name):
            visited = set()
            path = []
            while node is not None and node in tree:
                if node in visited:
                    print(f"[Loop Detected] in {name} at {node}")
                    break
                visited.add(node)
                path.append(node)
                node = tree[node]
            return path

        path_start = trace_path(self.start_tree, connect_node, "start_tree")
        path_goal = trace_path(self.goal_tree, connect_node, "goal_tree")  # <- FIXED

        if not path_start or not path_goal:
            print("[ERROR] Could not build complete path.")
            return []

        path_start.reverse()
        return path_start + path_goal


    def log_full_tree(self):
        print("\n--- Full Start Tree ---")
        for child, parent in self.start_tree.items():
            if parent is not None:
                print(f"{parent} -> {child}")
        print("\n--- Full Goal Tree ---")
        for child, parent in self.goal_tree.items():
            if parent is not None:
                print(f"{parent} -> {child}")

    def export_tree_as_dot(self, filename="tree.dot"):
        with open(filename, "w") as f:
            f.write("digraph G {\n")
            for child, parent in self.start_tree.items():
                if parent is not None:
                    f.write(f'    "{parent}" -> "{child}";\n')
            for child, parent in self.goal_tree.items():
                if parent is not None:
                    f.write(f'    "{parent}" -> "{child}";\n')
            f.write("}\n")
        print(f"[DOT Export] Tree exported to {filename}")

    def output(self):
        if not self.find or not self.path:
            print("[OUTPUT] No valid path found.")
            return

        cells = self.board.draw_board()
        color = self.board.colors["yellow"]

        for i, j in self.path:
            time.sleep(1.5 * DELAY)
            rect = cells[i][j]
            pygame.draw.rect(self.board.screen, color, rect)
            pygame.display.flip()

        self.draw_start_and_goal(cells)
        pygame.display.flip()
    def log_common_nodes(self):
        common_nodes = set(self.start_tree.keys()) & set(self.goal_tree.keys())
        if not common_nodes:
            print("[INFO] No common nodes between start_tree and goal_tree.")
        else:
            print(f"[INFO] Found {len(common_nodes)} common node(s):")
            for node in common_nodes:
                print(f" - {node}")
