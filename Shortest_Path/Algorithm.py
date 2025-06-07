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


    """
    Q-Learning Algorithm: Q(s,a) <-- Q(s,a) + alpha*((reward+best_future_reward) - Q(s,a))
    
    alpha: learning rate, within range 0.0~1.0
    epsilon: parameter for epsilon-greedy algorithm, within 0.0~1.0.
             if epsilon equals to 0.0, it'll be equivalent to 
             best greedy algorithm.
    """
    def __init__(self, board:Board, alpha=0.5, epsilon=0.1):
        self.board = board
        self.find = False

        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("Learning rate should be within 0.0 ~ 1.0")
        else:
            self.alpha = alpha
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("Learning rate should be within 0.0 ~ 1.0")
        else:
            self.epsilon = epsilon
    
    def initialize(self):
        """
        Create following information for solver:
        node_dict: key is coordinate of node; value is node
        """
        self.node_dict = {}
        for i in range(self.board.v_cells):
            for j in range(self.board.h_cells):
                # if (i,j) in self.board.wall:
                #     continue
                pos = (i,j)
                node = Node(pos, None, None)
                if pos == self.board.start:
                    self.start_node = node
                elif pos == self.board.target:
                    self.target_node = node
                
                self.node_dict[pos] = node
        
        self.q_values = defaultdict(dict)
        for pos in self.node_dict:
            neighbors = self.board.neighbors(pos, wall_included=True)
            for _, neighbor in neighbors:
                self.q_values[pos][neighbor] = 0

    def update_q_value(self, state:tuple, next_state:tuple, reward:int):
        """
        update q_values based on formula below:
        Q(s, a) = Q(s, a) + alpha * (current_reward + best_future_reward - Q(s,a))
        
        state: position of node -> tuple
        next_state: next position node after move -> tuple
        reward: current_reward -> int
        """
        old_q = self.q_values[state][next_state]
        new_est = reward + self.best_reward(next_state)
        new_q = old_q + self.alpha*(new_est - old_q)
        self.q_values[state][next_state] = new_q

    def best_reward(self, state:tuple)->float:
        """
        return best_future_reward from a state.

        state: position of node -> tuple
        """
        next_states = self.board.neighbors(state, wall_included=True)
        best = 0
        
        for _, next_state in next_states:
            best = max(best, self.q_values[state][next_state])

        return best

    def choose_action(self, state:tuple, available_actions:list, epsilon=True)->tuple:
        """
        return best action from a state

        state: position of node -> tuple
        available_actions: a list of all possible move -> list
        epsilon: if epsilon is True, epsilon greedy algorithm will be used.
                 else it'll return action with highest q_value.
        """
        t = int(time.time())
        random.seed(t)

        q_values = []
        for action in available_actions:
            q_value = self.q_values[state][action]
            q_values.append((q_value, action))

        q_values = sorted(q_values, key=lambda x:x[0], reverse=True)
        if not epsilon:
            return q_values[0][1]
        
        else:
            best_q = q_values[0]
            chosen = random.choices([best_q, q_values], weights=[1-self.epsilon, self.epsilon])
            if chosen != best_q:
                chosen = random.choice(q_values)
            return chosen[1]

    def solver(self, n):
        """
        Train AI to find shortest path using Q_Learning and DFS, do not allow AI to go back.
        """
        print('Training Start')
        search = self.initialize()

        self.board.draw_board(return_cells=False)
        pygame.display.flip()

        for i in range(n):
            stop = False
            last_state = None
            cur_state = None
            self.trail_path = [self.start_node.state]
            self.board.visited.add(self.start_node)
            while not stop and self.trail_path:
                # get current_state from last element of trail_path
                cur_state = self.trail_path[-1]

                # get all possible next_states have not been visited
                neighbors = self.board.neighbors(cur_state, wall_included=True)
                available_actions = [
                    neighbor 
                    for _, neighbor in neighbors
                    if self.node_dict[neighbor] not in self.board.visited
                ]

                # if every possible next_state has been visited and not find target, 
                # update q_value with -100 reward for last_state -> cur_state
                # and pop cur_state from trail_path. (It means the path is dead end)
                if len(available_actions) == 0 and len(self.board.visited) >= 3:
                    if self.trail_path[-1] == self.start_node.state:
                        self.trail_path.pop()
                    
                    else:
                        last_state = self.trail_path[-2]
                        self.update_q_value(
                            state=last_state,
                            next_state=cur_state,
                            reward=-100
                        )
                        self.trail_path.pop()
                    continue

                next_state = self.choose_action(cur_state, available_actions)
                # if next_state is wall, add to visited and update q_value with -10 reward
                if next_state in self.board.wall:
                    self.board.visited.add(self.node_dict[next_state])

                    self.update_q_value(
                        state=cur_state,
                        next_state=next_state,
                        reward=-10
                    )
                    continue
                    
                # if find target_node, update q_value with 100 reward
                elif next_state == self.target_node.state:
                    self.node_dict[next_state].parent = self.node_dict[cur_state]

                    self.update_q_value(
                        state=cur_state,
                        next_state=next_state,
                        reward=100
                    )
                    stop=True
                    self.find = True

                # normal path adding, update q_value with 0 reward
                else:
                    self.board.visited.add(self.node_dict[next_state])
                    self.trail_path.append(next_state)
                    self.node_dict[next_state].parent = self.node_dict[cur_state]

                    self.update_q_value(
                        state=cur_state,
                        next_state=next_state,
                        reward=0
                    )

                # draw condition of training
                cells = self.board.draw_board()   
                color = self.board.colors["purple"]                     
                for i, j in self.trail_path:
                    if (i, j) == self.board.start:
                        continue
                    rect = cells[i][j]
                    pygame.draw.rect(self.board.screen, color, rect)
                pygame.display.flip()
                
            if stop:
                self.board.clear_visited()
            
            if not self.trail_path:
                break

        # reset board.visited and board.path to ensure not effect output function
        self.board.clear_visited()
        print('Finish Training')

    def output(self):
        """
        Solve shortest path after training
        """
        # start from start node
        node = self.start_node
        visited = {node.state}
        self.board.path.append(node.state)
        # while node is not target, keep path adding
        count = 0
        while node != self.target_node:
            time.sleep(DELAY)
            self.board.draw_board(return_cells=False)

            neighbors = self.board.neighbors(node.state)
            available_actions = [
                neighbor 
                for _, neighbor in neighbors
                if neighbor not in visited
            ]
            best_action = self.choose_action(node.state, available_actions, epsilon=False)

            # get next node and append to board.path
            new_node = self.node_dict[best_action]    
            new_node.parent = node
            visited.add(new_node.state)
            if new_node.state in self.board.path:
                print("Train Fail")
                break

            self.board.path.append(new_node.state)
            node = new_node
            count += 1
            pygame.display.flip()
        
        if node == self.target_node:
            print('Total Step is {}'.format(count+1))


class BRRT(Search):
    def __init__(self, board: Board, max_iter=10000, step_size=1, p1=0.3, p2=0.5, alpha=0.4, beta=0.3, gamma=0.3):
        self.board = board
        self.max_iter = max_iter
        self.step_size = step_size
        self.find = False
        self.p1 = p1  # Probability for uniform sampling
        self.p2 = p2  # Probability for circle sampling when guided
        self.alpha = alpha  # Weight for d(s_i, t_j)
        self.beta = beta   # Weight for d(s_i, t_0)
        self.gamma = gamma  # Weight for d(t_j, s_0)

    def initialize(self):
        self.start_tree = {self.board.start: None}
        self.goal_tree = {self.board.target: None}
        self.start_nodes = [self.board.start]
        self.goal_nodes = [self.board.target]
        self.s0 = self.board.start  # Initial start node
        self.t0 = self.board.target  # Initial target node

        print("BRRT initialized with start at {} and target at {}".format(self.board.start, self.board.target))

    def add_to_tree(self, tree, tree_nodes, child, parent, tree_name="unknown"):
        if child != parent:  # Prevent self-loop
            tree[child] = parent
            tree_nodes.append(child)
            print(f"[Tree Build] ({tree_name}) Added node {child} with parent {parent}")
            print(f"[Tree Build] ({tree_name}) {parent} -> {child}")
        else:
            print(f"[Warning] ({tree_name}) Attempted to add self-loop at node {child}, skipped.")
    def get_adjacent_nodes(self, node):
        x, y = node
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Chỉ các hướng ngang/dọc
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board.v_cells and 0 <= ny < self.board.h_cells:
                if (nx, ny) not in self.board.wall:
                    neighbors.append((nx, ny))
        return neighbors

    def sample_points_in_circle(self, center, radius, num_samples):
        cx, cy = center
        candidates = []
        for _ in range(num_samples):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(0, radius)
            dx = int(round(r * math.cos(angle)))
            dy = int(round(r * math.sin(angle)))
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < self.board.v_cells and 0 <= ny < self.board.h_cells and (nx, ny) not in self.board.wall:
                candidates.append((nx, ny))
        return candidates

    def find_guide_pair(self, S, T):
        """Ensure valid guide nodes are returned, defaulting to s0 and t0 if necessary."""
        s_guide = random.choice(list(S.keys())) if S and len(S) > 1 else self.s0
        t_guide = random.choice(list(T.keys())) if T and len(T) > 1 else self.t0
        return (s_guide, t_guide)

    def sampling(self, S, T):
        """Ưu tiên sampling về phía đích (t_0)."""
        n = random.random()
        
        if n < self.p1:
            # Uniform sampling
            x_rand = (
                random.randint(0, self.board.v_cells - 1),
                random.randint(0, self.board.h_cells - 1)
            )
            while x_rand in self.board.wall:
                x_rand = (
                    random.randint(0, self.board.v_cells - 1),
                    random.randint(0, self.board.h_cells - 1)
                )
            return x_rand

        # Nếu không, dẫn hướng sampling về phía t0
        target = self.t0
        tree = S  # Cây hiện tại
        
        # Chọn node trong cây hiện tại gần nhất với đích
        nearest = self.get_best_node(list(tree.keys()), target)

        # Sinh điểm xung quanh nearest, nhưng ưu tiên hướng về target
        if nearest:
            direction = (target[0] - nearest[0], target[1] - nearest[1])
            dist = math.hypot(direction[0], direction[1])

            if dist > 0:
                ratio = min(self.step_size / dist, 1)
                dx = int(round(direction[0] * ratio))
                dy = int(round(direction[1] * ratio))
                x_rand = (nearest[0] + dx, nearest[1] + dy)

                if 0 <= x_rand[0] < self.board.v_cells and 0 <= x_rand[1] < self.board.h_cells:
                    if x_rand not in self.board.wall:
                        return x_rand

        # Dự phòng: trả về nearest
        return nearest


    def compute_distance(self, p1, p2):
        """Compute Euclidean distance between two points."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def heuristic(self, s_i, t_j):
        """Compute heuristic value h(s_i, t_j) = α·d(s_i, t_j) + β·d(s_i, t_0) + γ·d(t_j, s_0)"""
        d_si_tj = self.compute_distance(s_i, t_j)
        d_si_t0 = self.compute_distance(s_i, self.t0)
        d_tj_s0 = self.compute_distance(t_j, self.s0)
        return self.alpha * d_si_tj + self.beta * d_si_t0 + self.gamma * d_tj_s0
    def get_best_node(self, tree_nodes, target_node):
        """Chọn node trong cây gần nhất với target_node theo heuristic."""
        best_node = None
        best_score = float('inf')
        for node in tree_nodes:
            score = self.heuristic(node, target_node)
            if score < best_score:
                best_score = score
                best_node = node
        return best_node
    def step_from_to(self, start, target):
        """Tiến một bước từ `start` về phía `target`, với khoảng cách tối đa là `step_size`."""
        dx = target[0] - start[0]
        dy = target[1] - start[1]
        dist = math.hypot(dx, dy)

        if dist == 0:
            return start

        ratio = self.step_size / dist
        new_x = int(round(start[0] + dx * ratio))
        new_y = int(round(start[1] + dy * ratio))

        # Đảm bảo điểm mới nằm trong giới hạn bản đồ
        new_x = max(0, min(new_x, self.board.v_cells - 1))
        new_y = max(0, min(new_y, self.board.h_cells - 1))

        return (new_x, new_y)

    def extend_tree(self, tree, tree_nodes, target_node, other_tree):
        best_s_i = self.get_best_node(tree_nodes, target_node)

        if best_s_i is None:
            return None

        step = self.step_from_to(best_s_i, target_node)

        # Check if this step is already in the tree
        if step in tree:
            return None

        if step not in self.board.wall:
            tree_name = "start" if tree is self.start_tree else "goal"
            self.add_to_tree(tree, tree_nodes, step, best_s_i, tree_name=tree_name)

            # Kiểm tra vùng lân cận 3x3 xem có node nào thuộc cây còn lại không
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (step[0] + dx, step[1] + dy)
                    if neighbor in other_tree:
                        if step not in other_tree:
                            other_tree[step] = neighbor
                        if neighbor not in tree:
                            tree[neighbor] = step
                        print(f"[Connection] Trees connected at node {step}")
                        return step  # Đã kết nối thành công

        return None  # Không kết nối được

    def draw_tree(self, tree, cells, color):
        for node in tree:
            i, j = node
            rect = cells[i][j]
            pygame.draw.rect(self.board.screen, color, rect)

    def draw_start_and_goal(self, cells):
        start_i, start_j = self.board.start
        target_i, target_j = self.board.target
        pygame.draw.rect(self.board.screen, self.board.colors["red"], cells[start_i][start_j])
        pygame.draw.rect(self.board.screen, self.board.colors["blue"], cells[target_i][target_j])

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
                target_node = self.sampling(self.start_tree, self.goal_tree)
                new_node = self.extend_tree(self.start_tree, self.start_nodes, target_node, self.goal_tree)
                if new_node:
                    self.find = True
                    self.path = self.build_path(new_node)
                    print(f"[SUCCESS] Path built successfully from start to goal: {self.path}")
                    self.output()
                    break
                self.draw_tree(self.start_tree, cells, self.board.colors["green"])
            else:
                target_node = self.sampling(self.goal_tree, self.start_tree)
                new_node = self.extend_tree(self.goal_tree, self.goal_nodes, target_node, self.start_tree)
                if new_node:
                    self.find = True
                    self.path = self.build_path(new_node)
                    print(f"[SUCCESS] Path built successfully from goal to start: {self.path}")
                    self.output()
                    break
                self.draw_tree(self.goal_tree, cells, self.board.colors["purple"])

            toggle = not toggle
            self.draw_start_and_goal(cells)
            pygame.display.flip()


        if self.find:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                time.sleep(0.1)
    def interpolate_path(self, start, end):
        """Trả về danh sách các ô từ start đến end theo từng bước liền kề (4 hướng)."""
        path = [start]
        x1, y1 = start
        x2, y2 = end
        while (x1, y1) != (x2, y2):
            if x1 < x2: x1 += 1
            elif x1 > x2: x1 -= 1
            elif y1 < y2: y1 += 1
            elif y1 > y2: y1 -= 1
            path.append((x1, y1))
        return path

    def build_path(self, connect_node):
        def trace_path(tree, node):
            path = []
            while node is not None and node in tree:
                path.append(node)
                node = tree[node]
            return path

        path_start = trace_path(self.start_tree, connect_node)
        path_goal = trace_path(self.goal_tree, connect_node)

        if not path_start or not path_goal:
            print("[ERROR] Could not build complete path.")
            return []

        path_start.reverse()
        full_path = []

        # Nối từng cặp điểm bằng các bước nhỏ
        combined_path = path_start + path_goal[1:]
        for i in range(len(combined_path) - 1):
            seg = self.interpolate_path(combined_path[i], combined_path[i + 1])
            if i > 0:
                seg = seg[1:]  # tránh trùng điểm
            full_path.extend(seg)

        return full_path


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

