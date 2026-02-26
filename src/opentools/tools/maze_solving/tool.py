import os, sys, cv2, traceback, re, json, heapq
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from opentools.core.base import BaseTool
import numpy as np
class Maze_Solving_Tool(BaseTool):
    """Maze_Solving_Tool
    ---------------------
    Purpose:
        A sophisticated maze analysis tool that solves and analyzes grid-based mazes using computer vision and pathfinding algorithms. It can detect maze walls, start/end positions from colored arrows, and find optimal paths while analyzing turn patterns and movement statistics. The tool uses A* pathfinding with Manhattan distance heuristic and provides detailed path analysis including turns and directional movements.

    Core Capabilities:
        - Solves grid-based mazes using A* pathfinding algorithm
        - Detects maze walls and start/end positions from colored arrows
        - Analyzes path statistics (turns, directional movements, step counts)
        - Provides detailed movement analysis with turn sequences
        - Supports various maze dimensions and configurations

    Intended Use:
        Use this tool when you need to solve grid-based mazes, including finding optimal paths and analyzing turn patterns and movement statistics.

    Limitations:
        - May not handle complex maze configurations or dimensions
    """

    require_llm_engine = True
    def __init__(self, model_string="gpt-4o-mini", llm_engine=None):
        super().__init__(
            type='function',
            name="Maze_Solving_Tool",
            description="""A sophisticated maze analysis tool that solves and analyzes grid-based mazes using computer vision and pathfinding algorithms. It can detect maze walls, start/end positions from colored arrows, and find optimal paths while analyzing turn patterns and movement statistics. The tool uses A* pathfinding with Manhattan distance heuristic and provides detailed path analysis including turns and directional movements. CAPABILITIES: Solves grid-based mazes using A* pathfinding algorithm, detects maze walls and start/end positions from colored arrows, analyzes path statistics (turns, directional movements, step counts), provides detailed movement analysis with turn sequences, supports various maze dimensions and configurations. SYNONYMS: maze solver, puzzle solver, pathfinding tool, maze analyzer, grid puzzle solver, computer vision maze solver, A* algorithm tool, maze path finder. EXAMPLES: 'Find the optimal path in a 13 * 13 maze with minimum turns', 'Count total steps and turns in 8 * 8 maze solution', 'Analyze the path from start to end in this maze image'.""",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text query containing maze dimensions (e.g., '13 * 13') and specific requirements about the path analysis"
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Path to an image file showing a grid-based maze with black walls, white paths, and colored arrows for start (green) and end (blue) positions"
                    }
                },
                "required": ["query", "image_path"],
                "additionalProperties": False,
            },
            strict=True,
            category="puzzle_solving",
            tags=["maze_solver", "puzzle_solver", "pathfinding", "A*_algorithm", "grid_puzzle", "maze_analysis", "path_optimization", "turn_analysis"],
            limitation="Requires clear black and white contrast for wall detection, maze dimensions must be explicitly stated in query, only handles grid-based mazes with orthogonal movements, start and end positions must be marked with colored arrows, no support for diagonal movements, may not detect faded or unclear maze boundaries, green start arrow and blue end arrow must be visible",
            agent_type="Puzzle-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(query='Find the optimal path in a 13 * 13 maze with minimum turns', image_path='maze.png')",
                "description": "Find the optimal path in a 13 * 13 maze with minimum turns"
            },
            require_llm_engine=True,
            llm_engine=llm_engine,
        )
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        self.direction_names = ['RIGHT', 'DOWN', 'LEFT', 'UP']

    def matrix_size(self, query) -> tuple[int, int]:
        pattern = re.compile(r'\b(\d+)\s*(?:\*|x)\s*(\d+)\b', re.IGNORECASE)
        m = pattern.search(query)
        if m:
            rows, cols = map(int, m.groups())
            return rows, cols
        else:
            prompt = "Extract the size of the grid board from the query.Return the size in the format of 'rows*cols'. Query: " + query
            result = self.llm_engine.generate(prompt)
            if isinstance(result, dict):
                result = result.get('text')
            else:
                result = str(result)
            rows, cols = map(int, result.split('*'))
            return rows, cols

    def run(self, query: str, image_path: str) :
        rows, cols = self.matrix_size(query)
        maze, start, end  = self.maze_image_to_matrix(image_path, rows, cols)
        analysis = self.find_path(maze, start, end, rows, cols)
        return analysis

    def matrix_to_string(self,matrix):
        result = []
        for row in matrix:
            row_str = ""
            for cell in row:
                if cell == 1:
                    row_str += "█"
                elif cell == 0:
                    row_str += " "
                elif cell == 's':
                    row_str += "S"
                elif cell == 'e':
                    row_str += "E"
            result.append(row_str)
        return "\n".join(result)
            
    def maze_image_to_matrix(self,image_path, rows, cols):
        """
        Convert a maze image to a 2D matrix representation.
        Args:
            image_path: Path to the maze image
            rows: Number of rows in the maze 
            cols: Number of columns in the maze         
        Returns:
            2D list representing the maze where:
            - 0 = white (path)
            - 1 = black (wall)
            - 's' = start (green arrow)
            - 'e' = end (blue arrow)
        """
        if not os.path.isabs(image_path):
            image_path = os.path.join(r"C:\Users\daoqm\OpenTools\opentools\src\opentools\tools\test_file", image_path)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to RGB (OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale for wall detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Create binary image with high threshold
        _, binary = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        
        # Find maze boundaries
        rows_with_black = np.any(binary == 0, axis=1)
        cols_with_black = np.any(binary == 0, axis=0)
        
        y_start = np.argmax(rows_with_black)
        y_end = len(rows_with_black) - np.argmax(rows_with_black[::-1])
        x_start = np.argmax(cols_with_black)
        x_end = len(cols_with_black) - np.argmax(cols_with_black[::-1])
        
        # Extract maze region
        maze_region = img_rgb[y_start:y_end, x_start:x_end]
        maze_binary = binary[y_start:y_end, x_start:x_end]
        
        h, w = maze_binary.shape
        
        # Calculate cell dimensions
        cell_width = w / cols
        cell_height = h / rows
        
        # Initialize the matrix
        matrix = []
        
        # First pass: create basic maze structure (boundary and path)
        for row in range(rows):
            matrix_row = []
            for col in range(cols):
                # Calculate cell boundaries
                cell_y1 = int(row * cell_height)
                cell_y2 = int((row + 1) * cell_height)
                cell_x1 = int(col * cell_width)
                cell_x2 = int((col + 1) * cell_width)
                
                # Extract cell binary data
                cell_binary = maze_binary[cell_y1:cell_y2, cell_x1:cell_x2]
                
                # Determine if wall or path based on center pixels
                center_value = np.mean(cell_binary.astype(np.float32))
                if center_value < 128:  # Mostly black = wall
                    matrix_row.append(1)
                else:  # Mostly white = path
                    matrix_row.append(0)
            matrix.append(matrix_row)
            
        # Second pass: detect arrows on all boundary cells
        def detect_color_in_cell(row, col, color_type='green'):
            """
            Detect if a cell contains the specified color (green for start, blue for end)
            """
            cell_y1 = int(row * cell_height)
            cell_y2 = int((row + 1) * cell_height)
            cell_x1 = int(col * cell_width)
            cell_x2 = int((col + 1) * cell_width)
            
            cell = maze_region[cell_y1:cell_y2, cell_x1:cell_x2]

            if cell.size == 0:
                return False
            
            cell_float = cell.astype(np.float32)
            avg_color = np.mean(cell_float, axis=(0, 1))
            if color_type == 'green':
                # Green detection: G channel significantly higher than R and B
                return (avg_color[1] > avg_color[0] + 7 and 
                    avg_color[1] > avg_color[2] + 7 and
                    avg_color[1] > 150)  # Ensure it's actually greenish
            elif color_type == 'blue':
                # Blue detection: B channel higher than others
                return (avg_color[2] > avg_color[0] + 14 and 
                    avg_color[2] > avg_color[1] + 14 and
                    avg_color[2] > 150)  # Ensure it's actually bluish
            return False
        
        def get_boundary_cells():
            """
            Get all boundary cells (cells on the edges of the maze)
            Returns list of (row, col) tuples
            """
            boundary_cells = []
            # Top row
            for col in range(cols):
                boundary_cells.append((0, col))
            # Bottom row
            for col in range(cols):
                boundary_cells.append((rows-1, col))
            # Left column (excluding corners already added)
            for row in range(1, rows-1):
                boundary_cells.append((row, 0))
            # Right column (excluding corners already added)
            for row in range(1, rows-1):
                boundary_cells.append((row, cols-1))
            return boundary_cells        
        # Search for start arrow (green) on all boundary cells
        start_found = False
        boundary_cells = get_boundary_cells()
        start_arrow_pos = ()
       
        for row, col in boundary_cells:
            if matrix[row][col] == 0:  # Only check path cells
                if detect_color_in_cell(row, col, 'green'):
                    matrix[row][col] = 's'
                    start_arrow_pos = (row,col)
                    start_found = True
                    break
        
        # Search for end arrow (blue) on all boundary cells
        end_found = False
        end_arrow_pos = ()
        for row, col in boundary_cells:
            if matrix[row][col] == 0:  # Only check path cells
                if detect_color_in_cell(row, col, 'blue'):
                    matrix[row][col] = 'e'
                    end_arrow_pos = (row,col)
                    end_found = True
                    break
    
        if not start_found:
            print("Start arrow not detected by color, using fallback positioning...")
            fallback_positions = []
            # Left edge            
            for row in range(rows):
                if matrix[row][0] == 0:
                    fallback_positions.append((row, 0))
            
            for col in range(cols):
                if matrix[0][col] == 0:
                    fallback_positions.append((0, col))
        
            if fallback_positions:
                row, col = fallback_positions[0]
                matrix[row][col] = 's'
                start_arrow_pos = (row,col)
                print(f"Start placed at fallback position ({row}, {col})")
        
        if not end_found:
            print("End arrow not detected by color, using fallback positioning...")
            fallback_positions = []
            
            # Right edge
            for row in range(rows-1, -1, -1):
                if matrix[row][cols-1] == 0:
                    fallback_positions.append((row, cols-1))
            
            # Bottom edge
            for col in range(cols-1, -1, -1):
                if matrix[rows-1][col] == 0:
                    fallback_positions.append((rows-1, col))
        
            if fallback_positions:
                row, col = fallback_positions[0]
                matrix[row][col] = 'e'
                end_arrow_pos = (row,col)
                print(f"End placed at fallback position ({row}, {col})")
        
        return matrix,start_arrow_pos,end_arrow_pos

    def _is_valid(self, row, col, rows, cols, maze):
        """Check if position is valid and not a wall"""
        return (0 <= row < rows and 
                0 <= col < cols and 
                maze[row][col] != 1)
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_direction_index(self, from_pos, to_pos):
        """Get direction index from movement"""
        dr, dc = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
        for i, (r, c) in enumerate(self.directions):
            if (r, c) == (dr, dc):
                return i
        return -1
    
    def _calculate_turn_type(self, prev_dir, curr_dir):
        """Calculate turn type between two directions"""
        if prev_dir == -1:  # First move
            return 'NONE'
        
        # Calculate turn difference
        turn_diff = (curr_dir - prev_dir) % 4
        
        if turn_diff == 0:
            return 'STRAIGHT'
        elif turn_diff == 1:
            return 'RIGHT'
        elif turn_diff == 2:
            return 'U_TURN'
        elif turn_diff == 3:
            return 'LEFT'
    
    def find_path(self, maze, start, end, rows, cols):
        """Find optimal path using A* algorithm"""
        if not start or not end:
            return None, {"success": False, "error_type": "invalid_path"}
        
        # Priority queue: (f_score, g_score, position, direction, path)
        pq = [(0, 0, start, -1, [start])]
        visited = set()
        
        while pq:
            f_score, g_score, current, prev_direction, path = heapq.heappop(pq)
            
            if current == end:
                return self._analyze_path(path, maze)
            
            state = (current, prev_direction)
            if state in visited:
                continue
            visited.add(state)
            
            # Explore neighbors
            for i, (dr, dc) in enumerate(self.directions):
                new_row, new_col = current[0] + dr, current[1] + dc
                new_pos = (new_row, new_col)
                
                if not self._is_valid(new_row, new_col, rows, cols, maze):
                    continue
                
                new_g_score = g_score + 1
                h_score = self._manhattan_distance(new_pos, end)
                new_f_score = new_g_score + h_score
                
                new_path = path + [new_pos]
                
                heapq.heappush(pq, (new_f_score, new_g_score, new_pos, i, new_path))
        
        return None, {"success": False, "error_type": "invalid_path"}
    
    def _analyze_path(self, path, maze):
        """Analyze the path for turns and directional movements"""
        if len(path) < 2:
            return {"success": False, "error_type": "invalid_path"}
        
        analysis = {
            'total_steps': len(path),
            'left_turns': 0,
            'right_turns': 0,
            'total_turns': 0,
            'straight_moves': 0,
            'go_up': 0,
            'go_down': 0,
            'go_left': 0,
            'go_right': 0,
        }
        
        prev_direction = -1
        
        for i in range(1, len(path)):
            current_pos = path[i]
            prev_pos = path[i-1]
            
            # Get current direction
            current_direction = self._get_direction_index(prev_pos, current_pos)
            direction_name = self.direction_names[current_direction]
            
            # Count directional movements
            if current_direction == 0:  # RIGHT
                analysis['go_right'] += 1
            elif current_direction == 1:  # DOWN
                analysis['go_down'] += 1
            elif current_direction == 2:  # LEFT
                analysis['go_left'] += 1
            elif current_direction == 3:  # UP
                analysis['go_up'] += 1
            
            # Calculate turn type
            turn_type = self._calculate_turn_type(prev_direction, current_direction)
            
            # Count turns
            if turn_type == 'LEFT':
                analysis['left_turns'] += 1
                analysis['total_turns'] += 1
            elif turn_type == 'RIGHT':
                analysis['right_turns'] += 1
                analysis['total_turns'] += 1
            elif turn_type == 'STRAIGHT':
                analysis['straight_moves'] += 1
            prev_direction = current_direction
        analysis['success'] = True
        return analysis

    def llm_evaluation(self, query, response, expected_output):
        """
        Evaluate the response using LLM to check if it meets the query requirements.
        Args:
            query: The original query string
            response: The response from the tool
            expected_output: The expected output format or content
        Returns:
            bool: True if the response meets the requirements, False otherwise
        """
        prompt = f"Evaluate the following response based on the query:\nQuery: {query}\nResponse: {response}\nExpected Output: {expected_output}\nDoes the response meet the requirements? Answer with 'True' or 'False'."
        evaluation = self.llm_engine.generate(prompt)
        evaluation = evaluation.get("text") or str(evaluation)
        return evaluation.strip().lower() == 'true'

    def test(self):
        """
        Test the maze solving tool and write structured JSON results.
        """
        try:
            import json
            # Open testbench
            file_test = os.path.join(os.path.dirname(__file__), '..', 'test_file', 'data.json')
            with open(file_test, encoding='utf-8') as f:
                data = json.load(f)['maze']

            # Prepare result dict for JSON output
            file_result = os.path.join(os.path.dirname(__file__), 'test_result.json')
            test_result = {}
            test_result['Test-File length'] = len(data)
            run_accuracy = {'run_1': 0, 'run_2': 0, 'run_3': 0}

            for i in range(len(data)):
                test = data[i]
                question_result = {"id": f"maze_solver_{i + 1}"}
                if 'query' in test:
                    question_result['query'] = test['query']
                if 'answer' in test:
                    question_result['answer'] = test['answer']
                # Run 3 times for each test
                for j in range(0, 3):
                    run_result = {}
                    test['image_path'] = os.path.join(os.path.dirname(__file__), '..', 'test_file', test['image_path'])
                    result = self.run(test['query'], test['image_path'])
                    run_result['result'] = result
                    # Use llm_evaluation to check correctness
                    correctness = False
                    if self.llm_evaluation and self.llm_evaluation(test['query'], result, test['answer']):
                        correctness = True
                        run_accuracy[f'run_{j + 1}'] += 1
                        run_result['accuracy'] = 1
                    else:
                        run_result['accuracy'] = 0
                        print(f"Error in question {i + 1}: Expected {test['answer']}")
                    run_result['tool_call_pass'] = True
                    run_result['expected_solution'] = test['answer'] if 'answer' in test else None
                    run_result['correctness'] = correctness
                    question_result[f'run_{j + 1}'] = run_result
                test_result[f'Q{i + 1}'] = question_result
                print(f"Finish query: {i + 1}")

            # Calculate the accuracy of the tool
            test_result['Final_Accuracy'] = {
                'run_1': run_accuracy['run_1'] * 100 / len(data),
                'run_2': run_accuracy['run_2'] * 100 / len(data),
                'run_3': run_accuracy['run_3'] * 100 / len(data),
            }
            test_result['token_usage'] = self.llm_engine.get_token_usage() if hasattr(self.llm_engine, "get_token_usage") else {}

            print(test_result['Final_Accuracy'])
            with open(file_result, "w", encoding="utf-8") as output_file:
                json.dump(test_result, output_file, indent=2, default=str)
            return test_result['Final_Accuracy']
        except Exception as e:
            print(f"Error during test execution: {e}")
            return False


# Example usage
if __name__ == "__main__":
    tool = Maze_Solving_Tool()
    # tool.embed_tool()
    # tool.test()
    query = """This is maze having 13 * 11 cells. The empty cells are coloured white and the obstacle cells are coloured black. From an empty cell, you can only move up, down, left, or right to another adjacent empty cell. You cannot move diagonally between two empty cells and cannot step into a cell with an obstacle. The entry cell of the maze is shown with the green arrow. The exit cell of the maze is shown with the blue arrow. Suppose you have found the most optimal path in the maze between the entrance and exit, where you need to go through the least number of empty cells and you need to make the least number of left and right turns. What is the total number of left turns do you need to make in this optimal path?"""
    image_path =r'/home/daoqm/opentools/src/opentools/Benchmark/algopuzzlevqa/images/maze_0000.jpg'
    print(tool.run(query, image_path))