import sys, os, cv2, numpy as np, re, json, traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', '..')))
from opentools.core.base import BaseTool
from collections import deque
from typing import List, Tuple, Dict
class Woodslide_Solver_Tool(BaseTool):
    """Woodslide_Solver_Tool
    ---------------------
    Purpose:
        Advanced sliding block puzzle solver using computer vision and pathfinding algorithms. This tool specializes in analyzing and solving wooden block puzzles by detecting blocks of various sizes, determining their positions, and finding optimal move sequences to transform starting configurations into target configurations.

    Core Capabilities:
        - Solves sliding block puzzles with multiple block sizes (1x1, 1x2, 2x1, 2x2, etc.)
        - Analyzes puzzle images with computer vision
        - Finds minimum move solutions using BFS pathfinding
        - Handles complex configurations with multiple empty spaces
        - Validates move constraints and block collisions
        - Uses color detection to identify blocks
        - Implements sophisticated move validation for physical constraints

    Intended Use:
        Use this tool when you need to solve sliding block puzzles, including wooden block puzzles.

    Limitations:
        - Requires clear contrast between wooden blocks and empty spaces
        - Grid dimensions must be explicitly stated
        - Blocks can only move horizontally/vertically
        - Computational complexity increases with puzzle size
        - Image must show both start and end configurations side by side
    """
    # Default args for `opentools test Woodslide_Solver_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "woodslide_solver",
        "file_location": "woodslide_solver",
        "result_parameter": "result",
        "search_type": "similarity_eval",
    }
    require_llm_engine = True
    def __init__(self, model_string="gpt-4o-mini", llm_engine=None):
        super().__init__(
            type='function',
            name="Woodslide_Solver_Tool",
            description="""Advanced sliding block puzzle solver using computer vision and pathfinding algorithms. This tool specializes in analyzing and solving wooden block puzzles by detecting blocks of various sizes, determining their positions, and finding optimal move sequences to transform starting configurations into target configurations.CAPABILITIES: Solves sliding block puzzles with multiple block sizes (1x1, 1x2, 2x1, 2x2, etc.), analyzes puzzle images with computer vision, finds minimum move solutions using BFS pathfinding, handles complex configurations with multiple empty spaces, validates move constraints and block collisions.Uses color detection to identify blocks, implements sophisticated move validation for physical constraints.SYNONYMS: sliding puzzle solver, block puzzle solver, wooden puzzle solver, sliding block game solver, puzzle pathfinding tool, block arrangement solver, sliding tile puzzle solver, wooden block game solver.EXAMPLES: 'Solve a 5*4 sliding block puzzle with nine blocks: one 2*2, four 1*2, two 2*1, two 1*1', 'Find minimum moves to solve this 4*4 sliding puzzle', 'Analyze and solve this wooden block puzzle configuration'.""",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text query containing puzzle specifications including grid size (e.g., '5*4'), block counts and sizes, and solution requirements"
                    },
                    "image_path": {
                        "type": "string", 
                        "description": "Path to an image file showing both start and end configurations side by side, with wooden blocks in various shades of brown and empty spaces in white"
                    }
                },
                "required": ["query", "image_path"],
                "additionalProperties": False,
            },
            strict=True,
            category="puzzle_solving",
            tags=["sliding_puzzle", "block_puzzle", "computer_vision", "pathfinding", "puzzle_solver", "wooden_puzzle", "sliding_blocks", "game_solving", "image_analysis", "bfs_algorithm"],
            limitation="Requires clear contrast between wooden blocks and empty spaces, grid dimensions must be explicitly stated, blocks can only move horizontally/vertically, computational complexity increases with puzzle size, image must show both start and end configurations side by side.",
            agent_type="Puzzle-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(query='Solve a 5*4 sliding block puzzle with nine blocks: one 2*2, four 1*2, two 2*1, two 1*1. Find the minimum moves to reach the target configuration.', image='wood_slide_0001.jpg')",
                "description": "Solve a 5*4 sliding block puzzle with nine blocks: one 2*2, four 1*2, two 2*1, two 1*1. Find the minimum moves to reach the target configuration."
            },
            require_llm_engine=True,
            llm_engine=llm_engine,
        )
        self.output = "dict - Returns a dictionary with keys: solution (list of str, moves in 'BlockID:Direction' format), total_move (int, number of moves required), initial_config (2D list, starting grid with block IDs and 'EMP' for empty), final_config (2D list, target grid). If no solution found, solution is None and total_move is 0."
        self.image_path = None
        self.grid_rows = None
        self.grid_cols = None
        self.start_config = None
        self.end_config = None
        self.blocks = {}
        self.debug = False

    def run(self, query: str, image_path: str):
        """Solve sliding block puzzle using computer vision and pathfinding algorithms."""
        try:
            self.image_path = image_path
            self.grid_rows, self.grid_cols = self.matrix_size(query=query)
            self.decode_image()
            solution = self.solve()
            
            if solution is None:
                return {"result": "No solution found for the sliding block puzzle.", "success": False}
            
            result = {
                "solution": solution, 
                "total_move": len(solution) if solution else 0, 
            }
            # Expose total_move at the top level so the generic test
            # harness can compare it directly against numeric answers.
            return {"result": result, "total_move": result["total_move"], "success": True}
        except Exception as e:
            return {"error": f"Error solving puzzle: {str(e)}", "success": False, "error_type": "woodslide_solver_failed", "traceback": traceback.format_exc()}
        
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

        
    def decode_image(self):
        """Decode the image into start and end configurations."""
        # Load image
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError(f"Could not load image from {self.image_path}")
            
        height, width = img.shape[:2]
        
        # Split image into two halves (start and end configurations)
        mid_x = width // 2
        start_img = img[:, :mid_x]
        end_img = img[:, mid_x:]
        
        # First decode the start configuration to establish color-to-block mapping
        self.start_config, color_to_block = self._decode_configuration(start_img, "start", None)
        
        # Then decode the end configuration using the same color-to-block mapping
        self.end_config, _ = self._decode_configuration(end_img, "end", color_to_block)
        
    def _decode_configuration(self, img: np.ndarray, config_type: str, color_to_block_map=None) -> Tuple[np.ndarray, Dict]:
        """
        Decode a single configuration from an image using a grid-based approach.
        Returns the configuration and the color-to-block mapping.
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Find the puzzle area
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Find the main puzzle rectangle
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest rectangular contour
        puzzle_rect = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:  # Rectangle
                    max_area = area
                    puzzle_rect = cv2.boundingRect(contour)
        
        if puzzle_rect is None:
            # Fallback: assume entire image is the puzzle
            puzzle_rect = (0, 0, img.shape[1], img.shape[0])
        
        x, y, w, h = puzzle_rect
        margin = 5
        puzzle_img = img[y+margin:y+h-margin, x+margin:x+w-margin]
        
        # Calculate cell dimensions
        cell_height = (h - 2*margin) // self.grid_rows
        cell_width = (w - 2*margin) // self.grid_cols
        
        # Extract cell colors first - use a larger sampling area to avoid grid lines
        cell_colors = np.zeros((self.grid_rows, self.grid_cols, 3))
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Get the center region of each cell - use larger area to get true color
                y1 = row * cell_height + cell_height // 3
                y2 = (row + 1) * cell_height - cell_height // 3
                x1 = col * cell_width + cell_width // 3
                x2 = (col + 1) * cell_width - cell_width // 3
                
                # Ensure bounds
                y1 = max(0, y1)
                y2 = min(puzzle_img.shape[0], y2)
                x1 = max(0, x1)
                x2 = min(puzzle_img.shape[1], x2)
                
                if y2 > y1 and x2 > x1:
                    cell_region = puzzle_img[y1:y2, x1:x2]
                    # Use median instead of mean to be more robust to outliers
                    cell_colors[row, col] = np.median(cell_region.reshape(-1, 3), axis=0)
        
        # Now group cells into blocks using improved color matching
        config = np.full((self.grid_rows, self.grid_cols), 'EMP', dtype=object)
        visited = np.zeros((self.grid_rows, self.grid_cols), dtype=bool)
        block_id = ord('A')
        
        # First pass: identify empty cells
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                avg_color = cell_colors[row, col]
                if np.all(avg_color > 240):  # Very light = empty
                    config[row, col] = 'EMP'
                    visited[row, col] = True
        
        # Second pass: group non-empty cells by color similarity
        # Use a more sophisticated approach that considers all cells, not just adjacent ones
        color_groups = []
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if visited[row, col]:
                    continue
                
                cell_color = cell_colors[row, col]
                
                # Check if this color matches any existing group
                matched_group = None
                for i, group in enumerate(color_groups):
                    group_color = group['color']
                    color_diff = np.linalg.norm(cell_color - group_color)
                    if color_diff < 5:  # Very tight threshold as requested
                        matched_group = i
                        break
                
                if matched_group is not None:
                    color_groups[matched_group]['cells'].append((row, col))
                else:
                    color_groups.append({
                        'color': cell_color,
                        'cells': [(row, col)]
                    })
        
        # Now assign block IDs to color groups
        # But we need to ensure spatially connected cells get the same ID
        block_id = ord('A')
        color_to_block = {}  # Map average color to block character
        
        for group in color_groups:
            # Split this color group into spatially connected components
            group_cells = group['cells']
            group_color = group['color']
            components = self._find_connected_components(group_cells)
            
            for component in components:
                # If we have a color mapping from start config, try to use it
                block_char = None
                if color_to_block_map is not None:
                    # Find the best matching color from the start configuration
                    best_match = None
                    best_diff = float('inf')
                    
                    for ref_color_tuple, ref_block in color_to_block_map.items():
                        ref_color = np.array(ref_color_tuple)
                        color_diff = np.linalg.norm(group_color - ref_color)
                        if color_diff < best_diff and color_diff < 5:  # Allow some tolerance
                            best_diff = color_diff
                            best_match = ref_block
                    
                    if best_match:
                        block_char = best_match
                
                # If no match found or this is the start config, assign new block ID
                if block_char is None:
                    block_char = chr(block_id)
                    block_id += 1
                
                # Store color mapping for this block
                color_key = tuple(group_color.astype(int))
                color_to_block[color_key] = block_char
                
                for r, c in component:
                    config[r, c] = block_char
                    visited[r, c] = True
                
                # Store block information
                if config_type == "start":
                    min_r = min(r for r, c in component)
                    max_r = max(r for r, c in component)
                    min_c = min(c for r, c in component)
                    max_c = max(c for r, c in component)
                    
                    self.blocks[block_char] = {
                        'size': (max_r - min_r + 1, max_c - min_c + 1),
                        'cells': component
                    }
                
                # if self.debug:
                #     print(f"Block {block_char}: {len(component)} cells at {component}")
        
        # Convert color_to_block keys from tuples to arrays for returning
        color_to_block_return = {}
        for color_tuple, block in color_to_block.items():
            color_to_block_return[color_tuple] = block
        
        return config, color_to_block_return
    
    def _find_connected_components(self, cells: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """
        Find connected components among the given cells.
        Two cells are connected if they are adjacent (horizontally or vertically).
        """
        if not cells:
            return []
        
        # Create adjacency information
        cell_set = set(cells)
        visited = set()
        components = []
        
        for cell in cells:
            if cell in visited:
                continue
            
            # BFS to find all cells connected to this one
            component = []
            queue = [cell]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                
                visited.add(current)
                component.append(current)
                
                # Check all adjacent cells
                row, col = current
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    neighbor = (row + dr, col + dc)
                    if neighbor in cell_set and neighbor not in visited:
                        queue.append(neighbor)
            
            if component:
                components.append(component)
        
        return components
    
    def solve(self) -> List[str]:
        """
        Solve the sliding block puzzle using BFS.
        
        Returns:
            List of moves to solve the puzzle
        """
        if self.start_config is None or self.end_config is None:
            raise ValueError("Must decode image first")
            
        # Convert configurations to hashable strings for BFS
        start_state = self._config_to_string(self.start_config)
        end_state = self._config_to_string(self.end_config)
        
        if start_state == end_state:
            return []
        
        # BFS to find solution
        queue = deque([(start_state, [])])
        visited = {start_state}
        max_iterations = 10000  # Prevent infinite search
        iteration = 0
        
        while queue and iteration < max_iterations:
            iteration += 1
            current_state, moves = queue.popleft()
            
            if current_state == end_state:
                return moves
                
            # Generate all possible moves
            current_config = self._string_to_config(current_state)
            
            for move in self._get_possible_moves(current_config):
                new_config = self._apply_move(current_config, move)
                new_state = self._config_to_string(new_config)
                
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, moves + [move]))
        
        if iteration >= max_iterations:
            print(f"Search stopped after {max_iterations} iterations")
                    
        return None  # No solution found
    
    def _config_to_string(self, config: np.ndarray) -> str:
        """Convert configuration to hashable string."""
        rows = []
        for row in config:
            # Join cells with comma delimiter
            row_str = ','.join(str(cell) for cell in row)
            rows.append(row_str)
        # Join rows with pipe delimiter
        return '|'.join(rows)
    
    def _string_to_config(self, state: str) -> np.ndarray:
        """Convert string back to configuration."""
        rows = state.split('|')
        # Create a proper 2D array with the correct shape
        config = np.empty((self.grid_rows, self.grid_cols), dtype=object)
        for i, row_str in enumerate(rows):
            cells = row_str.split(',')
            for j, cell in enumerate(cells):
                config[i, j] = cell
        return config
    
    def _get_possible_moves(self, config: np.ndarray) -> List[str]:
        """Get all possible moves from current configuration."""
        moves = []
        
        # Find all unique blocks
        processed_blocks = set()
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                block = config[row, col]
                
                if block != 'EMP' and block not in processed_blocks:
                    processed_blocks.add(block)
                    
                    # Find all cells of this block
                    block_cells = []
                    for r in range(self.grid_rows):
                        for c in range(self.grid_cols):
                            if config[r, c] == block:
                                block_cells.append((r, c))
                    
                    # Try moving in each direction
                    for direction in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                        if self._can_move(config, block_cells, direction):
                            moves.append(f"{block}:{direction}")
                            
        return moves
    
    def _can_move(self, config: np.ndarray, block_cells: List[Tuple[int, int]], 
                  direction: str) -> bool:
        """Check if a block can move in the given direction."""
        dr, dc = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }[direction]
        
        # Check each cell of the block
        for row, col in block_cells:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if new_row < 0 or new_row >= self.grid_rows or \
               new_col < 0 or new_col >= self.grid_cols:
                return False
                
            # Check if destination is empty or part of the same block
            dest = config[new_row, new_col]
            if dest != 'EMP' and (new_row, new_col) not in block_cells:
                return False
                
        return True
    
    def _apply_move(self, config: np.ndarray, move: str) -> np.ndarray:
        """Apply a move to the configuration."""
        # Create a proper deep copy of the 2D array
        new_config = np.empty((self.grid_rows, self.grid_cols), dtype=object)
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                new_config[i, j] = config[i, j]
        
        block_id, direction = move.split(':')
        
        # Find all cells of the block
        block_cells = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if config[row, col] == block_id:
                    block_cells.append((row, col))
        
        # Clear current positions
        for row, col in block_cells:
            new_config[row, col] = 'EMP'
        
        # Move block
        dr, dc = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }[direction]
        
        # Place block in new positions
        for row, col in block_cells:
            new_row, new_col = row + dr, col + dc
            new_config[new_row, new_col] = block_id
            
        return new_config
    
    def visualize_configuration(self, config: np.ndarray):
        """Print configuration in a readable format."""
        print("\nConfiguration:")
        print("+" + "-" * (self.grid_cols * 4 - 1) + "+")
        for row in config:
            print("|", end="")
            for cell in row:
                if cell == 'EMP':
                    print("   ", end="|")
                else:
                    print(f" {cell} ", end="|")
            print()
            print("+" + "-" * (self.grid_cols * 4 - 1) + "+")
    
    def test(self, tool_test: str="woodslide_solver", file_location: str="woodslide_solver", result_parameter: str="result", search_type: str="exact_match"):
        return super().test(tool_test=tool_test, file_location=file_location, result_parameter=result_parameter, search_type=search_type)
    


if __name__ == "__main__":
    try:
        tool = Woodslide_Solver_Tool()
        tool.embed_tool()
        # For evaluation we compare the predicted minimum number of moves
        # against the ground-truth moves in `data.json` using exact_match.
        tool.test(
            tool_test="woodslide_solver",
            file_location="woodslide_solver",
            result_parameter="total_move",
            search_type="exact_match",
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()