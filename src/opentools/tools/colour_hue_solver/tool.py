import os, sys, cv2, io, re, traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from opentools.core.base import BaseTool
import numpy as np
from PIL import Image
class Colour_Hue_Solver_Tool(BaseTool):
    """Colour_Hue_Solver_Tool
    ---------------------
    Purpose:
        An advanced image processing tool that analyzes and solves color tile arrangement puzzles for both 1D and 2D boards. Uses computer vision and color space analysis to detect, compare, and optimize tile arrangements between two boards. The tool employs multiple sampling points and various color features to ensure accurate color matching, and uses the Hungarian algorithm to find the optimal mapping of tiles while calculating the minimum number of swaps needed.

    Core Capabilities:
        - Computer vision image analysis
        - Color space analysis (BGR, HSV, LAB)
        - Tile detection and extraction
        - Optimal tile mapping using Hungarian algorithm
        - Minimum swap calculation
        - Support for 1D and 2D grid layouts
        - Multiple sampling point color extraction
        - Cycle decomposition analysis
        - Comprehensive puzzle solving with detailed output

    Intended Use:
        Use this tool when you need to solve color tile arrangement puzzles, particularly for both 1D and 2D boards. It is particularly useful for finding the optimal mapping of tiles while calculating the minimum number of swaps needed.

    Limitations:
        - Requires clear images with distinguishable colored tiles
        - Board dimensions must be explicitly stated in query
        - Only handles 1D and 2D grid layouts
        - May not handle complex color tile arrangements
    """
    # Default args for `opentools test Colour_Hue_Solver_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "colour_hue_solver",
        "file_location": "colour_hue_solver",
        "result_parameter": "result",
        "search_type": "similarity_eval",
    }
    require_llm_engine = True
    def __init__(self, model_string="gpt-4o-mini", llm_engine=None):
        super().__init__(
            type='function',
            name="Colour_Hue_Solver_Tool",
            description="""An advanced image processing tool that analyzes and solves color tile arrangement puzzles for both 1D and 2D boards. Uses computer vision and color space analysis to detect, compare, and optimize tile arrangements between two boards. The tool employs multiple sampling points and various color features to ensure accurate color matching, and uses the Hungarian algorithm to find the optimal mapping of tiles while calculating the minimum number of swaps needed. CAPABILITIES: Computer vision image analysis, color space analysis (BGR, HSV, LAB), tile detection and extraction, optimal tile mapping using Hungarian algorithm, minimum swap calculation, support for 1D and 2D grid layouts, multiple sampling point color extraction, cycle decomposition analysis, comprehensive puzzle solving with detailed output. SYNONYMS: color tile solver, tile arrangement puzzle solver, color matching tool, tile swap calculator, color puzzle solver, board tile solver, color arrangement optimizer, tile mapping tool, color space analyzer, puzzle optimization tool. EXAMPLES: 'Solve this 3x4 color tile puzzle', 'Calculate minimum swaps for this tile arrangement', 'Find optimal mapping between two color boards', 'Solve color tile puzzle with 12 tiles'.""",
            parameters={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to an image file containing two side-by-side boards with colored tiles. Each board should have distinct colored tiles arranged in a grid pattern."
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional query string that may contain board dimensions (e.g., '3 * 4 board' for a 3x4 grid)"
                    }
                },
                "required": ["image_path"],
                "additionalProperties": False,
            },
            strict=False,
            category="image_processing",
            tags=["color_analysis", "puzzle_solving", "computer_vision", "tile_mapping", "color_matching", "image_processing", "algorithm", "optimization", "puzzle_tool", "color_solver"],
            limitation="Requires clear separation between the two boards, colors must be distinguishable and consistent, sensitive to lighting conditions, may struggle with similar or gradient colors, requires white background for board detection, colors should be solid without patterns, performance affected by glare or shadows, grid detection may fail with irregular tile spacing",
            agent_type="Puzzle-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {    
                "command": "reponse = tool.run(image_path='test.jpg', query='3 * 4 board')",    
                "description": "Solve a 3 * 4 board"
            },
            require_llm_engine=True,
            llm_engine=llm_engine,
        )
        
    def extract_board_dimensions(self, query):
        """Extract board dimensions from query string"""
        if query is None:
            return 1, 4  # Default to 1x4 (single row, 4 tiles)
        
        # Look for patterns like "3 * 4 board", "3x4 board", "3 × 4", etc.
        patterns = [
            r'(\d+)\s*[*×xX]\s*(\d+)\s*board',
            r'board.*?(\d+)\s*[*×xX]\s*(\d+)',
            r'(\d+)\s*by\s*(\d+)\s*board',
            r'(\d+)\s*rows?\s*and\s*(\d+)\s*columns?',
            r'(\d+)\s*rows?\s*[*×xX]\s*(\d+)\s*columns?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                rows, cols = int(match.group(1)), int(match.group(2))
                return rows, cols
        return 1, 4

        
    def detect_grid_structure(self, board_img, rows, cols):
        """Detect grid structure and extract individual tiles"""
        h, w = board_img.shape[:2]
        
        tile_height = h // rows
        tile_width = w // cols
        
        tiles = []
        positions = []
        
        for row in range(rows):
            for col in range(cols):
                y1 = row * tile_height
                y2 = (row + 1) * tile_height
                x1 = col * tile_width
                x2 = (col + 1) * tile_width
                
                tile = board_img[y1:y2, x1:x2]
                tiles.append(tile)
                positions.append((row, col))
        return tiles, positions
    
    def extract_tile_color(self, tile):
        """Extract representative color from a single tile"""
        h, w = tile.shape[:2]
        
        # Sample multiple points to get better color representation
        samples = []
        
        # Define sampling regions (avoid edges)
        margin = min(h, w) // 8
        regions = [
            (margin, h//2, margin, w//2),                    # top-left
            (margin, h//2, w//2, w-margin),                  # top-right
            (h//2, h-margin, margin, w//2),                  # bottom-left
            (h//2, h-margin, w//2, w-margin),                # bottom-right
            (h//3, 2*h//3, w//3, 2*w//3)                     # center
        ]
        
        for y1, y2, x1, x2 in regions:
            if y2 > y1 and x2 > x1:  # Valid region
                region = tile[y1:y2, x1:x2]
                if region.size > 0:
                    # Get mean color of the region
                    mean_color = np.mean(region.reshape(-1, 3), axis=0)
                    samples.append(mean_color)
        
        # Use median of samples for robustness
        if samples:
            tile_color = np.median(samples, axis=0)
        else:
            # Fallback to center pixel
            tile_color = tile[h//2, w//2]
        
        return tile_color
    
    def process_tile_board_image(self, image_data, rows=1, cols=4):
        try:
            if not os.path.isabs(image_data):
                image_data = os.path.join(os.path.dirname(__file__), image_data)
            image_data = cv2.imread(image_data)
            # Convert image data to numpy array
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                image = image_data
                    
            # Convert to grayscale to find colored regions
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Threshold to find non-white areas (the colored boards)
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours of the boards
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get the two largest contours (should be our boards)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            
            # Extract bounding boxes and sort by x-coordinate (left to right)
            boxes = [cv2.boundingRect(c) for c in contours]
            boxes.sort(key=lambda x: x[0])
            
            # Extract board A (left) and board B (right)
            board_a = image[boxes[0][1]:boxes[0][1]+boxes[0][3], boxes[0][0]:boxes[0][0]+boxes[0][2]]
            board_b = image[boxes[1][1]:boxes[1][1]+boxes[1][3], boxes[1][0]:boxes[1][0]+boxes[1][2]]

            
            # Step 2: Extract tiles and colors from each board
            tiles_a, positions_a = self.detect_grid_structure(board_a, rows, cols)
            tiles_b, positions_b = self.detect_grid_structure(board_b, rows, cols)
            
            print("\nExtracting colors from Board A:")
            colors_a = []
            for i, tile in enumerate(tiles_a):
                color = self.extract_tile_color(tile)
                colors_a.append(color)
                row, col = positions_a[i]
                print(f"  Tile ({row},{col}): RGB({int(color[2])}, {int(color[1])}, {int(color[0])})")
            
            print("\nExtracting colors from Board B:")
            colors_b = []
            for i, tile in enumerate(tiles_b):
                color = self.extract_tile_color(tile)
                colors_b.append(color)
                row, col = positions_b[i]
                print(f"  Tile ({row},{col}): RGB({int(color[2])}, {int(color[1])}, {int(color[0])})")
            
            # Step 3: Create bijective mapping using Hungarian algorithm
            def create_optimal_mapping(colors_from, colors_to):
                """
                Create a one-to-one mapping between tiles using optimal assignment
                """
                n = len(colors_from)
                
                # Create cost matrix using color distances
                cost_matrix = np.zeros((n, n))
                
                for i in range(n):
                    for j in range(n):
                        # Calculate color distance in LAB color space
                        color_from_rgb = np.array([colors_from[j][2], colors_from[j][1], colors_from[j][0]]).reshape(1, 1, 3).astype(np.uint8)
                        color_to_rgb = np.array([colors_to[i][2], colors_to[i][1], colors_to[i][0]]).reshape(1, 1, 3).astype(np.uint8)
                        
                        color_from_lab = cv2.cvtColor(color_from_rgb, cv2.COLOR_RGB2LAB)
                        color_to_lab = cv2.cvtColor(color_to_rgb, cv2.COLOR_RGB2LAB)
                        
                        # Euclidean distance in LAB space
                        cost_matrix[i, j] = np.linalg.norm(color_from_lab - color_to_lab)
                
                # For larger boards, use scipy's implementation if available
                try:
                    from scipy.optimize import linear_sum_assignment
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    mapping = [0] * n
                    for i, j in zip(row_ind, col_ind):
                        mapping[i] = j
                except ImportError:
                    # Fallback for small boards - try all permutations
                    if n <= 8:  # Reasonable limit for permutation approach
                        mapping = list(range(n))
                        min_cost = float('inf')
                        
                        from itertools import permutations
                        for perm in permutations(range(n)):
                            cost = sum(cost_matrix[i, perm[i]] for i in range(n))
                            if cost < min_cost:
                                min_cost = cost
                                mapping = list(perm)
                    else:
                        # For large boards without scipy, use greedy approach
                        mapping = []
                        used = set()
                        for i in range(n):
                            best_j = -1
                            best_cost = float('inf')
                            for j in range(n):
                                if j not in used and cost_matrix[i, j] < best_cost:
                                    best_cost = cost_matrix[i, j]
                                    best_j = j
                            mapping.append(best_j)
                            used.add(best_j)
                
                return mapping
            
            # Create mapping: mapping[i] = j means tile j from A goes to position i in B
            mapping = create_optimal_mapping(colors_a, colors_b)
            print(f"\nColor mapping (A->B positions): {mapping}")
            
            # Step 4: Calculate minimum swaps using cycle decomposition
            def count_cycles(permutation):
                n = len(permutation)
                visited = [False] * n
                num_cycles = 0
                
                for i in range(n):
                    if not visited[i]:
                        # Trace the cycle
                        j = i
                        while not visited[j]:
                            visited[j] = True
                            j = permutation[j]
                        num_cycles += 1
                
                return num_cycles
            
            # Create inverse permutation for swap calculation
            inverse_perm = [0] * len(mapping)
            for i, j in enumerate(mapping):
                inverse_perm[j] = i
            
            num_cycles = count_cycles(inverse_perm)
            min_swaps = len(mapping) - num_cycles
            
            print(f"\nNumber of cycles: {num_cycles}")
            print(f"Minimum swaps needed: {min_swaps}")
            
            # Visualize the mapping in grid format
            print("\nMapping visualization:")
            print("Board A -> Board B positions:")
            for row in range(rows):
                row_str = ""
                for col in range(cols):
                    idx = row * cols + col
                    target_idx = mapping[idx]
                    target_row, target_col = target_idx // cols, target_idx % cols
                    row_str += f"({row},{col})->({target_row},{target_col})  "
                print(f"  {row_str}")
            
            # Detailed cycle analysis
            print("\nSwap operations needed:")
            visited = [False] * len(inverse_perm)
            swap_count = 0
            analysis = {}
            analysis['result'] = min_swaps
            
            # Extract answer from query if it exists
            if hasattr(self, 'current_query') and self.current_query:
                answer_match = re.search(r'([A-D])\.\s*(\d+)', self.current_query)
                if answer_match and int(answer_match.group(2)) == min_swaps:
                    analysis['Answer'] = answer_match.group(1)
            
            cycle_details = []
            for i in range(len(inverse_perm)):
                if not visited[i] and inverse_perm[i] != i:
                    cycle = []
                    j = i
                    while not visited[j]:
                        visited[j] = True
                        cycle.append(j)
                        j = inverse_perm[j]
                    if len(cycle) > 1:
                        # Convert linear indices to grid positions for better visualization
                        cycle_positions = []
                        for idx in cycle:
                            row, col = idx // cols, idx % cols
                            cycle_positions.append(f"({row},{col})")
                        
                        cycle_str = ' → '.join(cycle_positions) + f" → {cycle_positions[0]}"
                        print(f"  Cycle {swap_count + 1}: {cycle_str} (requires {len(cycle)-1} swaps)")
                        cycle_details.append(f"{cycle_str} (requires {len(cycle)-1} swaps)")
                        swap_count += 1
            analysis['success'] = True
            return analysis
        except Exception as e:
            return {"error": f"Error: An unexpected error occurred: {str(e)}", "success": False, "traceback": traceback.format_exc()}
        
    def run(self, image_path, query=None):
        """Read image from file and solve the puzzle"""
        # Store query for later use
        self.current_query = query
        
        # Extract board dimensions from query
        rows, cols = self.extract_board_dimensions(query)
        
        return self.process_tile_board_image(image_path, rows, cols)

    def test(self, tool_test: str="colour_hue_solver", file_location: str="colour_hue_solver", result_parameter: str="result", search_type: str="exact_match"):
        return super().test(tool_test=tool_test, file_location=file_location, result_parameter=result_parameter, search_type=search_type)

# Example usage:
if __name__ == "__main__":
    print("Enhanced Tile Board Solver with M×N Grid Support")
    print("=" * 50)

    tool = Colour_Hue_Solver_Tool()
    tool.embed_tool()
    tool.test(tool_test="colour_hue_solver", file_location="colour_hue_solver", result_parameter="result", search_type='exact_match')