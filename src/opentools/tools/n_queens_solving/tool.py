import os, sys, re, cv2, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from opentools.core.base import BaseTool
from typing import List, Tuple, Optional

class N_Queens_Solving_Tool(BaseTool):
    """N_Queens_Solving_Tool
    ---------------------
    Purpose:
        A specialized chess puzzle solver that focuses on the N-Queens problem, combining computer vision and backtracking algorithms. It can detect existing queen positions from chessboard images, validate queen placements, and find optimal solutions for completing N-Queens configurations. The tool uses advanced image processing to identify chess pieces, particularly queens with crown shapes, and implements Manhattan distance calculations for position analysis.

    Core Capabilities:
        - Solves N-Queens puzzle using backtracking algorithms
        - Detects queen positions from chessboard images using computer vision
        - Calculates Manhattan distance between queen positions
        - Validates queen placements for conflict-free solutions
        - Provides complete board configurations and remaining queen positions

    Intended Use:
        Use this tool when you need to solve the N-Queens puzzle, including finding optimal solutions for completing N-Queens configurations.

    Limitations:
        - May not handle complex N-Queens configurations
    """
    # Default args for `opentools test N_Queens_Solving_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "n_queens_solving",
        "file_location": "n_queens_solving",
        "result_parameter": "result",
        "search_type": "exact_match",
    }
    require_llm_engine = True
    def __init__(self, model_string="gpt-4o-mini", llm_engine=None):
        super().__init__(
            type='function',
            name="N_Queens_Solving_Tool",
            description="""A specialized chess puzzle solver that focuses on the N-Queens problem, combining computer vision and backtracking algorithms. It can detect existing queen positions from chessboard images, validate queen placements, and find optimal solutions for completing N-Queens configurations. The tool uses advanced image processing to identify chess pieces, particularly queens with crown shapes, and implements Manhattan distance calculations for position analysis. CAPABILITIES: Solves N-Queens puzzle using backtracking algorithms, detects queen positions from chessboard images using computer vision, calculates Manhattan distance between queen positions, validates queen placements for conflict-free solutions, provides complete board configurations and remaining queen positions. SYNONYMS: N-Queens solver, chess puzzle solver, queen placement solver, backtracking algorithm tool, chess piece detector, computer vision chess solver, queen conflict solver, chessboard analyzer. EXAMPLES: 'Find Manhattan distance between two remaining queens in 8 * 8 board with 6 queens placed', 'Complete N-Queens solution for 6 * 6 board with 4 queens already placed', 'Solve the N-Queens puzzle from this chessboard image'.""",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text query containing chessboard dimensions (e.g., '8 * 8') and specific requirements about queen placements and distance calculations"
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Path to an image file showing a chessboard with existing queen pieces, where queens should be visibly distinct with crown shapes"
                    }
                },
                "required": ["query", "image_path"],
                "additionalProperties": False,
            },
            strict=True,
            category="puzzle_solving",
            tags=["n_queens", "chess_puzzle", "backtracking",    "queen_detection", "manhattan_distance", "chess_solver", "puzzle_solver", "algorithm"],
            limitation="Requires clear images with distinguishable queen pieces, queens must have visible crown shapes for detection, board dimensions must be explicitly stated in query, only handles standard N-Queens configurations, limited to square chessboards, cannot handle rotated or skewed board images, requires consistent lighting across the board, may struggle with non-standard chess piece designs, board must have clear black and white squares",
            agent_type="Puzzle-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(query='8 * 8', image_path='test.jpg')",
                "description": "Solve a 8 * 8 chessboard with 6 queens placed"
            },
            require_llm_engine=True,
            llm_engine=llm_engine,
        )
        self.board = None
        self.rows = None
        self.cols = None

    def run(self, query: str, image_path: str) :
        solution = self.solve_chess_problem(image_path, query)
        return solution
        
    def decode_board_from_image(self, image_path: str) -> List[List[str]]:
        """
        Decode a chess board image into a 2D matrix
        Returns matrix with '.' for empty squares and 'Q' for queens
        """        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
        # First, detect the actual chessboard boundaries
        board_region = self._detect_chessboard_region(img_rgb)
        if board_region is None:
            raise ValueError("Could not detect chessboard boundaries")
            
        x1, y1, x2, y2 = board_region
        cropped_board = img_rgb[y1:y1+y2, x1:x2+x1]
        
        # Now calculate cell dimensions from the cropped board
        height, width = cropped_board.shape[:2]
        cell_height = height // self.rows
        cell_width = width // self.cols
                
        # Initialize board
        board = [['.' for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Analyze each cell in the cropped board
        for row in range(self.rows):
            for col in range(self.cols):

                # Extract cell region from cropped board
                y1_cell = row * cell_height
                y2_cell = (row + 1) * cell_height
                x1_cell = col * cell_width
                x2_cell = (col + 1) * cell_width
                
                cell = cropped_board[y1_cell:y2_cell, x1_cell:x2_cell]
                
                # Check if cell contains a queen
                has_queen = self._detect_queen_in_cell(cell)
                if has_queen:
                    board[row][col] = 'Q'
                    print(f"Queen detected at ({row},{col})")
        
        self.board = board
        return board
    
    def _detect_queen_in_cell(self, cell: np.ndarray) -> bool:
        """
        Detect if a cell contains a queen piece - improved for crown detection
        """
        if cell.size == 0:
            return False
            
        # Convert to grayscale
        gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
        
        # Look for crown-like shapes (multiple peaks at top)
        crown_detected = self._detect_crown_shape(gray)
        # Look for dark outlined shapes
        outline_detected = self._detect_dark_outline(gray)
        # Check for significant contrast variation
        contrast_detected = self._detect_contrast_variation(gray)
        # Combine methods - queen likely if any strong indicator
        is_queen = crown_detected or outline_detected or contrast_detected
        
        return is_queen
    
    def _detect_crown_shape(self, gray: np.ndarray) -> bool:
        """
        Look for crown-like pattern with multiple peaks
        """
        if gray.shape[0] < 5 or gray.shape[1] < 5:
            return False
            
        # Look at the top portion for crown peaks
        top_portion = gray[:gray.shape[0]//2, :]
        
        # Find edges in the top portion
        edges = cv2.Canny(top_portion, 30, 100)
        
        # Look for multiple connected components (crown peaks)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count significant contours in top area
        significant_contours = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:  # Minimum area threshold
                significant_contours += 1
        
        return significant_contours >= 2  # Crown should have multiple peaks
    
    def _detect_dark_outline(self, gray: np.ndarray) -> bool:
        """
        Look for dark outlines typical of chess pieces
        """
        if gray.size == 0:
            return False
            
        # Find edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Also check for dark pixels (outline)
        mean_intensity = np.mean(gray)
        dark_threshold = mean_intensity * 0.6
        dark_pixels = np.sum(gray < dark_threshold)
        dark_ratio = dark_pixels / gray.size
        
        return edge_density > 0.05 or dark_ratio > 0.15
    
    def _detect_contrast_variation(self, gray: np.ndarray) -> bool:
        """
        Look for significant contrast variation indicating a piece
        """
        if gray.size == 0:
            return False
            
        # Standard deviation indicates variation
        std_dev = np.std(gray)
        
        # Also check intensity range
        intensity_range = np.max(gray) - np.min(gray)
        
        return std_dev > 25 or intensity_range > 80
    
    
    def _detect_chessboard_region(self, img_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the actual chessboard boundaries within the image
        Returns (x1, y1, x2, y2) coordinates of the chessboard
        """
        # Convert to grayscale for better edge detection
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Look for the largest square/rectangular region with grid pattern
        # Find edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for the largest rectangular contour
        largest_area = 0
        best_contour = None
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's rectangular (4 corners)
            if len(approx) >= 4:
                area = cv2.contourArea(contour)
                if area > largest_area:
                    largest_area = area
                    best_contour = contour
        
        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            return (x, y, x + w, y + h)
        return None
       
     
    def get_queen_positions(self) -> List[Tuple[int, int]]:
        """Get positions of all queens on the board"""
        positions = []
        if self.board is None:
            return positions
            
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row][col] == 'Q':
                    positions.append((row, col))
        return positions
    
    def is_safe(self, row: int, col: int, existing_queens: List[Tuple[int, int]]) -> bool:
        """
        Check if placing a queen at (row, col) is safe given existing queens
        """
        for queen_row, queen_col in existing_queens:
            # Check row conflict
            if queen_row == row:
                return False
            # Check column conflict
            if queen_col == col:
                return False
            # Check diagonal conflict
            if abs(queen_row - row) == abs(queen_col - col):
                return False
        return True
    
    def solve_nqueens(self) -> List[Tuple[int, int]]:
        """
        Solve the N-Queens problem for the current board state
        Returns positions where remaining queens should be placed
        """
        if self.board is None:
            raise ValueError("No board loaded")
        
        existing_queens = self.get_queen_positions()
        empty_rows = []
        
        # Find rows that don't have queens
        occupied_rows = {pos[0] for pos in existing_queens}
        for row in range(self.rows):
            if row not in occupied_rows:
                empty_rows.append(row)
        
        if len(existing_queens) + len(empty_rows) != self.cols:
            raise ValueError("Invalid board state - cannot place exactly one queen per row")
        
        # Find valid positions for remaining queens
        remaining_positions = []
        
        def backtrack(row_idx: int, current_positions: List[Tuple[int, int]]):
            if row_idx == len(empty_rows):
                remaining_positions.extend(current_positions)
                return True
            
            row = empty_rows[row_idx]
            all_queens = existing_queens + current_positions
            
            for col in range(self.cols):
                if self.is_safe(row, col, all_queens):
                    current_positions.append((row, col))
                    if backtrack(row_idx + 1, current_positions):
                        return True
                    current_positions.pop()
            
            return False
        
        backtrack(0, [])
        return remaining_positions
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def solve_and_get_distance(self) -> Optional[int]:
        """
        Solve the N-Queens problem and return Manhattan distance between
        the two newly placed queens (if exactly 2 are missing)
        """
        try:
            remaining_positions = self.solve_nqueens()
            
            if len(remaining_positions) == 2:
                return self.manhattan_distance(remaining_positions[0], remaining_positions[1])
            else:
                print(f"Expected 2 remaining queens, found {len(remaining_positions)}")
                return None
                
        except Exception as e:
            print(f"Error solving: {e}")
            return None

            
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
    
    def solve_chess_problem(self, image_path: str , query: str ):
        """
        Main function to solve chess N-Queens problem
        """
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(__file__), image_path)
        self.rows, self.cols =  self.matrix_size(query=query)
        try:
            board = self.decode_board_from_image(image_path)
        except Exception as e:
            print(f"Error decoding image: {e}")
            return {"success": False, "error": "error_decoding_image"}

            
        # Solve and get distance
        distance = self.solve_and_get_distance()
        
        if distance is not None:
            print(f"Manhattan distance between the two remaining queens: {distance}")
            
            # Show the complete solution
            remaining_positions = self.solve_nqueens()
            if remaining_positions:
                print(f"Remaining queens should be placed at: {remaining_positions}")
                
                # Create complete board
                complete_board = [row[:] for row in self.board]  # Deep copy
                for row, col in remaining_positions:
                    complete_board[row][col] = 'Q'
                
                print("Complete solution:")
                for row in complete_board:
                    print(' '.join(row))
        analysis = {"result": distance, "remaining_positions": remaining_positions, "success": True}
        return analysis
    def test(
            self,
            tool_test: str = "n_queens_solving",
            file_location: str = "n_queens_solving",
            result_parameter: str = "result",
            search_type: str = "exact_match",
        ):
            """Run the base tool test with text_detector-specific defaults."""
            return super().test(
                tool_test=tool_test,
                file_location=file_location,
                result_parameter=result_parameter,
                search_type=search_type,
            )

# Test with the example problem
if __name__ == "__main__":
    tool = N_Queens_Solving_Tool()
    tool.embed_tool()
    tool.test(tool_test="n_queens_solving", file_location="n_queens_solving", result_parameter="result", search_type="exact_match")