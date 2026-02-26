import os, sys, traceback, cv2, re, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from opentools.core.base import BaseTool
import numpy as np
from typing import Optional
class Rubik_Cube_Solver_Tool(BaseTool):
    """Rubik_Cube_Solver_Tool
    ---------------------
    Purpose:
        A sophisticated Rubik's Cube analysis tool that combines computer vision and cube manipulation algorithms. Detects and tracks the state of 3x3 Rubik's Cubes from images, processes standard cube notation (U, D, L, R, F, B), and simulates cube rotations with precise mechanics. CAPABILITIES: Analyzes Rubik's cube images using computer vision, detects cube faces and colors from unfolded layouts, processes standard cube notation and move sequences, simulates cube rotations and tracks state changes, counts specific colored squares on cube faces after move sequences.

    Core Capabilities:
        - Computer vision cube analysis
        - Cube face and color detection
        - Standard cube notation processing
        - Cube rotation simulation
        - State tracking and change detection
        - Color counting after move sequences

    Intended Use:
        Use this tool when you need to analyze a Rubik's cube, including detecting cube faces and colors, processing standard cube notation and move sequences, simulating cube rotations, tracking state changes, and counting specific colored squares after move sequences.

    Limitations:
        - May not handle complex cube configurations or move sequences
    """
    # Default args for `opentools test Rubik_Cube_Solver_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "rubik_cube_solver",
        "file_location": "rubik_cube_solver",
        "result_parameter": "result",
        "search_type": "similarity_eval",
    }
    FACE_MAP = {
        'UP': 'U', 'TOP': 'U', 'UPPER': 'U', 'U': 'U',
        'DOWN': 'D', 'BOTTOM': 'D', 'LOWER': 'D', 'D': 'D',
        'LEFT': 'L', 'L': 'L',
        'RIGHT': 'R', 'R': 'R',
        'FRONT': 'F', 'F': 'F',
        'BACK': 'B', 'B': 'B'
    }

    COLOR_ALIASES = {
        'gray': 'grey',
        'grey': 'grey',
        'silver': 'grey',
        'white': 'grey'
    }
    require_llm_engine = True
    def __init__(self, model_string="gpt-4o-mini", llm_engine=None ):
        super().__init__(
            type='function',
            name="Rubik_Cube_Solver_Tool",
            description="""A sophisticated Rubik's Cube analysis tool that combines computer vision and cube manipulation algorithms. Detects and tracks the state of 3x3 Rubik's Cubes from images, processes standard cube notation (U, D, L, R, F, B), and simulates cube rotations with precise mechanics. CAPABILITIES: Analyzes Rubik's cube images using computer vision, detects cube faces and colors from unfolded layouts, processes standard cube notation and move sequences, simulates cube rotations and tracks state changes, counts specific colored squares on cube faces after move sequences. SYNONYMS: Rubik's cube solver, cube state analyzer, cube rotation simulator, cube color counter, cube move processor, cube state tracker, cube notation parser, cube face analyzer, cube manipulation tool, cube state calculator. EXAMPLES: 'Count blue squares in the front face after U R2 B', 'Count grey squares in the up face after D2 U2', 'Analyze cube state after move sequence L R F'.""",
            parameters={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to an image showing a Rubik's cube unfolded layout with all six faces visible (Up, Down, Left, Right, Front, Back)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Text query containing cube rotation instructions using standard notation (U, D, L, R, F, B) and specific questions about the resulting cube state. MUST follow format for moving sequence: 'after [Moves]' and explicitly mention the target color and face."
                    }
                },
                "required": ["image_path", "query"],
                "additionalProperties": False,
            },
            strict=True,
            category="puzzle_solving",
            tags=["rubiks_cube", "cube_solver", "computer_vision", "cube_analysis", "cube_rotation", "puzzle_solver", "cube_state_tracker", "cube_notation", "color_detection", "cube_manipulation"],
            limitation="Requires unfolded cube layout showing all six faces clearly, limited to standard 3x3 Rubik's cube, needs clear color distinction between faces, sensitive to lighting and shadows, cannot handle partially visible faces, query must follow strict format for moving sequence: 'after [Moves]', move sequences must use standard notation: U, D, L, R, F, B with optional 2, 3, ' modifiers, colors must be standard Rubik's cube colors: red, green, blue, yellow, orange, grey, and the query must explicitly mention the target color and face to count.",
            agent_type="Puzzle-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(query='U R2 B', image_path='test.jpg')",
                "description": "Solve a Rubik's cube puzzle with the move sequence U R2 B"
            },
            require_llm_engine=True,
            llm_engine=llm_engine,
        )
        self.image_path = None
        self.matrix = {}

    
        
    def run(self, query: str, image_path: str) -> str:
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(__file__), image_path)
        self.image_path = image_path
        try:
            self.decode_image()
            result = self.solve_question(query)
            if isinstance(result, dict):
                if result.get("success") is False:
                    result.setdefault("matrix_color", self.matrix)
                    return result
                if result.get("success") is True and "result" in result:
                    result.setdefault("matrix_color", self.matrix)
                    return result
            sols = {'result': result, 'matrix_color': self.matrix, 'success': True}
            return sols
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "matrix_color": self.matrix or {}
            }

    def decode_image(self):
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError(f"Could not load the image from {self.image_path}")
        
        # Convert to RGB for consistent color processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img_rgb.shape[:2]
        
        # Improved face segmentation with larger margins to avoid borders
        face_width = width // 4
        face_height = height // 3
        
        # Increase margins to avoid border areas that might appear grey
        margin = 25
        
        # Define face regions more precisely
        matrix = {
            # U face: top center
            "U": img_rgb[margin:face_height-margin, 
                        face_width+margin:2*face_width-margin],
            
            # L face: middle left
            "L": img_rgb[face_height+margin:2*face_height-margin, 
                        margin:face_width-margin],
            
            # F face: middle center
            "F": img_rgb[face_height+margin:2*face_height-margin, 
                        face_width+margin:2*face_width-margin],
            
            # R face: middle right
            "R": img_rgb[face_height+margin:2*face_height-margin, 
                        2*face_width+margin:3*face_width-margin],
            
            # B face: middle far right (with extra margin to avoid edge effects)
            "B": img_rgb[face_height+margin:2*face_height-margin, 
                        3*face_width+margin*2:4*face_width-margin*2],
            
            # D face: bottom center
            "D": img_rgb[2*face_height+margin:3*face_height-margin, 
                        face_width+margin:2*face_width-margin],
        }
        
        for name, face in matrix.items():
            if face.size > 0:  # Check if face region is valid
                self.matrix[name] = self.parse_image_to_matrix(face)

                
        # Store original state for reference
                
    def parse_image_to_matrix(self, face):
        if face.size == 0:
            return []
            
        # Create a copy to avoid modifying original
        face_copy = face.copy()
        
        # Apply Gaussian blur to reduce noise
        face_blur = cv2.GaussianBlur(face_copy, (5, 5), 0)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(face_blur, cv2.COLOR_RGB2HSV)
        
        # Use adaptive thresholding to find grid lines
        gray = cv2.cvtColor(face_blur, cv2.COLOR_RGB2GRAY)
        
        # Apply morphological operations to enhance grid lines
        kernel = np.ones((3,3), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Use adaptive threshold instead of Canny
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If contour detection fails, use the entire face
        if not contours:
            face_img = face_copy
        else:
            # Find the largest rectangular contour
            max_area = 0
            best_rect = None
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) >= 4:  # At least 4 points for a quadrilateral
                        max_area = area
                        best_rect = cv2.boundingRect(contour)
            
            if best_rect:
                x, y, w, h = best_rect
                face_img = face_copy[y:y+h, x:x+w]
            else:
                # Fall back to using the entire face
                face_img = face_copy
        
        # Extract colors from 3x3 grid
        h, w = face_img.shape[:2]
        tile_h, tile_w = h // 3, w // 3
        
        colors = []
        for i in range(3):
            row_colors = []
            for j in range(3):
                # Sample from the center of each tile
                cy = int((i + 0.5) * tile_h)
                cx = int((j + 0.5) * tile_w)
                
                # Ensure we don't go out of bounds
                cy = min(cy, h-1)
                cx = min(cx, w-1)
                
                # Sample a small area around the center point for better accuracy
                sample_size = min(tile_h, tile_w) // 4
                y_start = max(0, cy - sample_size)
                y_end = min(h, cy + sample_size)
                x_start = max(0, cx - sample_size)
                x_end = min(w, cx + sample_size)
                
                # Get average color in the sample area
                sample_region = face_img[y_start:y_end, x_start:x_end]
                avg_color = np.mean(sample_region, axis=(0, 1))
                
                color_name = self.closest_rubik_color(avg_color)
                row_colors.append(color_name)
            colors.append(row_colors)
        
        return colors
    
    def closest_rubik_color(self, rgb):
        # Define standard Rubik's cube colors with more accurate RGB values
        rubik_colors = {
            'grey': [200, 200, 200],    # Lower threshold for white to catch grey tiles
            'yellow': [255, 215, 0],     # Gold-yellow
            'red': [220, 20, 20],        # Bright red
            'orange': [255, 140, 0],     # Orange
            'blue': [0, 70, 173],        # Royal blue
            'green': [0, 155, 72]        # Forest green
        }
        
        best_name = None
        best_dist = float('inf')
        
        for name, color in rubik_colors.items():
            # Calculate Euclidean distance in RGB space
            dist = np.sqrt(np.sum((rgb - np.array(color))**2))
            if dist < best_dist:
                best_dist = dist
                best_name = name
        
        # Special handling for white/grey detection
        r, g, b = rgb
        # If all RGB values are close to each other and relatively high, it's likely white
        if abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30 and min(r, g, b) > 150:
            best_name = 'grey'
                
        return best_name

    def rotate_face_clockwise(self, face_matrix):
        """Rotate a 3x3 face matrix 90 degrees clockwise"""
        if not face_matrix or len(face_matrix) != 3:
            return face_matrix
        
        # Transpose and reverse rows for 90-degree clockwise rotation
        rotated = [[face_matrix[2-j][i] for j in range(3)] for i in range(3)]
        return rotated

    def rotate_face_counterclockwise(self, face_matrix):
        """Rotate a 3x3 face matrix 90 degrees counterclockwise"""
        if not face_matrix or len(face_matrix) != 3:
            return face_matrix
        
        # Reverse rows and transpose for 90-degree counterclockwise rotation
        rotated = [[face_matrix[j][2-i] for j in range(3)] for i in range(3)]
        return rotated

    def execute_move(self, move, n=1):
        """Execute a single move n times"""
        for _ in range(n % 4):  # Only need to do up to 3 rotations
            if move == 'U':
                self.move_U()
            elif move == 'D':
                self.move_D()
            elif move == 'L':
                self.move_L()
            elif move == 'R':
                self.move_R()
            elif move == 'F':
                self.move_F()
            elif move == 'B':
                self.move_B()

    def move_U(self):
        """Rotate Up face clockwise"""
        # Rotate the U face itself
        self.matrix['U'] = self.rotate_face_clockwise(self.matrix['U'])
        
        # Rotate adjacent edges
        temp = [self.matrix['F'][0][i] for i in range(3)]
        for i in range(3):
            self.matrix['F'][0][i] = self.matrix['R'][0][i]
            self.matrix['R'][0][i] = self.matrix['B'][0][i]
            self.matrix['B'][0][i] = self.matrix['L'][0][i]
            self.matrix['L'][0][i] = temp[i]

    def move_D(self):
        """Rotate Down face clockwise"""
        # Rotate the D face itself
        self.matrix['D'] = self.rotate_face_clockwise(self.matrix['D'])
        
        # Rotate adjacent edges (opposite direction from U)
        temp = [self.matrix['F'][2][i] for i in range(3)]
        for i in range(3):
            self.matrix['F'][2][i] = self.matrix['L'][2][i]
            self.matrix['L'][2][i] = self.matrix['B'][2][i]
            self.matrix['B'][2][i] = self.matrix['R'][2][i]
            self.matrix['R'][2][i] = temp[i]

    def move_L(self):
        """Rotate Left face clockwise"""
        # Rotate the L face itself
        self.matrix['L'] = self.rotate_face_clockwise(self.matrix['L'])
        
        # Rotate adjacent edges
        temp = [self.matrix['U'][i][0] for i in range(3)]
        for i in range(3):
            self.matrix['U'][i][0] = self.matrix['B'][2-i][2]
            self.matrix['B'][2-i][2] = self.matrix['D'][i][0]
            self.matrix['D'][i][0] = self.matrix['F'][i][0]
            self.matrix['F'][i][0] = temp[i]

    def move_R(self):
        """Rotate Right face clockwise"""
        # Rotate the R face itself
        self.matrix['R'] = self.rotate_face_clockwise(self.matrix['R'])
        
        # Rotate adjacent edges
        temp = [self.matrix['U'][i][2] for i in range(3)]
        for i in range(3):
            self.matrix['U'][i][2] = self.matrix['F'][i][2]
            self.matrix['F'][i][2] = self.matrix['D'][i][2]
            self.matrix['D'][i][2] = self.matrix['B'][2-i][0]
            self.matrix['B'][2-i][0] = temp[i]

    def move_F(self):
        """Rotate Front face clockwise"""
        # Rotate the F face itself
        self.matrix['F'] = self.rotate_face_clockwise(self.matrix['F'])
        
        # Rotate adjacent edges
        temp = [self.matrix['U'][2][i] for i in range(3)]
        for i in range(3):
            self.matrix['U'][2][i] = self.matrix['L'][2-i][2]
            self.matrix['L'][2-i][2] = self.matrix['D'][0][2-i]
            self.matrix['D'][0][2-i] = self.matrix['R'][i][0]
            self.matrix['R'][i][0] = temp[i]

    def move_B(self):
        """Rotate Back face clockwise"""
        # Rotate the B face itself
        self.matrix['B'] = self.rotate_face_clockwise(self.matrix['B'])
        
        # Rotate adjacent edges
        temp = [self.matrix['U'][0][i] for i in range(3)]
        for i in range(3):
            self.matrix['U'][0][i] = self.matrix['R'][i][2]
            self.matrix['R'][i][2] = self.matrix['D'][2][2-i]
            self.matrix['D'][2][2-i] = self.matrix['L'][2-i][0]
            self.matrix['L'][2-i][0] = temp[i]

    def parse_move_sequence(self, sequence):
        """Parse a move sequence string into individual moves"""
        # Remove extra spaces and split by spaces
        moves = sequence.strip().split()
        parsed_moves = []
        
        for move in moves:
            # Use regex to parse move, number, and prime notation
            match = re.match(r'([UDLRFB])([\d])?(\')?' , move)
            if match:
                face = match.group(1)
                num_str = match.group(2)
                prime = match.group(3)
                
                # Calculate number of clockwise turns
                if prime:  # Counter-clockwise (prime) is 3 clockwise turns
                    num = 3
                else:
                    num = int(num_str) if num_str else 1
                
                parsed_moves.append((face, num))
        
        return parsed_moves

    def execute_sequence(self, sequence):
        """Execute a sequence of moves"""
        moves = self.parse_move_sequence(sequence)
        
        for face, num in moves:
            self.execute_move(face, num)
        
    def count_color_in_face(self, face_name, color):
        """Count the number of squares of a specific color in a face"""
        if face_name not in self.matrix:
            return 0
        
        count = 0
        face = self.matrix[face_name]
        for row in face:
            for square in row:
                if square == color:
                    count += 1
        return count

    def _normalize_color(self, color: Optional[str]) -> Optional[str]:
        if not color or not isinstance(color, str):
            return None
        normalized = color.strip().lower()
        return self.COLOR_ALIASES.get(normalized, normalized)

    def _normalize_face(self, face: Optional[str]) -> Optional[str]:
        if not face or not isinstance(face, str):
            return None
        normalized = face.strip().upper()
        face_letter = self.FACE_MAP.get(normalized, normalized)
        return face_letter if face_letter in 'UDLRFB' else None

    def _question_mentions_color_and_face(self, question: Optional[str]) -> bool:
        if not question or not isinstance(question, str):
            return False
        text = question.lower()
        color_words = {'red', 'green', 'blue', 'yellow', 'orange', 'grey', 'gray', 'white', 'silver'}
        face_words = {'front', 'back', 'left', 'right', 'up', 'down', 'top', 'bottom', 'upper', 'lower'}
        mentions_color = any(re.search(rf'\\b{word}\\b', text) for word in color_words)
        mentions_face = any(re.search(rf'\\b{word}\\b', text) for word in face_words)
        return mentions_color and mentions_face

    def _extract_color_face_regex(self, question: str) -> Optional[tuple]:
        color_patterns = [
            r"(\w+)\s+squares?\s+(?:in|on)\s+(?:the\s+)?(\w+)\s+face",
            r"(\w+)\s+squares?\s+(?:in|on)\s+(?:the\s+)?([UDLRFB])\s+face",
            r"(?:count\s+(?:of|the))?\s*(\w+)\s+(?:squares?\s+)?(?:in|on)\s+(?:the\s+)?(\w+)",
            r"number\s+of\s+(\w+)\s+squares?\s+(?:in|on)\s+(?:the\s+)?(\w+)"
        ]

        for pattern in color_patterns:
            color_match = re.search(pattern, question, re.IGNORECASE)
            if not color_match:
                continue

            color = self._normalize_color(color_match.group(1))
            face = self._normalize_face(color_match.group(2))

            if color and face:
                return color, face
        return None

    def _extract_color_face_llm(self, question: str) -> Optional[tuple]:
        if not getattr(self, "llm_engine", None):
            return None
        if not self._question_mentions_color_and_face(question):
            return None

        prompt = f"""You are extracting structured details from a Rubik's cube puzzle.
Question: {question}
Identify which color is being counted and which cube face that color should be counted on.
Respond ONLY with JSON: {{"color": "<color name lowercase>", "face": "<face letter>"}}.
The face letter must be one of U, D, L, R, F, B (Up, Down, Left, Right, Front, Back)."""

        response = self.llm_engine.generate(prompt)
        if isinstance(response, dict):
            raw_text = response.get("text")
        else:
            raw_text = str(response)

        if not isinstance(raw_text, str):
            return None

        raw_text = raw_text.strip()
        data = None
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r'"?color"?\s*[:=]\s*"?([a-zA-Z]+)"?.*"?face"?\s*[:=]\s*"?([a-zA-Z]+)"?', raw_text, re.IGNORECASE | re.DOTALL)
            if match:
                data = {"color": match.group(1), "face": match.group(2)}

        if not data:
            return None

        color = self._normalize_color(data.get("color"))
        face = self._normalize_face(data.get("face"))

        if color and face:
            return color, face
        return None

    def solve_question(self, question):
        """Solve a typical Rubik's cube question"""

        # PRIMARY METHOD: Try regex parsing first

        move_patterns = [
            r"after\s+'((?:[UDLRFB]\d?\s*)+)'",
            r"move sequence\s+'((?:[UDLRFB]\d?\s*)+)'",
        ]

        move_sequence = None
        for pattern in move_patterns:
            m = re.search(pattern, question, re.IGNORECASE)
            if not m:
                continue

            # 1) extract & normalize spaces
            raw = re.sub(r'\s+', ' ', m.group(1)).strip()     # → "U3 D3 U"

            # 2) validate with optional digit & optional prime
            valid_moves = re.findall(r"[UDLRFB]\d?'?", raw, re.IGNORECASE)

            if valid_moves:
                move_sequence = " ".join(valid_moves).upper()
                break


        # If regex failed, use LLM as fallback
        if not move_sequence:
            
            prompt = f"""You are a Rubik's cube expert. Extract the move sequence from the following question using only the standard notation (U, D, L, R, F, B) with optional modifiers (2 for double turns, ' for counterclockwise).
                Rules:
                1. Valid moves are U, D, L, R, F, B (uppercase only)
                2. Modifiers: 2 (double turn), 3 (triple turn), ' (counterclockwise)
                3. Return only the move sequence, nothing else
                4. Separate moves with spaces
                5. If no valid moves found, return "NO_MOVES"

                Examples:
                Input: "After doing U2 R' B, how many blue squares are on top?"
                Output: U2 R' B

                Input: "Count red squares after moving the upper face twice and right face once"
                Output: U2 R

                Question: {question}
                Moves: """
            
            # Get response from LLM
            sequence = self.llm_engine.generate(prompt)
            sequence = sequence.get("text") or str(sequence)
            print(sequence)
            if sequence == "NO_MOVES" or not sequence:
                raise ValueError("Could not find a valid move sequence in the question. Please use standard notation (U, D, L, R, F, B) with optional 2 or ' for moves.")
            
            move_sequence = sequence
        # Execute the sequence
        self.execute_sequence(move_sequence)
        
        color_face = self._extract_color_face_regex(question)
        if not color_face:
            color_face = self._extract_color_face_llm(question)

        if not color_face:
            raise ValueError("Query must include the target color and face (e.g., 'green squares on the front face'). Pass the full question, not only the move sequence.")

        color, face = color_face
        return self.count_color_in_face(face, color)

    def test(self, tool_test: str="rubik_cube_solver", file_location: str="rubik_cube_solver", result_parameter: str="result", search_type: str="exact_match"):
        return super().test(tool_test=tool_test, file_location=file_location, result_parameter=result_parameter, search_type=search_type)

# Example usage
if __name__ == "__main__":
    # Example question from your prompt    
    tool = Rubik_Cube_Solver_Tool()
    
    # Initialize solver
    tool.embed_tool()
    tool.test(tool_test="rubik_cube_solver", file_location="rubik_cube_solver", result_parameter="result", search_type="exact_match")
