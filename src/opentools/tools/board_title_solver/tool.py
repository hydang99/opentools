import os, sys, cv2, math, re, traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
import numpy as np
from collections import deque
class Board_Title_Solver_Tool(BaseTool):
    """
        Board_Title_Solver_Tool
        ---------------------
        Purpose:
            A tool for solving board tiling puzzles, particularly focusing on domino tiling problems. It analyzes checkerboard images with missing squares and determines if a valid tiling solution exists using dominoes. The tool employs computer vision for board detection and the Hopcroft-Karp algorithm for bipartite matching to solve the tiling problem.

        Core Capabilities:
            - Analyzes checkerboard images to detect missing squares
            - Determines if valid domino tiling exists
            - Provides detailed domino placement solutions
            - Handles various board dimensions
            - Uses computer vision for square detection
            - Implements Hopcroft-Karp algorithm for optimal matching\

        Intended Use:
            Use this tool when you need to solve board tiling puzzles, particularly domino tiling problems. It is particularly useful for solving puzzles with missing squares and determining if a valid tiling solution exists.

        Limitations:
            - Requires clear images with distinguishable light and dark squares
            - Board dimensions must be explicitly stated in query
            - Only handles 2x1 or 1x2 domino tiling problems
            - Assumes checkerboard pattern with alternating colors
            - Missing squares must be clearly marked in white
            - Grid detection may fail with irregular tile spacing
            - May struggle with similar or gradient colors
            - Requires white background for board detection
            - Colors should be solid without patterns
            - Grid detection may fail with irregular tile spacing
    """
    # Default args for `opentools test Board_Title_Solver_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "board_title_solver",
        "file_location": "board_title_solver",
        "result_parameter": "result",
        "search_type": "similarity_eval",
    }
    require_llm_engine = True
    def __init__(self, llm_engine=None, model_string="gpt-4o-mini"):
        super().__init__(
            type='function',
            name="Board_Title_Solver_Tool",
            description="""A tool for solving board tiling puzzles, particularly focusing on domino tiling problems. It analyzes checkerboard images with missing  squares and determines if a valid tiling solution exists using dominoes. The tool employs computer vision for board detection and the Hopcroft-Karp algorithm  for bipartite matching to solve the tiling problem. CAPABILITIES: Analyzes checkerboard images to detect missing squares, determines if valid domino tiling  exists, provides detailed domino placement solutions, handles various board dimensions, uses computer vision for square detection, implements Hopcroft-Karp  algorithm for optimal matching. SYNONYMS: board tiling solver, domino puzzle solver, checkerboard tiling, board coverage solver, tiling algorithm, puzzle  solver, board analysis tool. EXAMPLES: 'Can a 7 * 8 checkerboard with two missing squares be covered by 27 dominoes?', 'Is it possible to cover an 8 * 8 board with two corners removed using 31 dominoes?', 'Solve this 5 * 6 board tiling puzzle'.""",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The puzzle description containing board dimensions (e.g., '7 * 8') and tiling requirements"
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file showing the checkerboard with missing squares"
                    }
                },
                "required": ["query", "image_path"],
                "additionalProperties": False,
            },
            strict=True,
            category="puzzle_solving",
            tags=["board_tiling", "domino_puzzle", "checkerboard","algorithm", "puzzle_solver"],
            limitation="Requires clear images with distinguishable light and dark squares, board dimensions must be explicitly stated in query, only handles 2x1 or 1x2 domino tiling problems, assumes checkerboard pattern with alternating colors, missing squares must be clearly marked in white",
            agent_type="Puzzle-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),  
            demo_commands= {
                "command": "reponse = tool.run(query='7 * 8', image_path='test.jpg')",
                "description": "Solve a 7 * 8 checkerboard with two missing squares"
            },
            require_llm_engine=True,
            llm_engine=llm_engine,
            model_string=model_string,
        )
        
    def run(self, query: str, image_path: str) : 
        try:
            rows, cols = self.matrix_size(query)
            solution = self.solve(image_path, rows, cols)
        except Exception as e:
            return {
                "error": f"Error: {e}",
                "traceback": traceback.format_exc(),
                "success": False
            }
        return solution

    def matrix_size(self, query: str) -> tuple[int, int]:
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

    def solve(self, image_path: str, rows: int, cols: int):
        # Resolve the image path relative to the tool's directory
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(__file__), image_path)
        
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot load image at {image_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # compute the size of each cell
        H, W = img.shape[:2]
        cell_h = H // rows
        cell_w = W // cols

        # decode image into matrix
        present = np.ones((rows, cols), dtype=bool)
        missing = []

        for r in range(rows):
            for c in range(cols):
                y1 = r * cell_h + cell_h // 4
                y2 = (r + 1) * cell_h - cell_h // 4
                x1 = c * cell_w + cell_w // 4
                x2 = (c + 1) * cell_w - cell_w // 4
                patch = img[y1:y2, x1:x2]
                if np.mean(patch.astype(np.float32)) > 240:  # white = missing
                    present[r, c] = False
                    missing.append((r, c))

        # BUILD BIPARTITE GRAPH
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        dark_yellow, light_yellow = [], []
        pos2idx = {}
        adj = []

        for r in range(rows):
            for c in range(cols):
                # skip missing squares
                if not present[r, c]:
                    continue
                if (r + c) % 2 == 0:
                    pos2idx[(r, c)] = len(dark_yellow)
                    dark_yellow.append((r, c))
                    adj.append([])
                else:
                    pos2idx[(r, c)] = len(light_yellow)
                    light_yellow.append((r, c))

        for (r, c), u in pos2idx.items():
            if (r + c) % 2 != 0:  # only from dark yellow
                continue
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and present[nr, nc] and (nr + nc) % 2 == 1:
                    adj[u].append(pos2idx[(nr, nc)])

        #  SOLVE DOMINO TILING
        nL, nR = len(dark_yellow), len(light_yellow)
        matching, pairL = self.hopcroft_karp(adj, nL, nR)
        
        # Print detailed matching information
        pair_match  = []
        for i, light_idx in enumerate(pairL):
            if light_idx != -1:
                dark_pos = dark_yellow[i]
                light_pos = light_yellow[light_idx]
                pair_match.append(f"Domino {i+1}: Dark yellow square at {dark_pos} paired with light yellow square at {light_pos}")

        # A valid tiling exists if number of pairs equals half the number of present squares
        tilable = (matching * 2 == present.sum())
        solution = {
            "result": "Yes" if tilable else "No",
            "total_match": f"{matching*2} over {present.sum()}",
            "success": True
        }
        
        return solution

    def hopcroft_karp(self, adj, nL, nR):
        pairL = [-1] * nL  
        pairR = [-1] * nR 
        dist = [0] * nL   

        def bfs():
            queue = deque()
            for u in range(nL):
                if pairL[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = math.inf
            reachable = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if pairR[v] == -1:
                        reachable = True
                    elif dist[pairR[v]] == math.inf:
                        dist[pairR[v]] = dist[u] + 1
                        queue.append(pairR[v])
            return reachable

        def dfs(u):
            for v in adj[u]:
                if pairR[v] == -1 or (dist[pairR[v]] == dist[u] + 1 and dfs(pairR[v])):
                    pairL[u], pairR[v] = v, u
                    return True
            dist[u] = math.inf
            return False

        matching = 0
        while bfs():
            for u in range(nL):
                if pairL[u] == -1 and dfs(u):
                    matching += 1
        return matching, pairL

    def test(self, tool_test: str="board_title_solver", file_location: str="board_title_solver", result_parameter: str="result", search_type: str="exact_match"):
        return super().test(tool_test=tool_test, file_location=file_location, result_parameter=result_parameter, search_type=search_type)
    
if __name__ ==  "__main__":
    print("Board Title Solver Tool Test")
    tool = Board_Title_Solver_Tool()
    tool.embed_tool()
    try:
        tool.test(tool_test="board_title_solver", file_location='board_title_solver', result_parameter='result', search_type='exact_match')
    except Exception as e:
        print(f"Error during test execution: {e}")