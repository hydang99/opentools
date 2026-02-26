import os, sys, easyocr, traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..", "..", "..")))
from opentools.core.base import BaseTool
from opentools.core.factory import create_llm_engine
from typing import Optional, Dict, Any
from pydantic import BaseModel

class CalendarMetadata(BaseModel):
    month: str
    first_day: str
    first_date: int

class Calendar_Calculation_Tool(BaseTool):
    """Calendar_Calculation_Tool
    ---------------------
    Purpose:
        A sophisticated calendar analysis tool that solves date-related problems using computer vision and date calculation algorithms. It can extract calendar information from images and determine days of the week for dates across different years, handling both leap and non-leap years. The tool uses EasyOCR for text recognition and implements complex date calculation logic to track days across months and years.

    Core Capabilities:
        - Computer vision-based calendar text extraction using EasyOCR
        - Date calculation algorithms for leap and non-leap years
        - Day-of-week determination across different years (previous, current, next)
        - Month and day parsing from natural language queries
        - Leap year status detection and handling
        - Calendar layout analysis and day header recognition
        - Precise date arithmetic across month and year boundaries

    Intended Use:
        Use this tool when you need to solve date-related problems, including determining the day of the week for a given date, calculating dates across different years, and handling leap years.

    Limitations:
        - May not handle complex date expressions or symbolic date calculations
    """
    # Default args for `opentools test Calendar_Calculation_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "calendar_calculation",
        "file_location": "calendar_calculation",
        "result_parameter": "result",
        "search_type": "similarity_eval",
    }
    require_llm_engine = True
    def __init__(self, llm_engine=None, model_string="gpt-4o-mini"):
        super().__init__(
            type='function',
            name="Calendar_Calculation_Tool",
            description="""A sophisticated calendar analysis tool that solves date-related problems using computer vision and date calculation algorithms.
            It can extract calendar information from images and determine days of the week for dates across different years, handling both leap and non-leap years. 
            The tool uses EasyOCR for text recognition and implements complex date calculation logic to track days across months and years. CAPABILITIES: Computer 
            vision-based calendar text extraction using EasyOCR, date calculation algorithms for leap and non-leap years, day-of-week determination across different years
            (previous, current, next), month and day parsing from natural language queries, leap year status detection and handling, calendar layout analysis and day 
            header recognition, precise date arithmetic across month and year boundaries. SYNONYMS: calendar solver, date calculator, day of week finder, calendar 
            analysis tool, date arithmetic tool, leap year calculator, calendar problem solver, date computation tool. EXAMPLES: 'Last year and this year is non-leap 
            year. Which day of the week was on August 2 of the previous year?', 'If this is a leap year, what day will November 15 be next year?', 'If this is a leap 
            year, what day will November 15 of that year?'.""",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A text query describing the calendar problem to solve, including target date, year relation (same/previous/next), and leap year information. Must explicitly specify which year to calculate."
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Path to an image file containing a calendar page showing month layout with days of the week. Image must be clear with readable text."
                    }
                },
                "required": ["query", "image_path"],
                "additionalProperties": False,
            },
            strict=True,
            category="date_calculation",
            tags=["calendar", "date_calculation", "day_of_week", "computer_vision", "ocr", "leap_year", "date_arithmetic", "calendar_analysis", "date_solver"],
            limitation="IMAGE REQUIREMENTS: Requires clear, high-resolution calendar images with readable text. Calendar must show month name and day headers clearly. QUERY FORMAT: Queries must explicitly specify which year to calculate (this/that/next/previous year) - ambiguous year references will fail. CALCULATION LIMITS: Limited to calculations within one year before or after the shown month. Cannot handle complex date expressions or symbolic date calculations. DEPENDENCIES: Requires EasyOCR for text recognition and clear calendar layout. May fail with poor quality images or non-standard calendar formats.",
            agent_type="Puzzle-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(query='What day of the week was on August 2 of the previous year?', image_path='test.jpg')",
                "description": "Calculate the day of the week for August 2 of the previous year"
            },
            require_llm_engine=True,
            llm_engine=llm_engine,
        )

    
    def get_metadata(self):
        return super().get_metadata()
    
    def embed_tool(self):
        return super().embed_tool() 

    def extract_calendar_info(self, image_path):
        """
        Extract calendar info using EasyOCR with better logic
        """
        try:
            # Initialize reader
            reader = easyocr.Reader(['en'])
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.path.dirname(__file__), image_path) 
            results = reader.readtext(image_path)
            
            if not results:
                print("No text detected in image")
                return None
                
            # Sort results by vertical position (top to bottom)
            
            results.sort(key=lambda x: x[0][0][1])  # Sort by y-coordinate
            
            # Extract text with positions
            text_data = []
            for (bbox, text, prob) in results:
                # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                x_center = float(bbox[0][0] + bbox[2][0]) / 2
                y_center = float(bbox[0][1] + bbox[2][1]) / 2
                text_data.append({
                    'text': text.strip(),
                    'x': x_center,
                    'y': y_center,
                    'bbox': bbox
                })
            
            # Find month
            months = ["January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"]
            
            month_found = None
            month_y = 0
            for item in text_data:
                for month in months:
                    if month.lower() in item['text'].lower():
                        month_found = month
                        month_y = item['y']
                        break
                if month_found:
                    break
            
            # Find day headers (Mon, Tue, etc.)
            day_headers = []
            day_header_y = None
            days_short = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            days_full = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            # Group texts by approximate Y position (same row)
            rows = {}
            for item in text_data:
                # Round Y to nearest 10 to group items on same row
                row_y = round(item['y'] / 10) * 10
                if row_y not in rows:
                    rows[row_y] = []
                rows[row_y].append(item)
            
            # Find the row with day headers
            for row_y, items in rows.items():
                # Check if this row contains day headers
                day_count = 0
                for item in items:
                    if any(day in item['text'] for day in days_short):
                        day_count += 1
                
                if day_count >= 4:  # Found day header row
                    day_header_y = row_y
                    # Sort by x position (left to right)
                    items.sort(key=lambda x: x['x'])
                    day_headers = items
                    break
            
            # Find the number "1" (look for "1" instead of "2")
            one_position = None
            for item in text_data:
                if item['text'] == '2' or item['text'] == '2.':
                    # Make sure it's below the day headers
                    if day_header_y and item['y'] > day_header_y:
                        # Check if this is likely the first "1" (not 10, 11, etc.)
                        if one_position is None or item['y'] < one_position['y']:
                            one_position = item
            
            # Determine which day column the "1" falls under
            if one_position and day_headers:
                # Find which day header is closest to the "1" horizontally
                min_distance = float('inf')
                closest_day_idx = 0
                
                for idx, header in enumerate(day_headers):
                    distance = abs(header['x'] - one_position['x'])
                    if distance < min_distance:
                        min_distance = distance
                        closest_day_idx = idx
                
                # Map the header text to full day name
                header_text = day_headers[closest_day_idx]['text']
                
                # Find matching day
                first_day = None
                for i, day_short in enumerate(days_short):
                    if day_short.lower() in header_text.lower():
                        first_day = days_full[i-1]
                        break
                
                if not first_day:
                    # Try to use position if text matching fails
                    if closest_day_idx < len(days_full):
                        first_day = days_full[closest_day_idx-1]
                
                if month_found and first_day:
                    return {
                        "month": month_found,
                        "first_day": first_day,
                        "first_date": 1
                    }
            
            # If we only found month but not the day
            if month_found:
                return {
                    "month": month_found,
                    "first_day": "Unknown",
                    "first_date": 1
                }
            
            print("Failed to extract calendar information")
            return None
            
        except ImportError:
            print("EasyOCR not installed. Run: pip install easyocr")
            return None
        except Exception as e:
            print(f"Error in EasyOCR extraction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def parse_query(self, query_text):
        """
        Extract key information from the query using multiple parsing strategies.
        Falls back to alternative methods if primary method fails.
        """
        result: Dict[str, Optional[int]] = {
            "target_month": None,
            "target_day": None,
            "leap_year": None,
            "target_year": None,
            "current_year_is_leap": None  # Explicitly track current year leap status
        }
        
        # Extract month using multiple approaches
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }
        # Also include abbreviated months
        month_abbreviations = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "may": 5, "jun": 6, "jul": 7, "aug": 8,
            "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        
        query_lower = query_text.lower()
        
        # Find month
        for month_name, month_num in months.items():
            if month_name in query_lower:
                result["target_month"] = month_num
                break
        if result["target_month"] is None:
            for month_name, month_num in month_abbreviations.items():
                if month_name in query_lower:
                    result["target_month"] = month_num
                    break   
        
        # Extract day (look for patterns like "November 9", "April 11", "9 November", "11 April")
        import re
        
        # Pattern 1: Month followed by day (e.g., "November 9", "April 11")
        day_match = re.search(r'(\w+)\s+(\d+)', query_text)
        if day_match:
            # Check if the first group is a month name
            first_group = day_match.group(1).lower()
            if first_group in months or first_group in month_abbreviations:
                result["target_day"] = int(day_match.group(2))
        
        # Pattern 2: Day followed by month (e.g., "9 November", "11 April")
        if result["target_day"] is None:
            day_match = re.search(r'(\d+)\s+(\w+)', query_text)
            if day_match:
                # Check if the second group is a month name
                second_group = day_match.group(2).lower()
                if second_group in months or second_group in month_abbreviations:
                    result["target_day"] = int(day_match.group(1))
        
        # Determine year relation - look for the specific year being asked about
        # Search for patterns like "on August 29 of the previous year" or "what day will November 15 be next year"
        year_patterns = [
            r'(\w+\s+\d+)\s+of\s+the\s+(previous|next|that)\s+year',  # "August 29 of the previous year"
            r'(\d+\s+\w+)\s+of\s+the\s+(previous|next|that)\s+year',    # "29 August of the previous year"
            r'(\w+\s+\d+)\s+be\s+(next|previous|that)\s+year',           # "what day will November 15 be next year"
            r'(\d+\s+\w+)\s+be\s+(next|previous|that)\s+year',           # "what day will 15 November be next year"
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, query_lower)
            if match:
                year_relation = match.group(2)
                if year_relation == "previous":
                    result["target_year"] = -1
                elif year_relation == "next":
                    result["target_year"] = 1
                elif year_relation == "that":
                    result["target_year"] = 0
                break
            else:
                result["target_year"] = 0
        
        # Improved leap year parsing - handles both "leap year" and "non-leap year" mentions
        sentences = query_lower.split(".")
        current_year_leap_status = None
        previous_year_leap_status = None
        next_year_leap_status = None
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Check for current/this year leap status
            if "this year" in sentence or "current year" in sentence or ("calendar" in sentence and ("shows" in sentence or "of a" in sentence)):
                if "non-leap year" in sentence or "non leap year" in sentence:
                    current_year_leap_status = False
                elif "leap year" in sentence:
                    current_year_leap_status = True
            # Check for previous/last year leap status
            elif "previous year" in sentence or "last year" in sentence:
                if "non-leap year" in sentence or "non leap year" in sentence:
                    previous_year_leap_status = False
                elif "leap year" in sentence:
                    previous_year_leap_status = True
            # Check for next year leap status
            elif "next year" in sentence:
                if "non-leap year" in sentence or "non leap year" in sentence:
                    next_year_leap_status = False
                elif "leap year" in sentence:
                    next_year_leap_status = True
            # Generic leap year mentions (no specific year mentioned)
            elif "leap year" in sentence and "non-leap year" not in sentence and "non leap year" not in sentence:
                # If no year is specified, assume it refers to current year
                if current_year_leap_status is None:
                    current_year_leap_status = True
        
        # Store current year leap status explicitly
        if current_year_leap_status is not None:
            result["current_year_is_leap"] = 1 if current_year_leap_status else 0
        # Also try to infer from calendar description (e.g., "calendar of a month of a particular non-leap year")
        if result["current_year_is_leap"] is None:
            if "non-leap year" in query_lower and ("calendar" in query_lower or "shows" in query_lower or "particular" in query_lower):
                result["current_year_is_leap"] = 0
            elif "leap year" in query_lower and ("calendar" in query_lower or "shows" in query_lower or "particular" in query_lower) and "non-leap" not in query_lower:
                result["current_year_is_leap"] = 1
        
        # Determine leap_year value based on which year is leap
        # Priority: if asking about previous year and previous year is leap, set to -1
        # If asking about next year and next year is leap, set to 1
        # Otherwise, if current year is leap, set to 0
        if result["target_year"] == -1 and previous_year_leap_status is True:
            result["leap_year"] = -1
        elif result["target_year"] == 1 and next_year_leap_status is True:
            result["leap_year"] = 1
        elif current_year_leap_status is True:
            result["leap_year"] = 0
        elif previous_year_leap_status is True:
            result["leap_year"] = -1
        elif next_year_leap_status is True:
            result["leap_year"] = 1
        elif current_year_leap_status is False:
            # Current year is explicitly non-leap, but we still need to know which year IS leap
            if previous_year_leap_status is True:
                result["leap_year"] = -1
            elif next_year_leap_status is True:
                result["leap_year"] = 1
            # If current is non-leap and we don't know about others, leave as None for LLM fallback
        
        # --- LLM fallback for missing values ---
        missing_keys = [k for k, v in result.items() if v is None]
        if ("leap_year" in missing_keys):
            missing_keys.remove('leap_year')
        if missing_keys or result["leap_year"] is None:
            llm_prompt = f"""
            Extract the following information from this calendar question. If you cannot find a value, output 'None' for that field:
            1. Target month (e.g., 'August' or None) - look for both "August 15" and "15 August" formats
            2. Target day (e.g., '15' or None) - look for both "August 15" and "15 August" formats
            3. Year relation as an integer: -1 for previous year, 0 for current/this year, 1 for next year, None for does not mention. IMPORTANT: Look for the specific year being asked about, not just any year mentioned in the text. For example, if the question asks "what day was August 29 of the previous year", the year relation is -1 (previous year).
            4. Leap year status as an integer: 
               - 0 if current year is leap year (e.g., "calendar shows particular leap year", "this year is a leap year")
               - -1 if previous year was leap year (e.g., "previous year was a leap year", "last year was a leap year")
               - 1 if next year will be leap year (e.g., "next year is leap year")
               - IMPORTANT: If the query says "this year is a non-leap year and the previous year was a leap year", and you're asked about the previous year, the leap_year should be -1 (previous year was leap)
               - If the query says "this year is a non-leap year", that means current year is NOT leap, so look for which year IS leap (previous, current, or next)
               - None if it does not mention
            5. Current year leap status as 0 (non-leap) or 1 (leap): Look for explicit statements like "this year is a non-leap year", "calendar shows a particular non-leap year", "this year is a leap year", etc. If not mentioned, infer from context.
            Format: month|day|year_relation|leap_year|current_year_leap
            Question: {query_text}
            """
            llm_response = self.llm_engine.generate(llm_prompt)
            if isinstance(llm_response, dict):
                llm_response = llm_response.get('text')
            else:
                llm_response = str(llm_response)
            try:
                parts = [s.strip() for s in llm_response.strip().split('|')]
                if len(parts) >= 4:
                    month, day, year_rel, leap = parts[0], parts[1], parts[2], parts[3]
                    if result["target_month"] is None and month.lower() != 'none':
                        result["target_month"] = months.get(month.lower()) or month_abbreviations.get(month.lower()[:3])
                    if result["target_day"] is None and day.lower() != 'none':
                        result["target_day"] = int(day)
                    if result["target_year"] is None and year_rel.lower() != 'none':
                        try:
                            result["target_year"] = int(year_rel)
                        except Exception:
                            result["target_year"] = 0
                    if result["leap_year"] is None and leap.lower() != 'none':
                        try:
                            result["leap_year"] = int(leap)
                        except Exception:
                            pass
                    # Parse current_year_is_leap if provided
                    if len(parts) >= 5:
                        current_leap = parts[4]
                        if result["current_year_is_leap"] is None and current_leap.lower() != 'none':
                            try:
                                result["current_year_is_leap"] = int(current_leap)
                            except Exception:
                                pass
            except Exception as e:
                pass  
        return result

    def get_days_in_month(self, month, is_leap_year=False):
        """Return number of days in a given month"""
        days_in_months = {
            "January": 31, "February": 28, "March": 31, "April": 30,
            "May": 31, "June": 30, "July": 31, "August": 31,
            "September": 30, "October": 31, "November": 30, "December": 31
        }
        
        if month == "February" and is_leap_year:
            return 29
        return days_in_months[month]

    def calculate_day_of_week(self, calendar_info, query_info):
        """
        Calculate the day of week for the target date based on calendar info
        """
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        months_order = ["January", "February", "March", "April", "May", "June", 
                        "July", "August", "September", "October", "November", "December"]
        
        # Get starting point from calendar
        current_month = calendar_info["month"]
        current_first_day = calendar_info["first_day"]
        current_first_day_index = days_of_week.index(current_first_day)
        
        # Determine leap year status for each year
        current_year_is_leap = query_info["leap_year"] == 0
        
        # Calculate based on year relation
        if query_info["target_year"] == -1:  # Previous year  
            previous_year_is_leap = query_info["leap_year"] == -1
          
            days_shift = 366 if previous_year_is_leap else 365
            # Going backward, so subtract the shift
            days_to_jan1_current = self.days_from_jan1_to_date(current_month, 1, current_year_is_leap)
            jan1_current_index = (current_first_day_index - days_to_jan1_current) % 7
            jan1_previous_index = (jan1_current_index - days_shift) % 7

            # Calculate target date in previous year
            days_to_target = self.days_from_jan1_to_date(
                months_order[query_info["target_month"] - 1], 
                query_info["target_day"], 
                previous_year_is_leap
            )
            
            target_day_index = (jan1_previous_index + days_to_target) % 7
            return days_of_week[target_day_index]
        
        elif query_info["target_year"] == 0:  # Same year
            # Calculate days from current month 1st to target date
            current_month_index = months_order.index(current_month)
            target_month_index = query_info["target_month"] - 1
            
            days_difference = 0
            
            if current_month_index == target_month_index:
                # Same month
                days_difference = query_info["target_day"] - 1
            elif current_month_index < target_month_index:
                # Target is in a later month
                # Days remaining in current month
                days_difference += self.get_days_in_month(current_month, current_year_is_leap) - 1
                
                # Days in intermediate months
                for i in range(current_month_index + 1, target_month_index):
                    month_name = months_order[i]
                    days_difference += self.get_days_in_month(month_name, current_year_is_leap)
                
                # Days in target month
                days_difference += query_info["target_day"]
            else:
                # Target is in an earlier month (going backward)
                # Days back to beginning of current month
                days_difference -= 1
                
                # Days in intermediate months (going backward)
                for i in range(target_month_index + 1, current_month_index):
                    month_name = months_order[i]
                    days_difference -= self.get_days_in_month(month_name, current_year_is_leap)
                
                # Days back from end of target month to target day
                target_month_name = months_order[target_month_index]
                days_difference -= (self.get_days_in_month(target_month_name, current_year_is_leap) - query_info["target_day"])
            
            target_day_index = (current_first_day_index + days_difference) % 7
            return days_of_week[target_day_index]
        
        elif query_info["target_year"] == 1:  # Next year
            
            days_shift = 366 if current_year_is_leap else 365
            days_to_jan1_current = self.days_from_jan1_to_date(current_month, 1, current_year_is_leap)
            jan1_current_index = (current_first_day_index - days_to_jan1_current) % 7
            jan1_next_index = (jan1_current_index + days_shift) % 7
            # Calculate target date in next year
            next_year_is_leap = query_info["leap_year"] == 1
            days_to_target = self.days_from_jan1_to_date(
                months_order[query_info["target_month"] - 1], 
                query_info["target_day"], 
                next_year_is_leap
            )
            
            target_day_index = (jan1_next_index + days_to_target) % 7
            return days_of_week[target_day_index]
    
    def days_from_jan1_to_date(self, month_name, day, is_leap_year):
        """Calculate days from January 1 to given date (0-based)"""
        months_order = ["January", "February", "March", "April", "May", "June", 
                        "July", "August", "September", "October", "November", "December"]
        
        month_index = months_order.index(month_name)
        
        total_days = 0
        # Add days for complete months
        for i in range(month_index):
            total_days += self.get_days_in_month(months_order[i], is_leap_year)
        
        # Add days in target month (subtract 1 because Jan 1 is day 0)
        total_days += day - 1
        
        return total_days
    
    def run(self, image_path, query):
        """
        Main solver function that coordinates the solution with error handling
        """
        try:
            # Step 1: Extract calendar information
            calendar_info = self.extract_calendar_info(image_path)
            if not calendar_info:
                return {"message": "Error: Could not extract calendar information from image. Please ensure the image is clear and shows a complete calendar.", "success": False}
            # Step 2: Parse the query with fallback to LLM
            query_info = self.parse_query(query)

            # Step 3: Calculate the answer
            result_day = self.calculate_day_of_week(calendar_info, query_info)
            if not result_day:
                return {"error": "Error: Could not calculate the day of the week. Please check input information.", "success": False}

            return {'result': result_day, 'success': True}

        except Exception as e:
            return {"error": f"Error: An unexpected error occurred: {str(e)}", "success": False, "traceback": traceback.format_exc()}
    
    def test(self, tool_test: str="calendar_calculation", file_location: str="calendar_calculation", result_parameter: str="result", search_type: str="exact_match"):
        return super().test(tool_test=tool_test, file_location=file_location, result_parameter=result_parameter, search_type=search_type)

# Example usage
if __name__ == "__main__":
    tool = Calendar_Calculation_Tool()
    tool.embed_tool()
    query = r"What day of the week was January 16 of the previous year, given that this year is a non-leap year and the previous year was a leap year?"
    # res = tool.run(image='/home/daoqm/opentools/src/opentools/Benchmark/algopuzzlevqa/images/calendar_0011.jpg', query= query)
    # print(res)
    tool.test(tool_test="calendar_calculation", file_location="calendar_calculation", result_parameter='result', search_type='exact_match')