# source code: https://github.com/inclusionAI/AWorld/blob/main/examples/gaia/mcp_collections/tools/pubchem.py
import time, os, sys, requests, traceback
from datetime import datetime
from typing import Literal
from urllib.parse import quote
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
class CompoundData(BaseModel):
    """Structured compound data from PubChem."""

    cid: int | None = None
    name: str | None = None
    molecular_formula: str | None = None
    molecular_weight: float | None = None
    smiles: str | None = None
    inchi: str | None = None
    synonyms: list[str] = []


class PubChemMetadata(BaseModel):
    """Metadata for PubChem operation results."""

    query_type: str 
    query_value: str
    api_endpoint: str
    response_time: float
    total_results: int | None = None
    rate_limit_delay: float | None = None
    error_type: str | None = None
    timestamp: str

class Chemistry_Search_Tool(BaseTool):
    # Default args for `opentools test Chemistry_Search_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "chemistry_search",
        "file_location": "chemistry_search",
        "result_parameter": "result",
        "search_type": "similarity_eval",
    }

    """Chemistry_Search_Tool
    ---------------------
    Purpose:
        A comprehensive PubChem chemistry search tool that provides compound search, property retrieval, synonym lookup, similarity search, and chemical database capabilities.

    Core Capabilities:
        - Compound search by name, CID, SMILES, InChI, or molecular formula
        - Detailed chemical property retrieval (molecular weight, formula, SMILES, physicochemical properties)
        - Synonym and alternative name lookup
        - Chemical similarity search using Tanimoto similarity coefficient
        - PubChem database integration with rate limiting compliance
        - Multiple output formats (markdown, JSON, text)
        - Cross-platform compatibility with robust error handling

    Intended Use:
        Use this tool when you need to search for chemical compounds, retrieve their properties, find synonyms, and perform similarity searches using the PubChem database.

    Limitations:
        - May not handle complex chemical structures or properties
    """
    def __init__(self):
        super().__init__(
            type='function',
            name="Chemistry_Search_Tool",
            description="""A comprehensive PubChem chemistry search tool that provides compound search, property retrieval, synonym lookup, similarity search, and chemical database capabilities. CAPABILITIES: Compound search by name, CID, SMILES, InChI, or molecular formula, detailed chemical property retrieval (molecular weight, formula, SMILES, physicochemical properties), synonym and alternative name lookup, chemical similarity search using Tanimoto similarity coefficient, PubChem database integration with rate limiting compliance, multiple output formats (markdown, JSON, text), cross-platform compatibility with robust error handling. SYNONYMS: chemistry search, PubChem tool, compound finder, chemical database, molecular search, chemical property lookup, compound similarity, chemical identifier search, molecular formula search, chemical synonym finder. EXAMPLES: 'Search for aspirin compound', 'Get synonyms for CID 2244', 'Retrieve properties of caffeine', 'Find compounds similar to ethanol', 'Search compounds with formula C6H6'.""",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform",
                        "enum": ["search_compounds", "get_synonyms", "get_properties", "search_similar"]
                    },
                    "query": {
                        "type": "string",
                        "description": "Search terms for compound search (compound name, CID, SMILES, InChI, formula). Required for 'search_compounds' operation."
                    },
                    "cid": {
                        "type": "integer",
                        "description": "PubChem Compound ID. Required for 'get_synonyms', 'get_properties', 'search_similar' operations."
                    },
                    "search_type": {
                        "type": "string",
                        "description": "Type of search for compounds: 'name', 'cid', 'smiles', 'inchi', 'formula' (default: 'name')",
                        "enum": ["name", "cid", "smiles", "inchi", "formula"]
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1-100, default: 10)",
                        "minimum": 1,
                        "maximum": 100
                    },
                    "max_synonyms": {
                        "type": "integer",
                        "description": "Maximum number of synonyms to return (1-100, default: 20)",
                        "minimum": 1,
                        "maximum": 100
                    },
                    "properties": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of properties to retrieve (e.g., MolecularWeight, XLogP, TPSA). Default: common properties."
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Similarity threshold for similarity search (0.0-1.0, default: 0.9)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format: 'markdown', 'json', or 'text' (default: 'json')",
                        "enum": ["markdown", "json", "text"]
                    }
                },
                "required": ["operation"],
                "additionalProperties": False,
            },
            strict=False,
            category="chemistry",
            tags=["chemistry", "pubchem", "compound_search", "chemical_properties", "molecular_search", "chemical_database", "compound_similarity", "chemical_identifiers", "molecular_formula", "chemical_synonyms"],
            limitation="INTERNET DEPENDENCY: Requires stable internet connection for PubChem API access. RATE LIMITING: Subject to PubChem's 5 requests/second limit, may affect large batch operations. DATA AVAILABILITY: Some compounds may have limited data available, property availability depends on compound data completeness. SEARCH PERFORMANCE: Search speed depends on query complexity and network conditions. API RELIABILITY: Dependent on PubChem service availability and may experience temporary outages.",
            agent_type="Search-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(operation='search_compounds', query='aspirin')",
                "description": "Search for aspirin compound"
            }
        )
        self.workspace = Path(os.getcwd())
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.base_url_view = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
        self.request_delay = 0.5  # 500ms delay for more conservative rate limiting
        self.last_request_time = 0.0
        # Request timeout settings
        self.timeout = 60  # Increased timeout for complex queries
        # Initialize request session with headers
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "AWorld-PubChem-MCP/1.0 (https://github.com/aworld-framework)", "Accept": "application/json"}
        )

    def _rate_limit(self) -> float:
        """Enforce rate limiting to comply with PubChem usage policy.

        Returns:
            Actual delay time applied
        """
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.request_delay:
            delay = self.request_delay - time_since_last
            time.sleep(delay)
            self.last_request_time = time.time()
            return delay

        self.last_request_time = current_time
        return 0.0

    def _make_request(self, url: str, params: dict = None) -> tuple[dict | None, float]:
        """Make a rate-limited request to PubChem API.

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            Tuple of (response_data, response_time)

        Raises:
            requests.RequestException: For API request failures
        """
        start_time = time.time()

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response_time = time.time() - start_time

            if response.status_code == 200:
                return response.json(), response_time
            elif response.status_code == 202:
                # Handle async request - PubChem is processing the request
                try:
                    response_data = response.json()
                    if "Waiting" in response_data and "ListKey" in response_data["Waiting"]:
                        list_key = response_data["Waiting"]["ListKey"]
                        # Wait and poll for results multiple times
                        for poll_attempt in range(10):  # Try up to 10 times
                            time.sleep(5)  # Wait 5 seconds between polls
                            # Try different endpoint formats
                            poll_urls = [
                                f"{self.base_url}/listkey/{list_key}/cids/JSON",
                                f"{self.base_url}/listkey/{list_key}/JSON",
                                f"{self.base_url}/compound/listkey/{list_key}/property/Title,MolecularFormula,MolecularWeight,CanonicalSMILES,InChI/JSON"
                            ]
                            
                            for poll_url in poll_urls:
                                try:
                                    poll_response = self.session.get(poll_url, timeout=self.timeout)
                                    if poll_response.status_code == 200:
                                        poll_data = poll_response.json()
                                        
                                        # If we got CIDs, convert them to properties
                                        if "IdentifierList" in poll_data and "CID" in poll_data["IdentifierList"]:
                                            cids = poll_data["IdentifierList"]["CID"][:10]  # Limit to 10
                                            if cids:
                                                # Get properties for these CIDs
                                                cids_str = ",".join(map(str, cids))
                                                prop_url = f"{self.base_url}/compound/cid/{cids_str}/property/Title,MolecularFormula,MolecularWeight,CanonicalSMILES,InChI/JSON"
                                                prop_response = self.session.get(prop_url, timeout=self.timeout)
                                                if prop_response.status_code == 200:
                                                    return prop_response.json(), response_time + (poll_attempt + 1) * 2
                                        
                                        # If we got properties directly
                                        elif "PropertyTable" in poll_data:
                                            return poll_data, response_time + (poll_attempt + 1) * 2
                                            
                                except Exception:
                                    continue  # Try next URL
                        
                        # If all polls failed, return error
                        raise requests.RequestException(f"Async request timed out after multiple attempts")
                    else:
                        raise requests.RequestException(f"HTTP 202: {response.text}")
                except Exception as e:
                    raise requests.RequestException(f"Async request handling failed: {str(e)}")
            elif response.status_code == 503:
                raise requests.RequestException("PubChem service temporarily unavailable (503)")
            else:
                raise requests.RequestException(f"HTTP {response.status_code}: {response.text}")

        except requests.Timeout as e:
            response_time = time.time() - start_time
            raise requests.RequestException(f"Request timeout after {self.timeout}s") from e
        except requests.RequestException:
            response_time = time.time() - start_time
            raise

    def mcp_search_compounds(
        self,
        query: str = Field(description="Search query (compound name, CID, SMILES, InChI, etc.)"),
        search_type: Literal["name", "cid", "smiles", "inchi", "formula"] = Field(
            default="name",
            description="Type of search: name (compound name), cid (PubChem ID), smiles, inchi, or formula",
        ),
        max_results: int = Field(default=10, description="Maximum number of results to return (1-100)", ge=1, le=100),
    ) :
        """Search for chemical compounds in PubChem database.

        Supports multiple search types:
        - Name: Search by common or IUPAC names
        - CID: Search by PubChem Compound ID
        - SMILES: Search by SMILES notation
        - InChI: Search by InChI identifier
        - Formula: Search by molecular formula

        Args:
            query: Search term or identifier
            search_type: Type of search to perform
            max_results: Maximum number of compounds to return

        Returns:
            ActionResponse with compound search results and metadata
        """
        try:
            # Handle FieldInfo objects
            if isinstance(query, FieldInfo):
                query = query.default
            if isinstance(search_type, FieldInfo):
                search_type = search_type.default
            if isinstance(max_results, FieldInfo):
                max_results = max_results.default

            if not query or not query.strip():
                raise ValueError("Search query is required")

            # Build API URL based on search type - use simpler endpoints to avoid async issues
            if search_type == "cid":
                url = f"{self.base_url}/compound/cid/{quote(str(query))}/property/Title,MolecularFormula,MolecularWeight/JSON"
            elif search_type == "name":
                url = f"{self.base_url}/compound/name/{quote(query)}/property/Title,MolecularFormula,MolecularWeight/JSON"
            elif search_type == "smiles":
                # Use CID lookup first for SMILES to avoid async issues
                url = f"{self.base_url}/compound/smiles/{quote(query)}/cids/JSON"
            elif search_type == "inchi":
                # Use CID lookup first for InChI to avoid async issues
                url = f"{self.base_url}/compound/inchi/{quote(query)}/cids/JSON"
            elif search_type == "formula":
                # Use CID lookup first for formula to avoid async issues
                url = f"{self.base_url}/compound/formula/{quote(query)}/cids/JSON"
            else:
                raise ValueError(f"Unsupported search type: {search_type}")

            # Make API request with retry for formula searches
            max_retries = 5
            data = None
            response_time = None
            for attempt in range(max_retries):
                try:
                    data, response_time = self._make_request(url)
                    break
                except requests.RequestException as e:
                    if ("asynchronously" in str(e) or "timed out" in str(e)) and attempt < max_retries - 1:
                        time.sleep(10)  # Wait 10 seconds before retry
                        continue
                    else:
                        raise e

            # Parse results - handle both property and CID responses
            compounds = []
            if data:
                if "PropertyTable" in data and "Properties" in data["PropertyTable"]:
                    # Direct property response
                    properties_list = data["PropertyTable"]["Properties"][:max_results]
                    for prop in properties_list:
                        compound = CompoundData(
                            cid=prop.get("CID"),
                            name=prop.get("Title"),
                            molecular_formula=prop.get("MolecularFormula"),
                            molecular_weight=prop.get("MolecularWeight"),
                            smiles=prop.get("CanonicalSMILES"),
                            inchi=prop.get("InChI"),
                        )
                        compounds.append(compound)
                elif "IdentifierList" in data and "CID" in data["IdentifierList"]:
                    # CID response - need to get properties separately
                    cids = data["IdentifierList"]["CID"][:max_results]
                    if cids:
                        cids_str = ",".join(map(str, cids))
                        prop_url = f"{self.base_url}/compound/cid/{cids_str}/property/Title,MolecularFormula,MolecularWeight/JSON"
                        try:
                            prop_data, _ = self._make_request(prop_url)
                            if prop_data and "PropertyTable" in prop_data and "Properties" in prop_data["PropertyTable"]:
                                for prop in prop_data["PropertyTable"]["Properties"]:
                                    compound = CompoundData(
                                        cid=prop.get("CID"),
                                        name=prop.get("Title"),
                                        molecular_formula=prop.get("MolecularFormula"),
                                        molecular_weight=prop.get("MolecularWeight"),
                                    )
                                    compounds.append(compound)
                        except Exception as e:
                            print(f"Warning: Could not get properties for CIDs: {e}")
                            # Fallback: just return CIDs without full properties
                            for cid in cids:
                                compound = CompoundData(cid=cid)
                                compounds.append(compound)

            # Format results for LLM
            if compounds:
                result_lines = [f"Found {len(compounds)} compound(s) for query '{query}':\n"]

                for i, compound in enumerate(compounds, 1):
                    result_lines.append(f"{i}. **{compound.name}** (CID: {compound.cid})")
                    result_lines.append(f"   - Formula: {compound.molecular_formula}")
                    result_lines.append(f"   - Molecular Weight: {compound.molecular_weight} g/mol")
                    result_lines.append(f"   - SMILES: {compound.smiles}")
                    if compound.inchi:
                        result_lines.append(
                            f"   - InChI: {compound.inchi[:100]}..."
                            if len(compound.inchi) > 100
                            else f"   - InChI: {compound.inchi}"
                        )
                    result_lines.append("")

                message = "\n".join(result_lines)
            else:
                message = f"No compounds found for query '{query}' using search type '{search_type}'"
            # Prepare metadata
            metadata = PubChemMetadata(
                query_type=search_type,
                query_value=query,
                api_endpoint=url,
                response_time=response_time,
                total_results=len(compounds),
                timestamp=datetime.now().isoformat(),
            )
            return {
                "result": message,
                "success": True,
                "metadata": metadata.model_dump()
            }

        except ValueError as e:
            print(f"Invalid input: {str(e)}")
            return {
                "error": f"Invalid input: {str(e)}",
                "success": False,
                "metadata": PubChemMetadata(
                    query_type=search_type,
                    query_value=query,
                    api_endpoint=url,
                    response_time=response_time,
                    total_results=len(compounds),
                    timestamp=datetime.now().isoformat(),
                    error_type="invalid_input"
                ).model_dump(),
                "traceback": traceback.format_exc()
            }
        except requests.RequestException as e:
            print(f"PubChem API error: {str(e)}")
            return {
                "error": f"PubChem API error: {str(e)}",
                "success": False,
                "metadata": PubChemMetadata(
                    query_type=search_type,
                    query_value=query,
                    api_endpoint=url,
                    response_time=response_time or 0.0,
                    total_results=0,
                    timestamp=datetime.now().isoformat(),
                    error_type="pubchem_api_error"
                ).model_dump(),
                "traceback": traceback.format_exc()
            }
        except Exception as e:
            print(f"Search failed: {str(e)}: {traceback.format_exc()}")
            return {
                "error": f"Search failed: {str(e)}",
                "success": False,
                "metadata": PubChemMetadata(
                    query_type=search_type,
                    query_value=query,
                    api_endpoint=url,
                    response_time=response_time or 0.0,
                    total_results=0,
                    timestamp=datetime.now().isoformat(),
                    error_type="search_failed"
                ).model_dump(),
                "traceback": traceback.format_exc()
            }

    def mcp_get_compound_synonyms(
        self,
        cid: int = Field(description="PubChem Compound ID (CID)"),
        max_synonyms: int = Field(default=20, description="Maximum number of synonyms to return (1-100)", ge=1, le=100),
    ) :
        """Retrieve synonyms and alternative names for a PubChem compound.

        Args:
            cid: PubChem Compound ID
            max_synonyms: Maximum number of synonyms to return

        Returns:
            ActionResponse with compound synonyms and metadata
        """
        try:
            # Handle FieldInfo objectss
            if isinstance(cid, FieldInfo):
                cid = cid.default
            if isinstance(max_synonyms, FieldInfo):
                max_synonyms = max_synonyms.default

            if not cid or cid <= 0:
                raise ValueError("Valid PubChem CID is required")

            # Build API URL for synonyms
            url = f"{self.base_url}/compound/cid/{cid}/synonyms/JSON"

            # Make API request
            data = None
            response_time = None
            synonyms = []
            try:
                data, response_time = self._make_request(url)
            except requests.RequestException:
                raise  # Re-raise to be caught by outer exception handlers

            # Parse synonyms
            if data and "InformationList" in data and "Information" in data["InformationList"]:
                info_list = data["InformationList"]["Information"]
                if info_list and "Synonym" in info_list[0]:
                    synonyms = info_list[0]["Synonym"][:max_synonyms]

            # Format results for LLM
            if synonyms:
                result_lines = [f"Found {len(synonyms)} synonym(s) for CID {cid}:\n"]

                for i, synonym in enumerate(synonyms, 1):
                    result_lines.append(f"{i}. {synonym}")

                message = "\n".join(result_lines)
            else:
                message = f"No synonyms found for CID {cid}"
            # Prepare metadata
            metadata = PubChemMetadata(
                query_type="synonyms",
                query_value=str(cid),
                api_endpoint=url,
                response_time=response_time,
                total_results=len(synonyms),
                timestamp=datetime.now().isoformat(),
            )
            return {
                "result": message,
                "success": True,
                "metadata": metadata.model_dump()
            }

        except ValueError as e:
            print(f"Invalid input: {str(e)}")
            return {
                "error": f"Invalid input: {str(e)}",
                "success": False,
                "metadata": PubChemMetadata(
                    query_type="synonyms",
                    query_value=str(cid) if cid else None,
                    api_endpoint=url if 'url' in locals() else "",
                    response_time=response_time or 0.0,
                    total_results=0,
                    timestamp=datetime.now().isoformat(),
                    error_type="invalid_input"
                ).model_dump(),
                "traceback": traceback.format_exc()
            }
        except requests.RequestException as e:
            print(f"PubChem API error: {str(e)}")
            return {
                "error": f"PubChem API error: {str(e)}",
                "success": False,
                "metadata": PubChemMetadata(
                    query_type="synonyms",
                    query_value=str(cid) if cid else None,
                    api_endpoint=url if 'url' in locals() else "",
                    response_time=response_time or 0.0,
                    total_results=0,
                    timestamp=datetime.now().isoformat(),
                    error_type="pubchem_api_error"
                ).model_dump(),
                "traceback": traceback.format_exc()
            }
        except Exception as e:
            print(f"Synonym retrieval failed: {str(e)}: {traceback.format_exc()}")
            return {
                "error": f"Synonym retrieval failed: {str(e)}",
                "success": False,
                "metadata": PubChemMetadata(
                    query_type="synonyms",
                    query_value=str(cid) if cid else None,
                    api_endpoint=url if 'url' in locals() else "",
                    response_time=response_time or 0.0,
                    total_results=0,
                    timestamp=datetime.now().isoformat(),
                    error_type="synonym_retrieval_failed"
                ).model_dump(),
                "traceback": traceback.format_exc()
            }

    def mcp_get_compound_properties(
        self,
        cid: int = Field(description="PubChem Compound ID (CID)"),
        properties: list[str] = Field(
            default=[
                "MolecularWeight",
                "MolecularFormula",
                "CanonicalSMILES",
                "InChI",
                "XLogP",
                "TPSA",
                "HBondDonorCount",
                "HBondAcceptorCount",
            ],
            description="List of properties to retrieve (e.g., MolecularWeight, XLogP, TPSA)",
        ),
    ) :
        """Retrieve detailed chemical properties for a PubChem compound.

        Common properties include:
        - MolecularWeight: Molecular weight in g/mol
        - MolecularFormula: Chemical formula
        - CanonicalSMILES: SMILES notation
        - InChI: InChI identifier
        - XLogP: Partition coefficient
        - TPSA: Topological polar surface area
        - HBondDonorCount: Hydrogen bond donor count
        - HBondAcceptorCount: Hydrogen bond acceptor count

        Args:
            cid: PubChem Compound ID
            properties: List of property names to retrieve

        Returns:
            ActionResponse with compound properties and metadata
        """
        try:
            # Handle FieldInfo objects
            if isinstance(cid, FieldInfo):
                cid = cid.default
            if isinstance(properties, FieldInfo):
                properties = properties.default

            if not cid or cid <= 0:
                raise ValueError("Valid PubChem CID is required")

            if not properties:
                properties = ["MolecularWeight", "MolecularFormula", "CanonicalSMILES"]

            # Build API URL for properties
            props_str = ",".join(properties)
            url = f"{self.base_url}/compound/cid/{cid}/property/{props_str}/JSON"

            # Make API request
            data = None
            response_time = None
            compound_props = {}
            try:
                data, response_time = self._make_request(url)
            except requests.RequestException:
                raise  # Re-raise to be caught by outer exception handlers

            # Parse properties
            if data and "PropertyTable" in data and "Properties" in data["PropertyTable"]:
                props_data = data["PropertyTable"]["Properties"][0]
                compound_props = {k: v for k, v in props_data.items() if k != "CID"}

            # Format results for LLM
            if compound_props:
                result_lines = [f"Properties for PubChem CID {cid}:\n"]

                for prop_name, prop_value in compound_props.items():
                    if prop_name == "InChI" and isinstance(prop_value, str) and len(prop_value) > 100:
                        result_lines.append(f"**{prop_name}**: {prop_value[:100]}...")
                    else:
                        result_lines.append(f"**{prop_name}**: {prop_value}")

                message = "\n".join(result_lines)
            else:
                message = f"No properties found for CID {cid}"
            # Prepare metadata
            metadata = PubChemMetadata(
                query_type="properties",
                query_value=str(cid),
                api_endpoint=url,
                response_time=response_time,
                total_results=len(compound_props),
                timestamp=datetime.now().isoformat(),
            )
            return {
                "result": message,
                "success": True,
                "metadata": metadata.model_dump()
            }

        except ValueError as e:
            print(f"Invalid input: {str(e)}")
            return {
                "error": f"Invalid input: {str(e)}",
                "success": False,
                "metadata": PubChemMetadata(
                    query_type="properties",
                    query_value=str(cid) if cid else None,
                    api_endpoint=url if 'url' in locals() else "",
                    response_time=response_time or 0.0,
                    total_results=0,
                    timestamp=datetime.now().isoformat(),
                    error_type="invalid_input"
                ).model_dump(),
                "traceback": traceback.format_exc()
            }
        except requests.RequestException as e:
            print(f"PubChem API error: {str(e)}")
            return {
                "error": f"PubChem API error: {str(e)}",
                "success": False,
                "metadata": PubChemMetadata(
                    query_type="properties",
                    query_value=str(cid) if cid else None,
                    api_endpoint=url if 'url' in locals() else "",
                    response_time=response_time or 0.0,
                    total_results=0,
                    timestamp=datetime.now().isoformat(),
                    error_type="pubchem_api_error"
                ).model_dump(),
                "traceback": traceback.format_exc()
            }
        except Exception as e:
            print(f"Property retrieval failed: {str(e)}: {traceback.format_exc()}")
            return {
                "error": f"Property retrieval failed: {str(e)}",
                "success": False,
                "metadata": PubChemMetadata(
                    query_type="properties",
                    query_value=str(cid) if cid else None,
                    api_endpoint=url if 'url' in locals() else "",
                    response_time=response_time or 0.0,
                    total_results=0,
                    timestamp=datetime.now().isoformat(),
                    error_type="property_retrieval_failed"
                ).model_dump(),
                "traceback": traceback.format_exc()
            }

    def mcp_search_similar_compounds(
        self,
        cid: int = Field(description="PubChem Compound ID to find similar compounds for"),
        similarity_threshold: float = Field(
            default=0.9, description="Similarity threshold (0.0-1.0, higher = more similar)", ge=0.0, le=1.0
        ),
        max_results: int = Field(
            default=10, description="Maximum number of similar compounds to return (1-50)", ge=1, le=50
        ),
    ) :
        """Find structurally similar compounds using PubChem's similarity search.

        Uses Tanimoto similarity coefficient for 2D structure comparison.

        Args:
            cid: Reference compound CID for similarity search
            similarity_threshold: Minimum similarity score (0.0-1.0)
            max_results: Maximum number of similar compounds to return

        Returns:
            ActionResponse with similar compounds and metadata
        """
        try:
            # Handle FieldInfo objects
            if isinstance(cid, FieldInfo):
                cid = cid.default
            if isinstance(similarity_threshold, FieldInfo):
                similarity_threshold = similarity_threshold.default
            if isinstance(max_results, FieldInfo):
                max_results = max_results.default

            if not cid or cid <= 0:
                raise ValueError("Valid PubChem CID is required")

            # Build API URL for similarity search
            threshold_percent = int(similarity_threshold * 100)
            url = f"{self.base_url}/compound/fastsimilarity_2d/cid/{cid}/property/Title,MolecularFormula,MolecularWeight/JSON"
            params = {"Threshold": threshold_percent, "MaxRecords": max_results}

            # Make API request
            data, response_time = self._make_request(url, params)

            # Parse similar compounds
            similar_compounds: list[CompoundData] = []
            if data and "PropertyTable" in data and "Properties" in data["PropertyTable"]:
                properties_list = data["PropertyTable"]["Properties"]

                for prop in properties_list:
                    if prop.get("CID") != cid:  # Exclude the query compound itself
                        compound = CompoundData(
                            cid=prop.get("CID"),
                            name=prop.get("Title"),
                            molecular_formula=prop.get("MolecularFormula"),
                            molecular_weight=prop.get("MolecularWeight"),
                        )
                        similar_compounds.append(compound)

            # Format results for LLM
            if similar_compounds:
                result_lines = [
                    f"Found {len(similar_compounds)} compound(s) similar to CID {cid} (threshold: {similarity_threshold}):\n"
                ]

                for i, compound in enumerate(similar_compounds, 1):
                    result_lines.append(f"{i}. **{compound.name}** (CID: {compound.cid})")
                    result_lines.append(f"   - Formula: {compound.molecular_formula}")
                    result_lines.append(f"   - Molecular Weight: {compound.molecular_weight} g/mol")
                    result_lines.append("")

                message = "\n".join(result_lines)
            else:
                message = f"No similar compounds found for CID {cid} with similarity threshold {similarity_threshold}"
                
            metadata = PubChemMetadata(
                query_type="similarity",
                query_value=str(cid),
                api_endpoint=url,
                response_time=response_time,
                total_results=len(similar_compounds),
                timestamp=datetime.now().isoformat(),
            )


            return {
                "result": message,
                "success": True,
                "metadata": metadata.model_dump()
            }

        except ValueError as e:
            print(f"Invalid input: {str(e)}")
            return {
                "error": f"Invalid input: {str(e)}",
                "success": False,
                "metadata": PubChemMetadata(
                    query_type="similarity",
                    query_value=str(cid),
                    api_endpoint=url,
                    response_time=response_time,
                    total_results=len(similar_compounds),
                    timestamp=datetime.now().isoformat(),
                    error_type="invalid_input"
                ).model_dump(),
                "traceback": traceback.format_exc()
            }
        except requests.RequestException as e:
            print(f"PubChem API error: {str(e)}")
            return {
                "error": f"PubChem API error: {str(e)}",
                "success": False,
                "metadata": PubChemMetadata(
                    query_type="similarity",
                    query_value=str(cid),
                    api_endpoint=url,
                    response_time=response_time,
                    total_results=len(similar_compounds),
                    timestamp=datetime.now().isoformat(),
                    error_type="pubchem_api_error"
                ).model_dump(),
                "traceback": traceback.format_exc()
            }
        except Exception as e:
            print(f"Similarity search failed: {str(e)}: {traceback.format_exc()}")
            return {
                "error": f"Similarity search failed: {str(e)}",
                "success": False,
                "metadata": PubChemMetadata(
                    query_type="similarity",
                    query_value=str(cid),
                    api_endpoint=url,
                    response_time=response_time,
                    total_results=len(similar_compounds),
                    timestamp=datetime.now().isoformat(),
                    error_type="similarity_search_failed"
                ).model_dump(),
                "traceback": traceback.format_exc()
            }

    def run(
        self,
        operation: str = Field(default="search_compounds", description="The operation to perform"),
        query: str = Field(default=None, description="Search query for compound search"),
        cid: int = Field(default=None, description="PubChem Compound ID"),
        search_type: str = Field(default="name", description="Type of search for compounds"),
        max_results: int = Field(default=10, description="Maximum number of results"),
        max_synonyms: int = Field(default=20, description="Maximum number of synonyms"),
        properties: list = Field(default=None, description="List of properties to retrieve"),
        similarity_threshold: float = Field(default=0.9, description="Similarity threshold"),
        output_format: str = Field(default="json", description="Output format"),
    ):
        """Unified PubChem chemistry search interface.

        This tool provides comprehensive PubChem operations through a single interface:
        - Compound search with flexible criteria
        - Chemical property retrieval
        - Synonym and alternative name lookup
        - Chemical similarity search
        - Service capabilities and configuration

        Args:
            operation: The operation to perform (default: search_compounds)
            query: Search query (required for search_compounds operation)
            cid: PubChem Compound ID (required for get_synonyms, get_properties, search_similar operations)
            search_type: Type of search for compounds
            max_results: Maximum number of results to return
            max_synonyms: Maximum number of synonyms to return
            properties: List of properties to retrieve
            similarity_threshold: Similarity threshold for similarity search
            output_format: Format for the response output

        Returns:
            ActionResponse with operation results and metadata
        """
        # Handle FieldInfo objects
        if isinstance(operation, FieldInfo):
            operation = operation.default
        if isinstance(query, FieldInfo):
            query = query.default
        if isinstance(cid, FieldInfo):
            cid = cid.default
        if isinstance(search_type, FieldInfo):
            search_type = search_type.default
        if isinstance(max_results, FieldInfo):
            max_results = max_results.default
        if isinstance(max_synonyms, FieldInfo):
            max_synonyms = max_synonyms.default
        if isinstance(properties, FieldInfo):
            properties = properties.default
        if isinstance(similarity_threshold, FieldInfo):
            similarity_threshold = similarity_threshold.default
        if isinstance(output_format, FieldInfo):
            output_format = output_format.default

        try:
            # Validate required parameters for each operation
            if operation == "search_compounds":
                if not query:
                    return {
                        "error": "Error: 'query' parameter is REQUIRED for 'search_compounds' operation. Example: tool.run(operation='search_compounds', query='aspirin')",
                        "metadata": {"error_type": "missing_query", "operation": "search_compounds"},
                        "success": False
                    }
                return self.mcp_search_compounds(query, search_type, max_results)

            elif operation == "get_synonyms":
                if not cid:
                    return {
                        "error": "Error: 'cid' parameter is REQUIRED for 'get_synonyms' operation. Example: tool.run(operation='get_synonyms', cid=2244)",
                        "metadata": {"error_type": "missing_cid", "operation": "get_synonyms"},
                        "success": False
                    }
                return self.mcp_get_compound_synonyms(cid, max_synonyms)

            elif operation == "get_properties":
                if not cid:
                    return {
                        "error": "Error: 'cid' parameter is REQUIRED for 'get_properties' operation. Example: tool.run(operation='get_properties', cid=2244)",
                        "metadata": {"error_type": "missing_cid", "operation": "get_properties"},
                        "success": False
                    }
                if properties is None:
                    properties = [
                        "MolecularWeight",
                        "MolecularFormula",
                        "CanonicalSMILES",
                        "InChI",
                        "XLogP",
                        "TPSA",
                        "HBondDonorCount",
                        "HBondAcceptorCount",
                    ]
                return self.mcp_get_compound_properties(cid, properties)

            elif operation == "search_similar":
                if not cid:
                    return {
                        "error": "Error: 'cid' parameter is REQUIRED for 'search_similar' operation. Example: tool.run(operation='search_similar', cid=2244)",
                        "metadata": {"error_type": "missing_cid", "operation": "search_similar"},
                        "success": False
                    }
                return self.mcp_search_similar_compounds(cid, similarity_threshold, max_results)

            else:
                return {
                    "error": f"Error: Unknown operation '{operation}'. Supported operations: 'search_compounds' (requires query), 'get_synonyms' (requires cid), 'get_properties' (requires cid), 'search_similar' (requires cid)",
                    "metadata": {"error_type": "unknown_operation", "operation": operation},
                    "success": False
                }

        except Exception as e:
            error_msg = f"Failed to execute {operation} operation: {str(e)}"
            print(f"Chemistry operation error: {traceback.format_exc()}")

            return {
                "error": error_msg,
                "success": False,
                "metadata": PubChemMetadata(
                    query_type=operation,
                    query_value=query,
                    error_type="search_failed"
                ).model_dump(),
                "traceback": traceback.format_exc()
            }

    def test(self, tool_test: str="chemistry_search", file_location: str="chemistry_search", result_parameter: str="result", search_type: str="search_pattern"):
        return super().test(tool_test=tool_test, file_location=file_location, result_parameter=result_parameter, search_type=search_type)

# Example usage and entry point
if __name__ == "__main__":

    # Initialize and run the PubChem service
    try:
        service = Chemistry_Search_Tool()
        service.embed_tool()
        service.test(tool_test="chemistry_search", file_location="chemistry_search", result_parameter="result", search_type='search_pattern')

    except Exception as e:
        print(f"An error occurred: {e}: {traceback.format_exc()}")
