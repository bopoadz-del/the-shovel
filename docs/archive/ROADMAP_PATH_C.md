# Cerebrum-Blocks Architecture Audit & Path C Roadmap

## Executive Summary

This document provides a comprehensive analysis of the current Cerebrum-Blocks architecture and a detailed implementation roadmap for **Path C: The Real Block System**. The current system has fundamental data flow problems that prevent proper block composition, and the frontend bypasses the orchestrator entirely. This roadmap outlines the work needed to build a proper block interface contract, data flow architecture, and chain builder UI.

---

## Part 1: Current State Analysis

### 1.1 Files Analyzed

| File | Purpose | Key Issues |
|------|---------|------------|
| `app/core/universal_base.py` | Base block class | No input/output type contracts, ui_schema is UI-only, no data validation |
| `app/blocks/orchestrator.py` | Chain execution | Only does sequential execution, no type checking, context is just "passed through" |
| `app/static/index.html` | Frontend chain logic | **Bypasses orchestrator entirely** - implements its own `runChain()` |
| `app/routers/chain.py` | API endpoints | Thin wrapper, doesn't validate block compatibility |
| `app/blocks/chat.py` | Example block | Returns ad-hoc dict structure, no schema |
| `app/blocks/pdf.py` | Example block | Returns `{"text": ..., "pages": ...}` - no type contract |
| `app/blocks/ocr.py` | Example block | Returns `{"text": ..., "confidence": ...}` - different structure |
| `app/containers/construction.py` | Container example | Routes by action, processes pre-extracted text from chain |

### 1.2 Current Architecture Diagram (Text)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CURRENT ARCHITECTURE                            │
│                    (Broken Data Flow - Path A/B Legacy)                 │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (index.html)                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  runChain() - BUILT INTO FRONTEND                                   │  │
│  │                                                                     │  │
│  │  for step in steps:                                                 │  │
│  │    result = await client.execute(step.block, context, params)      │  │
│  │    context = result.result  ←── NO TYPE CHECKING                    │  │
│  │    results.push({step, block, result: context})                   │  │
│  │                                                                     │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │ PROBLEM: Frontend directly calls /v1/execute for each    │   │  │
│  │  │ block. The orchestrator at /v1/chain is NEVER used for   │   │  │
│  │  │ the main chain flow! It's only there as an API endpoint. │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP POST /v1/execute (per block)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           API LAYER (chain.py)                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  /v1/chain endpoint exists but:                                     │  │
│  │  - Frontend never uses it                                           │  │
│  │  - No block compatibility validation                                │  │
│  │  - Just wraps orchestrator.execute()                                │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR BLOCK                              │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  process(steps, input_data):                                        │  │
│  │    context = input_data  ←── RAW DATA, NO SCHEMA                   │  │
│  │    for step in steps:                                               │  │
│  │      block = resolve(step.block)                                    │  │
│  │      result = await block.execute(context, step.params)            │  │
│  │      context = result.get("result", result)  ←── NO VALIDATION     │  │
│  │    return {steps_executed, final_output: context}                   │  │
│  │                                                                     │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │ PROBLEM: Orchestrator blindly passes context between      │   │  │
│  │  │ blocks. It has NO IDEA what data types each block       │   │  │
│  │  │ produces or consumes. Zero contract enforcement.        │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
        ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
        │  PDF Block    │ │  OCR Block    │ │   Chat Block  │
        │  ─────────    │ │  ─────────    │ │   ─────────   │
        │  Returns:     │ │  Returns:     │ │   Returns:    │
        │  {            │ │  {            │ │   {           │
        │    text: str, │ │    text: str, │ │    text: str, │
        │    pages: int │ │    confidence │ │    provider,  │
        │  }            │ │    : float    │ │    tokens     │
        │               │ │  }            │ │  }            │
        └───────────────┘ └───────────────┘ └───────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌───────────────────────┐       ┌───────────────────────┐
        │   PROBLEM: No two    │       │   PROBLEM: Chat block │
        │   blocks agree on    │       │   expects "message"   │
        │   output structure!  │       │   but receives dict   │
        │                      │       │   from PDF/OCR        │
        │   PDF returns text   │       │   {text: ...,         │
        │   in "text" field.   │       │    pages: ...}       │
        │   OCR returns text   │       │                      │
        │   in "text" field.   │       │   Chat.process()      │
        │   Chat expects str   │       │   does:               │
        │   as input_data.     │       │   message = input_data│
        │                      │       │   if isinstance(...)  │
        │   How do we wire     │       │   else str(input_data)│
        │   PDF → Chat?        │       │   ← HACK!             │
        │   (Dict → String?)   │       │                      │
        └───────────────────────┘       └───────────────────────┘
```

### 1.3 Current Problems Documented

#### Problem 1: No Block Interface Contract

**Current State:**
```python
class ChatBlock(UniversalBlock):
    ui_schema = {
        "input": {"type": "text", ...},   # ← UI hint only
        "output": {"type": "text", ...}   # ← UI hint only
    }
    
    async def process(self, input_data: Any, params: Dict) -> Dict:
        # Can receive ANYTHING - no validation
        message = input_data if isinstance(input_data, str) else str(input_data)
        # Returns ad-hoc dict
        return {"status": "success", "text": "...", "tokens": {...}}
```

**What's Missing:**
- No declaration of what the block **requires** as input
- No declaration of what the block **produces** as output
- No validation that inputs match requirements
- No way for downstream blocks to know what they'll receive

#### Problem 2: Frontend Bypasses Orchestrator

**Current State (in index.html):**
```javascript
// Frontend implements its OWN chain logic!
async function runChain(label, steps, onProgress) {
    let context = null;
    const results = [];
    for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        // Direct API call per block - orchestrator is ignored
        const result = await client.execute(step.block, stepInput, step.params || {});
        context = result.result || result;  // Just grab whatever
        results.push({ step: i, block: step.block, result: context });
    }
    return { finalOutput: context, results };
}
```

**What's Wrong:**
- The `/v1/chain` endpoint exists but is **never used** for the main flow
- Chain logic is duplicated between frontend and orchestrator
- No single source of truth for chain execution
- Frontend must know how to wire blocks together

#### Problem 3: Data Structure Mismatch Between Blocks

**Example Chain: PDF → Construction → Chat**

```javascript
// Frontend manually wires this in selectDriveFile()
const result = await runChain(fileName, [
    { 
        block: 'pdf', 
        input: { action: 'extract_text', file_path: filePath },
        // PDF returns: {status, text, pages, filename}
    },
    { 
        block: 'construction', 
        input: (prev, allResults) => {
            // HACK: Construction expects {file_path, extracted_text}
            const ocrText = allResults[0]?.text || '';  // Knows PDF output structure!
            return { 
                file_path: filePath, 
                extracted_text: ocrText,  // Manual field mapping
                action: 'process_document' 
            };
        }
    },
    { 
        block: 'chat', 
        input: (prev, allResults) => {
            // HACK: Chat expects string, construction returns complex object
            const extracted = allResults[0]?.text || '';
            const construction = allResults[1];
            const constructionSummary = typeof construction === 'object' 
                ? JSON.stringify(construction, null, 2).substring(0, 2000)
                : String(construction).substring(0, 2000);
            // Manual construction of prompt
            return `You are a construction AI assistant...\n${extracted}\n${constructionSummary}`;
        }
    }
], ...);
```

**Problems:**
1. Frontend hardcodes knowledge of each block's output structure
2. No type checking - if PDF changes its output, chain breaks silently
3. Manual field mapping scattered throughout frontend code
4. Chat block receives constructed string instead of structured data

#### Problem 4: Container Routing Breaks Chain Semantics

**Current State:**
```python
class ConstructionContainer(UniversalContainer):
    async def route(self, action: str, input_data: Any, params: Dict) -> Dict:
        routes = {
            "process_document": self.process_document,
            "extract_quantities": self.extract_quantities,
            # ... more routes
        }
        handler = routes.get(action, self._status)
        return await handler(input_data, params)
```

**Problems:**
- Containers don't follow the same interface as blocks
- The `action` parameter is a side-channel that bypasses normal flow
- Orchestrator has special-case code to reject containers
- Can't compose containers in chains meaningfully

#### Problem 5: No Runtime Type Checking

**Current orchestrator logic:**
```python
# Just passes context through with no validation
result = await block.execute(context, step_params)
context = result.get("result", result)  # Whatever!
```

**Should be:**
```python
# Validate output matches declared schema
output = await block.execute(validated_input)
output_schema.validate(output)  # ← MISSING
# Check compatibility with next block
next_block.input_schema.validate(output)  # ← MISSING
```

---

## Part 2: Path C Roadmap - The Real Block System

### 2.1 Proposed Architecture Diagram (Text)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      PROPOSED ARCHITECTURE (Path C)                     │
│                     The Real Block System                               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         CHAIN BUILDER UI                               │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  Visual / Code-based Chain Construction                            │  │
│  │                                                                     │  │
│  │  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐   │  │
│  │  │   PDF       │ ──────► │Construction │ ──────► │    Chat     │   │  │
│  │  │  Block      │         │  Block      │         │   Block     │   │  │
│  │  └─────────────┘         └─────────────┘         └─────────────┘   │  │
│  │        │                       │                       │          │  │
│  │        ▼                       ▼                       ▼          │  │
│  │  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐   │  │
│  │  │  Output:    │  ✓      │  Input:     │  ✓      │  Input:     │   │  │
│  │  │  PDFFile    │ ───────►│  File +     │         │  String     │   │  │
│  │  │             │ match!  │  Extracted  │ match!  │  (prompt)   │   │  │
│  │  │  Produces:  │         │  Text       │         │             │   │  │
│  │  │  Extracted  │         │  Produces:  │         │  Receives:  │   │  │
│  │  │  Text       │         │  Analysis   │         │  String from│   │  │
│  │  │             │         │             │         │  upstream   │   │  │
│  │  └─────────────┘         └─────────────┘         └─────────────┘   │  │
│  │                                                                     │  │
│  │  Real-time validation: ✓ Compatible  ✗ Incompatible (red X)         │  │
│  │  Save/Load templates: template.json with versioned block refs      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ POST /v1/chain
                                    │ {steps: [{block, input_map, params}]}
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         API LAYER (chain.py)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  1. Parse chain request                                             │  │
│  │  2. Validate chain structure (DAG check, no cycles)                   │  │
│  │  3. Load block schemas from registry                                  │  │
│  │  4. Type-check entire chain before execution                          │  │
│  │  5. Return 400 if type mismatch, don't execute!                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ validated_chain
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     ENHANCED ORCHESTRATOR                               │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  execute_chain(steps, initial_input):                              │  │
│  │                                                                     │  │
│  │    # Build execution graph with typed edges                          │  │
│  │    graph = ExecutionGraph(steps)                                     │  │
│  │                                                                     │  │
│  │    for node in graph.topological_sort():                            │  │
│  │      block = registry.get(node.block_id)                          │  │
│  │                                                                     │  │
│  │      # Transform previous outputs to match input schema              │  │
│  │      validated_input = node.input_schema.transform(               │  │
│  │          graph.get_upstream_outputs(node)                          │  │
│  │      )                                                               │  │
│  │                                                                     │  │
│  │      # Execute with wrapped error handling                          │  │
│  │      result = await block.execute(validated_input, node.params)    │  │
│  │                                                                     │  │
│  │      # Validate output matches declared schema                       │  │
│  │      node.output_schema.validate(result)  ←── RUNTIME CHECK         │  │
│  │                                                                     │  │
│  │      # Store typed output for downstream blocks                      │  │
│  │      graph.set_node_output(node, result)                            │  │
│  │                                                                     │  │
│  │    return graph.get_final_output()                                   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
        ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
        │  PDF Block    │ │Construction   │ │   Chat Block  │
        │  ─────────    │ │Block          │ │   ─────────   │
        │               │ │───────        │ │               │
        │  INPUT_SCHEMA │ │               │ │  INPUT_SCHEMA │
        │  ───────────  │ │  INPUT_SCHEMA │ │  ───────────  │
        │  file_path:   │ │  ───────────  │ │  message:     │
        │    string     │ │  file_path:   │ │    string     │
        │  action:      │ │    string     │ │  context:     │
        │    enum       │ │  extracted_   │ │    object     │
        │               │ │    text:      │ │    (optional) │
        │  OUTPUT_SCHEMA│ │    string     │ │               │
        │  ───────────  │ │               │ │  OUTPUT_SCHEMA│
        │  text:        │ │  OUTPUT_SCHEMA│ │  ───────────  │
        │    string     │ │  ───────────  │ │  text:        │
        │  pages:       │ │  quantities:  │ │    string     │
        │    integer    │ │    object     │ │  tokens:      │
        │  filename:    │ │  analysis:    │ │    object     │
        │    string     │ │    string     │ │               │
        └───────────────┘ └───────────────┘ └───────────────┘
                    │               │               │
                    │               │               │
                    ▼               ▼               ▼
        ┌───────────────────────────────────────────────────────────┐
        │              SCHEMA REGISTRY (Central)                     │
        │  ┌─────────────────────────────────────────────────────┐  │
        │  │  {                                                  │  │
        │  │    "pdf": {                                         │  │
        │  │      "input": PDFFileSchema,                        │  │
        │  │      "output": PDFOutputSchema,                     │  │
        │  │      "version": "1.2.0"                             │  │
        │  │    },                                               │  │
        │  │    "chat": {                                        │  │
        │  │      "input": ChatInputSchema,                      │  │
        │  │      "output": ChatOutputSchema,                    │  │
        │  │      "version": "1.4.0"                             │  │
        │  │    }                                                │  │
        │  │  }                                                  │  │
        │  └─────────────────────────────────────────────────────┘  │
        └───────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    DATA FLOW TRANSFORMATION PIPE                        │
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐  │
│  │ PDF Output  │───►│  Mapper     │───►│Construction │───►│  Mapper  │  │
│  │ ──────────  │    │ ──────────  │    │ Input       │    │ ──────── │  │
│  │ {           │    │ Extracts    │    │ ──────────  │    │ Converts │  │
│  │   text: "..│───►│ text field  │───►│ {           │    │ analysis │  │
│  │   pages: 5  │    │ for         │    │   file_path,│    │ to       │  │
│  │ }           │    │ extracted_  │    │   extracted │    │ string   │  │
│  │             │    │ text param  │    │   _text     │    │ for Chat │  │
│  └─────────────┘    └─────────────┘    │ }           │    └──────────┘  │
│                                        └─────────────┘                  │
│  The mapper is generated from schema compatibility analysis:            │
│  - Field extraction: output.text → input.extracted_text                  │
│  - Type conversion: object → string (via JSON stringify or template)    │
│  - Validation: ensure required fields present                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Components

#### Component 1: Block Interface Contract

**New Base Class with Schema Support:**
```python
from pydantic import BaseModel, Field
from typing import Type, Generic, TypeVar

TInput = TypeVar('TInput', bound=BaseModel)
TOutput = TypeVar('TOutput', bound=BaseModel)

class TypedBlock(UniversalBlock, Generic[TInput, TOutput]):
    """
    Block with declared input/output schemas.
    All blocks MUST define these for Path C compatibility.
    """
    
    # Required: Pydantic model classes
    input_schema: Type[TInput]
    output_schema: Type[TOutput]
    
    # Optional: Schema version for migrations
    schema_version: str = "1.0.0"
    
    # Error handling standard
    error_schema: Type[BaseModel] = BlockError
    
    async def process(self, input_data: TInput) -> TOutput:
        """
        Process now receives VALIDATED input model, not raw dict.
        Must return output model instance.
        """
        raise NotImplementedError()
    
    async def execute(self, raw_input: Any, params: Dict = None) -> Dict:
        """
        Wrapper handles validation, execution, error standardization.
        """
        try:
            # Validate input against schema
            validated = self.input_schema.model_validate(raw_input)
            
            # Execute with typed input
            output = await self.process(validated)
            
            # Validate output (runtime check)
            validated_output = self.output_schema.model_validate(output)
            
            return {
                "status": "success",
                "result": validated_output.model_dump(),
                "schema_version": self.schema_version
            }
            
        except ValidationError as e:
            return {
                "status": "error",
                "error_type": "validation_error",
                "error": str(e),
                "schema": self.input_schema.model_json_schema()
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": "execution_error",
                "error": str(e),
                "block": self.name
            }
```

**Example Block Migration (PDF Block):**
```python
from pydantic import BaseModel, Field

class PDFFileInput(BaseModel):
    """Standard input for PDF block."""
    file_path: str = Field(..., description="Path to PDF file")
    action: str = Field(default="extract_text", enum=["extract_text", "metadata"])
    page_range: tuple[int, int] | None = None

class PDFTextOutput(BaseModel):
    """Standard output when extracting text."""
    text: str = Field(..., description="Extracted text content")
    pages: int = Field(..., ge=0, description="Number of pages processed")
    filename: str = Field(..., description="Original filename")
    extraction_method: str = Field(default="pymupdf")

class PDFMetadataOutput(BaseModel):
    """Standard output for metadata extraction."""
    title: str | None = None
    author: str | None = None
    pages: int
    file_size: int

class PDFBlock(TypedBlock[PDFFileInput, PDFTextOutput]):
    name = "pdf"
    version = "2.0.0"  # Major version bump for schema compatibility
    schema_version = "2.0.0"
    
    input_schema = PDFFileInput
    output_schema = PDFTextOutput
    
    async def process(self, input_data: PDFFileInput) -> PDFTextOutput:
        # Now we know EXACTLY what we receive
        doc = fitz.open(input_data.file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        # Return typed output
        return PDFTextOutput(
            text=text[:20000],
            pages=len(doc),
            filename=os.path.basename(input_data.file_path),
            extraction_method="pymupdf"
        )
```

#### Component 2: Data Flow Architecture

**Schema Registry:**
```python
class SchemaRegistry:
    """Central registry for all block schemas."""
    
    _schemas: Dict[str, BlockSchemas] = {}
    
    @classmethod
    def register(cls, block_name: str, schemas: BlockSchemas):
        cls._schemas[block_name] = schemas
    
    @classmethod
    def get_compatibility(cls, 
        upstream: str, 
        downstream: str
    ) -> CompatibilityReport:
        """
        Analyze if upstream output can feed downstream input.
        Returns compatibility score and field mappings.
        """
        up_schema = cls._schemas[upstream].output
        down_schema = cls._schemas[downstream].input
        
        # Field-by-field analysis
        mappings = {}
        missing_required = []
        type_conflicts = []
        
        for field_name, field_info in down_schema.model_fields.items():
            if field_name in up_schema.model_fields:
                # Direct field match
                up_type = up_schema.model_fields[field_name].annotation
                down_type = field_info.annotation
                
                if up_type == down_type:
                    mappings[field_name] = field_name  # 1:1 mapping
                else:
                    # Check if convertible
                    if cls._is_convertible(up_type, down_type):
                        mappings[field_name] = f"{field_name} (converted)"
                    else:
                        type_conflicts.append(field_name)
            elif field_info.is_required():
                missing_required.append(field_name)
        
        return CompatibilityReport(
            compatible=len(missing_required) == 0,
            score=len(mappings) / len(down_schema.model_fields),
            mappings=mappings,
            missing=missing_required,
            conflicts=type_conflicts
        )
```

**Data Transformer:**
```python
class DataTransformer:
    """Transforms data between block output and input schemas."""
    
    @staticmethod
    def create_mapping(
        source_schema: Type[BaseModel],
        target_schema: Type[BaseModel],
        field_map: Dict[str, str | Callable]
    ) -> Callable:
        """
        Create a transformer function from field mappings.
        
        field_map: {
            "target_field": "source_field",  # direct mapping
            "other_field": lambda src: src.text[:1000],  # computed
        }
        """
        def transform(source_data: Dict) -> Dict:
            result = {}
            for target_field, mapping in field_map.items():
                if callable(mapping):
                    result[target_field] = mapping(source_data)
                else:
                    result[target_field] = source_data.get(mapping)
            return result
        
        return transform
```

#### Component 3: Chain Builder UI

**Chain Template Format:**
```json
{
  "version": "2.0.0",
  "name": "PDF Analysis Chain",
  "description": "Extract and analyze PDF documents",
  "steps": [
    {
      "id": "step_1",
      "block": "pdf",
      "version": ">=2.0.0",
      "input_source": "initial",  // Gets from chain input
      "params": {
        "action": "extract_text"
      }
    },
    {
      "id": "step_2",
      "block": "construction",
      "version": ">=3.0.0",
      "input_source": "step_1",  // Gets from previous step
      "input_mapping": {
        // Explicit field mapping
        "file_path": "filename",
        "extracted_text": "text",
        "action": "const:process_document"  // constant value
      },
      "params": {}
    },
    {
      "id": "step_3",
      "block": "chat",
      "version": ">=2.0.0",
      "input_source": "step_2",
      "input_mapping": {
        // Template transformation
        "message": "template:Analyze this construction data:\n{{analysis}}"
      },
      "params": {
        "model": "deepseek-chat"
      }
    }
  ],
  "output_from": "step_3"
}
```

**Validation Rules:**
1. All referenced blocks must exist in registry
2. Input mappings must reference valid source fields
3. Type compatibility checked at save time
4. Circular dependencies rejected
5. Missing required fields flagged

#### Component 4: Migration Strategy

**Phase 1: Schema Foundation (Weeks 1-2)**
- Create schema infrastructure
- Migrate core blocks (pdf, ocr, chat, image)
- Maintain backward compatibility layer

**Phase 2: Orchestrator Rewrite (Weeks 3-4)**
- New execution engine with type checking
- Graph-based chain execution
- Error propagation and recovery

**Phase 3: Container Unification (Weeks 5-6)**
- Convert containers to typed blocks
- Remove action-based routing
- Standard parameter passing

**Phase 4: Frontend Chain Builder (Weeks 7-8)**
- Visual chain construction
- Real-time validation
- Template save/load

**Phase 5: Legacy Deprecation (Weeks 9-10)**
- Remove old orchestrator code paths
- Frontend uses /v1/chain exclusively
- Documentation and examples

---

## Part 3: Week-by-Week Breakdown

### Week 1-2: Schema Infrastructure

**Goals:**
- Create schema base classes and registry
- Define error handling standards
- Build backward compatibility layer

**Files to Create:**
```
app/core/
  ├── schema_base.py          # TypedBlock, SchemaRegistry
  ├── data_transformer.py      # Field mapping, type conversion
  └── compatibility.py           # CompatibilityReport, type checking

app/schemas/                   # Central schema definitions
  ├── __init__.py
  ├── common.py               # Shared types (FilePath, Confidence, etc.)
  ├── block_schemas.py        # Registry of all block schemas
  └── migrations.py           # Schema version migrations
```

**Files to Modify:**
```
app/core/universal_base.py
  - Add optional schema support (don't break existing blocks)
  - Add deprecation warnings for untyped blocks
```

**Deliverables:**
- [ ] SchemaRegistry with registration/lookup
- [ ] TypedBlock base class
- [ ] Compatibility analyzer
- [ ] DataTransformer with field mapping
- [ ] Backward compatibility tests

### Week 3-4: Orchestrator Rewrite

**Goals:**
- New execution engine with graph-based chains
- Runtime type validation
- Proper error propagation

**Files to Create:**
```
app/core/
  ├── execution_graph.py        # DAG representation of chains
  ├── chain_executor.py         # New execution engine
  └── error_recovery.py         # Retry, fallback strategies

app/blocks/
  └── orchestrator_v2.py        # New orchestrator block
```

**Files to Modify:**
```
app/routers/chain.py
  - Switch to new orchestrator
  - Add pre-execution validation
  - Better error responses
```

**Deliverables:**
- [ ] ExecutionGraph with topological sort
- [ ] ChainExecutor with type checking
- [ ] Input/output validation at each step
- [ ] Error recovery (continue_on_error, retry)
- [ ] Metrics and logging per step

### Week 5-6: Block & Container Migration

**Goals:**
- Migrate all core blocks to typed versions
- Convert containers to standard block interface
- Ensure backward compatibility

**Blocks to Migrate (Priority Order):**
1. `pdf` → `pdf_v2` (with schema)
2. `ocr` → `ocr_v2`
3. `chat` → `chat_v2`
4. `image` → `image_v2`
5. `construction` container → `construction_v2` block

**Files to Create:**
```
app/blocks/
  ├── pdf_v2.py
  ├── ocr_v2.py
  ├── chat_v2.py
  ├── image_v2.py
  └── construction_v2.py        # Container as typed block

app/schemas/
  ├── pdf.py                    # PDFFileInput, PDFTextOutput
  ├── ocr.py                    # OCRInput, OCROutput
  ├── chat.py                   # ChatInput, ChatOutput
  ├── image.py                  # ImageInput, ImageOutput
  └── construction.py           # ConstructionInput, ConstructionOutput
```

**Deliverables:**
- [ ] All core blocks with schemas
- [ ] Container routing via parameters, not action
- [ ] Backward compatibility shims
- [ ] Schema documentation

### Week 7-8: Chain Builder UI

**Goals:**
- Visual chain construction interface
- Real-time compatibility validation
- Template save/load

**Files to Create:**
```
app/static/
  ├── chain-builder.html        # Main UI
  ├── js/
  │   ├── chain-editor.js       # Visual editor logic
  │   ├── block-palette.js      # Available blocks sidebar
  │   ├── connection-validator.js # Real-time validation
  │   └── template-manager.js   # Save/load chains
  └── css/
      └── chain-builder.css

app/routers/
  └── templates.py              # Chain template CRUD API
```

**Files to Modify:**
```
app/static/index.html
  - Add chain builder link/button
  - Keep current UI for simple queries
```

**Deliverables:**
- [ ] Drag-drop chain builder
- [ ] Block compatibility visualization (green/red connections)
- [ ] Template save/load to backend
- [ ] Export chain as JSON
- [ ] Import and validate external chains

### Week 9-10: Testing & Documentation

**Goals:**
- Comprehensive test suite
- Migration guide
- Performance validation

**Files to Create:**
```
tests/
  ├── test_schemas.py
  ├── test_compatibility.py
  ├── test_chain_executor.py
  └── test_data_transformer.py

docs/
  ├── MIGRATION_GUIDE.md
  ├── BLOCK_SCHEMA_SPEC.md
  └── CHAIN_BUILDER_MANUAL.md
```

**Deliverables:**
- [ ] 90%+ test coverage on new code
- [ ] Migration guide for existing block authors
- [ ] Performance benchmarks (should not be slower)
- [ ] Deprecation timeline for old API

---

## Part 4: Files to Modify/Create Summary

### New Files (19 total)

| Path | Purpose | Week |
|------|---------|------|
| `app/core/schema_base.py` | TypedBlock, SchemaRegistry | 1 |
| `app/core/data_transformer.py` | Field mapping, type conversion | 1 |
| `app/core/compatibility.py` | Compatibility analysis | 1 |
| `app/core/execution_graph.py` | DAG chain representation | 3 |
| `app/core/chain_executor.py` | New execution engine | 3 |
| `app/core/error_recovery.py` | Retry, fallback strategies | 3 |
| `app/schemas/common.py` | Shared types | 1 |
| `app/schemas/pdf.py` | PDF block schemas | 5 |
| `app/schemas/ocr.py` | OCR block schemas | 5 |
| `app/schemas/chat.py` | Chat block schemas | 5 |
| `app/schemas/construction.py` | Construction block schemas | 5 |
| `app/blocks/pdf_v2.py` | Typed PDF block | 5 |
| `app/blocks/ocr_v2.py` | Typed OCR block | 5 |
| `app/blocks/chat_v2.py` | Typed chat block | 5 |
| `app/blocks/construction_v2.py` | Container as typed block | 5 |
| `app/routers/templates.py` | Chain template API | 7 |
| `app/static/chain-builder.html` | Chain builder UI | 7 |
| `app/static/js/chain-editor.js` | Chain editor logic | 7 |
| `app/static/js/connection-validator.js` | Real-time validation | 7 |

### Modified Files (8 total)

| Path | Changes | Week |
|------|---------|------|
| `app/core/universal_base.py` | Add schema support, deprecation warnings | 1 |
| `app/blocks/orchestrator.py` | Mark as deprecated, redirect to v2 | 3 |
| `app/routers/chain.py` | Switch to new orchestrator | 3 |
| `app/static/index.html` | Add chain builder entry point | 7 |
| `app/blocks/__init__.py` | Register v2 blocks | 5 |
| `app/blocks/pdf.py` | Add deprecation notice | 5 |
| `app/blocks/chat.py` | Add deprecation notice | 5 |
| `app/containers/construction.py` | Add deprecation notice | 5 |

---

## Part 5: Risk Assessment

### High Risk Items

#### Risk 1: Breaking Changes to Existing Chains
- **Impact:** HIGH - All current frontend chain code breaks
- **Likelihood:** CERTAIN if not handled
- **Mitigation:** 
  - Keep old blocks working alongside v2
  - Gradual migration with deprecation warnings
  - Backward compatibility layer in UniversalBlock
- **Timeline:** Migrate over 4 weeks, deprecate over 8 weeks

#### Risk 2: Performance Degradation from Validation
- **Impact:** MEDIUM - Slower chain execution
- **Likelihood:** POSSIBLE
- **Mitigation:**
  - Cache validation results
  - Skip validation in production mode (optional)
  - Benchmark before/after
- **Timeline:** Week 3-4 performance testing

#### Risk 3: Complex Schema Definitions Discourage Block Authors
- **Impact:** MEDIUM - Fewer community blocks
- **Likelihood:** POSSIBLE
- **Mitigation:**
  - Provide schema generators from existing blocks
  - Good documentation and examples
  - Optional schemas (graceful degradation to untyped)
- **Timeline:** Week 1-2 tooling, Week 9-10 docs

### Medium Risk Items

#### Risk 4: Frontend Chain Builder Complexity
- **Impact:** MEDIUM - Longer development time
- **Likelihood:** LIKELY
- **Mitigation:**
  - Start with code-based chain builder (JSON editor with validation)
  - Add visual builder in Phase 2
  - Reuse existing drag-drop libraries
- **Timeline:** Week 7-8, may extend to Week 10

#### Risk 5: Container Conversion Complexity
- **Impact:** MEDIUM - Construction container is large
- **Likelihood:** POSSIBLE
- **Mitigation:**
  - Convert one action at a time
  - Keep old container working during migration
  - Extensive testing with real PDFs
- **Timeline:** Week 5-6, dedicated testing

### Low Risk Items

#### Risk 6: Schema Version Migration Headaches
- **Impact:** LOW - Version mismatches
- **Likelihood:** UNLIKELY
- **Mitigation:**
  - Strict semver following
  - Migration functions in migrations.py
  - Registry supports multiple versions

#### Risk 7: Type System Limitations
- **Impact:** LOW - Can't express some data types
- **Likelihood:** UNLIKELY
- **Mitigation:**
  - Pydantic is very flexible
  - Custom validators for complex types
  - Binary data as file references, not inline

### Risk Matrix Summary

| Risk | Impact | Likelihood | Mitigation Cost | Priority |
|------|--------|------------|-----------------|----------|
| Breaking changes | HIGH | CERTAIN | MEDIUM | P1 |
| Performance | MEDIUM | POSSIBLE | LOW | P2 |
| Schema complexity | MEDIUM | POSSIBLE | MEDIUM | P3 |
| Builder complexity | MEDIUM | LIKELY | HIGH | P4 |
| Container conversion | MEDIUM | POSSIBLE | MEDIUM | P5 |
| Version migration | LOW | UNLIKELY | LOW | P6 |
| Type limitations | LOW | UNLIKELY | LOW | P7 |

---

## Part 6: Success Metrics

### Technical Metrics
- [ ] All core blocks have schemas defined
- [ ] Chain validation catches 100% of type mismatches before execution
- [ ] Runtime validation overhead < 10ms per block
- [ ] Frontend uses /v1/chain exclusively (no direct block calls in chains)
- [ ] All existing chains work without modification (backward compatibility)

### User Experience Metrics
- [ ] Chain builder UI can construct chains without writing code
- [ ] Real-time validation shows compatibility issues immediately
- [ ] Chain templates can be saved and loaded
- [ ] Error messages clearly indicate which block and field failed

### Adoption Metrics
- [ ] 80%+ of new blocks use schemas
- [ ] Frontend chains migrated to new system within 8 weeks
- [ ] Zero production incidents due to type mismatches

---

## Appendix A: Example Schema Definitions

### A.1 PDF Block Schema (Complete)

```python
# app/schemas/pdf.py
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional, Tuple

class PDFFileInput(BaseModel):
    """Input schema for PDF block operations."""
    
    file_path: str = Field(
        ..., 
        description="Absolute or relative path to PDF file",
        examples=["/uploads/drawing.pdf", "./docs/spec.pdf"]
    )
    
    action: Literal["extract_text", "extract_metadata", "extract_tables"] = Field(
        default="extract_text",
        description="Operation to perform on the PDF"
    )
    
    page_range: Optional[Tuple[int, int]] = Field(
        default=None,
        description="Optional (start, end) page range. None = all pages",
        examples=[(1, 5), (10, 20)]
    )
    
    extract_options: dict = Field(
        default_factory=dict,
        description="Format-specific options (e.g., table detection settings)"
    )
    
    @validator('file_path')
    def validate_path(cls, v):
        if not v.endswith('.pdf'):
            raise ValueError('File must be a PDF')
        return v

class PDFTextOutput(BaseModel):
    """Output schema for text extraction."""
    
    text: str = Field(
        ..., 
        description="Extracted text content",
        max_length=100000  # Limit for performance
    )
    
    pages: int = Field(
        ..., 
        ge=0,
        description="Number of pages successfully processed"
    )
    
    filename: str = Field(
        ..., 
        description="Original filename without path"
    )
    
    extraction_method: Literal["pymupdf", "pdfplumber", "ocr_fallback"] = Field(
        default="pymupdf",
        description="Method used for extraction"
    )
    
    confidence: Optional[float] = Field(
        default=None,
        ge=0, le=1,
        description="Confidence score if OCR was used"
    )

class PDFMetadataOutput(BaseModel):
    """Output schema for metadata extraction."""
    
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    pages: int
    file_size_bytes: int
    encrypted: bool
    
# Union type for different outputs
PDFOutput = PDFTextOutput | PDFMetadataOutput
```

### A.2 Chat Block Schema (Complete)

```python
# app/schemas/chat.py
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict

class Message(BaseModel):
    """Individual message in conversation."""
    role: Literal["system", "user", "assistant"] = "user"
    content: str

class ChatInput(BaseModel):
    """Input schema for chat block."""
    
    message: str = Field(
        ..., 
        description="User message text",
        max_length=50000
    )
    
    conversation_history: List[Message] = Field(
        default_factory=list,
        description="Previous messages for context",
        max_length=50
    )
    
    context: Optional[Dict] = Field(
        default=None,
        description="Additional structured context (e.g., extracted document data)"
    )
    
    model: str = Field(
        default="deepseek-chat",
        description="LLM model to use"
    )
    
    max_tokens: int = Field(
        default=2048,
        ge=1, le=8192,
        description="Maximum tokens in response"
    )
    
    temperature: float = Field(
        default=0.7,
        ge=0, le=2,
        description="Sampling temperature"
    )
    
    stream: bool = Field(
        default=False,
        description="If true, return stream generator"
    )

class ChatOutput(BaseModel):
    """Output schema for chat block."""
    
    text: str = Field(
        ..., 
        description="Generated response text"
    )
    
    model: str = Field(
        ..., 
        description="Model that generated the response"
    )
    
    provider: Literal["deepseek", "openai", "anthropic"] = Field(
        default="deepseek",
        description="API provider used"
    )
    
    tokens: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage stats: {prompt, completion, total}"
    )
    
    finish_reason: Literal["stop", "length", "content_filter"] = Field(
        default="stop",
        description="Why generation stopped"
    )
```

### A.3 Construction Block Schema (Complete)

```python
# app/schemas/construction.py
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any

class ConstructionInput(BaseModel):
    """Input schema for construction block."""
    
    # Primary inputs - can come from PDF, OCR, or direct upload
    file_path: Optional[str] = Field(
        default=None,
        description="Path to document (PDF, image, schedule file)"
    )
    
    extracted_text: Optional[str] = Field(
        default=None,
        description="Pre-extracted text content (from upstream PDF/OCR)"
    )
    
    # Operation selection (replaces container routing)
    operation: Literal[
        "process_document",
        "extract_quantities", 
        "analyze_spec",
        "cost_estimate",
        "schedule_risk",
        "contract_review",
        "safety_audit",
        "carbon_report"
    ] = Field(
        default="process_document",
        description="Analysis operation to perform"
    )
    
    # Operation-specific parameters
    doc_type: Literal["auto", "drawing", "specification", "contract", 
                     "schedule", "bom", "report", "bim", "image"] = "auto"
    
    project_location: str = Field(
        default="US National Average",
        description="For cost estimation - affects unit rates"
    )
    
    include_raw_output: bool = Field(
        default=False,
        description="Include full extracted text in output"
    )

class Measurement(BaseModel):
    """Individual measurement extracted from drawing."""
    type: Literal["dimension", "count", "area", "volume", "length"]
    value: float
    unit: str
    raw_text: str
    context: str  # Surrounding text
    confidence: float = Field(ge=0, le=1)

class Quantity(BaseModel):
    """Calculated construction quantity."""
    material: str
    quantity: float
    unit: str
    source_measurements: List[str]  # References to raw measurements

class RiskItem(BaseModel):
    """Identified project risk."""
    category: Literal["schedule", "cost", "quality", "safety", "contract"]
    description: str
    probability: Literal["low", "medium", "high"]
    impact: Literal["low", "medium", "high", "critical"]
    mitigation: str
    source: str  # What triggered this risk

class ConstructionOutput(BaseModel):
    """Output schema for construction block."""
    
    operation: str
    status: Literal["success", "error", "queued"]
    
    # Document analysis results
    doc_type: Optional[str] = None
    file_name: Optional[str] = None
    
    # Extracted data
    measurements: List[Measurement] = Field(default_factory=list)
    quantities: Dict[str, Quantity] = Field(default_factory=dict)
    detected_disciplines: List[str] = Field(default_factory=list)
    
    # Calculated results
    cost_estimate: Optional[Dict[str, float]] = None
    carbon_estimate: Optional[Dict[str, float]] = None
    
    # Risk analysis
    risks: List[RiskItem] = Field(default_factory=list)
    
    # Confidence scores
    confidence: Dict[str, float] = Field(default_factory=dict)
    
    # Raw output (if requested)
    raw_text: Optional[str] = None
    
    # Caching info
    cache_key: Optional[str] = None
    source: Literal["cache", "processor", "async_queue"] = "processor"
```

---

## Appendix B: Compatibility Examples

### B.1 PDF → Construction Compatibility Report

```python
# Generated by SchemaRegistry.get_compatibility("pdf", "construction")

CompatibilityReport(
    compatible=True,
    score=0.75,  # 3 of 4 fields mappable
    
    # Direct mappings (same name, compatible type)
    mappings={
        "file_path": Mapping(
            source_field="filename",
            target_field="file_path",
            transformation="path_join:uploads/"
        ),
        "extracted_text": Mapping(
            source_field="text",
            target_field="extracted_text",
            transformation="direct"
        ),
        "operation": Mapping(
            source_field=None,  # constant
            target_field="operation",
            transformation="const:process_document"
        ),
    },
    
    # Missing but optional
    missing_optional=["project_location", "include_raw_output"],
    
    # Missing and required
    missing_required=[],
    
    # Type conflicts
    conflicts=[],
    
    suggestions=[
        "Add 'doc_type' parameter to control processing mode",
        "Enable 'include_raw_output' to preserve full text"
    ]
)
```

### B.2 Construction → Chat Compatibility Report

```python
# Generated by SchemaRegistry.get_compatibility("construction", "chat")

CompatibilityReport(
    compatible=True,  # With transformation
    score=0.25,  # Needs significant transformation
    
    mappings={
        "message": Mapping(
            source_field=None,
            target_field="message",
            transformation="""template:Analyze this construction document:
            
Document: {{file_name}}
Type: {{doc_type}}

Extracted Quantities:
{{quantities | to_yaml}}

Risks Identified:
{{risks | map(attribute='description') | join('\n- ')}}

Please provide a summary and recommendations."""
        ),
        "context": Mapping(
            source_field=None,  # full object
            target_field="context",
            transformation="full_object"
        )
    },
    
    missing_optional=["conversation_history", "max_tokens", "temperature"],
    missing_required=[],
    conflicts=[],
    
    warnings=[
        "Large context may exceed token limits",
        "Consider using a template to format construction data for LLM"
    ]
)
```

---

## Conclusion

**Path C** represents a fundamental shift from ad-hoc data passing to a contract-based block system. While the migration requires significant work over 10 weeks, the benefits are substantial:

1. **Reliability:** Type checking prevents runtime errors from mismatched data
2. **Composability:** Blocks can be combined safely without hardcoded knowledge
3. **Discoverability:** Schemas document what each block does
4. **Tooling:** Chain builder UI can validate and assist users
5. **Maintainability:** Changes to block outputs are explicit and versioned

The key to success is maintaining **backward compatibility** throughout the migration, allowing gradual adoption rather than a big-bang rewrite.

---

*Document Version: 1.0*  
*Created: 2026-04-20*  
*Author: Subagent Architecture Audit*  
*Status: ROADMAP - Ready for Implementation*
