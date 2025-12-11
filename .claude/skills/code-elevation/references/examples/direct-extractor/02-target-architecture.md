# DirectExtractor - Target Architecture

## Design Principles

### Hexagonal Architecture (Ports & Adapters)
```
┌─────────────────────────────────────────────────────┐
│              Application Layer                      │
│  (DirectExtractionOrchestrator - thin coordinator)  │
└──────────────┬──────────────────────┬───────────────┘
               │                      │
       ┌───────▼────────┐    ┌────────▼──────────┐
       │  Domain Core   │    │  Domain Core      │
       │  (Business     │    │  (Pattern         │
       │   Logic)       │    │   Matching)       │
       └───────┬────────┘    └────────┬──────────┘
               │                      │
       ┌───────▼────────────────────┬─▼───────────┐
       │        Adapters             │             │
       │  (Infrastructure)           │             │
       └─────────────────────────────┴─────────────┘
```

### Layer Responsibilities

#### Domain Core (Pure Business Logic)
- `PatternMatcher`: Matches lines against regex patterns
- `FieldExtractor`: Extracts and converts field values
- `ItemPostProcessor`: Cleans and normalizes items
- `LinePreprocessor`: Joins multi-line items

**Characteristics:**
- No I/O
- No framework dependencies
- Easy to test
- Pure functions where possible

#### Application Layer (Orchestration)
- `DirectExtractionOrchestrator`: Coordinates domain objects
- Thin layer that wires dependencies
- No business logic

#### Adapters (Infrastructure)
- `FileSystemSchemaRepository`: Loads schemas from disk
- `PdfTextReader`: Reads text from PDFs
- `TextFileReader`: Reads text from .txt files

**Characteristics:**
- Implement interfaces defined by domain
- Can be swapped
- Tested via integration tests

## Class Structure

### 1. SchemaRepository (Port/Interface)
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class SchemaPattern:
    """Value object representing a regex pattern with field mappings."""
    name: str
    regex: str
    fields: List[Dict[str, Any]]

@dataclass
class ExtractionSchema:
    """Value object representing a complete extraction schema."""
    name: str
    patterns: List[SchemaPattern]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionSchema':
        """Factory method to create from dictionary."""
        patterns = [
            SchemaPattern(**p) for p in data.get("patterns", [])
        ]
        return cls(name=data.get("name", "unknown"), patterns=patterns)


class SchemaRepository(ABC):
    """Port: Interface for loading extraction schemas."""
    
    @abstractmethod
    def load(self, schema_name: str) -> ExtractionSchema:
        """Load a schema by name."""
        pass


class FileSystemSchemaRepository(SchemaRepository):
    """Adapter: Load schemas from JSON files on disk."""
    
    def __init__(self, search_paths: List[Path]):
        self.search_paths = search_paths
    
    def load(self, schema_name: str) -> ExtractionSchema:
        """Load schema from first matching file in search paths."""
        for base_path in self.search_paths:
            schema_path = base_path / f"{schema_name}.json"
            if schema_path.exists():
                with open(schema_path) as f:
                    data = json.load(f)
                return ExtractionSchema.from_dict(data)
        
        raise FileNotFoundError(
            f"Schema '{schema_name}' not found in: {self.search_paths}"
        )
```

**Benefits:**
- ✅ Easy to test (no file I/O in tests)
- ✅ Easy to extend (add DatabaseSchemaRepository, etc.)
- ✅ Single responsibility (schema loading only)

---

### 2. TextReader (Port/Interface)
```python
class TextReader(ABC):
    """Port: Interface for reading text from files."""
    
    @abstractmethod
    def read(self, file_path: Path) -> str:
        """Read text from a file."""
        pass


class PdfTextReader(TextReader):
    """Adapter: Read text from PDF files."""
    
    def read(self, file_path: Path) -> str:
        return read_pdf_text(file_path)  # Existing utility


class PlainTextReader(TextReader):
    """Adapter: Read text from .txt files."""
    
    def read(self, file_path: Path) -> str:
        return file_path.read_text()


class CompositeTextReader(TextReader):
    """Adapter: Route to appropriate reader based on file extension."""
    
    def __init__(self):
        self.readers = {
            '.pdf': PdfTextReader(),
            '.txt': PlainTextReader(),
        }
    
    def read(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        reader = self.readers.get(suffix)
        if not reader:
            raise ValueError(f"No reader for file type: {suffix}")
        return reader.read(file_path)
```

**Benefits:**
- ✅ Easy to test (inject mock readers)
- ✅ Easy to add formats (add reader for .docx, etc.)
- ✅ Follows Open/Closed Principle

---

### 3. LinePreprocessor (Domain Service)
```python
class LinePreprocessor:
    """Domain Service: Preprocess lines to join multi-line items."""
    
    def process(self, lines: List[str]) -> List[str]:
        """
        Join lines that appear to be part of the same item.
        
        Returns:
            List of preprocessed lines.
        """
        if not lines:
            return []
        
        # Remove empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        processed = []
        i = 0
        
        while i < len(lines):
            current = lines[i]
            
            # Check if starts with item pattern (qty + record)
            if re.match(r'^\d+\s+\d+', current):
                # Check if next line is continuation
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if self._is_continuation(next_line):
                        processed.append(f"{current} {next_line}")
                        i += 2
                        continue
                
                processed.append(current)
            else:
                processed.append(current)
            
            i += 1
        
        return processed
    
    def _is_continuation(self, line: str) -> bool:
        """Check if line is a continuation of previous item."""
        # Doesn't start with qty + record
        if re.match(r'^\d+\s+\d+', line):
            return False
        # Contains price info
        if "@" in line or "=" in line:
            return True
        return False
```

**Benefits:**
- ✅ Pure algorithm, no dependencies
- ✅ Easy to test with example inputs
- ✅ Can be replaced with different strategies

---

### 4. PatternMatcher (Domain Service)
```python
@dataclass
class MatchResult:
    """Value object representing a pattern match result."""
    matched: bool
    item: Optional[Dict[str, Any]] = None
    pattern_name: Optional[str] = None
    error: Optional[str] = None


class PatternMatcher:
    """Domain Service: Match lines against schema patterns."""
    
    def __init__(self, schema: ExtractionSchema):
        self.schema = schema
        self._compiled_patterns = self._compile_patterns()
    
    def match(self, line: str) -> MatchResult:
        """
        Try to match line against all patterns in schema.
        
        Returns:
            MatchResult with item if matched, else matched=False.
        """
        for pattern_def in self.schema.patterns:
            compiled = self._compiled_patterns[pattern_def.name]
            match = compiled.search(line)
            
            if match:
                try:
                    item = self._extract_item(match, pattern_def)
                    return MatchResult(
                        matched=True,
                        item=item,
                        pattern_name=pattern_def.name
                    )
                except Exception as e:
                    return MatchResult(
                        matched=False,
                        error=f"Extraction error: {e}"
                    )
        
        return MatchResult(matched=False)
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile all regex patterns once at initialization."""
        return {
            p.name: re.compile(p.regex)
            for p in self.schema.patterns
        }
    
    def _extract_item(
        self,
        match: re.Match,
        pattern: SchemaPattern
    ) -> Dict[str, Any]:
        """Extract item from regex match using pattern field definitions."""
        item = {}
        for field in pattern.fields:
            field_name = field["name"]
            field_type = field["type"]
            
            if "group" in field:
                raw_value = match.group(field["group"])
                item[field_name] = self._convert_value(raw_value, field_type)
            elif "value" in field:
                item[field_name] = field["value"]
        
        return item
    
    def _convert_value(self, raw_value: str, field_type: str) -> Any:
        """Convert string value to appropriate type."""
        converters = {
            "integer": int,
            "float": float,
            "string": str.strip,
        }
        converter = converters.get(field_type, str.strip)
        return converter(raw_value)
```

**Benefits:**
- ✅ Focused on pattern matching only
- ✅ Easy to test with example lines
- ✅ Immutable result objects

---

### 5. ItemPostProcessor (Domain Service)
```python
class ItemPostProcessor:
    """Domain Service: Clean and normalize extracted items."""
    
    def process(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean and normalize all items.
        
        Returns:
            List of processed items.
        """
        return [self._process_item(item) for item in items]
    
    def _process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item."""
        processed = item.copy()
        
        # Clean description
        if "description" in processed:
            desc = processed["description"]
            desc = re.sub(r'\s+', ' ', desc).strip()
            if desc and desc[0].islower():
                desc = desc[0].upper() + desc[1:]
            desc = re.sub(r'[.,;:]+$', '', desc)
            processed["description"] = desc
        
        # Ensure numeric types
        for field, expected_type in [
            ("qty", int),
            ("unit_cost", float),
            ("line_total", float)
        ]:
            if field in processed:
                try:
                    processed[field] = expected_type(processed[field])
                except (ValueError, TypeError):
                    pass  # Keep original if conversion fails
        
        return processed
```

**Benefits:**
- ✅ Pure data transformation
- ✅ Testable with example items
- ✅ Can add new normalizations easily

---

### 6. DirectExtractionOrchestrator (Application Service)
```python
class DirectExtractionOrchestrator:
    """
    Application Service: Orchestrates the extraction process.
    
    This class has minimal logic - it just wires together domain services
    and adapters to perform the extraction workflow.
    """
    
    def __init__(
        self,
        schema_repo: SchemaRepository,
        text_reader: TextReader,
        preprocessor: LinePreprocessor,
        matcher: PatternMatcher,
        postprocessor: ItemPostProcessor,
    ):
        self.schema_repo = schema_repo
        self.text_reader = text_reader
        self.preprocessor = preprocessor
        self.matcher = matcher
        self.postprocessor = postprocessor
    
    def extract(
        self,
        file_path: Path,
        context: ExtractionContext,
    ) -> ExtractionResult:
        """
        Orchestrate the extraction process.
        
        Workflow:
        1. Read text from file
        2. Preprocess lines
        3. Match patterns
        4. Post-process items
        5. Build result
        """
        # Read text
        text = self.text_reader.read(file_path)
        
        # Preprocess
        lines = text.split('\n')
        processed_lines = self.preprocessor.process(lines)
        
        # Match patterns
        items = []
        unparsed = []
        errors = []
        
        for line in processed_lines:
            result = self.matcher.match(line)
            if result.matched:
                items.append(result.item)
            elif result.error:
                errors.append(result.error)
            else:
                unparsed.append(line)
        
        # Post-process items
        items = self.postprocessor.process(items)
        
        # Build result
        return ExtractionResult(
            items=items,
            metadata={
                "source": file_path.name,
                "extraction_method": "direct",
            },
            artifacts={
                "raw_text": text,
                "unparsed_lines": unparsed,
            },
            diagnostics=Diagnostics(errors=errors),
        )
```

**Benefits:**
- ✅ < 50 lines of code
- ✅ All dependencies injected
- ✅ Easy to test orchestration logic
- ✅ Easy to modify workflow

---

### 7. Factory / Builder (Convenience)
```python
class DirectExtractorFactory:
    """Factory for creating DirectExtractor with default dependencies."""
    
    @staticmethod
    def create(
        schema_name: str = "default",
        search_paths: Optional[List[Path]] = None,
    ) -> DirectExtractionOrchestrator:
        """
        Create a DirectExtractor with default configuration.
        
        This factory provides the convenience of the old API while
        using the new architecture internally.
        """
        # Default search paths
        if search_paths is None:
            search_paths = [
                Path("schemas"),
                Path(__file__).parent.parent / "schemas",
            ]
        
        # Build dependencies
        schema_repo = FileSystemSchemaRepository(search_paths)
        schema = schema_repo.load(schema_name)
        
        text_reader = CompositeTextReader()
        preprocessor = LinePreprocessor()
        matcher = PatternMatcher(schema)
        postprocessor = ItemPostProcessor()
        
        # Return orchestrator
        return DirectExtractionOrchestrator(
            schema_repo=schema_repo,
            text_reader=text_reader,
            preprocessor=preprocessor,
            matcher=matcher,
            postprocessor=postprocessor,
        )


# Backward compatible API
class DirectExtractor(Extractor):
    """Wrapper that maintains backward compatibility."""
    
    def __init__(self, schema_name: str = "default", **kwargs):
        super().__init__(validator=kwargs.get("validator", "default"))
        self._orchestrator = DirectExtractorFactory.create(schema_name)
    
    def extract(self, pdf_path: Path) -> ExtractionResult:
        context = ExtractionContext(source_path=pdf_path)
        return self._orchestrator.extract(pdf_path, context)
```

---

## Testing Strategy

### Unit Tests (Fast, Isolated)
```python
# Test schema loading
def test_schema_repository_loads_valid_schema():
    repo = FileSystemSchemaRepository([Path("test/schemas")])
    schema = repo.load("test_schema")
    assert schema.name == "test_schema"
    assert len(schema.patterns) == 2

# Test line preprocessing
def test_preprocessor_joins_continuation_lines():
    preprocessor = LinePreprocessor()
    lines = [
        "1 123 Widget",
        "@ 5.00 /Ea. = 5.00"
    ]
    result = preprocessor.process(lines)
    assert result == ["1 123 Widget @ 5.00 /Ea. = 5.00"]

# Test pattern matching
def test_matcher_extracts_fields():
    schema = ExtractionSchema(
        name="test",
        patterns=[
            SchemaPattern(
                name="standard",
                regex=r"(\d+)\s+(\d+)\s+(.*)\s+@\s+([\d.]+)",
                fields=[
                    {"name": "qty", "type": "integer", "group": 1},
                    {"name": "record", "type": "string", "group": 2},
                ]
            )
        ]
    )
    matcher = PatternMatcher(schema)
    result = matcher.match("1 123 Widget @ 5.00")
    assert result.matched
    assert result.item["qty"] == 1
    assert result.item["record"] == "123"

# Test post-processing
def test_postprocessor_cleans_description():
    processor = ItemPostProcessor()
    items = [{"description": "  widget  with  spaces  "}]
    result = processor.process(items)
    assert result[0]["description"] == "Widget with spaces"
```

### Integration Tests (Slower, End-to-End)
```python
def test_full_extraction_pipeline():
    extractor = DirectExtractorFactory.create("default")
    result = extractor.extract(
        Path("test/fixtures/sample.pdf"),
        ExtractionContext()
    )
    assert len(result.items) > 0
    assert result.items[0]["qty"] > 0
```

---

## Migration Path

### Phase 1: Extract Helpers (Low Risk)
1. Create `LinePreprocessor` class
2. Move `_preprocess_lines` logic
3. Test both old and new implementations side by side
4. Switch DirectExtractor to use new class

### Phase 2: Extract Domain Services (Medium Risk)
1. Create `PatternMatcher` class
2. Move pattern matching logic
3. Test thoroughly
4. Switch DirectExtractor to use new class

### Phase 3: Extract Adapters (Medium Risk)
1. Create `SchemaRepository` interface and implementation
2. Create `TextReader` interface and implementations
3. Test with existing DirectExtractor
4. Switch to use new adapters

### Phase 4: Create Orchestrator (High Value)
1. Create `DirectExtractionOrchestrator`
2. Wire up all extracted classes
3. Test full workflow
4. Switch DirectExtractor to delegate to orchestrator

### Phase 5: Update Tests (Continuous)
1. Add unit tests for each new class
2. Reduce reliance on integration tests
3. Achieve >90% coverage with fast unit tests

---

## Summary

### Before (300 lines, 1 class, 8 responsibilities)
```
DirectExtractor
├── Schema loading
├── Text reading  
├── Line preprocessing
├── Pattern matching
├── Field extraction
├── Post-processing
├── Validation
└── Result building
```

### After (50 lines per class, 7 focused classes)
```
DirectExtractionOrchestrator (50 lines)
├── SchemaRepository (30 lines interface + 40 lines impl)
├── TextReader (20 lines interface + 30 lines impl)  
├── LinePreprocessor (60 lines)
├── PatternMatcher (80 lines)
├── ItemPostProcessor (50 lines)
└── DirectExtractorFactory (40 lines)
```

### Benefits Summary
- ✅ **Testability**: Unit tests with no I/O
- ✅ **Maintainability**: One class per responsibility
- ✅ **Extensibility**: Add new implementations without changing core
- ✅ **Clarity**: Each class has obvious purpose
- ✅ **Change Impact**: Modify one class for one change

### Metrics Improvement
| Metric | Before | After |
|--------|--------|-------|
| Lines per class | 300+ | 30-80 |
| Responsibilities | 8 | 1 per class |
| Test setup | Complex | Simple |
| Coupling | High | Low |
| Test speed | Slow | Fast |
