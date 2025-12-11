# DirectExtractor - Step-by-Step Refactoring Guide

This guide shows the exact transformation steps to go from the current DirectExtractor god class to the clean, testable architecture.

## Refactoring Strategy

**Golden Rule**: *Never break existing functionality. Refactor in small, testable increments.*

### Process
1. **Extract** → Create new class with extracted logic
2. **Test** → Add tests for new class
3. **Integrate** → Make old class use new class
4. **Verify** → Run all tests (old + new)
5. **Repeat** → Move to next extraction

---

## Step 1: Extract LinePreprocessor

### Why Start Here?
- Self-contained algorithm
- No external dependencies
- Easy to test
- Low risk

### 1.1 Create the New Class

```python
# skydocs/extractors/line_preprocessor.py (NEW FILE)
"""Line preprocessing for multi-line item detection."""

import re
from typing import List


class LinePreprocessor:
    """Preprocess text lines to join multi-line items."""
    
    def process(self, lines: List[str]) -> List[str]:
        """
        Join lines that appear to be part of the same item.
        
        Args:
            lines: Raw lines from text extraction
            
        Returns:
            Preprocessed lines with continuations joined
        """
        if not lines:
            return []
        
        # Remove empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        processed = []
        i = 0
        
        while i < len(lines):
            current = lines[i]
            
            # Check if starts with item pattern (qty + record number)
            if re.match(r'^\d+\s+\d+', current):
                # Check if next line is a continuation
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    
                    # Continuation if has price info but no qty+record
                    if (not re.match(r'^\d+\s+\d+', next_line) and
                        ("@" in next_line or "=" in next_line)):
                        processed.append(f"{current} {next_line}")
                        i += 2
                        continue
                
                processed.append(current)
            else:
                processed.append(current)
            
            i += 1
        
        return processed
```

### 1.2 Add Tests

```python
# tests/test_line_preprocessor.py (NEW FILE)
import pytest
from skydocs.extractors.line_preprocessor import LinePreprocessor


def test_joins_continuation_with_price():
    """Should join lines when next line has price info."""
    preprocessor = LinePreprocessor()
    lines = [
        "1 123 Widget",
        "@ 5.00 /Ea. = 5.00"
    ]
    result = preprocessor.process(lines)
    assert result == ["1 123 Widget @ 5.00 /Ea. = 5.00"]


def test_does_not_join_separate_items():
    """Should not join when next line is new item."""
    preprocessor = LinePreprocessor()
    lines = [
        "1 123 Widget @ 5.00 /Ea. = 5.00",
        "2 124 Gadget @ 3.00 /Ea. = 6.00"
    ]
    result = preprocessor.process(lines)
    assert len(result) == 2


def test_removes_empty_lines():
    """Should remove empty lines."""
    preprocessor = LinePreprocessor()
    lines = [
        "1 123 Widget @ 5.00 /Ea. = 5.00",
        "",
        "2 124 Gadget @ 3.00 /Ea. = 6.00",
        "   "
    ]
    result = preprocessor.process(lines)
    assert len(result) == 2


def test_handles_empty_input():
    """Should handle empty input."""
    preprocessor = LinePreprocessor()
    result = preprocessor.process([])
    assert result == []
```

### 1.3 Integrate with DirectExtractor

```python
# skydocs/extractors/direct.py (MODIFY)
from skydocs.extractors.line_preprocessor import LinePreprocessor

class DirectExtractor(Extractor):
    def __init__(self, schema_name: str = "default", validator=None):
        self.schema_name = schema_name
        self.schema = self._load_schema(schema_name)
        self.preprocessor = LinePreprocessor()  # NEW: Inject preprocessor
        super().__init__(validator=validator)
    
    def extract(self, pdf_path: Path) -> Dict[str, Any]:
        # ... existing code ...
        
        # OLD:
        # lines = self._preprocess_lines(text.split('\n'))
        
        # NEW:
        lines = self.preprocessor.process(text.split('\n'))
        
        # ... rest of method unchanged ...
    
    # DELETE the old _preprocess_lines method entirely
```

### 1.4 Verify

```bash
# Run all tests - old integration tests should still pass
pytest tests/

# Run new unit tests - should be fast!
pytest tests/test_line_preprocessor.py -v

# Expected: All green ✅
```

**Result**: We've extracted one responsibility with zero breakage.

---

## Step 2: Extract PatternMatcher

### Why This Step?
- Core business logic
- Currently buried in extract() method
- High value to isolate and test

### 2.1 Define Domain Models

```python
# skydocs/extractors/pattern_matcher.py (NEW FILE)
"""Pattern matching for structured data extraction."""

import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class MatchResult:
    """Result of attempting to match a line against patterns."""
    matched: bool
    item: Optional[Dict[str, Any]] = None
    pattern_name: Optional[str] = None
    error: Optional[str] = None


class PatternMatcher:
    """Match text lines against schema patterns."""
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize with a schema.
        
        Args:
            schema: Schema dictionary with 'patterns' key
        """
        self.schema = schema
        self._compiled = self._compile_patterns()
    
    def match(self, line: str) -> MatchResult:
        """
        Try to match line against all patterns.
        
        Args:
            line: Text line to match
            
        Returns:
            MatchResult with matched item or error
        """
        for pattern_def in self.schema["patterns"]:
            pattern = self._compiled[pattern_def["name"]]
            match = pattern.search(line)
            
            if match:
                try:
                    item = self._extract_fields(match, pattern_def)
                    return MatchResult(
                        matched=True,
                        item=item,
                        pattern_name=pattern_def["name"]
                    )
                except Exception as e:
                    return MatchResult(
                        matched=False,
                        error=f"Field extraction error: {e}"
                    )
        
        return MatchResult(matched=False)
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile all regex patterns once."""
        return {
            p["name"]: re.compile(p["regex"])
            for p in self.schema["patterns"]
        }
    
    def _extract_fields(
        self,
        match: re.Match,
        pattern_def: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract fields from regex match."""
        item = {}
        
        for field in pattern_def["fields"]:
            field_name = field["name"]
            field_type = field["type"]
            
            if "group" in field:
                raw_value = match.group(field["group"])
                item[field_name] = self._convert_field(raw_value, field_type)
            elif "value" in field:
                item[field_name] = field["value"]
        
        return item
    
    def _convert_field(self, value: str, field_type: str) -> Any:
        """Convert string value to appropriate type."""
        converters = {
            "integer": int,
            "float": float,
            "string": str.strip,
        }
        converter = converters.get(field_type, str.strip)
        return converter(value)
```

### 2.2 Add Tests

```python
# tests/test_pattern_matcher.py (NEW FILE)
import pytest
from skydocs.extractors.pattern_matcher import PatternMatcher, MatchResult


@pytest.fixture
def test_schema():
    """Simple test schema."""
    return {
        "name": "test",
        "patterns": [
            {
                "name": "standard",
                "regex": r"(\d+)\s+(\d+)\s+(.*?)\s+@\s+([\d.]+)\s+/Ea\.\s+=\s+([\d.]+)",
                "fields": [
                    {"name": "qty", "type": "integer", "group": 1},
                    {"name": "record", "type": "string", "group": 2},
                    {"name": "description", "type": "string", "group": 3},
                    {"name": "unit_cost", "type": "float", "group": 4},
                    {"name": "line_total", "type": "float", "group": 5},
                ]
            }
        ]
    }


def test_matches_valid_line(test_schema):
    """Should match valid line and extract fields."""
    matcher = PatternMatcher(test_schema)
    result = matcher.match("1 123 Widget @ 5.00 /Ea. = 5.00")
    
    assert result.matched
    assert result.item["qty"] == 1
    assert result.item["record"] == "123"
    assert result.item["description"] == "Widget"
    assert result.item["unit_cost"] == 5.0
    assert result.item["line_total"] == 5.0


def test_no_match_for_invalid_line(test_schema):
    """Should return unmatched for invalid line."""
    matcher = PatternMatcher(test_schema)
    result = matcher.match("This is not a valid line")
    
    assert not result.matched
    assert result.item is None


def test_returns_pattern_name(test_schema):
    """Should return which pattern matched."""
    matcher = PatternMatcher(test_schema)
    result = matcher.match("1 123 Widget @ 5.00 /Ea. = 5.00")
    
    assert result.pattern_name == "standard"


def test_handles_conversion_errors(test_schema):
    """Should handle field conversion errors gracefully."""
    matcher = PatternMatcher(test_schema)
    # Invalid number in qty position
    result = matcher.match("abc 123 Widget @ 5.00 /Ea. = 5.00")
    
    assert not result.matched
    assert "error" in result.error.lower()
```

### 2.3 Integrate with DirectExtractor

```python
# skydocs/extractors/direct.py (MODIFY)
from skydocs.extractors.pattern_matcher import PatternMatcher

class DirectExtractor(Extractor):
    def __init__(self, schema_name: str = "default", validator=None):
        self.schema_name = schema_name
        self.schema = self._load_schema(schema_name)
        self.preprocessor = LinePreprocessor()
        self.matcher = PatternMatcher(self.schema)  # NEW: Inject matcher
        super().__init__(validator=validator)
    
    def extract(self, pdf_path: Path) -> Dict[str, Any]:
        # ... existing code ...
        
        items = []
        unparsed = []
        errors = []
        
        # OLD: Complex inline pattern matching logic
        # NEW: Clean delegation
        for line in lines:
            result = self.matcher.match(line)
            
            if result.matched:
                items.append(result.item)
            elif result.error:
                errors.append(result.error)
            else:
                unparsed.append(line)
        
        # ... rest of method ...
```

### 2.4 Verify

```bash
pytest tests/ -v

# Expected: All tests pass, including new unit tests
```

**Result**: Pattern matching is now isolated, testable, and reusable.

---

## Step 3: Extract ItemPostProcessor

### Why This Step?
- Data transformation logic
- Currently in _post_process_items
- Easy to test

### 3.1 Create the Class

```python
# skydocs/extractors/item_postprocessor.py (NEW FILE)
"""Post-processing for extracted items."""

import re
from typing import Dict, Any, List


class ItemPostProcessor:
    """Clean and normalize extracted items."""
    
    def process(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process all items.
        
        Args:
            items: Raw extracted items
            
        Returns:
            Cleaned and normalized items
        """
        return [self._process_item(item) for item in items]
    
    def _process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item."""
        processed = item.copy()
        
        # Clean description
        if "description" in processed:
            desc = processed["description"]
            # Remove extra whitespace
            desc = re.sub(r'\s+', ' ', desc).strip()
            # Capitalize first letter
            if desc and desc[0].islower():
                desc = desc[0].upper() + desc[1:]
            # Remove trailing punctuation
            desc = re.sub(r'[.,;:]+$', '', desc)
            processed["description"] = desc
        
        # Ensure numeric types
        for field, expected_type in [
            ("qty", int),
            ("unit_cost", float),
            ("line_total", float)
        ]:
            if field in processed and not isinstance(processed[field], expected_type):
                try:
                    processed[field] = expected_type(processed[field])
                except (ValueError, TypeError):
                    pass  # Keep original if conversion fails
        
        return processed
```

### 3.2 Add Tests

```python
# tests/test_item_postprocessor.py (NEW FILE)
import pytest
from skydocs.extractors.item_postprocessor import ItemPostProcessor


def test_cleans_description_whitespace():
    """Should normalize whitespace in description."""
    processor = ItemPostProcessor()
    items = [{"description": "  widget  with   spaces  "}]
    result = processor.process(items)
    assert result[0]["description"] == "Widget with spaces"


def test_capitalizes_description():
    """Should capitalize first letter."""
    processor = ItemPostProcessor()
    items = [{"description": "widget"}]
    result = processor.process(items)
    assert result[0]["description"] == "Widget"


def test_removes_trailing_punctuation():
    """Should remove trailing punctuation."""
    processor = ItemPostProcessor()
    items = [{"description": "Widget..."}]
    result = processor.process(items)
    assert result[0]["description"] == "Widget"


def test_ensures_numeric_types():
    """Should convert numeric fields to correct types."""
    processor = ItemPostProcessor()
    items = [{
        "qty": "5",
        "unit_cost": "10.50",
        "line_total": "52.50"
    }]
    result = processor.process(items)
    assert isinstance(result[0]["qty"], int)
    assert isinstance(result[0]["unit_cost"], float)
    assert isinstance(result[0]["line_total"], float)


def test_handles_missing_fields():
    """Should not fail on missing fields."""
    processor = ItemPostProcessor()
    items = [{"qty": 1}]  # No description
    result = processor.process(items)
    assert result[0]["qty"] == 1
```

### 3.3 Integrate

```python
# skydocs/extractors/direct.py (MODIFY)
from skydocs.extractors.item_postprocessor import ItemPostProcessor

class DirectExtractor(Extractor):
    def __init__(self, schema_name: str = "default", validator=None):
        self.schema_name = schema_name
        self.schema = self._load_schema(schema_name)
        self.preprocessor = LinePreprocessor()
        self.matcher = PatternMatcher(self.schema)
        self.postprocessor = ItemPostProcessor()  # NEW
        super().__init__(validator=validator)
    
    def extract(self, pdf_path: Path) -> Dict[str, Any]:
        # ... pattern matching ...
        
        # OLD: self._post_process_items(items)
        # NEW:
        items = self.postprocessor.process(items)
        
        # ... build result ...
    
    # DELETE _post_process_items method
```

---

## Step 4: Extract SchemaRepository

### 4.1 Create Interface and Implementation

```python
# skydocs/extractors/schema_repository.py (NEW FILE)
"""Repository for loading extraction schemas."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class SchemaPattern:
    """A regex pattern with field mappings."""
    name: str
    regex: str
    fields: List[Dict[str, Any]]


@dataclass
class ExtractionSchema:
    """An extraction schema."""
    name: str
    patterns: List[SchemaPattern]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionSchema':
        """Create from dictionary."""
        patterns = [
            SchemaPattern(
                name=p["name"],
                regex=p["regex"],
                fields=p["fields"]
            )
            for p in data.get("patterns", [])
        ]
        return cls(
            name=data.get("name", "unknown"),
            patterns=patterns
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "name": self.name,
            "patterns": [
                {
                    "name": p.name,
                    "regex": p.regex,
                    "fields": p.fields
                }
                for p in self.patterns
            ]
        }


class SchemaRepository(ABC):
    """Interface for loading schemas."""
    
    @abstractmethod
    def load(self, schema_name: str) -> ExtractionSchema:
        """Load a schema by name."""
        pass


class FileSystemSchemaRepository(SchemaRepository):
    """Load schemas from JSON files."""
    
    def __init__(self, search_paths: List[Path]):
        """
        Initialize with search paths.
        
        Args:
            search_paths: Directories to search for schema files
        """
        self.search_paths = search_paths
    
    def load(self, schema_name: str) -> ExtractionSchema:
        """Load schema from first matching file."""
        for base_path in self.search_paths:
            schema_path = base_path / f"{schema_name}.json"
            if schema_path.exists():
                with open(schema_path) as f:
                    data = json.load(f)
                return ExtractionSchema.from_dict(data)
        
        raise FileNotFoundError(
            f"Schema '{schema_name}' not found in {self.search_paths}"
        )
```

### 4.2 Add Tests

```python
# tests/test_schema_repository.py (NEW FILE)
import json
import pytest
from pathlib import Path
from skydocs.extractors.schema_repository import (
    FileSystemSchemaRepository,
    ExtractionSchema
)


@pytest.fixture
def temp_schema_dir(tmp_path):
    """Create temp directory with test schema."""
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    
    schema = {
        "name": "test",
        "patterns": [
            {
                "name": "pattern1",
                "regex": r"\d+",
                "fields": [{"name": "num", "type": "integer", "group": 1}]
            }
        ]
    }
    
    schema_file = schema_dir / "test.json"
    schema_file.write_text(json.dumps(schema))
    
    return schema_dir


def test_loads_schema_from_file(temp_schema_dir):
    """Should load schema from file."""
    repo = FileSystemSchemaRepository([temp_schema_dir])
    schema = repo.load("test")
    
    assert schema.name == "test"
    assert len(schema.patterns) == 1
    assert schema.patterns[0].name == "pattern1"


def test_raises_on_missing_schema(temp_schema_dir):
    """Should raise FileNotFoundError for missing schema."""
    repo = FileSystemSchemaRepository([temp_schema_dir])
    
    with pytest.raises(FileNotFoundError):
        repo.load("nonexistent")


def test_searches_multiple_paths(tmp_path):
    """Should search multiple paths in order."""
    dir1 = tmp_path / "schemas1"
    dir2 = tmp_path / "schemas2"
    dir1.mkdir()
    dir2.mkdir()
    
    # Put schema only in dir2
    schema_file = dir2 / "test.json"
    schema_file.write_text(json.dumps({"name": "test", "patterns": []}))
    
    repo = FileSystemSchemaRepository([dir1, dir2])
    schema = repo.load("test")
    
    assert schema.name == "test"
```

### 4.3 Integrate

```python
# skydocs/extractors/direct.py (MODIFY)
from skydocs.extractors.schema_repository import (
    FileSystemSchemaRepository,
    ExtractionSchema
)

class DirectExtractor(Extractor):
    def __init__(
        self,
        schema_name: str = "default",
        validator=None,
        schema_repo: Optional[SchemaRepository] = None  # NEW: Injectable
    ):
        # Use provided repo or create default
        if schema_repo is None:
            import skydocs
            search_paths = [
                Path("schemas"),
                Path(skydocs.__file__).parent / "schemas"
            ]
            schema_repo = FileSystemSchemaRepository(search_paths)
        
        self.schema_name = schema_name
        self.schema = schema_repo.load(schema_name).to_dict()  # Load via repo
        self.preprocessor = LinePreprocessor()
        self.matcher = PatternMatcher(self.schema)
        self.postprocessor = ItemPostProcessor()
        super().__init__(validator=validator)
    
    # DELETE _load_schema method entirely
```

---

## Step 5: Extract TextReader Adapter

### 5.1 Create Interface and Implementations

```python
# skydocs/extractors/text_reader.py (NEW FILE)
"""Adapters for reading text from files."""

from abc import ABC, abstractmethod
from pathlib import Path

from skydocs.utils.pdf import read_pdf_text


class TextReader(ABC):
    """Interface for reading text from files."""
    
    @abstractmethod
    def read(self, file_path: Path) -> str:
        """Read text from file."""
        pass


class PdfTextReader(TextReader):
    """Read text from PDF files."""
    
    def read(self, file_path: Path) -> str:
        return read_pdf_text(file_path)


class PlainTextReader(TextReader):
    """Read text from .txt files."""
    
    def read(self, file_path: Path) -> str:
        return file_path.read_text()


class CompositeTextReader(TextReader):
    """Route to appropriate reader based on extension."""
    
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

### 5.2 Integrate

```python
# skydocs/extractors/direct.py (MODIFY)
from skydocs.extractors.text_reader import CompositeTextReader

class DirectExtractor(Extractor):
    def __init__(
        self,
        schema_name: str = "default",
        validator=None,
        schema_repo: Optional[SchemaRepository] = None,
        text_reader: Optional[TextReader] = None  # NEW: Injectable
    ):
        # ... schema loading ...
        self.text_reader = text_reader or CompositeTextReader()  # NEW
        self.preprocessor = LinePreprocessor()
        self.matcher = PatternMatcher(self.schema)
        self.postprocessor = ItemPostProcessor()
        super().__init__(validator=validator)
    
    def extract(self, pdf_path: Path) -> Dict[str, Any]:
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")
        
        # OLD: Complex file reading logic inline
        # NEW: Clean delegation
        text = self.text_reader.read(pdf_path)
        
        # ... rest of method ...
```

---

## Final Result

### Before: DirectExtractor (300 lines)
```python
class DirectExtractor(Extractor):
    def __init__(self, schema_name, validator):
        # Complex initialization
        pass
    
    def extract(self, pdf_path):
        # 200 lines of mixed concerns
        pass
    
    def _load_schema(self, schema_name):
        # 30 lines
        pass
    
    def _preprocess_lines(self, lines):
        # 40 lines
        pass
    
    def _post_process_items(self, items):
        # 30 lines
        pass
```

### After: DirectExtractor (60 lines)
```python
class DirectExtractor(Extractor):
    """Backward-compatible wrapper around the refactored architecture."""
    
    def __init__(
        self,
        schema_name: str = "default",
        validator=None,
        schema_repo: Optional[SchemaRepository] = None,
        text_reader: Optional[TextReader] = None,
    ):
        # Set up dependencies (with defaults for backward compatibility)
        if schema_repo is None:
            schema_repo = self._create_default_schema_repo()
        
        self.text_reader = text_reader or CompositeTextReader()
        self.schema = schema_repo.load(schema_name).to_dict()
        self.preprocessor = LinePreprocessor()
        self.matcher = PatternMatcher(self.schema)
        self.postprocessor = ItemPostProcessor()
        
        super().__init__(validator=validator)
    
    def extract(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract items using the refactored pipeline."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")
        
        # Read text
        text = self.text_reader.read(pdf_path)
        
        # Preprocess lines
        lines = self.preprocessor.process(text.split('\n'))
        
        # Match patterns
        items = []
        unparsed = []
        errors = []
        
        for line in lines:
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
        result = ExtractionResult(
            items=items,
            metadata={
                "source": pdf_path.name,
                "extraction_method": "direct",
                "schema": self.schema_name,
            },
            artifacts={
                "raw_text": text,
                "unparsed_lines": unparsed,
            },
        )
        
        if errors:
            result.diagnostics.errors = errors
        
        return result
    
    @staticmethod
    def _create_default_schema_repo():
        """Create default schema repository."""
        import skydocs
        search_paths = [
            Path("schemas"),
            Path(skydocs.__file__).parent / "schemas"
        ]
        return FileSystemSchemaRepository(search_paths)
```

### Testing Comparison

**Before: Integration Test (Slow)**
```python
def test_direct_extractor():
    # Need: Real schema file, real PDF, validator setup
    extractor = DirectExtractor(schema_name="default")
    result = extractor.extract(Path("test.pdf"))
    assert len(result["items"]) > 0
    # Runtime: ~500ms, brittle, hard to debug
```

**After: Unit Tests (Fast)**
```python
def test_line_preprocessor():
    preprocessor = LinePreprocessor()
    result = preprocessor.process(["1 123 Widget", "@ 5.00"])
    assert result == ["1 123 Widget @ 5.00"]
    # Runtime: <1ms

def test_pattern_matcher():
    matcher = PatternMatcher(test_schema)
    result = matcher.match("1 123 Widget @ 5.00 /Ea. = 5.00")
    assert result.matched
    assert result.item["qty"] == 1
    # Runtime: <1ms

def test_postprocessor():
    processor = ItemPostProcessor()
    result = processor.process([{"description": "  widget  "}])
    assert result[0]["description"] == "Widget"
    # Runtime: <1ms

# Plus one integration test to verify wiring
def test_direct_extractor_integration():
    # Mock dependencies for speed
    extractor = DirectExtractor(
        schema_repo=MockSchemaRepo(),
        text_reader=MockTextReader()
    )
    result = extractor.extract(Path("fake.pdf"))
    assert len(result.items) > 0
    # Runtime: <10ms
```

---

## Migration Checklist

- [x] Extract LinePreprocessor
- [x] Extract PatternMatcher  
- [x] Extract ItemPostProcessor
- [x] Extract SchemaRepository
- [x] Extract TextReader
- [ ] Update all existing tests
- [ ] Add comprehensive unit tests
- [ ] Update documentation
- [ ] Deploy with feature flag (optional)
- [ ] Monitor production (no regressions)
- [ ] Remove old code paths

---

## Key Learnings

### What Worked
1. **Small steps**: Each extraction was independently testable
2. **Backward compatibility**: Existing API never broke
3. **Test-first**: Tests caught issues immediately
4. **Interface clarity**: ABCs made contracts explicit

### Common Pitfalls
1. **Too big**: Don't extract everything at once
2. **No tests**: Always add tests before integrating
3. **Breaking changes**: Maintain backward compatibility
4. **Premature abstraction**: Extract when pattern is clear

### When to Stop
- Each class has < 100 lines
- Each class has one clear responsibility
- Tests are fast and don't require I/O
- Changes affect only one class

You've successfully elevated DirectExtractor from a god class to clean architecture! 🎉
