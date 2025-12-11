# DirectExtractor - Architectural Assessment

## Overview
`DirectExtractor` is a 300+ line class that extracts structured data from PDFs using regex patterns defined in schemas. While functionally correct, it violates multiple design principles and will become increasingly difficult to maintain as requirements evolve.

## Current Responsibilities (Too Many!)

### 1. Schema Management
```python
def _load_schema(self, schema_name: str) -> Dict[str, Any]:
    schema_path = Path(f"schemas/{schema_name}.json")
    if not schema_path.exists():
        # Try package directory
        import skydocs
        package_dir = Path(skydocs.__file__).parent
        schema_path = package_dir / "schemas" / f"{schema_name}.json"
    # ... JSON loading
```
**Problem:** Mixes business logic with file I/O and path resolution

### 2. Text Extraction
```python
if pdf_path.suffix.lower() == '.txt':
    with open(pdf_path, 'r') as f:
        text = f.read()
else:
    text = read_pdf_text(pdf_path)
```
**Problem:** Direct file system access, hard to test

### 3. Line Preprocessing
```python
def _preprocess_lines(self, lines: List[str]) -> List[str]:
    # 40+ lines of logic for joining multi-line items
    # Complex heuristics for detecting continuations
```
**Problem:** Complex algorithm buried in extractor class

### 4. Pattern Matching & Field Extraction
```python
for pattern_def in self.schema["patterns"]:
    pattern = pattern_def["regex"]
    match = re.search(pattern, line)
    if match:
        for field in pattern_def["fields"]:
            # Extract and convert fields
```
**Problem:** Parsing logic tightly coupled to orchestration

### 5. Item Post-Processing
```python
def _post_process_items(self, items: List[Dict[str, Any]]) -> None:
    for item in items:
        # Clean descriptions
        # Normalize numeric fields
        # Capitalize text
```
**Problem:** Data transformation mixed with extraction

### 6. Validator Resolution & Invocation
```python
def _resolve_validator(self, validator):
    if validator == "default":
        return self._get_default_validator()
    # Complex validator resolution logic
```
**Problem:** Extractor shouldn't know about validation concerns

### 7. Error Handling & Collection
```python
errors = []
unparsed = []
# Scattered throughout method
errors.append({"type": "conversion_error", ...})
```
**Problem:** Error handling interleaved with happy path

### 8. Result Building
```python
result = ExtractionResult(...)
result.legacy_payload = {...}  # Backward compatibility
```
**Problem:** Dual format support complicates every method

## Violation Analysis

### Single Responsibility Principle (SRP)
**Violated:** Class has 8+ distinct responsibilities
- Schema loading
- Text reading
- Line preprocessing
- Pattern matching
- Field extraction
- Post-processing
- Validation
- Result building

**Impact:** Any change to one responsibility risks breaking others

### Open/Closed Principle (OCP)
**Violated:** Adding new extraction strategies requires modifying the class
- Want to support different preprocessing strategies? Modify `_preprocess_lines`
- Want different post-processing? Modify `_post_process_items`
- Want new schema format? Modify `_load_schema`

**Impact:** Can't extend without modification, high regression risk

### Dependency Inversion Principle (DIP)
**Violated:** Depends on concrete implementations
```python
text = read_pdf_text(pdf_path)  # Direct function call
schema = json.load(f)            # Direct JSON dependency
```

**Impact:** Can't swap implementations, hard to test

### Interface Segregation Principle (ISP)
**Violated:** Clients forced to depend on methods they don't use
- Tests that only need pattern matching must instantiate entire extractor
- Must provide schema even when testing line preprocessing

**Impact:** Tests are complex and slow

## Testing Challenges

### Current Test Requirements
```python
def test_extract():
    # Need: Real schema file
    schema_name = "default"
    # Need: Real PDF or mock filesystem
    pdf_path = Path("test.pdf")
    # Need: Validator setup
    validator = ...
    
    extractor = DirectExtractor(schema_name=schema_name, validator=validator)
    result = extractor.extract(pdf_path)
    
    # Test couples all responsibilities together
```

### What We Want
```python
def test_line_preprocessor():
    preprocessor = LinePreprocessor()
    lines = ["1 123 Item", "  @ 5.00 /Ea. = 5.00"]
    result = preprocessor.process(lines)
    assert result == ["1 123 Item @ 5.00 /Ea. = 5.00"]

def test_pattern_matcher():
    matcher = PatternMatcher(schema)
    line = "1 123 Widget @ 5.00 /Ea. = 5.00"
    item = matcher.match(line)
    assert item["qty"] == 1
    
# Fast, focused, no I/O, no setup complexity
```

## Coupling Metrics

### High Coupling Indicators
- **Import Count**: 12 imports (many for infrastructure concerns)
- **Method Count**: 9 methods (many doing unrelated things)
- **Lines of Code**: 300+ lines
- **Cyclomatic Complexity**: High (nested loops, conditionals)
- **Afferent Coupling**: Multiple classes depend on DirectExtractor
- **Efferent Coupling**: DirectExtractor depends on many modules

### Ideal Coupling
- **Import Count**: 2-3 (only domain concepts)
- **Method Count**: 2-3 (focused on core responsibility)
- **Lines of Code**: 50-100 lines
- **Cyclomatic Complexity**: Low (delegated to specialized classes)

## Change Scenarios (Current Pain)

### Scenario 1: "Add support for YAML schemas"
**Current Impact**: Modify `_load_schema`, risk breaking JSON loading
**Effort**: High - need to test all extraction paths

### Scenario 2: "Change line preprocessing algorithm"
**Current Impact**: Modify `_preprocess_lines`, risk breaking extraction
**Effort**: High - need to test with real PDFs and schemas

### Scenario 3: "Add new field type (date)"
**Current Impact**: Modify field extraction logic in `extract`
**Effort**: Medium - scattered across method

### Scenario 4: "Test extraction without validation"
**Current Impact**: Must set `use_validator=False` but validator still created
**Effort**: Medium - can't truly isolate concerns

## Summary: Elevation Targets

### Primary Goals
1. ✅ Extract SchemaRepository (separate concern)
2. ✅ Extract LinePreprocessor (reusable algorithm)
3. ✅ Extract PatternMatcher (core business logic)
4. ✅ Extract FieldExtractor (type conversion)
5. ✅ Extract ItemPostProcessor (data cleaning)
6. ✅ Inject dependencies (TextReader interface)
7. ✅ Remove validator coupling (separate concern)

### Success Metrics
- **Lines per class**: < 100
- **Test complexity**: Unit tests with no I/O
- **Change impact**: Modify one class for one change
- **Cyclomatic complexity**: < 10 per method

### Pattern Selection
- **Repository Pattern**: Schema loading
- **Strategy Pattern**: Preprocessing algorithms
- **Adapter Pattern**: Text reading
- **Dependency Injection**: All infrastructure concerns
- **Service Layer**: Orchestration without business logic

## Next Steps
See `02-target-architecture.md` for the proposed design.
