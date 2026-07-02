## ADDED Requirements

### Requirement: get(component=...) calls SHALL NOT share a test function with always-run assertions
A test function that calls `roundtrip.get(component=...)` or
`roundtrip.get(storageClass=...)` SHALL NOT contain assertions that must always
execute regardless of butler availability.  When `lsst.daf.butler` is absent,
`get(component=...)` calls `pytest.skip()`, which exits the entire test function.
Butler-free assertions that must always run SHALL live in a separate test function
that contains no `get(component=...)` or `get(storageClass=...)` calls.

Both the component-read function and the butler-free function MAY pass
`storage_class=...` to `RoundtripX`; `storage_class` alone does not cause a skip
and its presence ensures full butler-path coverage when the butler is available.

#### Scenario: Always-run assertions are in a component-read-free function
- **WHEN** a test needs to assert on `roundtrip.result`, FITS headers, or other
  properties that do not require named component reads
- **THEN** those assertions SHALL be placed in a test function that contains no
  `roundtrip.get(component=...)` calls, so that `pytest.skip()` is never triggered

#### Scenario: Component reads are isolated in their own function
- **WHEN** a test needs to call `roundtrip.get(component=...)` or
  `roundtrip.get(storageClass=...)`
- **THEN** those calls SHALL be placed in a dedicated test function containing
  only component-read assertions; when butler is absent, the first
  `get(component=...)` call will `pytest.skip()` the entire function cleanly,
  wasting no coverage that should always run

#### Scenario: No always-run assertions are silently skipped by component reads
- **WHEN** `lsst.daf.butler` is not installed and a test function calls
  `roundtrip.get(component=...)`
- **THEN** only the assertions in that specific test function SHALL be skipped;
  assertions in the companion component-read-free function SHALL execute normally
