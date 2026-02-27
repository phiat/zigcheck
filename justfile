# zigcheck — property-based testing for Zig

# Run all tests
test:
    zig build test

# Run tests with verbose output
test-verbose:
    zig build test -- --verbose

# Build the library
build:
    zig build

# Clean build artifacts
clean:
    rm -rf .zig-cache zig-out zig-cache

# Run a specific test by name filter (usage: just test-filter "shrink")
test-filter FILTER:
    zig build test -- --test-filter "{{FILTER}}"

# Show project stats
stats:
    @echo "Files:      $(find src -name '*.zig' | wc -l) source files"
    @echo "Lines:      $(cat src/*.zig | wc -l) lines of code"
    @echo "Tests:      $(grep -r 'test "' src/*.zig | wc -l) tests"
    @echo "Generators: $(grep -c '^pub ' src/generators.zig) public exports"

# Push to all remotes
push:
    git push origin main
    git push github main

# List open issues
issues:
    bd list --status=open

# Format check (zig fmt --check)
fmt-check:
    zig fmt --check src/

# Format all source files
fmt:
    zig fmt src/
