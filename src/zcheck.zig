// zcheck — Property-based testing for Zig
//
// Generate random structured inputs, check properties, and automatically
// shrink failing cases to minimal counterexamples.

const std = @import("std");
const testing = std.testing;

// ── Core types ──────────────────────────────────────────────────────────

pub const Gen = @import("gen.zig").Gen;
pub const ShrinkIter = @import("shrink.zig").ShrinkIter;

// ── Built-in generators ─────────────────────────────────────────────────

pub const generators = @import("generators.zig");

// ── Runner ──────────────────────────────────────────────────────────────

const runner = @import("runner.zig");
pub const Config = runner.Config;
pub const CheckResult = runner.CheckResult;
pub const check = runner.check;
pub const forAll = runner.forAll;

// ── Convenience re-exports ──────────────────────────────────────────────

/// Auto-derive a generator for any supported type via comptime reflection.
pub const auto = generators.auto;

// ── Combinators ──────────────────────────────────────────────────────

// ── Slice and string generators ──────────────────────────────────────

pub const slice = generators.slice;
pub const sliceRange = generators.sliceRange;
pub const asciiChar = generators.asciiChar;
pub const asciiString = generators.asciiString;
pub const asciiStringRange = generators.asciiStringRange;
pub const alphanumeric = generators.alphanumeric;
pub const alphanumericString = generators.alphanumericString;
pub const string = generators.string;

// ── Combinators ──────────────────────────────────────────────────────

pub const constant = generators.constant;
pub const element = generators.element;
pub const oneOf = generators.oneOf;
pub const map = generators.map;
pub const filter = generators.filter;

// ── Tests (pull in all modules for `zig build test`) ────────────────────

test {
    _ = @import("gen.zig");
    _ = @import("generators.zig");
    _ = @import("shrink.zig");
    _ = @import("runner.zig");
}
