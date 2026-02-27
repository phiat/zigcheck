// zigcheck -- Property-based testing for Zig
//
// Generate random structured inputs, check properties, and automatically
// shrink failing cases to minimal counterexamples.

const std = @import("std");
const testing = std.testing;

// -- Core types -----------------------------------------------------------

pub const Gen = @import("gen.zig").Gen;
pub const ShrinkIter = @import("shrink.zig").ShrinkIter;

// -- Built-in generators --------------------------------------------------

pub const generators = @import("generators.zig");

// -- Runner ---------------------------------------------------------------

const runner = @import("runner.zig");
pub const Config = runner.Config;
pub const CheckResult = runner.CheckResult;
pub const check = runner.check;
pub const forAll = runner.forAll;
pub const forAllWith = runner.forAllWith;
pub const forAllZip = runner.forAllZip;
pub const forAllZipWith = runner.forAllZipWith;
pub const assume = runner.assume;
pub const assertEqual = runner.assertEqual;
pub const expectFailure = runner.expectFailure;
pub const expectFailureWith = runner.expectFailureWith;
pub const TestDiscarded = runner.TestDiscarded;
pub const counterexample = runner.counterexample;
pub const within = runner.within;
pub const forAllLabeled = runner.forAllLabeled;
pub const forAllLabeledWith = runner.forAllLabeledWith;
pub const checkLabeled = runner.checkLabeled;
pub const CheckResultLabeled = runner.CheckResultLabeled;
pub const CoverageResult = runner.CoverageResult;
pub const CoverageRequirement = runner.CoverageRequirement;
pub const forAllCover = runner.forAllCover;
pub const recheck = runner.recheck;
pub const forAllCollect = runner.forAllCollect;
pub const forAllTabulate = runner.forAllTabulate;
pub const conjoin = runner.conjoin;
pub const disjoin = runner.disjoin;
pub const PropertyContext = runner.PropertyContext;
pub const forAllCtx = runner.forAllCtx;
pub const forAllCtxWith = runner.forAllCtxWith;

// -- Convenience re-exports -----------------------------------------------

/// Auto-derive a generator for any supported type via comptime reflection.
pub const auto = generators.auto;

// -- Slice and string generators ------------------------------------------

pub const slice = generators.slice;
pub const sliceRange = generators.sliceRange;
pub const asciiChar = generators.asciiChar;
pub const asciiString = generators.asciiString;
pub const asciiStringRange = generators.asciiStringRange;
pub const alphanumeric = generators.alphanumeric;
pub const alphanumericString = generators.alphanumericString;
pub const string = generators.string;

// -- Combinators ----------------------------------------------------------

pub const constant = generators.constant;
pub const element = generators.element;
pub const oneOf = generators.oneOf;
pub const frequency = generators.frequency;
pub const map = generators.map;
pub const filter = generators.filter;
pub const FilterExhausted = generators.FilterExhausted;
pub const noShrink = generators.noShrink;
pub const shrinkMap = generators.shrinkMap;

// -- Derived generators ---------------------------------------------------

pub const shuffle = generators.shuffle;
pub const sublistOf = generators.sublistOf;
pub const orderedList = generators.orderedList;
pub const growingElements = generators.growingElements;
pub const flatMap = generators.flatMap;
pub const sized = generators.sized;
pub const resize = generators.resize;
pub const scale = generators.scale;
pub const mapSize = generators.mapSize;
pub const suchThatMap = generators.suchThatMap;
pub const FunWith = generators.FunWith;
pub const funGen = generators.funGen;
pub const build = generators.build;
pub const zip = generators.zip;
pub const ZipResult = generators.ZipResult;
pub const GenType = generators.GenType;
pub const zipMap = generators.zipMap;
pub const sliceOf = generators.sliceOf;
pub const sliceOfRange = generators.sliceOfRange;
pub const arrayOf = generators.arrayOf;
pub const unicodeChar = generators.unicodeChar;
pub const unicodeString = generators.unicodeString;

// -- Utility --------------------------------------------------------------

pub const sample = generators.sample;
pub const sampleWith = generators.sampleWith;

// -- Constrained integer generators ---------------------------------------

pub const positive = generators.positive;
pub const nonNegative = generators.nonNegative;
pub const nonZero = generators.nonZero;
pub const negative = generators.negative;

// -- Stateful testing -----------------------------------------------------

pub const StateMachine = @import("stateful.zig").StateMachine;
pub const StatefulConfig = @import("stateful.zig").StatefulConfig;

// -- Tests (pull in all modules for `zig build test`) ---------------------

test {
    _ = @import("gen.zig");
    _ = @import("generators.zig");
    _ = @import("shrink.zig");
    _ = @import("runner.zig");
    _ = @import("stateful.zig");
}
