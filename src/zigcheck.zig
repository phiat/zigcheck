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

// -- Fuzz integration -----------------------------------------------------

pub const fromFuzzInput = @import("fuzz.zig").fromFuzzInput;
pub const FuzzCheckResult = runner.FuzzCheckResult;
pub const checkFuzzOne = runner.checkFuzzOne;
pub const forAllFuzzOne = runner.forAllFuzzOne;

// -- Auto-derivation ------------------------------------------------------

pub const auto = @import("auto.zig").auto;

// -- Combinators (from combinators.zig) -----------------------------------

const combinators = @import("combinators.zig");
pub const constant = combinators.constant;
pub const element = combinators.element;
pub const oneOf = combinators.oneOf;
pub const frequency = combinators.frequency;
pub const map = combinators.map;
pub const mapAlloc = combinators.mapAlloc;
pub const filter = combinators.filter;
pub const FilterExhausted = combinators.FilterExhausted;
pub const noShrink = combinators.noShrink;
pub const shrinkMap = combinators.shrinkMap;
pub const shrinkMapAlloc = combinators.shrinkMapAlloc;
pub const flatMap = combinators.flatMap;
pub const sized = combinators.sized;
pub const resize = combinators.resize;
pub const scale = combinators.scale;
pub const mapSize = combinators.mapSize;
pub const suchThatMap = combinators.suchThatMap;
pub const FunWith = combinators.FunWith;
pub const funGen = combinators.funGen;
pub const build = combinators.build;
pub const zip = combinators.zip;
pub const ZipResult = combinators.ZipResult;
pub const GenType = combinators.GenType;
pub const zipMap = combinators.zipMap;
pub const sliceOf = combinators.sliceOf;
pub const sliceOfRange = combinators.sliceOfRange;
pub const arrayOf = combinators.arrayOf;

// -- Collections and strings (from collections.zig) -----------------------

const collections = @import("collections.zig");
pub const slice = collections.slice;
pub const sliceRange = collections.sliceRange;
pub const asciiChar = collections.asciiChar;
pub const asciiString = collections.asciiString;
pub const asciiStringRange = collections.asciiStringRange;
pub const alphanumeric = collections.alphanumeric;
pub const alphanumericString = collections.alphanumericString;
pub const string = collections.string;
pub const unicodeChar = collections.unicodeChar;
pub const unicodeString = collections.unicodeString;
pub const shuffle = collections.shuffle;
pub const sublistOf = collections.sublistOf;
pub const orderedList = collections.orderedList;
pub const growingElements = collections.growingElements;
pub const sample = collections.sample;
pub const sampleWith = collections.sampleWith;

// -- Constrained integer generators (from generators.zig) -----------------

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
    _ = @import("combinators.zig");
    _ = @import("collections.zig");
    _ = @import("auto.zig");
    _ = @import("shrink.zig");
    _ = @import("runner.zig");
    _ = @import("stateful.zig");
    _ = @import("fuzz.zig");
}
