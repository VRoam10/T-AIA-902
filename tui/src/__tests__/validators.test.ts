import { describe, expect, test } from "bun:test";

import {
  bool,
  num,
  vJson,
  vNonNegNumber,
  vNumber,
  vPosInt,
  vSeeds,
  wrap,
} from "../validators.ts";

describe("wrap", () => {
  test("wraps into [0, n) for negative and overflowing indices", () => {
    expect(wrap(0, 3)).toBe(0);
    expect(wrap(2, 3)).toBe(2);
    expect(wrap(3, 3)).toBe(0);
    expect(wrap(-1, 3)).toBe(2);
    expect(wrap(-4, 3)).toBe(2);
    expect(wrap(7, 3)).toBe(1);
  });
});

describe("num", () => {
  test("coerces finite numbers, else falls back", () => {
    expect(num("42", 0)).toBe(42);
    expect(num("3.5", 0)).toBe(3.5);
    expect(num("", 9)).toBe(0); // Number("") === 0 (finite)
    expect(num("abc", 9)).toBe(9);
    expect(num(undefined, 9)).toBe(9);
    expect(num(NaN, 9)).toBe(9);
  });
});

describe("bool", () => {
  test("treats true/'true'/'yes'/'y' as true, everything else false", () => {
    expect(bool(true)).toBe(true);
    expect(bool("true")).toBe(true);
    expect(bool("yes")).toBe(true);
    expect(bool("y")).toBe(true);
    expect(bool(false)).toBe(false);
    expect(bool("false")).toBe(false);
    expect(bool("1")).toBe(false);
    expect(bool(undefined)).toBe(false);
  });
});

describe("vNumber", () => {
  test("accepts numeric strings, rejects blank and non-numeric", () => {
    expect(vNumber("0")).toBeNull();
    expect(vNumber("-3.2")).toBeNull();
    expect(vNumber("")).not.toBeNull();
    expect(vNumber("   ")).not.toBeNull();
    expect(vNumber("abc")).not.toBeNull();
  });
});

describe("vPosInt", () => {
  test("accepts integers ≥ 1, rejects 0, negatives, and decimals", () => {
    expect(vPosInt("1")).toBeNull();
    expect(vPosInt("500")).toBeNull();
    expect(vPosInt("0")).not.toBeNull();
    expect(vPosInt("-1")).not.toBeNull();
    expect(vPosInt("1.5")).not.toBeNull();
    expect(vPosInt("abc")).not.toBeNull();
  });
});

describe("vNonNegNumber", () => {
  test("accepts numbers ≥ 0, rejects negatives and non-numeric", () => {
    expect(vNonNegNumber("0")).toBeNull();
    expect(vNonNegNumber("0.0")).toBeNull();
    expect(vNonNegNumber("12.5")).toBeNull();
    expect(vNonNegNumber("-0.1")).not.toBeNull();
    expect(vNonNegNumber("x")).not.toBeNull();
  });
});

describe("vSeeds", () => {
  test("accepts a comma list of numbers, rejects empty and non-numeric", () => {
    expect(vSeeds("0,1,2,3,4")).toBeNull();
    expect(vSeeds("0")).toBeNull();
    expect(vSeeds(" 1 , 2 ")).toBeNull(); // trims
    expect(vSeeds("")).not.toBeNull();
    expect(vSeeds(",,")).not.toBeNull();
    expect(vSeeds("0,a,2")).not.toBeNull();
  });
});

describe("vJson", () => {
  test("accepts valid JSON, rejects malformed", () => {
    expect(vJson("{}")).toBeNull();
    expect(vJson('{"lr":[0.1,0.5]}')).toBeNull();
    expect(vJson("[1,2,3]")).toBeNull();
    expect(vJson("{lr:")).not.toBeNull();
    expect(vJson("not json")).not.toBeNull();
  });
});
