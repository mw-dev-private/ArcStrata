"""A02 (1): a parameter whose schema can only hold a string keeps its text - in the direct parser and in the
incremental/streaming parser, under every chunking.  Everything else keeps the reading it had before: JSON-decoded
unless the schema says `type: "string"` (the reference below is that old reading, not the new result)."""
import copy
import json
import unittest

from serve.frontend import OutputParser, REF_DEPTH, declared_type, parse_tool_call, string_only

VALUES = ["true", "123", "null", "1.5", "1.10", "1e5", "[1,2]", "[1, 2]", '{"a":1}', '{"a": 1}', "007",
          "hello", "", " x ", '"q"', "a\nb", "-0", "12345678901234567890"]
DEFS = {"S": {"type": "string"}, "N": {"type": "integer"}, "SS": {"anyOf": [{"$ref": "#/$defs/S"}, {"enum": ["a"]}]},
        "A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/S"}, "LOOP": {"$ref": "#/$defs/LOOP"},
        "WITHNULL": {"anyOf": [{"$ref": "#/$defs/S"}, {"type": "null"}]}}

# schema -> True when it can only hold a string
CASES = {
    "type string": ({"type": "string"}, True),
    "type string + enum": ({"type": "string", "enum": ["a", "b"]}, True),
    "type [string]": ({"type": ["string"]}, True),
    "anyOf strings": ({"anyOf": [{"type": "string"}, {"type": "string", "enum": ["a"]}]}, True),
    "oneOf strings": ({"oneOf": [{"type": "string"}, {"enum": ["a", "b"]}]}, True),
    "enum strings, no type": ({"enum": ["true", "false"]}, True),
    "const string, no type": ({"const": "123"}, True),
    "ref to string": ({"$ref": "#/$defs/S"}, True),
    "ref to anyOf of refs": ({"$ref": "#/$defs/SS"}, True),
    "ref chain": ({"$ref": "#/$defs/A"}, True),
    "ref in anyOf": ({"anyOf": [{"$ref": "#/$defs/S"}, {"type": "string"}]}, True),
    "type integer": ({"type": "integer"}, False),
    "type number": ({"type": "number"}, False),
    "type boolean": ({"type": "boolean"}, False),
    "type array": ({"type": "array"}, False),
    "type object": ({"type": "object"}, False),
    "type [string, null]": ({"type": ["string", "null"]}, False),
    "type [string, boolean]": ({"type": ["string", "boolean"]}, False),
    "type [string, integer]": ({"type": ["string", "integer"]}, False),
    "anyOf string|null": ({"anyOf": [{"type": "string"}, {"type": "null"}]}, False),
    "anyOf string|boolean": ({"anyOf": [{"type": "string"}, {"type": "boolean"}]}, False),
    "ref to nullable string": ({"$ref": "#/$defs/WITHNULL"}, False),
    "ref to integer": ({"$ref": "#/$defs/N"}, False),
    "dangling ref": ({"$ref": "#/$defs/NOPE"}, False),
    "external ref": ({"$ref": "https://example.com/s.json"}, False),
    "anchor ref": ({"$ref": "#foo"}, False),
    "cyclic ref": ({"$ref": "#/$defs/LOOP"}, False),
    "enum with a non-string": ({"enum": ["a", 1]}, False),
    "empty enum": ({"enum": []}, False),
    "const non-string": ({"const": 5}, False),
    "empty anyOf": ({"anyOf": []}, False),
    "no type": ({}, False),
    "description only": ({"description": "free text"}, False),
    "undeclared property": (None, False),
}


def old_value(value, prop):
    """The reading before this change."""
    if (prop or {}).get("type") == "string":
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def tool(prop):
    props = {"z": {"type": "string"}}
    if prop is not None:
        props["arg"] = prop
    return {"name": "run", "parameters": {"type": "object", "properties": props, "$defs": copy.deepcopy(DEFS)}}


def call_text(value):
    return ("<tool_call>\n<function=run>\n<parameter=arg>\n" + value + "\n</parameter>\n<parameter=z>\ntail\n"
            "</parameter>\n</function>\n</tool_call>")


def parse(text, tools, stream, width):
    parser = OutputParser(thinking=False, tools=tools, stream_tools=stream)
    events = []
    for i in range(0, len(text), width):
        events += parser.feed(text[i:i + width])
    events += parser.finish("stop")
    return events


def calls(events):
    return [(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"]


class StringOnlyTests(unittest.TestCase):
    def test_classification_table(self):
        for label, (prop, expected) in CASES.items():
            with self.subTest(label):
                root = tool(prop)["parameters"]
                self.assertEqual(string_only(prop, root), expected)
                self.assertEqual(declared_type(prop, root) == "string",
                                 expected or (prop or {}).get("type") == "string")

    def test_declared_type_of_other_schemas_is_their_type_as_before(self):
        root = tool(None)["parameters"]
        for kind in ("integer", "number", "boolean", "array", "object"):
            self.assertEqual(declared_type({"type": kind}, root), kind)
        self.assertEqual(declared_type({"type": ["string", "null"]}, root), ["string", "null"])
        self.assertIsNone(declared_type({}, root))
        self.assertIsNone(declared_type(None, root))

    def test_ref_chain_longer_than_the_limit_is_not_followed(self):
        defs = {f"D{i}": {"$ref": f"#/$defs/D{i + 1}"} for i in range(REF_DEPTH + 3)}
        defs[f"D{REF_DEPTH + 3}"] = {"type": "string"}
        root = {"type": "object", "properties": {}, "$defs": defs}
        self.assertFalse(string_only({"$ref": "#/$defs/D0"}, root))
        self.assertTrue(string_only({"$ref": f"#/$defs/D{REF_DEPTH + 3}"}, root))

    def test_ref_pointer_escapes_and_lists(self):
        root = {"$defs": {"a/b": {"type": "string"}, "t~x": {"type": "string"}, "l": [{"type": "string"}]}}
        self.assertTrue(string_only({"$ref": "#/$defs/a~1b"}, root))
        self.assertTrue(string_only({"$ref": "#/$defs/t~0x"}, root))
        self.assertTrue(string_only({"$ref": "#/$defs/l/0"}, root))
        self.assertFalse(string_only({"$ref": "#/$defs/l/1"}, root))
        self.assertFalse(string_only({"$ref": "#/$defs/l/x"}, root))
        self.assertFalse(string_only({"$ref": "#/$defs/a~1b"}, None))     # no root: nothing to resolve against

    def test_direct_parser_matrix(self):
        for label, (prop, expected_string) in CASES.items():
            t = tool(prop)
            for value in VALUES:
                if value.startswith("\n") or value.endswith("\n"):
                    continue
                with self.subTest(schema=label, value=value):
                    body = call_text(value)[len("<tool_call>\n"):-len("\n</tool_call>")]
                    got = parse_tool_call(body, t).arguments
                    want = value if expected_string else old_value(value, (prop or {}))
                    self.assertEqual(got, {"arg": want, "z": "tail"})
                    if expected_string:
                        self.assertIsInstance(got["arg"], str)

    def test_output_parser_every_mode_matches_the_direct_parser(self):
        for label, (prop, expected_string) in CASES.items():
            t = tool(prop)
            for value in VALUES:
                text = call_text(value)
                want = value if expected_string else old_value(value, (prop or {}))
                for stream in (False, True):
                    for width in (1, 3, 8, len(text)):
                        with self.subTest(schema=label, value=value, stream=stream, width=width):
                            events = parse(text, [t], stream, width)
                            self.assertEqual(calls(events), [("run", {"arg": want, "z": "tail"})])

    def test_streamed_arguments_agree_with_the_final_call(self):
        """The pieces a client assembles from `tool_args` are the same JSON as the final call's arguments."""
        for label, (prop, expected_string) in CASES.items():
            t = tool(prop)
            for value in VALUES:
                text = call_text(value)
                for width in (1, 5, len(text)):
                    with self.subTest(schema=label, value=value, width=width):
                        events = parse(text, [t], True, width)
                        pieces = "".join(e.text for e in events if e.kind == "tool_args")
                        final = calls(events)[0][1]
                        self.assertEqual(json.loads(pieces), final)

    def test_two_string_only_calls_in_one_turn(self):
        t = tool({"enum": ["true", "false"]})
        text = call_text("true") + "\n" + call_text("false")
        for stream in (False, True):
            for width in (1, 7, len(text)):
                with self.subTest(stream=stream, width=width):
                    self.assertEqual(calls(parse(text, [t], stream, width)),
                                     [("run", {"arg": "true", "z": "tail"}), ("run", {"arg": "false", "z": "tail"})])

    def test_schemas_are_not_mutated(self):
        for label, (prop, _) in CASES.items():
            t = tool(prop)
            before = copy.deepcopy(t)
            for stream in (False, True):
                parse(call_text("true"), [t], stream, 4)
            parse_tool_call(call_text("true")[len("<tool_call>\n"):-len("\n</tool_call>")], t)
            self.assertEqual(t, before, label)

    def test_without_a_schema_values_are_json_first_as_before(self):
        body = "<function=run>\n<parameter=arg>\n123\n</parameter>\n</function>"
        self.assertEqual(parse_tool_call(body).arguments, {"arg": 123})
        self.assertEqual(parse_tool_call(body, {"name": "other"}).arguments, {"arg": 123})


if __name__ == "__main__":
    unittest.main()
