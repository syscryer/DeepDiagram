import unittest

from app.api.routes import StreamingTagParser


class TestStreamingTagParser(unittest.TestCase):
    def test_ignores_partial_closing_tags_in_design_concept(self):
        parser = StreamingTagParser()

        events = parser.feed("<design_concept>hello</design_con")

        self.assertIn(("design_concept_start", "", True), events)
        self.assertIn(("design_concept", "hello", True), events)
        self.assertNotIn(("design_concept", "hello</design_con", True), events)

    def test_ignores_partial_closing_tags_in_code(self):
        parser = StreamingTagParser()
        initial_events = parser.feed("<design_concept>ok</design_concept><code>")
        self.assertIn(("code_start", "", True), initial_events)

        events = parser.feed('{"a":1}</co')

        self.assertIn(("code", '{"a":1}', True), events)
        self.assertNotIn(("code", '{"a":1}</co', True), events)


if __name__ == "__main__":
    unittest.main()
