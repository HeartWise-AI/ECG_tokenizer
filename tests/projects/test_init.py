import unittest

class TestProjectsInit(unittest.TestCase):
    def test_imports(self):
        """Test that the imports from projects/__init__.py work correctly"""
        try:
            from projects import (
                LLMFinetuningProject,
                ECGTokenizerTrainingProject,
                BertReportClassifierProject
            )
            # If we get here, the imports worked
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Import error: {e}")

if __name__ == "__main__":
    unittest.main() 