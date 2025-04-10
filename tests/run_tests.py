import unittest
import sys
import os

def run_tests():
    """Discover and run all tests in the tests directory."""
    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the parent directory to the Python path so modules can be imported
    parent_dir = os.path.dirname(test_dir)
    sys.path.insert(0, parent_dir)
    
    # Print debug info
    print(f"Python path: {sys.path}")
    print(f"Test directory: {test_dir}")
    print(f"Parent directory: {parent_dir}")
    
    # List available modules in the parent directory
    print("Checking for modules:")
    for item in os.listdir(parent_dir):
        module_path = os.path.join(parent_dir, item)
        if os.path.isdir(module_path) and os.path.exists(os.path.join(module_path, "__init__.py")):
            print(f"  - Found module: {item}")
    
    # List test files for debugging
    print("Looking for test files:")
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                rel_path = os.path.relpath(os.path.join(root, file), test_dir)
                print(f"  - Found test file: {rel_path}")
    
    # Discover all tests in the tests directory
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return non-zero exit code if tests failed
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests()) 