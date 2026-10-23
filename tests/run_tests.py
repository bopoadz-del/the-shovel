"""Test runner and report generator for Cerebrum Blocks."""

import subprocess
import sys
import json
import time
from datetime import datetime
from pathlib import Path

def run_tests():
    """Run all tests and generate report."""
    print("=" * 70)
    print("CEREBRUM BLOCKS - TEST SUITE")
    print("=" * 70)
    print()
    
    start_time = time.time()
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=app",
        "--cov-report=term-missing",
        "--cov-report=html:tests/reports/htmlcov",
        "--json-report",
        "--json-report-file=tests/reports/test_report.json",
        "-x"  # Stop on first failure
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse results
    output = result.stdout + result.stderr
    
    # Generate report
    generate_text_report(output, result.returncode, time.time() - start_time)
    
    if result.returncode == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ TESTS FAILED (exit code: {result.returncode})")
    
    return result.returncode

def generate_text_report(output, exit_code, duration):
    """Generate a text-based test report."""
    report_path = Path("tests/reports/test_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CEREBRUM BLOCKS - TEST REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Duration: {duration:.2f} seconds\n")
        f.write(f"Status: {'PASSED' if exit_code == 0 else 'FAILED'}\n")
        f.write("=" * 80 + "\n\n")
        f.write(output)
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nReport saved to: {report_path}")

def generate_markdown_report():
    """Generate a markdown test report."""
    report_path = Path("tests/reports/TEST_REPORT.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try to load JSON report if it exists
    json_path = Path("tests/reports/test_report.json")
    test_data = None
    if json_path.exists():
        with open(json_path) as f:
            test_data = json.load(f)
    
    with open(report_path, "w") as f:
        f.write("# Cerebrum Blocks - Test Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        if test_data:
            summary = test_data.get("summary", {})
            f.write(f"- **Total Tests:** {summary.get('total', 'N/A')}\n")
            f.write(f"- **Passed:** {summary.get('passed', 'N/A')}\n")
            f.write(f"- **Failed:** {summary.get('failed', 'N/A')}\n")
            f.write(f"- **Skipped:** {summary.get('skipped', 'N/A')}\n")
            f.write(f"- **Duration:** {summary.get('duration', 'N/A'):.2f}s\n\n")
        else:
            f.write("Run tests to generate detailed report.\n\n")
        
        # Block Test Status
        f.write("## Block Test Coverage\n\n")
        
        blocks = [
            ("PDF", "test_pdf.py"),
            ("OCR", "test_ocr.py"),
            ("Chat", "test_chat.py"),
            ("Voice", "test_voice.py"),
            ("Search", "test_search.py"),
            ("Image", "test_image.py"),
            ("Translate", "test_translate.py"),
            ("Code", "test_code.py"),
            ("Web", "test_web.py"),
            ("Google Drive", "test_google_drive.py"),
            ("OneDrive", "test_onedrive.py"),
            ("Local Drive", "test_local_drive.py"),
            ("Android Drive", "test_android_drive.py"),
        ]
        
        f.write("| Block | Test File | Status |\n")
        f.write("|-------|-----------|--------|\n")
        
        for name, test_file in blocks:
            test_path = Path("tests/blocks") / test_file
            status = "✅ Ready" if test_path.exists() else "❌ Missing"
            f.write(f"| {name} | `{test_file}` | {status} |\n")
        
        f.write("\n")
        
        # Test Categories
        f.write("## Test Categories\n\n")
        f.write("### AI Blocks (9 blocks)\n")
        f.write("- PDF extraction and processing\n")
        f.write("- OCR text recognition\n")
        f.write("- AI chat completions\n")
        f.write("- Speech-to-text and text-to-speech\n")
        f.write("- Web search\n")
        f.write("- Image analysis and generation\n")
        f.write("- Text translation\n")
        f.write("- Code execution and analysis\n")
        f.write("- Web scraping\n\n")
        
        f.write("### Drive Blocks (4 blocks)\n")
        f.write("- Google Drive integration\n")
        f.write("- Microsoft OneDrive integration\n")
        f.write("- Local filesystem operations\n")
        f.write("- Android storage access\n\n")
        
        # Integration Tests
        f.write("## Integration Tests\n\n")
        f.write("- API endpoint tests\n")
        f.write("- Block chaining tests\n")
        f.write("- Common pipeline tests\n\n")
        
        # How to Run
        f.write("## How to Run Tests\n\n")
        f.write("```bash\n")
        f.write("# Run all tests\n")
        f.write("pytest tests/ -v\n\n")
        f.write("# Run with coverage\n")
        f.write("pytest tests/ --cov=app --cov-report=html\n\n")
        f.write("# Run specific block tests\n")
        f.write("pytest tests/blocks/test_pdf.py -v\n\n")
        f.write("# Run this report generator\n")
        f.write("python tests/run_tests.py\n")
        f.write("```\n\n")
        
        f.write("---\n\n")
        f.write("*This report was auto-generated by the Cerebrum Blocks test suite.*\n")
    
    print(f"Markdown report saved to: {report_path}")

def main():
    """Main entry point."""
    # Generate initial markdown report
    generate_markdown_report()
    
    # Check if pytest-cov is installed
    try:
        import pytest_cov
    except ImportError:
        print("\n⚠️  pytest-cov not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest-cov", "pytest-json-report"], check=True)
        print("✅ Dependencies installed.\n")
    
    # Run tests
    exit_code = run_tests()
    
    # Regenerate markdown report with data
    generate_markdown_report()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
