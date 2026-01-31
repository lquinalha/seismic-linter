"""
Command-line interface for seismic-linter.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Dict
from .analyzer import analyze_file
from .rules import Violation

def main():
    parser = argparse.ArgumentParser(
        description="Detect temporal causality violations in seismic ML code"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="File or directory to analyze"
    )
    
    args = parser.parse_args()
    
    path = args.path
    if not path.exists():
        print(f"Error: Path '{path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    violations_by_file: Dict[str, List[Violation]] = {}
    
    # Analyze single file or directory
    if path.is_file():
        if path.suffix == '.py':
            violations_by_file[str(path)] = analyze_file(path)
    else:
        for filepath in path.rglob("*.py"):
            violations_by_file[str(filepath)] = analyze_file(filepath)
    
    # Print results
    print("\n=== Seismic Linter Scan ===\n")
    total_violations = 0
    
    for filepath, violations in violations_by_file.items():
        if not violations:
            continue
            
        print(f"File: {filepath}")
        for v in violations:
            total_violations += 1
            icon = "⚠️" if v.severity == "warning" else "ℹ️"
            if v.severity == "error":
                icon = "❌"
                
            print(f"  {icon} [Line {v.lineno}] {v.rule_id}: {v.message}")
            if v.context:
                print(f"     Context: {v.context}")
        print("")
        
    if total_violations == 0:
        print("✅ No violations found.")
    else:
        print(f"Found {total_violations} potential violation(s).")
        # Exit with error if any violation is generic for now, can be strict later
        # sys.exit(1)

if __name__ == "__main__":
    main()
