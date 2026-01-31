"""
Core AST analyzer for detecting temporal leakage patterns.
"""
import ast
from typing import List, Dict, Optional
from pathlib import Path
from .rules import Violation, RULES

class TemporalAnalyzer(ast.NodeVisitor):
    """AST visitor that detects temporal leakage patterns."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.violations: List[Violation] = []
        self.imported_names: Dict[str, str] = {}  # alias -> original_name (e.g., 'tts' -> 'train_test_split')

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track imports to identify relevant functions."""
        if node.module == 'sklearn.model_selection':
            for alias in node.names:
                if alias.name == 'train_test_split':
                    asname = alias.asname or alias.name
                    self.imported_names[asname] = 'train_test_split'
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Inspect function calls for leakage patterns."""
        func_name = self._get_func_name(node.func)
        
        # Detect global statistical operations (T001)
        if func_name in ['mean', 'std', 'var', 'min', 'max', 'normalize']:
            self._check_global_statistics(node, func_name)
            
        # Detect sklearn operations (T003)
        # Check if function name matches a tracked import or is directly called as sklearn.model_selection.train_test_split
        is_tracked_tts = func_name in self.imported_names and self.imported_names[func_name] == 'train_test_split'
        
        # Also catch full qualified calls if possible (simplified: just checking name 'train_test_split' for now if not imported)
        # A more robust check would parse the attribute chain for fully qualified names.
        is_direct_tts = func_name == 'train_test_split' # Heuristic if they did "from sklearn.model_selection import *" or similar
        
        if is_tracked_tts or is_direct_tts:
            self._check_train_test_split(node)

        self.generic_visit(node)
        
    def _get_func_name(self, node: ast.AST) -> str:
        """Extract function name from Call node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ""
        
    def _check_global_statistics(self, node: ast.Call, func_name: str) -> None:
        """Check if statistics are computed globally without temporal awareness."""
        # Check if this is method call on DataFrame/Series (heuristic)
        if isinstance(node.func, ast.Attribute):
            # Look for patterns like: df.mean() or df['col'].mean()
            # If there's no groupby/rolling before stats, it's a potential violation
            
            parent_is_safe = self._has_safe_parent(node)
            
            if not parent_is_safe:
                rule = RULES["T001"]
                self.violations.append(Violation(
                    rule_id=rule.id,
                    message=f"Global {func_name}() may cause temporal leakage. "
                            f"Consider using rolling window or ensure temporal causality.",
                    filename=self.filename,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    severity=rule.severity,
                    context=f"Computing {func_name} without detected temporal boundaries"
                ))
    
    def _check_train_test_split(self, node: ast.Call) -> None:
        """Check if train_test_split respects temporal ordering (T003)."""
        shuffle_arg = None
        
        # Check keywords
        for keyword in node.keywords:
            if keyword.arg == 'shuffle':
                shuffle_arg = keyword.value
                break
        
        # Default for train_test_split is shuffle=True
        is_shuffle_false = False
        if shuffle_arg:
            if isinstance(shuffle_arg, ast.Constant): # python 3.8+
                if shuffle_arg.value is False:
                    is_shuffle_false = True
                elif shuffle_arg.value is True:
                    is_shuffle_false = False
        
        if not is_shuffle_false:
            rule = RULES["T003"]
            self.violations.append(Violation(
                rule_id=rule.id,
                message=rule.description,
                filename=self.filename,
                lineno=node.lineno,
                col_offset=node.col_offset,
                severity=rule.severity,
                context="Random splitting violates temporal causality"
            ))

    def _has_safe_parent(self, node: ast.Call) -> bool:
        """Check if call is preceded by a 'safe' operation like groupby or rolling."""
        if not isinstance(node.func, ast.Attribute):
            return False
            
        # Traverse down the chain to find the preceding function call
        # e.g. df.groupby().mean() -> node.func.value is Call(groupby)
        # e.g. df.groupby()['col'].mean() -> node.func.value is Subscript -> .value is Call(groupby)
        
        current = node.func.value
        while True:
            if isinstance(current, ast.Call):
                parent_func = self._get_func_name(current.func)
                if parent_func in ['groupby', 'rolling', 'expanding', 'resample']:
                    return True
                # If it's some other call, we might stop or continue depending on design, 
                # but for now, if it's not a known safe op, we assume unsafe/chain breaker.
                # However, chained calls like .reset_index() might be in between.
                # For this MVP, let's stop at the immediate call.
                return False
            elif isinstance(current, ast.Subscript):
                current = current.value
            elif isinstance(current, ast.Attribute):
                current = current.value
            else:
                return False

def analyze_file(filepath: Path) -> List[Violation]:
    """Analyze a Python file for temporal leakage violations."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        return analyze_code(source, filename=str(filepath))
    except SyntaxError as e:
        return [Violation(
            rule_id="E001",
            message=f"Syntax error: {e}",
            filename=str(filepath),
            lineno=e.lineno or 0,
            col_offset=e.offset or 0,
            severity="error"
        )]
    except Exception as e:
        # Fallback for other errors reading file
        return [Violation(
            rule_id="E000",
            message=f"Analysis error: {str(e)}",
            filename=str(filepath),
            lineno=0,
            col_offset=0,
            severity="error"
        )]

def analyze_code(source: str, filename: str = "<string>") -> List[Violation]:
    """Analyze a string of Python code for temporal leakage violations."""
    tree = ast.parse(source, filename=filename)
    analyzer = TemporalAnalyzer(filename)
    analyzer.visit(tree)
    return analyzer.violations
