
import pytest
from seismic_linter.analyzer import analyze_code

def test_t001_global_mean_trigger():
    """Test detection of global mean calculation."""
    code = """
import pandas as pd
df = pd.DataFrame({'mag': [1, 2, 3]})
print(df['mag'].mean())
"""
    violations = analyze_code(code)
    assert len(violations) == 1
    assert violations[0].rule_id == "T001"
    assert "Global mean()" in violations[0].message

def test_t001_rolling_mean_ignore():
    """Test that rolling/groupby operations are ignored."""
    code = """
import pandas as pd
df = pd.DataFrame({'mag': [1, 2, 3]})
print(df['mag'].rolling(10).mean())
print(df.groupby('grp')['mag'].mean())
"""
    violations = analyze_code(code)
    assert len(violations) == 0

def test_t003_random_split_trigger():
    """Test detection of random train_test_split (default shuffle=True)."""
    code = """
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, y)
"""
    violations = analyze_code(code)
    assert len(violations) == 1
    assert violations[0].rule_id == "T003"
    assert "shuffle=True" in violations[0].context or "Random splitting" in violations[0].context

def test_t003_explicit_shuffle_true_trigger():
    """Test detection of explicit shuffle=True."""
    code = """
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, y, shuffle=True)
"""
    violations = analyze_code(code)
    assert len(violations) == 1
    assert violations[0].rule_id == "T003"

def test_t003_shuffle_false_pass():
    """Test that shuffle=False passes."""
    code = """
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, y, shuffle=False)
"""
    violations = analyze_code(code)
    assert len(violations) == 0

def test_t003_aliased_import_trigger():
    """Test detection with aliased import."""
    code = """
from sklearn.model_selection import train_test_split as tts
X_train, X_test = tts(X, y, shuffle=True)
"""
    violations = analyze_code(code)
    assert len(violations) == 1
    assert violations[0].rule_id == "T003"
