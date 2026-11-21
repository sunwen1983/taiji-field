"""Thin wrapper: python scripts/run_exam.py [--limit N] [--output PATH]"""

from taiji_field.exam import main

if __name__ == "__main__":
    raise SystemExit(main())
