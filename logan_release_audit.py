#!/usr/bin/env python3
"""
LOGAN Release Audit Tool
Validates terminology, bounded claims, references, and CSV presence before arXiv submission.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict


class AuditReport:
    def __init__(self):
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.errors: List[str] = []

    def add_warning(self, msg: str):
        self.warnings.append(f"WARNING: {msg}")

    def add_info(self, msg: str):
        self.info.append(f"INFO: {msg}")

    def add_error(self, msg: str):
        self.errors.append(f"ERROR: {msg}")

    def has_issues(self) -> bool:
        return len(self.warnings) > 0 or len(self.errors) > 0

    def generate_markdown(self) -> str:
        lines = ["# LOGAN Release Audit Report\n"]

        if not self.has_issues():
            lines.append("## All Checks Passed!\n")
            lines.append("No issues found. Repository is ready for release.\n")
        else:
            if self.errors:
                lines.append("## Errors (Must Fix)\n")
                for error in self.errors:
                    lines.append(f"- {error}\n")
                lines.append("\n")

            if self.warnings:
                lines.append("## Warnings (Review Recommended)\n")
                for warning in self.warnings:
                    lines.append(f"- {warning}\n")
                lines.append("\n")

        if self.info:
            lines.append("## Information\n")
            for info in self.info:
                lines.append(f"- {info}\n")
            lines.append("\n")

        return "".join(lines)


def check_terminology(content: str, filename: str, report: AuditReport):
    """Check that Builder/Devil terminology is used consistently."""

    # Check for correct terminology
    builder_count = len(re.findall(r'\bBuilder\b', content, re.IGNORECASE))
    devil_count = len(re.findall(r'\bDevil\b', content, re.IGNORECASE))

    # Check for incorrect terminology
    synthesizer_count = len(re.findall(r'\bSynthesizer\b', content, re.IGNORECASE))
    verifier_count = len(re.findall(r'\bVerifier\b', content, re.IGNORECASE))

    if synthesizer_count > 0 or verifier_count > 0:
        report.add_error(
            f"{filename}: Found incorrect terminology - "
            f"Synthesizer: {synthesizer_count}, Verifier: {verifier_count}. "
            f"Should use Builder/Devil instead."
        )

    if builder_count > 0 or devil_count > 0:
        report.add_info(
            f"{filename}: Correct terminology found - "
            f"Builder: {builder_count}, Devil: {devil_count}"
        )


def check_bounded_claims(content: str, filename: str, report: AuditReport):
    """Check that claims are properly bounded with ≡_k notation."""

    # Look for bounded equivalence notation
    bounded_patterns = [
        r'\\equiv_k',
        r'\\equiv_\{k\}',
        r'up to depth.*k',
        r'depth.*k',
        r'k-round',
        r'bounded.*depth'
    ]

    bounded_found = any(re.search(pattern, content, re.IGNORECASE)
                       for pattern in bounded_patterns)

    if bounded_found:
        report.add_info(f"{filename}: Found bounded claim notation (≡_k, depth k, etc.)")

    # Look for problematic unbounded claims
    unbounded_patterns = [
        r'provably equivalent(?!\s+at\s+depth)',
        r'fully equivalent(?!\s+at\s+depth)',
        r'completely indistinguishable(?!\s+at\s+depth)',
    ]

    for pattern in unbounded_patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            context_start = max(0, match.start() - 50)
            context_end = min(len(content), match.end() + 50)
            context = content[context_start:context_end].replace('\n', ' ')
            report.add_warning(
                f"{filename}: Potentially unbounded claim found: '{match.group()}' "
                f"Context: ...{context}..."
            )


def check_latex_references(content: str, filename: str, report: AuditReport):
    """Check that LaTeX figure/table references exist."""

    # Find all \label{} definitions
    labels = set(re.findall(r'\\label\{([^}]+)\}', content))

    # Find all \ref{} and \cref{} uses
    refs = re.findall(r'\\(?:ref|cref)\{([^}]+)\}', content)

    # Check for undefined references
    undefined_refs = [ref for ref in refs if ref not in labels]

    if undefined_refs:
        report.add_warning(
            f"{filename}: Found {len(undefined_refs)} potentially undefined references: "
            f"{', '.join(set(undefined_refs[:5]))}"
            + (f" and {len(set(undefined_refs)) - 5} more" if len(set(undefined_refs)) > 5 else "")
        )

    if labels:
        report.add_info(f"{filename}: Found {len(labels)} label definitions")


def check_csv_files(results_dir: Path, report: AuditReport):
    """Check that CSV files exist and are readable."""

    if not results_dir.exists():
        report.add_error(f"Results directory not found: {results_dir}")
        return

    csv_files = list(results_dir.glob("*.csv"))

    if not csv_files:
        report.add_warning(f"No CSV files found in {results_dir}")
        return

    report.add_info(f"Found {len(csv_files)} CSV files in {results_dir}")

    for csv_file in csv_files:
        try:
            with open(csv_file, 'r') as f:
                lines = f.readlines()
                if len(lines) == 0:
                    report.add_warning(f"{csv_file.name}: Empty CSV file")
                elif len(lines) == 1:
                    report.add_warning(f"{csv_file.name}: Only header row, no data")
                else:
                    header = lines[0].strip()
                    sample_row = lines[1].strip() if len(lines) > 1 else ""
                    report.add_info(
                        f"{csv_file.name}: {len(lines)-1} data rows. "
                        f"Header: {header[:60]}..."
                    )
        except Exception as e:
            report.add_error(f"{csv_file.name}: Error reading file - {e}")


def audit_files(paper_path: str, readme_path: str, results_path: str) -> AuditReport:
    """Run all audit checks."""

    report = AuditReport()

    # Check paper
    paper_file = Path(paper_path)
    if paper_file.exists():
        with open(paper_file, 'r', encoding='utf-8') as f:
            paper_content = f.read()

        report.add_info(f"Checking paper: {paper_file}")
        check_terminology(paper_content, paper_file.name, report)
        check_bounded_claims(paper_content, paper_file.name, report)
        check_latex_references(paper_content, paper_file.name, report)
    else:
        report.add_error(f"Paper file not found: {paper_path}")

    # Check README
    readme_file = Path(readme_path)
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            readme_content = f.read()

        report.add_info(f"Checking README: {readme_file}")
        check_terminology(readme_content, readme_file.name, report)
    else:
        report.add_error(f"README file not found: {readme_path}")

    # Check CSV files
    results_dir = Path(results_path)
    check_csv_files(results_dir, report)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Audit LOGAN repository for release readiness"
    )
    parser.add_argument(
        '--paper',
        default='paper/main.tex',
        help='Path to main LaTeX paper file'
    )
    parser.add_argument(
        '--readme',
        default='README.md',
        help='Path to README file'
    )
    parser.add_argument(
        '--results',
        default='results',
        help='Path to results directory with CSV files'
    )
    parser.add_argument(
        '--output',
        default='release_audit_report.md',
        help='Output report filename'
    )

    args = parser.parse_args()

    print("Running LOGAN Release Audit...\n")

    report = audit_files(args.paper, args.readme, args.results)

    # Write report
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report.generate_markdown())

    print(f"Report written to: {output_path}\n")

    # Print summary to console
    if report.errors:
        print(f"ERRORS: {len(report.errors)} error(s) found")
    if report.warnings:
        print(f"WARNINGS: {len(report.warnings)} warning(s) found")
    if report.info:
        print(f"INFO: {len(report.info)} info message(s)")

    if not report.has_issues():
        print("\nAll checks passed! Repository is ready for release.")
        return 0
    else:
        print(f"\nIssues found. Review {output_path} for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
