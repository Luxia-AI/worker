#!/usr/bin/env python3
"""
Test script for Kafka integration.

Usage:
    python test_kafka_integration_script.py [options]

Options:
    --unit              Run only unit tests
    --integration       Run only integration tests (requires Kafka)
    --all               Run all tests (default)
    --verbose, -v       Verbose output
    --coverage          Generate coverage report
    --kafka-bootstrap   Kafka bootstrap servers (default: localhost:9092)

Examples:
    python test_kafka_integration_script.py --unit
    python test_kafka_integration_script.py --all --coverage
    python test_kafka_integration_script.py --integration --verbose
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent


def run_command(cmd: list, verbose: bool = False) -> int:
    """Run a shell command and return exit code."""
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'='*80}\n")

    result = subprocess.run(cmd, cwd=str(get_project_root()))
    return result.returncode


def run_unit_tests(verbose: bool = False, coverage: bool = False) -> int:
    """Run unit tests for Kafka integration."""
    print("\n" + "=" * 80)
    print("RUNNING UNIT TESTS")
    print("=" * 80 + "\n")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_kafka_integration.py",
        "-v" if verbose else "-q",
        "-m",
        "not integration",
    ]

    if coverage:
        cmd.extend(["--cov=app/kafka", "--cov-report=term-missing"])

    return run_command(cmd, verbose)


def run_integration_tests(
    verbose: bool = False,
    kafka_bootstrap: str = "localhost:9092",
) -> int:
    """Run integration tests with real Kafka."""
    print("\n" + "=" * 80)
    print("RUNNING INTEGRATION TESTS")
    print(f"Kafka Bootstrap: {kafka_bootstrap}")
    print("=" * 80 + "\n")

    env = os.environ.copy()
    env["KAFKA_BOOTSTRAP"] = kafka_bootstrap

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_kafka_integration.py",
        "-v" if verbose else "-q",
        "-m",
        "integration",
        "--run-integration",
    ]

    result = subprocess.run(cmd, cwd=str(get_project_root()), env=env)
    return result.returncode


def run_all_tests(
    verbose: bool = False,
    coverage: bool = False,
    kafka_bootstrap: str = "localhost:9092",
) -> int:
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RUNNING ALL KAFKA INTEGRATION TESTS")
    print(f"Kafka Bootstrap: {kafka_bootstrap}")
    print("=" * 80 + "\n")

    env = os.environ.copy()
    env["KAFKA_BOOTSTRAP"] = kafka_bootstrap

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_kafka_integration.py",
        "-v" if verbose else "-q",
    ]

    if coverage:
        cmd.extend(["--cov=app/kafka", "--cov-report=term-missing"])

    result = subprocess.run(cmd, cwd=str(get_project_root()), env=env)
    return result.returncode


def check_dependencies() -> bool:
    """Check if required test dependencies are installed."""
    print("\nChecking test dependencies...")

    required = ["pytest", "pytest-asyncio", "aiokafka"]
    missing = []

    for pkg in required:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", pkg],
            capture_output=True,
        )
        if result.returncode != 0:
            missing.append(pkg)

    if missing:
        print(f"\n[FAIL] Missing dependencies: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    print("[OK] All dependencies installed")
    return True


def check_kafka_available() -> bool:
    """Check if Kafka is available."""
    print("\nChecking Kafka availability...")

    try:
        import socket

        sock = socket.create_connection(("localhost", 9092), timeout=2)
        sock.close()
        print("[OK] Kafka broker is available at localhost:9092")
        return True
    except (socket.timeout, socket.error):
        print("[WARN] Kafka broker not available at localhost:9092")
        print("  (Integration tests will be skipped)")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Kafka integration test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests (requires Kafka)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests (default)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report",
    )
    parser.add_argument(
        "--kafka-bootstrap",
        default="localhost:9092",
        help="Kafka bootstrap servers (default: localhost:9092)",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies and exit",
    )
    parser.add_argument(
        "--check-kafka",
        action="store_true",
        help="Check Kafka availability and exit",
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        if args.check_deps:
            return 1
        print("⚠ Proceeding anyway (some tests may fail)\n")
    elif args.check_deps:
        return 0

    # Check Kafka
    kafka_available = check_kafka_available()
    if args.check_kafka:
        return 0 if kafka_available else 1

    # Determine which tests to run
    run_unit = args.unit or args.all or not (args.integration or args.unit)
    run_integration = args.integration or args.all

    exit_code = 0

    try:
        if run_unit:
            code = run_unit_tests(args.verbose, args.coverage)
            exit_code = max(exit_code, code)

        if run_integration:
            if kafka_available:
                code = run_integration_tests(args.verbose, args.kafka_bootstrap)
                exit_code = max(exit_code, code)
            else:
                print("\n[WARN] Skipping integration tests (Kafka not available)")
                if args.integration:
                    exit_code = 1

        # Print summary
        print("\n" + "=" * 80)
        if exit_code == 0:
            print("[SUCCESS] All tests passed!")
        else:
            print("[FAILED] Some tests failed")
        print("=" * 80 + "\n")

        return exit_code

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n[ERROR] Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
