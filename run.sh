#!/bin/bash
#source .venv/Scripts/activate
#uvicorn app.main:app --host 0.0.0.0 --port 9000 --reload

# Define a function to run a single check and report its status
run_check() {
    local check_name=$1
    shift
    local command="$@"

    echo ""
    echo "--- Running: $check_name ---"
    
    if $command; then
        echo "✅ Passed: $check_name"
        return 0
    else
        echo "❌ Failed: $check_name"
        return 1
    fi
}

# Use 'read' to get user input for which checks to run
read -p "Enter the checks you want to run (e.g., 'pytest ruff black isort flake8 bandit mypy', or 'all' for all checks): " input_checks

# An associative array to map check names to their commands
declare -A checks
checks[pytest]="pytest -q --disable-warnings"
checks[ruff]="ruff check . && ruff --fix --exit-zero ."
checks[black]="black --check app tests"
checks[isort]="isort --check-only app tests"
checks[flake8]="flake8 app tests"
checks[bandit]="bandit -r app"
checks[mypy]="mypy app"

# If the user enters 'all', use all available checks. Otherwise, parse their input.
if [[ "$input_checks" == "all" || "$input_checks" == "" ]]; then
    run_all=true
    checks_to_run=("${!checks[@]}")
else
    run_all=false
    checks_to_run=($input_checks)
fi

# A flag to track overall success
all_passed=true

# Loop through the checks to run and execute them
for check in "${checks_to_run[@]}"; do
    if [[ -v checks[$check] ]]; then
        run_check "$check" "${checks[$check]}"
        if [ $? -ne 0 ]; then
            all_passed=false
        fi
    else
        echo "⚠️  Warning: Unknown check '$check'. Skipping."
    fi
done

# Final summary based on the results
echo ""
echo "--- Summary ---"
if $all_passed; then
    echo "🎉 All selected checks passed successfully!"
    exit 0
else
    echo "🛑 Some checks failed. Please review the output above."
    exit 1
fi