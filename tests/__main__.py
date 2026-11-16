# Run the existing assertion-based checks: python -m tests.
import runpy


def main():
    for module in ("tests.test_model", "tests.test_accumulation", "tests.test_scheduler"):
        print(f"Running {module}", flush=True)
        runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
