import shutil


def check_program(program):
    """Checks if the specified program is installed"""
    path = shutil.which(program)
    if path is None:
        print(f"{program} is not installed.")
    else:
        print(f"{program} is installed at {path}.")


def main():
    # sudo apt install clang-format
    # python -m pip install cpplint
    # TODO cmake...
    programs = ["clang-format", "cpplint", "cmake"]
    for program in programs:
        check_program(program)


if __name__ == "__main__":
    main()
