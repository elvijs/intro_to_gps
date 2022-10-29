REQUIREMENTS="requirements.txt"
SOURCE_DIR="src"
NOTEBOOKS_DIR="notebooks"
TESTS_DIR="tests"


run_jupyter_server:
	jupyter notebook $(NOTEBOOKS_DIR)


format: black isort

black:
	black $(SOURCE_DIR) $(TESTS_DIR)

isort:
	isort $(SOURCE_DIR) $(TESTS_DIR)


check_format: check_black check_isort

check_black:
	black --check $(SOURCE_DIR) $(TESTS_DIR)

check_isort:
	isort --diff $(SOURCE_DIR) $(TESTS_DIR)


static_checks: mypy flake8


mypy:
	mypy --ignore-missing-imports $(SOURCE_DIR) $(TESTS_DIR)

flake8:
	flake8 $(SOURCE_DIR) $(TESTS_DIR)



test: test_code test_notebooks

test_code:
	pytest $(TESTS_DIR)

test_notebooks:
	pytest --nbmake $(NOTEBOOKS_DIR)


freeze_requirements:  # Note the M1 metal workaround in requirements file
	pip freeze | grep -v "tensorflow-metal" | grep -v "intro_to_gps" > $(REQUIREMENTS) && sed -i '' 's/tensorflow-macos/tensorflow/g' $(REQUIREMENTS)


install_deps:
	pip install -r $(REQUIREMENTS)

install: install_deps
	pip install -e .


# For those on Apple metal
m1_setup:
	pip install tensorflow-macos tensorflow-metal
