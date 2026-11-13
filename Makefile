MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
FUNCTIONS ?= data/input/functions_definition.json
TYPE ?= docs
RESULTS ?=

# export UV_PROJECT_ENVIRONMENT ?= $(HOME)/sgoinfre/.venv
# export UV_CACHE_DIR ?= $(HOME)/sgoinfre

export HF_HOME=/home/etchipoq/sgoinfre/.cache/huggingface


.PHONY: install run debug clean lint lint-strict build

# Terminal colors
C_RESET = \033[0m
C_BLUE = \033[1;34m
C_GREEN = \033[1;32m
C_YELLOW = \033[1;33m
C_RED = \033[1;31m

define msg_info
	@printf "$(C_BLUE)[INFO]$(C_RESET) %s\n" "$(1)"
endef

define msg_ok
	@printf "$(C_GREEN)[OK]$(C_RESET) %s\n" "$(1)"
endef

define msg_warn
	@printf "$(C_YELLOW)[WARN]$(C_RESET) %s\n" "$(1)"
endef

define msg_error
	@printf "$(C_RED)[ERROR]$(C_RESET) %s\n" "$(1)"
endef

define msg_step
	@printf "$(C_BLUE)>>$(C_RESET) %s\n" "$(1)"
endef

install:
	$(call msg_info, Installing necessar tools...)
	@uv sync
	$(call msg_ok,All Done.)

# Run your project
run: index search_dataset evaluate
	$(call msg_step,Running application...)
	
index: 
	@uv run python -m src index --max_chunk_size 2000

search_dataset:
ifeq ($(TYPE), docs)
	@uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
else ifeq ($(TYPE), code)
	@uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
else
	$(call msg_error,You must specify the dataset type (eg: TYPE=code or TYPE=docs)...)
endif

answer_dataset:
ifeq ($(RESULTS),)
	$(call msg_error,You must give the search results path (eg: RESULTS=dir/subdir/file)...)
else
	@uv run python -m src answer_dataset --student_search_results_path $(RESULTS)
endif

evaluate:
ifeq ($(TYPE), docs)
	@./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000
else ifeq ($(TYPE), code)
	@./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json --k 10 --max_context_length 2000
else
	$(call msg_error,You must specify the dataset type (eg: TYPE=code or TYPE=docs)...)
endif


# Debug mode
debug:
	$(call msg_warn,Starting debugger for maze application...)
	@python3 -m pdb -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions

# Clean cache files
clean:
	$(call msg_warn,Removing temporary and cache files...)
	@rm -rf .mypy_cache .venv
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	$(call msg_ok,Cleanup completed.)

# Linting (mandatory)
lint:
	$(call msg_step,Running flake8 checks...)
	@uv run python3 -m flake8 src/
	$(call msg_step,Running mypy checks...)
	@uv run python3 -m mypy src/ $(MYPY_FLAGS)
	$(call msg_ok,Lint checks passed.)

# Strict lint (optional)
lint-strict:
	$(call msg_step,Running strict lint checks...)
	@uv run python3 -m flake8  src/
	@uv run python3 -m mypy src/ --strict
	$(call msg_ok,Strict lint checks passed.)

#uv run python -m moulinette grade_student_answers --set private --student_answer_path ../data/output/function_calling_results.json

# ./exams/scripts/exam_retrieval.sh \
# --student-path . \
# --moulinette-path ./moulinette
