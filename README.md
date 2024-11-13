## Project Overview

This project is a Naive Bayes Classifier build we created to improve the organization and quality of a prior built by restructuring it with OOP principles. Now, the project is structure in a modular fashion, with smaller components that are easier to reuse and scale. In addition, we added some unit testing to ensure the Python code quality is at production level and includes best practices.

## Directory structure

```
assignment3/
│
├── nb/
│   ├── bin/
│   │   └── __init__.py
│   │   └── main.py
│   ├── __init__.py
│   └── nb.py
├── utils/
│   ├── __init__.py
│   └── load_data.py
├── tests/
│   ├── __init__.py
│   ├── test_naive_bayes.py
|   |__ data/
|   |   └── docs.txt
├── data/
│   ├── johnson/
│   ├── kennedy/
│   └── unlabeled/
├── pytest.ini
├── .gitignore
├── README.md
├── discussion.md
├── setup.py
├── environment.yml
└── index/
    └── index.py
```

## Installation and How to Run

### Installation

1. Close repository

```
git clone <project_url>
cd nb
```

2. Install required dependencies with the environment.yaml file

```
conda env create -f environment.yaml
conda activate <env_name>
```

3. Install project using setup.py

> Note: I suggest running the following from within the VSCode's terminal 

```
python setup.py install
```

### How to Run

1. Run the program using the following snippet of code

> Assuming you run the program from the nb folder:

```
python nb/bin/main.py -f data
```

2. Run tests using:

```
pytest
```

## Project Components

### nb

- nb.py: Contains the NaiveBayes class, which implements the training and testing of the Naive Bayes classifier.
- main.py: The main entry point for running the classifier.

### utils

- load_data.py: Includes helper functions for loading and processing data, such as extracting text from raw files.

### tests

- test_naive_bayes.py: Contains unit tests for the NaiveBayes class. These tests ensure the correctness of the classifier using pytest.

## Key Features

Some of the projects key features include:

1. Lidstone Smoothing: In the Naive Bayes classifier we implemented Lidstone smoothing to handle zero-probability cases. 

2. Modular Structure: The project is organized in a modular fashion, making it easy to expand, maintain, and test individual components.

3. Testing Framework: We have incorporated unit tests using pytest to ensure the correctness and reliability of the code.

