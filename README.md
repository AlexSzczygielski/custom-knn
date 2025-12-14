# custom-knn

## A final project for the Programming in Python Language engineering course.

Custom implementation of the **[k-Nearest Neighbors (k-NN) algorithm](docs/KNN_OVERVIEW.md)**, developed in Python and compared against scikit-learn. Project utilises a `github` repository, `PyScaffold` packaging and was developed in `VirtualEnv`.

<div align="center">

[![PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![Python](https://img.shields.io/badge/Python-blue?logo=python&logoColor=yellow)](https://www.python.org)
[![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

</div>

<div align="center">
<img width = 50% alt= "K-NN Visualization Graph" src= "docs/assets/knn_visualization.png">
<p>
    <em>
        Figure 1: K-Nearest Neighbors visualization. Source: 
        <a href="https://www.linkedin.com/pulse/k-nearest-neighbors-knn-algorithm-classification-nrxgf">
        Vrata Tech Solutions Pvt. Ltd. (LinkedIn)</a>
    </em>
</p>
</div>


## Comparison with scikit-learn
This custom k-NN implementation is tested and compared against scikit-learn's `KNeighborsClassifier` to verify correctness of operation. For further information please refer to the [analysis section](docs/ANALYSIS.md).


## Usage
1. Activate Python virtual environment
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run script
```bash
python src/knn.py
```

## Project Structure
```bash
custom-knn % tree -L1
.
├── AUTHORS.rst
├── CHANGELOG.rst
├── CONTRIBUTING.rst
├── docs # Documentation
├── LICENSE.txt
├── pyproject.toml
├── README.md
├── README.rst
├── setup.cfg
├── setup.py
├── src # k-NN implementation
├── tests # Unit tests
└── tox.ini
```


---

## Documentation:
### [K-Nearest Neighbors Algorithm Overview](docs/KNN_OVERVIEW.md)
### [Project requirements](docs/PROJECT_REQUIREMENTS.md)
### [Algorithm Implementation](docs/IMPLEMENTATION.md)
### [Testing](docs/TESTING.md)
### [Analysis (scikit learn)](docs/ANALYSIS.md)

---
<div align="center">
<img width = 20%  src= "https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Znak_graficzny_AGH.svg/2048px-Znak_graficzny_AGH.svg.png">
<img width = 20%  src= "https://iet.agh.edu.pl/wp-content/uploads/2021/05/Logo-WIET-2021.png">
</div>