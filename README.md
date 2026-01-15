
<img width="350" height="150" alt="image" src="https://github.com/user-attachments/assets/059b9588-4ae6-4951-9ab8-3f30c989e592" />

## MPSTAB: a cutting-edge quantum circuit simulator

Welcome to the `mpstab` official source page! The package offers a pure Python implementation of hybrid stabilizers-MPO formalism.
The package can be installed via `pip` after cloning this repository.

```shell
git clone https://github.com/MatteoRobbiati/mpstab.git
cd mpstab
pip install .
```

#### Software design (in my dreams)

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/925b99ba-bd72-4eb1-a895-9e3b518e4f6d" />

### Contribution guidelines

Contributions are super welcome! If you aim to contribute, please follow our style guidelines:

* The code formatting is managed through a series of linters, but we like to use `pre-commit`. This is a bot which helps in keeping the code style standardized. To use pre-commit, please install
  it following the [official installation instructions](https://pre-commit.com). Then enter the `mpstab` folder and run `pre-commit install`. From now on, everytime you will implement some lines of code,
  `pre-commit` will check your implementation and standardize the code with our policies. Then, you will need to repeat the `git add` and `git commit` procedure. Finally, feel free to push! 🫸
* Every new contribution requires at least one review from one member of the core team! Feel free to fork the project, open a pull request, and then one of us will have a look to your contribution 🫶
