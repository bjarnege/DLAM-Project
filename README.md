## Dependencies

Run:

```bash
poetry config virtualenvs.in-project true
```

to install virtualenv in the project folder for better ide support

Then install all dependencies:

```bash
poetry install
poetry shell
pip install torch
```

Start a shell session within

```bash
poetry shell
```

## Project To-Do List

This repository contains the project tasks and timeline for the team. Each task has a checkbox to track its completion.

### ✔️ Datensatz: (Bjarne)
- [x] OrganAMNIST, OrganCMNIST, OrganSMNIST concatenation
- Timeline: Until 08.07.2023

### Res-Net: (Simon, Bjarne)
- [x] Initial training on `X_real`, `c_real` and validation on `X_test`, `c_test`
- Timeline: Until 15.07.2023
- [ ] Subsequent training on `X_fake`, `c_fake` generated by CGAN and validation on `X_test`, `c_test`
- Timeline: Until 30.08.2023


### CGAN: (Ella, Ruben)
- [ ] Adjust architecture
- [ ] Generate `X_fake` dataset with labels `c_fake`
- Timeline: From 01.08.2023 to 26.08.2023

### Feinarbeit:
- [ ] Refinement
- Timeline: Until 30.08.2023

### Präsentation: (alle)
- [ ] Presentation at the end
- Timeline: Until 04.09.2023

Please check the tasks above as they are completed.
