<picture align="center">
  <img alt="HSPy Logo)" src="images/HSPyLogo.png">
</picture>

-----------------

# HSPy: Python Integration for Heimann Sensor Thermopiles
HSPy is a Python library that provides customers with a seamless way to integrate Heimann Sensor hardware and software into their Python applications. It offers a range of tools for:
- Live streaming, recording, and visualization of sensor data from our Appsets / Development Kits
- Converting sensor data recorded with ArraySoft (.bds/.txt) into NumPy arrays or pandas DataFrames for further analysis

With HSPy, developers can easily access, process, and analyze data from our systems directly within their Python workflows.

# Installation

HSPy uses [uv](https://docs.astral.sh/uv/) to manage its Python version, virtual environment and dependencies, and uv is the recommended way to install it. Installing with pip into an existing environment (e.g. Anaconda) is still possible, see [Alternative: Installation with pip](#alternative-installation-with-pip).

HSPy supports Python 3.10 to 3.13.

## Prerequisites
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (see [Install uv](#install-uv) below). A separate Python installation is **not** required: uv downloads a suitable Python version automatically.
- Git, either [Git for the command line](https://git-scm.com/downloads) or [GitHub Desktop](https://desktop.github.com/download/)
- [ArraySoft v2.30](https://cdn.website-editor.net/s/156d2965ff764637aaea150903bb0161/files/uploaded/SetupHTPAdGUIv2_30.rar?Expires=1709501376&Signature=p8nizQ9W1PC3O4uuuEpsxpqhTZc3t1vdlI1HImzhGcxrSur-9jmvYcf7EvJvU223HmZKhFJvr4dYW8PYtFwv1RVGjh626sN0ZQRICL6MBwOhqmevGODlUCFYjuGCMGWwlJCMpVz68dIYcBKjBS7MhEGKL~wCf1atkW82yr6eewPK3AJQmV0StLWQCi7Z4Q8epYWjGt4Xmuaa7wAcQJfFCBK1IKutkl52FPint4CqYarqQKfqpKJMn13SlVQSdp-RZbKZirddGrQkzTpYfi2BOuxThTn6C-FfXGQLC~Hnt1858gv96EKq23VJEGDmV~97rhQmj2YzXGySGpMnVv5xJQ__&Key-Pair-Id=K2NXBXLF010TJW) (if .bds or .txt files are to be parsed to pandas)

## Install uv
On Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

On macOS and Linux:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Afterwards, close and reopen your shell (and your editor, e.g. VS Code) so that uv is found, and check the installation with

```sh
uv --version
```

## Download HSPy
1. Clone this repository e.g. via

   ```sh
   git clone https://github.com/arehmer/HSPy.git
   ```

## Install HSPy
1. Open a shell and change into the cloned repository:

   ```sh
   cd path/to/HSPy
   ```
2. Run

   ```sh
   uv sync
   ```

   This command
   - downloads a compatible Python version if none is available,
   - creates a virtual environment in the folder `.venv` inside the repository,
   - installs all dependencies in the exact versions recorded in `uv.lock`,
   - installs HSPy in development (editable) mode, so that changes to the repository, e.g. after pulling new commits, are reflected immediately without reinstalling.

## Check installation
In the repository folder run

```sh
uv run python -c "import hspy; print('HSPy installed successfully')"
```

If the message is printed without an error, the installation was successful.

## Working with the environment
Run scripts inside the environment by prefixing them with `uv run`. No activation is needed:

```sh
uv run python my_script.py
```

Alternatively, activate the environment and use `python` as usual:

```sh
# Windows (PowerShell)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

In your IDE (e.g. VS Code or PyCharm), select the interpreter in `.venv` (`.venv\Scripts\python.exe` on Windows, `.venv/bin/python` on macOS/Linux).

## Updating HSPy
To get the latest version, pull the new commits and synchronize the environment:

```sh
git pull
uv sync
```

## Using HSPy in your own uv project
To use HSPy as a dependency of your own project managed with uv, add it directly from GitHub:

```sh
uv add git+https://github.com/arehmer/HSPy.git
```

or, if you have cloned the repository and want changes to it to take effect immediately, as an editable dependency:

```sh
uv add --editable path/to/HSPy
```

## Alternative: Installation with pip
HSPy can still be installed with pip into an existing environment (venv or conda). Activate the environment, then run

```sh
pip install -e path/to/HSPy
```

This requires pip 21.3 or newer (`python -m pip install --upgrade pip`). Note that pip installs dependencies based on the version ranges in `pyproject.toml`, not the exact versions recorded in `uv.lock`.

# Development
`uv sync` also installs the development tools (pytest, ruff). Common tasks:

```sh
uv run pytest          # run the tests
uv run ruff check .    # lint the code
uv add <package>       # add a dependency (updates pyproject.toml and uv.lock)
uv build               # build wheel and source distribution into dist/
```

Commit both `pyproject.toml` and `uv.lock` whenever dependencies change. The `.venv` folder must not be committed.
