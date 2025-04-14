# MultiOptForest -- Holisoils

This is a modified [Voila](https://github.com/voila-dashboards/voila) build
of the MultiOptForest optimization tool
implemented as a Jupyter Notebook.
See [`code/README.md`](./code/README.md)
for information on the project.

## Developing

To run the project on your machine,
you'll need a Python environment with its dependencies.
A Nix shell containing such an environment is available in `flake.nix`
with the command `nix develop`.

To run the project with our custom theme and hot-reloading enabled,
run the command
```bash
voila code --template material --autoreload=1
```
For more options see `voila --help`.

If not using the Nix shell which does this automatically,
note that the environment variable
`JUPYTER_PATH=$(PWD)/share/jupyter` needs to be set
for Voila to find the theme.

### Project structure

The theme is a modified version of [voila-material](https://github.com/voila-dashboards/voila-material/),
and its files are found under `./share/jupyter/`.
Custom CSS is all in `voila/templates/material/index.html.j2`.

Often modifications to the theme are not enough
and we need to touch the notebook Python code
written by the MultiOptForest researchers.
The notebooks are under `./code`,
where the most relevant file is `python/MultiFunc.py`.

### Publishing

This project can be published to Artifactory as a Podman container.
For now, it must be done manually,
as we don't have CI runner that can do this on gitlab.jyu.fi.

To prepare for the build, configure `jfrog-cli` to publish to repo.kopla.jyu.fi,
make sure Podman is installed,
and update the build version in `release-container.mk`.
Then run `make container-dist` to build the container
and `make container-publish` to push it to the staging repository
(this is a repo where old builds are automatically removed,
useful to host test builds).
Finally, use `make container-promote` to promote the build into the stable repo.

A Nomad template to run the published project on Kopla test servers
can be found in the [specs repo](https://gitlab.kopla.jyu.fi/infra/config/specs/-/blob/master/nomad-native/multioptforest.test.tpl.hcl)
(only available for Kopla developers).

## Voici

We initially tried to get this project running with [Voici](https://github.com/voila-dashboards/voici),
which runs the code in the user's browser with WebAssembly
(whereas Voila runs it on the server,
which is a potential scalability problem).
This ran into issues with dependencies
(`geopandas` and `ortools` in particular)
that haven't been packaged for WASM yet.
If you'd like to try Voici again, a patch allowing
its installation with Nix is provided in this repo.
See the commented-out part in `flake.nix`.
The WASM builds are defined [in this repo](https://github.com/emscripten-forge/recipes),
where `geopandas` still has an open issue at the time of writing
and `ortools` hasn't been attempted to be packaged at all.
