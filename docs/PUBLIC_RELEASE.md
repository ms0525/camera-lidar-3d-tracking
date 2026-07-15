# Public release checklist

Complete every item before changing the repository visibility or deploying the
Streamlit application.

## Licensing decision

- [ ] Choose one Ultralytics licensing path: publish the complete project under
  `AGPL-3.0-only`, or obtain an Ultralytics Enterprise License before any use
  that will not satisfy the AGPL requirements.
- [ ] Keep `LICENSE`, `NOTICE`, source/configuration files, dependency manifests,
  and build/run scripts in the public repository.
- [ ] Confirm that the source link in the running app resolves to the exact
  deployed revision.

## Audit the publication set

- [ ] Run the release audit from the repository root:

  ```powershell
  python scripts\audit_public_release.py
  ```

- [ ] Inspect every tracked and unignored candidate, not only the files shown by
  the IDE:

  ```powershell
  git status --short
  git ls-files --cached --others --exclude-standard
  git diff --cached --stat
  git diff --cached
  ```

- [ ] Confirm the candidate set contains none of the following:

  - KITTI images, point clouds, labels, calibration, archives, or derived
    camera/LiDAR captures;
  - model weights or exports such as `.pt`, `.pth`, `.onnx`, `.engine`, or
    `.safetensors`;
  - raw inference, training, evaluation, tracking, or dashboard-cache output;
  - screenshots, screen-camera metadata, videos, notebooks with output, or
    presentations;
  - passwords, tokens, keys, cookies, `.env` files, Streamlit secrets, private
    paths, or other credentials.

- [ ] If sensitive material was ever committed, remove it from all Git history
  and rotate every exposed credential. Adding a file to `.gitignore` is not
  sufficient.

## Validate the release

- [ ] Run the project tests in the documented environment:

  ```powershell
  python -m unittest discover -s tests
  ```

- [ ] Run `python scripts\audit_public_release.py` again after the final staged
  changes.
- [ ] Verify the README setup from a clean clone without local datasets, model
  files, caches, or machine-specific configuration.

## Public Streamlit deployment

- [ ] Deploy `app/streamlit_app.py` with `app/requirements.txt` in portfolio
  preview mode only.
- [ ] Leave `DASHBOARD_ENABLE_LIVE` and `DASHBOARD_TRUSTED_LOCAL` unset or set to
  `0`. Never enable trusted local mode on an internet-facing host.
- [ ] Set `DASHBOARD_SOURCE_URL` to an HTTPS URL for the exact deployed commit or
  immutable release, for example
  `https://github.com/OWNER/REPOSITORY/tree/COMMIT_SHA`.
- [ ] Confirm the deployed source URL is visible and includes the complete
  corresponding source and configuration used by that deployment.
- [ ] Confirm the hosted application displays only synthetic portfolio data and
  does not accept server-local dataset or model paths.
