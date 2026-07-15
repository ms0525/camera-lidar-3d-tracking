# Dashboard deployment

The dashboard has two deliberately separate operating modes:

- **Public preview:** generated synthetic camera and LiDAR frames, with no KITTI download, model weights, GPU, or local drive required.
- **Local live mode:** YOLO11 and YOLO26 inference over a KITTI Tracking sequence using the full project environment.

The public preview is the recommended portfolio deployment. It loads quickly, avoids shipping the multi-gigabyte KITTI dataset, and does not expose machine-specific filesystem paths.

## Deploy the public preview on Streamlit Community Cloud

1. Push the project to a GitHub repository. Do not commit downloaded KITTI data, private credentials, or unapproved model artifacts.
2. In Streamlit Community Cloud, create an app from that repository and choose `app/streamlit_app.py` as the entry point.
3. Community Cloud will use `app/requirements.txt`, which intentionally contains only the dependencies needed by the lightweight preview.
4. Leave `DASHBOARD_ENABLE_LIVE` and `DASHBOARD_TRUSTED_LOCAL` unset (or set
   both to `0`). The app will start in synthetic preview mode.
5. In the app's advanced settings, set the root-level Streamlit secret below
   to the immutable GitHub URL for the exact deployed commit. Root-level
   Streamlit secrets are exposed as environment variables:

   ```toml
   DASHBOARD_SOURCE_URL = "https://github.com/OWNER/REPOSITORY/tree/COMMIT_SHA"
   ```

   Confirm that the resulting **Source code and AGPL license** link is visible
   and resolves without authentication.
6. After deployment, add the resulting HTTPS app URL to the project entry in
   your portfolio or CV.

See Streamlit's official [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) for the current UI and supported Python versions.

## Run local live mode with AMD ROCm

Install the complete project dependencies, including Streamlit, in a verified
Windows ROCm environment. The launchers use the repository-local
`.venv\Scripts\python.exe` by default. From the repository root, run:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install `
  -r requirements.txt -r app\requirements.txt
```

If the ROCm environment is elsewhere, pass its Python executable with
`-PythonPath "<path-to-rocm-python.exe>"`; no machine-specific environment path
is stored in the repository.

The root requirements pin `setuptools==80.10.2` because
`deep-sort-realtime==1.3.2` still imports `pkg_resources`, which was removed
from Setuptools 82. Newer Setuptools versions prevent the Deep SORT embedder
from starting.

Then launch:

```powershell
.\scripts\run_streamlit_rocm.ps1 `
  -DatasetRoot "<path-to-kitti-tracking>" `
  -Sequence 0000 `
  -Yolo11Model "<path-to-yolo11-weights>" `
  -Yolo26Model "<path-to-yolo26-weights>" `
  -Device 0 `
  -EmbedderGpu 0 `
  -Port 8501
```

The launcher validates the dataset sequence, calibration, point clouds, images, weights, Python executable, and Streamlit installation before starting the app. It also discovers the MSVC and ROCm Clang headers required by MIOpen, isolates temporary and library caches, and confirms that PyTorch can access the selected AMD GPU. Its environment changes are process-local and are restored when Streamlit exits. KITTI labels are optional; when available, the app can display ground truth.

`-DatasetRoot` is required. It may point either to the directory containing
`training` or directly to the `training` split. The default Python executable
is repository-relative:

```text
<repository>\.venv\Scripts\python.exe
```

The cache root defaults to the current user's local application-data directory
at `%LOCALAPPDATA%\3D_Detection\ROCm`; pass `-CacheRoot` to use another writable
directory. GPU visibility defaults to device `0`. Override it with
`-VisibleGpu`. Header locations are discovered automatically; if discovery is
not possible, pass verified directories with `-MsvcInclude` and
`-RocmClangInclude`.

The training launcher follows the same Python and cache conventions, requires
its own Ultralytics-format dataset root, and writes runs below the repository's
ignored `runs` directory unless `-Project` is supplied:

```powershell
.\scripts\train_kitti_rocm.ps1 `
  -DatasetRoot "<path-to-ultralytics-kitti-dataset>" `
  -Model "<path-to-compatible-yolo-weights>"
```

## Dashboard environment variables

The launcher sets the live-mode values automatically. Set
`DASHBOARD_SOURCE_URL` separately for a public deployment:

| Variable | Meaning | Example |
| --- | --- | --- |
| `DASHBOARD_ENABLE_LIVE` | Enables local model inference when set to `1` | `1` |
| `DASHBOARD_TRUSTED_LOCAL` | Second guard set by the localhost-only launcher | `1` |
| `DASHBOARD_DATASET_ROOT` | KITTI Tracking root or training split | `<path-to-kitti-tracking>` |
| `DASHBOARD_SEQUENCE` | One- to four-digit sequence, normalized to four digits | `0000` |
| `DASHBOARD_YOLO11_MODEL` | YOLO11 weights path | `<path-to-yolo11-weights>` |
| `DASHBOARD_YOLO26_MODEL` | YOLO26 weights path | `<path-to-yolo26-weights>` |
| `DASHBOARD_DEVICE` | Ultralytics device selector; ROCm uses PyTorch's CUDA-compatible API | `0` |
| `DASHBOARD_EMBEDDER_GPU` | Runs the DeepSORT appearance embedder on the GPU when set to `1` | `0` |
| `DASHBOARD_SOURCE_URL` | Immutable public source URL for the deployed revision | `https://github.com/OWNER/REPOSITORY/tree/COMMIT_SHA` |

Live mode is intentionally guarded by both `DASHBOARD_ENABLE_LIVE=1` and
`DASHBOARD_TRUSTED_LOCAL=1`. The launcher binds Streamlit to `127.0.0.1`, so
its editable server-side paths and raw local errors are not exposed to remote
visitors. File paths are configuration, not values to hard-code in the
application or commit to source control.

## Hosting limitations

Streamlit Community Cloud does not have access to a visitor's local dataset,
model files, or native-Windows AMD ROCm installation. Do not enable live mode
there. Running two detector/tracker pipelines and rendering full-resolution
point clouds may also exceed the CPU, memory, storage, or startup limits of a
lightweight public host.

If public live inference is required later, use a separate authenticated,
GPU-capable service with fixed allow-listed asset identifiers (not arbitrary
filesystem inputs), and keep the Streamlit UI as a client or deploy it
alongside the inference service. The local Windows ROCm setup cannot simply be
copied to Community Cloud. Do not set the two local-live guards on a public,
unauthenticated deployment.

For a responsive public demo:

- keep synthetic or properly licensed precomputed assets small;
- decimate LiDAR point clouds before serialization;
- cache immutable assets and keep playback state per user session;
- avoid downloading weights or datasets at app startup; and
- show that predicted 3D boxes are approximate when their dimensions or yaw use class priors.

## Dataset and model licensing

Before publishing real KITTI frames, point clouds, annotations, derived assets, or downloadable weights, review the current terms yourself:

- [Official KITTI dataset page and license](https://www.cvlibs.net/datasets/kitti/)
- [Official KITTI Terms of Service](https://www.cvlibs.net/datasets/kitti/terms_of_service.php)
- [Official Ultralytics licensing guide](https://www.ultralytics.com/license)
- [Ultralytics AGPL-3.0 license text](https://www.ultralytics.com/legal/agpl-3-0-software-license)

KITTI describes its datasets under CC BY-NC-SA 3.0, including attribution, non-commercial, and share-alike conditions. Ultralytics offers AGPL-3.0 and commercial licensing paths. A public portfolio deployment does not remove those obligations. The synthetic preview is designed to avoid redistributing KITTI material, but it does not replace a license review for the source code, models, or any later real-data deployment.
