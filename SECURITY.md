# Security policy

## Supported code

Security fixes apply to the current default branch and the currently deployed
portfolio-preview revision. Older local snapshots are not maintained.

## Reporting a vulnerability

Report vulnerabilities privately through the repository's GitHub **Security**
tab: open **Advisories**, then choose **Report a vulnerability** when private
vulnerability reporting is enabled. Include affected versions, reproduction
steps, impact, and any suggested mitigation.

Do not open a public issue containing an exploit, credential, private path, or
sensitive dataset detail. If the private reporting option is unavailable, the
repository maintainer should enable GitHub private vulnerability reporting
before accepting a report; this policy intentionally does not invent a contact
address.

## Deployment boundary

The hosted application is designed for synthetic portfolio-preview mode only.
Trusted local mode can read server filesystem paths, load model checkpoints,
process datasets, use substantial CPU/GPU resources, and expose local metadata.
It is not an upload service and is not hardened for untrusted users.

For every public or internet-facing deployment:

- leave `DASHBOARD_ENABLE_LIVE` and `DASHBOARD_TRUSTED_LOCAL` unset or set to
  `0`;
- do not expose the local ROCm launcher through a reverse proxy, tunnel, shared
  network interface, or public container;
- use `app/requirements.txt` and synthetic preview data only;
- never place KITTI data, model weights, secrets, or machine-local paths on the
  public host; and
- keep `DASHBOARD_SOURCE_URL` pointed at the exact source revision being run.

Run trusted local inference only on a machine and network controlled by the
operator. Treat datasets and model checkpoints as untrusted input: obtain them
from known sources, verify them where possible, and remember that serialized
model files can be unsafe to load.
