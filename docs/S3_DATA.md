# Hosted CHASE data over S3

We host ready-to-use CHASE observations on an S3-compatible server so you can skip the portal, the signed-URL expiry dance, and the manual downloads. Any S3 client works: boto3, s3fs, rclone, the AWS CLI.

| | |
|---|---|
| Endpoint | `https://s3.preservemy.world` |
| Region name | `garage` |
| Data bucket | `chase-data` (read-only) |
| Results bucket | `chase-results` (read/write for researchers) |

**Getting credentials:** access keys are free for research use. Ask Waleed (open an issue on this repo or email) and you get an access-key/secret pair. Keys are read-only on the data; nothing here is writable by you except `chase-results`.

## What's hosted

```
chase-data/
  events/2026/07/04/M3.2/          2026-07-04 M3.2 flare, 24 scans
    manifest.csv                   filename, sequence, line, size, sha256
    validation.csv                 astropy validation of every file
    raw/HA/RSM20260704T*_HA.fits   24 files, ~259 MiB each (118 channels)
    raw/FE/RSM20260704T*_FE.fits   24 files, ~111 MiB each (46 channels)
```

More events and full observing days are added under `events/` and `observations/` as we ingest them; list the bucket to see what's current.

## Python (anywhere)

```python
import boto3

s3 = boto3.client("s3",
    endpoint_url="https://s3.preservemy.world",
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
    region_name="garage")

# See what's there
r = s3.list_objects_v2(Bucket="chase-data", Prefix="events/2026/07/04/M3.2/raw/HA/")
for o in r["Contents"]:
    print(o["Key"], o["Size"])

# Grab one scan and open it
key = "events/2026/07/04/M3.2/raw/HA/RSM20260704T132940_0000_HA.fits"
s3.download_file("chase-data", key, "scan0000_HA.fits")

from astropy.io import fits
with fits.open("scan0000_HA.fits", memmap=True) as hdul:
    cube = hdul[1].data      # (118, 2313, 2304) = (wavelength, y, x)
```

Download the folder once, then everything in this repo works on it:

```bash
chase ./data --patch 1292 1452 1976 2136 --contrast --temperature halpha
```

## Google Colab

Store the three values as Colab Secrets (key icon in the sidebar), never in cells:

| Secret name | Value |
|---|---|
| `CHASE_S3_ENDPOINT` | `https://s3.preservemy.world` |
| `CHASE_S3_ACCESS_KEY` | your access key |
| `CHASE_S3_SECRET_KEY` | your secret key |

```python
!pip -q install enchase boto3

from google.colab import userdata
import boto3
s3 = boto3.client("s3",
    endpoint_url=userdata.get("CHASE_S3_ENDPOINT"),
    aws_access_key_id=userdata.get("CHASE_S3_ACCESS_KEY"),
    aws_secret_access_key=userdata.get("CHASE_S3_SECRET_KEY"),
    region_name="garage")
```

Then download to `/content/` and process as above. For the ~259 MiB HA cubes, always download first and open with `memmap=True`; do not stream cubes over HTTP repeatedly.

## s3fs

```python
import s3fs, pandas as pd
fs = s3fs.S3FileSystem(key="YOUR_ACCESS_KEY", secret="YOUR_SECRET_KEY",
                       client_kwargs={"endpoint_url": "https://s3.preservemy.world"})

fs.ls("chase-data/events/2026/07/04/M3.2/raw/HA/")
with fs.open("chase-data/events/2026/07/04/M3.2/manifest.csv") as f:
    manifest = pd.read_csv(f)
```

## Saving your results

Research keys can write to `chase-results`. Use a folder with your name:

```python
s3.upload_file("temperature_map.npz", "chase-results",
               "2026/07/04/M3.2/yourname/temperature_map.npz")
```

## Integrity

Every hosted file has a SHA256 in the dataset's `manifest.csv`. To verify a download:

```python
import hashlib
h = hashlib.sha256(open("scan0000_HA.fits", "rb").read()).hexdigest()
```

Compare against the manifest row for that filename.
