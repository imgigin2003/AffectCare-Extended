# visualize_api.py
# Renders the MFCC spectrogram for a single audio clip — the exact (13, 173)
# "image" the CNN actually reads — as a base64 PNG, and returns it as JSON so
# the Node backend can hand it straight to the browser.

import base64
import io
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt  # noqa: E402

from preprocessing import load_and_format  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: python3 visualize_api.py <audio_file>"}))
        sys.exit(1)

    filepath = sys.argv[1]

    try:
        # (1, 13, 173) -> (13, 173): 13 MFCC bands over ~173 time frames
        mfcc = load_and_format(filepath)[0]

        fig, ax = plt.subplots(figsize=(6, 2.4), dpi=140)
        im = ax.imshow(mfcc, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
        plt.close(fig)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")

        # A couple of cheap descriptive stats for the UI caption
        stats = {
            "n_mfcc": int(mfcc.shape[0]),
            "n_frames": int(mfcc.shape[1]),
            "energy_min": round(float(mfcc.min()), 2),
            "energy_max": round(float(mfcc.max()), 2),
        }

        print(json.dumps({"mfcc_png": encoded, "stats": stats}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
